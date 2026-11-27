// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <boost/range/iterator_range.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

#include "holoflow/runtime/compiler.hh"
#include "holotask/pipelines/mtd.hh"
#include "holotask/sinks/npy_volume.hh"
#include "holotask/sources/pgm_sequence.hh"
#include "python_oracle.hh"
#include "sync_task_runner.hh"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace {
using namespace holoflow::core;
using namespace holotask::syncs;
using Complex = std::complex<float>;

template <typename T> std::vector<std::byte> bytes(const std::vector<T> &data) {
  std::vector<std::byte> result(data.size() * sizeof(T));
  std::memcpy(result.data(), data.data(), result.size());
  return result;
}
void expect_close(const std::vector<std::byte> &actual, const std::vector<std::byte> &expected,
                  float atol = 2e-5f, float rtol = 2e-4f) {
  ASSERT_EQ(actual.size(), expected.size());
  ASSERT_EQ(actual.size() % sizeof(float), 0);
  for (size_t i = 0; i < actual.size(); i += sizeof(float)) {
    float a, e;
    std::memcpy(&a, actual.data() + i, sizeof(float));
    std::memcpy(&e, expected.data() + i, sizeof(float));
    ASSERT_TRUE(std::isfinite(a)) << i / sizeof(float);
    ASSERT_NEAR(a, e, atol + rtol * std::abs(e)) << "float element " << i / sizeof(float);
  }
}
holonp_test::RunResult compare(const ISyncTaskFactory &factory, const std::string &op,
                               std::vector<TDesc> descs, std::vector<std::vector<std::byte>> data,
                               nlohmann::json settings, size_t outputs = 1, float atol = 2e-5f) {
  const auto run = holonp_test::run_sync_factory(factory, descs, data, settings);
  const auto reference =
      holonp_test::invoke_oracle({op, outputs, descs, data, settings}, HOLOTASK_MTD_ORACLE);
  EXPECT_EQ(run.output_bytes.size(), reference.output_bytes.size());
  for (size_t i = 0; i < run.output_bytes.size(); ++i)
    expect_close(run.output_bytes[i], reference.output_bytes[i], atol);
  return run;
}
std::vector<Complex> complex_field(size_t frames, int side) {
  std::vector<Complex> values(frames * static_cast<size_t>(side) * side);
  for (size_t frame = 0; frame < frames; ++frame)
    for (int y = 0; y < side; ++y)
      for (int x = 0; x < side; ++x) {
        const float xn = static_cast<float>(x) / (side - 1),
                    yn = static_cast<float>(y) / (side - 1);
        const float amplitude =
            1.0f + 0.1f * xn + 0.2f * yn + 0.1f * std::cos(17 * xn + static_cast<float>(frame));
        const float phase = 4.5f * std::sin(6.0f * xn) + 0.4f * std::cos(5.0f * yn) +
                            static_cast<float>(frame) * 0.03f;
        values[(frame * side + y) * side + x] = std::polar(amplitude, phase);
      }
  return values;
}
class TemporaryDirectory {
public:
  TemporaryDirectory() {
    static std::atomic<int> number{0};
    path_ = std::filesystem::temp_directory_path() /
            ("holotask_mtd_" +
             std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
             std::to_string(number++));
    std::filesystem::create_directories(path_);
  }
  ~TemporaryDirectory() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  std::filesystem::path path() const { return path_; }

private:
  std::filesystem::path path_;
};
void pgm(const std::filesystem::path &path, int side, const std::vector<uint8_t> &pixels) {
  std::ofstream out(path, std::ios::binary);
  out << "P5\n# test\n" << side << ' ' << side << "\n255\n";
  out.write(reinterpret_cast<const char *>(pixels.data()),
            static_cast<std::streamsize>(pixels.size()));
}

TEST(MTDFieldRecovery, MatchesIndependentFourierReference) {
  constexpr int      side = 32;
  std::vector<float> images(2 * side * side);
  for (size_t i = 0; i < images.size(); ++i)
    images[i] = 120.0f + 12.0f * std::cos(0.98f * static_cast<float>(i % side) +
                                          0.79f * static_cast<float>((i / side) % side));
  compare(MTDFieldRecoveryFactory{}, "field", {TDesc({2, side, side}, DType::F32, MemLoc::Host)},
          {bytes(images)}, MTDFieldRecoverySettings{.half_size = 8, .center_x = 10, .center_y = 10},
          2, 0.02f);
}
TEST(MTDFieldRecovery, ZeroFrameUsesFirstArgmaxIndex) {
  const auto run = holonp_test::run_sync_factory(
      MTDFieldRecoveryFactory{}, std::vector<TDesc>{TDesc({1, 8, 8}, DType::U16, MemLoc::Host)},
      std::vector<std::vector<std::byte>>{bytes(std::vector<uint16_t>(64, 0))},
      MTDFieldRecoverySettings{.half_size = 2, .center_x = 4, .center_y = 4, .tukey_alpha = 0});
  EXPECT_EQ(run.output_bytes[1], bytes(std::vector<float>{0, 0}));
  EXPECT_EQ(run.output_bytes[0], bytes(std::vector<Complex>(16)));
}
TEST(MTDFieldRecovery, RejectsInvalidCropAndNonfiniteWindow) {
  const std::vector<TDesc> descs{TDesc({1, 16, 16}, DType::U8, MemLoc::Host)};
  EXPECT_THROW((void)MTDFieldRecoveryFactory{}.infer(
                   descs, MTDFieldRecoverySettings{.half_size = 8, .center_x = 2, .center_y = 8}),
               std::invalid_argument);
  EXPECT_THROW(
      (void)MTDFieldRecoveryFactory{}.infer(
          descs,
          MTDFieldRecoverySettings{.half_size = 4, .center_x = 8, .center_y = 8, .tukey_alpha = 2}),
      std::invalid_argument);
}
TEST(MTDRytov, MirroredVolkovAndMaskedFitMatchReference) {
  constexpr int        side = 16;
  std::vector<uint8_t> mask(side * side, 255);
  for (int y = 5; y < 11; ++y)
    for (int x = 5; x < 11; ++x)
      mask[static_cast<size_t>(y) * side + x] = 0;
  compare(
      RytovCorrectionFactory{}, "rytov",
      {TDesc({2, side, side}, DType::CF32, MemLoc::Device),
       TDesc({1, side, side}, DType::U8, MemLoc::Host)},
      {bytes(complex_field(2, side)), bytes(mask)},
      RytovCorrectionSettings{.polynomial_degree = 2, .sample_step = 1, .background_margin = 1});
}
TEST(MTDRytov, WrappedModeMatchesReference) {
  constexpr int side = 12;
  compare(
      RytovCorrectionFactory{}, "rytov",
      {TDesc({1, side, side}, DType::CF32, MemLoc::Device),
       TDesc({side, side}, DType::U8, MemLoc::Host)},
      {bytes(complex_field(1, side)), bytes(std::vector<uint8_t>(side * side, 255))},
      RytovCorrectionSettings{
          .polynomial_degree = 2, .sample_step = 1, .background_margin = 0, .unwrap_phase = false});
}
TEST(MTDRytov, EmptyBackgroundMaskIsRejected) {
  const std::vector<TDesc> descs{TDesc({1, 16, 16}, DType::CF32, MemLoc::Device),
                                 TDesc({16, 16}, DType::U8, MemLoc::Host)};
  EXPECT_THROW(holonp_test::run_sync_factory(
                   RytovCorrectionFactory{}, descs,
                   std::vector<std::vector<std::byte>>{bytes(complex_field(1, 16)),
                                                       bytes(std::vector<uint8_t>(256, 0))},
                   RytovCorrectionSettings{}),
               std::invalid_argument);
}
TEST(MTDLinear3D, VolumeAndCoverageMatchReference) {
  compare(
      MTDLinear3DReconstructionFactory{}, "linear",
      {TDesc({3, 8, 8}, DType::CF32, MemLoc::Device), TDesc({3, 2}, DType::F32, MemLoc::Device)},
      {bytes(complex_field(3, 8)), bytes(std::vector<float>{3, 4, 4, 3, 4, 4})},
      MTDLinear3DReconstructionSettings{
          .dimension = 16, .hologram_size = 32, .pixel_pitch_object = 1e-7},
      2, 1e-6f);
}
TEST(MTDGPS, PartialCoverageMatchesReferenceIncludingEvenNonMultipleOfFour) {
  for (const size_t dim : {size_t{6}, size_t{8}}) {
    auto               volume = complex_field(dim, static_cast<int>(dim));
    std::vector<float> coverage(volume.size());
    for (size_t i = 0; i < coverage.size(); ++i)
      coverage[i] = i % 3 == 0 ? 2.0f : 0.0f;
    compare(GPSRefinement3DFactory{}, "gps",
            {TDesc({dim, dim, dim}, DType::CF32, MemLoc::Device),
             TDesc({dim, dim, dim}, DType::F32, MemLoc::Device)},
            {bytes(volume), bytes(coverage)},
            GPSRefinement3DSettings{.delta_n_min = -0.1f,
                                    .delta_n_max = 0.1f,
                                    .kappa_min   = 0,
                                    .kappa_max   = 0.05f,
                                    .iterations  = 3},
            1, 2e-6f);
  }
}
TEST(MTDGPS, InvalidBoundsAndCancelledExecution) {
  const std::vector<TDesc> descs{TDesc({4, 4, 4}, DType::CF32, MemLoc::Device),
                                 TDesc({4, 4, 4}, DType::F32, MemLoc::Device)};
  EXPECT_THROW((void)GPSRefinement3DFactory{}.infer(
                   descs, GPSRefinement3DSettings{.delta_n_min = 1, .delta_n_max = 0}),
               std::invalid_argument);
  curaii::CudaStream stream;
  auto task = GPSRefinement3DFactory{}.create(descs, GPSRefinement3DSettings{}, {stream.get()});
  std::atomic<bool> cancelled{true};
  SyncCtx           context{.cancelled = &cancelled};
  EXPECT_EQ(task->execute(context), OpResult::Cancelled);
}
TEST(MTDTransforms, SeparateOutputsResetViewOffsetAndPreserveInputs) {
  TDesc field({2, 16, 16}, DType::CF32, MemLoc::Device);
  field.offset     = 128;
  const auto rytov = RytovCorrectionFactory{}.infer(
      std::vector<TDesc>{field, TDesc({16, 16}, DType::U8, MemLoc::Host)},
      RytovCorrectionSettings{});
  EXPECT_TRUE(rytov.in_place.empty());
  EXPECT_EQ(rytov.input_descs[0].offset, 128);
  EXPECT_EQ(rytov.output_descs[0].offset, 0);
  EXPECT_EQ(rytov.output_descs[0].shape, field.shape);
  TDesc volume({8, 8, 8}, DType::CF32, MemLoc::Device);
  volume.offset  = 256;
  const auto gps = GPSRefinement3DFactory{}.infer(
      std::vector<TDesc>{volume, TDesc({8, 8, 8}, DType::F32, MemLoc::Device)},
      GPSRefinement3DSettings{});
  EXPECT_TRUE(gps.in_place.empty());
  EXPECT_EQ(gps.input_descs[0].offset, 256);
  EXPECT_EQ(gps.output_descs[0].offset, 0);
  EXPECT_EQ(gps.output_descs[0].shape, volume.shape);
}
TEST(MTDVolumeSlice, AllAxesAndComponentsMatchReference) {
  std::vector<Complex> volume(4 * 3 * 5);
  for (size_t i = 0; i < volume.size(); ++i)
    volume[i] = Complex(static_cast<float>(i), -static_cast<float>(i) / 2);
  for (int axis = 0; axis < 3; ++axis)
    for (const std::string component : {"real", "imaginary", "magnitude"})
      compare(VolumeSliceFactory{}, "slice", {TDesc({4, 3, 5}, DType::CF32, MemLoc::Device)},
              {bytes(volume)},
              VolumeSliceSettings{.axis = axis, .index = 2, .component = component});
}
TEST(MTDNpyWriter, PreservesComplexValuesForNumpy) {
  TemporaryDirectory directory;
  const auto         path     = directory.path() / "volume.npy";
  const auto         data     = bytes(complex_field(2, 4));
  const auto         settings = holotask::sinks::NpyVolumeSettings{.path = path.string()};
  (void)holonp_test::run_sync_factory(
      holotask::sinks::NpyVolumeFactory{},
      std::vector<TDesc>{TDesc({2, 4, 4}, DType::CF32, MemLoc::Device)},
      std::vector<std::vector<std::byte>>{data}, settings);
  const auto loaded = holonp_test::invoke_oracle({"load_npy", 1, {}, {}, {{"path", path.string()}}},
                                                 HOLOTASK_MTD_ORACLE);
  EXPECT_EQ(data, loaded.output_bytes[0]);
  EXPECT_TRUE(std::filesystem::is_regular_file(path.string() + ".json"));
  EXPECT_THROW((void)holotask::sinks::NpyVolumeFactory{}.infer(
                   std::vector<TDesc>{TDesc({2, 4, 4}, DType::CF32, MemLoc::Device)}, settings),
               std::invalid_argument);
}
TEST(MTDPgmSource, FiniteScanPublishesSharedStorageAndStops) {
  TemporaryDirectory directory;
  pgm(directory.path() / "i000.pgm", 2, {1, 2, 3, 4});
  holotask::sources::PgmSequenceFactory factory;
  const auto                            settings = holotask::sources::PgmSequenceSettings{
      .directory = directory.path().string(), .repeat = false};
  auto               task = factory.create({}, settings, {});
  Storage            shared{MemLoc::Host, 0, nullptr};
  std::vector<TView> outputs{{factory.infer({}, settings).output_descs[0], &shared}};
  SyncCtx            ctx{.outputs = outputs};
  EXPECT_EQ(task->execute(ctx), OpResult::Ok);
  EXPECT_EQ(outputs[0].storage, &shared);
  ASSERT_NE(shared.ptr, nullptr);
  EXPECT_EQ(static_cast<unsigned char>(shared.ptr[0]), 1);
  EXPECT_EQ(task->execute(ctx), OpResult::Eof);
}
TEST(MTDPgmSource, RejectsExtraMatchingFrames) {
  TemporaryDirectory directory;
  pgm(directory.path() / "i000.pgm", 2, {1, 2, 3, 4});
  pgm(directory.path() / "i001.pgm", 2, {1, 2, 3, 4});
  EXPECT_THROW(
      (void)holotask::sources::PgmSequenceFactory{}.infer(
          {}, holotask::sources::PgmSequenceSettings{.directory      = directory.path().string(),
                                                     .repeat         = false,
                                                     .expected_count = 1}),
      std::invalid_argument);
}
TEST(MTDPipeline, RejectsOutputAliasesAndMetadataCollisions) {
  TemporaryDirectory                      directory;
  const auto                              volume = directory.path() / "volume.npy";
  holotask::pipelines::MTDDatasetSettings dataset{.frame_count = 1};
  for (const auto &slice : {volume, directory.path() / "." / "volume.npy",
                            std::filesystem::path(volume.string() + ".json")})
    EXPECT_THROW((void)holotask::pipelines::build_mtd_pipeline(
                     dataset, {.volume_path = volume, .slice_path = slice, .overwrite = true}),
                 std::invalid_argument);
  EXPECT_THROW((void)holotask::pipelines::build_mtd_pipeline(
                   dataset, {.volume_path = volume,
                             .overwrite   = true,
                             .graph_path  = std::filesystem::path(volume.string() + ".json")}),
               std::invalid_argument);
}
#ifdef _WIN32
TEST(MTDNpyWriter, FailedOverwritePreservesPreviousPair) {
  TemporaryDirectory       directory;
  const auto               path          = directory.path() / "volume.npy";
  const auto               metadata_path = std::filesystem::path(path.string() + ".json");
  const std::vector<TDesc> descs{TDesc({2, 4, 4}, DType::CF32, MemLoc::Device)};
  const auto               data = bytes(complex_field(2, 4));
  auto                     settings =
      holotask::sinks::NpyVolumeSettings{.path = path.string(), .metadata = {{"version", 1}}};
  (void)holonp_test::run_sync_factory(holotask::sinks::NpyVolumeFactory{}, descs,
                                      std::vector<std::vector<std::byte>>{data}, settings);
  const auto read_file = [](const std::filesystem::path &file) {
    std::ifstream input(file, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(input), {});
  };
  const auto original_volume = read_file(path), original_metadata = read_file(metadata_path);
  settings.overwrite           = true;
  settings.metadata["version"] = 2;
  const auto locked = CreateFileW(metadata_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  ASSERT_NE(locked, INVALID_HANDLE_VALUE);
  const std::vector<std::vector<std::byte>> replacement{bytes(std::vector<Complex>(32))};
  EXPECT_THROW((void)holonp_test::run_sync_factory(holotask::sinks::NpyVolumeFactory{}, descs,
                                                   replacement, settings),
               std::filesystem::filesystem_error);
  CloseHandle(locked);
  EXPECT_EQ(read_file(path), original_volume);
  EXPECT_EQ(read_file(metadata_path), original_metadata);
  (void)holonp_test::run_sync_factory(holotask::sinks::NpyVolumeFactory{}, descs, replacement,
                                      settings);
  const auto loaded = holonp_test::invoke_oracle({"load_npy", 1, {}, {}, {{"path", path.string()}}},
                                                 HOLOTASK_MTD_ORACLE);
  EXPECT_EQ(loaded.output_bytes[0], bytes(std::vector<Complex>(32)));
  EXPECT_EQ(nlohmann::json::parse(read_file(metadata_path))["version"], 2);
  for (const auto &entry : std::filesystem::directory_iterator(directory.path()))
    EXPECT_EQ(entry.path().filename().string().find(".backup-"), std::string::npos);
}
#endif
TEST(MTDPipeline, CompilerSchedulerAndWriterMatchReference) {
  TemporaryDirectory   directory;
  constexpr int        holo = 32, side = 16;
  constexpr size_t     frames = 3;
  std::vector<uint8_t> images(frames * holo * holo), mask(side * side, 255);
  for (size_t i = 0; i < images.size(); ++i)
    images[i] =
        static_cast<uint8_t>(120 + 40 * std::cos(0.98 * static_cast<double>(i % holo) +
                                                 0.79 * static_cast<double>((i / holo) % holo)));
  for (size_t frame = 0; frame < frames; ++frame)
    pgm(directory.path() / ("i00" + std::to_string(frame) + ".pgm"), holo,
        {images.begin() + static_cast<std::ptrdiff_t>(frame * holo * holo),
         images.begin() + static_cast<std::ptrdiff_t>((frame + 1) * holo * holo)});
  pgm(directory.path() / "Image_mask.pgm", side, mask);
  holotask::pipelines::MTDDatasetSettings dataset{
      .acquisition_directory = directory.path(),
      .frame_count           = 3,
      .recovery              = {.half_size = 8, .center_x = 10, .center_y = 10},
      .rytov                 = {.polynomial_degree = 2, .sample_step = 1, .background_margin = 1},
      .linear                = {.dimension = 16, .hologram_size = holo, .pixel_pitch_object = 1e-7},
      .gps                   = {.iterations = 3}};
  const auto output = directory.path() / "gps.npy";
  auto       graph  = holotask::pipelines::build_mtd_pipeline(dataset, {.volume_path = output});
  const std::vector<std::pair<std::string, std::filesystem::path>> branches{
      {"mtd_field_recovery", directory.path() / "fields.npy"},
      {"mtd_linear3d", directory.path() / "linear.npy"}};
  for (const auto &[name, path] : branches) {
    for (const auto vertex : boost::make_iterator_range(boost::vertices(graph))) {
      if (graph[vertex].name != name)
        continue;
      const auto writer = boost::add_vertex(
          NodeSpec{.name     = name + "_export",
                   .kind     = "NpyVolumeWriter",
                   .settings = holotask::sinks::NpyVolumeSettings{.path = path.string()},
                   .debug    = false},
          graph);
      boost::add_edge(vertex, writer, EdgeSpec{0, 0}, graph);
      break;
    }
  }
  Registry registry;
  holotask::pipelines::register_mtd_tasks(registry);
  holoflow::runtime::Compiler  compiler(registry, {.enable_profiling = false});
  auto                         compiled = compiler.compile(graph);
  holoflow::runtime::Scheduler scheduler(compiled->graph, compiled->sections, compiled->resources);
  scheduler.start();
  scheduler.wait();
  const auto loaded = holonp_test::invoke_oracle(
      {"load_npy", 1, {}, {}, {{"path", output.string()}}}, HOLOTASK_MTD_ORACLE);
  const auto expected =
      holonp_test::invoke_oracle({"pipeline",
                                  3,
                                  {TDesc({frames, holo, holo}, DType::U8, MemLoc::Host),
                                   TDesc({side, side}, DType::U8, MemLoc::Host)},
                                  {bytes(images), bytes(mask)},
                                  {{"recovery", dataset.recovery},
                                   {"rytov", dataset.rytov},
                                   {"linear", dataset.linear},
                                   {"gps", dataset.gps}}},
                                 HOLOTASK_MTD_ORACLE);
  expect_close(loaded.output_bytes[0], expected.output_bytes[0], 2e-5f, 2e-3f);
  for (size_t i = 0; i < branches.size(); ++i) {
    const auto branch = holonp_test::invoke_oracle(
        {"load_npy", 1, {}, {}, {{"path", branches[i].second.string()}}}, HOLOTASK_MTD_ORACLE);
    expect_close(branch.output_bytes[0], expected.output_bytes[i + 1], i == 0 ? 0.02f : 2e-5f,
                 2e-3f);
  }
}
} // namespace
