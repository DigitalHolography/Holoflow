// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow_event/router.hh"
#include "holofile/holofile.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/sources/fresnel_qin.hh"
#include "holotask/sources/fresnel_qout.hh"
#include "holotask/syncs/convolution.hh"
#include "holotask/syncs/correct_phase.hh"
#include "holotask/syncs/cross_correlation2.hh"
#include "holotask/syncs/cuda_stream_synchronize.hh"
#include "holotask/syncs/mean_abs.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/registration.hh"
#include "holotask/syncs/unfold2d.hh"
#include "holotask/syncs/wrap2pi.hh"
#include "holotask/sinks/holofile.hh"

#include "sync_task_runner.hh"
#include "tensor_test_buffer.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

template <typename T> std::vector<std::byte> as_bytes(const std::vector<T> &values) {
  std::vector<std::byte> bytes(values.size() * sizeof(T));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return bytes;
}

template <typename T> std::vector<T> from_bytes(const std::vector<std::byte> &bytes) {
  EXPECT_EQ(bytes.size() % sizeof(T), 0U);
  std::vector<T> values(bytes.size() / sizeof(T));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

TDesc device_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}

TDesc host_desc(std::vector<size_t> shape, DType dtype) {
  return TDesc(std::move(shape), dtype, MemLoc::Host);
}

template <typename T>
void expect_near_values(const std::vector<std::byte> &actual, const std::vector<T> &expected,
                        float tolerance = 1e-5F) {
  const auto values = from_bytes<T>(actual);
  ASSERT_EQ(values.size(), expected.size());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_NEAR(static_cast<float>(values[i]), static_cast<float>(expected[i]), tolerance)
        << "element " << i;
  }
}

struct ComplexValue {
  float real;
  float imag;
};

} // namespace

TEST(HoloTaskHeadlessTest, MemcpyRoundTripsBetweenHostAndDevice) {
  const TDesc host_input = host_desc({4}, DType::F32);
  const std::vector<float> expected{1.25F, -2.5F, 3.75F, 4.0F};
  const auto input_bytes = as_bytes(expected);

  holotask::syncs::MemcpyFactory factory;
  const auto to_device = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{host_input}, std::vector<std::vector<std::byte>>{input_bytes},
      holotask::syncs::MemcpySettings{.target = holotask::syncs::MemcpySettings::Target::Device});

  ASSERT_EQ(to_device.output_descs[0].mem_loc, MemLoc::Device);
  const TDesc device_input = device_desc({4}, DType::F32);
  const auto to_host = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{device_input},
      std::vector<std::vector<std::byte>>{to_device.output_bytes[0]},
      holotask::syncs::MemcpySettings{.target = holotask::syncs::MemcpySettings::Target::Host});

  ASSERT_EQ(to_host.output_descs[0].mem_loc, MemLoc::Host);
  expect_near_values<float>(to_host.output_bytes[0], expected, 0.0F);
}

TEST(HoloTaskHeadlessTest, CudaStreamSynchronizePreservesTensorValues) {
  holotask::syncs::CudaStreamSynchronizeFactory factory;
  const TDesc desc = device_desc({2, 3}, DType::F32);
  const std::vector<float> expected{1.F, 2.F, 3.F, 4.F, 5.F, 6.F};
  curaii::CudaStream stream;
  auto task = factory.create(std::vector<TDesc>{desc},
                             holotask::syncs::CudaStreamSynchronizeSettings{}, {stream.get()});
  holonp_test::TensorTestBuffer buffer(desc);
  buffer.upload(as_bytes(expected));
  auto view = buffer.view();
  std::atomic<bool> cancelled{false};
  holoflow::core::SyncCtx ctx{.inputs = {&view, 1},
                              .outputs = {&view, 1},
                              .cancelled = &cancelled,
                              .event_writer = nullptr,
                              .event_reader = nullptr};
  task->bind_logger(spdlog::default_logger());
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));
  expect_near_values<float>(buffer.download(), expected, 0.0F);
}

TEST(HoloTaskHeadlessTest, MeanAbsReducesSelectedAxisAndKeepsDimensions) {
  holotask::syncs::MeanAbsFactory factory;
  const TDesc desc = device_desc({2, 3}, DType::F32);
  const std::vector<float> input{-1.F, 2.F, -3.F, 4.F, -5.F, 6.F};

  const auto reduced = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      holotask::syncs::MeanAbsSettings{.axis = {1}, .keepdims = false});
  ASSERT_EQ(reduced.output_descs[0].shape, (std::vector<size_t>{2}));
  expect_near_values<float>(reduced.output_bytes[0], {2.F, 5.F}, 1e-6F);

  const auto kept = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      holotask::syncs::MeanAbsSettings{.axis = {-1}, .keepdims = true});
  ASSERT_EQ(kept.output_descs[0].shape, (std::vector<size_t>{2, 1}));
  expect_near_values<float>(kept.output_bytes[0], {2.F, 5.F}, 1e-6F);
}

TEST(HoloTaskHeadlessTest, Wrap2PiHandlesNegativeAndMultipleTurns) {
  holotask::syncs::Wrap2PiFactory factory;
  const TDesc desc = device_desc({4}, DType::F32);
  const float pi = std::acos(-1.0F);
  const std::vector<float> input{-0.5F, 0.5F, 2.0F * pi + 0.25F, -2.0F * pi - 0.75F};
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      holotask::syncs::Wrap2PiSettings{});

  expect_near_values<float>(result.output_bytes[0],
                            {2.0F * pi - 0.5F, 0.5F, 0.25F, 2.0F * pi - 0.75F}, 2e-6F);
}

TEST(HoloTaskHeadlessTest, Unfold2DExtractsOverlappingWindowsInRowMajorOrder) {
  holotask::syncs::Unfold2DFactory factory;
  const TDesc desc = device_desc({3, 3}, DType::F32);
  const std::vector<float> input{1.F, 2.F, 3.F, 4.F, 5.F, 6.F, 7.F, 8.F, 9.F};
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      holotask::syncs::Unfold2DSettings{.win_h = 2, .win_w = 2, .stride_y = 1, .stride_x = 1});

  ASSERT_EQ(result.output_descs[0].shape, (std::vector<size_t>{2, 2, 2, 2}));
  expect_near_values<float>(result.output_bytes[0],
                            {1.F, 2.F, 4.F, 5.F, 2.F, 3.F, 5.F, 6.F,
                             4.F, 5.F, 7.F, 8.F, 5.F, 6.F, 8.F, 9.F},
                            0.0F);
}

TEST(HoloTaskHeadlessTest, CorrectPhaseAppliesTheNegativePhaseConvention) {
  holotask::syncs::CorrectPhaseFactory factory;
  const TDesc input_desc = device_desc({1, 2}, DType::F32);
  const TDesc phase_desc = device_desc({1, 2}, DType::F32);
  const float pi = std::acos(-1.0F);
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc, phase_desc},
      std::vector<std::vector<std::byte>>{as_bytes(std::vector<float>{1.F, 2.F}),
                                          as_bytes(std::vector<float>{0.F, 0.5F * pi})},
      holotask::syncs::CorrectPhaseSettings{});

  const auto output = from_bytes<ComplexValue>(result.output_bytes[0]);
  ASSERT_EQ(output.size(), 2U);
  EXPECT_NEAR(output[0].real, 1.F, 1e-6F);
  EXPECT_NEAR(output[0].imag, 0.F, 1e-6F);
  EXPECT_NEAR(output[1].real, 0.F, 1e-6F);
  EXPECT_NEAR(output[1].imag, -2.F, 1e-6F);
}

TEST(HoloTaskHeadlessTest, FresnelQuadraticInputAndOutputProduceFiniteFields) {
  const TDesc z_desc = device_desc({1}, DType::F32);
  const float z = 1.0F;
  const auto input = std::vector<std::vector<std::byte>>{as_bytes(std::vector<float>{z})};

  holotask::sources::FresnelQinFactory qin;
  const auto qin_result = holonp_test::run_sync_factory(
      qin, std::vector<TDesc>{z_desc}, input,
      holotask::sources::FresnelQinSettings{.lambda = 1.0F, .dx = 0.25F, .dy = 0.25F,
                                            .nx = 2, .ny = 2});
  ASSERT_EQ(qin_result.output_descs[0].shape, (std::vector<size_t>{2, 2}));

  holotask::sources::FresnelQoutFactory qout;
  const auto qout_result = holonp_test::run_sync_factory(
      qout, std::vector<TDesc>{z_desc}, input,
      holotask::sources::FresnelQoutSettings{.lambda = 1.0F, .dx = 0.25F, .dy = 0.25F,
                                             .nx = 2, .ny = 2});
  const auto output = from_bytes<ComplexValue>(qout_result.output_bytes[0]);
  ASSERT_EQ(output.size(), 4U);
  for (const auto value : output) {
    EXPECT_TRUE(std::isfinite(value.real));
    EXPECT_TRUE(std::isfinite(value.imag));
    EXPECT_NEAR(std::hypot(value.real, value.imag), 1.0F, 1e-5F);
  }
}

TEST(HoloTaskHeadlessTest, HolofileSourceReadsAValidTinyRecording) {
  const auto path = std::filesystem::temp_directory_path() / "holoflow_tiny_source_test.holo";
  const holofile::Header header{
      .magic_number       = holofile::Header::MAGIC_NUMBER_LE,
      .version            = holofile::Header::CURRENT_VERSION,
      .bits_per_pixel     = 8,
      .frame_width        = 2,
      .frame_height       = 2,
      .frame_count        = 1,
      .data_size_in_bytes = 4,
      .endianness         = holofile::Header::LITTLE_ENDIAN,
  };
  const std::vector<std::uint8_t> pixels{1, 2, 3, 4};
  {
    holofile::Writer writer(path.string(), header, holofile::Footer{.pipeline_settings = {}});
    writer.write_frames(pixels.data(), 1);
    writer.write_footer();
  }

  holotask::sources::HolofileFactory factory;
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{}, std::vector<std::vector<std::byte>>{},
      holotask::sources::HolofileSettings{.path = path.string(),
                                          .load_kind = holotask::sources::HolofileSettings::LoadKind::Live,
                                          .start_frame = 0,
                                          .end_frame = 1,
                                          .batch_size = 1,
                                          .max_fps = std::nullopt,
                                          .keep_cursor = true});

  ASSERT_EQ(result.output_descs[0].shape, (std::vector<size_t>{1, 2, 2}));
  const auto output = from_bytes<std::uint8_t>(result.output_bytes[0]);
  EXPECT_EQ(output, pixels);
  std::filesystem::remove(path);
}

TEST(HoloTaskHeadlessTest, HolofileWriterWritesAValidTinyRecording) {
  const auto path = std::filesystem::temp_directory_path() / "holoflow_tiny_writer_test.holo";
  const TDesc desc = host_desc({1, 2, 2}, DType::U8);
  const std::vector<std::uint8_t> pixels{9, 8, 7, 6};

  holotask::sinks::HolofileFactory factory;
  const auto settings = holotask::sinks::HolofileSettings{
      .path = path.string(), .count = 1, .pipeline_settings = nlohmann::json{{"test", true}},
      .use_buffer = true};
  const std::vector<TDesc> input_descs{desc};
  auto task = factory.create(input_descs, settings, {});
  task->bind_logger(spdlog::default_logger());

  holoflow_event::Router router;
  auto                    handles = router.bind_node("writer");
  ASSERT_TRUE(router.ui_try_send(
      "writer", nlohmann::json{{"type", "start_recording"}, {"record_path", path.string()}}));
  router.tick();

  holonp_test::TensorTestBuffer input(desc);
  input.upload(as_bytes(pixels));
  auto view = input.view();
  std::atomic<bool> cancelled{false};
  holoflow::core::SyncCtx ctx{.inputs = {&view, 1},
                              .outputs = {},
                              .cancelled = &cancelled,
                              .event_writer = &handles.out,
                              .event_reader = &handles.in};
  ASSERT_EQ(task->execute(ctx), holoflow::core::OpResult::Ok);
  task.reset();

  {
    holofile::Reader reader(path.string());
    ASSERT_EQ(reader.header().frame_width, 2U);
    ASSERT_EQ(reader.header().frame_height, 2U);
    ASSERT_EQ(reader.header().frame_count, 1U);
    std::vector<std::uint8_t> output(4);
    reader.read_frames(output.data(), 1);
    EXPECT_EQ(output, pixels);
  }
  std::filesystem::remove(path);
}

TEST(HoloTaskHeadlessTest, ConvolutionAcceptsAOneByOneKernel) {
  const auto path = std::filesystem::temp_directory_path() / "holoflow_one_by_one_kernel.json";
  {
    std::ofstream file(path);
    file << R"({"kernel":[[1.0]]})";
  }

  holotask::syncs::ConvolutionFactory factory;
  const TDesc desc = device_desc({2, 2}, DType::F32);
  const std::vector<float> input{1.F, 2.F, 3.F, 4.F};
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      holotask::syncs::ConvolutionSettings{.kernel_file = path.string(), .divide = false});

  ASSERT_EQ(result.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  for (const auto value : from_bytes<float>(result.output_bytes[0])) {
    EXPECT_TRUE(std::isfinite(value));
  }

  const auto divided = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(input)},
      holotask::syncs::ConvolutionSettings{.kernel_file = path.string(), .divide = true});
  for (const auto value : from_bytes<float>(divided.output_bytes[0])) {
    EXPECT_TRUE(std::isfinite(value));
  }
  std::filesystem::remove(path);
}

TEST(HoloTaskHeadlessTest, CorrectPhaseAcceptsComplexInput) {
  holotask::syncs::CorrectPhaseFactory factory;
  const TDesc input_desc = device_desc({1, 1}, DType::CF32);
  const TDesc phase_desc = device_desc({1, 1}, DType::F32);
  const float pi = std::acos(-1.0F);
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{input_desc, phase_desc},
      std::vector<std::vector<std::byte>>{
          as_bytes(std::vector<ComplexValue>{{1.F, 2.F}}), as_bytes(std::vector<float>{0.5F * pi})},
      holotask::syncs::CorrectPhaseSettings{});

  const auto output = from_bytes<ComplexValue>(result.output_bytes[0]);
  ASSERT_EQ(output.size(), 1U);
  EXPECT_NEAR(output[0].real, 2.F, 1e-6F);
  EXPECT_NEAR(output[0].imag, -1.F, 1e-6F);
}

TEST(HoloTaskHeadlessTest, Unfold2DRejectsAnOversizedWindow) {
  holotask::syncs::Unfold2DFactory factory;
  const TDesc desc = device_desc({2, 2}, DType::F32);
  EXPECT_THROW(factory.infer(std::vector<TDesc>{desc},
                             holotask::syncs::Unfold2DSettings{
                                 .win_h = 3, .win_w = 2, .stride_y = 1, .stride_x = 1}),
               std::invalid_argument);
}

TEST(HoloTaskHeadlessTest, CrossCorrelation2PreservesTheMovingShape) {
  holotask::syncs::CrossCorrelation2Factory factory;
  const TDesc desc = device_desc({2, 2}, DType::F32);
  const std::vector<float> image{1.F, 2.F, 3.F, 4.F};
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc, desc},
      std::vector<std::vector<std::byte>>{as_bytes(image), as_bytes(image)},
      holotask::syncs::CrossCorrelation2Settings{});

  ASSERT_EQ(result.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  const auto output = from_bytes<float>(result.output_bytes[0]);
  ASSERT_EQ(output.size(), 4U);
  for (const auto value : output)
    EXPECT_TRUE(std::isfinite(value));
  EXPECT_GT(*std::max_element(output.begin(), output.end()), 0.25F);
}

TEST(HoloTaskHeadlessTest, RegistrationKeepsAConstantFrameFinite) {
  holotask::syncs::RegistrationFactory factory;
  const TDesc desc = device_desc({4, 4}, DType::F32);
  const std::vector<float> image(16, 1.0F);
  const auto result = holonp_test::run_sync_factory(
      factory, std::vector<TDesc>{desc}, std::vector<std::vector<std::byte>>{as_bytes(image)},
      holotask::syncs::RegistrationSettings{.radius = 0.9F});

  ASSERT_EQ(result.output_descs[0].shape, (std::vector<size_t>{4, 4}));
  for (const auto value : from_bytes<float>(result.output_bytes[0]))
    EXPECT_TRUE(std::isfinite(value));
}
