#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include <utility>

#include "holotask/syncs/resize.hh"
#include "sync_task_runner.hh"

namespace {
using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

TEST(ResizeTest, StretchesLandscapeU8ToSquare) {
  const TDesc input_desc({1, 2, 3}, DType::U8, MemLoc::Host);
  const std::array<std::byte, 6> input{
      std::byte{0}, std::byte{10}, std::byte{20}, std::byte{30}, std::byte{40}, std::byte{50}};
  const std::vector<TDesc> inputs{input_desc};
  const std::vector<std::vector<std::byte>> data{std::vector(input.begin(), input.end())};

  const auto result = holonp_test::run_sync_factory(
      holotask::syncs::ResizeFactory{}, inputs, data,
      holotask::syncs::ResizeSettings{3, 3});

  EXPECT_EQ(result.output_descs[0].shape, (std::vector<std::size_t>{1, 3, 3}));
  ASSERT_EQ(result.output_bytes[0].size(), 9);
  EXPECT_EQ(static_cast<unsigned char>(result.output_bytes[0][3]), 15);
  EXPECT_EQ(static_cast<unsigned char>(result.output_bytes[0][4]), 25);
  EXPECT_EQ(static_cast<unsigned char>(result.output_bytes[0][8]), 50);
}

TEST(ResizeTest, PreservesU16AndBatch) {
  const TDesc input_desc({2, 2, 1}, DType::U16, MemLoc::Host);
  const std::array<std::byte, 8> input{
      std::byte{0x00}, std::byte{0x01}, std::byte{0x00}, std::byte{0x02},
      std::byte{0x00}, std::byte{0x03}, std::byte{0x00}, std::byte{0x04}};
  const std::vector<TDesc> inputs{input_desc};
  const std::vector<std::vector<std::byte>> data{std::vector(input.begin(), input.end())};
  const auto result = holonp_test::run_sync_factory(
      holotask::syncs::ResizeFactory{}, inputs, data,
      holotask::syncs::ResizeSettings{2, 2});

  EXPECT_EQ(result.output_descs[0].dtype, DType::U16);
  EXPECT_EQ(result.output_descs[0].shape, (std::vector<std::size_t>{2, 2, 2}));
  const std::vector<std::byte> expected{
      std::byte{0x00}, std::byte{0x01}, std::byte{0x00}, std::byte{0x01},
      std::byte{0x00}, std::byte{0x02}, std::byte{0x00}, std::byte{0x02},
      std::byte{0x00}, std::byte{0x03}, std::byte{0x00}, std::byte{0x03},
      std::byte{0x00}, std::byte{0x04}, std::byte{0x00}, std::byte{0x04}};
  EXPECT_EQ(result.output_bytes[0], expected);
}

TEST(ResizeTest, CudaBilinearMatchesCpuForU8) {
  const TDesc input_desc({1, 2, 3}, DType::U8, MemLoc::Device);
  const std::array<std::byte, 6> input{
      std::byte{0}, std::byte{10}, std::byte{20}, std::byte{30}, std::byte{40}, std::byte{50}};
  const std::vector<TDesc> inputs{input_desc};
  const std::vector<std::vector<std::byte>> data{std::vector(input.begin(), input.end())};
  const auto settings = holotask::syncs::ResizeSettings{
      3, 3, holotask::syncs::ResizeInterpolation::Bilinear,
      holotask::syncs::ResizeAlgorithm::CudaBilinear};

  const auto result = holonp_test::run_sync_factory(holotask::syncs::ResizeFactory{}, inputs, data,
                                                     settings);

  EXPECT_EQ(result.output_descs[0].mem_loc, MemLoc::Device);
  ASSERT_EQ(result.output_bytes[0].size(), 9);
  EXPECT_EQ(static_cast<unsigned char>(result.output_bytes[0][3]), 15);
  EXPECT_EQ(static_cast<unsigned char>(result.output_bytes[0][4]), 25);
  EXPECT_EQ(static_cast<unsigned char>(result.output_bytes[0][8]), 50);
}

TEST(ResizeTest, CudaBilinearSupportsU16AndLargeSquareOutput) {
  const TDesc input_desc({1, 320, 512}, DType::U16, MemLoc::Device);
  std::vector<std::byte> input(input_desc.num_bytes());
  auto *pixels = reinterpret_cast<std::uint16_t *>(input.data());
  for (std::size_t y = 0; y < 320; ++y) {
    for (std::size_t x = 0; x < 512; ++x) {
      pixels[y * 512 + x] = static_cast<std::uint16_t>((x + y) % 65536);
    }
  }
  const std::vector<TDesc> inputs{input_desc};
  const std::vector<std::vector<std::byte>> data{std::move(input)};
  const auto settings = holotask::syncs::ResizeSettings{
      512, 512, holotask::syncs::ResizeInterpolation::Bilinear,
      holotask::syncs::ResizeAlgorithm::CudaBilinear};

  const auto result = holonp_test::run_sync_factory(holotask::syncs::ResizeFactory{}, inputs, data,
                                                     settings);

  EXPECT_EQ(result.output_descs[0].shape, (std::vector<std::size_t>{1, 512, 512}));
  EXPECT_EQ(result.output_descs[0].mem_loc, MemLoc::Device);
  ASSERT_EQ(result.output_bytes[0].size(), 512u * 512u * sizeof(std::uint16_t));
  const auto *output = reinterpret_cast<const std::uint16_t *>(result.output_bytes[0].data());
  EXPECT_EQ(output[0], 0);
  EXPECT_EQ(output[511], 511);
  EXPECT_EQ(output[511 * 512], 319);
}

TEST(ResizeTest, RejectsInvalidInputAndDimensions) {
  holotask::syncs::ResizeFactory factory;
  const std::vector<TDesc> f32{TDesc({1, 2, 3}, DType::F32, MemLoc::Host)};
  const std::vector<TDesc> device{TDesc({1, 2, 3}, DType::U8, MemLoc::Device)};
  const std::vector<TDesc> u8{TDesc({1, 2, 3}, DType::U8, MemLoc::Host)};
  EXPECT_THROW(factory.infer(f32,
                             holotask::syncs::ResizeSettings{3, 3}), std::invalid_argument);
  EXPECT_THROW(factory.infer(device,
                             holotask::syncs::ResizeSettings{3, 3}), std::invalid_argument);
  EXPECT_THROW(factory.infer(u8,
                             holotask::syncs::ResizeSettings{
                                 3, 3, holotask::syncs::ResizeInterpolation::Bilinear,
                                 holotask::syncs::ResizeAlgorithm::CudaBilinear}),
               std::invalid_argument);
  EXPECT_THROW(factory.infer(u8,
                             holotask::syncs::ResizeSettings{0, 3}), std::invalid_argument);
}

TEST(ResizeTest, ReusesTaskWhenSettingsAreUnchanged) {
  holotask::syncs::ResizeFactory factory;
  const std::vector<TDesc> inputs{TDesc({1, 2, 3}, DType::U8, MemLoc::Host)};
  const auto settings = holotask::syncs::ResizeSettings{3, 3};
  auto task = factory.create(inputs, settings, {});
  auto *old = task.get();
  task = factory.update(std::move(task), inputs, settings, {});
  EXPECT_EQ(task.get(), old);
  task = factory.update(std::move(task), inputs, holotask::syncs::ResizeSettings{4, 4}, {});
  EXPECT_NE(task.get(), old);
}
} // namespace
