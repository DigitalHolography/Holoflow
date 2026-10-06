#include <gtest/gtest.h>

#include <array>
#include <cstddef>
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
