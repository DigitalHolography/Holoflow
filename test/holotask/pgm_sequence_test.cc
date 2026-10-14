// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holotask/sources/pgm_sequence.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::SyncCtx;
using holoflow::core::TDesc;
using holoflow::core::TView;
using holotask::sources::PgmSequenceFactory;
using holotask::sources::PgmSequenceSettings;

class TempDirectory {
public:
  TempDirectory() {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    path_             = std::filesystem::temp_directory_path() /
                        ("holoflow_pgm_sequence_" + std::to_string(unique));
    std::filesystem::create_directories(path_);
  }

  ~TempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
};

void write_binary_pgm(const std::filesystem::path &path, std::size_t width, std::size_t height,
                      unsigned int max_value, const std::vector<std::uint8_t> &raster) {
  std::ofstream output(path, std::ios::binary);
  ASSERT_TRUE(output) << "Could not create " << path;
  output << "P5\n# header comment\n"
         << width << ' ' << height << "\n# comment before maxval\n"
         << max_value << '\n';
  output.write(reinterpret_cast<const char *>(raster.data()),
               static_cast<std::streamsize>(raster.size()));
  ASSERT_TRUE(output) << "Could not write " << path;
}

nlohmann::json make_settings(const std::filesystem::path &directory, int batch_size) {
  return PgmSequenceSettings{
      .directory  = directory.string(),
      .batch_size = batch_size,
  };
}

std::vector<std::uint8_t> execute_and_read_u8(PgmSequenceFactory   &factory,
                                              const nlohmann::json &settings,
                                              const TDesc          &expected_desc) {
  const auto inferred = factory.infer({}, settings);
  EXPECT_EQ(inferred.output_descs.front().shape, expected_desc.shape);
  EXPECT_EQ(inferred.output_descs.front().dtype, expected_desc.dtype);

  holoflow::core::SyncCreateCtx create_ctx{};
  auto                          task = factory.create({}, settings, create_ctx);
  std::vector<TView>            outputs(1);
  SyncCtx                       execution_ctx{
      .inputs       = {},
      .outputs      = outputs,
      .cancelled    = nullptr,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  EXPECT_EQ(task->execute(execution_ctx), OpResult::Ok);
  const auto               *pixels = reinterpret_cast<const std::uint8_t *>(outputs.front().data());
  const auto                count  = expected_desc.num_elements();
  std::vector<std::uint8_t> result(pixels, pixels + count);
  task->release_output(0);
  return result;
}

TEST(PgmSequenceTest, ReadsCommentedFramesInOrderAndSkipsMaskFile) {
  TempDirectory directory;
  write_binary_pgm(directory.path() / "i001.pgm", 2, 2, 255, {5, 6, 7, 8});
  write_binary_pgm(directory.path() / "Image_mask.pgm", 1, 1, 255, {99});
  write_binary_pgm(directory.path() / "i000.pgm", 2, 2, 255, {1, 2, 3, 4});

  PgmSequenceFactory factory;
  const auto         settings = make_settings(directory.path(), 2);
  const TDesc        expected_desc({2, 2, 2}, DType::U8, MemLoc::Host);

  EXPECT_EQ(execute_and_read_u8(factory, settings, expected_desc),
            (std::vector<std::uint8_t>{1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(PgmSequenceTest, ReadsSixteenBitBinaryPixelsAsBigEndian) {
  TempDirectory directory;
  write_binary_pgm(directory.path() / "i000.pgm", 2, 1, 65535, {0x01, 0x02, 0xAB, 0xCD});

  PgmSequenceFactory factory;
  const auto         settings = make_settings(directory.path(), 1);
  const auto         inferred = factory.infer({}, settings);

  ASSERT_EQ(inferred.output_descs.front().dtype, DType::U16);
  ASSERT_EQ(inferred.output_descs.front().shape, (std::vector<std::size_t>{1, 1, 2}));

  holoflow::core::SyncCreateCtx create_ctx{};
  auto                          task = factory.create({}, settings, create_ctx);
  std::vector<TView>            outputs(1);
  SyncCtx                       execution_ctx{
      .inputs       = {},
      .outputs      = outputs,
      .cancelled    = nullptr,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  ASSERT_EQ(task->execute(execution_ctx), OpResult::Ok);
  const auto *pixels = reinterpret_cast<const std::uint16_t *>(outputs.front().data());
  ASSERT_NE(pixels, nullptr);
  EXPECT_EQ(pixels[0], 0x0102);
  EXPECT_EQ(pixels[1], 0xABCD);
  task->release_output(0);
}

TEST(PgmSequenceTest, ReadsAsciiPgmWithComments) {
  TempDirectory directory;
  {
    std::ofstream output(directory.path() / "i000.pgm");
    ASSERT_TRUE(output);
    output << "P2\n# header comment\n2 2\n255\n1 20\n# raster comment\n3 4\n";
  }

  PgmSequenceFactory factory;
  const auto         settings = make_settings(directory.path(), 1);
  const TDesc        expected_desc({1, 2, 2}, DType::U8, MemLoc::Host);

  EXPECT_EQ(execute_and_read_u8(factory, settings, expected_desc),
            (std::vector<std::uint8_t>{1, 20, 3, 4}));
}

TEST(PgmSequenceTest, UpdatingUnchangedSequenceKeepsItsCurrentFrame) {
  TempDirectory directory;
  write_binary_pgm(directory.path() / "i000.pgm", 1, 1, 255, {10});
  write_binary_pgm(directory.path() / "i001.pgm", 1, 1, 255, {20});

  PgmSequenceFactory            factory;
  const auto                    settings = make_settings(directory.path(), 1);
  const auto                    desc     = factory.infer({}, settings).output_descs.front();
  holoflow::core::SyncCreateCtx create_ctx{};
  auto                          task = factory.create({}, settings, create_ctx);
  std::vector<TView>            outputs(1);
  SyncCtx                       execution_ctx{
      .inputs       = {},
      .outputs      = outputs,
      .cancelled    = nullptr,
      .event_writer = nullptr,
      .event_reader = nullptr,
  };

  ASSERT_EQ(task->execute(execution_ctx), OpResult::Ok);
  task->release_output(0);
  task = factory.update(std::move(task), {}, settings, create_ctx);
  ASSERT_EQ(task->execute(execution_ctx), OpResult::Ok);

  const auto *pixels = reinterpret_cast<const std::uint8_t *>(outputs.front().data());
  ASSERT_NE(pixels, nullptr);
  EXPECT_EQ(pixels[0], 20);
  task->release_output(0);
  EXPECT_EQ(desc.shape, (std::vector<std::size_t>{1, 1, 1}));
}

} // namespace
