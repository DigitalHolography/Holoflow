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

#include <boost/graph/adjacency_list.hpp>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "holoflow/core/tensor.hh"

// -------------------------------------------------------------------------------------------------
// Tensor descriptors
// -------------------------------------------------------------------------------------------------

using TDesc  = holoflow::core::TDesc;
using DType  = holoflow::core::DType;
using MemLoc = holoflow::core::MemLoc;

TEST(TensorDescriptorTest, CreatesContiguousStridesAndSizes) {
  const TDesc desc({2, 3, 4}, DType::F32, MemLoc::Host);

  EXPECT_EQ(desc.rank(), 3);
  EXPECT_EQ(desc.strides, (std::vector<size_t>{48, 16, 4}));
  EXPECT_EQ(desc.num_elements(), 24);
  EXPECT_EQ(desc.num_bytes(), 96);
}

TEST(TensorDescriptorTest, HandlesEmptyAndZeroElementShapes) {
  EXPECT_EQ(TDesc({}, DType::U8, MemLoc::Host).num_bytes(), 0);
  EXPECT_EQ(TDesc({3, 0, 2}, DType::U16, MemLoc::Host).num_elements(), 0);
}

TEST(TensorDescriptorTest, DetectsElementCountOverflow) {
  const TDesc desc({std::numeric_limits<size_t>::max(), 2}, DType::U8, MemLoc::Host);
  EXPECT_THROW((void)desc.num_elements(), std::overflow_error);
}

TEST(TensorDescriptorTest, SerializesDtypeMemoryAndStrides) {
  const TDesc original({2, 3}, DType::CF32, MemLoc::Device, std::vector<size_t>{32, 8});
  const auto  encoded = nlohmann::json(original);
  const auto  decoded = encoded.get<TDesc>();

  EXPECT_EQ(decoded.shape, original.shape);
  EXPECT_EQ(decoded.strides, original.strides);
  EXPECT_EQ(decoded.dtype, original.dtype);
  EXPECT_EQ(decoded.mem_loc, original.mem_loc);
  EXPECT_THROW((void)nlohmann::json("bad").get<DType>(), std::invalid_argument);
  EXPECT_THROW((void)nlohmann::json("bad").get<MemLoc>(), std::invalid_argument);
}

TEST(TensorDescriptorTest, CoversEveryEnumAndConstructor) {
  using holoflow::core::DType;
  using holoflow::core::MemLoc;

  EXPECT_EQ(holoflow::core::size_of(DType::U8), 1);
  EXPECT_EQ(holoflow::core::size_of(DType::U16), 2);
  EXPECT_EQ(holoflow::core::size_of(DType::F32), 4);
  EXPECT_EQ(holoflow::core::size_of(DType::CF32), 8);
  EXPECT_EQ(holoflow::core::to_string(DType::U8), "U8");
  EXPECT_EQ(holoflow::core::to_string(DType::U16), "U16");
  EXPECT_EQ(holoflow::core::to_string(DType::F32), "F32");
  EXPECT_EQ(holoflow::core::to_string(DType::CF32), "CF32");
  EXPECT_EQ(holoflow::core::to_string(MemLoc::Host), "Host");
  EXPECT_EQ(holoflow::core::to_string(MemLoc::Device), "Device");

  for (const auto dtype : {DType::U8, DType::U16, DType::F32, DType::CF32}) {
    EXPECT_EQ(nlohmann::json(dtype).get<DType>(), dtype);
  }
  for (const auto location : {MemLoc::Host, MemLoc::Device}) {
    EXPECT_EQ(nlohmann::json(location).get<MemLoc>(), location);
  }

  const holoflow::core::TDesc offset_desc({2, 3}, DType::U16, MemLoc::Host, 7);
  EXPECT_EQ(offset_desc.strides, (std::vector<size_t>{6, 2}));
  EXPECT_EQ(offset_desc.offset, 7);
  const holoflow::core::TDesc custom_desc({2, 3}, DType::U16, MemLoc::Host,
                                          std::vector<size_t>{16, 4}, 9);
  EXPECT_EQ(custom_desc.strides, (std::vector<size_t>{16, 4}));
  EXPECT_EQ(custom_desc.offset, 9);
  EXPECT_EQ(custom_desc.num_bytes(), 32);
}

// -------------------------------------------------------------------------------------------------
// Tensor
// -------------------------------------------------------------------------------------------------

TEST(TensorTest, AllocatesHostStorageAndExposesOffsetView) {
  holoflow::core::Tensor tensor(
      holoflow::core::TDesc({4}, holoflow::core::DType::F32, holoflow::core::MemLoc::Host));
  ASSERT_NE(tensor.data(), nullptr);
  EXPECT_EQ(std::as_const(tensor).data(), tensor.data());
  EXPECT_EQ(tensor.desc().num_bytes(), 16);
  auto view = tensor.view();
  EXPECT_FALSE(view.is_nullptr());
  EXPECT_EQ(view.data(), tensor.data());

  std::array<std::byte, 16> bytes{};
  holoflow::core::Storage   storage{holoflow::core::MemLoc::Host, bytes.size(), bytes.data()};
  holoflow::core::TView     offset_view{
      holoflow::core::TDesc({2}, holoflow::core::DType::U8, holoflow::core::MemLoc::Host, 3),
      &storage};
  EXPECT_EQ(offset_view.data(), bytes.data() + 3);
  EXPECT_TRUE(holoflow::core::TView{}.is_nullptr());
}
