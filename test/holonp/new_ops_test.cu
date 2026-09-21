#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>
#include <cuComplex.h>
#include <nlohmann/json.hpp>

#include "holoflow/core/tensor.hh"
#include "holonp/argmin.hh"
#include "holonp/angle.hh"
#include "holonp/clip.hh"
#include "holonp/imag.hh"
#include "holonp/isfinite.hh"
#include "holonp/log.hh"
#include "holonp/maximum.hh"
#include "holonp/minimum.hh"
#include "holonp/real.hh"
#include "holonp/sqrt.hh"
#include "holonp/std.hh"
#include "holonp/sum.hh"
#include "holonp/var.hh"
#include "python_oracle.hh"
#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {
const std::filesystem::path kOracle{HOLONP_TEST_ORACLE_SCRIPT};
TDesc desc(std::vector<size_t> shape, DType dtype = DType::F32) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}
template <typename T> std::vector<std::byte> bytes(const std::vector<T>& values) {
  std::vector<std::byte> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}
void expect_f32(const std::vector<std::byte>& actual, const std::vector<std::byte>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  const auto* a = reinterpret_cast<const float*>(actual.data());
  const auto* e = reinterpret_cast<const float*>(expected.data());
  for (size_t i = 0; i < actual.size() / sizeof(float); ++i)
    EXPECT_NEAR(a[i], e[i], 2e-4f * std::max(std::abs(e[i]), 1.0f));
}
template <typename Factory>
void expect_oracle(Factory& factory, const char* op, const TDesc& input,
                   const std::vector<std::byte>& data, const nlohmann::json& settings) {
  const std::vector<TDesc> descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto run = holonp_test::run_sync_factory(factory, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op=op, .n_outputs=1, .input_descs=descriptors,
                                   .input_bytes=inputs, .settings=settings};
  const auto expected = holonp_test::invoke_oracle(request, kOracle);
  expect_f32(run.output_bytes[0], expected.output_bytes[0]);
}
}

TEST(NewReductionTest, GlobalSumStdVarAndArgmin) {
  const TDesc input = desc({2, 3});
  const auto data = bytes(std::vector<float>{3.f, 1.f, 2.f, 8.f, 5.f, 7.f});
  const nlohmann::json settings{{"axis", nullptr}, {"keepdims", true}};
  holonp::SumFactory sum; expect_oracle(sum, "sum", input, data, settings);
  holonp::StdFactory standard; expect_oracle(standard, "std", input, data, settings);
  holonp::VarFactory variance; expect_oracle(variance, "var", input, data, settings);
  holonp::ArgminFactory argmin;
  const std::vector<TDesc> descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto run = holonp_test::run_sync_factory(argmin, descriptors, inputs, settings);
  ASSERT_EQ(run.output_bytes[0].size(), sizeof(std::uint16_t));
  EXPECT_EQ(*reinterpret_cast<const std::uint16_t*>(run.output_bytes[0].data()), 1);
}

TEST(NewReductionTest, AxisAndKeepdimsMatchOracle) {
  const TDesc input = desc({2, 3});
  const auto data = bytes(std::vector<float>{3.f, 1.f, 2.f, 8.f, 5.f, 7.f});

  holonp::SumFactory sum;
  expect_oracle(sum, "sum", input, data, {{"axis", 1}});
  expect_oracle(sum, "sum", input, data, {{"axis", -1}, {"keepdims", true}});

  holonp::StdFactory standard;
  expect_oracle(standard, "std", input, data, {{"axis", 1}});
  holonp::VarFactory variance;
  expect_oracle(variance, "var", input, data, {{"axis", 0}, {"keepdims", true}});
}

TEST(NewReductionTest, HonorsNonContiguousInputStrides) {
  TDesc input = desc({2, 3});
  input.strides = {16, 4};
  const auto data = bytes(std::vector<float>{3.f, 1.f, 2.f, 99.f, 8.f, 5.f, 7.f, 88.f});
  const std::vector<TDesc> descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};

  holonp::SumFactory sum;
  const auto run = holonp_test::run_sync_factory(sum, descriptors, inputs, {{"axis", 1}});
  ASSERT_EQ(run.output_bytes[0].size(), 2 * sizeof(float));
  const auto* output = reinterpret_cast<const float*>(run.output_bytes[0].data());
  EXPECT_FLOAT_EQ(output[0], 6.f);
  EXPECT_FLOAT_EQ(output[1], 20.f);
}

TEST(NewUnaryTest, SqrtLogClipAndFinite) {
  const TDesc input = desc({4});
  const auto data = bytes(std::vector<float>{1.f, 4.f, 9.f, 16.f});
  holonp::SqrtFactory sqrt; expect_oracle(sqrt, "sqrt", input, data, {});
  holonp::LogFactory log; expect_oracle(log, "log", input, data, {});
  holonp::ClipFactory clip;
  expect_oracle(clip, "clip", input, data, {{"min", 2.f}, {"max", 10.f}});
  holonp::IsfiniteFactory finite; expect_oracle(finite, "isfinite", input, data, {});
}

TEST(NewUnaryTest, ComplexSqrtPreservesComplexOutput) {
  const TDesc input = desc({2}, DType::CF32);
  const auto data = bytes(std::vector<cuFloatComplex>{make_cuFloatComplex(3.f, 4.f),
                                                       make_cuFloatComplex(0.f, -4.f)});
  holonp::SqrtFactory sqrt;
  const std::vector<TDesc> descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto inferred = sqrt.infer(descriptors, {});
  ASSERT_EQ(inferred.output_descs[0].dtype, DType::CF32);
  const auto run = holonp_test::run_sync_factory(sqrt, descriptors, inputs, {});
  const auto* output = reinterpret_cast<const cuFloatComplex*>(run.output_bytes[0].data());
  EXPECT_NEAR(output[0].x, 2.f, 1e-5f);
  EXPECT_NEAR(output[0].y, 1.f, 1e-5f);
  EXPECT_NEAR(output[1].x, 1.4142135f, 1e-5f);
  EXPECT_NEAR(output[1].y, -1.4142135f, 1e-5f);
}

TEST(NewUnaryTest, ComplexLogIsRejected) {
  holonp::LogFactory log;
  const std::vector<TDesc> descriptors{desc({2}, DType::CF32)};
  EXPECT_THROW(log.infer(descriptors, {}), std::invalid_argument);
}

TEST(NewBinaryTest, MaximumAndMinimum) {
  const TDesc input = desc({4});
  const auto a = bytes(std::vector<float>{1.f, 8.f, 3.f, 7.f});
  const auto b = bytes(std::vector<float>{4.f, 2.f, 6.f, 5.f});
  holonp::MaximumFactory maximum;
  const std::vector<TDesc> descriptors{input, input};
  const std::vector<std::vector<std::byte>> inputs{a, b};
  const auto max_run = holonp_test::run_sync_factory(maximum, descriptors, inputs, {});
  EXPECT_EQ(max_run.output_bytes[0], bytes(std::vector<float>{4.f, 8.f, 6.f, 7.f}));
  holonp::MinimumFactory minimum;
  const auto min_run = holonp_test::run_sync_factory(minimum, descriptors, inputs, {});
  EXPECT_EQ(min_run.output_bytes[0], bytes(std::vector<float>{1.f, 2.f, 3.f, 5.f}));
}

TEST(NewSettingsTest, ClipRejectsInvertedBounds) {
  holonp::ClipFactory factory;
  const std::vector<TDesc> descriptors{desc({2})};
  EXPECT_THROW(factory.infer(descriptors, {{"min", 2.f}, {"max", 1.f}}), std::invalid_argument);
}
