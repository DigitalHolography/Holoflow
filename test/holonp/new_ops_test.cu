#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <cuComplex.h>
#include <filesystem>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "holoflow/core/tensor.hh"
#include "holonp/angle.hh"
#include "holonp/argmin.hh"
#include "holonp/clip.hh"
#include "holonp/convolve.hh"
#include "holonp/correlate.hh"
#include "holonp/diff.hh"
#include "holonp/gradient.hh"
#include "holonp/histogram.hh"
#include "holonp/imag.hh"
#include "holonp/isfinite.hh"
#include "holonp/log.hh"
#include "holonp/lstsq.hh"
#include "holonp/maximum.hh"
#include "holonp/median.hh"
#include "holonp/minimum.hh"
#include "holonp/norm.hh"
#include "holonp/percentile.hh"
#include "holonp/pinv.hh"
#include "holonp/quantile.hh"
#include "holonp/real.hh"
#include "holonp/sqrt.hh"
#include "holonp/std.hh"
#include "holonp/sum.hh"
#include "holonp/svd.hh"
#include "holonp/var.hh"
#include "python_oracle.hh"
#include "sync_task_runner.hh"

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::TDesc;

namespace {
const std::filesystem::path kOracle{HOLONP_TEST_ORACLE_SCRIPT};
TDesc                       desc(std::vector<size_t> shape, DType dtype = DType::F32) {
  return TDesc(std::move(shape), dtype, MemLoc::Device);
}
template <typename T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  std::vector<std::byte> result(values.size() * sizeof(T));
  std::memcpy(result.data(), values.data(), result.size());
  return result;
}
void expect_f32(const std::vector<std::byte> &actual, const std::vector<std::byte> &expected) {
  ASSERT_EQ(actual.size(), expected.size());
  const auto *a = reinterpret_cast<const float *>(actual.data());
  const auto *e = reinterpret_cast<const float *>(expected.data());
  for (size_t i = 0; i < actual.size() / sizeof(float); ++i)
    EXPECT_NEAR(a[i], e[i], 2e-4f * std::max(std::abs(e[i]), 1.0f));
}
template <typename Factory>
void expect_oracle(Factory &factory, const char *op, const TDesc &input,
                   const std::vector<std::byte> &data, const nlohmann::json &settings) {
  const std::vector<TDesc>                  descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto run = holonp_test::run_sync_factory(factory, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = op,
                                   .n_outputs   = 1,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  expect_f32(run.output_bytes[0], expected.output_bytes[0]);
}
} // namespace

TEST(NewReductionTest, GlobalSumStdVarAndArgmin) {
  const TDesc          input = desc({2, 3});
  const auto           data  = bytes(std::vector<float>{3.f, 1.f, 2.f, 8.f, 5.f, 7.f});
  const nlohmann::json settings{{"axis", nullptr}, {"keepdims", true}};
  holonp::SumFactory   sum;
  expect_oracle(sum, "sum", input, data, settings);
  holonp::StdFactory standard;
  expect_oracle(standard, "std", input, data, settings);
  holonp::VarFactory variance;
  expect_oracle(variance, "var", input, data, settings);
  holonp::ArgminFactory                     argmin;
  const std::vector<TDesc>                  descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto run = holonp_test::run_sync_factory(argmin, descriptors, inputs, settings);
  ASSERT_EQ(run.output_bytes[0].size(), sizeof(std::uint16_t));
  EXPECT_EQ(*reinterpret_cast<const std::uint16_t *>(run.output_bytes[0].data()), 1);
}

TEST(NewReductionTest, GlobalReductionsProduceNumPyScalars) {
  const TDesc          input = desc({2, 3});
  const auto           data  = bytes(std::vector<float>{3.f, 1.f, 2.f, 8.f, 5.f, 7.f});
  const nlohmann::json settings{{"axis", nullptr}};

  holonp::SumFactory                        sum;
  const std::vector<TDesc>                  descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto sum_run = holonp_test::run_sync_factory(sum, descriptors, inputs, settings);
  holonp_test::OracleInput sum_request{.op          = "sum",
                                       .n_outputs   = 1,
                                       .input_descs = descriptors,
                                       .input_bytes = inputs,
                                       .settings    = settings};
  const auto               sum_oracle = holonp_test::invoke_oracle(sum_request, kOracle);
  expect_f32(sum_run.output_bytes[0], sum_oracle.output_bytes[0]);

  holonp::ArgminFactory argmin;
  const auto argmin_run = holonp_test::run_sync_factory(argmin, descriptors, inputs, settings);
  ASSERT_EQ(argmin_run.output_bytes[0].size(), sizeof(std::uint16_t));
  EXPECT_EQ(*reinterpret_cast<const std::uint16_t *>(argmin_run.output_bytes[0].data()), 1);
}

TEST(NewReductionTest, AxisAndKeepdimsMatchOracle) {
  const TDesc input = desc({2, 3});
  const auto  data  = bytes(std::vector<float>{3.f, 1.f, 2.f, 8.f, 5.f, 7.f});

  holonp::SumFactory sum;
  expect_oracle(sum, "sum", input, data, {{"axis", 1}});
  expect_oracle(sum, "sum", input, data, {{"axis", -1}, {"keepdims", true}});

  holonp::StdFactory standard;
  expect_oracle(standard, "std", input, data, {{"axis", 1}});
  holonp::VarFactory variance;
  expect_oracle(variance, "var", input, data, {{"axis", 0}, {"keepdims", true}});
}

TEST(NewReductionTest, HonorsNonContiguousInputStrides) {
  TDesc input     = desc({2, 3});
  input.strides   = {16, 4};
  const auto data = bytes(std::vector<float>{3.f, 1.f, 2.f, 99.f, 8.f, 5.f, 7.f, 88.f});
  const std::vector<TDesc>                  descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};

  holonp::SumFactory sum;
  const auto         run = holonp_test::run_sync_factory(sum, descriptors, inputs, {{"axis", 1}});
  ASSERT_EQ(run.output_bytes[0].size(), 2 * sizeof(float));
  const auto *output = reinterpret_cast<const float *>(run.output_bytes[0].data());
  EXPECT_FLOAT_EQ(output[0], 6.f);
  EXPECT_FLOAT_EQ(output[1], 20.f);
}

TEST(NewReductionTest, MedianAndQuantileMatchNumpy) {
  const TDesc input = desc({2, 4});
  const auto  data  = bytes(std::vector<float>{7.f, 1.f, 4.f, 2.f, 9.f, 3.f, 8.f, 6.f});

  holonp::MedianFactory median;
  expect_oracle(median, "median", input, data, {});
  const nlohmann::json axis_settings{{"axis", -1}, {"keepdims", true}};
  expect_oracle(median, "median", input, data, axis_settings);

  holonp::QuantileFactory quantile;
  const nlohmann::json    quantile_settings{{"q", 0.25f}, {"axis", 1}, {"keepdims", true}};
  expect_oracle(quantile, "quantile", input, data, quantile_settings);

  const auto                 invalid = nlohmann::json{{"q", 1.1f}};
  const std::array<TDesc, 1> descriptors{input};
  EXPECT_THROW(quantile.infer(descriptors, invalid), std::invalid_argument);
}

TEST(NewStatisticsTest, PercentileAndHistogramMatchNumpy) {
  const TDesc input = desc({2, 4});
  const auto  data  = bytes(std::vector<float>{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 8.f});

  holonp::PercentileFactory percentile;
  expect_oracle(percentile, "percentile", input, data,
                {{"q", 75.f}, {"axis", 1}, {"keepdims", false}});

  const TDesc                               histogram_input = desc({8});
  holonp::HistogramFactory                  histogram;
  const nlohmann::json                      settings{{"bins", 4}, {"min", 0.f}, {"max", 8.f}};
  const std::vector<TDesc>                  descriptors{histogram_input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto run = holonp_test::run_sync_factory(histogram, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = "histogram",
                                   .n_outputs   = 2,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  expect_f32(run.output_bytes[0], expected.output_bytes[0]);
  expect_f32(run.output_bytes[1], expected.output_bytes[1]);
}

TEST(NewNormTest, MatchesNumpyForGlobalAndAxisReductions) {
  const TDesc         input = desc({6});
  const auto          data  = bytes(std::vector<float>{3.f, 4.f, 0.f, 5.f, 12.f, 0.f});
  holonp::NormFactory norm;
  expect_oracle(norm, "norm", input, data, {{"axis", nullptr}, {"keepdims", true}});
  const TDesc axis_input = desc({2, 3});
  expect_oracle(norm, "norm", axis_input, data, {{"axis", 1}, {"ord", 1.0}});
}

TEST(NewNormTest, SupportsComplexInputAndInfinityNorm) {
  const TDesc input = desc({2}, DType::CF32);
  const auto  data  = bytes(
      std::vector<cuFloatComplex>{make_cuFloatComplex(3.f, 4.f), make_cuFloatComplex(0.f, 12.f)});
  holonp::NormFactory norm;
  expect_oracle(norm, "norm", input, data, {{"ord", "inf"}});
}

TEST(NewDiffTest, MatchesNumpyForHigherOrderAxisDifference) {
  const TDesc input = desc({2, 5});
  const auto  data =
      bytes(std::vector<float>{1.f, 4.f, 9.f, 16.f, 25.f, 2.f, 8.f, 18.f, 32.f, 50.f});
  holonp::DiffFactory diff;
  expect_oracle(diff, "diff", input, data, {{"axis", 1}, {"n", 2}});
}

TEST(NewDiffTest, SupportsComplexInput) {
  const TDesc         input = desc({3}, DType::CF32);
  const auto          data  = bytes(std::vector<cuFloatComplex>{
      make_cuFloatComplex(1.f, 1.f), make_cuFloatComplex(3.f, 2.f), make_cuFloatComplex(6.f, 4.f)});
  holonp::DiffFactory diff;
  const auto          descriptors = std::vector<TDesc>{input};
  const auto          inputs      = std::vector<std::vector<std::byte>>{data};
  const auto          run = holonp_test::run_sync_factory(diff, descriptors, inputs, {{"axis", 0}});
  holonp_test::OracleInput request{.op          = "diff",
                                   .n_outputs   = 1,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = {{"axis", 0}}};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  ASSERT_EQ(run.output_bytes[0].size(), expected.output_bytes[0].size());
  const auto *actual = reinterpret_cast<const float *>(run.output_bytes[0].data());
  const auto *oracle = reinterpret_cast<const float *>(expected.output_bytes[0].data());
  for (size_t i = 0; i < run.output_bytes[0].size() / sizeof(float); ++i)
    EXPECT_NEAR(actual[i], oracle[i], 1e-4f * std::max(std::abs(oracle[i]), 1.0f));
}

TEST(NewGradientTest, ProducesAllAxisOutputsMatchingNumpy) {
  const TDesc input = desc({3, 4});
  const auto  data =
      bytes(std::vector<float>{1.f, 2.f, 4.f, 7.f, 2.f, 4.f, 8.f, 14.f, 3.f, 6.f, 12.f, 21.f});
  holonp::GradientFactory                   gradient;
  const std::vector<TDesc>                  descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const nlohmann::json                      settings{{"axis", nullptr}, {"edge_order", 2}};
  const auto run = holonp_test::run_sync_factory(gradient, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = "gradient",
                                   .n_outputs   = 2,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  ASSERT_EQ(run.output_bytes.size(), expected.output_bytes.size());
  for (size_t i = 0; i < run.output_bytes.size(); ++i)
    expect_f32(run.output_bytes[i], expected.output_bytes[i]);
}

TEST(NewGradientTest, SupportsAxisSelectionSpacingAndComplexInput) {
  const TDesc                               input = desc({2, 4}, DType::CF32);
  const auto                                data  = bytes(std::vector<cuFloatComplex>{
      make_cuFloatComplex(0.f, 1.f), make_cuFloatComplex(2.f, 3.f), make_cuFloatComplex(6.f, 7.f),
      make_cuFloatComplex(12.f, 13.f), make_cuFloatComplex(1.f, 2.f), make_cuFloatComplex(3.f, 5.f),
      make_cuFloatComplex(7.f, 8.f), make_cuFloatComplex(13.f, 14.f)});
  holonp::GradientFactory                   gradient;
  const std::vector<TDesc>                  descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const nlohmann::json settings{{"axis", 1}, {"spacing", 2.0}, {"edge_order", 1}};
  const auto           run = holonp_test::run_sync_factory(gradient, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = "gradient",
                                   .n_outputs   = 1,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  ASSERT_EQ(run.output_bytes.size(), 1u);
  ASSERT_EQ(run.output_bytes[0].size(), expected.output_bytes[0].size());
  const auto *actual = reinterpret_cast<const float *>(run.output_bytes[0].data());
  const auto *oracle = reinterpret_cast<const float *>(expected.output_bytes[0].data());
  for (size_t i = 0; i < run.output_bytes[0].size() / sizeof(float); ++i)
    EXPECT_NEAR(actual[i], oracle[i], 2e-4f * std::max(std::abs(oracle[i]), 1.0f));
}

TEST(NewConvolutionTest, MatchesNumpyForModes) {
  const TDesc                               a      = desc({3});
  const TDesc                               b      = desc({2});
  const auto                                a_data = bytes(std::vector<float>{1.f, 2.f, 3.f});
  const auto                                b_data = bytes(std::vector<float>{4.f, 5.f});
  const std::vector<TDesc>                  descriptors{a, b};
  const std::vector<std::vector<std::byte>> inputs{a_data, b_data};

  for (const char *mode : {"full", "same", "valid"}) {
    holonp::ConvolveFactory factory;
    const nlohmann::json    settings{{"mode", mode}};
    const auto run = holonp_test::run_sync_factory(factory, descriptors, inputs, settings);
    holonp_test::OracleInput request{.op          = "convolve",
                                     .n_outputs   = 1,
                                     .input_descs = descriptors,
                                     .input_bytes = inputs,
                                     .settings    = settings};
    const auto               expected = holonp_test::invoke_oracle(request, kOracle);
    expect_f32(run.output_bytes[0], expected.output_bytes[0]);
  }
}

TEST(NewCorrelationTest, MatchesNumpyForComplexInput) {
  const TDesc a      = desc({3}, DType::CF32);
  const TDesc b      = desc({2}, DType::CF32);
  const auto  a_data = bytes(std::vector<cuFloatComplex>{make_cuFloatComplex(1.f, 2.f),
                                                         make_cuFloatComplex(3.f, 1.f),
                                                         make_cuFloatComplex(2.f, -1.f)});
  const auto  b_data = bytes(
      std::vector<cuFloatComplex>{make_cuFloatComplex(2.f, 1.f), make_cuFloatComplex(1.f, -2.f)});
  const std::vector<TDesc>                  descriptors{a, b};
  const std::vector<std::vector<std::byte>> inputs{a_data, b_data};
  holonp::CorrelateFactory                  factory;
  const nlohmann::json                      settings{{"mode", "same"}};
  const auto run = holonp_test::run_sync_factory(factory, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = "correlate",
                                   .n_outputs   = 1,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  expect_f32(run.output_bytes[0], expected.output_bytes[0]);
}

TEST(NewSVDTest, ThinRealSVDMatchesNumpySingularValuesAndReconstructs) {
  const TDesc              input = desc({2, 3});
  const auto               data  = bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  holonp::SVDFactory       factory;
  const std::vector<TDesc> descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const nlohmann::json                      settings{{"full_matrices", false}};
  const auto                                inferred = factory.infer(descriptors, settings);
  ASSERT_EQ(inferred.output_descs.size(), 3u);
  EXPECT_EQ(inferred.output_descs[0].shape, (std::vector<size_t>{2, 2}));
  EXPECT_EQ(inferred.output_descs[1].shape, (std::vector<size_t>{2}));
  EXPECT_EQ(inferred.output_descs[2].shape, (std::vector<size_t>{2, 3}));

  const auto run = holonp_test::run_sync_factory(factory, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = "svd",
                                   .n_outputs   = 3,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  expect_f32(run.output_bytes[1], expected.output_bytes[1]);

  const auto              *u  = reinterpret_cast<const float *>(run.output_bytes[0].data());
  const auto              *s  = reinterpret_cast<const float *>(run.output_bytes[1].data());
  const auto              *vh = reinterpret_cast<const float *>(run.output_bytes[2].data());
  const std::vector<float> reconstructed{
      u[0] * s[0] * vh[0] + u[1] * s[1] * vh[3], u[0] * s[0] * vh[1] + u[1] * s[1] * vh[4],
      u[0] * s[0] * vh[2] + u[1] * s[1] * vh[5], u[2] * s[0] * vh[0] + u[3] * s[1] * vh[3],
      u[2] * s[0] * vh[1] + u[3] * s[1] * vh[4], u[2] * s[0] * vh[2] + u[3] * s[1] * vh[5]};
  const auto original = std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  for (size_t i = 0; i < original.size(); ++i)
    EXPECT_NEAR(reconstructed[i], original[i], 2e-3f);
}

TEST(NewSVDTest, RejectsUnsupportedModes) {
  holonp::SVDFactory       factory;
  const std::vector<TDesc> input{desc({2, 3}, DType::CF32)};
  EXPECT_THROW(factory.infer(input, {}), std::invalid_argument);
  const std::vector<TDesc> real_input{desc({2, 3})};
  EXPECT_THROW(factory.infer(real_input, {{"full_matrices", true}}), std::invalid_argument);
}

TEST(NewPinvTest, MatchesNumpyForRectangularMatrix) {
  const TDesc              input = desc({2, 3});
  const auto               data  = bytes(std::vector<float>{1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  holonp::PinvFactory      factory;
  const std::vector<TDesc> descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const nlohmann::json                      settings{{"rcond", 1e-6f}};
  const auto run = holonp_test::run_sync_factory(factory, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = "pinv",
                                   .n_outputs   = 1,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  expect_f32(run.output_bytes[0], expected.output_bytes[0]);
}

TEST(NewLstsqTest, MatchesNumpyForOverdeterminedVectorSystem) {
  const TDesc              a      = desc({3, 2});
  const TDesc              b      = desc({3});
  const auto               a_data = bytes(std::vector<float>{1.f, 0.f, 0.f, 1.f, 1.f, 1.f});
  const auto               b_data = bytes(std::vector<float>{1.f, 2.f, 4.f});
  holonp::LstsqFactory     factory;
  const std::vector<TDesc> descriptors{a, b};
  const std::vector<std::vector<std::byte>> inputs{a_data, b_data};
  const nlohmann::json                      settings{{"rcond", 1e-6f}};
  const auto                                inferred = factory.infer(descriptors, settings);
  ASSERT_EQ(inferred.output_descs.size(), 4u);
  EXPECT_EQ(inferred.output_descs[0].shape, (std::vector<size_t>{2}));
  EXPECT_EQ(inferred.output_descs[1].shape, (std::vector<size_t>{1}));
  EXPECT_TRUE(inferred.output_descs[2].shape.empty());
  EXPECT_EQ(inferred.output_descs[3].shape, (std::vector<size_t>{2}));

  const auto run = holonp_test::run_sync_factory(factory, descriptors, inputs, settings);
  holonp_test::OracleInput request{.op          = "lstsq",
                                   .n_outputs   = 4,
                                   .input_descs = descriptors,
                                   .input_bytes = inputs,
                                   .settings    = settings};
  const auto               expected = holonp_test::invoke_oracle(request, kOracle);
  expect_f32(run.output_bytes[0], expected.output_bytes[0]);
  expect_f32(run.output_bytes[1], expected.output_bytes[1]);
  ASSERT_EQ(run.output_bytes[2].size(), sizeof(std::uint16_t));
  EXPECT_EQ(*reinterpret_cast<const std::uint16_t *>(run.output_bytes[2].data()), 2);
  expect_f32(run.output_bytes[3], expected.output_bytes[3]);

  const TDesc              matrix_b      = desc({3, 2});
  const auto               matrix_b_data = bytes(std::vector<float>{1.f, 2.f, 2.f, 1.f, 4.f, 3.f});
  const std::vector<TDesc> matrix_descriptors{a, matrix_b};
  const std::vector<std::vector<std::byte>> matrix_inputs{a_data, matrix_b_data};
  const auto                                matrix_run =
      holonp_test::run_sync_factory(factory, matrix_descriptors, matrix_inputs, settings);
  holonp_test::OracleInput matrix_request{.op          = "lstsq",
                                          .n_outputs   = 4,
                                          .input_descs = matrix_descriptors,
                                          .input_bytes = matrix_inputs,
                                          .settings    = settings};
  const auto               matrix_expected = holonp_test::invoke_oracle(matrix_request, kOracle);
  expect_f32(matrix_run.output_bytes[0], matrix_expected.output_bytes[0]);
  expect_f32(matrix_run.output_bytes[1], matrix_expected.output_bytes[1]);
  ASSERT_EQ(matrix_run.output_bytes[2].size(), sizeof(std::uint16_t));
  EXPECT_EQ(*reinterpret_cast<const std::uint16_t *>(matrix_run.output_bytes[2].data()), 2);
  expect_f32(matrix_run.output_bytes[3], matrix_expected.output_bytes[3]);
}

TEST(NewUnaryTest, SqrtLogClipAndFinite) {
  const TDesc         input = desc({4});
  const auto          data  = bytes(std::vector<float>{1.f, 4.f, 9.f, 16.f});
  holonp::SqrtFactory sqrt;
  expect_oracle(sqrt, "sqrt", input, data, {});
  holonp::LogFactory log;
  expect_oracle(log, "log", input, data, {});
  holonp::ClipFactory clip;
  expect_oracle(clip, "clip", input, data, {{"min", 2.f}, {"max", 10.f}});
  holonp::IsfiniteFactory finite;
  expect_oracle(finite, "isfinite", input, data, {});
}

TEST(NewUnaryTest, ComplexSqrtPreservesComplexOutput) {
  const TDesc input = desc({2}, DType::CF32);
  const auto  data  = bytes(
      std::vector<cuFloatComplex>{make_cuFloatComplex(3.f, 4.f), make_cuFloatComplex(0.f, -4.f)});
  holonp::SqrtFactory                       sqrt;
  const std::vector<TDesc>                  descriptors{input};
  const std::vector<std::vector<std::byte>> inputs{data};
  const auto                                inferred = sqrt.infer(descriptors, {});
  ASSERT_EQ(inferred.output_descs[0].dtype, DType::CF32);
  const auto  run    = holonp_test::run_sync_factory(sqrt, descriptors, inputs, {});
  const auto *output = reinterpret_cast<const cuFloatComplex *>(run.output_bytes[0].data());
  EXPECT_NEAR(output[0].x, 2.f, 1e-5f);
  EXPECT_NEAR(output[0].y, 1.f, 1e-5f);
  EXPECT_NEAR(output[1].x, 1.4142135f, 1e-5f);
  EXPECT_NEAR(output[1].y, -1.4142135f, 1e-5f);
}

TEST(NewUnaryTest, ComplexLogIsRejected) {
  holonp::LogFactory       log;
  const std::vector<TDesc> descriptors{desc({2}, DType::CF32)};
  EXPECT_THROW(log.infer(descriptors, {}), std::invalid_argument);
}

TEST(NewBinaryTest, MaximumAndMinimum) {
  const TDesc                               input = desc({4});
  const auto                                a     = bytes(std::vector<float>{1.f, 8.f, 3.f, 7.f});
  const auto                                b     = bytes(std::vector<float>{4.f, 2.f, 6.f, 5.f});
  holonp::MaximumFactory                    maximum;
  const std::vector<TDesc>                  descriptors{input, input};
  const std::vector<std::vector<std::byte>> inputs{a, b};
  const auto max_run = holonp_test::run_sync_factory(maximum, descriptors, inputs, {});
  EXPECT_EQ(max_run.output_bytes[0], bytes(std::vector<float>{4.f, 8.f, 6.f, 7.f}));
  holonp::MinimumFactory minimum;
  const auto             min_run = holonp_test::run_sync_factory(minimum, descriptors, inputs, {});
  EXPECT_EQ(min_run.output_bytes[0], bytes(std::vector<float>{1.f, 2.f, 3.f, 5.f}));
}

TEST(NewSettingsTest, ClipRejectsInvertedBounds) {
  holonp::ClipFactory      factory;
  const std::vector<TDesc> descriptors{desc({2})};
  EXPECT_THROW(factory.infer(descriptors, {{"min", 2.f}, {"max", 1.f}}), std::invalid_argument);
}
