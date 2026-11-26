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

#include "holonp/matmul.hh"
#include "utils/tensor_common.hh"

#include <cuComplex.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const MatmulSettings &s) { j = {{"math_mode", s.math_mode}}; }

void from_json(const nlohmann::json &j, MatmulSettings &s) {
  s.math_mode = j.is_null() ? "default" : j.value("math_mode", std::string("default"));
}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Matmul: " + msg);
}

struct Layout {
  holoflow::core::TDesc output_desc;
  int                   lda, ldb;
  cublasOperation_t     opa, opb;
  int                   m;
  int                   k;
  int                   n;
  size_t                batch_count;
  size_t                a_batch_stride;
  size_t                b_batch_stride;
  size_t                c_batch_stride;
};

Layout infer_layout(std::span<const holoflow::core::TDesc> input_descs) {
  check(input_descs.size() == 2, "expected exactly 2 inputs");
  const auto &a = input_descs[0];
  const auto &b = input_descs[1];

  check(a.mem_loc == holoflow::core::MemLoc::Device && b.mem_loc == holoflow::core::MemLoc::Device,
        "only Device tensors are supported");

  check(a.dtype == b.dtype, "inputs must have the same dtype");
  check(a.dtype == holoflow::core::DType::F32 || a.dtype == holoflow::core::DType::CF32 ||
            a.dtype == holoflow::core::DType::F16,
        "supported dtypes are F16, F32 and CF32");
  check(a.shape.size() >= 2 && b.shape.size() >= 2, "inputs must have rank >= 2");
  check(a.shape.size() == b.shape.size(), "batched inputs must have the same rank");

  const auto rank = static_cast<int>(a.shape.size());
  const auto m    = a.shape[static_cast<size_t>(rank - 2)];
  const auto k    = a.shape[static_cast<size_t>(rank - 1)];
  const auto kb   = b.shape[static_cast<size_t>(rank - 2)];
  const auto n    = b.shape[static_cast<size_t>(rank - 1)];
  check(k == kb, "inner matrix dimensions must match");
  check(m > 0 && k > 0 && n > 0, "matrix dimensions must be nonzero");
  check(m <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            k <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            n <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "matrix dimensions exceed cuBLAS limits");

  std::vector<size_t> output_shape(a.shape.begin(), a.shape.end() - 2);
  for (int i = 0; i < rank - 2; ++i) {
    check(a.shape[static_cast<size_t>(i)] == b.shape[static_cast<size_t>(i)],
          "batch dimensions must match");
  }
  output_shape.push_back(m);
  output_shape.push_back(n);

  size_t batch_count = 1;
  for (int i = 0; i < rank - 2; ++i)
    batch_count *= a.shape[static_cast<size_t>(i)];

  const auto matrix_layout = [&](const auto &d) {
    const auto element = holoflow::core::size_of(d.dtype);
    const auto rows = d.shape[rank - 2], cols = d.shape[rank - 1];
    const auto row = d.strides[rank - 2], col = d.strides[rank - 1];
    const bool row_major = col == element && row / element >= cols && row % element == 0;
    const auto leading   = row_major ? row : col;
    check(row_major || (row == element && col / element >= rows && col % element == 0),
          "unsupported matrix strides");
    check(leading / element <= static_cast<size_t>(std::numeric_limits<int>::max()),
          "leading dimension exceeds cuBLAS limits");
    const auto span =
        row_major ? (rows - 1) * row + cols * element : (cols - 1) * col + rows * element;
    const auto stride = rank == 2 ? span : d.strides[rank - 3];
    check(rank == 2 || d.shape[rank - 3] == 1 || stride >= span, "overlapping batch matrices");
    size_t expected = stride;
    for (int axis = rank - 3; axis >= 0; --axis) {
      check(d.shape[axis] <= 1 || d.strides[axis] == expected, "batch strides must be uniform");
      expected *= d.shape[axis];
    }
    return std::tuple{static_cast<int>(leading / element), row_major ? CUBLAS_OP_N : CUBLAS_OP_T,
                      stride};
  };
  const auto [lda, opa, stride_a] = matrix_layout(a);
  const auto [ldb, opb, stride_b] = matrix_layout(b);
  return Layout{
      .output_desc = holoflow::core::TDesc(output_shape, a.dtype, holoflow::core::MemLoc::Device),
      .lda         = lda,
      .ldb         = ldb,
      .opa         = opa,
      .opb         = opb,
      .m           = static_cast<int>(m),
      .k           = static_cast<int>(k),
      .n           = static_cast<int>(n),
      .batch_count = batch_count,
      .a_batch_stride = stride_a,
      .b_batch_stride = stride_b,
      .c_batch_stride = m * n,
  };
}

class Matmul : public holoflow::core::ISyncTask {
public:
  Matmul(MatmulSettings settings, holoflow::core::TDesc a_desc, holoflow::core::TDesc b_desc,
         Layout layout, cudaStream_t stream)
      : settings_(std::move(settings)), a_desc_(std::move(a_desc)), b_desc_(std::move(b_desc)),
        layout_(std::move(layout)), stream_(stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_.get(), stream_));
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;
  bool                     supports_cuda_graph() const noexcept override { return true; }
  void                     record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    holoflow::core::SyncCtx execution{ctx.inputs, ctx.outputs};
    (void)execute(execution);
  }

  const MatmulSettings        &settings() const { return settings_; }
  const holoflow::core::TDesc &a_desc() const { return a_desc_; }
  const holoflow::core::TDesc &b_desc() const { return b_desc_; }
  void                         update_stream(cudaStream_t stream) {
    if (stream_ != stream) {
      stream_ = stream;
      CUBLAS_CHECK(cublasSetStream(cublas_.get(), stream_));
    }
  }

private:
  MatmulSettings        settings_;
  holoflow::core::TDesc a_desc_;
  holoflow::core::TDesc b_desc_;
  Layout                layout_;
  cudaStream_t          stream_;
  curaii::CublasHandle  cublas_;
};

} // namespace

holoflow::core::OpResult Matmul::execute(holoflow::core::SyncCtx &ctx) {
  constexpr float alpha_f = 1.0f;
  constexpr float beta_f  = 0.0f;
  using holoflow::core::DType;
  for (size_t batch = 0; batch < layout_.batch_count; ++batch) {
    const auto *a        = ctx.inputs[0].data() + batch * layout_.a_batch_stride;
    const auto *b        = ctx.inputs[1].data() + batch * layout_.b_batch_stride;
    auto       *c        = ctx.outputs[0].data() +
                           batch * layout_.c_batch_stride * holoflow::core::size_of(a_desc_.dtype);
    const auto [m, n, k] = std::tuple{layout_.m, layout_.n, layout_.k};
    if (a_desc_.dtype == DType::CF32) {
      const auto alpha = make_cuComplex(1.0f, 0.0f);
      const auto beta  = make_cuComplex(0.0f, 0.0f);
      CUBLAS_CHECK(cublasCgemm(cublas_.get(), layout_.opb, layout_.opa, n, m, k, &alpha,
                               reinterpret_cast<const cuFloatComplex *>(b), layout_.ldb,
                               reinterpret_cast<const cuFloatComplex *>(a), layout_.lda, &beta,
                               reinterpret_cast<cuFloatComplex *>(c), n));
    } else if (a_desc_.dtype == DType::F32 && settings_.math_mode == "default") {
      CUBLAS_CHECK(cublasSgemm(cublas_.get(), layout_.opb, layout_.opa, n, m, k, &alpha_f,
                               reinterpret_cast<const float *>(b), layout_.ldb,
                               reinterpret_cast<const float *>(a), layout_.lda, &beta_f,
                               reinterpret_cast<float *>(c), n));
    } else {
      const auto type = a_desc_.dtype == DType::F16 ? CUDA_R_16F : CUDA_R_32F;
      const auto compute =
          settings_.math_mode == "fast_f16" ? CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_32F;
      CUBLAS_CHECK(cublasGemmEx(cublas_.get(), layout_.opb, layout_.opa, n, m, k, &alpha_f, b, type,
                                layout_.ldb, a, type, layout_.lda, &beta_f, c, type, n, compute,
                                CUBLAS_GEMM_DEFAULT));
    }
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult MatmulFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<MatmulSettings>();
  check(settings.math_mode == "default" ||
            (settings.math_mode == "fast_f16" && !input_descs.empty() &&
             input_descs[0].dtype == holoflow::core::DType::F32),
        "unsupported math mode");
  const auto layout = infer_layout(input_descs);
  return holoflow::core::InferResult{.input_descs   = {input_descs[0], input_descs[1]},
                                     .output_descs  = {layout.output_desc},
                                     .in_place      = {},
                                     .owned_inputs  = {false, false},
                                     .owned_outputs = {false},
                                     .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
MatmulFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                      const nlohmann::json                  &jsettings,
                      const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  return std::make_unique<Matmul>(jsettings.get<MatmulSettings>(), input_descs[0], input_descs[1],
                                  infer_layout(input_descs), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
MatmulFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                      std::span<const holoflow::core::TDesc>     input_descs,
                      const nlohmann::json                      &jsettings,
                      const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);
  auto *old_matmul = dynamic_cast<Matmul *>(old_task.get());
  if (old_matmul != nullptr && input_descs.size() == 2 &&
      old_matmul->settings() == jsettings.get<MatmulSettings>() &&
      utils::same_desc(input_descs[0], old_matmul->a_desc()) &&
      utils::same_desc(input_descs[1], old_matmul->b_desc())) {
    old_matmul->update_stream(ctx.stream);
    return old_task;
  }
  return create(input_descs, jsettings, ctx);
}

} // namespace holonp
