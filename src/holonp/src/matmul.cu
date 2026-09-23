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
#include <vector>

#include "curaii/cublas.hh"
#include "curaii/cuda.hh"

namespace holonp {

void to_json(nlohmann::json &j, const MatmulSettings &) { j = nlohmann::json::object(); }

void from_json(const nlohmann::json &, MatmulSettings &) {}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Matmul: " + msg);
}

struct Layout {
  holoflow::core::TDesc output_desc;
  int                   rank;
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
  check(utils::is_c_contiguous(a) && utils::is_c_contiguous(b), "inputs must be C-contiguous");
  check(a.dtype == b.dtype, "inputs must have the same dtype");
  check(a.dtype == holoflow::core::DType::F32 || a.dtype == holoflow::core::DType::CF32,
        "supported dtypes are F32 and CF32");
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

  return Layout{
      .output_desc = holoflow::core::TDesc(output_shape, a.dtype, holoflow::core::MemLoc::Device),
      .rank        = rank,
      .m           = static_cast<int>(m),
      .k           = static_cast<int>(k),
      .n           = static_cast<int>(n),
      .batch_count = batch_count,
      .a_batch_stride = m * k,
      .b_batch_stride = k * n,
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
  const auto     *a       = reinterpret_cast<const float *>(ctx.inputs[0].data());
  const auto     *b       = reinterpret_cast<const float *>(ctx.inputs[1].data());
  auto           *c       = reinterpret_cast<float *>(ctx.outputs[0].data());

  for (size_t batch = 0; batch < layout_.batch_count; ++batch) {
    const auto a_offset = static_cast<long long>(batch * layout_.a_batch_stride);
    const auto b_offset = static_cast<long long>(batch * layout_.b_batch_stride);
    const auto c_offset = static_cast<long long>(batch * layout_.c_batch_stride);

    if (a_desc_.dtype == holoflow::core::DType::F32) {
      CUBLAS_CHECK(cublasSgemm(cublas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, layout_.n, layout_.m,
                               layout_.k, &alpha_f, b + b_offset, layout_.n, a + a_offset,
                               layout_.k, &beta_f, c + c_offset, layout_.n));
    } else {
      const auto *a_complex      = reinterpret_cast<const cuFloatComplex *>(ctx.inputs[0].data());
      const auto *b_complex      = reinterpret_cast<const cuFloatComplex *>(ctx.inputs[1].data());
      auto       *c_complex      = reinterpret_cast<cuFloatComplex *>(ctx.outputs[0].data());
      const cuFloatComplex alpha = make_cuComplex(1.0f, 0.0f);
      const cuFloatComplex beta  = make_cuComplex(0.0f, 0.0f);
      CUBLAS_CHECK(cublasCgemm(cublas_.get(), CUBLAS_OP_N, CUBLAS_OP_N, layout_.n, layout_.m,
                               layout_.k, &alpha, b_complex + b_offset, layout_.n,
                               a_complex + a_offset, layout_.k, &beta, c_complex + c_offset,
                               layout_.n));
    }
  }

  CUDA_CHECK(cudaGetLastError());
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult MatmulFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                 const nlohmann::json &jsettings) const {
  (void)jsettings;
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
