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

#include "holonp/transpose.hh"

#include <algorithm>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <vector>

namespace holonp {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const TransposeSettings &s) {
  j = nlohmann::json{{"axes", s.axes}};
}

void from_json(const nlohmann::json &j, TransposeSettings &s) {
  if (j.contains("axes"))
    j.at("axes").get_to(s.axes);
}

namespace {

// -------------------------------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------------------------------

constexpr int kKernelMaxNDim = 16;

inline void check(bool cond, const std::string &msg) {
  if (!cond)
    throw std::invalid_argument("Transpose: " + msg);
}

std::vector<int> normalize_axes(const std::vector<int> &axes, int ndim) {
  check(static_cast<int>(axes.size()) == ndim, "axes length must match ndim");

  std::vector<int> norm;
  norm.reserve(ndim);
  std::vector<bool> seen(ndim, false);

  for (int a : axes) {
    if (a < 0)
      a += ndim;
    check(a >= 0 && a < ndim, "axis out of bounds");
    check(!seen[a], "duplicate axis in permutation");
    seen[a] = true;
    norm.push_back(a);
  }
  return norm;
}

// Ensure strides exist or create contiguous default
std::vector<size_t> get_strides_bytes(const holoflow::core::TDesc &desc) {
  std::vector<size_t> strides(desc.shape.size());
  if (!desc.strides.empty()) {
    for (size_t i = 0; i < desc.strides.size(); ++i)
      strides.at(i) = desc.strides.at(i);
  } else {
    size_t acc = holoflow::core::size_of(desc.dtype);
    for (size_t i = desc.shape.size(); i-- > 0;) {
      strides.at(i) = acc;
      acc *= desc.shape.at(i);
    }
  }
  return strides;
}

// -------------------------------------------------------------------------------------------------
// Transpose task implementation
// -------------------------------------------------------------------------------------------------

class Transpose : public holoflow::core::ISyncTask {
public:
  explicit Transpose(cudaStream_t stream) : stream_(stream) {}

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;
  void                     update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  cudaStream_t stream_;
};

} // namespace

holoflow::core::OpResult Transpose::execute(holoflow::core::SyncCtx &ctx) {
  (void)ctx;
  // auto *idata = reinterpret_cast<const std::uint8_t *>(ctx.inputs[0].data());
  // auto *odata = reinterpret_cast<std::uint8_t *>(ctx.outputs[0].data());

  // 1. Short-circuit: The framework respected our in-place request.
  // This is a true zero-cost view transpose.
  return holoflow::core::OpResult::Ok;
}

// -------------------------------------------------------------------------------------------------
// TransposeFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
TransposeFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings) const {
  check(input_descs.size() == 1, "expected 1 input");
  const auto &idesc = input_descs[0];

  const int ndim = static_cast<int>(idesc.shape.size());
  check(ndim > 0 && ndim <= kKernelMaxNDim,
        "invalid ndim (max " + std::to_string(kKernelMaxNDim) + ")");

  auto settings = jsettings.get<TransposeSettings>();
  auto axes     = normalize_axes(settings.axes, ndim);

  // 1. Get input strides (calculate dense ones if empty)
  auto in_strides = get_strides_bytes(idesc);

  // 2. Permute shape AND strides
  std::vector<size_t> oshape(ndim);
  std::vector<size_t> ostrides(ndim);
  for (int i = 0; i < ndim; ++i) {
    oshape[i]   = idesc.shape[axes[i]];
    ostrides[i] = in_strides[axes[i]];
  }

  // 3. Create output descriptor using the permuted strides and identical offset
  holoflow::core::TDesc odesc(oshape, idesc.dtype, idesc.mem_loc, ostrides, idesc.offset);

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {{0, 0}}, // Crucial: Request aliasing from graph engine
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
TransposeFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                         const nlohmann::json                  &jsettings,
                         const holoflow::core::SyncCreateCtx   &ctx) const {
  (void)infer(input_descs, jsettings);
  return std::make_unique<Transpose>(ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
TransposeFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                         std::span<const holoflow::core::TDesc>     input_descs,
                         const nlohmann::json                      &jsettings,
                         const holoflow::core::SyncCreateCtx       &ctx) const {
  (void)infer(input_descs, jsettings);

  auto *old_transpose = dynamic_cast<Transpose *>(old_task.get());
  if (old_transpose == nullptr || input_descs.size() != 1) {
    return create(input_descs, jsettings, ctx);
  }

  old_transpose->update_stream(ctx.stream);
  return old_task;
}

} // namespace holonp
