// Copyright 2025 Digital Holography Foundation
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

#include "holotask/syncs/extract_ranges.hh"

#include "logger.hh"
#include <cuda_runtime.h>

namespace holotask::syncs {

void to_json(nlohmann::json &j, const Range &range) {
  j = nlohmann::json{{"start", range.start}, {"end", range.end}};
}

void from_json(const nlohmann::json &j, Range &range) {
  j.at("start").get_to(range.start);
  j.at("end").get_to(range.end);
}

void to_json(nlohmann::json &j, const ExtractRangesSettings &settings) {
  j = nlohmann::json{{"x_ranges", settings.x_ranges},
                     {"y_ranges", settings.y_ranges},
                     {"z_ranges", settings.z_ranges}};
}

void from_json(const nlohmann::json &j, ExtractRangesSettings &settings) {
  j.at("x_ranges").get_to(settings.x_ranges);
  j.at("y_ranges").get_to(settings.y_ranges);
  j.at("z_ranges").get_to(settings.z_ranges);
}

std::size_t compute_total_length(const std::vector<Range> &ranges) {
  std::size_t total = 0;
  for (const auto &range : ranges) {
    total += (range.end - range.start);
  }
  return total;
}

std::vector<CopyOp> precompute_copy_ops(const ExtractRangesSettings &settings) {
  std::vector<CopyOp> copy_ops;

  std::size_t dst_z_offset = 0;
  for (const auto &z_range : settings.z_ranges) {
    const std::size_t z_length = z_range.end - z_range.start;

    std::size_t dst_y_offset = 0;
    for (const auto &y_range : settings.y_ranges) {
      const std::size_t y_length = y_range.end - y_range.start;

      std::size_t dst_x_offset = 0;
      for (const auto &x_range : settings.x_ranges) {
        const std::size_t x_length = x_range.end - x_range.start;

        CopyOp op;
        op.src_pos =
            make_cudaPos(static_cast<size_t>(x_range.start) * sizeof(float),
                         static_cast<size_t>(y_range.start), static_cast<size_t>(z_range.start));

        op.dst_pos =
            make_cudaPos(static_cast<size_t>(dst_x_offset) * sizeof(float),
                         static_cast<size_t>(dst_y_offset), static_cast<size_t>(dst_z_offset));

        op.extent = make_cudaExtent(static_cast<size_t>(x_length) * sizeof(float),
                                    static_cast<size_t>(y_length), static_cast<size_t>(z_length));

        copy_ops.push_back(op);

        dst_x_offset += x_length;
      }

      dst_y_offset += y_length;
    }

    dst_z_offset += z_length;
  }

  logger()->info("[ExtractRanges] Pre-computed {} copy operations", copy_ops.size());
  return copy_ops;
}

ExtractRanges::ExtractRanges(cudaStream_t stream, std::size_t input_width, std::size_t input_height,
                             std::size_t output_width, std::size_t output_height,
                             std::vector<CopyOp> &&copy_ops)
    : stream_(stream), input_width_(input_width), input_height_(input_height),
      output_width_(output_width), output_height_(output_height), copy_ops_(std::move(copy_ops)) {}

holoflow::core::OpResult ExtractRanges::execute(holoflow::core::SyncCtx &ctx) {

  holoflow::core::TView &input  = ctx.inputs[0];
  holoflow::core::TView &output = ctx.outputs[0];

  cudaPitchedPtr src_ptr = make_cudaPitchedPtr((void *)input.data, input_width_ * sizeof(float),
                                               input_width_, input_height_);

  cudaPitchedPtr dst_ptr = make_cudaPitchedPtr((void *)output.data, output_width_ * sizeof(float),
                                               output_width_, output_height_);

  for (const auto &op : copy_ops_) {
    cudaMemcpy3DParms params = {0};
    params.srcPtr            = src_ptr;
    params.dstPtr            = dst_ptr;
    params.srcPos            = op.src_pos;
    params.dstPos            = op.dst_pos;
    params.extent            = op.extent;
    params.kind              = cudaMemcpyDeviceToDevice;

    CUDA_CHECK(cudaMemcpy3DAsync(&params, stream_));
  }

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
ExtractRangesFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                            const nlohmann::json                  &jsettings) const {

  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[ExtractRangesFactory::infer] error: {}", msg);
      throw std::invalid_argument("ExtractRangesFactory inference error: " + msg);
    }
  };

  auto settings = jsettings.get<ExtractRangesSettings>();

  check(input_descs.size() == 1, "expected exactly one input");
  check(input_descs[0].dtype == holoflow::core::DType::F32, "only Float32 data type is supported");
  const auto &idesc = input_descs[0];

  check(idesc.rank() == 3, "expected 3D tensor");
  check(!settings.z_ranges.empty(), "z_ranges cannot be empty");
  check(!settings.y_ranges.empty(), "y_ranges cannot be empty");
  check(!settings.x_ranges.empty(), "x_ranges cannot be empty");

  const std::size_t input_depth  = idesc.shape[0];
  const std::size_t input_height = idesc.shape[1];
  const std::size_t input_width  = idesc.shape[2];

  // Validate all ranges
  for (const auto &range : settings.z_ranges) {
    check(range.start < range.end, "Invalid Z range: start >= end");
    check(range.end <= input_depth, "Z range exceeds input depth");
  }

  for (const auto &range : settings.y_ranges) {
    check(range.start < range.end, "Invalid Y range: start >= end");
    check(range.end <= input_height, "Y range exceeds input height");
  }

  for (const auto &range : settings.x_ranges) {
    check(range.start < range.end, "Invalid X range: start >= end");
    check(range.end <= input_width, "X range exceeds input width");
  }

  const std::size_t output_depth  = compute_total_length(settings.z_ranges);
  const std::size_t output_height = compute_total_length(settings.y_ranges);
  const std::size_t output_width  = compute_total_length(settings.x_ranges);

  logger()->info("[ExtractRangesFactory] Input shape: [{}, {}, {}], Output shape: [{}, {}, {}]",
                 input_depth, input_height, input_width, output_depth, output_height, output_width);

  auto odesc  = idesc;
  odesc.shape = {output_depth, output_height, output_width};

  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
ExtractRangesFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                             const nlohmann::json                  &jsettings,
                             const holoflow::core::SyncCreateCtx   &ctx) const {

  auto infer_result = this->infer(input_descs, jsettings);
  auto settings     = jsettings.get<ExtractRangesSettings>();

  std::vector<CopyOp> copy_ops = precompute_copy_ops(settings);

  const auto &input_desc  = infer_result.input_descs[0];
  const auto &output_desc = infer_result.output_descs[0];

  auto *task = new ExtractRanges(ctx.stream, input_desc.shape[2], input_desc.shape[1],
                                 output_desc.shape[2], output_desc.shape[1], std::move(copy_ops));
  return std::unique_ptr<holoflow::core::ISyncTask>(task);
}

} // namespace holotask::syncs