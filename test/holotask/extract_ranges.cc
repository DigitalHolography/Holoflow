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

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <vector>

#include "holoflow/core/tasks.hh"
#include "holotask/syncs/extract_ranges.hh"

using namespace holotask::syncs;
using namespace holoflow::core;

static void cuda_check(cudaError_t e, const char *msg = nullptr) {
  if (e != cudaSuccess) {
    if (msg)
      fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
    else
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
    std::abort();
  }
}

static float *gpu_alloc_and_upload(const std::vector<float> &host) {
  float *ptr = nullptr;
  cuda_check(cudaMalloc(&ptr, host.size() * sizeof(float)), "cudaMalloc input");
  cuda_check(cudaMemcpy(ptr, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
             "cudaMemcpy H->D input");
  return ptr;
}

static std::vector<float> gpu_download(float *ptr, size_t count) {
  std::vector<float> out(count);
  cuda_check(cudaMemcpy(out.data(), ptr, count * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy D->H output");
  return out;
}

// TEST(ExtractRanges, CorrectCopy3D) {
//   cudaStream_t stream = nullptr;
//   cuda_check(cudaStreamCreate(&stream), "cudaStreamCreate");

//   const size_t D = 4, H = 4, W = 6;
//   const size_t total = D * H * W;

//   std::vector<float> host_in(total);
//   for (size_t i = 0; i < total; ++i)
//     host_in[i] = static_cast<float>(i);

//   float *d_input = gpu_alloc_and_upload(host_in);

//   ExtractRangesSettings settings;
//   settings.x_ranges = {{1, 3}, {4, 6}}; // widths: 2 + 2 = 4
//   settings.y_ranges = {{0, 1}, {2, 4}}; // heights: 1 + 2 = 3
//   settings.z_ranges = {{1, 3}};         // depth: 2

//   nlohmann::json jsettings = settings;

//   TDesc in_desc;
//   in_desc.dtype = DType::F32;
//   in_desc.shape = {D, H, W};

//   std::array<TDesc, 1> in_descs = {in_desc};

//   ExtractRangesFactory factory;

//   InferResult infer_res = factory.infer(in_descs, jsettings);
//   ASSERT_EQ(infer_res.output_descs.size(), 1u);

//   const TDesc &out_desc  = infer_res.output_descs[0];
//   const size_t outD      = out_desc.shape[0];
//   const size_t outH      = out_desc.shape[1];
//   const size_t outW      = out_desc.shape[2];
//   const size_t out_total = outD * outH * outW;

//   float *d_output = nullptr;
//   cuda_check(cudaMalloc(&d_output, out_total * sizeof(float)), "cudaMalloc output");

//   SyncCreateCtx create_ctx;
//   create_ctx.stream = stream;

//   std::unique_ptr<ISyncTask> task = factory.create(in_descs, jsettings, create_ctx);
//   ASSERT_NE(task, nullptr);

//   TView input_view;
//   input_view.data = reinterpret_cast<std::byte *>(d_input);
//   input_view.desc = in_desc;

//   TView output_view;
//   output_view.data = reinterpret_cast<std::byte *>(d_output);
//   output_view.desc = out_desc;

//   std::array<TView, 1> ins  = {input_view};
//   std::array<TView, 1> outs = {output_view};

//   std::atomic<bool> cancel_flag(false);

//   SyncCtx ctx{.inputs       = std::span<TView>(ins),
//               .outputs      = std::span<TView>(outs),
//               .cancelled    = &cancel_flag,
//               .event_writer = nullptr,
//               .event_reader = nullptr};

//   OpResult res = task->execute(ctx);
//   ASSERT_EQ(res, OpResult::Ok);

//   cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

//   auto host_out = gpu_download(d_output, out_total);

//   std::vector<float> expected;
//   expected.reserve(out_total);

//   for (const auto &zr : settings.z_ranges) {
//     for (size_t z = zr.start; z < zr.end; ++z) {
//       for (const auto &yr : settings.y_ranges) {
//         for (size_t y = yr.start; y < yr.end; ++y) {
//           for (const auto &xr : settings.x_ranges) {
//             for (size_t x = xr.start; x < xr.end; ++x) {
//               expected.push_back(host_in[z * (H * W) + y * W + x]);
//             }
//           }
//         }
//       }
//     }
//   }

//   ASSERT_EQ(expected.size(), host_out.size());
//   for (size_t i = 0; i < expected.size(); ++i) {
//     EXPECT_FLOAT_EQ(expected[i], host_out[i]) << "Mismatch at index " << i;
//   }

//   cuda_check(cudaFree(d_input), "cudaFree input");
//   cuda_check(cudaFree(d_output), "cudaFree output");
//   cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
// }