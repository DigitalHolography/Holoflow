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

#pragma once

#include <nlohmann/json.hpp>

#include "curaii/cuda.hh"
#include "curaii/cufft.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holovibes::tasks::syncs {

struct RegistrationSettings {
    float radius = 0.9f;
};

void to_json(nlohmann::json& j, const RegistrationSettings& s);
void from_json(const nlohmann::json& j, RegistrationSettings& s);

class Registration : public holoflow::core::ISyncTask {
public:
    Registration(
        RegistrationSettings settings,
        const holoflow::core::TDesc& input_desc,
        const holoflow::core::TDesc& output_desc,
        cudaStream_t stream,
        // Center mean
        DevPtr<float> d_mean_centered,
        // Cross correlation
        bool ref_initialized,
        size_t freq_size,
        curaii::CufftHandle r2c_handle,
        curaii::CufftHandle c2r_handle,
        DevPtr<float> d_ref,
        DevPtr<float> d_xcorr,
        DevPtr<cuFloatComplex> d_freq1,
        DevPtr<cuFloatComplex> d_freq2,
        // CUB sum
        size_t sum_tmp_bytes,
        DevPtr<uint8_t> d_sum_tmp,
        DevPtr<float> d_sum,
        // CUB argmax
        size_t amax_tmp_bytes,
        DevPtr<uint8_t> d_amax_tmp,
        DevPtr<float> d_max,
        DevPtr<int64_t> d_max_idx,
        // CUB select
        size_t select_tmp_bytes,
        DevPtr<uint8_t> d_select_tmp,
        DevPtr<int> d_select_count,
        DevPtr<uint8_t> d_select_roi,
        DevPtr<float> d_selected);

    holoflow::core::OpResult execute(holoflow::core::SyncCtx& ctx) override;

private:
    void xcorr(float* odata, const float* idata);
    void mask_circle(float* odata, const float* idata, std::size_t b, std::size_t h, std::size_t w);
    void center_mean(float* odata, const float* idata, std::size_t b, std::size_t h, std::size_t w);
    std::pair<int64_t, int64_t> get_shifts(float* xcorr, std::size_t b, std::size_t h, std::size_t w);

    std::pair<float, float> get_shifts_subpixel(float* xcorr, std::size_t h, std::size_t w);
    void apply_shifts(float* odata, const float* idata, float shift_x, float shift_y, std::size_t b, std::size_t h, std::size_t w);

    RegistrationSettings settings_;
    holoflow::core::TDesc input_desc_;
    holoflow::core::TDesc output_desc_;
    cudaStream_t stream_;

    // Center mean
    DevPtr<float> d_mean_centered_;

    // Cross correlation
    bool ref_initialized_;
    size_t freq_size_;
    curaii::CufftHandle r2c_handle_;
    curaii::CufftHandle c2r_handle_;
    DevPtr<float> d_ref_;
    DevPtr<float> d_xcorr_;
    DevPtr<cuFloatComplex> d_freq1_;
    DevPtr<cuFloatComplex> d_freq2_;

    // CUB sum
    size_t sum_tmp_bytes_;
    DevPtr<uint8_t> d_sum_tmp_;
    DevPtr<float> d_sum_;

    // CUB argmax
    size_t amax_tmp_bytes_;
    DevPtr<uint8_t> d_amax_tmp_;
    DevPtr<float> d_max_;
    DevPtr<int64_t> d_max_idx_;

    // CUB select
    size_t select_tmp_bytes_;
    DevPtr<uint8_t> d_select_tmp_;
    DevPtr<int> d_select_count_;
    DevPtr<uint8_t> d_select_roi_;
    DevPtr<float> d_selected_;
};

class RegistrationFactory : public holoflow::core::ISyncTaskFactory {
public:
    holoflow::core::InferResult infer(
        std::span<const holoflow::core::TDesc> input_descs,
        const nlohmann::json& jsettings) const override;

    std::unique_ptr<holoflow::core::ISyncTask> create(
        std::span<const holoflow::core::TDesc> input_descs,
        const nlohmann::json& jsettings,
        const holoflow::core::SyncCreateCtx& ctx) const override;
};

} // namespace holovibes::tasks::syncs