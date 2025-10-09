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
#include <vector>
#include <memory>
#include <span>
#include <string>
#include <cufft.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

namespace holovibes::tasks::syncs {

struct FFTConvolutionData {
  cufftHandle fft_plan;
  DevPtr<cufftComplex> d_padded_input;
  DevPtr<cufftComplex> d_padded_kernel;
  DevPtr<cufftComplex> d_freq_input;
  DevPtr<cufftComplex> d_freq_kernel;
  unsigned int padded_width;
  unsigned int padded_height;
  
  ~FFTConvolutionData() {
    if (fft_plan) {
      cufftDestroy(fft_plan);
    }
  }
};


/// @brief Settings for convolution task
struct ConvolutionSettings {
    std::string kernel_file;
    bool divide;
};

/// @name JSON serialization
/// @{
void to_json(nlohmann::json& j, const ConvolutionSettings& s);
void from_json(const nlohmann::json& j, ConvolutionSettings& s);
/// @}

/// @brief 2D convolution task for image processing using FFT
class Convolution : public holoflow::core::ISyncTask {
public:
    ~Convolution() override = default;

    /// @brief Execute convolution operation
    holoflow::core::OpResult execute(holoflow::core::SyncCtx& ctx) override;

private:
    Convolution(ConvolutionSettings settings,
                const holoflow::core::TDesc& input_desc,
                const holoflow::core::TDesc& output_desc,
                cudaStream_t stream,
                DevPtr<float>&& d_kernel,
                const int kernel_width,
                const int kernel_height);

    friend class ConvolutionFactory;

    ConvolutionSettings settings_;
    holoflow::core::TDesc input_desc_;
    holoflow::core::TDesc output_desc_;
    cudaStream_t stream_;
    DevPtr<float> d_kernel_;
    const int kernel_width_;
    const int kernel_height_;
    
    mutable std::unique_ptr<FFTConvolutionData> fft_data_;

    void create_fft_data();
};

/// @brief Factory for creating Convolution tasks
class ConvolutionFactory : public holoflow::core::ISyncTaskFactory {
public:
    ~ConvolutionFactory() override = default;

    /// @brief Infer task metadata and requirements
    holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                      const nlohmann::json& jsettings) const override;

    /// @brief Create a new Convolution task instance
    std::unique_ptr<holoflow::core::ISyncTask> create(
        std::span<const holoflow::core::TDesc> input_descs,
        const nlohmann::json& jsettings,
        const holoflow::core::SyncCreateCtx& ctx) const override;
};

} // namespace holovibes::tasks::syncs