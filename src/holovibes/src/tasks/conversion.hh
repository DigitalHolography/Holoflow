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
#include "holoflow/core/tasks.hh"

template <typename T> using DevPtr  = curaii::unique_device_ptr<T>;
template <typename T> using HostPtr = curaii::unique_host_ptr<T>;

namespace holovibes::tasks {

/// @brief Settings for the data type conversion task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
///   "target": "U8|U16|F32|CF32",
///   "strategy": "Real|Scaled|Modulus|Argument"
/// }
/// @endcode
struct ConversionSettings {
  /// @brief Target data type.
  enum class Target {
    U8,   ///< 8-bit unsigned integer
    U16,  ///< 16-bit unsigned integer
    F32,  ///< 32-bit floating point
    CF32, ///< 32-bit complex floating point
  };

  /// @brief Conversion strategy.
  enum class Strategy {
    Real,    ///< Real part
    Scaled,  ///< Scaled to target range
    Modulus, ///< Modulus (magnitude)
    Argument ///< Argument (phase)
  };

  Target   target;   ///< Target data type.
  Strategy strategy; ///< Conversion strategy.
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref ConversionSettings and
/// @ref ConversionSettings::Target and @ref ConversionSettings::Strategy.
/// @{
void to_json(nlohmann::json &j, const ConversionSettings::Target &t);
void from_json(const nlohmann::json &j, ConversionSettings::Target &t);
void to_json(nlohmann::json &j, const ConversionSettings::Strategy &s);
void from_json(const nlohmann::json &j, ConversionSettings::Strategy &s);
void to_json(nlohmann::json &j, const ConversionSettings &cs);
void from_json(const nlohmann::json &j, ConversionSettings &cs);
/// @}

class Conversion : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  Conversion(const ConversionSettings &settings, size_t min_temp_storage_bytes,
             DevPtr<uint8_t> &&d_min_temp_storage, DevPtr<std::byte> &&d_min,
             size_t max_temp_storage_bytes, DevPtr<uint8_t> &&d_max_temp_storage,
             DevPtr<std::byte> &&d_max, cudaStream_t stream);

  void launch_u8_cf32_real(holoflow::core::CTView in, holoflow::core::TView out);
  void launch_u16_cf32_real(holoflow::core::CTView in, holoflow::core::TView out);
  void launch_f32_u8_scaled(holoflow::core::CTView in, holoflow::core::TView out);
  void launch_f32_u16_scaled(holoflow::core::CTView in, holoflow::core::TView out);
  void launch_cf32_f32_modulus(holoflow::core::CTView in, holoflow::core::TView out);
  void launch_cf32_f32_argument(holoflow::core::CTView in, holoflow::core::TView out);

  friend class ConversionFactory;

  ConversionSettings settings_;
  size_t             min_temp_storage_bytes_;
  DevPtr<uint8_t>    d_min_temp_storage_;
  DevPtr<std::byte>  d_min_;
  size_t             max_temp_storage_bytes_;
  DevPtr<uint8_t>    d_max_temp_storage_;
  DevPtr<std::byte>  d_max_;
  cudaStream_t       stream_;
};

class ConversionFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holovibes::tasks