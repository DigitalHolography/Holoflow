#pragma once

#include <nlohmann/json.hpp>

#include "holoflow/core/tasks.hh"

namespace holovibes::tasks {

/// @brief Settings for the FFT shift task.
/// @details
/// JSON schema (informal):
/// @code{.json}
/// {
/// }
/// @endcode
struct FFTShiftSettings {
  // Empty for now
};

/// @name JSON serialization
/// @brief nlohmann::json adapters for @ref FFTShiftSettings.
/// @{
void to_json(nlohmann::json &j, const FFTShiftSettings &s);
void from_json(const nlohmann::json &j, FFTShiftSettings &s);
/// @}

class FFTShift : public holoflow::core::ISyncTask {
public:
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override;

private:
  FFTShift(const FFTShiftSettings &settings, cudaStream_t stream);

  friend class FFTShiftFactory;

  FFTShiftSettings settings_;
  cudaStream_t     stream_;
};

class FFTShiftFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;

  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holovibes::tasks