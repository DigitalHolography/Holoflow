#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/curaii.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class PercentileClipTask : public Task {
public:
  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class PercentileClipTaskFactory;

private:
  PercentileClipTask(const TaskMeta &meta, CudaStreamRef stream,
                     unique_device_ptr<float> d_lower_thresh,
                     unique_device_ptr<float> d_upper_thresh,
                     unique_device_ptr<uint8_t> d_sort_tmp,
                     unique_device_ptr<float> d_roi_values,
                     unique_device_ptr<uint8_t> d_select_tmp,
                     unique_device_ptr<uint8_t> d_roi_mask,
                     unique_device_ptr<int> d_roi_count, size_t sort_tmp_bytes,
                     size_t select_tmp_bytes, float pct_low, float pct_high,
                     float roi_radius);

  // Device buffers
  unique_device_ptr<float> d_lower_thresh_; // 1 float
  unique_device_ptr<float> d_upper_thresh_; // 1 float

  unique_device_ptr<uint8_t> d_sort_tmp_;   // radix‑sort workspace
  unique_device_ptr<uint8_t> d_select_tmp_; // flag‑select workspace

  unique_device_ptr<float> d_roi_values_; // compacted ROI pixels
  unique_device_ptr<uint8_t> d_roi_mask_; // 0/1 ellipse mask
  unique_device_ptr<int> d_roi_count_;    // number of pixels kept

  // Scalar parameters
  size_t sort_tmp_bytes_;
  size_t select_tmp_bytes_;

  float pct_low_;
  float pct_high_;
  float roi_radius_;
};

class PercentileClipTaskFactory : public TaskFactory {
public:
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params) override;

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params,
         CudaStreamRef stream) override;

private:
  struct Params {
    float lower_percentile;
    float upper_percentile;
    float radius;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, lower_percentile, upper_percentile,
                                   radius);
  };
};

} // namespace dh