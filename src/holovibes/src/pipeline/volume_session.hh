// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "curaii/cuda.hh"
#include "holoflow/core/tensor.hh"
#include <QByteArray>
#include <string>

namespace holovibes::pipeline {
/// Worker-thread-owned GPU cache shared by quantitative tomography and focus stacks.
class VolumeSession {
public:
  QByteArray slice(int axis, int index, const std::string &component, int count = 1,
                   bool transpose = false);
  int        width() const { return desc_.shape.empty() ? 0 : static_cast<int>(desc_.shape[2]); }
  int        height() const { return desc_.shape.empty() ? 0 : static_cast<int>(desc_.shape[1]); }
  int        depth() const { return desc_.shape.empty() ? 0 : static_cast<int>(desc_.shape[0]); }

protected:
  void retain(const holoflow::core::TDesc &desc, const std::byte *data);

private:
  curaii::CudaStream                   stream_;
  curaii::unique_device_ptr<std::byte> volume_;
  curaii::unique_device_ptr<float>     plane_;
  holoflow::core::TDesc                desc_;
};
} // namespace holovibes::pipeline
