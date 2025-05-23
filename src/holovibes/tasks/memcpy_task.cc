#include "holovibes/tasks/memcpy_task.hh"

#include <cuda_runtime.h>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"

namespace holovibes::tasks {

// ==========================================================================
//                     MemcpyParams Implementation
// ==========================================================================

void to_json(nlohmann::json &j, const MemcpyParams &p) {
  static const std::map<MemcpyParams::Kind, std::string> kind_to_string = {
      {MemcpyParams::Kind::HostToHost, "HostToHost"},
      {MemcpyParams::Kind::HostToDevice, "HostToDevice"},
      {MemcpyParams::Kind::DeviceToHost, "DeviceToHost"},
      {MemcpyParams::Kind::DeviceToDevice, "DeviceToDevice"}};

  DH_CHECK(kind_to_string.contains(p.kind));
  j = nlohmann::json{{"kind", kind_to_string.at(p.kind)}};
}

void from_json(const nlohmann::json &j, MemcpyParams &p) {
  static const std::map<std::string, MemcpyParams::Kind> string_to_kind = {
      {"HostToHost", MemcpyParams::Kind::HostToHost},
      {"HostToDevice", MemcpyParams::Kind::HostToDevice},
      {"DeviceToHost", MemcpyParams::Kind::DeviceToHost},
      {"DeviceToDevice", MemcpyParams::Kind::DeviceToDevice}};

  auto kind_str = j.at("kind").get<std::string>();
  if (!string_to_kind.contains(kind_str)) {
    throw std::invalid_argument("Invalid MemcpyParams kind: " + kind_str);
  }

  p.kind = string_to_kind.at(kind_str);
}

// ==========================================================================
//                     Memcpy Implementation
// ==========================================================================

Memcpy::Memcpy(const dh::TaskMeta &meta, cudaStream_t stream,
               cudaMemcpyKind kind)
    : dh::Task(meta, stream), kind_(kind) {}

void Memcpy::run(dh::TensorView input, dh::TensorView output) {
  CUDA_CHECK(cudaMemcpyAsync(output.data(), input.data(), input.size_in_bytes(),
                             kind_, stream_));
}

// ==========================================================================
//                     MemcpyFactory Implementation
// ==========================================================================

dh::TaskMeta MemcpyFactory::type_check(const dh::TensorMeta &imeta,
                                       const json           &jparams) {
  // Unpack parameters
  const MemcpyParams params = jparams.get<MemcpyParams>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // Tensor meta sanity
  std::map<MemcpyParams::Kind, dh::MemoryLocation> kind_to_src_loc = {
      {MemcpyParams::Kind::HostToHost, dh::MemoryLocation::HOST},
      {MemcpyParams::Kind::HostToDevice, dh::MemoryLocation::HOST},
      {MemcpyParams::Kind::DeviceToHost, dh::MemoryLocation::DEVICE},
      {MemcpyParams::Kind::DeviceToDevice, dh::MemoryLocation::DEVICE},
  };
  check(imeta.memory_location() == kind_to_src_loc.at(params.kind),
        "invalid memcpy: tensor is in the wrong memory for this kind");

  // Success
  std::map<MemcpyParams::Kind, dh::MemoryLocation> kind_to_dst_loc = {
      {MemcpyParams::Kind::HostToHost, dh::MemoryLocation::HOST},
      {MemcpyParams::Kind::HostToDevice, dh::MemoryLocation::DEVICE},
      {MemcpyParams::Kind::DeviceToHost, dh::MemoryLocation::HOST},
      {MemcpyParams::Kind::DeviceToDevice, dh::MemoryLocation::DEVICE},
  };
  auto           location = kind_to_dst_loc.at(params.kind);
  dh::TensorMeta ometa(imeta.data_type(), location, imeta.shape());
  return dh::TaskMeta(imeta, ometa, false);
}

std::unique_ptr<dh::Task> MemcpyFactory::create(const dh::TensorMeta &imeta,
                                                const json           &jparams,
                                                cudaStream_t          stream) {
  // Validate
  auto meta   = type_check(imeta, jparams);
  auto params = jparams.get<MemcpyParams>();

  // Extract memcpy kind
  std::map<MemcpyParams::Kind, cudaMemcpyKind> kind_to_cuda_kind = {
      {MemcpyParams::Kind::HostToHost, cudaMemcpyHostToHost},
      {MemcpyParams::Kind::HostToDevice, cudaMemcpyHostToDevice},
      {MemcpyParams::Kind::DeviceToHost, cudaMemcpyDeviceToHost},
      {MemcpyParams::Kind::DeviceToDevice, cudaMemcpyDeviceToDevice},
  };
  auto cuda_kind = kind_to_cuda_kind.at(params.kind);

  // 3) Assemble task
  auto *task = new Memcpy(meta, stream, cuda_kind);
  return std::unique_ptr<Memcpy>(task);
}

} // namespace holovibes::tasks