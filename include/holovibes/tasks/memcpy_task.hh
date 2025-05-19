#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "holoflow/task.hh"

using json = nlohmann::json;

namespace holovibes::tasks {

struct MemcpyParams {
  enum class Kind {
    HostToHost,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
  };

  Kind kind;
};

void to_json(nlohmann::json &j, const MemcpyParams &p);
void from_json(const nlohmann::json &j, MemcpyParams &p);

class Memcpy : public dh::Task {
public:
  void run(dh::TensorView itens, dh::TensorView otens) override;

  friend class MemcpyFactory;

private:
  Memcpy(const dh::TaskMeta &meta, cudaStream_t stream, cudaMemcpyKind kind);

  cudaMemcpyKind kind_;
};

class MemcpyFactory : public dh::TaskFactory {
public:
  dh::TaskMeta type_check(const dh::TensorMeta &imeta,
                          const json &params) override;

  std::unique_ptr<dh::Task> create(const dh::TensorMeta &imeta,
                                   const json &params,
                                   cudaStream_t stream) override;
};

} // namespace holovibes::tasks
