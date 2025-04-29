#pragma once

#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>

#include "curaii/v2/cuda.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/tensor.hh"

using json = nlohmann::json;

namespace nlohmann {
template <typename T> struct adl_serializer<std::optional<T>> {
  static void to_json(json &j, std::optional<T> const &opt) {
    if (opt) {
      j = *opt;
    } else {
      j = nullptr;
    }
  }

  static void from_json(json const &j, std::optional<T> &opt) {
    if (j.is_null()) {
      opt = std::nullopt;
    } else {
      opt = j.get<T>();
    }
  }
};
} // namespace nlohmann

namespace dh {

class GateAccumulator : public Accumulator {
public:
  std::optional<TensorView> write_tensor() override;

  void commit_write() override;

  std::optional<TensorView> read_tensor() override;

  void commit_read() override;

  void handle_event(const json &event) override;

  friend class GateAccumulatorFactory;

private:
  GateAccumulator(const AccumulatorMeta &meta, cudaStream_t stream,
                  Accumulator::EventListeners event_listeners, bool is_on,
                  std::optional<size_t> target,
                  curaii::cuda::unique_host_ptr<uint8_t> h_buffer,
                  curaii::cuda::unique_device_ptr<uint8_t> d_buffer);

  std::atomic<bool> write_;
  std::atomic<bool> is_on_;
  std::optional<size_t> target_;
  size_t frames_passed_;

  uint8_t *buffer_;
  curaii::cuda::unique_host_ptr<uint8_t> h_buffer_;
  curaii::cuda::unique_device_ptr<uint8_t> d_buffer_;
};

class GateAccumulatorFactory : public AccumulatorFactory {
  AccumulatorMeta type_check(const TensorMeta &imeta,
                             const json &params) override;

  std::unique_ptr<Accumulator>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream,
         Accumulator::EventListeners event_listeners) override;

private:
  struct Params {
    bool is_on;
    std::optional<size_t> target;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, is_on, target);
  };
};

} // namespace dh