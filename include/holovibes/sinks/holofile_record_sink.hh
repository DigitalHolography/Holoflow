#pragma once

#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>

#include "curaii/v2/cuda.hh"
#include "holoflow/sink.hh"
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

class HolofileRecordSink : public Sink {
public:
  void run(TensorView itens) override;

  friend class QtDisplaySinkFactory;

private:
};

} // namespace dh