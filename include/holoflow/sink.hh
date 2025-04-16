#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/cuda_runtime.hh"
#include "holoflow/error.hh"
#include "holoflow/tensor.hh"

using json = nlohmann::json;

namespace dh {

class SinkMeta {
public:
  SinkMeta(const TensorMeta &imeta);

  const TensorMeta &imeta() const;

private:
  TensorMeta imeta_;
};

class Sink {
public:
  Sink(const SinkMeta &meta, CudaStreamRef stream);

  virtual ~Sink() = default;

  Sink(const Sink &) = delete;
  Sink &operator=(const Sink &) = delete;
  Sink(Sink &&) = delete;
  Sink &operator=(Sink &&) = delete;

  virtual tl::expected<void, Error> run(TensorView itens) = 0;

  const SinkMeta &meta() const;

  const TensorMeta &imeta() const;

protected:
  SinkMeta meta_;
  CudaStreamRef stream_;
};

class SinkFactory {
public:
  SinkFactory() = default;

  virtual ~SinkFactory() = default;

  SinkFactory(const SinkFactory &) = delete;
  SinkFactory &operator=(const SinkFactory &) = delete;
  SinkFactory(SinkFactory &&) = delete;
  SinkFactory &operator=(SinkFactory &&) = delete;

  virtual tl::expected<SinkMeta, Error> type_check(const TensorMeta &imeta,
                                                   const json &params) = 0;

  virtual tl::expected<std::unique_ptr<Sink>, Error>
  create(const TensorMeta &imeta, const json &params, CudaStreamRef stream) = 0;
};

} // namespace dh