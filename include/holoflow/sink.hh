#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

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

std::ostream &operator<<(std::ostream &os, const SinkMeta &meta);

class Sink {
public:
  Sink(const SinkMeta &meta, cudaStream_t stream);

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
  cudaStream_t stream_;
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
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream) = 0;
};

} // namespace dh