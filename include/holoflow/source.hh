#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "holoflow/error.hh"
#include "holoflow/tensor.hh"

using json = nlohmann::json;

namespace dh {

class SourceMeta {
public:
  SourceMeta(const TensorMeta &ometa);

  const TensorMeta &ometa() const;

private:
  TensorMeta ometa_;
};

std::ostream &operator<<(std::ostream &os, const SourceMeta &meta);

class Source {
public:
  Source(const SourceMeta &meta, cudaStream_t stream);

  virtual ~Source() = default;

  Source(const Source &) = delete;
  Source &operator=(const Source &) = delete;
  Source(Source &&) = delete;
  Source &operator=(Source &&) = delete;

  virtual tl::expected<void, Error> run(TensorView otens) = 0;

  const SourceMeta &meta() const;

  const TensorMeta &ometa() const;

protected:
  SourceMeta meta_;
  cudaStream_t stream_;
};

class SourceFactory {
public:
  SourceFactory() = default;

  virtual ~SourceFactory() = default;

  SourceFactory(const SourceFactory &) = delete;
  SourceFactory &operator=(const SourceFactory &) = delete;
  SourceFactory(SourceFactory &&) = delete;
  SourceFactory &operator=(SourceFactory &&) = delete;

  virtual tl::expected<SourceMeta, Error> type_check(const json &params) = 0;

  virtual tl::expected<std::unique_ptr<Source>, Error>
  create(const json &params, cudaStream_t stream) = 0;
};

} // namespace dh