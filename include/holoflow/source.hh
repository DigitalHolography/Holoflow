#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/cuda_runtime.hh"
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

class Source {
public:
  Source(const SourceMeta &meta, CudaStreamRef stream);

  virtual ~Source() = default;

  Source(const Source &) = delete;
  Source &operator=(const Source &) = delete;
  Source(Source &&) = delete;
  Source &operator=(Source &&) = delete;

  virtual void run(TensorView otens) = 0;

  const SourceMeta &meta() const;

  const TensorMeta &ometa() const;

protected:
  SourceMeta meta_;
  CudaStreamRef stream_;
};

class SourceFactory {
public:
  SourceFactory() = default;

  virtual ~SourceFactory() = default;

  SourceFactory(const SourceFactory &) = delete;
  SourceFactory &operator=(const SourceFactory &) = delete;
  SourceFactory(SourceFactory &&) = delete;
  SourceFactory &operator=(SourceFactory &&) = delete;

  virtual SourceMeta type_check(const json &params) = 0;

  virtual std::unique_ptr<Source> create(const json &params,
                                         CudaStreamRef stream) = 0;
};

} // namespace dh