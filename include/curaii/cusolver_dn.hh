#pragma once

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <fmt/base.h>
#include <tl/expected.hpp>

namespace dh {

class CusolverDnParams {
public:
  explicit CusolverDnParams(cusolverDnParams_t params) noexcept;

  cusolverDnParams_t params() const noexcept;

private:
  cusolverDnParams_t params_;
};

class CusolverEigMode {
  explicit CusolverEigMode(cusolverEigMode_t eig_mode) noexcept;

  cusolverEigMode_t eig_mode() const noexcept;

private:
  cusolverEigMode_t eig_mode_;
};

} // namespace dh