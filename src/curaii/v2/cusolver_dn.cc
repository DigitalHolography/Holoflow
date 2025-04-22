#include "curaii/v2/cusolver_dn.hh"

#include <cstddef>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cusolver_common.hh"
#include "curaii/v2/logger.hh"

namespace curaii::cusolverdn {

Handle::Handle() { CUSOLVER_CHECK(cusolverDnCreate(&handle_)); }

Handle::Handle(cusolverDnHandle_t raw) noexcept : handle_(raw) {}

Handle::Handle(Handle &&other) noexcept : handle_(other.handle_) {
  other.handle_ = nullptr;
}

Handle &Handle::operator=(Handle &&other) noexcept {
  if (this != &other) {
    reset();
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

Handle::~Handle() noexcept {
  if (handle_) {
    CUSOLVER_CHECK_NT(cusolverDnDestroy(handle_));
  }
}

cusolverDnHandle_t Handle::get() const noexcept { return handle_; }

cusolverDnHandle_t Handle::release() noexcept {
  auto tmp = handle_;
  handle_ = nullptr;
  return tmp;
}

void Handle::reset(cusolverDnHandle_t raw) noexcept {
  if (handle_) {
    CUSOLVER_CHECK_NT(cusolverDnDestroy(handle_));
  }
  handle_ = raw;
}

Handle::operator bool() const noexcept { return handle_ != nullptr; }

Params::Params() { CUSOLVER_CHECK(cusolverDnCreateParams(&params_)); }

Params::Params(cusolverDnParams_t raw) noexcept : params_(raw) {}

Params::Params(Params &&other) noexcept : params_(other.params_) {
  other.params_ = nullptr;
}

Params &Params::operator=(Params &&other) noexcept {
  if (this != &other) {
    reset();
    params_ = other.params_;
    other.params_ = nullptr;
  }
  return *this;
}

Params::~Params() noexcept {
  if (params_) {
    CUSOLVER_CHECK_NT(cusolverDnDestroyParams(params_));
  }
}

cusolverDnParams_t Params::get() const noexcept { return params_; }

cusolverDnParams_t Params::release() noexcept {
  auto tmp = params_;
  params_ = nullptr;
  return tmp;
}

void Params::reset(cusolverDnParams_t raw) noexcept {
  if (params_) {
    CUSOLVER_CHECK_NT(cusolverDnDestroyParams(params_));
  }
  params_ = raw;
}

Params::operator bool() const noexcept { return params_ != nullptr; }

} // namespace curaii::cusolverdn