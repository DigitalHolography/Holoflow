// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "curaii/cufft.hh"
#include "curaii/nvrtc.hh"
#include "holoflow/runtime/detail/nvrtc_cache.hh"

#include <cuda.h>

#include <initializer_list>
#include <map>
#include <mutex>
#include <tuple>
#include <utility>

namespace holoflow::runtime::detail {

class CachedLto {
public:
  CachedLto() = default;

  CachedLto(std::string source, std::string name, std::vector<std::string> options,
            bool memoize = false)
      : recipe_(std::move(name), std::move(source), std::move(options)),
        disk_("lto-" + std::get<0>(recipe_), std::get<1>(recipe_), std::get<2>(recipe_),
              std::to_string(CUDA_VERSION)),
        memoize_(memoize) {
    const tracing::ScopedTrace scope("NVRTC Callback " + std::get<0>(recipe_), "detail");
    if (memoize_) {
      std::scoped_lock lock(memory_mutex_);
      if (auto it = memory_.find(recipe_); it != memory_.end()) {
        bytes_ = it->second;
        return;
      }
    }
    bytes_ = disk_.read();
    hit_   = !bytes_.empty();
    if (!hit_)
      compile();
  }

  const char       *data() const { return bytes_.data(); }
  size_t            size() const { return bytes_.size(); }
  bool              hit() const { return hit_; }
  std::vector<char> take() { return std::move(bytes_); }

  void rebuild() {
    if (hit_) {
      hit_ = false;
      compile();
    }
  }

  void publish() {
    if (compiled_)
      disk_.write(bytes_);
    if (memoize_) {
      std::scoped_lock lock(memory_mutex_);
      memory_.try_emplace(recipe_, bytes_);
    }
  }

private:
  using Recipe = std::tuple<std::string, std::string, std::vector<std::string>>;
  inline static std::mutex                          memory_mutex_;
  inline static std::map<Recipe, std::vector<char>> memory_;

  Recipe            recipe_;
  NvrtcCache        disk_;
  bool              memoize_  = false;
  bool              hit_      = false;
  bool              compiled_ = false;
  std::vector<char> bytes_;

  void compile() {
    const auto &[name, source, options] = recipe_;
    const tracing::ScopedTrace scope("NVRTC Compile " + name, "detail");
    curaii::NvrtcProgram       program(source.c_str(), name.c_str(), 0, nullptr, nullptr);
    std::vector<const char *>  args;
    args.reserve(options.size());
    for (const auto &option : options)
      args.push_back(option.c_str());
    try {
      NVRTC_CHECK(nvrtcCompileProgram(program.get(), static_cast<int>(args.size()), args.data()));
      size_t size = 0;
      NVRTC_CHECK(nvrtcGetLTOIRSize(program.get(), &size));
      bytes_.resize(size);
      NVRTC_CHECK(nvrtcGetLTOIR(program.get(), bytes_.data()));
      compiled_ = true;
    } catch (const curaii::NvrtcError &error) {
      size_t size = 0;
      NVRTC_CHECK(nvrtcGetProgramLogSize(program.get(), &size));
      std::string log(size, '\0');
      NVRTC_CHECK(nvrtcGetProgramLog(program.get(), log.data()));
      throw std::runtime_error(name + ": " + error.what() + "\n" + log);
    }
  }
};

template <typename Configure>
void configure_callback_plan(curaii::CufftHandle &plan, cudaStream_t stream,
                             std::initializer_list<CachedLto *> callbacks, Configure configure) {
  CUFFT_CHECK(cufftSetStream(plan.get(), stream));
  try {
    configure();
  } catch (const curaii::CufftError &) {
    bool hit = false;
    for (auto *callback : callbacks)
      hit |= callback->hit();
    if (!hit)
      throw;
    const tracing::ScopedTrace scope("NVRTC Cached Callback Retry", "detail");
    plan.reset(0);
    for (auto *callback : callbacks)
      callback->rebuild();
    plan = curaii::CufftHandle{};
    CUFFT_CHECK(cufftSetStream(plan.get(), stream));
    configure();
  }
  for (auto *callback : callbacks)
    callback->publish();
}

} // namespace holoflow::runtime::detail
