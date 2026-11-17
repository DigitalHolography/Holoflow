// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0.

#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <filesystem>
#include <stdexcept>
#include <vector>

namespace holotask::detail {

inline std::filesystem::path containing_module_directory() {
  static const char anchor = 0;
  HMODULE           module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&anchor), &module))
    throw std::runtime_error("Cannot locate Holoflow runtime module");
  std::vector<wchar_t> buffer(MAX_PATH);
  for (;;) {
    const auto length =
        GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (!length)
      throw std::runtime_error("Cannot locate Holoflow runtime module path");
    if (length < buffer.size() - 1)
      return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
    buffer.resize(buffer.size() * 2);
  }
}

inline std::filesystem::path cuda_include_directory() {
  const auto module = containing_module_directory();
  for (const auto &candidate :
       {module / "_assets/nvrtc/cuda/include", module / "../share/holovibes/nvrtc/cuda/include",
        std::filesystem::path{HOLOFLOW_CUDA_INCLUDE_DIR}}) {
    if (std::filesystem::is_regular_file(candidate / "cuda_runtime.h"))
      return candidate;
  }
  throw std::runtime_error("CUDA runtime-compilation headers are missing");
}

} // namespace holotask::detail
