// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "holoflow/runtime/tracing.hh"
#include "runtime_assets.hh"
#include <tlhelp32.h>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <span>

#include <nvrtc.h>

namespace holotask::detail {

class NvrtcCache {
public:
  NvrtcCache(std::string_view kind, std::string_view source_id,
             std::span<const std::string> options, const std::filesystem::path &link_input = {}) {
    try {
      if (const auto *disabled = std::getenv("HOLOFLOW_NVRTC_CACHE_DISABLE");
          disabled && std::string_view(disabled) == "1")
        return;
      const auto *override_dir = _wgetenv(L"HOLOFLOW_NVRTC_CACHE_DIR");
      const auto *local        = _wgetenv(L"LOCALAPPDATA");
      if ((!override_dir || !*override_dir) && (!local || !*local))
        return;
      auto dir = override_dir && *override_dir
                     ? std::filesystem::path(override_dir)
                     : std::filesystem::path(local) / "Holoflow/NvrtcCache/v1";

      append(kind);
      append(source_id);
      append(HOLOFLOW_NVRTC_ASSET_VERSION);
      for (const auto &option : options)
        append(option);
      if (!link_input.empty())
        append(file_identity(link_input));
      const auto toolchain = toolchain_identity(!link_input.empty());
      if (toolchain.empty())
        return;
      append(toolchain);
      path_ = dir / (std::string(kind) + "-" + std::to_string(digest(key_)) + ".bin");
    } catch (...) {
      path_.clear();
    }
  }

  std::vector<char> read() const noexcept {
    if (path_.empty())
      return {};
    try {
      holoflow::runtime::tracing::ScopedTrace lookup("NVRTC Disk Cache Lookup", "detail");
      std::ifstream                           file(path_, std::ios::binary | std::ios::ate);
      const auto                              size = file.tellg();
      if (size < 0 || static_cast<uint64_t>(size) > max_bytes)
        return {};
      file.seekg(0);
      std::string key(key_.size(), '\0');
      uint64_t    checksum{};
      if (!file.read(key.data(), key.size()) || key != key_ ||
          !file.read(reinterpret_cast<char *>(&checksum), sizeof(checksum)))
        return {};
      const auto payload_size = static_cast<uint64_t>(size) - key.size() - sizeof(checksum);
      if (payload_size == 0)
        return {};
      std::vector<char> payload(static_cast<size_t>(payload_size));
      if (!file.read(payload.data(), payload.size()) ||
          checksum != digest({payload.data(), payload.size()}))
        return {};
      holoflow::runtime::tracing::ScopedTrace hit("NVRTC Disk Cache Hit", "detail");
      return payload;
    } catch (...) {
      return {};
    }
  }

  void write(std::span<const char> payload) const noexcept {
    if (path_.empty() || payload.empty() ||
        payload.size() + key_.size() + sizeof(uint64_t) > max_bytes)
      return;
    wchar_t temporary[MAX_PATH]{};
    try {
      holoflow::runtime::tracing::ScopedTrace timing("NVRTC Disk Cache Write", "detail");
      const auto                              dir = path_.parent_path();
      std::filesystem::create_directories(dir);
      if (!GetTempFileNameW(dir.c_str(), L"hfc", 0, temporary))
        return;
      std::ofstream file(std::filesystem::path(temporary), std::ios::binary | std::ios::trunc);
      const auto    checksum = digest({payload.data(), payload.size()});
      file.write(key_.data(), key_.size());
      file.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));
      file.write(payload.data(), payload.size());
      file.close();
      if (file)
        MoveFileExW(temporary, path_.c_str(), MOVEFILE_REPLACE_EXISTING);
    } catch (...) {
    }
    if (*temporary)
      DeleteFileW(temporary);
  }

private:
  static constexpr uint64_t max_bytes = 64 * 1024 * 1024;
  std::string               key_;
  std::filesystem::path     path_;

  void append(std::string_view value) {
    key_ += std::to_string(value.size()) + ":" + std::string(value);
  }

  static uint64_t digest(std::string_view data) noexcept {
    uint64_t hash = 14695981039346656037ull;
    for (const unsigned char byte : data) {
      hash ^= byte;
      hash *= 1099511628211ull;
    }
    return hash;
  }

  static std::string file_identity(const std::filesystem::path &path) {
    return path.string() + ":" + std::to_string(std::filesystem::file_size(path)) + ":" +
           std::to_string(std::filesystem::last_write_time(path).time_since_epoch().count());
  }

  static std::string toolchain_identity(bool linked) {
    int major{}, minor{};
    if (nvrtcVersion(&major, &minor) != NVRTC_SUCCESS)
      return {};
    const auto builtins_name =
        L"nvrtc-builtins64_" + std::to_wstring(major) + std::to_wstring(minor) + L".dll";
    const auto handle = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, GetCurrentProcessId());
    if (handle == INVALID_HANDLE_VALUE)
      return {};
    const std::unique_ptr<void, decltype(&CloseHandle)> snapshot(handle, CloseHandle);
    std::filesystem::path                               nvrtc, builtins, jitlink;
    MODULEENTRY32W                                      entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL valid = Module32FirstW(snapshot.get(), &entry); valid;
         valid      = Module32NextW(snapshot.get(), &entry)) {
      const std::wstring_view name(entry.szModule);
      if (name.starts_with(L"nvrtc64_"))
        nvrtc = entry.szExePath;
      else if (name == builtins_name)
        builtins = entry.szExePath;
      else if (name.starts_with(L"nvJitLink_"))
        jitlink = entry.szExePath;
    }
    if (nvrtc.empty() || (linked && jitlink.empty()))
      return {};
    if (builtins.empty())
      builtins = nvrtc.parent_path() / builtins_name;
    return file_identity(nvrtc) + "|" + file_identity(builtins) +
           (linked ? "|" + file_identity(jitlink) : "");
  }
};

} // namespace holotask::detail
