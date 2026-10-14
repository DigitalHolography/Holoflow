// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "holotask/sinks/npy_volume.hh"

#include <algorithm>
#include <bit>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>

#include "curaii/cuda.hh"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif

namespace holotask::sinks {
void to_json(nlohmann::json &j, const NpyVolumeSettings &s) {
  j = {{"path", s.path}, {"metadata", s.metadata}, {"overwrite", s.overwrite}};
}
void from_json(const nlohmann::json &j, NpyVolumeSettings &s) {
  j.at("path").get_to(s.path);
  s.metadata  = j.value("metadata", nlohmann::json::object());
  s.overwrite = j.value("overwrite", false);
}

namespace {
using namespace holoflow::core;
constexpr size_t download_chunk = 8 * 1024 * 1024;

bool contiguous(const TDesc &desc) {
  size_t stride = size_of(desc.dtype);
  if (desc.shape.size() != desc.strides.size())
    return false;
  for (size_t i = desc.rank(); i-- > 0;) {
    if (desc.shape[i] == 0 || desc.strides[i] != stride)
      return false;
    if (desc.shape[i] > std::numeric_limits<size_t>::max() / stride)
      return false;
    stride *= desc.shape[i];
  }
  return true;
}

class PendingFile {
public:
  explicit PendingFile(std::filesystem::path destination) : destination_(std::move(destination)) {
    path_ = destination_;
    path_ +=
        ".partial-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  }
  ~PendingFile() {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  const std::filesystem::path &path() const { return path_; }
  void                         commit(bool overwrite) {
    if (!overwrite && std::filesystem::exists(destination_))
      throw std::runtime_error("NPY output already exists: " + destination_.string());
#ifdef _WIN32
    const DWORD flags = MOVEFILE_WRITE_THROUGH | (overwrite ? MOVEFILE_REPLACE_EXISTING : 0);
    if (!MoveFileExW(path_.c_str(), destination_.c_str(), flags))
      throw std::system_error(static_cast<int>(GetLastError()), std::system_category(),
                              "Could not publish NPY output");
#else
    std::filesystem::rename(path_, destination_);
#endif
  }

private:
  std::filesystem::path path_, destination_;
};

// Preserve both previous outputs until the complete NPY/metadata pair is published.
class OutputPublication {
public:
  OutputPublication(std::filesystem::path destination, bool overwrite)
      : destination_(std::move(destination)) {
    if (!std::filesystem::exists(destination_))
      return;
    if (!overwrite || !std::filesystem::is_regular_file(destination_))
      throw std::runtime_error("NPY output cannot be replaced: " + destination_.string());
    backup_ = destination_;
    backup_ +=
        ".backup-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::rename(destination_, backup_);
  }
  ~OutputPublication() {
    std::error_code error;
    if (complete_) {
      if (!backup_.empty())
        std::filesystem::remove(backup_, error);
      return;
    }
    if (published_)
      std::filesystem::remove(destination_, error);
    if (!backup_.empty())
      std::filesystem::rename(backup_, destination_, error);
  }
  void publish(PendingFile &file) {
    file.commit(false);
    published_ = true;
  }
  void complete() noexcept { complete_ = true; }

private:
  std::filesystem::path destination_, backup_;
  bool                  published_ = false, complete_ = false;
};

std::string header(const TDesc &desc) {
  std::ostringstream text;
  text << "{'descr': '" << (desc.dtype == DType::CF32 ? "<c8" : "<f4")
       << "', 'fortran_order': False, 'shape': (";
  for (const auto size : desc.shape)
    text << size << ", ";
  text << "), }";
  auto result = text.str();
  result.append((64 - (10 + result.size() + 1) % 64) % 64, ' ');
  result += '\n';
  if (result.size() > 65535)
    throw std::runtime_error("NPY header exceeds version 1 length limit");
  return result;
}

class NpyVolumeWriter : public ISyncTask {
public:
  NpyVolumeWriter(TDesc input, NpyVolumeSettings s, cudaStream_t stream)
      : input_(std::move(input)), s_(std::move(s)), stream_(stream),
        staging_(
            curaii::make_unique_host_ptr<std::byte>(std::min(download_chunk, input_.num_bytes()))) {
  }
  OpResult execute(SyncCtx &ctx) override {
    static_assert(std::endian::native == std::endian::little);
    const std::filesystem::path destination(s_.path);
    if (!destination.parent_path().empty())
      std::filesystem::create_directories(destination.parent_path());
    PendingFile   file(destination), sidecar(destination.string() + ".json");
    std::ofstream output(file.path(), std::ios::binary);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    const auto          dictionary = header(input_);
    const auto          length     = static_cast<uint16_t>(dictionary.size());
    const unsigned char prefix[]   = {0x93,
                                      'N',
                                      'U',
                                      'M',
                                      'P',
                                      'Y',
                                      1,
                                      0,
                                      static_cast<unsigned char>(length & 255),
                                      static_cast<unsigned char>(length >> 8)};
    output.write(reinterpret_cast<const char *>(prefix), sizeof(prefix));
    output.write(dictionary.data(), static_cast<std::streamsize>(dictionary.size()));
    for (size_t offset = 0; offset < input_.num_bytes(); offset += download_chunk) {
      if (ctx.cancelled != nullptr && ctx.cancelled->load(std::memory_order_relaxed))
        return OpResult::Cancelled;
      const size_t bytes  = std::min(download_chunk, input_.num_bytes() - offset);
      const auto  *source = ctx.inputs[0].data() + offset;
      if (input_.mem_loc == MemLoc::Device) {
        CUDA_CHECK(cudaMemcpyAsync(staging_.get(), source, bytes, cudaMemcpyDeviceToHost, stream_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        source = staging_.get();
      }
      output.write(reinterpret_cast<const char *>(source), static_cast<std::streamsize>(bytes));
    }
    output.close();
    auto metadata           = s_.metadata;
    metadata["shape"]       = input_.shape;
    metadata["dtype"]       = input_.dtype;
    metadata["array_order"] = "C";
    if (!metadata.contains("axes"))
      metadata["axes"] = input_.rank() == 3 ? "zyx" : "yx";
    std::ofstream info(sidecar.path());
    info.exceptions(std::ios::badbit | std::ios::failbit);
    info << metadata.dump(2) << '\n';
    info.close();
    OutputPublication volume_publication(destination, s_.overwrite);
    OutputPublication metadata_publication(destination.string() + ".json", s_.overwrite);
    volume_publication.publish(file);
    metadata_publication.publish(sidecar);
    volume_publication.complete();
    metadata_publication.complete();
    return OpResult::Ok;
  }

private:
  TDesc                              input_;
  NpyVolumeSettings                  s_;
  cudaStream_t                       stream_;
  curaii::unique_host_ptr<std::byte> staging_;
};
} // namespace

holoflow::core::InferResult NpyVolumeFactory::infer(std::span<const TDesc> inputs,
                                                    const nlohmann::json  &settings) const {
  const auto s = settings.get<NpyVolumeSettings>();
  if (inputs.size() != 1)
    throw std::invalid_argument("NPY writer requires one tensor");
  const auto &in = inputs[0];
  if ((in.rank() != 2 && in.rank() != 3) || !contiguous(in) ||
      (in.dtype != DType::F32 && in.dtype != DType::CF32))
    throw std::invalid_argument("NPY writer expects contiguous F32/CF32 planes or volumes");
  if (s.path.empty() || !s.metadata.is_object())
    throw std::invalid_argument("NPY output path and object metadata are required");
  if (!s.overwrite &&
      (std::filesystem::exists(s.path) || std::filesystem::exists(s.path + ".json")))
    throw std::invalid_argument("NPY output already exists; use overwrite explicitly");
  return {.input_descs   = {in},
          .output_descs  = {},
          .owned_inputs  = {false},
          .owned_outputs = {},
          .kind          = TaskKind::Sync};
}
std::unique_ptr<ISyncTask> NpyVolumeFactory::create(std::span<const TDesc> inputs,
                                                    const nlohmann::json  &settings,
                                                    const SyncCreateCtx   &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<NpyVolumeWriter>(inputs[0], settings.get<NpyVolumeSettings>(),
                                           ctx.stream);
}
} // namespace holotask::sinks
