#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/curaii.hh"
#include "holofile/holofile.hh"
#include "holoflow/error.hh"
#include "holoflow/source.hh"

using json = nlohmann::json;

namespace dh {

class HolofileSource : public Source {
public:
  ~HolofileSource() = default;

  tl::expected<void, Error> run(TensorView otens) override;

  friend class HolofileSourceFactory;

private:
  enum class LoadKind {
    READ_LIVE,
    LOAD_IN_CPU,
    LOAD_IN_GPU,
  };

  HolofileSource(const SourceMeta &meta, cudaStream_t stream,
                 const std::string &path, int start_frame, int end_frame,
                 int batch_size, LoadKind load_kind, HolofileReader reader,
                 uint8_t *internal_buffer, unique_host_ptr<uint8_t> host_buffer,
                 unique_device_ptr<uint8_t> device_buffer);

  std::string path_;
  int start_frame_;
  int end_frame_;
  int batch_size_;
  LoadKind load_kind_;
  int frame_index_;
  HolofileReader reader_;
  HolofileHeader header_;
  uint8_t *internal_buffer_;
  unique_host_ptr<uint8_t> host_buffer_;
  unique_device_ptr<uint8_t> device_buffer_;
};

class HolofileSourceFactory : public SourceFactory {
public:
  tl::expected<SourceMeta, Error> type_check(const json &params);

  tl::expected<std::unique_ptr<Source>, Error> create(const json &params,
                                                      cudaStream_t stream);

private:
  struct Params {
    std::string path;
    int start_frame;
    int end_frame;
    int batch_size;
    std::string load_kind;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, path, start_frame, end_frame,
                                   batch_size, load_kind);
  };
};

} // namespace dh