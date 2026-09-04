#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "holoflow/core/tasks.hh"

namespace npyfile {

struct Header {
  std::vector<std::size_t> shape;
  std::string              dtype;
};

class Writer {
public:
  explicit Writer(const std::string &path, const Header &header);
  ~Writer();

  Writer(const Writer &)            = delete;
  Writer &operator=(const Writer &) = delete;

  void write_frames(const void *data, std::size_t frame_count, std::size_t frame_byte_size);

private:
  FILE *file_ = nullptr;
};

} // namespace npyfile

namespace holotask::sinks {

struct NpyfileSettings {
  std::string path;
  int         count;
  bool        use_buffer = true;

  bool operator==(const NpyfileSettings &) const = default;
};

void to_json(nlohmann::json &j, const NpyfileSettings &settings);
void from_json(const nlohmann::json &j, NpyfileSettings &settings);

class NpyfileFactory : public holoflow::core::ISyncTaskFactory {
public:
  holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc> input_descs,
                                    const nlohmann::json &jsettings) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  create(std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
  std::unique_ptr<holoflow::core::ISyncTask>
  update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
         std::span<const holoflow::core::TDesc> input_descs, const nlohmann::json &jsettings,
         const holoflow::core::SyncCreateCtx &ctx) const override;
};

} // namespace holotask::sinks
