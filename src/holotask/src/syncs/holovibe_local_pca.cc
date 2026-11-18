#include "holotask/syncs/holovibe_local_pca.hh"
#include "holovibe_local_pca_processor.hh"
#include "logger.hh"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace holotask::syncs {
void to_json(nlohmann::json &j, const HolovibeLocalPcaSettings &s) {
  j = {{"width_x", s.width_x}, {"width_y", s.width_y},
       {"stride_x", s.stride_x}, {"stride_y", s.stride_y},
       {"rank_factor", s.rank_factor}};
}
void from_json(const nlohmann::json &j, HolovibeLocalPcaSettings &s) {
  s.width_x = j.value("width_x", 32);
  s.width_y = j.value("width_y", 32);
  s.stride_x = j.value("stride_x", 16);
  s.stride_y = j.value("stride_y", 16);
  s.rank_factor = j.value("rank_factor", 1.f);
}

namespace {
class HolovibeLocalPca final : public holoflow::core::ISyncTask {
public:
  HolovibeLocalPca(const holoflow::core::TDesc &d, HolovibeLocalPcaSettings s,
                   cudaStream_t stream)
      : settings_(s), frames_(d.shape[0]),
        kernels_(((d.shape[2] + s.stride_x - 1) / s.stride_x) *
                 ((d.shape[1] + s.stride_y - 1) / s.stride_y)),
        processor_(static_cast<unsigned>(d.shape[2]),
                                 static_cast<unsigned>(d.shape[1]),
                                 static_cast<unsigned>(d.shape[0]), s.width_x, s.width_y,
                                 s.stride_x, s.stride_y, stream) {}
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    const auto started = std::chrono::steady_clock::now();
    logger()->info("[HolovibeLocalPca] processing {} frames across {} kernels ({}x{}, steps {}x{}, rank factor {})",
                   frames_, kernels_, settings_.width_x, settings_.width_y,
                   settings_.stride_x, settings_.stride_y, settings_.rank_factor);
    std::jthread progress([this, started](std::stop_token stop) {
      std::mutex mutex;
      std::condition_variable_any wake;
      std::unique_lock lock(mutex);
      while (!stop.stop_requested()) {
        wake.wait_for(lock, stop, std::chrono::seconds(5), [] { return false; });
        if (!stop.stop_requested())
          logger()->info("[HolovibeLocalPca] still processing ({} s elapsed)",
                         std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::steady_clock::now() - started).count());
      }
    });
    const bool completed = processor_.process(
        reinterpret_cast<const cuComplex *>(ctx.inputs[0].data()),
        reinterpret_cast<cuComplex *>(ctx.outputs[0].data()), settings_.rank_factor,
        ctx.cancelled);
    progress.request_stop();
    logger()->info("[HolovibeLocalPca] {} {}-frame block in {} ms",
                   completed ? "completed" : "cancelled", frames_,
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - started).count());
    return completed ? holoflow::core::OpResult::Ok : holoflow::core::OpResult::Cancelled;
  }
private:
  HolovibeLocalPcaSettings settings_;
  size_t frames_;
  size_t kernels_;
  holovibes::compute::LocalPca processor_;
};
}

holoflow::core::InferResult HolovibeLocalPcaFactory::infer(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json) const {
  const auto s = json.get<HolovibeLocalPcaSettings>();
  if (inputs.size() != 1) throw std::invalid_argument("Holovibe Local-PCA expects one input");
  const auto &d = inputs.front();
  if (d.rank() != 3 || d.dtype != holoflow::core::DType::CF32 ||
      d.mem_loc != holoflow::core::MemLoc::Device || d.shape[0] == 0 ||
      d.shape[1] == 0 || d.shape[2] == 0 ||
      d.strides != holoflow::core::TDesc(d.shape, d.dtype, d.mem_loc).strides ||
      s.width_x < 1 || s.width_y < 1 || s.stride_x < 1 || s.stride_y < 1 ||
      s.stride_x > s.width_x || s.stride_y > s.width_y ||
      !std::isfinite(s.rank_factor) || s.rank_factor < 0.f)
    throw std::invalid_argument("Holovibe Local-PCA requires contiguous CF32 [T,H,W], positive kernel/stride and nonnegative rank factor");
  return {.input_descs = {d}, .output_descs = {d}, .in_place = {},
          .owned_inputs = {false}, .owned_outputs = {false},
          .kind = holoflow::core::TaskKind::Sync};
}
std::unique_ptr<holoflow::core::ISyncTask> HolovibeLocalPcaFactory::create(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
    const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<HolovibeLocalPca>(inputs.front(),
                                             settings.get<HolovibeLocalPcaSettings>(), ctx.stream);
}
std::unique_ptr<holoflow::core::ISyncTask> HolovibeLocalPcaFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask> old_task,
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
    const holoflow::core::SyncCreateCtx &ctx) const {
  old_task.reset();
  return create(inputs, settings, ctx);
}
} // namespace holotask::syncs
