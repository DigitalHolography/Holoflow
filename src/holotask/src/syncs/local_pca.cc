#include "holotask/syncs/local_pca.hh"
#include "local_pca_processor.hh"
#include "logger.hh"
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace holotask::syncs {
void to_json(nlohmann::json &j, const LocalPcaSettings &s) {
  j = {{"width_x", s.width_x}, {"width_y", s.width_y},
       {"stride_x", s.stride_x}, {"stride_y", s.stride_y},
       {"rank_factor", s.rank_factor}};
}
void from_json(const nlohmann::json &j, LocalPcaSettings &s) {
  s.width_x = j.value("width_x", 32);
  s.width_y = j.value("width_y", 32);
  s.stride_x = j.value("stride_x", 16);
  s.stride_y = j.value("stride_y", 16);
  s.rank_factor = j.value("rank_factor", 1.f);
}

namespace {
class LocalPca final : public holoflow::core::ISyncTask {
public:
  LocalPca(const holoflow::core::TDesc &d, LocalPcaSettings s,
                   cudaStream_t stream)
      : settings_(s), desc_(d), stream_(stream),
        processor_(static_cast<unsigned>(d.shape[2]),
                                 static_cast<unsigned>(d.shape[1]),
                                 static_cast<unsigned>(d.shape[0]), s.width_x, s.width_y,
                                 s.stride_x, s.stride_y, stream) {}
  bool can_reuse(const holoflow::core::TDesc &d, const LocalPcaSettings &s,
                 cudaStream_t stream) const {
    return desc_.shape == d.shape && desc_.strides == d.strides &&
           desc_.dtype == d.dtype && desc_.mem_loc == d.mem_loc && stream_ == stream &&
           settings_.width_x == s.width_x && settings_.width_y == s.width_y &&
           settings_.stride_x == s.stride_x && settings_.stride_y == s.stride_y;
  }
  void set_rank_factor(float rank_factor) { settings_.rank_factor = rank_factor; }
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (!started_) {
      logger()->info("[LocalPca] started: {} frames/window, {}x{} kernel, {}x{} step, rank factor {}",
                     desc_.shape[0], settings_.width_x, settings_.width_y,
                     settings_.stride_x, settings_.stride_y, settings_.rank_factor);
      started_at_ = std::chrono::steady_clock::now();
      next_progress_ = started_at_ + std::chrono::seconds(1);
      started_ = true;
    }
    const bool completed = processor_.process(
        reinterpret_cast<const cuComplex *>(ctx.inputs[0].data()),
        reinterpret_cast<cuComplex *>(ctx.outputs[0].data()), settings_.rank_factor,
        ctx.cancelled);
    if (completed) {
      ++windows_submitted_;
      const auto now = std::chrono::steady_clock::now();
      if (logger()->should_log(spdlog::level::debug) && now >= next_progress_) {
        const auto seconds = std::chrono::duration<double>(now - started_at_).count();
        logger()->debug("[LocalPca] {} temporal windows submitted in {:.1f} s ({:.1f} windows/s)",
                        windows_submitted_, seconds, windows_submitted_ / seconds);
        next_progress_ = now + std::chrono::seconds(1);
      }
    }
    return completed ? holoflow::core::OpResult::Ok : holoflow::core::OpResult::Cancelled;
  }
private:
  LocalPcaSettings settings_;
  holoflow::core::TDesc desc_;
  cudaStream_t stream_;
  holovibes::compute::LocalPca processor_;
  std::chrono::steady_clock::time_point started_at_;
  std::chrono::steady_clock::time_point next_progress_;
  bool started_ = false;
  size_t windows_submitted_ = 0;
};
}

holoflow::core::InferResult LocalPcaFactory::infer(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &json) const {
  const auto s = json.get<LocalPcaSettings>();
  if (inputs.size() != 1) throw std::invalid_argument("Local PCA expects one input");
  const auto &d = inputs.front();
  if (d.rank() != 3 || d.dtype != holoflow::core::DType::CF32 ||
      d.mem_loc != holoflow::core::MemLoc::Device || d.shape[0] == 0 ||
      d.shape[1] == 0 || d.shape[2] == 0 ||
      d.strides != holoflow::core::TDesc(d.shape, d.dtype, d.mem_loc).strides ||
      s.width_x < 1 || s.width_y < 1 || s.stride_x < 1 || s.stride_y < 1 ||
      s.stride_x > s.width_x || s.stride_y > s.width_y ||
      !std::isfinite(s.rank_factor) || s.rank_factor < 0.f)
    throw std::invalid_argument("Local PCA requires contiguous CF32 [T,H,W], positive kernel/stride and nonnegative rank factor");
  return {.input_descs = {d}, .output_descs = {d}, .in_place = {},
          .owned_inputs = {false}, .owned_outputs = {false},
          .kind = holoflow::core::TaskKind::Sync};
}
std::unique_ptr<holoflow::core::ISyncTask> LocalPcaFactory::create(
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
    const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  return std::make_unique<LocalPca>(inputs.front(),
                                             settings.get<LocalPcaSettings>(), ctx.stream);
}
std::unique_ptr<holoflow::core::ISyncTask> LocalPcaFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask> old_task,
    std::span<const holoflow::core::TDesc> inputs, const nlohmann::json &settings,
    const holoflow::core::SyncCreateCtx &ctx) const {
  (void)infer(inputs, settings);
  const auto config = settings.get<LocalPcaSettings>();
  auto *task = dynamic_cast<LocalPca *>(old_task.get());
  if (task && task->can_reuse(inputs.front(), config, ctx.stream)) {
    task->set_rank_factor(config.rank_factor);
    return old_task;
  }
  old_task.reset();
  return create(inputs, settings, ctx);
}
} // namespace holotask::syncs
