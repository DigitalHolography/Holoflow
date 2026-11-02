// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "display_mesi_history.hh"

#include <QByteArray>
#include <QMetaObject>
#include <QThread>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <cuda_runtime_api.h>

#include "bug.hh"
#include "ui/widgets/mesi_dashboard_widget.hh"
#include "ui/widgets/mesi_exposure_plot_widget.hh"
#include "ui/widgets/tensor_display_widget.hh"

namespace holovibes::tasks::sinks {
namespace {

struct RoiFit {
  double beta          = 0.0;
  double rho           = 0.0;
  double tau_frames    = 0.0;
  double squared_error = std::numeric_limits<double>::infinity();
};

double model_shape(const holotask::syncs::MesiModelTerms &terms, double rho) {
  return rho * rho * terms.double_decay + 2.0 * rho * (1.0 - rho) * terms.single_decay +
         (1.0 - rho) * (1.0 - rho);
}

RoiFit evaluate_roi_fit(const std::vector<double>                          &observed,
                        const std::vector<holotask::syncs::MesiModelTerms> &terms,
                        const holotask::syncs::MesiRemiSettings &settings, double rho,
                        double tau_frames) {
  const bool fixed       = settings.estimator == holotask::syncs::MesiEstimator::FixedLut;
  double     numerator   = 0.0;
  double     denominator = 0.0;
  for (size_t index = 0; index < observed.size(); ++index) {
    const double shape = model_shape(terms[index], rho);
    numerator += observed[index] * shape;
    denominator += shape * shape;
  }
  if (!(denominator > 0.0))
    return {};
  const double beta  = fixed ? settings.beta : std::clamp(numerator / denominator, 0.0, 1.0);
  double       error = 0.0;
  for (size_t index = 0; index < observed.size(); ++index) {
    const double residual = observed[index] - beta * model_shape(terms[index], rho);
    error += residual * residual;
  }
  return {.beta = beta, .rho = rho, .tau_frames = tau_frames, .squared_error = error};
}

RoiFit fit_roi_curve(const std::vector<double>               &observed,
                     const holotask::syncs::MesiRemiSettings &settings) {
  if (observed.size() != settings.exposure_frames.size() || observed.empty() ||
      !std::all_of(observed.begin(), observed.end(),
                   [](double value) { return std::isfinite(value) && value >= 0.0; }))
    return {};

  const bool   fixed     = settings.estimator == holotask::syncs::MesiEstimator::FixedLut;
  const int    tau_steps = fixed ? std::clamp(settings.tau_samples, 16, 512) : 128;
  const int    rho_steps = fixed ? 1 : 51;
  const double log_min   = std::log(settings.tau_min_frames);
  const double log_max   = std::log(settings.tau_max_frames);
  RoiFit       best;
  const auto   evaluate = [&](double log_tau, double rho) {
    const double                                 tau_frames = std::exp(log_tau);
    std::vector<holotask::syncs::MesiModelTerms> terms;
    terms.reserve(settings.exposure_frames.size());
    for (const int frames : settings.exposure_frames)
      terms.push_back(holotask::syncs::mesi_model_terms(settings, frames, tau_frames));
    return evaluate_roi_fit(observed, terms, settings, rho, tau_frames);
  };

  for (int tau_index = 0; tau_index < tau_steps; ++tau_index) {
    const double log_tau =
        log_min + (log_max - log_min) * static_cast<double>(tau_index) / (tau_steps - 1);
    const double                                 tau_frames = std::exp(log_tau);
    std::vector<holotask::syncs::MesiModelTerms> terms;
    terms.reserve(settings.exposure_frames.size());
    for (const int frames : settings.exposure_frames)
      terms.push_back(holotask::syncs::mesi_model_terms(settings, frames, tau_frames));
    for (int rho_index = 0; rho_index < rho_steps; ++rho_index) {
      const double rho = fixed ? settings.rho : static_cast<double>(rho_index) / (rho_steps - 1);
      const RoiFit candidate = evaluate_roi_fit(observed, terms, settings, rho, tau_frames);
      if (candidate.squared_error < best.squared_error)
        best = candidate;
    }
  }

  if (!std::isfinite(best.squared_error))
    return best;
  double tau_step = (log_max - log_min) / (tau_steps - 1);
  double rho_step = fixed ? 0.0 : 1.0 / (rho_steps - 1);
  for (int iteration = 0; iteration < 7; ++iteration) {
    const double center_log_tau = std::log(best.tau_frames);
    const double center_rho     = best.rho;
    for (int dt = -1; dt <= 1; ++dt) {
      for (int dr = fixed ? 0 : -1; dr <= (fixed ? 0 : 1); ++dr) {
        const double log_tau   = std::clamp(center_log_tau + dt * tau_step, log_min, log_max);
        const double rho       = std::clamp(center_rho + dr * rho_step, 0.0, 1.0);
        const RoiFit candidate = evaluate(log_tau, rho);
        if (candidate.squared_error < best.squared_error)
          best = candidate;
      }
    }
    tau_step *= 0.5;
    rho_step *= 0.5;
  }
  return best;
}

class DisplayMesiHistoryTask final : public holoflow::core::ISyncTask {
public:
  DisplayMesiHistoryTask(DisplayMesiHistorySettings        settings,
                         QPointer<ui::MesiDashboardWidget> dashboard, cudaStream_t stream,
                         std::vector<uint64_t> cached_indices, uint64_t cache_revision,
                         uint64_t map_revision)
      : settings_(settings), dashboard_(std::move(dashboard)), stream_(stream),
        k2_planes_(settings.model_settings.exposure_frames.size()), parameter_planes_(3),
        cached_packets_(settings.source_cycle_packets, false), cache_revision_(cache_revision),
        map_revision_(map_revision) {
    for (const uint64_t index : cached_indices)
      if (index < cached_packets_.size())
        cached_packets_[static_cast<size_t>(index)] = true;
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    if (dashboard_.isNull())
      return holoflow::core::OpResult::Ok;
    const auto    &desc         = ctx.inputs[0].desc;
    const int      height       = static_cast<int>(desc.shape[1]);
    const int      width        = static_cast<int>(desc.shape[2]);
    const uint64_t packet_index = sample_index_++;
    const size_t   packet_position =
        settings_.source_cycle_packets == 0
            ? 0
            : static_cast<size_t>(packet_index % settings_.source_cycle_packets);
    const uint64_t revision = dashboard_->curve_cache_revision();
    if (revision != cache_revision_) {
      std::fill(cached_packets_.begin(), cached_packets_.end(), false);
      cache_revision_ = revision;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_map_refresh_) {
      next_map_refresh_ = now + std::chrono::milliseconds(33);
      QByteArray map_bytes;
      map_bytes.resize(static_cast<qsizetype>(static_cast<size_t>(width) * height * sizeof(float)));
      auto &map = ctx.inputs[2];
      if (map.desc.mem_loc == holoflow::core::MemLoc::Device)
        CUDA_CHECK(cudaMemcpyAsync(map_bytes.data(), map.data(),
                                   static_cast<size_t>(map_bytes.size()), cudaMemcpyDeviceToHost,
                                   stream_));
      else
        std::memcpy(map_bytes.data(), map.data(), static_cast<size_t>(map_bytes.size()));

      std::optional<ui::MesiPixelValues> values;
      const QPoint                       hovered = dashboard_->pixel_to_sample();
      if (hovered.x() >= 0 && hovered.x() < width && hovered.y() >= 0 && hovered.y() < height) {
        values.emplace();
        values->pixel        = hovered;
        values->packet_index = settings_.source_cycle_packets == 0 ? packet_index : packet_position;
        values->k2.resize(k2_planes_.size());
        const int64_t offset     = static_cast<int64_t>(hovered.y()) * width + hovered.x();
        const int64_t plane_size = static_cast<int64_t>(width) * height;
        const auto    copy_pixel = [&](size_t input, size_t channel, float &destination) {
          auto       &tensor = ctx.inputs[input];
          const auto *source = reinterpret_cast<const float *>(tensor.data()) +
                               static_cast<int64_t>(channel) * plane_size + offset;
          if (tensor.desc.mem_loc == holoflow::core::MemLoc::Device)
            CUDA_CHECK(cudaMemcpyAsync(&destination, source, sizeof(float), cudaMemcpyDeviceToHost,
                                       stream_));
          else
            destination = *source;
        };
        for (size_t channel = 0; channel < values->k2.size(); ++channel)
          copy_pixel(0, channel, values->k2[channel]);
        copy_pixel(1, 0, values->beta);
        copy_pixel(1, 1, values->rho);
        copy_pixel(1, 2, values->tau_seconds);
      }
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      if (values) {
        const size_t offset = static_cast<size_t>(values->pixel.y()) * width + values->pixel.x();
        std::memcpy(&values->map_value, map_bytes.constData() + offset * sizeof(float),
                    sizeof(float));
      }
      QPointer<ui::MesiDashboardWidget> dashboard = dashboard_;
      QMetaObject::invokeMethod(
          dashboard,
          [dashboard, map_bytes = std::move(map_bytes), width, height, values = std::move(values),
           map_revision = map_revision_] {
            if (!dashboard || dashboard->map_revision() != map_revision)
              return;
            dashboard->tau_map_widget()->presentTensor(map_bytes, width, height,
                                                       holoflow::core::DType::F32);
            if (values)
              dashboard->set_pixel_values(*values);
          },
          Qt::QueuedConnection);
    }
    if (!cached_packets_.empty() && cached_packets_[packet_position])
      return holoflow::core::OpResult::Ok;
    const QRect selected_roi = dashboard_->selected_roi();
    QRect       roi          = selected_roi;
    if (roi.isEmpty())
      roi = QRect(0, 0, width, height);
    roi = roi.intersected(QRect(0, 0, width, height));
    if (roi.isEmpty())
      return holoflow::core::OpResult::Ok;

    const size_t row_bytes     = static_cast<size_t>(roi.width()) * sizeof(float);
    const size_t roi_pixels    = static_cast<size_t>(roi.width()) * roi.height();
    const auto copy_roi_planes = [&](size_t input_index, std::vector<std::vector<float>> &planes) {
      auto       &tensor     = ctx.inputs[input_index];
      const auto &input_desc = tensor.desc;
      const auto *source     = reinterpret_cast<const float *>(tensor.data());
      for (size_t channel = 0; channel < planes.size(); ++channel) {
        planes[channel].resize(roi_pixels);
        const auto *src = source + static_cast<int64_t>(channel) * width * height +
                          static_cast<int64_t>(roi.y()) * width + roi.x();
        if (input_desc.mem_loc == holoflow::core::MemLoc::Device) {
          CUDA_CHECK(cudaMemcpy2DAsync(planes[channel].data(), row_bytes, src,
                                       static_cast<size_t>(width) * sizeof(float), row_bytes,
                                       roi.height(), cudaMemcpyDeviceToHost, stream_));
        } else {
          for (int row = 0; row < roi.height(); ++row) {
            std::memcpy(planes[channel].data() + static_cast<size_t>(row) * roi.width(),
                        src + static_cast<int64_t>(row) * width, row_bytes);
          }
        }
      }
    };
    auto &k2_planes        = k2_planes_;
    auto &parameter_planes = parameter_planes_;
    copy_roi_planes(0, k2_planes);
    if (roi_pixels == 1)
      copy_roi_planes(1, parameter_planes);
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    ui::MesiExposureSample curve_sample;
    curve_sample.exposure_frames   = settings_.model_settings.exposure_frames;
    curve_sample.exposure_fraction = settings_.model_settings.exposure_fraction;
    curve_sample.roi_pixels        = roi_pixels;
    curve_sample.single_pixel      = roi_pixels == 1;
    if (curve_sample.single_pixel) {
      curve_sample.pixel_beta        = parameter_planes[0][0];
      curve_sample.pixel_rho         = parameter_planes[1][0];
      curve_sample.pixel_tau_seconds = parameter_planes[2][0];
    }
    curve_sample.window_index =
        settings_.source_cycle_packets == 0 ? packet_index : packet_position;
    curve_sample.fixed_beta_rho =
        settings_.model_settings.estimator == holotask::syncs::MesiEstimator::FixedLut;
    std::vector<double> sums(k2_planes.size(), 0.0);
    for (size_t pixel = 0; pixel < roi_pixels; ++pixel) {
      bool valid = true;
      for (const auto &plane : k2_planes) {
        if (!std::isfinite(plane[pixel]) || plane[pixel] < 0.0F) {
          valid = false;
          break;
        }
      }
      if (!valid)
        continue;
      for (size_t channel = 0; channel < k2_planes.size(); ++channel)
        sums[channel] += k2_planes[channel][pixel];
      ++curve_sample.valid_pixels;
    }
    for (size_t channel = 0; channel < k2_planes.size(); ++channel) {
      curve_sample.exposure_seconds.push_back(
          static_cast<double>(curve_sample.exposure_frames[channel]) *
          settings_.model_settings.frame_period_seconds);
      const double single_pixel_k2 = k2_planes[channel][0];
      curve_sample.measured_k2.push_back(
          curve_sample.valid_pixels > 0 ? sums[channel] / curve_sample.valid_pixels
          : curve_sample.single_pixel && std::isfinite(single_pixel_k2)
              ? single_pixel_k2
              : std::numeric_limits<double>::quiet_NaN());
    }
    if (curve_sample.valid_pixels > 0) {
      // This diagnostic fits the ROI mean curve, independently of the per-pixel REMI maps.
      const RoiFit fit = fit_roi_curve(curve_sample.measured_k2, settings_.model_settings);
      if (std::isfinite(fit.squared_error) && fit.beta > 0.0) {
        curve_sample.fit_valid   = true;
        curve_sample.beta        = fit.beta;
        curve_sample.rho         = fit.rho;
        curve_sample.tau_seconds = fit.tau_frames * settings_.model_settings.frame_period_seconds;
        curve_sample.rmse        = std::sqrt(fit.squared_error / k2_planes.size());
        curve_sample.tau_at_bound =
            fit.tau_frames <= settings_.model_settings.tau_min_frames * 1.001 ||
            fit.tau_frames >= settings_.model_settings.tau_max_frames * 0.999;
        auto fitted_settings = settings_.model_settings;
        fitted_settings.beta = fit.beta;
        fitted_settings.rho  = fit.rho;
        for (int frames = 1; frames <= fitted_settings.exposure_frames.back(); ++frames) {
          curve_sample.model_exposure_seconds.push_back(frames *
                                                        fitted_settings.frame_period_seconds);
          curve_sample.model_k2.push_back(
              holotask::syncs::mesi_model_k2(fitted_settings, frames, fit.tau_frames));
        }
      }
    }

    QPointer<ui::MesiDashboardWidget> dashboard = dashboard_;
    if (!cached_packets_.empty())
      cached_packets_[packet_position] = true;
    const uint64_t posted_revision = cache_revision_;
    QMetaObject::invokeMethod(
        dashboard,
        [dashboard, selected_roi, posted_revision,
         curve_sample = std::move(curve_sample)]() mutable {
          if (!dashboard || dashboard->selected_roi() != selected_roi ||
              dashboard->curve_cache_revision() != posted_revision)
            return;
          dashboard->append_curve_sample(std::move(curve_sample));
        },
        Qt::QueuedConnection);
    return holoflow::core::OpResult::Ok;
  }

private:
  DisplayMesiHistorySettings            settings_;
  QPointer<ui::MesiDashboardWidget>     dashboard_;
  cudaStream_t                          stream_;
  uint64_t                              sample_index_ = 0;
  std::chrono::steady_clock::time_point next_map_refresh_{};
  std::vector<std::vector<float>>       k2_planes_;
  std::vector<std::vector<float>>       parameter_planes_;
  std::vector<bool>                     cached_packets_;
  uint64_t                              cache_revision_ = 0;
  uint64_t                              map_revision_   = 0;
};

} // namespace

void to_json(nlohmann::json &j, const DisplayMesiHistorySettings &settings) {
  j                         = {{"tau_min_seconds", settings.tau_min_seconds}};
  j["tau_max_seconds"]      = settings.tau_max_seconds;
  j["model_settings"]       = settings.model_settings;
  j["source_cycle_packets"] = settings.source_cycle_packets;
  j["curve_cache_key"]      = settings.curve_cache_key;
}

void from_json(const nlohmann::json &j, DisplayMesiHistorySettings &settings) {
  settings.tau_min_seconds      = j.value("tau_min_seconds", settings.tau_min_seconds);
  settings.tau_max_seconds      = j.value("tau_max_seconds", settings.tau_max_seconds);
  settings.model_settings       = j.at("model_settings").get<holotask::syncs::MesiRemiSettings>();
  settings.source_cycle_packets = j.value("source_cycle_packets", size_t{0});
  settings.curve_cache_key      = j.value("curve_cache_key", std::string{});
}

DisplayMesiHistoryFactory::DisplayMesiHistoryFactory(ui::MesiDashboardWidget *dashboard_widget)
    : dashboard_widget_(dashboard_widget) {}

holoflow::core::InferResult
DisplayMesiHistoryFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                 const nlohmann::json                  &jsettings) const {
  const auto settings = jsettings.get<DisplayMesiHistorySettings>();
  if (input_descs.size() != 3 || input_descs[0].rank() != 3 || input_descs[1].rank() != 3 ||
      input_descs[2].rank() != 3 ||
      input_descs[0].shape[0] != settings.model_settings.exposure_frames.size() ||
      input_descs[1].shape[0] != 3 || input_descs[2].shape[0] != 1 ||
      input_descs[0].shape[1] != input_descs[1].shape[1] ||
      input_descs[0].shape[2] != input_descs[1].shape[2] ||
      input_descs[0].shape[1] != input_descs[2].shape[1] ||
      input_descs[0].shape[2] != input_descs[2].shape[2] ||
      input_descs[0].dtype != holoflow::core::DType::F32 ||
      input_descs[1].dtype != holoflow::core::DType::F32 ||
      input_descs[2].dtype != holoflow::core::DType::F32 ||
      !std::isfinite(settings.tau_min_seconds) || settings.tau_min_seconds < 0.0 ||
      !std::isfinite(settings.tau_max_seconds) ||
      settings.tau_max_seconds <= settings.tau_min_seconds) {
    throw std::invalid_argument("DisplayMesiHistory expects K², parameter, and display F32 maps");
  }
  return {.input_descs   = {input_descs[0], input_descs[1], input_descs[2]},
          .output_descs  = {},
          .in_place      = {},
          .owned_inputs  = {false, false, false},
          .owned_outputs = {},
          .kind          = holoflow::core::TaskKind::Sync};
}

std::unique_ptr<holoflow::core::ISyncTask>
DisplayMesiHistoryFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                                  const nlohmann::json                  &jsettings,
                                  const holoflow::core::SyncCreateCtx   &ctx) const {
  infer(input_descs, jsettings);
  const auto settings = jsettings.get<DisplayMesiHistorySettings>();
  if (dashboard_widget_.isNull()) {
    throw std::invalid_argument("MESI dashboard is unavailable");
  }
  QPointer<ui::MesiDashboardWidget> dashboard = dashboard_widget_;
  const int                         width     = static_cast<int>(input_descs[0].shape[2]);
  const int                         height    = static_cast<int>(input_descs[0].shape[1]);
  const float                       tau_min   = static_cast<float>(settings.tau_min_seconds);
  const float                       tau_max   = static_cast<float>(settings.tau_max_seconds);
  std::vector<uint64_t>             cached_indices;
  uint64_t                          cache_revision = 0;
  uint64_t                          map_revision   = 0;
  auto update_dashboard = [dashboard, width, height, tau_min, tau_max, &settings, &cached_indices,
                           &cache_revision, &map_revision] {
    if (dashboard) {
      map_revision = dashboard->begin_map_run();
      dashboard->set_map_size(width, height);
      dashboard->set_exposure_levels(settings.model_settings.exposure_frames);
      dashboard->set_tau_range(tau_min, tau_max);
      cached_indices =
          dashboard->prepare_curve_cache(settings.curve_cache_key, settings.source_cycle_packets);
      cache_revision = dashboard->curve_cache_revision();
    }
  };
  if (QThread::currentThread() == dashboard->thread()) {
    update_dashboard();
  } else {
    QMetaObject::invokeMethod(dashboard, std::move(update_dashboard), Qt::BlockingQueuedConnection);
  }
  return std::make_unique<DisplayMesiHistoryTask>(settings, dashboard_widget_, ctx.stream,
                                                  std::move(cached_indices), cache_revision,
                                                  map_revision);
}

std::unique_ptr<holoflow::core::ISyncTask> DisplayMesiHistoryFactory::update(
    std::unique_ptr<holoflow::core::ISyncTask>, std::span<const holoflow::core::TDesc> input_descs,
    const nlohmann::json &jsettings, const holoflow::core::SyncCreateCtx &ctx) const {
  return create(input_descs, jsettings, ctx);
}

} // namespace holovibes::tasks::sinks
