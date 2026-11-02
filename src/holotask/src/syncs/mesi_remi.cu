// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "holotask/syncs/mesi_remi.hh"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curaii/cuda.hh"
#include "logger.hh"

namespace holotask::syncs {

// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const MesiRemiSettings &s) {
  j = nlohmann::json{{"exposure_frames", s.exposure_frames},
                     {"frame_period_seconds", s.frame_period_seconds},
                     {"exposure_fraction", s.exposure_fraction},
                     {"beta", s.beta},
                     {"rho", s.rho},
                     {"tau_min_frames", s.tau_min_frames},
                     {"tau_max_frames", s.tau_max_frames},
                     {"tau_samples", s.tau_samples},
                     {"estimator", static_cast<int>(s.estimator)},
                     {"beta_smoothing_size", s.beta_smoothing_size}};
}

void from_json(const nlohmann::json &j, MesiRemiSettings &s) {
  s.exposure_frames      = j.value("exposure_frames", s.exposure_frames);
  s.frame_period_seconds = j.value("frame_period_seconds", s.frame_period_seconds);
  s.exposure_fraction    = j.value("exposure_fraction", s.exposure_fraction);
  s.beta                 = j.value("beta", s.beta);
  s.rho                  = j.value("rho", s.rho);
  s.tau_min_frames       = j.value("tau_min_frames", s.tau_min_frames);
  s.tau_max_frames       = j.value("tau_max_frames", s.tau_max_frames);
  s.tau_samples          = j.value("tau_samples", s.tau_samples);
  s.estimator = static_cast<MesiEstimator>(j.value("estimator", static_cast<int>(s.estimator)));
  s.beta_smoothing_size = j.value("beta_smoothing_size", s.beta_smoothing_size);
}

namespace {

template <typename T> using DevPtr = curaii::unique_device_ptr<T>;

void check(bool condition, const std::string &message) {
  if (!condition) {
    throw std::invalid_argument("MesiRemi: " + message);
  }
}

bool is_c_contiguous(const holoflow::core::TDesc &desc) {
  if (desc.shape.size() != desc.strides.size()) {
    return false;
  }

  size_t expected = holoflow::core::size_of(desc.dtype);
  for (size_t i = desc.shape.size(); i-- > 0;) {
    if (desc.strides[i] != expected ||
        (desc.shape[i] != 0 && expected > std::numeric_limits<size_t>::max() / desc.shape[i])) {
      return false;
    }
    expected *= desc.shape[i];
  }
  return true;
}

double exposure_average(int frames, double frame_period, double gate_duration, double tau,
                        double decay_multiplier) {
  const double rate      = decay_multiplier / tau;
  const double gate_z    = rate * gate_duration;
  double       same_gate = 1.0;
  if (std::abs(gate_z) >= 1.0e-3) {
    same_gate = 2.0 * (std::expm1(-gate_z) + gate_z) / (gate_z * gate_z);
  } else {
    const double z2 = gate_z * gate_z;
    same_gate       = 1.0 - gate_z / 3.0 + z2 / 12.0 - z2 * gate_z / 60.0 + z2 * z2 / 360.0;
  }

  if (frames == 1) {
    return same_gate;
  }

  const double gate_average =
      std::abs(gate_z) >= 1.0e-3
          ? -std::expm1(-gate_z) / gate_z
          : 1.0 - gate_z / 2.0 + gate_z * gate_z / 6.0 - gate_z * gate_z * gate_z / 24.0;
  double integrated = static_cast<double>(frames) * same_gate;
  for (int lag = 1; lag < frames; ++lag) {
    const double start_gap = std::max(static_cast<double>(lag) * frame_period - gate_duration, 0.0);
    const double covariance = gate_average * gate_average * std::exp(-rate * start_gap);
    integrated += 2.0 * static_cast<double>(frames - lag) * covariance;
  }
  return integrated / (static_cast<double>(frames) * frames);
}

double model_k2(const MesiRemiSettings &settings, int exposure_frames, double tau_frames) {
  return mesi_model_k2(settings, exposure_frames, tau_frames);
}

void validate_settings(const MesiRemiSettings &s, size_t input_channels) {
  check(s.estimator == MesiEstimator::FixedLut || s.estimator == MesiEstimator::Remi ||
            s.estimator == MesiEstimator::SpatialRemi,
        "unknown estimator");
  check(s.exposure_frames.size() >= 3 && s.exposure_frames.size() <= 10,
        "exposure_frames must contain between 3 and 10 dyadic groups");
  check(s.exposure_frames.size() == input_channels,
        "exposure_frames count must match the input channel count");
  int previous = 0;
  for (const int frames : s.exposure_frames) {
    check(frames > 0 && (frames & (frames - 1)) == 0,
          "exposure frame counts must be positive powers of two");
    check(frames == (previous == 0 ? 1 : previous * 2),
          "exposure frame counts must start at 1 and double each channel");
    previous = frames;
  }
  check(std::isfinite(s.frame_period_seconds) && s.frame_period_seconds > 0.0,
        "frame_period_seconds must be positive and finite");
  check(std::isfinite(s.exposure_fraction) && s.exposure_fraction > 0.0 &&
            s.exposure_fraction <= 1.0,
        "exposure_fraction must be in (0, 1]");
  if (s.estimator == MesiEstimator::FixedLut) {
    check(std::isfinite(s.beta) && s.beta > 0.0 && s.beta <= 1.0, "beta must be in (0, 1]");
    check(std::isfinite(s.rho) && s.rho > 0.0 && s.rho <= 1.0, "rho must be in (0, 1]");
  } else {
    check(input_channels >= 5, "REMI requires at least five exposure channels (16 frames)");
    check(s.exposure_fraction == 1.0,
          "REMI coefficients require continuous exposures (exposure_fraction = 1)");
  }
  if (s.estimator == MesiEstimator::SpatialRemi) {
    check(s.beta_smoothing_size == 0 ||
              (s.beta_smoothing_size >= 1 && s.beta_smoothing_size <= 2047 &&
               (s.beta_smoothing_size & 1) == 1),
          "beta_smoothing_size must be zero (automatic) or odd and between 1 and 2047");
  }
  check(std::isfinite(s.tau_min_frames) && s.tau_min_frames > 0.0 &&
            std::isfinite(s.tau_max_frames) && s.tau_max_frames > s.tau_min_frames,
        "tau bounds must be positive and strictly increasing");
  check(std::isfinite(s.tau_min_frames * s.frame_period_seconds) &&
            s.tau_min_frames * s.frame_period_seconds > 0.0 &&
            std::isfinite(s.tau_max_frames * s.frame_period_seconds),
        "tau bounds in seconds must be finite and representable");
  if (s.estimator == MesiEstimator::FixedLut) {
    check(s.tau_samples >= 5 && s.tau_samples <= 2048, "tau_samples must be between 5 and 2048");
  }

  if (s.estimator == MesiEstimator::FixedLut) {
    double max_sensitivity = 0.0;
    for (const int exposure : s.exposure_frames) {
      double low      = model_k2(s, exposure, s.tau_min_frames);
      double high     = model_k2(s, exposure, s.tau_max_frames);
      max_sensitivity = std::max(max_sensitivity, std::abs(high - low));
    }
    check(std::isfinite(max_sensitivity) && max_sensitivity > 1.0e-7,
          "calibration and tau bounds do not produce a measurable tau dependence");
  }
}

__device__ float lut_fit_error(const float *__restrict__ observations,
                               const float *__restrict__ model_table, int channel_count,
                               int64_t pixel_count, int64_t pixel, int candidate) {
  float error = 0.0F;
  for (int channel = 0; channel < channel_count; ++channel) {
    const float observed = observations[static_cast<int64_t>(channel) * pixel_count + pixel];
    const float expected = model_table[static_cast<int64_t>(candidate) * channel_count + channel];
    const float delta    = observed - expected;
    error += delta * delta;
  }
  return error;
}

__global__ void fit_tau_kernel(const float *__restrict__ observations,
                               const float *__restrict__ model_table, float *__restrict__ output,
                               int channel_count, int tau_samples, int64_t pixel_count,
                               double tau_min_frames, double tau_max_frames,
                               double frame_period_seconds) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixel_count) {
    return;
  }

  bool  valid   = true;
  float minimum = FLT_MAX;
  float maximum = 0.0F;
  for (int channel = 0; channel < channel_count; ++channel) {
    const float value = observations[static_cast<int64_t>(channel) * pixel_count + pixel];
    if (!isfinite(value) || value < 0.0F) {
      valid = false;
      break;
    }
    minimum = fminf(minimum, value);
    maximum = fmaxf(maximum, value);
  }
  if (!valid || maximum - minimum <= 1.0e-10F) {
    output[pixel] = 0.0F;
    return;
  }

  float best_error = FLT_MAX;
  int   best_index = -1;
  for (int candidate = 0; candidate < tau_samples; ++candidate) {
    const float error =
        lut_fit_error(observations, model_table, channel_count, pixel_count, pixel, candidate);
    if (error < best_error) {
      best_error = error;
      best_index = candidate;
    }
  }

  // Zero marks pixels whose data do not identify tau within the configured search interval.
  if (best_index <= 0 || best_index >= tau_samples - 1) {
    output[pixel] = 0.0F;
    return;
  }
  const float left_error =
      lut_fit_error(observations, model_table, channel_count, pixel_count, pixel, best_index - 1);
  const float right_error =
      lut_fit_error(observations, model_table, channel_count, pixel_count, pixel, best_index + 1);
  const float curvature         = left_error - 2.0F * best_error + right_error;
  float       sub_sample_offset = 0.0F;
  if (isfinite(curvature) && curvature > 0.0F) {
    sub_sample_offset = fminf(fmaxf(0.5F * (left_error - right_error) / curvature, -1.0F), 1.0F);
  }
  const double fraction =
      (static_cast<double>(best_index) + sub_sample_offset) / static_cast<double>(tau_samples - 1);
  const double tau_frames =
      exp(log(tau_min_frames) + (log(tau_max_frames) - log(tau_min_frames)) * fraction);
  output[pixel] = static_cast<float>(tau_frames * frame_period_seconds);
}

// The empirical coefficients and formulae below follow sn-lab/Speckle-REMI's
// fullMESIestimates.m and fullMESIestimates_spatial.m (MESIeqDmuFits.mat).
// Their published fit assumes contiguous exposures. The reference calls its input K,
// but its K0=beta and Kcrit equations use the squared-contrast values supplied here.
__device__ __constant__ double remi_pf0[11] = {
    -0.020475343735539206, -0.000249703516761583, 0.19313859057944238,   -0.36047918793935946,
    0.28297723593911367,   -0.15969192842790683,  0.008759628227215873,  -0.04280014577879131,
    0.7405740344362672,    -1.283491167540141,    1.3018503390755502e-07};
__device__ __constant__ double remi_pf1[11] = {
    0.05329785249108524, -0.2678137035936505, 0.5297325548934846,     -0.5226901782034629,
    0.29932083427794304, -0.1306824240412107, -0.0029070896252958033, -0.03972798704852216,
    0.4639364195355631,  -0.7649256681428883, 5.171982469352355e-08};

struct RemiCurve {
  double k[10]{};
  double kf[8]{};
  double derivative[8]{};
  double t_peak     = 0.0;
  double peak_slope = 0.0;
  double b_k0       = 0.0;
  double b_k1       = 0.0;
  double k_p0       = 0.0;
  double k_p1       = 0.0;
};

__device__ double remi_polynomial(double rho, bool dynamic) {
  const double *coefficients = dynamic ? remi_pf1 : remi_pf0;
  double        result       = coefficients[0];
  for (int i = 1; i < 11; ++i) {
    result = result * rho + coefficients[i];
  }
  return result;
}

__device__ bool remi_prepare(const float *observations, int channels, int64_t pixels, int64_t pixel,
                             RemiCurve &curve) {
  constexpr double log_step = 0.30102999566398119521; // log10(2)
  double           smoothed[10];
  for (int i = 0; i < channels; ++i) {
    const double k2 = observations[static_cast<int64_t>(i) * pixels + pixel];
    if (!isfinite(k2) || k2 < 0.0) {
      return false;
    }
    curve.k[i] = k2;
  }
  if (curve.k[0] - curve.k[channels - 1] <= 1.0e-6) {
    return false;
  }
  for (int i = 0; i < channels; ++i) {
    const int left  = i > 0 ? i - 1 : i;
    const int right = i + 1 < channels ? i + 1 : i;
    double    sum   = 0.0;
    for (int j = left; j <= right; ++j) {
      sum += curve.k[j];
    }
    smoothed[i] = 0.5 * (curve.k[i] + sum / (right - left + 1));
  }
  const int derivatives = channels - 2;
  for (int i = 0; i < derivatives; ++i) {
    const double raw    = (smoothed[i + 2] - smoothed[i]) / (2.0 * log_step);
    curve.kf[i]         = smoothed[i] + log_step * raw;
    curve.derivative[i] = raw;
  }
  double filtered[8];
  int    minimum = 0;
  for (int i = 0; i < derivatives; ++i) {
    const int left  = i > 0 ? i - 1 : i;
    const int right = i + 1 < derivatives ? i + 1 : i;
    double    sum   = 0.0;
    for (int j = left; j <= right; ++j) {
      sum += curve.derivative[j];
    }
    filtered[i] = fmin(sum / (right - left + 1), 0.0);
    if (filtered[i] < filtered[minimum]) {
      minimum = i;
    }
  }
  minimum         = max(1, min(minimum, derivatives - 2));
  const double x0 = minimum * log_step;
  const double x1 = (minimum + 1) * log_step;
  const double x2 = (minimum + 2) * log_step;
  const double b1 = filtered[minimum - 1];
  const double b2 = (filtered[minimum] - b1) / log_step;
  const double b3 =
      ((filtered[minimum + 1] - filtered[minimum]) / log_step - b2) / (2.0 * log_step);
  if (!isfinite(b3) || fabs(b3) < 1.0e-12) {
    return false;
  }
  const double log_peak    = fmin(fmax((b3 * (x0 + x1) - b2) / (2.0 * b3), x0 - 1.0), x2 + 1.0);
  curve.t_peak             = pow(10.0, log_peak);
  curve.peak_slope         = b1 + b2 * (log_peak - x0) + b3 * (log_peak - x0) * (log_peak - x1);
  const double     first_x = 2.0 / curve.t_peak;
  const double     last_x  = pow(2.0, channels - 2) / curve.t_peak;
  const double     x0_root = sqrt(first_x);
  const double     x0_pow  = first_x * x0_root;
  const double     x1_root = sqrt(last_x);
  const double     x1_pow  = last_x * x1_root;
  constexpr double ln10    = 2.30258509299404568402;
  curve.b_k0               = -sqrt(1.0 - 0.045106 * x0_root + 1.8508 * first_x - 1.1288 * x0_pow +
                                   1.3459 * first_x * first_x) /
                             ln10;
  curve.b_k1               = -2.0 *
                             sqrt(1.0 + 1.03401 * x0_root + 0.997606 * first_x + 0.08485 * x0_pow +
                                  0.0942633 * first_x * first_x) /
                             ln10;
  curve.k_p0               = sqrt(1.0 - 0.00128996 * x1_root + 1.16603 * last_x - 0.17042 * x1_pow +
                                  pow(0.9182 * last_x, 2.0)) /
                             (0.9182 * last_x * ln10);
  curve.k_p1 = sqrt(sqrt(1.0 + 0.98233 * x1_root + 0.754227 * last_x + 0.04413 * x1_pow +
                         pow(0.23324 * last_x, 2.0)) /
                    (0.23324 * last_x)) /
               ln10;
  for (int i = 0; i < derivatives; ++i) {
    curve.derivative[i] = filtered[i];
  }
  return isfinite(curve.t_peak) && isfinite(curve.peak_slope) && isfinite(curve.b_k0) &&
         isfinite(curve.b_k1) && isfinite(curve.k_p0) && isfinite(curve.k_p1);
}

__device__ bool remi_initial(const RemiCurve &curve, int channels, bool spatial, double &beta,
                             double &dm) {
  const int    last  = channels - 3;
  const double beta0 = fmin(fmax(curve.kf[0] + curve.b_k0 * curve.derivative[0], 0.0), 1.0);
  if (beta0 <= 0.0) {
    return false;
  }
  const double kinf0 = fmin(fmax(curve.kf[last] + curve.k_p0 * curve.derivative[last], 0.0), 1.0);
  const double rho0  = fmin(fmax(1.0 - sqrt(kinf0 / beta0), 0.0), 1.0);
  const double d0    = beta0 * remi_polynomial(rho0, false);
  const double d1    = beta0 * remi_polynomial(rho0, true);
  if (!isfinite(d1 - d0) || fabs(d1 - d0) < 1.0e-12) {
    return false;
  }
  dm = fmin(fmax(((spatial ? curve.peak_slope * 1.05 : curve.peak_slope) - d0) / (d1 - d0), 0.0),
            1.0);
  if (spatial) {
    dm = round(dm);
  } else {
    const double centered = dm - 0.5;
    dm                    = 0.5 + copysign(sqrt(fabs(centered) / 2.0), centered);
  }
  const double b_k = 1.0 / (dm / curve.b_k1 + (1.0 - dm) / curve.b_k0);
  beta             = fmin(fmax(curve.kf[0] + b_k * curve.derivative[0], 0.0), 1.0);
  return isfinite(beta) && beta > 0.0;
}

__device__ double remi_finish(const RemiCurve &curve, int channels, double beta, double dm,
                              bool spatial, double tau_min, double tau_max, double frame_period,
                              double *rho_out) {
  const int last = channels - 3;
  if (spatial) {
    const double kinf0 = fmin(fmax(curve.kf[last] + curve.k_p0 * curve.derivative[last], 0.0), 1.0);
    const double rho0  = fmin(fmax(1.0 - sqrt(kinf0 / beta), 0.0), 1.0);
    const double d0    = beta * remi_polynomial(rho0, false);
    const double d1    = beta * remi_polynomial(rho0, true);
    if (fabs(d1 - d0) < 1.0e-12) {
      return 0.0;
    }
    dm = fmin(fmax((curve.peak_slope - d0) / (d1 - d0), 0.0), 1.0);
  }
  const double k_p  = 1.0 / (dm / curve.k_p1 + (1.0 - dm) / curve.k_p0);
  const double kinf = fmin(fmax(curve.kf[last] + k_p * curve.derivative[last], 0.0), 1.0);
  const double rho  = fmin(fmax(1.0 - sqrt(kinf / beta), 0.0), 1.0);
  *rho_out          = rho;
  // Use the paper's beta/sqrt(2) Kcrit cap. The MATLAB repository uses 0.8*beta instead.
  const double kcrit =
      fmin(beta * (rho * rho * (0.5 - dm + (6.0 * dm + 0.5) / exp(2.0)) +
                   rho * (1.0 - rho) * ((108.0 * dm + 4.0) / exp(1.0) - 40.0 * dm) +
                   (1.0 - rho) * (1.0 - rho)),
           beta / sqrt(2.0));
  if (!isfinite(kcrit) || kcrit <= 0.0) {
    return 0.0;
  }
  // Eq. 24 uses the last exposure at or above Kcrit, even for a nonmonotonic curve.
  // Use the edge pair to extrapolate when Kcrit lies outside the measured exposures.
  constexpr double log_step   = 0.30102999566398119521;
  int              left_index = -1;
  for (int i = 0; i < channels; ++i) {
    if (curve.k[i] >= kcrit) {
      left_index = i;
    }
  }
  if (left_index < 0) {
    left_index = 0;
  } else if (left_index == channels - 1) {
    left_index = channels - 2;
  }
  const double left       = curve.k[left_index];
  const double right      = curve.k[left_index + 1];
  const double difference = right - left;
  if (!isfinite(difference) || fabs(difference) <= 1.0e-12) {
    return 0.0;
  }
  const double fraction = (kcrit - left) / difference;
  if (!isfinite(fraction)) {
    return 0.0;
  }
  const double tau_frames = pow(10.0, (left_index + fraction) * log_step);
  if (!isfinite(tau_frames))
    return tau_max * frame_period;
  // The reference clamps extrapolated estimates to its reporting interval.
  return fmin(fmax(tau_frames, tau_min), tau_max) * frame_period;
}

__global__ void remi_kernel(const float *observations, const float *smoothed_beta, float *output,
                            int channels, int64_t pixels, double tau_min, double tau_max,
                            double frame_period, bool spatial, bool preliminary) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixels) {
    return;
  }
  RemiCurve curve;
  double    beta = 0.0;
  double    dm   = 0.0;
  if (!remi_prepare(observations, channels, pixels, pixel, curve) ||
      !remi_initial(curve, channels, spatial, beta, dm)) {
    if (preliminary) {
      output[pixel] = 0.0F;
    } else {
      output[pixel]              = 0.0F;
      output[pixels + pixel]     = 0.0F;
      output[2 * pixels + pixel] = 0.0F;
    }
    return;
  }
  if (preliminary) {
    output[pixel] = static_cast<float>(beta);
    return;
  }
  if (spatial) {
    beta = smoothed_beta[pixel];
  }
  if (beta <= 0.0 || !isfinite(beta)) {
    output[pixel] = output[pixels + pixel] = output[2 * pixels + pixel] = 0.0F;
    return;
  }
  double       rho = 0.0;
  const double tau =
      remi_finish(curve, channels, beta, dm, spatial, tau_min, tau_max, frame_period, &rho);
  output[pixel]              = static_cast<float>(beta);
  output[pixels + pixel]     = static_cast<float>(rho);
  output[2 * pixels + pixel] = static_cast<float>(tau);
}

__global__ void fixed_coefficients_kernel(float *output, int64_t pixels, float beta, float rho) {
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= pixels) {
    return;
  }
  output[pixel]          = beta;
  output[pixels + pixel] = rho;
}

__global__ void smooth_beta_kernel(const float *input, float *output, int width, int height,
                                   int filter_size, bool horizontal) {
  const int pixel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= width * height) {
    return;
  }
  const int    x        = pixel % width;
  const int    y        = pixel / width;
  const int    radius   = filter_size / 2;
  const double sigma    = filter_size / 2.0;
  double       weighted = 0.0;
  double       weights  = 0.0;
  for (int step = -radius; step <= radius; ++step) {
    const int   nx    = horizontal ? max(0, min(width - 1, x + step)) : x;
    const int   ny    = horizontal ? y : max(0, min(height - 1, y + step));
    const float value = input[ny * width + nx];
    if (value > 0.0F && isfinite(value)) {
      const double weight = exp(-static_cast<double>(step * step) / (2.0 * sigma * sigma));
      weighted += weight * value;
      weights += weight;
    }
  }
  output[pixel] = weights > 0.0 ? static_cast<float>(weighted / weights) : 0.0F;
}

class MesiRemi final : public holoflow::core::ISyncTask {
public:
  MesiRemi(MesiRemiSettings settings, holoflow::core::TDesc input_desc, int width, int height,
           int64_t pixel_count, int beta_smoothing_size, DevPtr<float> &&model_table,
           DevPtr<float> &&beta, DevPtr<float> &&beta_horizontal, DevPtr<float> &&beta_smoothed,
           cudaStream_t stream)
      : settings_(std::move(settings)), input_desc_(std::move(input_desc)), width_(width),
        height_(height), pixel_count_(pixel_count), beta_smoothing_size_(beta_smoothing_size),
        model_table_(std::move(model_table)), beta_(std::move(beta)),
        beta_horizontal_(std::move(beta_horizontal)), beta_smoothed_(std::move(beta_smoothed)),
        stream_(stream) {}

  bool supports_cuda_graph() const noexcept override { return true; }

  void record_cuda_graph(holoflow::core::CudaGraphCtx &ctx) override {
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx execution{ctx.inputs, ctx.outputs, &cancelled, nullptr, nullptr};
    (void)execute(execution);
  }

  holoflow::core::OpResult execute(holoflow::core::SyncCtx &ctx) override {
    constexpr int block_size = 256;
    const auto  grid_size = static_cast<unsigned int>((pixel_count_ + block_size - 1) / block_size);
    const auto *observations     = reinterpret_cast<const float *>(ctx.inputs[0].data());
    auto       *tau_output       = reinterpret_cast<float *>(ctx.outputs[0].data());
    auto       *parameter_output = reinterpret_cast<float *>(ctx.outputs[1].data());
    // The image and history sinks consume separate outputs, so both need the fitted tau.
    auto        *parameter_tau = parameter_output + 2 * pixel_count_;
    const size_t tau_bytes     = static_cast<size_t>(pixel_count_) * sizeof(float);
    if (settings_.estimator == MesiEstimator::FixedLut) {
      fixed_coefficients_kernel<<<grid_size, block_size, 0, stream_>>>(
          parameter_output, pixel_count_, static_cast<float>(settings_.beta),
          static_cast<float>(settings_.rho));
      fit_tau_kernel<<<grid_size, block_size, 0, stream_>>>(
          observations, model_table_.get(), tau_output,
          static_cast<int>(settings_.exposure_frames.size()), settings_.tau_samples, pixel_count_,
          settings_.tau_min_frames, settings_.tau_max_frames, settings_.frame_period_seconds);
      CUDA_CHECK(
          cudaMemcpyAsync(parameter_tau, tau_output, tau_bytes, cudaMemcpyDeviceToDevice, stream_));
    } else {
      const bool spatial = settings_.estimator == MesiEstimator::SpatialRemi;
      if (spatial) {
        remi_kernel<<<grid_size, block_size, 0, stream_>>>(
            observations, nullptr, beta_.get(), static_cast<int>(settings_.exposure_frames.size()),
            pixel_count_, settings_.tau_min_frames, settings_.tau_max_frames,
            settings_.frame_period_seconds, true, true);
        if (beta_smoothing_size_ > 1) {
          smooth_beta_kernel<<<grid_size, block_size, 0, stream_>>>(
              beta_.get(), beta_horizontal_.get(), width_, height_, beta_smoothing_size_, true);
          smooth_beta_kernel<<<grid_size, block_size, 0, stream_>>>(
              beta_horizontal_.get(), beta_smoothed_.get(), width_, height_, beta_smoothing_size_,
              false);
        }
      }
      remi_kernel<<<grid_size, block_size, 0, stream_>>>(
          observations,
          spatial ? (beta_smoothing_size_ > 1 ? beta_smoothed_.get() : beta_.get()) : nullptr,
          parameter_output, static_cast<int>(settings_.exposure_frames.size()), pixel_count_,
          settings_.tau_min_frames, settings_.tau_max_frames, settings_.frame_period_seconds,
          spatial, false);
      CUDA_CHECK(
          cudaMemcpyAsync(tau_output, parameter_tau, tau_bytes, cudaMemcpyDeviceToDevice, stream_));
    }
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

  const MesiRemiSettings      &settings() const { return settings_; }
  const holoflow::core::TDesc &input_desc() const { return input_desc_; }
  void                         update_stream(cudaStream_t stream) { stream_ = stream; }

private:
  MesiRemiSettings      settings_;
  holoflow::core::TDesc input_desc_;
  int                   width_;
  int                   height_;
  int64_t               pixel_count_;
  int                   beta_smoothing_size_;
  DevPtr<float>         model_table_;
  DevPtr<float>         beta_;
  DevPtr<float>         beta_horizontal_;
  DevPtr<float>         beta_smoothed_;
  cudaStream_t          stream_;
};

} // namespace

MesiModelTerms mesi_model_terms(const MesiRemiSettings &settings, int exposure_frames,
                                double tau_frames) {
  const double frame_period = settings.frame_period_seconds;
  const double tau          = tau_frames * frame_period;
  const double gate         = settings.exposure_fraction * frame_period;
  return {.single_decay = exposure_average(exposure_frames, frame_period, gate, tau, 1.0),
          .double_decay = exposure_average(exposure_frames, frame_period, gate, tau, 2.0)};
}

double mesi_model_k2(const MesiRemiSettings &settings, int exposure_frames, double tau_frames) {
  const auto   terms = mesi_model_terms(settings, exposure_frames, tau_frames);
  const double rho   = settings.rho;
  return settings.beta * (rho * rho * terms.double_decay +
                          2.0 * rho * (1.0 - rho) * terms.single_decay + (1.0 - rho) * (1.0 - rho));
}

// -------------------------------------------------------------------------------------------------
// MesiRemiFactory
// -------------------------------------------------------------------------------------------------

holoflow::core::InferResult
MesiRemiFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                       const nlohmann::json                  &jsettings) const {
  check(input_descs.size() == 1, "expected exactly one input");
  const auto  settings = jsettings.get<MesiRemiSettings>();
  const auto &input    = input_descs[0];
  check(input.rank() == 3, "input must have shape [exposure_channels, height, width]");
  check(input.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(input.dtype == holoflow::core::DType::F32, "input must have dtype F32");
  check(is_c_contiguous(input), "input must be C-contiguous");
  check(input.shape[0] > 0 && input.shape[1] > 0 && input.shape[2] > 0,
        "input dimensions must be positive");
  check(input.shape[0] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            input.shape[1] <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
            input.shape[2] <= static_cast<size_t>(std::numeric_limits<int>::max()),
        "input dimensions must fit in int");
  const auto pixels = input.shape[1] * input.shape[2];
  check(input.shape[1] == 0 || pixels / input.shape[1] == input.shape[2],
        "image dimensions overflow");
  check(pixels <= static_cast<size_t>(std::numeric_limits<int64_t>::max()),
        "pixel count must fit in int64");
  validate_settings(settings, input.shape[0]);

  auto tau_shape       = input.shape;
  tau_shape[0]         = 1;
  auto parameter_shape = input.shape;
  parameter_shape[0]   = 3; // beta, rho, tau maps in that order.
  auto tau_output = holoflow::core::TDesc(tau_shape, holoflow::core::DType::F32, input.mem_loc);
  auto parameter_output =
      holoflow::core::TDesc(parameter_shape, holoflow::core::DType::F32, input.mem_loc);
  return holoflow::core::InferResult{
      .input_descs   = {input},
      .output_descs  = {tau_output, parameter_output},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false, false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
MesiRemiFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                        const nlohmann::json                  &jsettings,
                        const holoflow::core::SyncCreateCtx   &ctx) const {
  const auto  result              = infer(input_descs, jsettings);
  const auto  settings            = jsettings.get<MesiRemiSettings>();
  const auto &input               = input_descs[0];
  const auto  height              = static_cast<int>(input.shape[1]);
  const auto  width               = static_cast<int>(input.shape[2]);
  const auto  pixel_count         = static_cast<int64_t>(input.shape[1] * input.shape[2]);
  int         beta_smoothing_size = settings.beta_smoothing_size;
  if (settings.estimator == MesiEstimator::SpatialRemi && beta_smoothing_size == 0) {
    // fullMESIestimates_spatial.m uses ceil(sqrt(numel(K0)) / 8) by default.
    // Keep an odd width for the centered separable Gaussian used here.
    beta_smoothing_size =
        static_cast<int>(std::ceil(std::sqrt(static_cast<double>(pixel_count)) / 8.0));
    beta_smoothing_size = std::max(beta_smoothing_size, 1);
    if ((beta_smoothing_size & 1) == 0) {
      ++beta_smoothing_size;
    }
  }
  DevPtr<float> table;
  DevPtr<float> beta;
  DevPtr<float> beta_horizontal;
  DevPtr<float> beta_smoothed;
  if (settings.estimator == MesiEstimator::FixedLut) {
    const size_t table_size =
        static_cast<size_t>(settings.tau_samples) * settings.exposure_frames.size();
    table = curaii::make_unique_device_ptr<float>(table_size);
    std::vector<float> host_table(table_size);
    for (int candidate = 0; candidate < settings.tau_samples; ++candidate) {
      const double fraction =
          static_cast<double>(candidate) / static_cast<double>(settings.tau_samples - 1);
      const double tau_frames = std::exp(
          std::log(settings.tau_min_frames) +
          (std::log(settings.tau_max_frames) - std::log(settings.tau_min_frames)) * fraction);
      for (size_t channel = 0; channel < settings.exposure_frames.size(); ++channel) {
        host_table[static_cast<size_t>(candidate) * settings.exposure_frames.size() + channel] =
            static_cast<float>(model_k2(settings, settings.exposure_frames[channel], tau_frames));
      }
    }
    CUDA_CHECK(cudaMemcpyAsync(table.get(), host_table.data(), table_size * sizeof(float),
                               cudaMemcpyHostToDevice, ctx.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream));
  } else if (settings.estimator == MesiEstimator::SpatialRemi) {
    beta = curaii::make_unique_device_ptr<float>(static_cast<size_t>(pixel_count));
    if (beta_smoothing_size > 1) {
      beta_horizontal = curaii::make_unique_device_ptr<float>(static_cast<size_t>(pixel_count));
      beta_smoothed   = curaii::make_unique_device_ptr<float>(static_cast<size_t>(pixel_count));
    }
  }
  (void)result;
  return std::make_unique<MesiRemi>(
      settings, input, width, height, pixel_count, beta_smoothing_size, std::move(table),
      std::move(beta), std::move(beta_horizontal), std::move(beta_smoothed), ctx.stream);
}

std::unique_ptr<holoflow::core::ISyncTask>
MesiRemiFactory::update(std::unique_ptr<holoflow::core::ISyncTask> old_task,
                        std::span<const holoflow::core::TDesc>     input_descs,
                        const nlohmann::json                      &jsettings,
                        const holoflow::core::SyncCreateCtx       &ctx) const {
  holoflow::core::ExecutionUpdateGuard update_guard(ctx.execution_invalidation);
  (void)infer(input_descs, jsettings);
  const auto settings = jsettings.get<MesiRemiSettings>();
  auto      *old_fit  = dynamic_cast<MesiRemi *>(old_task.get());
  if (old_fit != nullptr && settings == old_fit->settings() &&
      input_descs[0].shape == old_fit->input_desc().shape &&
      input_descs[0].strides == old_fit->input_desc().strides &&
      input_descs[0].dtype == old_fit->input_desc().dtype &&
      input_descs[0].mem_loc == old_fit->input_desc().mem_loc &&
      input_descs[0].offset == old_fit->input_desc().offset) {
    old_fit->update_stream(ctx.stream);
    return old_task;
  }
  ctx.invalidate_execution();
  return create(input_descs, jsettings, ctx);
}

} // namespace holotask::syncs
