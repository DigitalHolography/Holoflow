// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0.

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <numbers>
#include <span>
#include <vector>

#include <spdlog/spdlog.h>

#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow_event/router.hh"
#include "holotask/syncs/registration.hh"
#include "tensor_test_buffer.hh"

namespace {

using holoflow::core::DType;
using holoflow::core::MemLoc;
using holoflow::core::OpResult;
using holoflow::core::TDesc;

std::vector<std::byte> as_bytes(std::span<const float> values) {
  std::vector<std::byte> bytes(values.size_bytes());
  std::memcpy(bytes.data(), values.data(), values.size_bytes());
  return bytes;
}

std::vector<float> from_bytes(std::span<const std::byte> bytes) {
  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

std::vector<float> make_reference(size_t height, size_t width) {
  std::vector<float> image(height * width, 0.0f);
  for (size_t y = height / 4; y < 3 * height / 4; ++y) {
    for (size_t x = width / 4; x < 3 * width / 4; ++x) {
      const auto hash      = (x * 37 + y * 101 + x * y * 13) % 97;
      image[y * width + x] = static_cast<float>(hash) - 48.0f;
    }
  }
  return image;
}

int wrap(int value, int size) {
  value %= size;
  return value < 0 ? value + size : value;
}

std::vector<float> translate_bilinear(const std::vector<float> &input, size_t height, size_t width,
                                      float dx, float dy) {
  std::vector<float> output(input.size());
  for (int y = 0; y < static_cast<int>(height); ++y) {
    for (int x = 0; x < static_cast<int>(width); ++x) {
      const float src_x = static_cast<float>(x) - dx;
      const float src_y = static_cast<float>(y) - dy;
      const int   x0    = static_cast<int>(std::floor(src_x));
      const int   y0    = static_cast<int>(std::floor(src_y));
      const float fx    = src_x - static_cast<float>(x0);
      const float fy    = src_y - static_cast<float>(y0);
      const float i00 =
          input[wrap(y0, static_cast<int>(height)) * width + wrap(x0, static_cast<int>(width))];
      const float i10 =
          input[wrap(y0, static_cast<int>(height)) * width + wrap(x0 + 1, static_cast<int>(width))];
      const float i01 =
          input[wrap(y0 + 1, static_cast<int>(height)) * width + wrap(x0, static_cast<int>(width))];
      const float i11 = input[wrap(y0 + 1, static_cast<int>(height)) * width +
                              wrap(x0 + 1, static_cast<int>(width))];
      output[static_cast<size_t>(y) * width + static_cast<size_t>(x)] =
          (1.0f - fx) * (1.0f - fy) * i00 + fx * (1.0f - fy) * i10 + (1.0f - fx) * fy * i01 +
          fx * fy * i11;
    }
  }
  return output;
}

std::vector<float> rigid_transform_bilinear(const std::vector<float> &input, size_t height,
                                            size_t width, float angle_degrees, float dx, float dy) {
  std::vector<float> output(input.size());
  const float        radians = angle_degrees * static_cast<float>(std::numbers::pi / 180.0);
  const float        cosine  = std::cos(radians);
  const float        sine    = std::sin(radians);
  const float        cx      = 0.5f * static_cast<float>(width - 1);
  const float        cy      = 0.5f * static_cast<float>(height - 1);
  for (int y = 0; y < static_cast<int>(height); ++y) {
    for (int x = 0; x < static_cast<int>(width); ++x) {
      const float px    = static_cast<float>(x) - cx - dx;
      const float py    = static_cast<float>(y) - cy - dy;
      const float src_x = cx + cosine * px + sine * py;
      const float src_y = cy - sine * px + cosine * py;
      const int   x0    = static_cast<int>(std::floor(src_x));
      const int   y0    = static_cast<int>(std::floor(src_y));
      const float fx    = src_x - static_cast<float>(x0);
      const float fy    = src_y - static_cast<float>(y0);
      const float i00 =
          input[wrap(y0, static_cast<int>(height)) * width + wrap(x0, static_cast<int>(width))];
      const float i10 =
          input[wrap(y0, static_cast<int>(height)) * width + wrap(x0 + 1, static_cast<int>(width))];
      const float i01 =
          input[wrap(y0 + 1, static_cast<int>(height)) * width + wrap(x0, static_cast<int>(width))];
      const float i11 = input[wrap(y0 + 1, static_cast<int>(height)) * width +
                              wrap(x0 + 1, static_cast<int>(width))];
      output[static_cast<size_t>(y) * width + static_cast<size_t>(x)] =
          (1.0f - fx) * (1.0f - fy) * i00 + fx * (1.0f - fy) * i10 + (1.0f - fx) * fy * i01 +
          fx * fy * i11;
    }
  }
  return output;
}

double mse(std::span<const float> lhs, std::span<const float> rhs) {
  double error = 0.0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const double delta = static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]);
    error += delta * delta;
  }
  return error / static_cast<double>(lhs.size());
}

class RegistrationHarness {
public:
  RegistrationHarness(size_t height, size_t width)
      : desc_({1, height, width}, DType::F32, MemLoc::Device),
        valid_desc_({1}, DType::U8, MemLoc::Device), input_(desc_), output_(desc_),
        validity_(valid_desc_), handles_(router_.bind_node("registration")) {
    const holotask::syncs::RegistrationSettings settings{.radius = 1.0f};
    const std::array                            input_descs{desc_};
    task_ = factory_.create(input_descs, settings, {.stream = stream_.get()});
    task_->bind_logger(spdlog::default_logger());
  }

  void start_recording() {
    ASSERT_TRUE(router_.ui_try_send("registration", {{"type", "start_recording"}}));
    router_.tick();
  }

  void stop_recording() {
    ASSERT_TRUE(router_.ui_try_send("registration", {{"type", "stop_recording"}}));
    router_.tick();
  }

  std::vector<float> execute(const std::vector<float> &frame) {
    input_.upload(as_bytes(frame));
    std::array              inputs{input_.view()};
    std::array              outputs{output_.view(), validity_.view()};
    std::atomic<bool>       cancelled{false};
    holoflow::core::SyncCtx ctx{inputs, outputs, &cancelled, nullptr, &handles_.in};
    EXPECT_EQ(task_->execute(ctx), OpResult::Ok);
    CUDA_CHECK(cudaStreamSynchronize(stream_.get()));
    return from_bytes(output_.download());
  }

  bool valid() const {
    const auto bytes = validity_.download();
    return bytes[0] != std::byte{0};
  }

private:
  TDesc                                      desc_;
  TDesc                                      valid_desc_;
  curaii::CudaStream                         stream_;
  holonp_test::TensorTestBuffer              input_;
  holonp_test::TensorTestBuffer              output_;
  holonp_test::TensorTestBuffer              validity_;
  holoflow_event::Router                     router_;
  holoflow_event::Router::NodeHandles        handles_;
  holotask::syncs::RegistrationFactory       factory_;
  std::unique_ptr<holoflow::core::ISyncTask> task_;
};

TEST(RegistrationTest, AlignsKnownIntegerTranslationsOnNonSquareImages) {
  constexpr size_t height    = 80;
  constexpr size_t width     = 96;
  const auto       reference = make_reference(height, width);

  for (const auto [dx, dy] : std::array{std::pair{10.0f, 0.0f}, std::pair{-7.0f, 4.0f}}) {
    RegistrationHarness registration(height, width);
    registration.start_recording();
    EXPECT_EQ(registration.execute(reference), reference);
    EXPECT_LT(mse(registration.execute(reference), reference), 1e-6);
    EXPECT_TRUE(registration.valid());

    const auto shifted = translate_bilinear(reference, height, width, dx, dy);
    const auto aligned = registration.execute(shifted);
    EXPECT_TRUE(registration.valid());
    EXPECT_LT(mse(aligned, reference), 0.1);
  }
}

TEST(RegistrationTest, ImprovesKnownSubpixelTranslation) {
  constexpr size_t height    = 64;
  constexpr size_t width     = 80;
  const auto       reference = make_reference(height, width);
  const auto       shifted   = translate_bilinear(reference, height, width, 2.4f, -1.7f);

  RegistrationHarness registration(height, width);
  registration.start_recording();
  (void)registration.execute(reference);
  const auto aligned = registration.execute(shifted);

  EXPECT_TRUE(registration.valid());
  EXPECT_LT(mse(aligned, reference), 0.4 * mse(shifted, reference));
}

TEST(RegistrationTest, ImprovesCombinedTranslationAndRotation) {
  constexpr size_t height    = 96;
  constexpr size_t width     = 112;
  const auto       reference = make_reference(height, width);

  for (const float angle : std::array{1.0f, -1.0f, 3.0f, -3.0f}) {
    const auto moved = rigid_transform_bilinear(reference, height, width, angle, 3.0f, -2.0f);
    RegistrationHarness registration(height, width);
    registration.start_recording();
    (void)registration.execute(reference);
    const auto aligned = registration.execute(moved);

    EXPECT_TRUE(registration.valid()) << "angle=" << angle;
    EXPECT_LT(mse(aligned, reference), 0.35 * mse(moved, reference)) << "angle=" << angle;
  }
}

TEST(RegistrationTest, ResetsReferenceForEachRecording) {
  constexpr size_t height = 80;
  constexpr size_t width  = 96;
  const auto       first  = make_reference(height, width);
  auto             second = first;
  for (auto &value : second)
    value = -value;

  RegistrationHarness registration(height, width);
  registration.start_recording();
  (void)registration.execute(first);
  registration.stop_recording();
  registration.start_recording();

  EXPECT_EQ(registration.execute(second), second);
  const auto shifted = translate_bilinear(second, height, width, -7.0f, 4.0f);
  EXPECT_LT(mse(registration.execute(shifted), second), 0.1);
  EXPECT_TRUE(registration.valid());
}

TEST(RegistrationTest, RejectsUncorrelatedAndExcessiveMotion) {
  constexpr size_t    height    = 64;
  constexpr size_t    width     = 80;
  const auto          reference = make_reference(height, width);
  RegistrationHarness registration(height, width);
  registration.start_recording();
  (void)registration.execute(reference);

  std::vector<float> noise(reference.size());
  for (size_t i = 0; i < noise.size(); ++i)
    noise[i] = static_cast<float>((i * 7919 + 104729) % 1009) - 504.0f;
  (void)registration.execute(noise);
  EXPECT_FALSE(registration.valid());

  std::vector<float> blink(reference.size(), 0.0f);
  (void)registration.execute(blink);
  EXPECT_FALSE(registration.valid());

  auto non_finite                   = reference;
  non_finite[non_finite.size() / 2] = std::numeric_limits<float>::quiet_NaN();
  (void)registration.execute(non_finite);
  EXPECT_FALSE(registration.valid());

  (void)registration.execute(translate_bilinear(reference, height, width, 20.0f, 0.0f));
  EXPECT_FALSE(registration.valid());
}

} // namespace
