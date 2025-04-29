#include "holovibes/sinks/qt_display_sink.hh"

#include <QCoreApplication>
#include <cassert>
#include <cstdlib>
#include <nvtx3/nvToolsExt.h>
#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     QtDisplaySink Implementation
// ==========================================================================

QtDisplaySink::QtDisplaySink(const SinkMeta &meta, cudaStream_t stream,
                             curaii::cuda::unique_host_ptr<uint8_t> h_buffer,
                             TensorMeta h_meta)
    : Sink(meta, stream), frame_displayed_(true),
      last_display_time_(std::chrono::steady_clock::now()),
      h_buffer_(std::move(h_buffer)), host_meta_(h_meta) {}

void QtDisplaySink::run(TensorView itens) {
  holovibes_logger()->trace("running qt display sink");

  // 1) Limit fps
  auto now = std::chrono::steady_clock::now();
  if (now - last_display_time_ < std::chrono::milliseconds(1000 / 25)) {
    holovibes_logger()->trace("Skipping frame to maintain 30fps");
    return;
  }

  // 2) Skip until previous frame is displayed
  if (!frame_displayed_) {
    if (now - last_display_time_ < std::chrono::microseconds(100)) {
      holovibes_logger()->trace(
          "Skipping frame because prev was not displayed");
      return;
    } else {
      holovibes_logger()->warn(
          "Prev frame was not displayed in the timeout of 100ms");
    }
  }

  frame_displayed_.store(false, std::memory_order_release);

  auto kind = itens.memory_location() == MemoryLocation::DEVICE
                  ? cudaMemcpyDeviceToHost
                  : cudaMemcpyHostToHost;

  CUDA_CHECK(cudaMemcpyAsync(h_buffer_.get(), itens.data(),
                             itens.size_in_bytes(), kind, stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));

  TensorView host_view(h_buffer_.get(), host_meta_);
  emit frame_ready(host_view);

  // auto startTime = std::chrono::steady_clock::now();
  // nvtxRangePush(L"[QtDisplaySink] wait for frame displayed");
  // while (!frame_displayed_) {
  //   // std::this_thread::yield();
  //   auto elapsed = std::chrono::steady_clock::now() - startTime;
  //   if (elapsed > std::chrono::milliseconds(1000 / 25)) {
  //     holovibes_logger()->warn("Timeout waiting for frame to be displayed.");
  //     break;
  //   }
  // }

  // nvtxRangePop();
  last_display_time_ = std::chrono::steady_clock::now();
  holovibes_logger()->trace("FINISHED");
}

void QtDisplaySink::on_frame_displayed() { frame_displayed_ = true; }

// ==========================================================================
//                     QtDisplaySinkFactory Implementation
// ==========================================================================

QtDisplaySinkFactory::QtDisplaySinkFactory(TensorDisplayWidget &widget)
    : widget_(widget) {}

SinkMeta QtDisplaySinkFactory::type_check(const TensorMeta &imeta,
                                          const json &) {
  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Tensor meta sanity
  check(imeta.shape().size() == 3, "tensor rank != 3");
  check(imeta.shape().at(0) == 1, "tensor dim 0 != 1");
  check(imeta.data_type() == DataType::U8, "tensor data_type != U8");
  check(imeta.strides().at(2) == 1, "tensor stride 2 != 1");
  check(imeta.strides().at(1) == imeta.shape().at(2),
        "tensor stride 1 != tensor dim 2");
  check(imeta.strides().at(0) == imeta.shape().at(2) * imeta.shape().at(1),
        "tensor stride 0 != tensor dim 2 * tensor stride 1");

  // 2) Success
  return SinkMeta(imeta);
}

std::unique_ptr<Sink> QtDisplaySinkFactory::create(const TensorMeta &imeta,
                                                   const json &jparams,
                                                   cudaStream_t stream) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);

  auto host =
      curaii::cuda::make_unique_host_ptr<uint8_t>(imeta.size_in_bytes());

  TensorMeta host_meta(imeta.data_type(), MemoryLocation::HOST,
                       {imeta.shape().at(1), imeta.shape().at(2)});

  // 2) Assemble sink
  auto *sink = new QtDisplaySink(meta, stream, std::move(host), host_meta);

  // 3) Connect sink to display widget
  QObject::connect(sink, &QtDisplaySink::frame_ready, &widget_,
                   &TensorDisplayWidget::show_tensor, Qt::QueuedConnection);

  QObject::connect(
      &widget_, &TensorDisplayWidget::frame_displayed, sink,
      [sink]() { sink->on_frame_displayed(); }, Qt::QueuedConnection);

  return std::unique_ptr<QtDisplaySink>(sink);
}

} // namespace dh
