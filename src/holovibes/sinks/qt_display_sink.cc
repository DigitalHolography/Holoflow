#include "holovibes/sinks/qt_display_sink.hh"

#include <cassert>
#include <cstdlib>
#include <spdlog/spdlog.h>

#include "bug_buster/bug_buster.hh"
#include "curaii/curaii.hh"
#include "holovibes/holovibes.hh"

namespace dh {

// ==========================================================================
//                     QtDisplaySink Implementation
// ==========================================================================

QtDisplaySink::QtDisplaySink(const SinkMeta &meta, CudaStreamRef stream)
    : Sink(meta, stream), frame_displayed_(true),
      last_display_time_(std::chrono::steady_clock::now()) {}

tl::expected<void, Error> QtDisplaySink::run(TensorView itens) {
  holovibes_logger()->trace("running qt display sink");

  // Throttle to 30 fps: skip frame if less than ~33ms since last display.
  auto now = std::chrono::steady_clock::now();
  if (now - last_display_time_ < std::chrono::milliseconds(1000 / 120)) {
    // Too soon: drop this frame.
    holovibes_logger()->trace("Skipping frame to maintain 30fps");
    return {};
  }

  frame_displayed_.store(false, std::memory_order_release);

  auto host = make_unique_host_ptr<uint8_t>(itens.size_in_bytes());
  DH_CHECK(cudaMemcpyAsync(host.get(), itens.data(), itens.size_in_bytes(),
                           cudaMemcpyDeviceToHost,
                           stream_.stream()) == cudaSuccess);

  DH_CHECK(stream_.try_synchronize());
  TensorMeta host_meta(itens.data_type(), MemoryLocation::HOST,
                       {itens.shape().at(1), itens.shape().at(2)});
  TensorView host_view(host.get(), host_meta);
  emit frame_ready(host_view);

  // Wait for frame to be displayed.
  while (!frame_displayed_) {
    std::this_thread::yield();
  }
  last_display_time_ = std::chrono::steady_clock::now();
  holovibes_logger()->trace("FINISHED");
  return {};
}

void QtDisplaySink::on_frame_displayed() { frame_displayed_ = true; }

// ==========================================================================
//                     QtDisplaySinkFactory Implementation
// ==========================================================================

QtDisplaySinkFactory::QtDisplaySinkFactory(TensorDisplayWidget &widget)
    : widget_(widget) {}

tl::expected<SinkMeta, Error>
QtDisplaySinkFactory::type_check(const TensorMeta &imeta, const json &) {
  if (imeta.shape().size() != 3) {
    holovibes_logger()->warn("Invalid rank: {}", imeta.shape().size());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().at(0) != 1) {
    holovibes_logger()->warn("Invalid batch size: {}", imeta.shape().at(0));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.data_type() != DataType::U8) {
    holovibes_logger()->warn("Invalid data type: {}", (int)imeta.data_type());
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.strides().at(2) != 1) {
    holovibes_logger()->warn("Invalid stride: {}", imeta.strides().at(2));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.strides().at(1) != imeta.shape().at(2)) {
    holovibes_logger()->warn("Invalid stride: {}", imeta.strides().at(1));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.strides().at(0) != imeta.shape().at(2) * imeta.shape().at(1)) {
    holovibes_logger()->warn("Invalid stride: {}", imeta.strides().at(0));
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return SinkMeta(imeta);
}

tl::expected<std::unique_ptr<Sink>, Error>
QtDisplaySinkFactory::create(const TensorMeta &imeta, const json &jparams,
                             CudaStreamRef stream) {
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    holovibes_logger()->warn("type check failed");
    return tl::unexpected(meta_result.error());
  }
  auto meta = meta_result.value();

  auto *sink = new QtDisplaySink(meta, stream);

  QObject::connect(sink, &QtDisplaySink::frame_ready, &widget_,
                   &TensorDisplayWidget::show_tensor, Qt::QueuedConnection);

  QObject::connect(
      &widget_, &TensorDisplayWidget::frame_displayed, sink,
      [sink]() { sink->on_frame_displayed(); }, Qt::QueuedConnection);

  return std::unique_ptr<QtDisplaySink>(sink);
}

} // namespace dh
