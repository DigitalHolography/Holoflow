#include "holovibes/sinks/qt_display_sink.hh"

#include <glog/logging.h>

#include "curaii/curaii.hh"

namespace dh {

// ==========================================================================
//                     QtDisplaySink Implementation
// ==========================================================================

QtDisplaySink::QtDisplaySink(const SinkMeta &meta, cudaStream_t stream)
    : Sink(meta, stream), frame_displayed_(true) {}

tl::expected<void, Error> QtDisplaySink::run(TensorView itens) {
  LOG(INFO) << "running qt display sink";
  frame_displayed_.store(false, std::memory_order_release);

  auto host = make_unique_host_ptr<uint8_t>(itens.size_in_bytes());
  CHECK(cudaMemcpyAsync(host.get(), itens.data(), itens.size_in_bytes(),
                        cudaMemcpyDeviceToHost, stream_) == cudaSuccess);

  CHECK(cudaStreamSynchronize(stream_) == cudaSuccess);
  TensorMeta host_meta(itens.data_type(), MemoryLocation::HOST,
                       {itens.shape().at(1), itens.shape().at(2)});
  TensorView host_view(host.get(), host_meta);
  emit frame_ready(host_view);

  // Wait for frame to be displayed.
  while (!frame_displayed_) {
    std::this_thread::yield();
  }
  // std::this_thread::sleep_for(std::chrono::milliseconds(100));
  LOG(INFO) << "FINISHED";
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
    LOG(WARNING) << "Invalid rank: " << imeta.shape().size();
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.shape().at(0) != 1) {
    LOG(WARNING) << "Invalid batch size: " << imeta.shape().at(0);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.data_type() != DataType::U8) {
    LOG(WARNING) << "Invalid data type: " << imeta.data_type();
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.strides().at(2) != 1) {
    LOG(WARNING) << "Invalid stride: " << imeta.strides().at(2);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.strides().at(1) != imeta.shape().at(2)) {
    LOG(WARNING) << "Invalid stride: " << imeta.strides().at(1);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  if (imeta.strides().at(0) != imeta.shape().at(2) * imeta.shape().at(1)) {
    LOG(WARNING) << "Invalid stride: " << imeta.strides().at(0);
    return tl::unexpected(Error::INTERNAL_ERROR);
  }

  return SinkMeta(imeta);
}

tl::expected<std::unique_ptr<Sink>, Error>
QtDisplaySinkFactory::create(const TensorMeta &imeta, const json &jparams,
                             cudaStream_t stream) {
  auto meta_result = type_check(imeta, jparams);
  if (!meta_result) {
    LOG(WARNING) << "type check failed";
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