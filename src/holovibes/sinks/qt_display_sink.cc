#include "holovibes/sinks/qt_display_sink.hh"

#include <glog/logging.h>

namespace dh {

// ==========================================================================
//                     QtDisplaySink Implementation
// ==========================================================================

QtDisplaySink::QtDisplaySink(const SinkMeta &meta, cudaStream_t stream)
    : Sink(meta, stream) {}

tl::expected<void, Error> QtDisplaySink::run(TensorView itens) {
  emit frame_ready(itens);
  return {};
}

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

  if (imeta.strides().at(0) != imeta.shape().at(2)) {
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

  return std::unique_ptr<QtDisplaySink>(sink);
}

} // namespace dh