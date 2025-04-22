#pragma once

#include <QObject>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "holoflow/error.hh"
#include "holoflow/sink.hh"
#include "holoflow/tensor.hh"
#include "holovibes/ui/tensor_display_widget.hh"

using json = nlohmann::json;

namespace dh {

class QtDisplaySink : public QObject, public Sink {
  Q_OBJECT

public:
  ~QtDisplaySink() override = default;

  void run(TensorView itens) override;

  friend class QtDisplaySinkFactory;

signals:
  void frame_ready(TensorView itens);

private:
  QtDisplaySink(const SinkMeta &meta, cudaStream_t stream);

  void on_frame_displayed();

  std::atomic<bool> frame_displayed_;
  std::chrono::steady_clock::time_point last_display_time_;
};

class QtDisplaySinkFactory : public SinkFactory {
public:
  QtDisplaySinkFactory(TensorDisplayWidget &widget);

  ~QtDisplaySinkFactory() override = default;

  SinkMeta type_check(const TensorMeta &imeta, const json &params) override;

  std::unique_ptr<Sink> create(const TensorMeta &imeta, const json &params,
                               cudaStream_t stream) override;

private:
  TensorDisplayWidget &widget_;
};

} // namespace dh