// Copyright 2026 Digital Holography Foundation
// Licensed under the Apache License, Version 2.0.

#include "array_collector.hh"
#include "fft_fusion.hh"
#include "finite_graph.hh"
#include "latency.hh"
#include "phase_fusion.hh"
#include "spatial_flatfield_fusion.hh"
#include "throughput.hh"
#include <algorithm>
#include <condition_variable>
#include <cuda_runtime_api.h>
#include <limits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ranges>
#include <set>
#include <thread>

#include "holofile/holofile.hh"
#include "holoflow/core/registry.hh"
#include "holoflow/runtime/compiler.hh"
#include "holoflow/runtime/diagnostics.hh"
#include "holoflow/runtime/graph_display.hh"
#include "holoflow/runtime/tracing.hh"
#include "holonp/abs.hh"
#include "holonp/add.hh"
#include "holonp/arange.hh"
#include "holonp/asarray.hh"
#include "holonp/copy.hh"
#include "holonp/divide.hh"
#include "holonp/exp.hh"
#include "holonp/fft2.hh"
#include "holonp/fftshift.hh"
#include "holonp/irfft2.hh"
#include "holonp/matmul.hh"
#include "holonp/max.hh"
#include "holonp/mean.hh"
#include "holonp/meshgrid.hh"
#include "holonp/min.hh"
#include "holonp/multiply.hh"
#include "holonp/remainder.hh"
#include "holonp/reshape.hh"
#include "holonp/rfft2.hh"
#include "holonp/slice.hh"
#include "holonp/square.hh"
#include "holonp/subtract.hh"
#include "holonp/sum.hh"
#include "holonp/transpose.hh"
#include "holonp/window_ops.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/dual_reader_batch_queue.hh"
#include "holotask/asyncs/pca_decomposition_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/sinks/holofile.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/causal_sliding_average.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/correct_phase.hh"
#include "holotask/syncs/filter2d.hh"
#include "holotask/syncs/flatfield.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/matmul_lowering.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/normalize.hh"
#include "holotask/syncs/pca.hh"
#include "holotask/syncs/pct_clip.hh"
#include "holotask/syncs/shack_hartmann_slopes.hh"
#include "holotask/syncs/short_time_fresnel_diffraction.hh"
#include "holotask/syncs/wrap2pi.hh"
#include "holotask/syncs/zernike_from_slopes.hh"
#include "holotask/syncs/zernike_phase.hh"

namespace py = pybind11;
namespace hc = holoflow::core;
namespace hr = holoflow::runtime;
using Json   = nlohmann::json;

namespace {
class VoidTask final : public hc::ISyncTask {
public:
  hc::OpResult execute(hc::SyncCtx &) override { return hc::OpResult::Ok; }
  bool         supports_cuda_graph() const noexcept override { return true; }
  void         record_cuda_graph(hc::CudaGraphCtx &) override {}
};
class VoidFactory final : public hc::ISyncTaskFactory {
public:
  hc::InferResult infer(std::span<const hc::TDesc> inputs, const Json &) const override {
    if (inputs.size() != 1)
      throw std::invalid_argument("Void expects one tensor");
    return {.input_descs   = {inputs[0]},
            .output_descs  = {},
            .in_place      = {},
            .owned_inputs  = {false},
            .owned_outputs = {},
            .kind          = hc::TaskKind::Sync};
  }
  std::unique_ptr<hc::ISyncTask> create(std::span<const hc::TDesc> inputs, const Json &settings,
                                        const hc::SyncCreateCtx &) const override {
    (void)infer(inputs, settings);
    return std::make_unique<VoidTask>();
  }
};

std::unique_ptr<hc::Registry> make_registry(bool include_lowered_tasks = false) {
  auto r = std::make_unique<hc::Registry>();
  if (include_lowered_tasks) {
    r->register_sync("FusedSpatialFlatfield",
                     std::make_unique<holotask::syncs::FusedSpatialFlatfieldFactory>());
    r->register_sync("FusedPhaseCorrection",
                     std::make_unique<holotask::syncs::FusedPhaseCorrectionFactory>());
    r->register_sync("FusedFFT2", std::make_unique<holoflow::python::FusedFft2Factory>(*r));
    r->register_sync("FusedMatmul", std::make_unique<holotask::syncs::FusedMatmulFactory>());
  }
  r->register_sync("ArrayCollector", std::make_unique<holoflow::python::ArrayCollectorFactory>());
  r->register_sync("Holofile", std::make_unique<holotask::sources::HolofileFactory>());
  r->register_sync("HolofileWriter", std::make_unique<holotask::sinks::HolofileFactory>());
  r->register_sync("Memcpy", std::make_unique<holotask::syncs::MemcpyFactory>());
  r->register_sync("Conversion", std::make_unique<holotask::syncs::ConversionFactory>());
  r->register_sync("Reshape", std::make_unique<holonp::ReshapeFactory>());
  r->register_sync("Mean", std::make_unique<holonp::MeanFactory>());
  r->register_sync("Min", std::make_unique<holonp::MinFactory>());
  r->register_sync("Max", std::make_unique<holonp::MaxFactory>());
  r->register_sync("Subtract", std::make_unique<holonp::SubtractFactory>());
  r->register_sync("Divide", std::make_unique<holonp::DivideFactory>());
  r->register_sync("Filter2D", std::make_unique<holotask::syncs::Filter2DFactory>());
  r->register_sync("Pad", std::make_unique<holonp::PadFactory>());
  r->register_sync("SlidingWindowView", std::make_unique<holonp::SlidingWindowViewFactory>());
  r->register_sync("Sum", std::make_unique<holonp::SumFactory>());
  r->register_sync("RFFT2", std::make_unique<holonp::RFFT2Factory>());
  r->register_sync("IRFFT2", std::make_unique<holonp::IRFFT2Factory>());
  r->register_sync("Flatfield", std::make_unique<holotask::syncs::FlatfieldFactory>());
  r->register_sync("Matmul", std::make_unique<holonp::MatmulFactory>());
  r->register_sync("Eigh", std::make_unique<holotask::syncs::EighFactory>());
  r->register_sync("Pca", std::make_unique<holotask::syncs::PcaFactory>());
  r->register_sync("PcaProjection", std::make_unique<holotask::syncs::PcaProjectionFactory>());
  r->register_async("PcaDecompositionQueue",
                    std::make_unique<holotask::asyncs::PcaDecompositionQueueFactory>());
  r->register_sync("PctClip", std::make_unique<holotask::syncs::PctClipFactory>());
  r->register_sync("FresnelDiffraction",
                   std::make_unique<holotask::syncs::FresnelDiffractionFactory>());
  r->register_sync("PcaCovariance", std::make_unique<holotask::syncs::PcaCovarianceFactory>());
  r->register_sync("SymmetricEigendecomposition",
                   std::make_unique<holotask::syncs::SymmetricEigendecompositionFactory>());
  r->register_async("MultiReaderBatchQueue",
                    std::make_unique<holotask::asyncs::MultiReaderBatchQueueFactory>());
  r->register_async("BatchQueue", std::make_unique<holotask::asyncs::BatchQueueFactory>());
  r->register_async("SlidingAverage", std::make_unique<holotask::asyncs::SlidingAverageFactory>());
  r->register_sync("Arange", std::make_unique<holonp::ArangeFactory>());
  r->register_sync("AsArray", std::make_unique<holonp::AsArrayFactory>());
  r->register_sync("Meshgrid", std::make_unique<holonp::MeshgridFactory>());
  r->register_sync("Square", std::make_unique<holonp::SquareFactory>());
  r->register_sync("Add", std::make_unique<holonp::AddFactory>());
  r->register_sync("Multiply", std::make_unique<holonp::MultiplyFactory>());
  r->register_sync("Remainder", std::make_unique<holonp::RemainderFactory>());
  r->register_sync("Exp", std::make_unique<holonp::ExpFactory>());
  r->register_sync("FFT2", std::make_unique<holonp::FFT2Factory>());
  r->register_sync("Abs", std::make_unique<holonp::AbsFactory>());
  r->register_sync("FFTShiftNp", std::make_unique<holonp::FFTShiftFactory>());
  r->register_sync("Void", std::make_unique<VoidFactory>());
  r->register_sync("Throughput", std::make_unique<holoflow::python::ThroughputFactory>());
  r->register_sync("LatencyStart", std::make_unique<holoflow::python::LatencyMarkerFactory>());
  r->register_sync("LatencyEnd", std::make_unique<holoflow::python::LatencyMarkerFactory>());
  r->register_async("DualReaderBatchQueue",
                    std::make_unique<holotask::asyncs::DualReaderBatchQueueFactory>());
  r->register_sync("ShortTimeFresnelDiffraction",
                   std::make_unique<holotask::syncs::ShortTimeFresnelDiffractionFactory>());
  r->register_sync("CausalSlidingAverage",
                   std::make_unique<holotask::syncs::CausalSlidingAverageFactory>());
  r->register_sync("ShackHartmannSlopes",
                   std::make_unique<holotask::syncs::ShackHartmannSlopesFactory>());
  r->register_sync("ZernikeFromSlopes",
                   std::make_unique<holotask::syncs::ZernikeFromSlopesFactory>());
  r->register_sync("ZernikePhase", std::make_unique<holotask::syncs::ZernikePhaseFactory>());
  r->register_sync("CorrectPhase", std::make_unique<holotask::syncs::CorrectPhaseFactory>());
  r->register_sync("Normalize", std::make_unique<holotask::syncs::NormalizeFactory>());
  r->register_sync("Wrap2Pi", std::make_unique<holotask::syncs::Wrap2PiFactory>());
  r->register_sync("Copy", std::make_unique<holonp::CopyFactory>());
  r->register_sync("Transpose", std::make_unique<holonp::TransposeFactory>());
  r->register_sync("Slice", std::make_unique<holonp::SliceFactory>());
  return r;
}

class ExecutionError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

class ProfileCapture {
public:
  explicit ProfileCapture(const std::string &directory) : path_(directory) {
    std::filesystem::create_directories(path_);
    if (hr::tracing::Session::available()) {
      session_ = hr::tracing::Session::start();
      if (!session_)
        throw std::runtime_error("Another process-wide Perfetto capture is active");
    }
  }
  ~ProfileCapture() {
    try {
      close();
    } catch (...) { /* Explicit close reports export failures. */
    }
  }
  bool available() const { return hr::tracing::Session::available(); }
  void begin_trace() {
    trace_ = std::make_unique<hr::tracing::ScopedTrace>("Python Graph Construction");
  }
  void end_trace() { trace_.reset(); }
  void close() {
    end_trace();
    hr::flush_diagnostics();
    if (session_) {
      auto session = std::move(session_);
      session->stop_and_save(path_ / "trace_events.perfetto-trace");
    }
  }

private:
  std::filesystem::path                     path_;
  std::unique_ptr<hr::tracing::Session>     session_;
  std::unique_ptr<hr::tracing::ScopedTrace> trace_;
};

class Executable {
public:
  explicit Executable(const hc::GraphSpec &spec, const std::string &log_dir, bool capture,
                      std::unique_ptr<hr::CompilerOutput> previous = {}, bool fuse_tasks = true)
      : capture_(!log_dir.empty() && capture ? std::make_unique<ProfileCapture>(log_dir) : nullptr),
        log_dir_(log_dir), registry_(make_registry(true)),
        compiler_(std::make_unique<hr::Compiler>(
            *registry_,
            hr::Compiler::Config{
                .log_dir             = log_dir,
                .dump_dot_on_failure = !log_dir.empty(),
                .verbose_tracing     = false,
                .enable_profiling    = false,
                .lower_tasks         = fuse_tasks ? holoflow::python::lower_fft_tasks : nullptr,
                .lower_matmul_tasks  = fuse_tasks ? holotask::syncs::lower_matmul_tasks : nullptr,
                .lower_phase_tasks   = fuse_tasks ? holoflow::python::lower_phase_tasks : nullptr,
                .lower_spatial_flatfield_tasks =
                    fuse_tasks ? holoflow::python::lower_spatial_flatfield_tasks : nullptr})) {
    for (const auto v : boost::make_iterator_range(vertices(spec))) {
      if (spec[v].kind == "FusedFFT2" || spec[v].kind == "FusedMatmul" ||
          spec[v].kind == "FusedPhaseCorrection" || spec[v].kind == "FusedSpatialFlatfield")
        throw std::invalid_argument(spec[v].kind + " is an internal compiler task");
      node_ids_.insert(spec[v].name);
    }
    for (const auto v : boost::make_iterator_range(vertices(spec)))
      names_.push_back(spec[v].name);
    output_ = compiler_->compile(holoflow::python::prepare_finite_graph(spec, *registry_),
                                 std::move(previous));
    for (const auto &[id, task] : output_->resources.tasks) {
      if (auto *probe = dynamic_cast<holoflow::python::ThroughputTask *>(task.get()))
        probes_.emplace(id, probe);
    }
    scheduler_ = std::make_unique<hr::Scheduler>(
        output_->graph, output_->sections, output_->resources, std::chrono::milliseconds{1000},
        hr::Scheduler::FailureCallback{}, hr::Scheduler::FailurePolicy::Capture);
    configure_latency();
  }
  ~Executable() {
    try {
      if (scheduler_ && scheduler_->is_running()) {
        stop();
        wait();
      }
    } catch (...) { /* Explicit wait reports failures. */
    }
  }
  void check_failure() const {
    if (!scheduler_)
      throw std::runtime_error("Executable is closed");
    try {
      scheduler_->rethrow_failure();
    } catch (const std::exception &e) {
      throw ExecutionError(e.what());
    }
  }
  void start() {
    check_failure();
    const auto started = Clock::now();
    scheduler_->start();
    if (!probes_.empty() || latency_) {
      for (const auto &[id, probe] : probes_)
        probe->start(started);
      if (latency_)
        latency_->start(started);
      reporter_ = std::jthread([this](std::stop_token stop) { report_metrics(stop); });
    }
  }
  void stop() {
    if (scheduler_->is_running())
      scheduler_->request_stop();
  }
  void wait() {
    scheduler_->wait();
    const auto ended = Clock::now();
    if (reporter_.joinable()) {
      reporter_.request_stop();
      reporter_.join();
      const auto      counts = scheduler_->progress().at("successful_emissions");
      std::lock_guard lock(events_mutex_);
      for (const auto &[id, probe] : probes_)
        pending_events_[id] =
            probe->summary(counts.value(id, uint64_t{0}), ended, scheduler_->completed());
      if (latency_) {
        auto summary                  = latency_->snapshot();
        summary["type"]               = "latency_summary";
        summary["completed"]          = scheduler_->completed();
        pending_events_[latency_->id] = std::move(summary);
      }
    }
    check_failure();
    std::lock_guard lock(events_mutex_);
    if (reporting_failure_)
      std::rethrow_exception(reporting_failure_);
  }
  void close(bool finalize_diagnostics = true) {
    if (!scheduler_)
      return;
    std::exception_ptr failure;
    try {
      stop();
      wait();
    } catch (...) {
      failure = std::current_exception();
    }
    if (reporter_.joinable()) {
      reporter_.request_stop();
      reporter_.join();
    }
    scheduler_.reset();
    output_.reset();
    compiler_.reset();
    registry_.reset();
    if (!log_dir_.empty() && finalize_diagnostics) {
      try {
        hr::flush_diagnostics();
        if (capture_)
          capture_->close();
      } catch (...) {
        if (!failure)
          failure = std::current_exception();
      }
    }
    if (failure)
      std::rethrow_exception(failure);
  }
  std::unique_ptr<hr::CompilerOutput> take_compilation() {
    check_failure();
    if (!scheduler_->completed() || scheduler_->is_running() || reporter_.joinable())
      throw std::invalid_argument("Graph update requires a completed and joined finite run");
    hr::flush_diagnostics();
    if (capture_)
      capture_->close();
    scheduler_.reset();
    return std::move(output_);
  }
  bool completed() const {
    check_failure();
    return scheduler_->completed();
  }
  py::array array_result(size_t vertex) const {
    check_failure();
    if (!scheduler_->completed())
      throw std::runtime_error("Execution has not completed successfully");
    auto *collector = dynamic_cast<holoflow::python::ArrayCollector *>(
        output_->resources.tasks.at(names_.at(vertex)).get());
    if (!collector)
      throw std::invalid_argument("Result handle does not identify an array collector");
    return collector->array();
  }
  bool is_running() const { return scheduler_->is_running(); }
  bool stop_requested() const { return scheduler_->stop_requested(); }
  bool send_event(const std::string &node_id, const std::string &data) {
    check_failure();
    if (!node_ids_.contains(node_id))
      throw std::invalid_argument("Unknown node ID: " + node_id);
    return scheduler_->ui_try_send(node_id, Json::parse(data));
  }
  std::optional<std::pair<std::string, std::string>> receive_event() {
    check_failure();
    auto event = scheduler_->ui_try_receive();
    if (event)
      return std::pair{event->node_id, event->data.dump()};
    std::lock_guard lock(events_mutex_);
    if (reporting_failure_)
      std::rethrow_exception(reporting_failure_);
    if (pending_events_.empty())
      return std::nullopt;
    // Round-robin among pending nodes, even if one node refreshes much faster.
    auto next = pending_events_.upper_bound(last_reported_);
    if (next == pending_events_.end())
      next = pending_events_.begin();
    auto entry     = pending_events_.extract(next);
    last_reported_ = entry.key();
    return std::pair{entry.key(), entry.mapped().dump()};
  }
  std::string metrics() const {
    Json result = Json::object();
    for (const auto &[id, m] : scheduler_->metrics())
      result[id] = {{"average_duration_ms", m.average_duration_ms},
                    {"runs_per_second", m.runs_per_second},
                    {"host_throughput_bytes_per_second", m.host_throughput_bytes_per_second},
                    {"device_throughput_bytes_per_second", m.device_throughput_bytes_per_second},
                    {"sample_count", m.sample_count},
                    {"individual_timing_available", m.individual_timing_available}};
    return result.dump();
  }
  std::string progress() const {
    check_failure();
    return scheduler_->progress().dump();
  }
  std::string latency(const std::string &id) const {
    check_failure();
    if (!latency_ || id != latency_->id)
      throw std::invalid_argument("Unknown latency output node: " + id);
    return latency_->snapshot().dump();
  }

private:
  using Clock = holoflow::python::ThroughputTask::Clock;

  static uint64_t latency_setting(const Json &settings, const char *key, uint64_t minimum,
                                  uint64_t maximum = std::numeric_limits<uint64_t>::max()) {
    const auto &value = settings.at(key);
    if (!value.is_number_integer() || (!value.is_number_unsigned() && value.get<int64_t>() < 0))
      throw std::invalid_argument(std::string("Invalid latency setting: ") + key);
    const auto result = value.get<uint64_t>();
    if (result < minimum || result > maximum)
      throw std::invalid_argument(std::string("Invalid latency setting: ") + key);
    return result;
  }

  void configure_latency() {
    const auto &graph = output_->graph;
    using Vertex      = hr::GraphPlan::vertex_descriptor;
    std::optional<Vertex> start, end;
    for (auto v : boost::make_iterator_range(vertices(graph))) {
      const auto &kind = graph[v].spec.kind;
      if (kind != "LatencyStart" && kind != "LatencyEnd")
        continue;
      auto &marker = kind == "LatencyStart" ? start : end;
      if (marker)
        throw std::invalid_argument("Only one latency probe pair is supported");
      marker = v;
    }
    if (!start && !end)
      return;
    if (!start || !end)
      throw std::invalid_argument("Latency requires both a start and an end probe");
    const auto &settings = graph[*end].spec.settings;
    if (settings.at("source_node_id") != graph[*start].spec.name)
      throw std::invalid_argument("Latency source_node_id must identify the start probe");
    if (in_degree(*start, graph) != 1 ||
        graph[source(*in_edges(*start, graph).first, graph)].spec.kind != "Holofile")
      throw std::invalid_argument("Latency start must directly follow Holofile");
    if (out_degree(*end, graph) != 1)
      throw std::invalid_argument("Latency end requires one downstream sink");
    const auto &sink     = graph[target(*out_edges(*end, graph).first, graph)].spec.kind;
    auto        producer = *end;
    do {
      producer = source(*in_edges(producer, graph).first, graph);
    } while (graph[producer].spec.kind == "Throughput");
    // Host queue publication follows the producer's stream barrier.
    const bool host_ready =
        graph[producer].spec.kind == "BatchQueue" &&
        graph[producer].infer.output_descs[0].mem_loc == hc::MemLoc::Host &&
        std::ranges::any_of(output_->sections, [&](const auto &section) {
          return !section.has_synchronizing_async_producer &&
                 std::ranges::find(section.async_prod, producer) != section.async_prod.end();
        });
    if (sink != "Void" && !(sink == "ArrayCollector" && host_ready))
      throw std::invalid_argument(
          "Latency end requires Void or a NumPy sink after a synchronized host queue");
    const auto section = std::ranges::find_if(output_->sections, [&](const auto &section) {
      return std::ranges::find(section.sync_topo, *end) != section.sync_topo.end();
    });
    if (section == output_->sections.end() || section->has_synchronizing_async_producer)
      throw std::invalid_argument(
          "Latency end requires a section with an unconditional CUDA barrier");
    std::set<Vertex>    upstream;
    std::vector<Vertex> pending{*end};
    while (!pending.empty()) {
      const auto v = pending.back();
      pending.pop_back();
      if (upstream.insert(v).second)
        for (auto e : boost::make_iterator_range(in_edges(v, graph)))
          pending.push_back(source(e, graph));
    }
    if (!upstream.contains(*start))
      throw std::invalid_argument("Latency end must depend on the start probe");
    auto *source_settings = &graph[*start].spec.settings;
    latency_              = std::make_unique<holoflow::python::LatencyMeasurement>(
        graph[*end].spec.name, graph[*start].infer.input_descs[0].shape.at(0),
        latency_setting(*source_settings, "history_batches", 1, std::numeric_limits<size_t>::max()),
        latency_setting(settings, "first_frame", 0), latency_setting(settings, "window_frames", 1),
        latency_setting(settings, "stride_frames", 1),
        static_cast<int>(
            latency_setting(settings, "interval_ms", 1, std::numeric_limits<int>::max())));
    scheduler_->observe_sync_node(*start, false,
                                  [probe = latency_.get()](auto now) { probe->input(now); });
    scheduler_->observe_sync_node(*end, !host_ready,
                                  [probe = latency_.get()](auto now) { probe->output(now); });
  }

  void report_metrics(std::stop_token stop) noexcept {
    try {
      std::condition_variable_any timer;
      std::mutex                  timer_mutex;
      std::unique_lock            lock(timer_mutex);
      while (!stop.stop_requested()) {
        auto deadline = Clock::time_point::max();
        for (const auto &[id, probe] : probes_)
          deadline = std::min(deadline, probe->deadline());
        if (latency_)
          deadline = std::min(deadline, latency_->deadline());
        timer.wait_until(lock, stop, deadline, [] { return false; });
        if (stop.stop_requested() || scheduler_->stop_requested())
          return;
        const auto counts =
            probes_.empty() ? Json::object() : scheduler_->progress().at("successful_emissions");
        const auto      now = Clock::now();
        std::lock_guard events_lock(events_mutex_);
        for (const auto &[id, probe] : probes_)
          if (probe->deadline() <= now)
            pending_events_[id] = probe->sample(counts.value(id, uint64_t{0}), now);
        if (latency_ && latency_->deadline() <= now)
          pending_events_[latency_->id] = latency_->sample(now);
      }
    } catch (...) {
      std::lock_guard lock(events_mutex_);
      reporting_failure_ = std::current_exception();
    }
  }

  // Destruction order matters: scheduler borrows output; compiler borrows registry.
  std::unique_ptr<ProfileCapture>                           capture_;
  std::string                                               log_dir_;
  std::unique_ptr<hc::Registry>                             registry_;
  std::unique_ptr<hr::Compiler>                             compiler_;
  std::unique_ptr<hr::CompilerOutput>                       output_;
  std::unique_ptr<holoflow::python::LatencyMeasurement>     latency_;
  std::unique_ptr<hr::Scheduler>                            scheduler_;
  std::set<std::string>                                     node_ids_;
  std::vector<std::string>                                  names_;
  std::map<std::string, holoflow::python::ThroughputTask *> probes_;
  std::mutex                                                events_mutex_;
  std::map<std::string, Json>                               pending_events_;
  std::string                                               last_reported_;
  std::exception_ptr                                        reporting_failure_;
  std::jthread                                              reporter_;
};

class Graph {
public:
  explicit Graph(hc::GraphSpec spec) : spec_(std::move(spec)) {}
  explicit Graph(const std::string &json) : spec_(hc::from_json(Json::parse(json))) {}
  std::string                 json() const { return hc::to_json(spec_).dump(); }
  std::string                 dot() const { return hc::to_dot(spec_); }
  std::unique_ptr<Executable> compile(const std::string &log_dir, bool capture,
                                      Executable *previous, bool fuse_tasks) const {
    auto output = previous ? previous->take_compilation() : nullptr;
    return std::make_unique<Executable>(spec_, log_dir, capture, std::move(output), fuse_tasks);
  }

private:
  hc::GraphSpec spec_;
};

class Builder {
public:
  Builder() : registry_(make_registry()) {}
  std::pair<size_t, std::vector<hc::TDesc>>
  add(const std::string &kind, const std::vector<std::pair<size_t, size_t>> &inputs,
      const std::string &settings_json, const std::optional<std::string> &node_id) {
    if (finished_)
      throw std::invalid_argument("Trace is already finalized");
    if (node_id && (node_id->empty() || explicit_ids_.contains(*node_id)))
      throw std::invalid_argument("Node ID must be nonempty and unique: " + *node_id);
    auto                   settings = Json::parse(settings_json);
    std::vector<hc::TDesc> descs;
    for (const auto &[vertex, port] : inputs)
      descs.push_back(outputs_.at(vertex).at(port));
    hc::InferResult inferred;
    try {
      if (registry_->is_sync_registered(kind))
        inferred = registry_->get_sync(kind).infer(descs, settings);
      else
        inferred = registry_->get_async(kind).infer(descs, settings);
    } catch (const std::exception &e) {
      throw std::invalid_argument(kind + " inference failed for inputs " + Json(descs).dump() +
                                  ": " + e.what());
    }
    const auto v = add_vertex(hc::NodeSpec{node_id.value_or(""), kind, settings}, spec_);
    for (size_t i = 0; i < inputs.size(); ++i)
      add_edge(inputs[i].first, v,
               hc::EdgeSpec{static_cast<int>(inputs[i].second), static_cast<int>(i)}, spec_);
    if (node_id)
      explicit_ids_.insert(*node_id);
    outputs_.push_back(inferred.output_descs);
    return {v, std::move(inferred.output_descs)};
  }
  std::unique_ptr<Graph> finalize() {
    if (finished_ || num_vertices(spec_) == 0)
      throw std::invalid_argument("Cannot finalize an empty or finished trace");
    auto   ids  = explicit_ids_;
    size_t next = 0;
    for (const auto v : boost::make_iterator_range(vertices(spec_))) {
      if (!spec_[v].name.empty())
        continue;
      std::string id;
      do {
        id = "node_" + std::to_string(next++);
      } while (ids.contains(id));
      spec_[v].name = id;
      ids.insert(id);
    }
    finished_ = true;
    return std::make_unique<Graph>(spec_);
  }

private:
  std::unique_ptr<hc::Registry>       registry_;
  hc::GraphSpec                       spec_;
  std::vector<std::vector<hc::TDesc>> outputs_;
  std::set<std::string>               explicit_ids_;
  bool                                finished_ = false;
};
} // namespace

PYBIND11_MODULE(_native, m) {
  m.doc()               = "Native Holoflow graph tracing and execution";
  m.attr("__version__") = HOLOFLOW_PYTHON_VERSION;
  py::register_exception<ExecutionError>(m, "ExecutionError");
  py::enum_<holoflow::core::DType>(m, "DType")
      .value("uint8", holoflow::core::DType::U8)
      .value("uint16", holoflow::core::DType::U16)
      .value("float16", holoflow::core::DType::F16)
      .value("float32", holoflow::core::DType::F32)
      .value("complex32", holoflow::core::DType::CF16)
      .value("complex64", holoflow::core::DType::CF32);
  py::class_<hc::TDesc>(m, "Descriptor")
      .def_readonly("shape", &hc::TDesc::shape)
      .def_readonly("dtype", &hc::TDesc::dtype)
      .def_readonly("strides", &hc::TDesc::strides)
      .def_readonly("offset", &hc::TDesc::offset)
      .def_property_readonly("device", [](const hc::TDesc &d) {
        return d.mem_loc == hc::MemLoc::Host ? "host" : "device";
      });
  py::class_<Builder>(m, "Builder")
      .def(py::init<>())
      .def("add", &Builder::add)
      .def("finalize", &Builder::finalize);
  py::class_<Graph>(m, "Graph")
      .def(py::init<const std::string &>())
      .def("to_json", &Graph::json)
      .def("to_dot", &Graph::dot)
      .def("compile", &Graph::compile, py::arg("log_dir") = "", py::arg("capture") = true,
           py::arg("previous") = nullptr, py::arg("fuse_tasks") = true,
           py::call_guard<py::gil_scoped_release>());
  py::class_<Executable>(m, "Executable", py::release_gil_before_calling_cpp_dtor())
      .def("start", &Executable::start, py::call_guard<py::gil_scoped_release>())
      .def("stop", &Executable::stop)
      .def("wait", &Executable::wait, py::call_guard<py::gil_scoped_release>())
      .def("close", &Executable::close, py::arg("finalize_diagnostics") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("completed", &Executable::completed)
      .def("array_result", &Executable::array_result)
      .def("is_running", &Executable::is_running)
      .def("stop_requested", &Executable::stop_requested)
      .def("check_failure", &Executable::check_failure)
      .def("send_event", &Executable::send_event)
      .def("receive_event", &Executable::receive_event)
      .def("metrics", &Executable::metrics)
      .def("latency", &Executable::latency)
      .def("progress", &Executable::progress);
  py::class_<ProfileCapture>(m, "ProfileCapture")
      .def(py::init<const std::string &>())
      .def_property_readonly("available", &ProfileCapture::available)
      .def("begin_trace", &ProfileCapture::begin_trace)
      .def("end_trace", &ProfileCapture::end_trace)
      .def("close", &ProfileCapture::close, py::call_guard<py::gil_scoped_release>());
  m.def("flush_diagnostics", &hr::flush_diagnostics, py::call_guard<py::gil_scoped_release>());
  m.def("perfetto_available", &hr::tracing::Session::available);
  m.def("supports_sm120", [] {
    int device = 0, major = 0, minor = 0;
    return cudaGetDevice(&device) == cudaSuccess &&
           cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device) ==
               cudaSuccess &&
           cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device) ==
               cudaSuccess &&
           major == 12 && minor == 0;
  });
  m.def("file_frame_count", [](const std::string &path) {
    const auto count = holofile::Reader(path).header().frame_count;
    if (count > static_cast<uint32_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("File frame count exceeds supported range");
    return static_cast<int>(count);
  });
}
