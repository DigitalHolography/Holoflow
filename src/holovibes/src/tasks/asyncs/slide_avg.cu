#include "slide_avg.hh"

#include <cmath>
#include <span>
#include <stdexcept>

#include <cub/cub.cuh>
#include <math_constants.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

namespace holovibes::tasks::asyncs {

// -------------------------------------------------------------------------------------------------
// JSON Serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const SlidingAverageSettings &s) {
  j = nlohmann::json{
      {"target_capacity", s.target_capacity},
      {"window_size", s.window_size},
  };
}

void from_json(const nlohmann::json &j, SlidingAverageSettings &s) {
  j.at("target_capacity").get_to(s.target_capacity);
  j.at("window_size").get_to(s.window_size);
}

namespace {

// -------------------------------------------------------------------------------------------------
// CUDA Kernels
// -------------------------------------------------------------------------------------------------

template <typename T>
__global__ void slide_avg_kernel(const T *push_data,
                                 const T *pop_data,
                                 T       *running_avg,
                                 T       *odata,
                                 int      nx,
                                 int      ny,
                                 int      avg_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  running_avg[idx] += push_data[idx] / avg_size;
  running_avg[idx] -= pop_data[idx] / avg_size;
  odata[idx] = running_avg[idx];
}

} // namespace

// -------------------------------------------------------------------------------------------------
// SlidingAverage Implementation
// -------------------------------------------------------------------------------------------------

SlidingAverage::SlidingAverage(SlidingAverageSettings       settings,
                               const holoflow::core::TDesc &idesc,
                               holoflow::core::TDesc       &odesc,
                               cudaStream_t                 producer_stream,
                               cudaStream_t                 consumer_stream,
                               size_t                       nb_slots,
                               size_t                       element_size,
                               DevPtr<std::byte>          &&d_buffer,
                               DevPtr<float>              &&d_running_avg) :
    settings_(std::move(settings)),
    idesc_(idesc),
    odesc_(odesc),
    producer_stream_(producer_stream),
    consumer_stream_(consumer_stream),
    nb_slots_(nb_slots),
    element_size_(element_size),
    d_buffer_(std::move(d_buffer)),
    d_running_avg_(std::move(d_running_avg)),
    avg_idx_(nb_slots - settings.window_size),
    write_idx_(0),
    read_idx_(nb_slots - 1) {}

int SlidingAverage::writer_size() const {
  const int w = write_idx_.load(std::memory_order_relaxed);
  const int r = read_idx_.load(std::memory_order_acquire);
  return (w >= r) ? (w - r) : (w - r + nb_slots_);
}

int SlidingAverage::reader_size() const {
  const int w = write_idx_.load(std::memory_order_acquire);
  const int r = read_idx_.load(std::memory_order_relaxed);
  return (w >= r) ? (w - r) : (w - r + nb_slots_);
}

std::byte *SlidingAverage::get_slot_ptr(size_t index) const {
  return d_buffer_.get() + (index * element_size_);
}

size_t SlidingAverage::next_slot_index(size_t current_index) const {
  size_t next = current_index + 1;
  return (next == nb_slots_) ? 0 : next;
}

std::optional<holoflow::core::TView> SlidingAverage::acquire_input(int index) {
  if (index != 0) {
    throw std::out_of_range("SlidingAverage::acquire_input: invalid index");
  }

  // Ensure we have space (at least 1 slot free)
  if (nb_slots_ - writer_size() <= 1) {
    return std::nullopt;
  }

  const int idx = write_idx_.load(std::memory_order_relaxed);

  return holoflow::core::TView{
      .data = get_slot_ptr(idx),
      .desc = idesc_,
  };
}

void SlidingAverage::release_output(int index) {
  if (index != 0) {
    throw std::out_of_range("SlidingAverage::release_output: invalid index");
  }

  const int current = read_idx_.load(std::memory_order_relaxed);
  read_idx_.store(next_slot_index(current), std::memory_order_release);
}
holoflow::core::OpResult SlidingAverage::try_push(holoflow::core::AsyncPushCtx &) {
  // Load Indices
  const size_t w_idx = write_idx_.load(std::memory_order_relaxed);
  const size_t a_idx = avg_idx_.load(std::memory_order_relaxed);

  // Prepare Kernel Arguments
  const float *push_data = reinterpret_cast<const float *>(get_slot_ptr(w_idx));
  const float *pop_data  = reinterpret_cast<const float *>(get_slot_ptr(a_idx));
  float       *out_data  = reinterpret_cast<float *>(get_slot_ptr(a_idx));

  const int height      = idesc_.shape.at(1);
  const int width       = idesc_.shape.at(2);
  const int window_size = static_cast<int>(settings_.window_size);

  // Launch Kernel
  const dim3 block_dim(16, 16);
  const dim3 grid_dim((width + block_dim.x - 1) / block_dim.x,
                      (height + block_dim.y - 1) / block_dim.y);

  slide_avg_kernel<<<grid_dim, block_dim, 0, producer_stream_>>>(push_data,
                                                                 pop_data,
                                                                 d_running_avg_.get(),
                                                                 out_data,
                                                                 width,
                                                                 height,
                                                                 window_size);

  // Note: Synchronizing here is required because the consumer stream may read
  // the output data immediately after the update of write_idx_ below.
  CUDA_CHECK(cudaStreamSynchronize(producer_stream_));

  // Update Indices
  avg_idx_.store(next_slot_index(a_idx), std::memory_order_release);
  write_idx_.store(next_slot_index(w_idx), std::memory_order_release);

  return holoflow::core::OpResult::Ok;
}

holoflow::core::OpResult SlidingAverage::try_pop(holoflow::core::AsyncPopCtx &ctx) {
  // Ensure we have data to read (at least window_size + 1 slots)
  if (reader_size() <= settings_.window_size) {
    return holoflow::core::OpResult::NotReady;
  }

  const int idx = read_idx_.load(std::memory_order_relaxed);

  ctx.outputs[0] = holoflow::core::TView{
      .data = get_slot_ptr(idx),
      .desc = odesc_,
  };

  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult
SlidingAverageFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                             const nlohmann::json                  &jsettings) const {
  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::logic_error(std::string(what));
    }
  };

  auto settings = jsettings.get<SlidingAverageSettings>();

  // validate
  check(input_descs.size() == 1, "unexpected input port");
  auto idesc = input_descs[0];
  check(idesc.dtype == holoflow::core::DType::F32, "input must be float32");
  check(idesc.rank() == 3, "input must be 3D");
  check(idesc.shape[0] == 1, "input batch size must be 1");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "input must be in device memory");
  check(settings.target_capacity > 0, "target_capacity must be greater or equal than zero");
  check(settings.window_size > 0, "window_size must be greater or equal than zero");

  auto odesc        = input_descs[0];
  odesc.shape.at(0) = 1;

  // Success
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {true},
      .owned_outputs = {true},
      .kind          = holoflow::core::TaskKind::Async,
  };
}

std::unique_ptr<holoflow::core::IAsyncTask>
SlidingAverageFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                              const nlohmann::json                  &jsettings,
                              const holoflow::core::AsyncCreateCtx  &ctx) const {
  using curaii::make_unique_device_ptr;

  auto        result   = infer(input_descs, jsettings);
  auto        settings = jsettings.get<SlidingAverageSettings>();
  const auto &idesc    = input_descs[0];

  int    nb_slots     = settings.target_capacity + settings.window_size;
  size_t element_size = idesc.num_bytes();
  size_t buffer_size  = nb_slots * element_size;

  auto d_buffer      = make_unique_device_ptr<std::byte>(buffer_size);
  auto d_running_avg = make_unique_device_ptr<float>(element_size / sizeof(float));
  CUDA_CHECK(cudaMemsetAsync(d_running_avg.get(), 0, element_size, ctx.producer_stream));
  CUDA_CHECK(cudaMemsetAsync(d_buffer.get(), 0, buffer_size, ctx.producer_stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx.producer_stream));

  // Success
  auto task = new SlidingAverage(settings,
                                 idesc,
                                 result.output_descs[0],
                                 ctx.producer_stream,
                                 ctx.consumer_stream,
                                 nb_slots,
                                 element_size,
                                 std::move(d_buffer),
                                 std::move(d_running_avg));
  return std::unique_ptr<holoflow::core::IAsyncTask>(task);
}

std::unique_ptr<holoflow::core::IAsyncTask>
SlidingAverageFactory::update(std::unique_ptr<holoflow::core::IAsyncTask> old_task,
                              std::span<const holoflow::core::TDesc>      input_descs,
                              const nlohmann::json                       &jsettings,
                              const holoflow::core::AsyncCreateCtx       &ctx) const {
  return create(input_descs, jsettings, ctx);
}

} // namespace holovibes::tasks::asyncs