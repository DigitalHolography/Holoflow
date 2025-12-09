#include "holotask/asyncs/slide_avg.hh"

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"

#include <cub/cub.cuh>
#include <math_constants.h>

#include "bug.hh"
#include "curaii/cuda.hh"
#include "holoflow/core/tasks.hh"
#include "logger.hh"
#include <span>

namespace holotask::asyncs {

void to_json(nlohmann::json &j, const SlidingAverageSettings &s) {
  j = nlohmann::json{{"target_capacity", s.target_capacity}, {"window_size", s.window_size}};
}

void from_json(const nlohmann::json &j, SlidingAverageSettings &s) {
  j.at("target_capacity").get_to(s.target_capacity);
  j.at("window_size").get_to(s.window_size);
}

namespace {

__global__ void f32_add_avg_kernel(const float *idata, float *odata, int nx, int ny, int avg_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  odata[idx] += idata[idx] / avg_size;
}

__global__ void f32_sub_avg_kernel(const float *idata, float *odata, int nx, int ny, int avg_size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) {
    return;
  }

  int idx = y * nx + x;
  odata[idx] -= idata[idx] / avg_size;
}

} // namespace

SlidingAverage::SlidingAverage(SlidingAverageSettings settings, const holoflow::core::TDesc &idesc,
                               holoflow::core::TDesc &odesc, cudaStream_t producer_stream,
                               cudaStream_t consumer_stream, size_t nb_slots, size_t element_size,
                               DevPtr<std::byte> &&d_buffer, DevPtr<float> &&d_running_avg)
    : settings_(std::move(settings)), idesc_(idesc), odesc_(odesc),
      producer_stream_(producer_stream), consumer_stream_(consumer_stream), nb_slots_(nb_slots),
      element_size_(element_size), d_buffer_(std::move(d_buffer)),
      d_running_avg_(std::move(d_running_avg)), avg_idx_(nb_slots - settings.window_size),
      write_idx_(0), read_idx_(nb_slots - 1) {}

int SlidingAverage::writer_size() const {
  int write_idx = write_idx_.load(std::memory_order_relaxed);
  int read_idx  = read_idx_.load(std::memory_order_acquire);
  int diff      = write_idx - read_idx;
  if (write_idx < read_idx) {
    diff += nb_slots_;
  }
  return diff;
}

int SlidingAverage::reader_size() const {
  int write_idx = write_idx_.load(std::memory_order_acquire);
  int read_idx  = read_idx_.load(std::memory_order_relaxed);
  int diff      = write_idx - read_idx;
  if (write_idx < read_idx) {
    diff += nb_slots_;
  }
  return diff;
}

std::optional<holoflow::core::TView> SlidingAverage::acquire_input(int index) {
  if (index != 0) {
    throw std::out_of_range("SlidingAverage::acquire_input: invalid index");
  }

  if (nb_slots_ - writer_size() <= 1) {
    return std::nullopt;
  }

  int        write_idx = write_idx_.load(std::memory_order_relaxed);
  std::byte *data      = d_buffer_.get() + write_idx * element_size_;
  return holoflow::core::TView{
      .data = data,
      .desc = idesc_,
  };
}

void SlidingAverage::release_output(int index) {
  if (index != 0) {
    throw std::out_of_range("SlidingAverage::release_output: invalid index");
  }

  int read_idx      = read_idx_.load(std::memory_order_relaxed);
  int next_read_idx = read_idx + 1;
  if (next_read_idx == nb_slots_) {
    next_read_idx = 0;
  }
  read_idx_.store(next_read_idx, std::memory_order_release);
}

holoflow::core::OpResult SlidingAverage::try_push(holoflow::core::AsyncPushCtx &) {
  // Compute offsets
  size_t   write_idx  = write_idx_.load(std::memory_order_relaxed);
  uint8_t *write_data = reinterpret_cast<uint8_t *>(d_buffer_.get()) + write_idx * element_size_;
  size_t   avg_idx    = avg_idx_.load(std::memory_order_relaxed);
  uint8_t *avg_data   = reinterpret_cast<uint8_t *>(d_buffer_.get()) + avg_idx * element_size_;
  logger()->trace("[SlidingAverage::try_push] write_idx={} avg_idx={}", write_idx, avg_idx);

  size_t ny = idesc_.shape.at(1);
  size_t nx = idesc_.shape.at(2);

  // Cuda workload
  dim3 block_size(16, 16);
  dim3 grid_size((nx + block_size.x - 1) / block_size.x, (ny + block_size.y - 1) / block_size.y);

  f32_add_avg_kernel<<<grid_size, block_size, 0, producer_stream_>>>(
      reinterpret_cast<float *>(write_data), d_running_avg_.get(), nx, ny, settings_.window_size);

  f32_sub_avg_kernel<<<grid_size, block_size, 0, producer_stream_>>>(
      reinterpret_cast<float *>(avg_data), d_running_avg_.get(), nx, ny, settings_.window_size);

  CUDA_CHECK(cudaMemcpyAsync(avg_data, d_running_avg_.get(), element_size_,
                             cudaMemcpyDeviceToDevice, producer_stream_));

  CUDA_CHECK(cudaStreamSynchronize(producer_stream_));

  // Update indices
  size_t next_avg_idx = avg_idx + 1;
  if (next_avg_idx == nb_slots_)
    next_avg_idx = 0;

  avg_idx_.store(next_avg_idx, std::memory_order_release);

  size_t next_write_idx = write_idx + 1;
  if (next_write_idx == nb_slots_)
    next_write_idx = 0;

  write_idx_.store(next_write_idx, std::memory_order_release);

  // Success
  return holoflow::core::OpResult::Ok;
}

holoflow::core::OpResult SlidingAverage::try_pop(holoflow::core::AsyncPopCtx &ctx) {
  if (reader_size() <= settings_.window_size) {
    return holoflow::core::OpResult::NotReady;
  }

  int        read_idx = read_idx_.load(std::memory_order_relaxed);
  std::byte *data     = d_buffer_.get() + read_idx * element_size_;
  ctx.outputs[0]      = holoflow::core::TView{
           .data = data,
           .desc = odesc_,
  };
  logger()->trace("[SlidingAverage::try_pop] read_idx={}", read_idx);
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
  auto task = new SlidingAverage(settings, idesc, result.output_descs[0], ctx.producer_stream,
                                 ctx.consumer_stream, nb_slots, element_size, std::move(d_buffer),
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

} // namespace holotask::asyncs