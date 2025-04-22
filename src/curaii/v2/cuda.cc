#include "curaii/v2/cuda.hh"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include "curaii/v2/logger.hh"

namespace curaii::cuda {

Error::Error(cudaError_t code, const char *what, const char *file, int line)
    : std::runtime_error(Error::make_message(code, what, file, line)),
      code_(code) {}

std::string Error::make_message(cudaError_t code, const char *what,
                                const char *file, int line) {
  std::ostringstream os;
  os << "CUDA error: " << cudaGetErrorString(code) << " ("
     << static_cast<int>(code) << ")\n"
     << "  expression : " << what << '\n'
     << "  location   : " << file << ':' << line;
  return os.str();
}

cudaError_t Error::code() const noexcept { return code_; }

void HostDeleter::operator()(void *ptr) const noexcept {
  if (ptr) {
    CUDA_CHECK_NT(cudaFreeHost(ptr));
  }
}

DeviceDeleter::DeviceDeleter(cudaStream_t s) noexcept : stream(s) {}

void DeviceDeleter::operator()(void *ptr) const noexcept {
  if (!ptr)
    return;

  if (stream) {
    CUDA_CHECK_NT(cudaFreeAsync(ptr, stream));
  } else {
    CUDA_CHECK_NT(cudaFree(ptr));
  }
}

Stream::Stream(unsigned flags, int priority) {
  CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, flags, priority));
}

Stream::Stream(Stream &&other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

Stream &Stream::operator=(Stream &&other) noexcept {
  if (this != &other) {
    reset();
    stream_ = other.stream_;
    other.stream_ = nullptr;
  }
  return *this;
}

Stream::~Stream() noexcept {
  if (stream_) {
    CUDA_CHECK_NT(cudaStreamDestroy(stream_));
  }
}

cudaStream_t Stream::get() const noexcept { return stream_; }

cudaStream_t Stream::release() noexcept {
  auto tmp = stream_;
  stream_ = nullptr;
  return tmp;
}

void Stream::reset(cudaStream_t s) noexcept {
  if (stream_) {
    CUDA_CHECK_NT(cudaStreamDestroy(stream_));
  }
  stream_ = s;
}

Stream::operator bool() const noexcept { return stream_ != nullptr; }

} // namespace curaii::cuda

namespace curaii::cuda::detail {

void log_cuda_failure(spdlog::level::level_enum lvl, cudaError_t code,
                      const char *expr, const char *file, int line) {
  logger()->log(lvl, "CUDA error {} ({}): \"{}\"  [{}:{}]",
                cudaGetErrorString(code), // e.g. "invalid argument"
                static_cast<int>(code),   // numeric code
                expr,                     // the failed expression
                file, line);              // location
}

} // namespace curaii::cuda::detail
