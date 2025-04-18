#include "curaii/v2/cuda.hh"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>
#include <string>

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

} // namespace curaii::cuda

namespace curaii::cuda::detail {

inline void log_cuda_failure(spdlog::level::level_enum lvl, cudaError_t code,
                             const char *expr, const char *file, int line) {
  logger()->log(lvl, "CUDA error {} ({}): \"{}\"  [{}:{}]",
                cudaGetErrorString(code), // e.g. "invalid argument"
                static_cast<int>(code),   // numeric code
                expr,                     // the failed expression
                file, line);              // location
}

} // namespace curaii::cuda::detail
