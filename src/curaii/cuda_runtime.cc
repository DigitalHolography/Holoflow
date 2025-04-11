#include "curaii/cuda_runtime.hh"

#include <cuda_runtime.h>
#include <fmt/base.h>
#include <fmt/format.h>
#include <tl/expected.hpp>

#include "curaii/logger.hh"

#define UNREACHABLE(msg)                                                       \
  do {                                                                         \
    dh::curaii_logger()->critical("Unreachable code reached at {}:{} - {}",    \
                                  __FILE__, __LINE__, msg);                    \
    std::abort();                                                              \
  } while (0)

// ==========================================================================
//                     CudaError Implementation
// ==========================================================================

dh::CudaError::CudaError(cudaError_t error) noexcept : error_(error) {}

const char *dh::CudaError::message() const noexcept {
  return cudaGetErrorString(error_);
}

cudaError_t dh::CudaError::error() const noexcept { return error_; }

auto fmt::formatter<dh::CudaError>::format(dh::CudaError error,
                                           format_context &ctx) const
    -> format_context::iterator {
  string_view name;
  switch (error.error()) {
  case cudaSuccess:
    name = "cudaSuccess";
    break;
  case cudaErrorInvalidValue:
    name = "cudaErrorInvalidValue";
    break;
  case cudaErrorMemoryAllocation:
    name = "cudaErrorMemoryAllocation";
    break;
  case cudaErrorInitializationError:
    name = "cudaErrorInitializationError";
    break;
  case cudaErrorCudartUnloading:
    name = "cudaErrorCudartUnloading";
    break;
  case cudaErrorProfilerDisabled:
    name = "cudaErrorProfilerDisabled";
    break;
  case cudaErrorProfilerNotInitialized:
    name = "cudaErrorProfilerNotInitialized";
    break;
  case cudaErrorProfilerAlreadyStarted:
    name = "cudaErrorProfilerAlreadyStarted";
    break;
  case cudaErrorProfilerAlreadyStopped:
    name = "cudaErrorProfilerAlreadyStopped";
    break;
  case cudaErrorInvalidConfiguration:
    name = "cudaErrorInvalidConfiguration";
    break;
  case cudaErrorInvalidPitchValue:
    name = "cudaErrorInvalidPitchValue";
    break;
  case cudaErrorInvalidSymbol:
    name = "cudaErrorInvalidSymbol";
    break;
  case cudaErrorInvalidHostPointer:
    name = "cudaErrorInvalidHostPointer";
    break;
  case cudaErrorInvalidDevicePointer:
    name = "cudaErrorInvalidDevicePointer";
    break;
  case cudaErrorInvalidTexture:
    name = "cudaErrorInvalidTexture";
    break;
  case cudaErrorInvalidTextureBinding:
    name = "cudaErrorInvalidTextureBinding";
    break;
  case cudaErrorInvalidChannelDescriptor:
    name = "cudaErrorInvalidChannelDescriptor";
    break;
  case cudaErrorInvalidMemcpyDirection:
    name = "cudaErrorInvalidMemcpyDirection";
    break;
  case cudaErrorAddressOfConstant:
    name = "cudaErrorAddressOfConstant";
    break;
  case cudaErrorTextureFetchFailed:
    name = "cudaErrorTextureFetchFailed";
    break;
  case cudaErrorTextureNotBound:
    name = "cudaErrorTextureNotBound";
    break;
  case cudaErrorSynchronizationError:
    name = "cudaErrorSynchronizationError";
    break;
  case cudaErrorInvalidFilterSetting:
    name = "cudaErrorInvalidFilterSetting";
    break;
  case cudaErrorInvalidNormSetting:
    name = "cudaErrorInvalidNormSetting";
    break;
  case cudaErrorMixedDeviceExecution:
    name = "cudaErrorMixedDeviceExecution";
    break;
  case cudaErrorNotYetImplemented:
    name = "cudaErrorNotYetImplemented";
    break;
  case cudaErrorMemoryValueTooLarge:
    name = "cudaErrorMemoryValueTooLarge";
    break;
  case cudaErrorStubLibrary:
    name = "cudaErrorStubLibrary";
    break;
  case cudaErrorInsufficientDriver:
    name = "cudaErrorInsufficientDriver";
    break;
  case cudaErrorCallRequiresNewerDriver:
    name = "cudaErrorCallRequiresNewerDriver";
    break;
  case cudaErrorInvalidSurface:
    name = "cudaErrorInvalidSurface";
    break;
  case cudaErrorDuplicateVariableName:
    name = "cudaErrorDuplicateVariableName";
    break;
  case cudaErrorDuplicateTextureName:
    name = "cudaErrorDuplicateTextureName";
    break;
  case cudaErrorDuplicateSurfaceName:
    name = "cudaErrorDuplicateSurfaceName";
    break;
  case cudaErrorDevicesUnavailable:
    name = "cudaErrorDevicesUnavailable";
    break;
  case cudaErrorIncompatibleDriverContext:
    name = "cudaErrorIncompatibleDriverContext";
    break;
  case cudaErrorMissingConfiguration:
    name = "cudaErrorMissingConfiguration";
    break;
  case cudaErrorPriorLaunchFailure:
    name = "cudaErrorPriorLaunchFailure";
    break;
  case cudaErrorLaunchMaxDepthExceeded:
    name = "cudaErrorLaunchMaxDepthExceeded";
    break;
  case cudaErrorLaunchFileScopedTex:
    name = "cudaErrorLaunchFileScopedTex";
    break;
  case cudaErrorLaunchFileScopedSurf:
    name = "cudaErrorLaunchFileScopedSurf";
    break;
  case cudaErrorSyncDepthExceeded:
    name = "cudaErrorSyncDepthExceeded";
    break;
  case cudaErrorLaunchPendingCountExceeded:
    name = "cudaErrorLaunchPendingCountExceeded";
    break;
  case cudaErrorInvalidDeviceFunction:
    name = "cudaErrorInvalidDeviceFunction";
    break;
  case cudaErrorNoDevice:
    name = "cudaErrorNoDevice";
    break;
  case cudaErrorInvalidDevice:
    name = "cudaErrorInvalidDevice";
    break;
  case cudaErrorDeviceNotLicensed:
    name = "cudaErrorDeviceNotLicensed";
    break;
  case cudaErrorSoftwareValidityNotEstablished:
    name = "cudaErrorSoftwareValidityNotEstablished";
    break;
  case cudaErrorStartupFailure:
    name = "cudaErrorStartupFailure";
    break;
  case cudaErrorInvalidKernelImage:
    name = "cudaErrorInvalidKernelImage";
    break;
  case cudaErrorDeviceUninitialized:
    name = "cudaErrorDeviceUninitialized";
    break;
  case cudaErrorMapBufferObjectFailed:
    name = "cudaErrorMapBufferObjectFailed";
    break;
  case cudaErrorUnmapBufferObjectFailed:
    name = "cudaErrorUnmapBufferObjectFailed";
    break;
  case cudaErrorArrayIsMapped:
    name = "cudaErrorArrayIsMapped";
    break;
  case cudaErrorAlreadyMapped:
    name = "cudaErrorAlreadyMapped";
    break;
  case cudaErrorNoKernelImageForDevice:
    name = "cudaErrorNoKernelImageForDevice";
    break;
  case cudaErrorAlreadyAcquired:
    name = "cudaErrorAlreadyAcquired";
    break;
  case cudaErrorNotMapped:
    name = "cudaErrorNotMapped";
    break;
  case cudaErrorNotMappedAsArray:
    name = "cudaErrorNotMappedAsArray";
    break;
  case cudaErrorNotMappedAsPointer:
    name = "cudaErrorNotMappedAsPointer";
    break;
  case cudaErrorECCUncorrectable:
    name = "cudaErrorECCUncorrectable";
    break;
  case cudaErrorUnsupportedLimit:
    name = "cudaErrorUnsupportedLimit";
    break;
  case cudaErrorDeviceAlreadyInUse:
    name = "cudaErrorDeviceAlreadyInUse";
    break;
  case cudaErrorPeerAccessUnsupported:
    name = "cudaErrorPeerAccessUnsupported";
    break;
  case cudaErrorInvalidPtx:
    name = "cudaErrorInvalidPtx";
    break;
  case cudaErrorInvalidGraphicsContext:
    name = "cudaErrorInvalidGraphicsContext";
    break;
  case cudaErrorNvlinkUncorrectable:
    name = "cudaErrorNvlinkUncorrectable";
    break;
  case cudaErrorJitCompilerNotFound:
    name = "cudaErrorJitCompilerNotFound";
    break;
  case cudaErrorUnsupportedPtxVersion:
    name = "cudaErrorUnsupportedPtxVersion";
    break;
  case cudaErrorJitCompilationDisabled:
    name = "cudaErrorJitCompilationDisabled";
    break;
  case cudaErrorUnsupportedExecAffinity:
    name = "cudaErrorUnsupportedExecAffinity";
    break;
  case cudaErrorUnsupportedDevSideSync:
    name = "cudaErrorUnsupportedDevSideSync";
    break;
  case cudaErrorContained:
    name = "cudaErrorContained";
    break;
  case cudaErrorInvalidSource:
    name = "cudaErrorInvalidSource";
    break;
  case cudaErrorFileNotFound:
    name = "cudaErrorFileNotFound";
    break;
  case cudaErrorSharedObjectSymbolNotFound:
    name = "cudaErrorSharedObjectSymbolNotFound";
    break;
  case cudaErrorSharedObjectInitFailed:
    name = "cudaErrorSharedObjectInitFailed";
    break;
  case cudaErrorOperatingSystem:
    name = "cudaErrorOperatingSystem";
    break;
  case cudaErrorInvalidResourceHandle:
    name = "cudaErrorInvalidResourceHandle";
    break;
  case cudaErrorIllegalState:
    name = "cudaErrorIllegalState";
    break;
  case cudaErrorLossyQuery:
    name = "cudaErrorLossyQuery";
    break;
  case cudaErrorSymbolNotFound:
    name = "cudaErrorSymbolNotFound";
    break;
  case cudaErrorNotReady:
    name = "cudaErrorNotReady";
    break;
  case cudaErrorIllegalAddress:
    name = "cudaErrorIllegalAddress";
    break;
  case cudaErrorLaunchOutOfResources:
    name = "cudaErrorLaunchOutOfResources";
    break;
  case cudaErrorLaunchTimeout:
    name = "cudaErrorLaunchTimeout";
    break;
  case cudaErrorLaunchIncompatibleTexturing:
    name = "cudaErrorLaunchIncompatibleTexturing";
    break;
  case cudaErrorPeerAccessAlreadyEnabled:
    name = "cudaErrorPeerAccessAlreadyEnabled";
    break;
  case cudaErrorPeerAccessNotEnabled:
    name = "cudaErrorPeerAccessNotEnabled";
    break;
  case cudaErrorSetOnActiveProcess:
    name = "cudaErrorSetOnActiveProcess";
    break;
  case cudaErrorContextIsDestroyed:
    name = "cudaErrorContextIsDestroyed";
    break;
  case cudaErrorAssert:
    name = "cudaErrorAssert";
    break;
  case cudaErrorTooManyPeers:
    name = "cudaErrorTooManyPeers";
    break;
  case cudaErrorHostMemoryAlreadyRegistered:
    name = "cudaErrorHostMemoryAlreadyRegistered";
    break;
  case cudaErrorHostMemoryNotRegistered:
    name = "cudaErrorHostMemoryNotRegistered";
    break;
  case cudaErrorHardwareStackError:
    name = "cudaErrorHardwareStackError";
    break;
  case cudaErrorIllegalInstruction:
    name = "cudaErrorIllegalInstruction";
    break;
  case cudaErrorMisalignedAddress:
    name = "cudaErrorMisalignedAddress";
    break;
  case cudaErrorInvalidAddressSpace:
    name = "cudaErrorInvalidAddressSpace";
    break;
  case cudaErrorInvalidPc:
    name = "cudaErrorInvalidPc";
    break;
  case cudaErrorLaunchFailure:
    name = "cudaErrorLaunchFailure";
    break;
  case cudaErrorCooperativeLaunchTooLarge:
    name = "cudaErrorCooperativeLaunchTooLarge";
    break;
  case cudaErrorTensorMemoryLeak:
    name = "cudaErrorTensorMemoryLeak";
    break;
  case cudaErrorNotPermitted:
    name = "cudaErrorNotPermitted";
    break;
  case cudaErrorNotSupported:
    name = "cudaErrorNotSupported";
    break;
  case cudaErrorSystemNotReady:
    name = "cudaErrorSystemNotReady";
    break;
  case cudaErrorSystemDriverMismatch:
    name = "cudaErrorSystemDriverMismatch";
    break;
  case cudaErrorCompatNotSupportedOnDevice:
    name = "cudaErrorCompatNotSupportedOnDevice";
    break;
  case cudaErrorMpsConnectionFailed:
    name = "cudaErrorMpsConnectionFailed";
    break;
  case cudaErrorMpsRpcFailure:
    name = "cudaErrorMpsRpcFailure";
    break;
  case cudaErrorMpsServerNotReady:
    name = "cudaErrorMpsServerNotReady";
    break;
  case cudaErrorMpsMaxClientsReached:
    name = "cudaErrorMpsMaxClientsReached";
    break;
  case cudaErrorMpsMaxConnectionsReached:
    name = "cudaErrorMpsMaxConnectionsReached";
    break;
  case cudaErrorMpsClientTerminated:
    name = "cudaErrorMpsClientTerminated";
    break;
  case cudaErrorCdpNotSupported:
    name = "cudaErrorCdpNotSupported";
    break;
  case cudaErrorCdpVersionMismatch:
    name = "cudaErrorCdpVersionMismatch";
    break;
  case cudaErrorStreamCaptureUnsupported:
    name = "cudaErrorStreamCaptureUnsupported";
    break;
  case cudaErrorStreamCaptureInvalidated:
    name = "cudaErrorStreamCaptureInvalidated";
    break;
  case cudaErrorStreamCaptureMerge:
    name = "cudaErrorStreamCaptureMerge";
    break;
  case cudaErrorStreamCaptureUnmatched:
    name = "cudaErrorStreamCaptureUnmatched";
    break;
  case cudaErrorStreamCaptureUnjoined:
    name = "cudaErrorStreamCaptureUnjoined";
    break;
  case cudaErrorStreamCaptureIsolation:
    name = "cudaErrorStreamCaptureIsolation";
    break;
  case cudaErrorStreamCaptureImplicit:
    name = "cudaErrorStreamCaptureImplicit";
    break;
  case cudaErrorCapturedEvent:
    name = "cudaErrorCapturedEvent";
    break;
  case cudaErrorStreamCaptureWrongThread:
    name = "cudaErrorStreamCaptureWrongThread";
    break;
  case cudaErrorTimeout:
    name = "cudaErrorTimeout";
    break;
  case cudaErrorGraphExecUpdateFailure:
    name = "cudaErrorGraphExecUpdateFailure";
    break;
  case cudaErrorExternalDevice:
    name = "cudaErrorExternalDevice";
    break;
  case cudaErrorInvalidClusterSize:
    name = "cudaErrorInvalidClusterSize";
    break;
  case cudaErrorFunctionNotLoaded:
    name = "cudaErrorFunctionNotLoaded";
    break;
  case cudaErrorInvalidResourceType:
    name = "cudaErrorInvalidResourceType";
    break;
  case cudaErrorInvalidResourceConfiguration:
    name = "cudaErrorInvalidResourceConfiguration";
    break;
  case cudaErrorUnknown:
    name = "cudaErrorUnknown";
    break;
  case cudaErrorApiFailureBase:
    name = "cudaErrorApiFailureBase";
    break;
  default:
    UNREACHABLE("Invalid cuda error");
  }
  return fmt::format_to(ctx.out(), "{}: {}", name, error.message());
}

// ==========================================================================
//                     CudaStreamFlags Implementation
// ==========================================================================

dh::CudaStreamFlags::CudaStreamFlags(unsigned int flags) noexcept
    : flags_(flags) {}

unsigned int dh::CudaStreamFlags::flags() const noexcept { return flags_; }

auto fmt::formatter<dh::CudaStreamFlags>::format(dh::CudaStreamFlags flags,
                                                 format_context &ctx) const
    -> format_context::iterator {
  std::string_view name;
  switch (flags.flags()) {
  case cudaStreamDefault:
    name = "cudaStreamDefault";
    break;
  case cudaStreamNonBlocking:
    name = "cudaStreamNonBlocking";
    break;
  default:
    UNREACHABLE("Invalid cuda stream flags");
  }
  return formatter<string_view>::format(name, ctx);
}

// ==========================================================================
//                     CudaStreamRef Implementation
// ==========================================================================

dh::CudaStreamRef::CudaStreamRef(cudaStream_t stream) noexcept
    : stream_(stream) {}

dh::CudaStreamRef dh::CudaStreamRef::from_raw(cudaStream_t stream) noexcept {
  return CudaStreamRef(stream);
}

tl::expected<void, dh::CudaError>
dh::CudaStreamRef::try_synchronize() const noexcept {
  if (auto error = cudaStreamSynchronize(stream_); error != cudaSuccess) {
    curaii_logger()->warn(
        "[CudaStreamRef::try_synchronize] failed with error: \"{}\"",
        CudaError(error));
  }

  return {};
}

cudaStream_t dh::CudaStreamRef::stream() const noexcept { return stream_; }

// ==========================================================================
//                     CudaStream Implementation
// ==========================================================================

dh::CudaStream::CudaStream(cudaStream_t stream) noexcept : stream_(stream) {}

dh::CudaStream::CudaStream(CudaStream &&other) noexcept
    : stream_(other.stream_) {
  other.stream_ = 0;
}

dh::CudaStream &dh::CudaStream::operator=(CudaStream &&other) noexcept {
  if (this != &other) {
    if (stream_) {
      if (auto error = cudaStreamDestroy(stream_); error != cudaSuccess) {
        curaii_logger()->warn(
            "[CudaStream::operator=] failed with error: \"{}\"",
            CudaError(error));
      }
    }
    stream_ = other.stream_;
    other.stream_ = 0;
  }
  return *this;
}

dh::CudaStream::~CudaStream() {
  if (stream_) {
    if (auto error = cudaStreamSynchronize(stream_); error != cudaSuccess) {
      curaii_logger()->warn("[CudaStream::operator=] failed with error: \"{}\"",
                            CudaError(error));
    }

    if (auto error = cudaStreamDestroy(stream_); error != cudaSuccess) {
      curaii_logger()->warn("[CudaStream::operator=] failed with error: \"{}\"",
                            CudaError(error));
    }
  }
}

tl::expected<dh::CudaStream, dh::CudaError>
dh::CudaStream::try_create() noexcept {
  cudaStream_t stream;
  if (auto error = cudaStreamCreate(&stream); error != cudaSuccess) {
    curaii_logger()->warn("[CudaStream::try_create] failed with error: \"{}\"",
                          CudaError(error));
  }

  return CudaStream(stream);
}

tl::expected<void, dh::CudaError>
dh::CudaStream::try_synchronize() const noexcept {
  if (auto error = cudaStreamSynchronize(stream_); error != cudaSuccess) {
    curaii_logger()->warn(
        "[CudaStream::try_synchronize] failed with error: \"{}\"",
        CudaError(error));
  }

  return {};
}

dh::CudaStreamRef dh::CudaStream::ref() noexcept {
  return CudaStreamRef(stream_);
}

cudaStream_t dh::CudaStream::stream() const noexcept { return stream_; }