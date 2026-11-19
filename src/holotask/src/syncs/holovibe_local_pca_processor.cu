#include "holovibe_local_pca_processor.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include "logger.hh"

namespace holotask::syncs {
std::vector<char> compile_local_pca_heev_cubin(int frames, int architecture);
}

namespace
{
void check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("Local PCA CUDA error: ") + cudaGetErrorString(status));
}
void check(cublasStatus_t status)
{
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("Local PCA cuBLAS error: " + std::to_string(static_cast<int>(status)));
}
void check(cusolverStatus_t status)
{
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("Local PCA cuSOLVER error: " + std::to_string(static_cast<int>(status)));
}

template <typename T>
struct DeviceBuffer
{
    T* ptr = nullptr;
    ~DeviceBuffer() { cudaFree(ptr); }
    void allocate(size_t count) { check(cudaMalloc(reinterpret_cast<void**>(&ptr), count * sizeof(T))); }
};

struct BlasHandle
{
    cublasHandle_t value = nullptr;
    BlasHandle() { check(cublasCreate(&value)); }
    ~BlasHandle() { cublasDestroy(value); }
};
struct SolverHandle
{
    cusolverDnHandle_t value = nullptr;
    SolverHandle() { check(cusolverDnCreate(&value)); }
    ~SolverHandle() { cusolverDnDestroy(value); }
};

unsigned grid_size(size_t count) { return static_cast<unsigned>(std::min<size_t>((count + 255) / 256, 65535)); }

// Gather one width_x * width_y * frames tile. Modulo indexing implements periodic
// spatial borders, including corners and tiles larger than the image itself.
__global__ void gather_tile(const cuComplex* input,
                            cuComplex* tile,
                            size_t count,
                            unsigned width,
                            unsigned height,
                            unsigned width_x,
                            size_t tile_pixels,
                            unsigned origin_x,
                            unsigned origin_y)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const size_t local = i % tile_pixels;
        const unsigned x = (origin_x + local % width_x) % width;
        const unsigned y = (origin_y + local / width_x) % height;
        tile[i] = input[(i / tile_pixels) * pixels + static_cast<size_t>(y) * width + x];
    }
}

// cuSOLVER orders the eigenvalues increasingly. Compute the Shannon effective
// rank of the Gram matrix, e = exp(-sum(p * log(p))), p = lambda/sum(lambda),
// and choose r = min(frames, ceil(rank_factor * e)). Scale BEFORE rounding so
// small factor adjustments can move the cutoff even when e is not an integer.
// Then zero the LAST r eigenvectors to remove the strongest components. A zero
// spectrum has rank zero; negative roundoff eigenvalues carry no probability.
__global__ void remove_effective_rank(cuComplex* vectors,
                                      const float* eigenvalues,
                                      unsigned frames,
                                      const int* solver_info,
                                      int* error,
                                      float rank_factor)
{
    const size_t batch = blockIdx.x;
    vectors += batch * static_cast<size_t>(frames) * frames;
    eigenvalues += batch * static_cast<size_t>(frames);
    solver_info += batch;
    __shared__ unsigned removed;
    if (threadIdx.x == 0)
    {
        removed = 0;
        double total = 0.;
        for (unsigned i = 0; i < frames; ++i)
        {
            if (!isfinite(eigenvalues[i]))
                atomicCAS(error, 0, -1);
            total += fmax(0., static_cast<double>(eigenvalues[i]));
        }
        if (*solver_info != 0)
            atomicCAS(error, 0, *solver_info);
        if (total > 0. && rank_factor > 0.f)
        {
            double entropy = 0.;
            for (unsigned i = 0; i < frames; ++i)
            {
                const double p = fmax(0., static_cast<double>(eigenvalues[i])) / total;
                if (p > 0.)
                    entropy -= p * log(p);
            }
            // Avoid rounding a numerically perturbed integer rank up by one.
            const double scaled_rank = static_cast<double>(rank_factor) * exp(entropy);
            removed = static_cast<unsigned>(fmax(1., fmin(static_cast<double>(frames), ceil(scaled_rank - 1e-6))));
        }
    }
    __syncthreads();

    const size_t first_removed = static_cast<size_t>(frames - removed) * frames;
    const size_t size = static_cast<size_t>(frames) * frames;
    for (size_t i = first_removed + threadIdx.x; i < size; i += blockDim.x)
        vectors[i] = make_cuComplex(0.f, 0.f);
}

// Scatter residuals to their original pixel/time coordinates. Count every spatial
// contribution once (at t=0). The count can vary at wrapped borders when an image
// dimension is not divisible by the stride, so a fixed overlap factor is incorrect.
__global__ void accumulate_tile(const cuComplex* tile,
                                cuComplex* output,
                                unsigned* coverage,
                                size_t count,
                                unsigned width,
                                unsigned height,
                                unsigned width_x,
                                size_t tile_pixels,
                                unsigned origin_x,
                                unsigned origin_y)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const size_t local = i % tile_pixels;
        const unsigned x = (origin_x + local % width_x) % width;
        const unsigned y = (origin_y + local / width_x) % height;
        const size_t pixel = static_cast<size_t>(y) * width + x;
        const size_t target = (i / tile_pixels) * pixels + pixel;
        atomicAdd(&output[target].x, tile[i].x);
        atomicAdd(&output[target].y, tile[i].y);
        if (i < tile_pixels)
            atomicAdd(coverage + pixel, 1u);
    }
}

__global__ void average_overlaps(cuComplex* output, const unsigned* coverage, size_t pixels, size_t count)
{
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
    {
        const unsigned weight = coverage[i % pixels];
        output[i] = weight ? make_cuComplex(output[i].x / weight, output[i].y / weight) : make_cuComplex(0.f, 0.f);
    }
}
} // namespace

namespace holovibes::compute
{
struct LocalPca::Impl
{
    unsigned width, height, frames, width_x, width_y, stride_x, stride_y;
    size_t pixels, tile_pixels;
    unsigned batch_capacity;
    cudaStream_t stream;
    BlasHandle blas;
    std::unique_ptr<SolverHandle> solver;
    DeviceBuffer<cuComplex> tile, scores, covariance, workspace;
    DeviceBuffer<float> eigenvalues;
    DeviceBuffer<unsigned> coverage;
    DeviceBuffer<int> solver_info, error;
    int workspace_size = 0;
    CUmodule dx_module = nullptr;
    CUfunction dx_kernel = nullptr;
    unsigned dx_block_x = 0, dx_shared_memory = 0;

    ~Impl() { if (dx_module) cuModuleUnload(dx_module); }

    void solve(cuComplex* matrix, float* values, int* info, unsigned batches)
    {
        if (dx_kernel)
        {
            CUdeviceptr matrix_arg = reinterpret_cast<CUdeviceptr>(matrix);
            CUdeviceptr values_arg = reinterpret_cast<CUdeviceptr>(values);
            CUdeviceptr info_arg = reinterpret_cast<CUdeviceptr>(info);
            void* args[] = {&matrix_arg, &values_arg, &info_arg};
            const auto result = cuLaunchKernel(dx_kernel, batches, 1, 1, dx_block_x, 1, 1,
                                               dx_shared_memory, reinterpret_cast<CUstream>(stream),
                                               args, nullptr);
            if (result != CUDA_SUCCESS)
                throw std::runtime_error("Local PCA cuSolverDx launch failed: " + std::to_string(result));
        }
        else
            for (unsigned batch = 0; batch < batches; ++batch)
                check(cusolverDnCheevd(solver->value, CUSOLVER_EIG_MODE_VECTOR,
                                       CUBLAS_FILL_MODE_LOWER, frames,
                                       matrix + static_cast<size_t>(batch) * frames * frames, frames,
                                       values + static_cast<size_t>(batch) * frames,
                                       workspace.ptr, workspace_size, info + batch));
    }

    Impl(unsigned w, unsigned h, unsigned n, unsigned wx, unsigned wy, unsigned sx, unsigned sy, cudaStream_t s)
        : width(w)
        , height(h)
        , frames(n)
        , width_x(wx)
        , width_y(wy)
        , stride_x(sx)
        , stride_y(sy)
        , pixels(static_cast<size_t>(w) * h)
        , tile_pixels(static_cast<size_t>(wx) * wy)
        , batch_capacity(1)
        , stream(s)
    {
        if (!w || !h || !n || n > 65535 || !wx || !wy || wx > 65535 || wy > 65535 || !sx || !sy || sx > wx || sy > wy ||
            tile_pixels > std::numeric_limits<int>::max())
            throw std::invalid_argument("Local PCA needs positive dimensions, 1..65535 frames/tile sides, "
                                        "and strides no larger than their tile sides");

        check(cublasSetStream(blas.value, stream));
        check(cublasSetMathMode(blas.value, CUBLAS_PEDANTIC_MATH));

        // Allocate separate scratch for every spatial tile so one launch can solve them all.
        const size_t tile_count = ((static_cast<size_t>(w) + sx - 1) / sx) *
                                  ((static_cast<size_t>(h) + sy - 1) / sy);
        if (tile_count > std::numeric_limits<unsigned>::max())
            throw std::invalid_argument("Local PCA tile count exceeds CUDA grid capacity");
        batch_capacity = static_cast<unsigned>(tile_count);
        tile.allocate(tile_pixels * frames * batch_capacity);
        scores.allocate(tile_pixels * frames * batch_capacity);
        covariance.allocate(static_cast<size_t>(frames) * frames * batch_capacity);
        eigenvalues.allocate(static_cast<size_t>(frames) * batch_capacity);
        coverage.allocate(pixels);
        solver_info.allocate(batch_capacity);
        error.allocate(1);
        try
        {
            int device = 0, major = 0, minor = 0, max_shared = 0;
            check(cudaGetDevice(&device));
            check(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
            check(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
            check(cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
            const auto cubin = holotask::syncs::compile_local_pca_heev_cubin(
                static_cast<int>(frames), major * 10 + minor);
            if (cuModuleLoadDataEx(&dx_module, cubin.data(), 0, nullptr, nullptr) != CUDA_SUCCESS ||
                cuModuleGetFunction(&dx_kernel, dx_module, "local_pca_heev") != CUDA_SUCCESS)
                throw std::runtime_error("could not load cuSolverDx kernel");
            const auto read_constant = [&](const char* name) {
                CUdeviceptr address = 0;
                size_t bytes = 0;
                unsigned value = 0;
                if (cuModuleGetGlobal(&address, &bytes, dx_module, name) != CUDA_SUCCESS ||
                    bytes != sizeof(value) ||
                    cuMemcpyDtoH(&value, address, sizeof(value)) != CUDA_SUCCESS)
                    throw std::runtime_error("could not read cuSolverDx kernel configuration");
                return value;
            };
            dx_block_x = read_constant("local_pca_block_x");
            dx_shared_memory = read_constant("local_pca_shared_memory");
            if (dx_shared_memory > static_cast<unsigned>(max_shared) ||
                cuFuncSetAttribute(dx_kernel, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                   static_cast<int>(dx_shared_memory)) != CUDA_SUCCESS)
                throw std::runtime_error("cuSolverDx shared memory exceeds device limit");
            holotask::logger()->info("[HolovibeLocalPca] cuSolverDx enabled for {} frames", frames);
        }
        catch (const std::exception& e)
        {
            holotask::logger()->warn("[HolovibeLocalPca] cuSolverDx unavailable for {} frames: {}; using cuSOLVER", frames, e.what());
            dx_kernel = nullptr;
            if (dx_module) { cuModuleUnload(dx_module); dx_module = nullptr; }
        }
        if (!dx_kernel)
        {
            solver = std::make_unique<SolverHandle>();
            check(cusolverDnSetStream(solver->value, stream));
            check(cusolverDnCheevd_bufferSize(solver->value, CUSOLVER_EIG_MODE_VECTOR,
                                              CUBLAS_FILL_MODE_LOWER, frames, covariance.ptr,
                                              frames, eigenvalues.ptr, &workspace_size));
            workspace.allocate(workspace_size);
        }
    }
};

LocalPca::LocalPca(unsigned width,
                   unsigned height,
                   unsigned frames,
                   unsigned width_x,
                   unsigned width_y,
                   unsigned stride_x,
                   unsigned stride_y,
                   cudaStream_t stream)
    : impl_(std::make_unique<Impl>(width, height, frames, width_x, width_y, stride_x, stride_y, stream))
{
}
LocalPca::~LocalPca() = default;

bool LocalPca::process(const cuComplex* input, cuComplex* output, float rank_factor,
                       const std::atomic<bool>* cancelled)
{
    if (!input || !output || input == output)
        throw std::invalid_argument("Local PCA requires distinct, nonnull input and output buffers");
    if (!std::isfinite(rank_factor) || rank_factor < 0.f)
        throw std::invalid_argument("Local PCA rank factor must be finite and nonnegative");
    auto& a = *impl_;
    if (cancelled && cancelled->load(std::memory_order_acquire)) return false;
    const cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
    const size_t tile_count = a.tile_pixels * a.frames, count = a.pixels * a.frames;
    // At factor zero, every local projector is the identity and overlap averaging
    // gives the original cube. Copy it directly to avoid needless eigensolves.
    if (rank_factor == 0.f)
    {
        check(cudaMemcpyAsync(output, input, count * sizeof(cuComplex), cudaMemcpyDeviceToDevice, a.stream));
        check(cudaStreamSynchronize(a.stream));
        return true;
    }
    const int m = static_cast<int>(a.tile_pixels), n = static_cast<int>(a.frames);
    check(cudaMemsetAsync(output, 0, count * sizeof(cuComplex), a.stream));
    check(cudaMemsetAsync(a.coverage.ptr, 0, a.pixels * sizeof(unsigned), a.stream));
    check(cudaMemsetAsync(a.error.ptr, 0, sizeof(int), a.stream));

    const size_t tiles_x = (static_cast<size_t>(a.width) + a.stride_x - 1) / a.stride_x;
    const size_t tiles_y = (static_cast<size_t>(a.height) + a.stride_y - 1) / a.stride_y;
    const size_t total_tiles = tiles_x * tiles_y;
    for (size_t first = 0; first < total_tiles; first += a.batch_capacity)
    {
        if (cancelled && cancelled->load(std::memory_order_acquire))
        {
            check(cudaStreamSynchronize(a.stream));
            return false;
        }
        const unsigned batches = static_cast<unsigned>(std::min<size_t>(a.batch_capacity, total_tiles - first));
        for (unsigned batch = 0; batch < batches; ++batch)
        {
            const size_t index = first + batch;
            const unsigned x = static_cast<unsigned>((index % tiles_x) * a.stride_x);
            const unsigned y = static_cast<unsigned>((index / tiles_x) * a.stride_y);
            auto* tile = a.tile.ptr + static_cast<size_t>(batch) * tile_count;
            auto* covariance = a.covariance.ptr + static_cast<size_t>(batch) * n * n;
            gather_tile<<<grid_size(tile_count), 256, 0, a.stream>>>(input, tile, tile_count,
                a.width, a.height, a.width_x, a.tile_pixels, x, y);
            check(cudaGetLastError());
            // Match classic PCA's uncentered complex covariance: C = H^H H.
            check(cublasCgemm(a.blas.value, CUBLAS_OP_C, CUBLAS_OP_N, n, n, m,
                              &one, tile, m, tile, m, &zero, covariance, n));
        }
        a.solve(a.covariance.ptr, a.eigenvalues.ptr, a.solver_info.ptr, batches);
        remove_effective_rank<<<batches, 256, 0, a.stream>>>(a.covariance.ptr,
            a.eigenvalues.ptr, a.frames, a.solver_info.ptr, a.error.ptr, rank_factor);
        check(cudaGetLastError());

        // Reconstruct each residual in its own temporal basis, then average overlaps.
        for (unsigned batch = 0; batch < batches; ++batch)
        {
            const size_t index = first + batch;
            const unsigned x = static_cast<unsigned>((index % tiles_x) * a.stride_x);
            const unsigned y = static_cast<unsigned>((index / tiles_x) * a.stride_y);
            auto* tile = a.tile.ptr + static_cast<size_t>(batch) * tile_count;
            auto* scores = a.scores.ptr + static_cast<size_t>(batch) * tile_count;
            auto* covariance = a.covariance.ptr + static_cast<size_t>(batch) * n * n;
            check(cublasCgemm(a.blas.value, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                              &one, tile, m, covariance, n, &zero, scores, m));
            check(cublasCgemm(a.blas.value, CUBLAS_OP_N, CUBLAS_OP_C, m, n, n,
                              &one, scores, m, covariance, n, &zero, tile, m));
            accumulate_tile<<<grid_size(tile_count), 256, 0, a.stream>>>(tile, output,
                a.coverage.ptr, tile_count, a.width, a.height, a.width_x, a.tile_pixels, x, y);
            check(cudaGetLastError());
        }
    }

    average_overlaps<<<grid_size(count), 256, 0, a.stream>>>(output, a.coverage.ptr, a.pixels, count);
    check(cudaGetLastError());
    // Check all eigensolves once per temporal block, avoiding a host round trip
    // for every spatial tile. Failed decompositions must not produce a displayed map.
    // int error = 0;
    // check(cudaMemcpyAsync(&error, a.error.ptr, sizeof(int), cudaMemcpyDeviceToHost, a.stream));
    // check(cudaStreamSynchronize(a.stream));
    // if (error)
    //     throw std::runtime_error("Local PCA eigendecomposition failed: " + std::to_string(error));
    return true;
}
} // namespace holovibes::compute
