#include "curaii/cuda.hh"
#include "holonp/elementwise_minmax.hh"
#include <cmath>
#include <stdexcept>
namespace holonp {
void to_json(nlohmann::json &j, const ElementwiseMinMaxSettings &) { j = nlohmann::json::object(); }
void from_json(const nlohmann::json &, ElementwiseMinMaxSettings &) {}
namespace {
template <bool Max> __global__ void k(const float *a, const float *b, float *o, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    o[i] = Max ? fmaxf(a[i], b[i]) : fminf(a[i], b[i]);
}
class T : public holoflow::core::ISyncTask {
public:
  T(bool m, cudaStream_t s) : m(m), s(s) {}
  holoflow::core::OpResult execute(holoflow::core::SyncCtx &c) override {
    size_t n = c.inputs[0].desc.num_elements();
    int    b = 256, g = (int)((n + b - 1) / b);
    if (m)
      k<true><<<g, b, 0, s>>>((float *)c.inputs[0].data(), (float *)c.inputs[1].data(),
                              (float *)c.outputs[0].data(), n);
    else
      k<false><<<g, b, 0, s>>>((float *)c.inputs[0].data(), (float *)c.inputs[1].data(),
                               (float *)c.outputs[0].data(), n);
    CUDA_CHECK(cudaGetLastError());
    return holoflow::core::OpResult::Ok;
  }

private:
  bool         m;
  cudaStream_t s;
};
holoflow::core::InferResult in(std::span<const holoflow::core::TDesc> x) {
  if (x.size() != 2 || x[0].dtype != holoflow::core::DType::F32 || x[1].dtype != x[0].dtype ||
      x[0].shape != x[1].shape)
    throw std::invalid_argument("elementwise min/max: inputs must be equal F32 tensors");
  return {{x[0], x[1]}, {{x[0].shape, x[0].dtype, holoflow::core::MemLoc::Device}},
          {},           {false, false},
          {false},      holoflow::core::TaskKind::Sync};
}
template <bool M>
std::unique_ptr<holoflow::core::ISyncTask> cr(std::span<const holoflow::core::TDesc> x,
                                              const nlohmann::json &,
                                              const holoflow::core::SyncCreateCtx &c) {
  in(x);
  return std::make_unique<T>(M, c.stream);
}
} // namespace
holoflow::core::InferResult MaximumFactory::infer(std::span<const holoflow::core::TDesc> x,
                                                  const nlohmann::json &) const {
  return in(x);
}
std::unique_ptr<holoflow::core::ISyncTask>
MaximumFactory::create(std::span<const holoflow::core::TDesc> x, const nlohmann::json &j,
                       const holoflow::core::SyncCreateCtx &c) const {
  return cr<true>(x, j, c);
}
std::unique_ptr<holoflow::core::ISyncTask>
MaximumFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                       std::span<const holoflow::core::TDesc> x, const nlohmann::json &j,
                       const holoflow::core::SyncCreateCtx &c) const {
  return cr<true>(x, j, c);
}
holoflow::core::InferResult MinimumFactory::infer(std::span<const holoflow::core::TDesc> x,
                                                  const nlohmann::json &) const {
  return in(x);
}
std::unique_ptr<holoflow::core::ISyncTask>
MinimumFactory::create(std::span<const holoflow::core::TDesc> x, const nlohmann::json &j,
                       const holoflow::core::SyncCreateCtx &c) const {
  return cr<false>(x, j, c);
}
std::unique_ptr<holoflow::core::ISyncTask>
MinimumFactory::update(std::unique_ptr<holoflow::core::ISyncTask>,
                       std::span<const holoflow::core::TDesc> x, const nlohmann::json &j,
                       const holoflow::core::SyncCreateCtx &c) const {
  return cr<false>(x, j, c);
}
} // namespace holonp
