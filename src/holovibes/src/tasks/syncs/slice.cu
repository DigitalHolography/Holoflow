#include "slice.hh"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include "bug.hh"
#include "logger.hh"

namespace holovibes::tasks::syncs {

// -------------------------------------------------------------------------------------------------
// JSON Serialization
// -------------------------------------------------------------------------------------------------

void to_json(nlohmann::json &j, const SliceSettings::AxisRange &ar) {
  if (ar.start)
    j["start"] = ar.start.value();
  if (ar.end)
    j["end"] = ar.end.value();
  if (ar.step)
    j["step"] = ar.step.value();
}

void from_json(const nlohmann::json &j, SliceSettings::AxisRange &ar) {
  if (j.contains("start"))
    ar.start = j.at("start").get<size_t>();
  if (j.contains("end"))
    ar.end = j.at("end").get<size_t>();
  if (j.contains("step"))
    ar.step = j.at("step").get<size_t>();
}

void to_json(nlohmann::json &j, const SliceSettings &ss) { j["ranges"] = ss.ranges; }

void from_json(const nlohmann::json &j, SliceSettings &ss) { j.at("ranges").get_to(ss.ranges); }

// -------------------------------------------------------------------------------------------------
// Slice Implementation
// -------------------------------------------------------------------------------------------------

Slice::Slice(const ResolvedSliceParams &params, cudaStream_t stream) :
    params_(params),
    stream_(stream) {}

holoflow::core::OpResult Slice::execute(holoflow::core::SyncCtx &ctx) {
  const size_t Nz = params_.dims[0].out_size;
  const size_t Ny = params_.dims[1].out_size;
  const size_t Nx = params_.dims[2].out_size;

  auto       &iview = ctx.inputs[0];
  auto       &oview = ctx.outputs[0];
  const auto *idata = static_cast<const std::byte *>(iview.data);
  auto       *odata = static_cast<std::byte *>(oview.data);

  auto offset =
      params_.dims[0].start * iview.desc.shape[1] * iview.desc.shape[2] * size_of(iview.desc.dtype);

  if (Nx == iview.desc.shape[2] && Ny == iview.desc.shape[1]) {
    // Fast path: contiguous copy
    size_t copy_bytes = Nz * Ny * Nx * size_of(iview.desc.dtype);
    CUDA_CHECK(
        cudaMemcpyAsync(odata, idata + offset, copy_bytes, cudaMemcpyDeviceToDevice, stream_));
    return holoflow::core::OpResult::Ok;
  }

  HOLOVIBES_UNIMPLEMENTED();
}

// -------------------------------------------------------------------------------------------------
// SliceFactory Implementation
// -------------------------------------------------------------------------------------------------

Slice::ResolvedRange SliceFactory::resolve_range(const SliceSettings::AxisRange &range,
                                                 size_t input_dim_size) const {
  size_t N = input_dim_size;

  size_t step = range.step.value_or(1);
  if (step == 0) {
    throw std::invalid_argument("Slice error: step size must be non-zero.");
  }

  size_t start = range.start.value_or(0);
  if (start >= N) {
    throw std::invalid_argument("Slice error: start index is out of bounds.");
  }

  size_t end = range.end.value_or(N);
  if (end > N) {
    end = N;
  }

  if (start >= end) {
    throw std::invalid_argument("Slice error: effective start index must be less than end index.");
  }

  size_t range_length = end - start;
  size_t out_size     = (range_length + step - 1) / step;

  return {start, step, out_size};
}

Slice::ResolvedSliceParams
SliceFactory::resolve_params(const SliceSettings         &settings,
                             const holoflow::core::TDesc &input_desc) const {
  (void)settings;
  (void)input_desc;
  throw std::logic_error("Not implemented");
}

holoflow::core::InferResult SliceFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                const nlohmann::json &jsettings) const {

  const auto check = [&](bool condition, const std::string &msg) {
    if (!condition) {
      logger()->error("[SliceFactory::infer] {}", msg);
      throw std::invalid_argument("Slice error: " + msg);
    }
  };

  auto settings = jsettings.get<SliceSettings>();
  check(input_descs.size() == 1, "Expected exactly one input");

  const auto &idesc = input_descs[0];
  const int   rank  = static_cast<int>(idesc.rank());
  check(rank == 3, "Slice task currently only supports 3D tensors (B, H, W)");
  check(idesc.mem_loc == holoflow::core::MemLoc::Device, "Input must be on Device");

  // We only care about the last three dimensions (B, H, W)
  // Map dimensions to indices in the TDesc shape array: B->0, H->1, W->2
  const int    DIM_MAP[3]      = {0, 1, 2};
  const size_t input_shapes[3] = {idesc.shape[0], idesc.shape[1], idesc.shape[2]};

  // Prepare resolved parameters and output descriptor
  Slice::ResolvedSliceParams params;
  auto                       odesc = idesc;

  for (int i = 0; i < 3; ++i) { // i=0: B, i=1: H, i=2: W
    // Get user range for this axis, or an empty one (default)
    const SliceSettings::AxisRange user_range =
        (i < settings.ranges.size()) ? settings.ranges[i] : SliceSettings::AxisRange{};

    // Resolve parameters
    Slice::ResolvedRange resolved = resolve_range(user_range, input_shapes[i]);

    // Store resolved parameters
    params.dims[i] = resolved;

    // Update output shape
    odesc.shape[DIM_MAP[i]] = resolved.out_size;
  }

  // Store the final output descriptor for the task
  params.output_desc = odesc;

  // The InferResult requires the *final* output descriptor
  return holoflow::core::InferResult{
      .input_descs   = {idesc},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {false},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
SliceFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                     const nlohmann::json                  &jsettings,
                     const holoflow::core::SyncCreateCtx   &ctx) const {

  // Perform inference and get the resolved parameters indirectly
  auto infer_res = infer(input_descs, jsettings);

  // Rerun resolve logic to get the full ResolvedSliceParams struct
  auto        settings = jsettings.get<SliceSettings>();
  const auto &idesc    = input_descs[0];

  Slice::ResolvedSliceParams params;
  const size_t               input_shapes[3] = {idesc.shape[0], idesc.shape[1], idesc.shape[2]};

  for (int i = 0; i < 3; ++i) { // B, H, W
    const SliceSettings::AxisRange user_range =
        (i < settings.ranges.size()) ? settings.ranges[i] : SliceSettings::AxisRange{};
    params.dims[i] = resolve_range(user_range, input_shapes[i]);
  }

  params.output_desc = infer_res.output_descs[0];

  // Note: The Slice task only stores the resolved parameters and stream
  return std::unique_ptr<holoflow::core::ISyncTask>(new Slice(params, ctx.stream));
}

} // namespace holovibes::tasks::syncs