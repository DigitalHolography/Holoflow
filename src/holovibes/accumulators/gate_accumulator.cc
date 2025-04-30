#include "holovibes/accumulators/gate_accumulator.hh"

#include <atomic>
#include <cstddef>
#include <cuda_runtime.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/tensor.hh"

namespace holovibes::accumulators {

// ==========================================================================
//                     GateParams Implementation
// ==========================================================================

void to_json(nlohmann::json &j, const GateParams &p) {
  j = nlohmann::json{{"is_on", p.is_on}};
  if (p.target.has_value()) {
    j["target"] = p.target.value();
  }
}

void from_json(const nlohmann::json &j, GateParams &p) {
  j.at("is_on").get_to(p.is_on);
  if (j.contains("target")) {
    p.target = j.at("target").get<size_t>();
  } else {
    p.target = std::nullopt;
  }
}

// ==========================================================================
//                     Gate Implementation
// ==========================================================================

Gate::Gate(const dh::AccumulatorMeta &meta, cudaStream_t stream,
           dh::Accumulator::EventListeners event_listeners, bool is_on,
           std::optional<size_t> target,
           curaii::cuda::unique_host_ptr<uint8_t> h_buffer,
           curaii::cuda::unique_device_ptr<uint8_t> d_buffer)
    : dh::Accumulator(meta, stream, event_listeners), write_(true),
      is_on_(is_on), target_(target), h_buffer_(std::move(h_buffer)),
      d_buffer_(std::move(d_buffer)), frames_passed_(0) {
  DH_CHECK((d_buffer_ && !h_buffer_) || (h_buffer_ && !d_buffer_));
  buffer_ = d_buffer_ ? d_buffer_.get() : h_buffer_.get();
}

std::optional<dh::TensorView> Gate::write_tensor() {
  if (!write_) {
    return std::nullopt;
  }

  return dh::TensorView(buffer_, meta_.imeta());
}

void Gate::commit_write() {
  if (!is_on_) {
    return;
  }

  write_.store(false);
}

std::optional<dh::TensorView> Gate::read_tensor() {
  if (write_ || !is_on_) {
    return std::nullopt;
  }

  return dh::TensorView(buffer_, meta_.imeta());
}

void Gate::commit_read() {
  frames_passed_ += meta_.ometa().shape().at(0);
  if (target_ && frames_passed_ >= *target_) {
    is_on_ = false;
    json out;
    out["action"] = "stopped";
    out["status"] = "ok";
    emit_event(out);
  }

  write_.store(true);
}

void Gate::handle_event(const json &event) {
  // 0) Unpack parameters
  std::string action = event.at("action").get<std::string>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  json out;

  try {
    // 1) Event sanity
    const std::unordered_set<std::string> valid_actions = {"start", "stop"};
    check(valid_actions.contains(action), "action invalid");

    const std::unordered_map<std::string, bool> expected_state = {
        {"start", false}, {"stop", true}};
    check(is_on_ == expected_state.at(action),
          "invalid action for current state");

    // 2) Perform the action and emit success event
    if (action == "start") {
      frames_passed_ = 0;
      is_on_ = true;
      out["action"] = "started";
      out["status"] = "ok";
    } else { // action == "stop"
      is_on_ = false;
      out["action"] = "stopped";
      out["status"] = "ok";
    }
  } catch (const std::exception &e) {
    // 3) On failure, emit the corresponding *_failed event
    if (action == "start") {
      out["action"] = "start_failed";
    } else { // action == "stop"
      out["action"] = "stop_failed";
    }
    out["status"] = e.what();
  }

  // 4) Finally, fire the event so your listeners (the Worker) pick it up
  emit_event(out);
}

// ==========================================================================
//                     GateFactory Implementation
// ==========================================================================
dh::AccumulatorMeta GateFactory::type_check(const dh::TensorMeta &imeta,
                                            const json &jparams) {
  // 0) Unpack parameters
  const GateParams params = jparams.get<GateParams>();

  const auto check = [&](bool cond, std::string_view what) {
    if (!cond) {
      throw std::invalid_argument(std::string(what));
    }
  };

  // 1) Tensor meta sanity
  if (params.target) {
    check(*params.target % imeta.shape().at(0) == 0,
          "tensor dim 0 must divide target");
  }

  // 3) Success
  return dh::AccumulatorMeta(imeta, imeta);
}

std::unique_ptr<dh::Accumulator>
GateFactory::create(const dh::TensorMeta &imeta, const json &jparams,
                    cudaStream_t stream,
                    dh::Accumulator::EventListeners event_listeners) {
  // 1) Validate
  auto meta = type_check(imeta, jparams);
  auto params = jparams.get<GateParams>();

  // 2) Buffer size
  auto buffer_size = meta.imeta().size_in_bytes();

  // 3) Allocation
  curaii::cuda::unique_host_ptr<uint8_t> host_buffer = nullptr;
  curaii::cuda::unique_device_ptr<uint8_t> device_buffer = nullptr;
  switch (meta.imeta().memory_location()) {
  case dh::MemoryLocation::HOST:
    host_buffer = curaii::cuda::make_unique_host_ptr<uint8_t>(buffer_size);
    break;
  case dh::MemoryLocation::DEVICE:
    device_buffer =
        curaii::cuda::make_unique_device_ptr<uint8_t>(buffer_size, stream);
    break;
  }

  // 4) Assemble accumulator
  auto *accumulator =
      new Gate(meta, stream, event_listeners, params.is_on, params.target,
               std::move(host_buffer), std::move(device_buffer));

  return std::unique_ptr<Gate>(accumulator);
}

} // namespace holovibes::accumulators