#include "holonp/empty.hh"

#include <stdexcept>

namespace holonp {

void to_json(nlohmann::json &j, const EmptySettings &s) {
  j = nlohmann::json{
      {"shape", s.shape},
      {"order", s.order},
  };

  if (s.dtype) {
    j["dtype"] = *s.dtype;
  }
  if (s.device) {
    j["device"] = *s.device;
  }
}

void from_json(const nlohmann::json &j, EmptySettings &s) {
  j.at("shape").get_to(s.shape);

  if (j.contains("order")) {
    j.at("order").get_to(s.order);
  } else {
    s.order = "C";
  }

  if (j.contains("dtype")) {
    s.dtype = j.at("dtype").get<holoflow::core::DType>();
  } else {
    s.dtype = std::nullopt;
  }

  if (j.contains("device")) {
    s.device = j.at("device").get<holoflow::core::MemLoc>();
  } else {
    s.device = std::nullopt;
  }
}

namespace {

inline void check(bool cond, const std::string &msg) {
  if (!cond) {
    throw std::invalid_argument("EmptyFactory inference error: " + msg);
  }
}

} // namespace

Empty::Empty(const EmptySettings &settings) : settings_(settings) {}

holoflow::core::OpResult Empty::execute(holoflow::core::SyncCtx &ctx) {
  (void)ctx;
  return holoflow::core::OpResult::Ok;
}

holoflow::core::InferResult EmptyFactory::infer(std::span<const holoflow::core::TDesc> input_descs,
                                                const nlohmann::json &jsettings) const {
  const auto settings = jsettings.get<EmptySettings>();
  const auto dtype    = settings.dtype.value_or(holoflow::core::DType::F32);
  const auto memloc   = settings.device.value_or(holoflow::core::MemLoc::Device);

  check(input_descs.empty(), "expected zero inputs");
  check(settings.order == "C", "only C order is supported");
  check(memloc == holoflow::core::MemLoc::Device, "only Device output is supported (for now)");

  holoflow::core::TDesc odesc({settings.shape}, dtype, memloc);

  return holoflow::core::InferResult{
      .input_descs   = {},
      .output_descs  = {odesc},
      .in_place      = {},
      .owned_inputs  = {},
      .owned_outputs = {false},
      .kind          = holoflow::core::TaskKind::Sync,
  };
}

std::unique_ptr<holoflow::core::ISyncTask>
EmptyFactory::create(std::span<const holoflow::core::TDesc> input_descs,
                     const nlohmann::json                  &jsettings,
                     const holoflow::core::SyncCreateCtx   &ctx) const {
  auto infer    = this->infer(input_descs, jsettings);
  auto settings = jsettings.get<EmptySettings>();
  (void)infer;
  (void)ctx;

  return std::unique_ptr<holoflow::core::ISyncTask>(new Empty(settings));
}

} // namespace holonp
