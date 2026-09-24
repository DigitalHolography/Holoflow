#pragma once
#include "holoflow/core/tasks.hh"
#include <nlohmann/json.hpp>
#include <span>
namespace holonp { struct ElementwiseMinMaxSettings { bool operator==(const ElementwiseMinMaxSettings&) const=default; }; void to_json(nlohmann::json&,const ElementwiseMinMaxSettings&); void from_json(const nlohmann::json&,ElementwiseMinMaxSettings&);
#define HOLONP_BINARY_FACTORY(Name) class Name##Factory:public holoflow::core::ISyncTaskFactory { public: holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,const nlohmann::json&) const override; std::unique_ptr<holoflow::core::ISyncTask> create(std::span<const holoflow::core::TDesc>,const nlohmann::json&,const holoflow::core::SyncCreateCtx&) const override; std::unique_ptr<holoflow::core::ISyncTask> update(std::unique_ptr<holoflow::core::ISyncTask>,std::span<const holoflow::core::TDesc>,const nlohmann::json&,const holoflow::core::SyncCreateCtx&) const override; };
using MaximumSettings=ElementwiseMinMaxSettings; using MinimumSettings=ElementwiseMinMaxSettings; HOLONP_BINARY_FACTORY(Maximum) HOLONP_BINARY_FACTORY(Minimum) }
