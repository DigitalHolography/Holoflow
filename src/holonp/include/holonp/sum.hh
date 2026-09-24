#pragma once
#include <nlohmann/json.hpp>
#include <span>
#include <vector>
#include "holoflow/core/tasks.hh"
namespace holonp { struct SumSettings { std::vector<int> axis; bool keepdims=false; bool operator==(const SumSettings&) const=default; }; void to_json(nlohmann::json&,const SumSettings&); void from_json(const nlohmann::json&,SumSettings&); class SumFactory:public holoflow::core::ISyncTaskFactory { public: holoflow::core::InferResult infer(std::span<const holoflow::core::TDesc>,const nlohmann::json&) const override; std::unique_ptr<holoflow::core::ISyncTask> create(std::span<const holoflow::core::TDesc>,const nlohmann::json&,const holoflow::core::SyncCreateCtx&) const override; std::unique_ptr<holoflow::core::ISyncTask> update(std::unique_ptr<holoflow::core::ISyncTask>,std::span<const holoflow::core::TDesc>,const nlohmann::json&,const holoflow::core::SyncCreateCtx&) const override; }; }
