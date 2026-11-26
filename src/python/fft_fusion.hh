// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "holoflow/core/registry.hh"
#include "holoflow/core/tensor.hh"
#include "holoflow/runtime/compiler.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/short_time_fresnel_diffraction.hh"

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <spdlog/spdlog.h>

namespace holoflow::python {

class FusedFft2Fallback final : public core::ISyncTask {
  struct Stage {
    std::unique_ptr<core::ISyncTask> task;
    std::unique_ptr<core::Tensor>    output;
    core::TDesc                      desc;
    bool                             binary;
    bool                             view;
  };
  std::vector<Stage> stages_;

public:
  FusedFft2Fallback(const core::Registry &registry, std::span<const core::TDesc> inputs,
                    const nlohmann::json &settings, const core::SyncCreateCtx &ctx) {
    auto        field = inputs[0];
    const auto &chain = settings.at("chain");
    for (size_t i = 0; i < chain.size(); ++i) {
      const auto &stage   = chain[i];
      const auto &kind    = stage.at("kind").get_ref<const std::string &>();
      const auto &params  = stage.at("settings");
      const auto &factory = registry.get_sync(kind);
      std::array  descriptors{field, inputs[1]};
      const auto  args     = std::span{descriptors}.first(kind == "Multiply" ? 2 : 1);
      const auto  inferred = factory.infer(args, params);
      field                = inferred.output_descs.at(0);
      const bool view = !inferred.in_place.empty() && inferred.in_place.front().read_only_alias;
      auto output = view || i + 1 == chain.size() ? nullptr : std::make_unique<core::Tensor>(field);
      stages_.push_back(
          {factory.create(args, params, ctx), std::move(output), field, kind == "Multiply", view});
    }
  }

  core::OpResult execute(core::SyncCtx &ctx) override {
    auto field = ctx.inputs[0];
    for (size_t i = 0; i < stages_.size(); ++i) {
      auto         &stage = stages_[i];
      std::array    inputs{field, ctx.inputs[1]};
      std::array    outputs{stage.view     ? core::TView{stage.desc, field.storage}
                            : stage.output ? stage.output->view()
                                           : ctx.outputs[0]};
      core::SyncCtx call{std::span{inputs}.first(stage.binary ? 2 : 1), outputs, ctx.cancelled,
                         nullptr, nullptr};
      const auto    result = stage.task->execute(call);
      if (result != core::OpResult::Ok)
        return result;
      field = outputs[0];
    }
    return core::OpResult::Ok;
  }
};

class FusedFft2Factory final : public core::ISyncTaskFactory {
  const core::Registry &registry_;

public:
  explicit FusedFft2Factory(const core::Registry &registry) : registry_(registry) {}

  core::InferResult infer(std::span<const core::TDesc> inputs,
                          const nlohmann::json        &settings) const override {
    auto field = inputs[0];
    for (const auto &stage : settings.at("chain")) {
      const auto &kind = stage.at("kind").get_ref<const std::string &>();
      std::array  args{field, inputs[1]};
      field = registry_.get_sync(kind)
                  .infer(std::span{args}.first(kind == "Multiply" ? 2 : 1), stage.at("settings"))
                  .output_descs.at(0);
    }
    return {.input_descs   = {inputs[0], inputs[1]},
            .output_descs  = {field},
            .in_place      = {},
            .owned_inputs  = {false, false},
            .owned_outputs = {false},
            .kind          = core::TaskKind::Sync};
  }

  std::unique_ptr<core::ISyncTask> create(std::span<const core::TDesc> inputs,
                                          const nlohmann::json        &settings,
                                          const core::SyncCreateCtx   &ctx) const override {
    if (settings.contains("tile")) {
      using namespace holotask::syncs;
      const auto tile = settings.at("tile").get<std::array<size_t, 2>>();
      const ShortTimeFresnelDiffractionSettings config{.lambda    = 1,
                                                       .dx        = 1,
                                                       .dy        = 1,
                                                       .z         = 1,
                                                       .win_h     = tile[0],
                                                       .win_w     = tile[1],
                                                       .stride_y  = tile[0],
                                                       .stride_x  = tile[1],
                                                       .phase_ref = STFDPhaseReference::GLOBAL,
                                                       .output_magnitude = true};
      try {
        return ShortTimeFresnelDiffractionFactory{}.create_with_lens(inputs, config, ctx);
      } catch (const std::exception &error) {
        spdlog::warn("Fused tiled FFT2 setup failed, using primitive tasks: {}", error.what());
        return std::make_unique<FusedFft2Fallback>(registry_, inputs, settings, ctx);
      }
    }
    holotask::syncs::FresnelDiffractionSettings config;
    config.lambda = config.dx = config.dy = config.z = 1;
    config.axes                                      = {-2, -1};
    config.output_magnitude                          = true;
    config.skip_phase_shift                          = true;
    config.use_fused_f16 = config.reduce_magnitude = settings.at("fp16").get<bool>();
    const auto &chain                              = settings.at("chain");
    const bool  widened                            = chain.front().at("kind") == "Conversion";
    const auto &multiply =
        registry_.get_sync(chain.at(widened ? 1 : 0).at("kind").get<std::string>());
    const auto &magnitude =
        registry_.get_sync(chain.at(widened ? 3 : 2).at("kind").get<std::string>());
    try {
      return holotask::syncs::FresnelDiffractionFactory{}.create_with_lens(
          inputs, config, ctx, multiply.fft_fusion().device_source,
          magnitude.fft_fusion().device_source);
    } catch (const std::exception &error) {
      spdlog::warn("Fused FFT2 setup failed, using primitive tasks: {}", error.what());
      return std::make_unique<FusedFft2Fallback>(registry_, inputs, settings, ctx);
    }
  }
};

inline std::optional<core::GraphSpec> lower_fft_tasks(const runtime::GraphPlan &graph,
                                                      const core::Registry     &registry) {
  using Vertex = runtime::GraphPlan::vertex_descriptor;
  using Role   = core::ITaskFactory::FftFusionRole;
  using DType  = core::DType;
  auto role    = [&](Vertex v) { return registry.get(graph[v].spec.kind).fft_fusion(); };
  auto input   = [&](Vertex v, int port) -> std::optional<Vertex> {
    for (auto edge : boost::make_iterator_range(in_edges(v, graph)))
      if (graph[edge].spec.in_idx == port && graph[edge].spec.out_idx == 0)
        return source(edge, graph);
    return {};
  };
  auto single  = [&](Vertex v) { return out_degree(v, graph) == 1; };
  auto compact = [](const core::TDesc &d) {
    return d.mem_loc == core::MemLoc::Device && d.offset == 0 &&
           d.strides == core::TDesc(d.shape, d.dtype, d.mem_loc).strides &&
           std::ranges::find(d.shape, 0u) == d.shape.end();
  };
  struct Replacement {
    Vertex         field, lens;
    core::NodeSpec node;
  };
  std::map<Vertex, Replacement> replacements;
  std::set<Vertex>              removed;
  const auto                    replace = [&](Vertex field, Vertex lens, Vertex terminal,
                                              const std::vector<Vertex> &chain, bool fp16,
                                              std::optional<std::array<size_t, 2>> tile) {
    auto node   = graph[terminal].spec;
    node.kind   = "FusedFFT2";
    auto stages = nlohmann::json::array();
    auto names  = nlohmann::json::array();
    for (auto v : chain) {
      stages.push_back({{"kind", graph[v].spec.kind}, {"settings", graph[v].spec.settings}});
      names.push_back(graph[v].spec.name);
      if (v != terminal)
        removed.insert(v);
    }
    node.settings = {{"chain", stages},
                     {"absorbed", names},
                     {"fp16", fp16},
                     {"backend", fp16 ? "SM120 cuFFTDx" : "cuFFT callbacks"}};
    if (tile) {
      node.settings["tile"]    = *tile;
      node.settings["backend"] = "tiled cuFFT callbacks";
    }
    replacements.emplace(terminal, Replacement{field, lens, std::move(node)});
  };
  const auto desc = [&](Vertex v) -> const core::TDesc & { return graph[v].infer.output_descs[0]; };
  const auto match_tiles = [&](Vertex pack, Vertex fft, Vertex abs) {
    if (graph[pack].spec.kind != "Reshape" ||
        graph[pack].spec.settings.find("copy") == graph[pack].spec.settings.end() ||
        graph[pack].spec.settings["copy"] != true)
      return;
    auto transposed = input(pack, 0);
    if (!transposed || !single(*transposed) || graph[*transposed].spec.kind != "Transpose")
      return;
    auto split = input(*transposed, 0);
    if (!split || !single(*split) || graph[*split].spec.kind != "Reshape" ||
        !graph[*split].spec.settings.contains("copy") ||
        graph[*split].spec.settings["copy"] != false)
      return;
    auto cropped = input(*split, 0);
    if (!cropped || !single(*cropped))
      return;
    const bool crop = graph[*cropped].spec.kind == "Slice";
    auto       mul  = crop ? input(*cropped, 0) : cropped;
    if (!mul || !single(*mul) || role(*mul).role != Role::Load)
      return;
    auto field = input(*mul, 0);
    auto lens  = input(*mul, 1);
    if (!field || !lens || !compact(desc(*field)) || !compact(desc(*lens)) ||
        graph[*lens].infer.constness != core::ConstInferenceState::Constant)
      return;

    const auto &full  = desc(*field);
    const auto &tiles = desc(*transposed);
    if (full.rank() < 2 || tiles.rank() != full.rank() + 2 ||
        (full.dtype != DType::F32 && full.dtype != DType::CF32) ||
        desc(*lens).dtype != DType::CF32 || !compact(desc(*mul)) ||
        desc(*mul).shape != full.shape || tiles.dtype != DType::CF32 || tiles.offset != 0 ||
        desc(pack).shape != tiles.shape || desc(abs).shape != tiles.shape ||
        desc(abs).dtype != DType::F32)
      return;
    const size_t n = full.rank() - 2;
    const size_t h = full.shape[n], w = full.shape[n + 1];
    const size_t th = tiles.shape[n + 2], tw = tiles.shape[n + 3];
    if (!th || !tw || desc(*lens).shape != std::vector<size_t>{h, w} || tiles.shape[n] != h / th ||
        tiles.shape[n + 1] != w / tw)
      return;
    auto       strides = desc(*mul).strides;
    const auto pixel   = strides.back();
    strides.resize(n);
    strides.insert(strides.end(), {th * w * pixel, tw * pixel, w * pixel, pixel});
    for (size_t i = 0; i < tiles.rank(); ++i) {
      if ((i < n && tiles.shape[i] != full.shape[i]) ||
          (tiles.shape[i] > 1 && tiles.strides[i] != strides[i]))
        return;
    }

    std::vector<Vertex> chain;
    if (graph[*field].spec.kind == "Conversion" && single(*field) &&
        graph[*field].spec.settings.value("strategy", std::string("Real")) == "Real" &&
        full.dtype == DType::F32) {
      if (auto original = input(*field, 0);
          original && compact(desc(*original)) && desc(*original).dtype == DType::F16) {
        chain.push_back(*field);
        field = original;
      }
    }
    chain.push_back(*mul);
    if (crop)
      chain.push_back(*cropped);
    chain.insert(chain.end(), {*split, *transposed, pack, fft, abs});
    replace(*field, *lens, abs, chain, false, std::array{th, tw});
  };
  for (auto fft : boost::make_iterator_range(vertices(graph))) {
    if (role(fft).role != Role::Transform || !single(fft))
      continue;
    const auto &fft_settings = graph[fft].spec.settings;
    if (fft_settings.value("inverse", false) ||
        fft_settings.value("norm", std::string("backward")) != "backward" ||
        fft_settings.value("axes", std::vector<int>{}) != std::vector<int>{-2, -1})
      continue;
    const auto mul = input(fft, 0);
    const auto abs = target(*out_edges(fft, graph).first, graph);
    if (!mul || !single(*mul) || role(abs).role != Role::Magnitude || input(abs, 0) != fft)
      continue;
    if (role(*mul).role != Role::Load) {
      if (!fft_settings.value("prefer_fp16", false))
        match_tiles(*mul, fft, abs);
      continue;
    }
    auto       field = input(*mul, 0);
    const auto lens  = input(*mul, 1);
    if (!field || !lens || graph[*lens].infer.constness != core::ConstInferenceState::Constant)
      continue;
    const bool          fp16     = fft_settings.value("prefer_fp16", false);
    Vertex              terminal = abs;
    std::vector<Vertex> chain{*mul, fft, abs};
    if (fp16) {
      int            device = 0;
      cudaDeviceProp properties{};
      if (cudaGetDevice(&device) != cudaSuccess ||
          cudaGetDeviceProperties(&properties, device) != cudaSuccess || properties.major != 12 ||
          properties.minor != 0 || !single(abs))
        continue;
      terminal = target(*out_edges(abs, graph).first, graph);
      if (input(terminal, 0) != abs || role(terminal).role != Role::Mean)
        continue;
      chain.push_back(terminal);
      if (graph[*field].spec.kind == "Conversion" && single(*field)) {
        if (auto original = input(*field, 0)) {
          auto from = graph[*original].infer.output_descs[0].dtype;
          auto to   = graph[*field].infer.output_descs[0].dtype;
          if (((from == DType::F16 && to == DType::F32) ||
               (from == DType::CF16 && to == DType::CF32)) &&
              graph[*field].spec.settings.value("strategy", std::string("Real")) == "Real") {
            chain.insert(chain.begin(), *field);
            field = original;
          }
        }
      }
    }
    const auto &d   = graph[*field].infer.output_descs[0];
    const auto &l   = graph[*lens].infer.output_descs[0];
    const auto &out = graph[terminal].infer.output_descs[0];
    if (!compact(d) || !compact(l) || !compact(out) || l.dtype != DType::CF32 ||
        out.dtype != DType::F32 || d.rank() < 2 ||
        l.shape != std::vector<size_t>{d.shape[d.rank() - 2], d.shape.back()})
      continue;
    if (fp16) {
      if (d.rank() < 3 || l.shape != std::vector<size_t>{320, 512} ||
          (d.dtype != DType::F16 && d.dtype != DType::CF16 && d.dtype != DType::F32))
        continue;
      const auto &settings = graph[terminal].spec.settings;
      auto        axes     = settings.at("axis").get<std::vector<int>>();
      if (axes.size() != 1 || (axes[0] != -3 && axes[0] != static_cast<int>(d.rank()) - 3))
        continue;
      auto shape = d.shape;
      if (settings.value("keepdims", false))
        shape[d.rank() - 3] = 1;
      else
        shape.erase(shape.end() - 3);
      if (out.shape != shape)
        continue;
    } else if ((d.dtype != DType::F32 && d.dtype != DType::CF32) || out.shape != d.shape) {
      continue;
    }
    replace(*field, *lens, terminal, chain, fp16, {});
  }
  if (replacements.empty())
    return {};

  core::GraphSpec                                      spec;
  std::map<Vertex, core::GraphSpec::vertex_descriptor> mapped;
  for (auto v : boost::make_iterator_range(vertices(graph))) {
    if (!removed.contains(v)) {
      auto it = replacements.find(v);
      mapped.emplace(v,
                     add_vertex(it == replacements.end() ? graph[v].spec : it->second.node, spec));
    }
  }
  for (auto edge : boost::make_iterator_range(edges(graph))) {
    const auto from = source(edge, graph), to = target(edge, graph);
    if (!removed.contains(from) && !removed.contains(to) && !replacements.contains(to))
      add_edge(mapped.at(from), mapped.at(to), graph[edge].spec, spec);
  }
  for (const auto &[v, replacement] : replacements) {
    add_edge(mapped.at(replacement.field), mapped.at(v), core::EdgeSpec{0, 0}, spec);
    add_edge(mapped.at(replacement.lens), mapped.at(v), core::EdgeSpec{0, 1}, spec);
  }
  return spec;
}

} // namespace holoflow::python
