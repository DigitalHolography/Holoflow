// // Copyright 2025 Digital Holography Foundation
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

// #include <gtest/gtest.h>

// #include <algorithm>
// #include <boost/graph/graph_traits.hpp>
// #include <filesystem>
// #include <fstream>
// #include <functional>
// #include <map>
// #include <nlohmann/json.hpp>
// #include <span>
// #include <string>
// #include <utility>
// #include <vector>

// #include "factory_maker.hh"
// #include "graph_builder.hh"
// #include "holoflow/core/graph_spec.hh"
// #include "holoflow/core/registry.hh"
// #include "holoflow/runtime/compiler.hh"
// #include "holoflow/runtime/graph_display.hh"
// #include "holotask/asyncs/batch_queue.hh"
// #include "holotask/asyncs/slide_avg.hh"
// #include "holotask/sinks/holofile.hh"
// #include "holotask/sources/ametek_s710_euresys_coaxlink_octo.hh"
// #include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"
// #include "holotask/sources/holofile.hh"
// #include "holotask/syncs/angular_spectrum.hh"
// #include "holotask/syncs/average.hh"
// #include "holotask/syncs/conversion.hh"
// #include "holotask/syncs/convolution.hh"
// #include "holotask/syncs/crop.hh"
// #include "holotask/syncs/fft_shift.hh"
// #include "holotask/syncs/filter2d.hh"
// #include "holotask/syncs/fresnel_diffraction.hh"
// #include "holotask/syncs/log.hh"
// #include "holotask/syncs/memcpy.hh"
// #include "holotask/syncs/pca.hh"
// #include "holotask/syncs/pct_clip.hh"
// #include "holotask/syncs/registration.hh"
// #include "holotask/syncs/reshape.hh"
// #include "holotask/syncs/rotation.hh"
// #include "holotask/syncs/stft.hh"

// using namespace holotask;

// namespace holoflow::runtime {
// namespace {

// using holoflow::core::TaskKind;
// using holoflow::test::AsyncFactorySpec;
// using holoflow::test::copy_descs;
// using holoflow::test::GraphBuilder;
// using holoflow::test::make_desc;
// using holoflow::test::make_infer_result;
// using holoflow::test::RecordingAsyncFactory;
// using holoflow::test::RecordingSyncFactory;
// using holoflow::test::StubAsyncTask;
// using holoflow::test::StubSyncTask;
// using holoflow::test::SyncFactorySpec;

// std::vector<std::filesystem::path> list_json_files(const std::filesystem::path &dir) {
//   std::vector<std::filesystem::path> json_files;
//   for (const auto &entry : std::filesystem::directory_iterator(dir)) {
//     if (entry.is_regular_file() && entry.path().extension() == ".json") {
//       json_files.push_back(entry.path());
//     }
//   }
//   return json_files;
// }

// enum class Expect { Pass, Fail };

// struct CompilerCase {
//   std::filesystem::path path;
//   Expect                expect;
// };

// std::string gtest_case_name(const ::testing::TestParamInfo<CompilerCase> &info) {
//   auto s = info.param.path.generic_string();

//   // Gtest names must be [a-zA-Z0-9_]
//   for (auto &c : s) {
//     if (!std::isalnum(c)) {
//       c = '_';
//     }
//   }

//   // Avoid leading digit
//   if (!s.empty() && std::isdigit(s[0])) {
//     s = "_" + s;
//   }

//   // Avoid empty name
//   if (s.empty()) {
//     s = "empty";
//   }

//   return s;
// }

// std::vector<CompilerCase> discover_compiler_cases() {
//   const std::filesystem::path root = HOLOFLOW_TESTDATA_DIR;
//   std::vector<CompilerCase>   cases;

//   for (const auto &path : list_json_files(root / "compiler" / "ok")) {
//     cases.push_back(CompilerCase{path, Expect::Pass});
//   }

//   for (const auto &path : list_json_files(root / "compiler" / "fail")) {
//     cases.push_back(CompilerCase{path, Expect::Fail});
//   }

//   return cases;
// }

// template <class F, class... Args>
// void reg_sync(holoflow::core::Registry &r, std::string_view name, Args &&...args) {
//   r.register_sync(std::string{name}, std::make_unique<F>(std::forward<Args>(args)...));
// }

// template <class F, class... Args>
// void reg_async(holoflow::core::Registry &r, std::string_view name, Args &&...args) {
//   r.register_async(std::string{name}, std::make_unique<F>(std::forward<Args>(args)...));
// }

// class CompilerSuite : public ::testing::TestWithParam<CompilerCase> {
// protected:
//   core::Registry registry;

//   void SetUp() override {
//     // clang-format off
//     reg_async<asyncs::BatchQueueFactory>(registry, "BatchQueue");
//     reg_async<asyncs::SlidingAverageFactory>(registry, "SlidingAverage");
//     reg_sync<sinks::HolofileFactory>(registry, "HolofileWriter");
//     reg_sync<sources::HolofileFactory>(registry, "Holofile");
//     reg_sync<sources::AmetekS710EuresysCoaxlinkOctoFactory>(registry,
//     "AmetekS710EuresysCoaxlinkOcto");
//     reg_sync<sources::AmetekS711EuresysCoaxlinkQSFPFactory>(registry,
//     "AmetekS711EuresysCoaxlinkQSFP+"); reg_sync<syncs::AngularSpectrumFactory>(registry,
//     "AngularSpectrum"); reg_sync<syncs::AverageFactory>(registry, "Average");
//     reg_sync<syncs::ConversionFactory>(registry, "Conversion");
//     reg_sync<syncs::FFTShiftFactory>(registry, "FFTShift");
//     reg_sync<syncs::FresnelDiffractionFactory>(registry, "FresnelDiffraction");
//     reg_sync<syncs::MemcpyFactory>(registry, "Memcpy");
//     reg_sync<syncs::PcaFactory>(registry, "Pca");
//     reg_sync<syncs::PctClipFactory>(registry, "PctClip");
//     reg_sync<syncs::StftFactory>(registry, "Stft");
//     reg_sync<syncs::ConvolutionFactory>(registry, "Convolution");
//     reg_sync<syncs::Filter2DFactory>(registry, "Filter2D");
//     reg_sync<syncs::LogFactory>(registry, "Log");
//     reg_sync<syncs::RegistrationFactory>(registry, "Registration");
//     reg_sync<syncs::ReshapeFactory>(registry, "Reshape");
//     reg_sync<syncs::CropFactory>(registry, "Crop");
//     reg_sync<syncs::RotationFactory>(registry, "Rotation");
//     // clang-format on

//     SyncFactorySpec sink_spec;
//     sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//       auto input_descs = copy_descs(inputs);
//       return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//     };
//     registry.register_sync("DisplayTensorXY",
//                            std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));
//   }
// };

// TEST_P(CompilerSuite, Compile) {
//   const auto &param = GetParam();
//   const auto  path  = param.path;

//   // Load graph spec
//   nlohmann::json j;
//   {
//     std::ifstream in(path);
//     if (!in.is_open()) {
//       FAIL() << "Failed to open test case file: " << path;
//     }
//     in >> j;
//   }

//   auto     g = holoflow::core::from_json(j);
//   Compiler compiler{registry};
//   if (param.expect == Expect::Pass) {
//     EXPECT_NO_THROW(compiler.compile(g)) << "Test case: " << path;
//   } else {
//     EXPECT_THROW(compiler.compile(g), std::exception) << "Test case: " << path;
//   }
// }

// INSTANTIATE_TEST_SUITE_P(CompilerCases, CompilerSuite,
//                          ::testing::ValuesIn(discover_compiler_cases()), gtest_case_name);

// GraphPlan::vertex_descriptor find_plan_vertex(const GraphPlan &graph, const std::string &name) {
//   for (auto v : boost::make_iterator_range(boost::vertices(graph))) {
//     if (graph[v].spec.name == name) {
//       return v;
//     }
//   }
//   ADD_FAILURE() << "Missing plan vertex " << name;
//   return *boost::vertices(graph).first;
// }

// TEST(CompilerValidation, EmptyGraphFails) {
//   core::Registry  registry;
//   Compiler        compiler(registry);
//   core::GraphSpec graph;
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerValidation, SimpleGraphSucceeds) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({1})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   AsyncFactorySpec async_spec;
//   async_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Async, std::move(input_descs),
//                              {make_desc({1}), make_desc({1})});
//   };
//   auto *queue_factory = new RecordingAsyncFactory(std::move(async_spec));
//   registry.register_async("queue", std::unique_ptr<RecordingAsyncFactory>(queue_factory));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("queue", "queue", "queue", {{"answer", 42}});
//   builder.add_node("snk", "snk", "sink");
//   builder.add_node("snk2", "snk2", "sink");
//   builder.add_edge("src", "queue", 0, 0);
//   builder.add_edge("queue", "snk", 0, 0);
//   builder.add_edge("queue", "snk2", 1, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     output = compiler.compile(graph);
// }

// TEST(CompilerValidation, DuplicateNodeNameFails) {
//   core::Registry registry;
//   registry.register_sync("noop", std::make_unique<RecordingSyncFactory>());

//   GraphBuilder builder;
//   builder.add_node("n1", "dup", "noop");
//   builder.add_node("n2", "dup", "noop");
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerValidation, DuplicateEdgeDestinationFails) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({2})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("root", "root", "source");
//   builder.add_node("leaf", "leaf", "sink");
//   builder.add_edge("root", "leaf", 0, 0);
//   builder.add_edge("root", "leaf", 0, 1);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerValidation, MissingFactoryFails) {
//   core::Registry registry;
//   GraphBuilder   builder;
//   builder.add_node("node", "node", "unknown");
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerValidation, MultipleSourcesFail) {
//   core::Registry registry;
//   registry.register_sync("noop", std::make_unique<RecordingSyncFactory>());

//   GraphBuilder builder;
//   builder.add_node("a", "a", "noop");
//   builder.add_node("b", "b", "noop");
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerValidation, MultipleInputsFail) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({4})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec mid_spec;
//   mid_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {make_desc({4})});
//   };
//   registry.register_sync("mid", std::make_unique<RecordingSyncFactory>(std::move(mid_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("root", "root", "source");
//   builder.add_node("mid", "mid", "mid");
//   builder.add_node("leaf", "leaf", "sink");
//   builder.add_edge("root", "mid", 0, 0);
//   builder.add_edge("mid", "leaf", 0, 0);
//   builder.add_edge("root", "leaf", 0, 1);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerTyping, PropagatesEdgeDescriptors) {
//   core::Registry registry;

//   auto source_desc = make_desc({4, 4}, core::DType::F32);

//   SyncFactorySpec source_spec;
//   source_spec.infer = [source_desc](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {source_desc});
//   };
//   auto *source_factory = new RecordingSyncFactory(std::move(source_spec));
//   registry.register_sync("source", std::unique_ptr<RecordingSyncFactory>(source_factory));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   auto *sink_factory = new RecordingSyncFactory(std::move(sink_spec));
//   registry.register_sync("sink", std::unique_ptr<RecordingSyncFactory>(sink_factory));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     output = compiler.compile(graph);

//   ASSERT_EQ(sink_factory->infer_calls().size(), 1u);
//   const auto &infer_call = sink_factory->infer_calls().front();
//   ASSERT_EQ(infer_call.input_descs.size(), 1u);
//   EXPECT_EQ(infer_call.input_descs.front().shape, source_desc.shape);
//   EXPECT_EQ(infer_call.input_descs.front().dtype, source_desc.dtype);
//   EXPECT_EQ(infer_call.input_descs.front().mem_loc, source_desc.mem_loc);

//   auto plan  = output->graph;
//   auto src_v = find_plan_vertex(plan, "src");
//   auto snk_v = find_plan_vertex(plan, "snk");
//   ASSERT_FALSE(plan[src_v].out_tids.empty());
//   EXPECT_EQ(plan[src_v].out_tids.front(), plan[snk_v].in_tids.front());
// }

// TEST(CompilerTensorIds, AssignsIdsAndTidsVectors) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({1})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec mid_spec;
//   mid_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {make_desc({1})});
//   };
//   registry.register_sync("mid", std::make_unique<RecordingSyncFactory>(std::move(mid_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("mid", "mid", "mid");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "mid", 0, 0);
//   builder.add_edge("mid", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     output = compiler.compile(graph);

//   const auto &plan  = output->graph;
//   auto        src_v = find_plan_vertex(plan, "src");
//   auto        mid_v = find_plan_vertex(plan, "mid");
//   auto        snk_v = find_plan_vertex(plan, "snk");

//   ASSERT_EQ(plan[src_v].out_tids.size(), 1u);
//   EXPECT_EQ(plan[src_v].out_tids.front(), 0);
//   ASSERT_EQ(plan[mid_v].in_tids.size(), 1u);
//   EXPECT_EQ(plan[mid_v].in_tids.front(), 0);
//   ASSERT_EQ(plan[mid_v].out_tids.size(), 1u);
//   EXPECT_EQ(plan[mid_v].out_tids.front(), 1);
//   ASSERT_EQ(plan[snk_v].in_tids.size(), 1u);
//   EXPECT_EQ(plan[snk_v].in_tids.front(), 1);
// }

// TEST(CompilerTemporal, RejectsMultipleOwnedConsumers) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({8})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec consumer_spec;
//   consumer_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     std::vector<bool> owns(inputs.size(), true);
//     auto              input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {}, owns);
//   };
//   registry.register_sync("left", std::make_unique<RecordingSyncFactory>(consumer_spec));
//   registry.register_sync("right", std::make_unique<RecordingSyncFactory>(consumer_spec));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("left", "left", "left");
//   builder.add_node("right", "right", "right");
//   builder.add_edge("src", "left", 0, 0);
//   builder.add_edge("src", "right", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerTemporal, AllowsSingleOwnedConsumer) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({8})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     std::vector<bool> owns(inputs.size(), true);
//     auto              input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {}, owns);
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_NO_THROW({
//     auto output = compiler.compile(graph);
//     EXPECT_EQ(boost::num_vertices(output->graph), 2u);
//   });
// }

// TEST(CompilerSpatial, RejectsMultipleOwnersOfSameTensor) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({1})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec inplace_spec;
//   inplace_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     std::vector<bool>          owns_in(inputs.size(), true);
//     std::vector<bool>          owns_out(1, true);
//     std::vector<core::InPlace> inplace     = {{0, 0}};
//     auto                       input_descs = copy_descs(inputs);
//     std::vector<core::TDesc>   output_descs{input_descs.front()};
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), std::move(output_descs),
//                              owns_in, owns_out, inplace);
//   };
//   registry.register_sync("mid", std::make_unique<RecordingSyncFactory>(std::move(inplace_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("mid", "mid", "mid");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "mid", 0, 0);
//   builder.add_edge("mid", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerSections, BuildsSingleSyncSection) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({1})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec mid_spec;
//   mid_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {make_desc({1})});
//   };
//   registry.register_sync("mid", std::make_unique<RecordingSyncFactory>(std::move(mid_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("mid", "mid", "mid");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "mid", 0, 0);
//   builder.add_edge("mid", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     output = compiler.compile(graph);

//   ASSERT_EQ(output->sections.size(), 1u);
//   const auto &section = output->sections.front();
//   ASSERT_EQ(section.sync_topo.size(), 3u);
//   EXPECT_EQ(output->graph[section.sync_topo[0]].spec.name, "src");
//   EXPECT_EQ(output->graph[section.sync_topo[1]].spec.name, "mid");
//   EXPECT_EQ(output->graph[section.sync_topo[2]].spec.name, "snk");
//   EXPECT_TRUE(section.async_cons.empty());
//   EXPECT_TRUE(section.async_prod.empty());
// }

// TEST(CompilerSections, AsyncNodeGetsProducerAndConsumerStreams) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({1})});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   AsyncFactorySpec async_spec;
//   async_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Async, std::move(input_descs), {make_desc({1})});
//   };
//   auto *queue_factory = new RecordingAsyncFactory(std::move(async_spec));
//   registry.register_async("queue", std::unique_ptr<RecordingAsyncFactory>(queue_factory));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("queue", "queue", "queue");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "queue", 0, 0);
//   builder.add_edge("queue", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     output = compiler.compile(graph);

//   ASSERT_EQ(output->sections.size(), 2u);
//   const auto &producer_section = output->sections[0];
//   const auto &consumer_section = output->sections[1];

//   ASSERT_EQ(producer_section.sync_topo.size(), 1u);
//   EXPECT_EQ(output->graph[producer_section.sync_topo[0]].spec.name, "src");
//   ASSERT_EQ(producer_section.async_prod.size(), 1u);
//   EXPECT_EQ(output->graph[producer_section.async_prod[0]].spec.name, "queue");

//   ASSERT_EQ(consumer_section.async_cons.size(), 1u);
//   EXPECT_EQ(output->graph[consumer_section.async_cons[0]].spec.name, "queue");
//   ASSERT_EQ(consumer_section.sync_topo.size(), 1u);
//   EXPECT_EQ(output->graph[consumer_section.sync_topo[0]].spec.name, "snk");

//   const auto &tasks = output->resources.tasks;
//   ASSERT_EQ(tasks.size(), 3u);

//   auto *src_task = dynamic_cast<StubSyncTask *>(tasks.at("src").get());
//   ASSERT_NE(src_task, nullptr);
//   EXPECT_EQ(src_task->ctx().stream, producer_section.stream);

//   auto *snk_task = dynamic_cast<StubSyncTask *>(tasks.at("snk").get());
//   ASSERT_NE(snk_task, nullptr);
//   EXPECT_EQ(snk_task->ctx().stream, consumer_section.stream);

//   auto *queue_task = dynamic_cast<StubAsyncTask *>(tasks.at("queue").get());
//   ASSERT_NE(queue_task, nullptr);
//   EXPECT_EQ(queue_task->ctx().producer_stream, producer_section.stream);
//   EXPECT_EQ(queue_task->ctx().consumer_stream, consumer_section.stream);
// }

// TEST(CompilerTyping, IncompatibleTensorsFail) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {},
//                              {make_desc({4, 4}, core::DType::F32)}); // 4x4 F32 tensor
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     if (inputs.empty() || inputs[0].shape != std::vector<size_t>{8, 8}) {
//       throw std::logic_error("Input tensor size mismatch");
//     }
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   EXPECT_THROW(compiler.compile(graph), std::logic_error);
// }

// TEST(CompilerReuse, CallsUpdateWithPreviousOutput) {
//   core::Registry registry;

//   SyncFactorySpec spec;
//   spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {});
//   };
//   bool update_called = false;
//   spec.update        = [&update_called](std::unique_ptr<core::ISyncTask> old_task,
//                                  std::span<const core::TDesc>     input_descs,
//                                  const nlohmann::json &settings, const core::SyncCreateCtx &ctx)
//                                  {
//     (void)input_descs;
//     (void)settings;
//     update_called = true;
//     EXPECT_NE(old_task, nullptr);
//     return std::make_unique<StubSyncTask>(ctx);
//   };

//   auto *factory = new RecordingSyncFactory(std::move(spec));
//   registry.register_sync("noop", std::unique_ptr<RecordingSyncFactory>(factory));

//   GraphBuilder builder;
//   builder.add_node("node", "node", "noop");
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     first = compiler.compile(graph);
//   ASSERT_EQ(factory->create_calls().size(), 1u);
//   ASSERT_TRUE(factory->update_calls().empty());

//   auto second = compiler.compile(graph, std::move(first));
//   EXPECT_TRUE(update_called);
//   ASSERT_EQ(factory->update_calls().size(), 1u);
//   EXPECT_NE(factory->update_calls().front().previous, nullptr);
//   EXPECT_EQ(factory->create_calls().size(), 1u);
//   ASSERT_NE(second, nullptr);
//   EXPECT_EQ(second->resources.tasks.size(), 1u);
// }

// TEST(CompilerResources, OwnedTensorsNotPreallocated) {
//   core::Registry registry;

//   SyncFactorySpec source_spec;
//   source_spec.infer = [](std::span<const core::TDesc>, const nlohmann::json &) {
//     std::vector<bool> owned_outputs = {false};
//     return make_infer_result(TaskKind::Sync, {}, {make_desc({4, 4})}, {}, owned_outputs);
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     std::vector<bool> owned_inputs = {true};
//     auto              input_descs  = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {}, owned_inputs);
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     output = compiler.compile(graph);

//   const auto &tensors = output->resources.tensors;
//   EXPECT_TRUE(tensors.empty());
// }

// TEST(CompilerResources, TensorsAllocated) {
//   core::Registry registry;

//   auto            source_desc = make_desc({4, 4}, core::DType::F32, core::MemLoc::Device);
//   SyncFactorySpec source_spec;
//   source_spec.infer = [source_desc](std::span<const core::TDesc>, const nlohmann::json &) {
//     return make_infer_result(TaskKind::Sync, {}, {source_desc});
//   };
//   registry.register_sync("source",
//   std::make_unique<RecordingSyncFactory>(std::move(source_spec)));

//   auto            mid_output_desc = make_desc({2, 8}, core::DType::F32, core::MemLoc::Device);
//   SyncFactorySpec mid_spec;
//   mid_spec.infer = [mid_output_desc](std::span<const core::TDesc> inputs, const nlohmann::json &)
//   {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {mid_output_desc});
//   };
//   registry.register_sync("mid", std::make_unique<RecordingSyncFactory>(std::move(mid_spec)));

//   SyncFactorySpec sink_spec;
//   sink_spec.infer = [](std::span<const core::TDesc> inputs, const nlohmann::json &) {
//     auto input_descs = copy_descs(inputs);
//     return make_infer_result(TaskKind::Sync, std::move(input_descs), {});
//   };
//   registry.register_sync("sink", std::make_unique<RecordingSyncFactory>(std::move(sink_spec)));

//   GraphBuilder builder;
//   builder.add_node("src", "src", "source");
//   builder.add_node("mid", "mid", "mid");
//   builder.add_node("snk", "snk", "sink");
//   builder.add_edge("src", "mid", 0, 0);
//   builder.add_edge("mid", "snk", 0, 0);
//   auto graph = builder.finish();

//   Compiler compiler(registry);
//   auto     output = compiler.compile(graph);

//   const auto &tensors = output->resources.tensors;

//   ASSERT_EQ(tensors.size(), 2u);

//   EXPECT_TRUE(tensors.count(0) > 0);
//   EXPECT_TRUE(tensors.count(1) > 0);

//   auto plan  = output->graph;
//   auto src_v = find_plan_vertex(plan, "src");
//   auto mid_v = find_plan_vertex(plan, "mid");
//   auto snk_v = find_plan_vertex(plan, "snk");

//   EXPECT_EQ(plan[src_v].out_tids.front(), 0);
//   EXPECT_EQ(plan[mid_v].in_tids.front(), 0);
//   EXPECT_EQ(plan[mid_v].out_tids.front(), 1);
//   EXPECT_EQ(plan[snk_v].in_tids.front(), 1);
// }

// } // namespace
// } // namespace holoflow::runtime
