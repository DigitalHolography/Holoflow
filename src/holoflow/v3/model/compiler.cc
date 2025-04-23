#include "holoflow/v3/model/compiler.hh"

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graphviz.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <stack>
#include <string>
#include <variant>
#include <vector>

#include "bug_buster/bug_buster.hh"
#include "curaii/v2/cuda.hh"
#include "holoflow/accumulator.hh"
#include "holoflow/holoflow.hh"
#include "holoflow/sink.hh"
#include "holoflow/source.hh"
#include "holoflow/task.hh"
#include "holoflow/v3/model/descriptor.hh"
#include "holoflow/v3/model/graph.hh"
#include "holoflow/v3/model/model.hh"

namespace holoflow::model {

void ModelCompiler::add_factory(const std::string &type,
                                std::unique_ptr<dh::SourceFactory> factory) {
  dh::holoflow_logger()->trace(
      "[ModelCompiler::add_factory] Adding source factory of type: {}", type);

  if (!factory) {
    throw std::invalid_argument("Factory cannot be null");
  }

  if (source_factories_.contains(type) || task_factories_.contains(type) ||
      sink_factories_.contains(type) || accumulator_factories_.contains(type)) {
    throw std::runtime_error("Factory already exists for type: " + type);
  }

  source_factories_[type] = std::move(factory);
}

void ModelCompiler::add_factory(const std::string &type,
                                std::unique_ptr<dh::SinkFactory> factory) {
  dh::holoflow_logger()->trace(
      "[ModelCompiler::add_factory] Adding sink factory of type: {}", type);

  if (!factory) {
    throw std::invalid_argument("Factory cannot be null");
  }

  if (source_factories_.contains(type) || task_factories_.contains(type) ||
      sink_factories_.contains(type) || accumulator_factories_.contains(type)) {
    throw std::runtime_error("Factory already exists for type: " + type);
  }

  sink_factories_[type] = std::move(factory);
}

void ModelCompiler::add_factory(const std::string &type,
                                std::unique_ptr<dh::TaskFactory> factory) {
  dh::holoflow_logger()->trace(
      "[ModelCompiler::add_factory] Adding task factory of type: {}", type);

  if (!factory) {
    throw std::invalid_argument("Factory cannot be null");
  }

  if (source_factories_.contains(type) || task_factories_.contains(type) ||
      sink_factories_.contains(type) || accumulator_factories_.contains(type)) {
    throw std::runtime_error("Factory already exists for type: " + type);
  }

  task_factories_[type] = std::move(factory);
}

void ModelCompiler::add_factory(
    const std::string &type, std::unique_ptr<dh::AccumulatorFactory> factory) {
  dh::holoflow_logger()->trace(
      "[ModelCompiler::add_factory] Adding accumulator factory of type: {}",
      type);

  if (!factory) {
    throw std::invalid_argument("Factory cannot be null");
  }

  if (source_factories_.contains(type) || task_factories_.contains(type) ||
      sink_factories_.contains(type) || accumulator_factories_.contains(type)) {
    throw std::runtime_error("Factory already exists for type: " + type);
  }

  accumulator_factories_[type] = std::move(factory);
}

void ModelCompiler::build_compiler_graph(const DescriptorGraph &graph) {
  for (const auto &v : boost::make_iterator_range(vertices(graph))) {
    const auto &node = graph[v];
    NodeProperties node_properties;
    node_properties.descriptor_ = node;

    node_properties.metrics_.num_executions_ =
        std::make_shared<std::atomic<int>>(0);
    node_properties.metrics_.last_reset_time_ =
        std::chrono::steady_clock::now();
    node_properties.metrics_.execution_throughput_ =
        std::make_shared<std::atomic<int>>(0);

    if (source_factories_.contains(node.type)) {
      node_properties.kind_ = NodeKind::Source;
      SourceProperties source_properties;
      node_properties.type_specific_ = source_properties;
    } else if (sink_factories_.contains(node.type)) {
      node_properties.kind_ = NodeKind::Sink;
      SinkProperties sink_properties;
      node_properties.type_specific_ = sink_properties;
    } else if (task_factories_.contains(node.type)) {
      node_properties.kind_ = NodeKind::Task;
      TaskProperties task_properties;
      node_properties.type_specific_ = task_properties;
    } else if (accumulator_factories_.contains(node.type)) {
      node_properties.kind_ = NodeKind::Accumulator;
      AccumulatorProperties accumulator_properties;
      node_properties.type_specific_ = accumulator_properties;
    } else {
      throw std::runtime_error("Unknown node type: " + node.type);
    }

    Vertex v_compiler = boost::add_vertex(node_properties, model_.graph_);
    model_.graph_[v_compiler].descriptor_ = node;
    model_.node_map_[node.id] = v_compiler;
  }

  for (const auto &e : boost::make_iterator_range(edges(graph))) {
    const auto &source = graph[boost::source(e, graph)];
    const auto &target = graph[boost::target(e, graph)];

    auto source_it = model_.node_map_.find(source.id);
    auto target_it = model_.node_map_.find(target.id);

    DH_CHECK(source_it != model_.node_map_.end(), "Source node not found: {}",
             source.id);
    DH_CHECK(target_it != model_.node_map_.end(), "Target node not found: {}",
             target.id);

    Vertex source_vertex = source_it->second;
    Vertex target_vertex = target_it->second;
    boost::add_edge(source_vertex, target_vertex, model_.graph_);
  }
}

void ModelCompiler::validate_no_orphan_nodes() const {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[node].kind_ == NodeKind::Source) {
      continue; // Skip source nodes
    }
    if (in_degree(node, model_.graph_) == 0) {
      throw std::runtime_error("Orphan node found: " +
                               model_.graph_[node].descriptor_.id);
    }
  }
}

void ModelCompiler::validate_single_parent_node() const {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[node].kind_ == NodeKind::Source) {
      continue; // Skip source nodes
    }
    if (in_degree(node, model_.graph_) > 1) {
      throw std::runtime_error("Node has multiple parents: " +
                               model_.graph_[node].descriptor_.id);
    }
  }
}

void ModelCompiler::validate_no_childless_nodes() const {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[node].kind_ == NodeKind::Sink) {
      continue; // Skip sink nodes
    }
    if (out_degree(node, model_.graph_) == 0) {
      throw std::runtime_error("Childless node found: " +
                               model_.graph_[node].descriptor_.id);
    }
  }
}

void ModelCompiler::validate_sources_are_orphan() const {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[node].kind_ == NodeKind::Source) {
      if (in_degree(node, model_.graph_) > 0) {
        throw std::runtime_error("Source node has parents: " +
                                 model_.graph_[node].descriptor_.id);
      }
    }
  }
}

void ModelCompiler::validate_sinks_are_childless() const {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[node].kind_ == NodeKind::Sink) {
      if (out_degree(node, model_.graph_) > 0) {
        throw std::runtime_error("Sink node has children: " +
                                 model_.graph_[node].descriptor_.id);
      }
    }
  }
}

void ModelCompiler::validate_source_is_unique() const {
  size_t source_count =
      std::count_if(boost::make_iterator_range(vertices(model_.graph_)).begin(),
                    boost::make_iterator_range(vertices(model_.graph_)).end(),
                    [this](const auto &node) {
                      return model_.graph_[node].kind_ == NodeKind::Source;
                    });

  if (source_count == 0) {
    throw std::runtime_error("No source nodes found in the graph");
  }

  if (source_count != 1) {
    throw std::runtime_error("Multiple source nodes found in the graph");
  }
}

struct AssignCudaStreamDFSVisitor : public boost::default_dfs_visitor {
  AssignCudaStreamDFSVisitor(Graph &graph,
                             std::vector<curaii::cuda::Stream> &streams)
      : graph_(graph), streams_(streams) {}

  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph &) {
    auto &node = graph_[u];

    // Source and accumulators are start point of asynchronous sub-pipeline.
    // They need to create a new stream.
    if (node.kind_ == NodeKind::Source || node.kind_ == NodeKind::Accumulator) {
      if (!node.common_.stream_) {
        curaii::cuda::Stream stream;
        node.common_.stream_ = stream.get();
        streams_.push_back(std::move(stream));
      }

      stream_stack_.push(*node.common_.stream_);
    }

    // Other nodes inherit the stream from the parent node.
    else {
      DH_CHECK(!stream_stack_.empty());
      node.common_.stream_ = stream_stack_.top();
    }
  }

  template <typename Vertex, typename Graph>
  void finish_vertex(Vertex u, const Graph &) {
    auto &node = graph_[u];
    if (node.kind_ == NodeKind::Source || node.kind_ == NodeKind::Accumulator) {
      stream_stack_.pop();
    }
  }

  Graph &graph_;
  std::vector<curaii::cuda::Stream> &streams_;
  std::stack<cudaStream_t> stream_stack_;
};

void ModelCompiler::assign_cuda_streams() {
  AssignCudaStreamDFSVisitor visitor(model_.graph_, model_.streams_);
  std::vector<boost::default_color_type> color_map(num_vertices(model_.graph_));
  auto color_map_property = boost::make_iterator_property_map(
      color_map.begin(), get(boost::vertex_index, model_.graph_));

  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[v].kind_ == NodeKind::Source &&
        color_map_property[v] == boost::white_color) {
      boost::depth_first_visit(model_.graph_, v, visitor, color_map_property);
    }
  }
}

struct TypeCheckingDFSVisitor : public boost::default_dfs_visitor {
  TypeCheckingDFSVisitor(
      Graph &graph,
      const std::map<std::string, std::unique_ptr<dh::SourceFactory>>
          &source_factories,
      const std::map<std::string, std::unique_ptr<dh::SinkFactory>>
          &sink_factories,
      const std::map<std::string, std::unique_ptr<dh::TaskFactory>>
          &task_factories,
      const std::map<std::string, std::unique_ptr<dh::AccumulatorFactory>>
          &accumulator_factories)
      : graph_(graph), source_factories_(source_factories),
        sink_factories_(sink_factories), task_factories_(task_factories),
        accumulator_factories_(accumulator_factories) {}

  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph &) {
    auto &node = graph_[u];
    if (node.kind_ == NodeKind::Source) {
      DH_CHECK(source_factories_.contains(node.descriptor_.type));
      auto factory = source_factories_.at(node.descriptor_.type).get();
      auto meta = factory->type_check(node.descriptor_.config);
      std::get<SourceProperties>(node.type_specific_).source_meta_ = meta;
      imeta_stack_.push(meta.ometa());
    } else if (node.kind_ == NodeKind::Sink) {
      DH_CHECK(!imeta_stack_.empty());
      DH_CHECK(sink_factories_.contains(node.descriptor_.type));
      auto factory = sink_factories_.at(node.descriptor_.type).get();
      auto imeta = imeta_stack_.top();
      auto meta = factory->type_check(imeta, node.descriptor_.config);
      std::get<SinkProperties>(node.type_specific_).sink_meta_ = meta;
    } else if (node.kind_ == NodeKind::Task) {
      DH_CHECK(!imeta_stack_.empty());
      DH_CHECK(task_factories_.contains(node.descriptor_.type));
      auto factory = task_factories_.at(node.descriptor_.type).get();
      auto imeta = imeta_stack_.top();
      auto meta = factory->type_check(imeta, node.descriptor_.config);
      std::get<TaskProperties>(node.type_specific_).task_meta_ = meta;
      imeta_stack_.push(meta.ometa());
    } else if (node.kind_ == NodeKind::Accumulator) {
      DH_CHECK(!imeta_stack_.empty());
      DH_CHECK(accumulator_factories_.contains(node.descriptor_.type));
      auto factory = accumulator_factories_.at(node.descriptor_.type).get();
      auto imeta = imeta_stack_.top();
      auto meta = factory->type_check(imeta, node.descriptor_.config);
      std::get<AccumulatorProperties>(node.type_specific_).accumulator_meta_ =
          meta;
      imeta_stack_.push(meta.ometa());
    }
  }

  template <typename Vertex, typename Graph>
  void finish_vertex(Vertex u, const Graph &) {
    auto &node = graph_[u];
    if (node.kind_ != NodeKind::Sink) {
      DH_CHECK(!imeta_stack_.empty());
      imeta_stack_.pop();
    }
  }

private:
  const std::map<std::string, std::unique_ptr<dh::SourceFactory>>
      &source_factories_;
  const std::map<std::string, std::unique_ptr<dh::SinkFactory>>
      &sink_factories_;
  const std::map<std::string, std::unique_ptr<dh::TaskFactory>>
      &task_factories_;
  const std::map<std::string, std::unique_ptr<dh::AccumulatorFactory>>
      &accumulator_factories_;

  Graph &graph_;
  std::stack<dh::TensorMeta> imeta_stack_;
};

void ModelCompiler::perform_type_checking() {
  TypeCheckingDFSVisitor visitor(model_.graph_, source_factories_,
                                 sink_factories_, task_factories_,
                                 accumulator_factories_);
  std::vector<boost::default_color_type> color_map(num_vertices(model_.graph_));
  auto color_map_property = boost::make_iterator_property_map(
      color_map.begin(), get(boost::vertex_index, model_.graph_));

  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[v].kind_ == NodeKind::Source &&
        color_map_property[v] == boost::white_color) {
      boost::depth_first_visit(model_.graph_, v, visitor, color_map_property);
    }
  }
}

void ModelCompiler::validate_single_inlined_or_accumulator_child() const {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    size_t nb_inlined_children = std::count_if(
        boost::make_iterator_range(out_edges(node, model_.graph_)).begin(),
        boost::make_iterator_range(out_edges(node, model_.graph_)).end(),
        [this](const auto &edge) {
          auto target = boost::target(edge, model_.graph_);
          auto task = model_.graph_[target];
          return model_.graph_[target].kind_ == NodeKind::Task &&
                 std::get<TaskProperties>(model_.graph_[target].type_specific_)
                     .task_meta_->inlined();
        });

    size_t nb_accumulator_children = std::count_if(
        boost::make_iterator_range(out_edges(node, model_.graph_)).begin(),
        boost::make_iterator_range(out_edges(node, model_.graph_)).end(),
        [this](const auto &edge) {
          auto target = boost::target(edge, model_.graph_);
          return model_.graph_[target].kind_ == NodeKind::Accumulator;
        });

    if (nb_inlined_children + nb_accumulator_children > 1) {
      throw std::runtime_error(
          "Node has more than one inlined or accumulator child: " +
          model_.graph_[node].descriptor_.id);
    }
  }
}

struct ValidateNonInlinedBetweenAccumulatorsDFSVisitor
    : public boost::default_dfs_visitor {
  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph &g) {
    auto &node = g[u];
    dh::holoflow_logger()->trace(
        "[ValidateNonInlinedBetweenAccumulatorsDFSVisitor::"
        "discover_vertex] Visiting node: "
        "{}",
        node.descriptor_.id);

    if (node.kind_ == NodeKind::Task &&
        std::get<TaskProperties>(node.type_specific_).task_meta_->inlined()) {
      DH_CHECK(!seen_non_inlined_.empty());
      seen_non_inlined_.push(seen_non_inlined_.top());
    } else if (node.kind_ == NodeKind::Accumulator) {
      seen_non_inlined_.push(false);
    } else {
      seen_non_inlined_.push(true);
    }
  }

  template <typename Vertex, typename Graph>
  void finish_vertex(Vertex, const Graph &) {
    DH_CHECK(!seen_non_inlined_.empty());
    seen_non_inlined_.pop();
  }

  template <typename Edge, typename Graph>
  void examine_edge(Edge e, const Graph &g) {
    auto source = boost::source(e, g);
    auto target = boost::target(e, g);
    auto &source_node = g[source];
    auto &target_node = g[target];

    dh::holoflow_logger()->trace("[ValidateNonInlinedBetweenAccumulatorsDFSVisi"
                                 "tor::examine_edge] Edge from {} "
                                 "to {}",
                                 source_node.descriptor_.id,
                                 target_node.descriptor_.id);

    DH_CHECK(!seen_non_inlined_.empty());
    if (target_node.kind_ == NodeKind::Accumulator &&
        !seen_non_inlined_.top()) {
      throw std::runtime_error("No non-inlined child found when reaching "
                               "accumulator node: " +
                               target_node.descriptor_.id);
    }
  }

private:
  std::stack<bool> seen_non_inlined_;
};

void ModelCompiler::validate_non_inlined_child_between_accumulators() const {
  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[v].kind_ == NodeKind::Accumulator) {
      dh::holoflow_logger()->debug(
          "[ModelCompiler::validate_non_inlined_child_between_accumulators] "
          "Starting DFS from accumulator node: {}",
          model_.graph_[v].descriptor_.id);

      ValidateNonInlinedBetweenAccumulatorsDFSVisitor visitor;
      std::vector<boost::default_color_type> color_map(
          num_vertices(model_.graph_));
      auto color_map_property = boost::make_iterator_property_map(
          color_map.begin(), get(boost::vertex_index, model_.graph_));

      bool visited_root = false;
      auto terminate_dfs = [&visited_root](Vertex v, const Graph &g) {
        if (!visited_root) {
          visited_root = true;
          return false;
        }

        return g[v].kind_ == NodeKind::Accumulator;
      };

      boost::depth_first_visit(model_.graph_, v, visitor, color_map_property,
                               terminate_dfs);
    }
  }
}

struct ResetTensorIdsVisitor {
  void operator()(SourceProperties &source) { source.otens_id_ = std::nullopt; }
  void operator()(SinkProperties &sink) { sink.itens_id_ = std::nullopt; }
  void operator()(TaskProperties &task) {
    task.itens_id_ = std::nullopt;
    task.otens_id_ = std::nullopt;
  }
  void operator()(AccumulatorProperties &accumulator) {
    accumulator.itens_id_ = std::nullopt;
    accumulator.otens_id_ = std::nullopt;
  }
};

void ModelCompiler::reset_non_accumulator_tensor_ids() {
  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    auto &node = model_.graph_[v];
    if (node.kind_ != NodeKind::Accumulator) {
      std::visit(ResetTensorIdsVisitor{}, node.type_specific_);
    }
  }
}

struct SetItensIdVisitor {
  SetItensIdVisitor(int id) : id_(id) {}

  void operator()(SourceProperties &) {
    DH_BUG("Source should not have "
           "itens_id_");
  }
  void operator()(SinkProperties &sink) { sink.itens_id_ = id_; }
  void operator()(TaskProperties &task) { task.itens_id_ = id_; }
  void operator()(AccumulatorProperties &accumulator) {
    accumulator.itens_id_ = id_;
  }

  int id_;
};

struct SetOtensIdVisitor {
  SetOtensIdVisitor(int id) : id_(id) {}

  void operator()(SourceProperties &source) { source.otens_id_ = id_; }
  void operator()(SinkProperties &) {
    DH_BUG("Sink should not have "
           "otens_id_");
  }
  void operator()(TaskProperties &task) { task.otens_id_ = id_; }
  void operator()(AccumulatorProperties &accumulator) {
    accumulator.otens_id_ = id_;
  }

  int id_;
};

void ModelCompiler::assign_accumulator_tensor_ids() {
  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    auto &node = model_.graph_[v];
    if (node.kind_ != NodeKind::Accumulator) {
      continue;
    }

    auto &accumulator = std::get<AccumulatorProperties>(node.type_specific_);
    // Set current node's itens_id_ and otens_id_
    if (!accumulator.itens_id_) {
      int new_tens_id = model_.next_tens_id_++;
      accumulator.itens_id_ = new_tens_id;
    }
    if (!accumulator.otens_id_) {
      int new_tens_id = model_.next_tens_id_++;
      accumulator.otens_id_ = new_tens_id;
    }

    // Set the itens_id_ of the children
    for (auto e : boost::make_iterator_range(out_edges(v, model_.graph_))) {
      auto target = boost::target(e, model_.graph_);
      auto &target_node = model_.graph_[target];
      std::visit(SetItensIdVisitor(*accumulator.otens_id_),
                 target_node.type_specific_);
    }

    // Set the otens_id_ of the parent and itens_id_ of the siblings
    for (auto e : boost::make_iterator_range(in_edges(v, model_.graph_))) {
      auto source = boost::source(e, model_.graph_);
      auto &source_node = model_.graph_[source];
      std::visit(SetOtensIdVisitor(*accumulator.itens_id_),
                 source_node.type_specific_);

      for (auto sibling_e :
           boost::make_iterator_range(out_edges(source, model_.graph_))) {
        auto sibling = boost::target(sibling_e, model_.graph_);
        auto &sibling_node = model_.graph_[sibling];
        std::visit(SetItensIdVisitor(*accumulator.itens_id_),
                   sibling_node.type_specific_);
      }
    }
  }
}

struct AssignInlinedTensorIdsDFSVisitor : public boost::default_dfs_visitor {
  AssignInlinedTensorIdsDFSVisitor(Graph &graph, int &next_tens_id)
      : next_tens_id_(next_tens_id), graph_(graph) {}

  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph &) {
    auto &node = graph_[u];
    if (node.kind_ != NodeKind::Task ||
        !std::get<TaskProperties>(node.type_specific_).task_meta_->inlined()) {
      return;
    }

    auto itens_id = std::get<TaskProperties>(node.type_specific_).itens_id_;
    if (!itens_id) {
      return;
    }

    // Set current node's otens_id_
    std::visit(SetOtensIdVisitor(*itens_id), node.type_specific_);

    // Set the itens_id_ of the children
    for (auto e : boost::make_iterator_range(out_edges(u, graph_))) {
      auto target = boost::target(e, graph_);
      auto &target_node = graph_[target];
      std::visit(SetItensIdVisitor(*itens_id), target_node.type_specific_);
    }
  }

  template <typename Vertex, typename Graph>
  void finish_vertex(Vertex u, const Graph &) {
    auto &node = graph_[u];
    if (node.kind_ != NodeKind::Task ||
        !std::get<TaskProperties>(node.type_specific_).task_meta_->inlined()) {
      return;
    }

    auto otens_id = std::get<TaskProperties>(node.type_specific_).otens_id_;
    if (!otens_id) {
      return;
    }

    // Set current node's itens_id_
    std::visit(SetItensIdVisitor(*otens_id), node.type_specific_);

    // Set the otens_id_ of the parent and itens_id_ of the siblings
    for (auto e : boost::make_iterator_range(in_edges(u, graph_))) {
      auto source = boost::source(e, graph_);
      auto &source_node = graph_[source];
      std::visit(SetOtensIdVisitor(*otens_id), source_node.type_specific_);

      for (auto sibling_e :
           boost::make_iterator_range(out_edges(source, graph_))) {
        auto sibling = boost::target(sibling_e, graph_);
        auto &sibling_node = graph_[sibling];
        std::visit(SetItensIdVisitor(*otens_id), sibling_node.type_specific_);
      }
    }
  }

private:
  Graph &graph_;
  int &next_tens_id_;
};

void ModelCompiler::assign_inlined_tensor_ids() {
  AssignInlinedTensorIdsDFSVisitor visitor(model_.graph_, model_.next_tens_id_);
  std::vector<boost::default_color_type> color_map(num_vertices(model_.graph_));
  auto color_map_property = boost::make_iterator_property_map(
      color_map.begin(), get(boost::vertex_index, model_.graph_));

  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[v].kind_ == NodeKind::Source &&
        color_map_property[v] == boost::white_color) {
      boost::depth_first_visit(model_.graph_, v, visitor, color_map_property);
    }
  }
}

struct AssignNonInlinedTensorIdsDFSVisitor : public boost::default_dfs_visitor {
  AssignNonInlinedTensorIdsDFSVisitor(Graph &graph, int &next_tens_id)
      : next_tens_id_(next_tens_id), graph_(graph) {}

  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph &) {
    auto &node = graph_[u];
    if (node.kind_ != NodeKind::Source &&
        (node.kind_ != NodeKind::Task ||
         std::get<TaskProperties>(node.type_specific_).task_meta_->inlined())) {
      return;
    }

    if (node.kind_ == NodeKind::Source) {
      auto otens_id = std::get<SourceProperties>(node.type_specific_).otens_id_;
      if (otens_id) {
        return;
      }
    }

    if (node.kind_ == NodeKind::Task) {
      auto otens_id = std::get<TaskProperties>(node.type_specific_).otens_id_;
      if (otens_id) {
        return;
      }
    }

    // Set current node's otens_id_
    int new_tens_id = next_tens_id_++;
    std::visit(SetOtensIdVisitor(new_tens_id), node.type_specific_);

    // Set the otens_id_ of the children
    for (auto e : boost::make_iterator_range(out_edges(u, graph_))) {
      auto target = boost::target(e, graph_);
      auto &target_node = graph_[target];
      std::visit(SetItensIdVisitor(new_tens_id), target_node.type_specific_);
    }
  }

private:
  Graph &graph_;
  int &next_tens_id_;
};

void ModelCompiler::assign_non_inlined_tensor_ids() {
  AssignNonInlinedTensorIdsDFSVisitor visitor(model_.graph_,
                                              model_.next_tens_id_);
  std::vector<boost::default_color_type> color_map(num_vertices(model_.graph_));
  auto color_map_property = boost::make_iterator_property_map(
      color_map.begin(), get(boost::vertex_index, model_.graph_));

  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    if (model_.graph_[v].kind_ == NodeKind::Source &&
        color_map_property[v] == boost::white_color) {
      boost::depth_first_visit(model_.graph_, v, visitor, color_map_property);
    }
  }
}

void ModelCompiler::call_factories() {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    auto &node_properties = model_.graph_[node];
    auto type = node_properties.descriptor_.type;
    auto config = node_properties.descriptor_.config;
    auto stream = *node_properties.common_.stream_;
    if (node_properties.kind_ == NodeKind::Source) {
      auto factory = source_factories_.at(type).get();
      auto source = factory->create(config, stream);
      std::get<SourceProperties>(node_properties.type_specific_).source_ =
          source.get();
      model_.sources_.push_back(std::move(source));
    } else if (node_properties.kind_ == NodeKind::Sink) {
      auto imeta = std::get<SinkProperties>(node_properties.type_specific_)
                       .sink_meta_->imeta();
      auto factory = sink_factories_.at(type).get();
      auto sink = factory->create(imeta, config, stream);
      std::get<SinkProperties>(node_properties.type_specific_).sink_ =
          sink.get();
      model_.sinks_.push_back(std::move(sink));
    } else if (node_properties.kind_ == NodeKind::Task) {
      auto imeta = std::get<TaskProperties>(node_properties.type_specific_)
                       .task_meta_->imeta();
      auto factory = task_factories_.at(type).get();
      auto task = factory->create(imeta, config, stream);
      std::get<TaskProperties>(node_properties.type_specific_).task_ =
          task.get();
      model_.tasks_.push_back(std::move(task));
    } else if (node_properties.kind_ == NodeKind::Accumulator) {
      auto imeta =
          std::get<AccumulatorProperties>(node_properties.type_specific_)
              .accumulator_meta_->imeta();
      auto factory = accumulator_factories_.at(type).get();
      auto accumulator = factory->create(imeta, config, stream);
      std::get<AccumulatorProperties>(node_properties.type_specific_)
          .accumulator_ = accumulator.get();
      model_.accumulators_.push_back(std::move(accumulator));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
}

struct CreateTensorSlotsVisitor {
  CreateTensorSlotsVisitor(std::map<int, TensorSlot> &tensor_slots)
      : tensor_slots_(tensor_slots) {}

  void operator()(SourceProperties &source) {
    tensor_slots_.emplace(*source.otens_id_,
                          TensorSlot(source.source_meta_->ometa()));
  }
  void operator()(SinkProperties &sink) {
    tensor_slots_.emplace(*sink.itens_id_,
                          TensorSlot(sink.sink_meta_->imeta()));
  }
  void operator()(TaskProperties &task) {
    tensor_slots_.emplace(*task.itens_id_,
                          TensorSlot(task.task_meta_->imeta()));
    tensor_slots_.emplace(*task.otens_id_,
                          TensorSlot(task.task_meta_->ometa()));
  }
  void operator()(AccumulatorProperties &accumulator) {
    tensor_slots_.emplace(*accumulator.itens_id_,
                          TensorSlot(accumulator.accumulator_meta_->imeta()));
    tensor_slots_.emplace(*accumulator.otens_id_,
                          TensorSlot(accumulator.accumulator_meta_->ometa()));
  }

private:
  std::map<int, TensorSlot> &tensor_slots_;
};

void ModelCompiler::create_tensor_slots() {
  for (const auto &node : boost::make_iterator_range(vertices(model_.graph_))) {
    auto &node_properties = model_.graph_[node];
    std::visit(CreateTensorSlotsVisitor(model_.tensor_slots_),
               node_properties.type_specific_);
  }
}

struct ContainsTensorIdVisitor {
  ContainsTensorIdVisitor(int tens_id) : tens_id_(tens_id) {}

  bool operator()(const SourceProperties &source) {
    return source.otens_id_ == tens_id_;
  }
  bool operator()(const SinkProperties &sink) {
    return sink.itens_id_ == tens_id_;
  }
  bool operator()(const TaskProperties &task) {
    return task.itens_id_ == tens_id_ || task.otens_id_ == tens_id_;
  }
  bool operator()(const AccumulatorProperties &accumulator) {
    return accumulator.itens_id_ == tens_id_ ||
           accumulator.otens_id_ == tens_id_;
  }

private:
  int tens_id_;
};

bool ModelCompiler::is_accumulator_tensor(int tens_id) const {
  return std::any_of(
      boost::make_iterator_range(vertices(model_.graph_)).begin(),
      boost::make_iterator_range(vertices(model_.graph_)).end(),
      [this, tens_id](const auto &node) {
        return model_.graph_[node].kind_ == NodeKind::Accumulator &&
               std::visit(ContainsTensorIdVisitor(tens_id),
                          model_.graph_[node].type_specific_);
      });
}

void ModelCompiler::allocate_non_accumulator_tensor_slots() {
  for (auto &[id, slot] : model_.tensor_slots_) {
    if (is_accumulator_tensor(id)) {
      continue;
    }

    switch (slot.meta.memory_location()) {
    case dh::MemoryLocation::HOST:
      dh::holoflow_logger()->debug("allocation tens_id: {} on host", id);
      slot.host_data = curaii::cuda::make_unique_host_ptr<uint8_t>(
          slot.meta.size_in_bytes());
      slot.device_data.reset();
      slot.data = slot.host_data.get();
      break;
    case dh::MemoryLocation::DEVICE:
      dh::holoflow_logger()->debug("allocation tens_id: {} on device", id);
      slot.device_data = curaii::cuda::make_unique_device_ptr<uint8_t>(
          slot.meta.size_in_bytes());
      slot.host_data.reset();
      slot.data = slot.device_data.get();
      break;
    }
  }
}

void ModelCompiler::select_pes_roots() {
  for (auto v : boost::make_iterator_range(vertices(model_.graph_))) {
    auto &node = model_.graph_[v];
    if (node.kind_ == NodeKind::Source || node.kind_ == NodeKind::Accumulator) {
      model_.pes_roots_.push_back(v);
    }
  }
}

Model ModelCompiler::compile(const DescriptorGraph &descriptor_graph) {
  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] Compiling model from descriptor graph");

  model_ = std::move(Model());

  dh::holoflow_logger()->debug("[ModelCompiler::compile] building compiler "
                               "graph from descriptor graph");
  build_compiler_graph(descriptor_graph);

  dh::holoflow_logger()->debug("[ModelCompiler::compile] validating no "
                               "orphan nodes in compiler graph");
  validate_no_orphan_nodes();

  dh::holoflow_logger()->debug("[ModelCompiler::compile] validating single "
                               "parent node in compiler graph");
  validate_single_parent_node();

  dh::holoflow_logger()->debug("[ModelCompiler::compile] validating no "
                               "childless nodes in compiler graph");
  validate_no_childless_nodes();

  dh::holoflow_logger()->debug("[ModelCompiler::compile] validating sources "
                               "are orphan in compiler graph");
  validate_sources_are_orphan();

  dh::holoflow_logger()->debug("[ModelCompiler::compile] validating sinks "
                               "are childless in compiler graph");
  validate_sinks_are_childless();

  dh::holoflow_logger()->debug("[ModelCompiler::compile] validating source "
                               "is unique in compiler graph");
  validate_source_is_unique();

  dh::holoflow_logger()->debug("[ModelCompiler::compile] assigning CUDA "
                               "streams to nodes in compiler graph");
  assign_cuda_streams();

  dh::holoflow_logger()->debug("[ModelCompiler::compile] performing type "
                               "checking on nodes in compiler graph");
  perform_type_checking();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] validating single inlined or accumulator "
      "child in compiler graph");
  validate_single_inlined_or_accumulator_child();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] validating non-inlined child between "
      "accumulators in compiler graph");
  validate_non_inlined_child_between_accumulators();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] resetting non-accumulator tensor IDs in "
      "compiler graph");
  reset_non_accumulator_tensor_ids();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] assigning accumulator tensor IDs in "
      "compiler graph");
  assign_accumulator_tensor_ids();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] assigning inlined tensor IDs in compiler "
      "graph (1st pass)");
  assign_inlined_tensor_ids();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] assigning non-inlined tensor IDs in compiler "
      "graph");
  assign_non_inlined_tensor_ids();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] assigning inlined tensor IDs in compiler "
      "graph (2nd pass)");
  assign_inlined_tensor_ids();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] calling factories in compiler graph");
  call_factories();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] creating tensor slots in compiler graph");
  create_tensor_slots();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] allocating non-accumulator tensor slots in "
      "compiler graph");
  allocate_non_accumulator_tensor_slots();

  dh::holoflow_logger()->debug(
      "[ModelCompiler::compile] selecting PES roots in compiler graph");
  select_pes_roots();

  return std::move(model_);
}

} // namespace holoflow::model