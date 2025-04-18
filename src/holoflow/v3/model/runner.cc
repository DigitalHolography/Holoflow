#include "holoflow/v3/model/runner.hh"

#include <atomic>
#include <boost/graph/depth_first_search.hpp>
#include <chrono>
#include <fmt/core.h>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>
#include <thread>
#include <vector>

#include "bug_buster/bug_buster.hh"
#include "holoflow/holoflow.hh"
#include "holoflow/v3/model/model.hh"

namespace holoflow::model {

Runner::Runner(Model &model) : model_(model) {}

void Runner::start() {
  if (running_.exchange(true)) {
    throw std::runtime_error("Runner is already running");
  }

  curaii::cuda::peek_at_last_error();

  stop_.store(false);
  thread_ = std::thread([this]() { run(); });
}

void Runner::stop() {
  if (stop_.exchange(true)) {
    throw std::runtime_error("Runner is already stopping");
  }

  stop_.store(true);
  thread_.join();

  if (!running_.exchange(false)) {
    throw std::runtime_error("Runner is not running");
  }
}

struct AllocatePesDFSVisitor : public boost::default_dfs_visitor {
  AllocatePesDFSVisitor(std::map<int, TensorSlot> &tensor_slots,
                        size_t pes_root, std::atomic<bool> &stop)
      : tensor_slots_(tensor_slots), pes_root_(pes_root), stop_(stop) {}

  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph &g) {
    if (stop_.load()) {
      dh::holoflow_logger()->debug(
          "[Runner::allocate_pes] stop_ is set, stopping DFS");
      return;
    }

    if (u == pes_root_ && g[u].kind_ == NodeKind::Accumulator) {
      auto prop = std::get<AccumulatorProperties>(g[u].type_specific_);
      auto view = prop.accumulator_->read_tensor();
      while (!view && !stop_.load()) {
        view = prop.accumulator_->read_tensor();
      }

      if (stop_.load()) {
        dh::holoflow_logger()->debug(
            "[Runner::allocate_pes] stop_ is set, stopping DFS");
        return;
      }

      auto &otens = tensor_slots_.at(*prop.otens_id_);
      otens.data = reinterpret_cast<uint8_t *>(view->data());
    } else if (g[u].kind_ == NodeKind::Accumulator) {
      auto prop = std::get<AccumulatorProperties>(g[u].type_specific_);
      auto view = prop.accumulator_->write_tensor();
      while (!view && !stop_.load()) {
        view = prop.accumulator_->write_tensor();
      }

      if (stop_.load()) {
        dh::holoflow_logger()->debug(
            "[Runner::allocate_pes] stop_ is set, stopping DFS");
        return;
      }

      auto &itens = tensor_slots_.at(*prop.itens_id_);
      itens.data = reinterpret_cast<uint8_t *>(view->data());
    }
  }

private:
  std::map<int, TensorSlot> &tensor_slots_;
  std::atomic<bool> &stop_;
  size_t pes_root_;
};

void Runner::allocate_pes(size_t pes_root) {
  AllocatePesDFSVisitor visitor(model_.tensor_slots_, pes_root, stop_);
  std::vector<boost::default_color_type> color_map(num_vertices(model_.graph_));
  auto color_map_property = boost::make_iterator_property_map(
      color_map.begin(), get(boost::vertex_index, model_.graph_));

  bool visited_root = false;
  auto terminate_dfs = [&visited_root, this](Vertex v, const Graph &g) {
    if (stop_.load()) {
      dh::holoflow_logger()->debug(
          "[Runner::allocate_pes] stop_ is set, stopping DFS");
      return true;
    }

    if (!visited_root) {
      visited_root = true;
      return false;
    }

    return g[v].kind_ == NodeKind::Accumulator;
  };

  boost::depth_first_visit(model_.graph_, pes_root, visitor, color_map_property,
                           terminate_dfs);
}

struct FreePesDFSVisitor : public boost::default_dfs_visitor {
  FreePesDFSVisitor(std::map<int, TensorSlot> &tensor_slots, size_t pes_root,
                    std::atomic<bool> &stop)
      : tensor_slots_(tensor_slots), pes_root_(pes_root), stop_(stop) {}

  template <typename Vertex, typename Graph>
  void discover_vertex(Vertex u, const Graph &g) {
    if (stop_.load()) {
      dh::holoflow_logger()->debug(
          "[Runner::free_pes] stop_ is set, stopping DFS");
      return;
    }

    if (u == pes_root_ && g[u].kind_ == NodeKind::Accumulator) {
      auto prop = std::get<AccumulatorProperties>(g[u].type_specific_);
      prop.accumulator_->commit_read();
    } else if (g[u].kind_ == NodeKind::Accumulator) {
      auto prop = std::get<AccumulatorProperties>(g[u].type_specific_);
      prop.accumulator_->commit_write();
    }
  }

private:
  std::map<int, TensorSlot> &tensor_slots_;
  std::atomic<bool> &stop_;
  size_t pes_root_;
};

void Runner::free_pes(size_t pes_root) {
  FreePesDFSVisitor visitor(model_.tensor_slots_, pes_root, stop_);
  std::vector<boost::default_color_type> color_map(num_vertices(model_.graph_));
  auto color_map_property = boost::make_iterator_property_map(
      color_map.begin(), get(boost::vertex_index, model_.graph_));

  bool visited_root = false;
  auto terminate_dfs = [&visited_root, this](Vertex v, const Graph &g) {
    if (stop_.load()) {
      dh::holoflow_logger()->debug(
          "[Runner::free_pes] stop_ is set, stopping DFS");
      return true;
    }

    if (!visited_root) {
      visited_root = true;
      return false;
    }

    return g[v].kind_ == NodeKind::Accumulator;
  };

  boost::depth_first_visit(model_.graph_, pes_root, visitor, color_map_property,
                           terminate_dfs);
}

struct ExecNodeVisitor {
  ExecNodeVisitor(std::map<int, TensorSlot> &tensor_slots)
      : tensor_slots_(tensor_slots) {}

  void operator()(SourceProperties &source) {
    auto otens = tensor_slots_.at(*source.otens_id_).view();
    DH_CHECK(source.source_->run(otens));
  }
  void operator()(SinkProperties &sink) {
    auto itens = tensor_slots_.at(*sink.itens_id_).view();
    DH_CHECK(sink.sink_->run(itens));
  }
  void operator()(TaskProperties &task) {
    auto itens = tensor_slots_.at(*task.itens_id_).view();
    auto otens = tensor_slots_.at(*task.otens_id_).view();
    DH_CHECK(task.task_->run(itens, otens));
  }
  void operator()(AccumulatorProperties &) {}

private:
  std::map<int, TensorSlot> &tensor_slots_;
};

void Runner::exec_pes_rec(Vertex v) {
  auto &node = model_.graph_[v];

  dh::holoflow_logger()->debug("[Runner::exec_pes_rec] Executing node {}",
                               node.descriptor_.id);

  if (stop_.load()) {
    dh::holoflow_logger()->debug(
        "[Runner::exec_pes_rec] stop_ is set, stopping DFS");
    return;
  }

  ExecNodeVisitor visitor(model_.tensor_slots_);
  std::visit(visitor, node.type_specific_);

  // Metrics are updated every 5s
  node.metrics_.num_executions_->fetch_add(1);
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
      now - node.metrics_.last_reset_time_);
  if (duration.count() > 5e6) {
    auto num_executions = node.metrics_.num_executions_->load();
    auto throughput = num_executions / (duration.count() / 1e6);
    node.metrics_.execution_throughput_->store((int)throughput);
    node.metrics_.last_reset_time_ = now;
    node.metrics_.num_executions_->store(0);

    // If source display throughput
    if (node.kind_ == NodeKind::Source) {
      auto prop = std::get<SourceProperties>(node.type_specific_);
      auto ometa = prop.source_meta_->ometa();
      auto batch_size = ometa.shape().at(0);
      dh::holoflow_logger()->info(
          "[Runner::exec_pes_rec] Source {} throughput: {}",
          node.descriptor_.id, throughput * (int)batch_size);
    }
  }

  auto is_inlined = [](const auto &node) {
    return node.kind_ == NodeKind::Task &&
           std::get<TaskProperties>(node.type_specific_).task_meta_->inlined();
  };

  // Process non-inlined children first
  for (const auto &child :
       boost::make_iterator_range(out_edges(v, model_.graph_))) {
    auto child_v = target(child, model_.graph_);
    auto child_node = model_.graph_[child_v];
    if (is_inlined(child_node) || child_node.kind_ == NodeKind::Accumulator) {
      continue;
    }

    exec_pes_rec(child_v);
  }

  // Process inlined children
  for (const auto &child :
       boost::make_iterator_range(out_edges(v, model_.graph_))) {
    auto child_v = target(child, model_.graph_);
    auto child_node = model_.graph_[child_v];
    if (!is_inlined(child_node) || child_node.kind_ == NodeKind::Accumulator) {
      continue;
    }

    exec_pes_rec(child_v);
  }
}

void Runner::exec_pes(Vertex pes_root) { exec_pes_rec(pes_root); }

void Runner::run() {
  std::vector<std::thread> threads;

  for (auto &pes_root : model_.pes_roots_) {
    threads.emplace_back([this, &pes_root]() {
      auto &name = model_.graph_[pes_root].descriptor_.id;
      dh::holoflow_logger()->debug("[Runner::run] PES \"{}\" started", name);
      while (!stop_.load()) {
        auto &node = model_.graph_[pes_root];
        dh::holoflow_logger()->debug("[Runner::run] Executing pes root {}",
                                     node.descriptor_.id);

        nvtxRangePush(fmt::format("[PES::{}] allocate", name).c_str());
        allocate_pes(pes_root);
        node.common_.stream_->synchronize();
        nvtxRangePop();

        nvtxRangePush(fmt::format("[PES::{}] exec", name).c_str());
        exec_pes(pes_root);
        node.common_.stream_->synchronize();
        nvtxRangePop();

        nvtxRangePush(fmt::format("[PES::{}] free", name).c_str());
        free_pes(pes_root);
        node.common_.stream_->synchronize();
        nvtxRangePop();
      }

      dh::holoflow_logger()->debug("[Runner::run] PES \"{}\" stopped", name);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  dh::holoflow_logger()->info("[Runner::run] model stoped");
}

} // namespace holoflow::model