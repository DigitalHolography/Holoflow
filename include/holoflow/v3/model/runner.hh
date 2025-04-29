#pragma once

#include <atomic>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>

#include "holoflow/v3/model/model.hh"

using json = nlohmann::json;

namespace holoflow::model {

class Runner {
public:
  explicit Runner(Model &model);

  void start();
  void stop();

  void send_event(const std::string &name, const json &event);

private:
  void run();

  void allocate_pes(size_t pes_root);
  void free_pes(size_t pes_root);
  void exec_pes_rec(Vertex v);
  void exec_pes(size_t pes_root);

  Model &model_;
  std::thread thread_;
  std::atomic<bool> running_{false};
  std::atomic<bool> stop_{false};
};

} // namespace holoflow::model