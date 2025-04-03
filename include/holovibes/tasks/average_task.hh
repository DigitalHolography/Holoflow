#pragma once

#include <cuComplex.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <tl/expected.hpp>

#include "curaii/curaii.hh"
#include "holoflow/error.hh"
#include "holoflow/task.hh"

using json = nlohmann::json;

namespace dh {

class AverageTask : public Task {
public:
  tl::expected<void, Error> run(TensorView input, TensorView output) override;

  friend class AverageTaskFactory;

private:
  enum class Kind {
    U8_AVG,
    F32_AVG,
    CF32_AVG,
  };

  AverageTask(const TaskMeta &meta, cudaStream_t stream, int begin, int end,
              Kind kind);

  int begin_;
  int end_;
  Kind kind_;
};

class AverageTaskFactory : public TaskFactory {
public:
  tl::expected<TaskMeta, Error> type_check(const TensorMeta &imeta,
                                           const json &params);

  tl::expected<std::unique_ptr<Task>, Error>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream);

private:
  struct Params {
    int begin;
    int end;

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(Params, begin, end);
  };
};

} // namespace dh