#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>

#include "curaii/cuda_runtime.hh"
#include "holoflow/error.hh"
#include "holoflow/tensor.hh"

using json = nlohmann::json;

namespace dh {

/**
 * @brief Metadata for a computational task.
 *
 * This class defines the input and output tensor metadata, as well as whether
 * the task is inlined (i.e., modifying the input in place).
 */
class TaskMeta {
public:
  /**
   * @brief Constructs task metadata.
   * @param imeta Metadata for the input tensor.
   * @param ometa Metadata for the output tensor.
   * @param inlined Whether the task modifies the input in place.
   */
  TaskMeta(const TensorMeta &imeta, const TensorMeta &ometa, bool inlined);

  /**
   * @brief Returns the input tensor metadata.
   * @return A reference to the input TensorMeta.
   */
  const TensorMeta &imeta() const;

  /**
   * @brief Returns the output tensor metadata.
   * @return A reference to the output TensorMeta.
   */
  const TensorMeta &ometa() const;

  /**
   * @brief Checks if the task is inlined.
   * @return True if the task modifies the input tensor in place.
   */
  bool inlined() const;

private:
  TensorMeta imeta_; ///< Input tensor metadata.
  TensorMeta ometa_; ///< Output tensor metadata.
  bool inlined_;     ///< Whether the task modifies the input tensor in place.
};

/**
 * @brief Represents a computational task that operates on tensors.
 *
 * A Task is an abstract base class for GPU operations on input tensors,
 * producing an output tensor. Each task is associated with a CUDA stream for
 * execution.
 */
class Task {
public:
  /**
   * @brief Constructs a task with the given metadata and stream.
   * @param meta The task metadata.
   * @param stream The stream.
   */
  Task(const TaskMeta &meta, CudaStreamRef stream);

  /**
   * @brief Virtual destructor.
   */
  virtual ~Task() = default;

  // Disable copy and move operations.
  Task(const Task &) = delete;
  Task &operator=(const Task &) = delete;
  Task(Task &&) = delete;
  Task &operator=(Task &&) = delete;

  /**
   * @brief Executes the task using input and output tensor views.
   * @param itens The input tensor view.
   * @param otens The output tensor view.
   * @return A tl::expected indicating success or an error.
   */
  virtual void run(TensorView itens, TensorView otens) = 0;

  /**
   * @brief Returns the metadata associated with the task.
   * @return A reference to the TaskMeta.
   */
  const TaskMeta &meta() const;

  /**
   * @brief Returns the metadata of the input tensor.
   * @return A reference to the input TensorMeta.
   */
  const TensorMeta &imeta() const;

  /**
   * @brief Returns the metadata of the output tensor.
   * @return A reference to the output TensorMeta.
   */
  const TensorMeta &ometa() const;

  /**
   * @brief Checks if the task is inlined.
   * @return True if the task modifies the input tensor in place.
   */
  bool inlined() const;

protected:
  TaskMeta meta_; ///< Metadata defining input/output tensors and inlining.
  CudaStreamRef stream_; ///< CUDA stream associated with the task execution.
};

/**
 * @brief Factory interface for creating tasks.
 *
 * A TaskFactory provides a mechanism for type-checking input tensors and
 * creating new Task instances based on parameters and input metadata.
 */
class TaskFactory {
public:
  /**
   * @brief Default constructor.
   */
  TaskFactory() = default;

  /**
   * @brief Virtual destructor.
   */
  virtual ~TaskFactory() = default;

  // Disable copy and move operations.
  TaskFactory(const TaskFactory &) = delete;
  TaskFactory &operator=(const TaskFactory &) = delete;
  TaskFactory(TaskFactory &&) = delete;
  TaskFactory &operator=(TaskFactory &&) = delete;

  /**
   * @brief Checks whether the input metadata is valid for a specific task type.
   * @param imeta The input tensor metadata.
   * @param params Additional parameters in JSON format.
   * @return A tl::expected containing the produced TaskMeta or an error.
   */
  virtual TaskMeta type_check(const TensorMeta &imeta, const json &params) = 0;

  /**
   * @brief Creates a task instance with the given parameters.
   * @param imeta The input tensor metadata.
   * @param params Additional parameters in JSON format.
   * @param stream The CUDA stream to associate with the task.
   * @return A tl::expected containing a unique_ptr to a Task or an error.
   *
   * @note The stream is not owned by the task and must be kept alive until the
   * task is destroyed.
   */
  virtual std::unique_ptr<Task>
  create(const TensorMeta &imeta, const json &params, CudaStreamRef stream) = 0;
};

} // namespace dh