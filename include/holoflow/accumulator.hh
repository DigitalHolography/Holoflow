#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>

#include "holoflow/error.hh"
#include "holoflow/tensor.hh"

using json = nlohmann::json;

namespace dh {

/**
 * @brief Metadata for an accumulator.
 *
 * This class defines the input and output tensor metadata.
 */
class AccumulatorMeta {
public:
  /**
   * @brief Constructs accumulator metadata.
   * @param imeta Metadata for the input tensor.
   * @param ometa Metadata for the output tensor.
   */
  AccumulatorMeta(const TensorMeta &imeta, const TensorMeta &ometa);

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

private:
  TensorMeta imeta_; ///< Input tensor metadata.
  TensorMeta ometa_; ///< Output tensor metadata.
};

/**
 * @brief Represents an accumulator that operates on tensors.
 *
 * An Accumulator is an abstract base class that act as a FIFO structure for
 * tensors.
 *
 * It manages the memory for its input and output tensors, meaning they do not
 * need external allocation.
 *
 * It allows merging and splitting tensors on their first dimension (usually the
 * batch size). i.e. (32x128x128) + (32x128x128) = (64x128x128), and
 * (64x128x128) = (32x128x128) + (32x128x128). This can be used to resize a
 * batch of images for instance.
 */
class Accumulator {
public:
  /**
   * @brief Constructs an accumulator with the given metadata.
   * @param meta The task metadata.
   * @param stream The stream.
   */
  Accumulator(const AccumulatorMeta &meta, cudaStream_t stream);

  /**
   * @brief Virtual destructor.
   */
  virtual ~Accumulator() = default;

  // Disable copy and move operations.
  Accumulator(const Accumulator &) = delete;
  Accumulator &operator=(const Accumulator &) = delete;
  Accumulator(Accumulator &&) = delete;
  Accumulator &operator=(Accumulator &&) = delete;

  /**
   * @brief Provide a tensor that the user can write to.
   * @return A tl::expected indicating success or an error.
   *
   * @note A full queue is considered as a success therefore the user must check
   * if the TensorView returned is null.
   *
   * @note The tensor written will be accessible for reading only after
   * `commit_write` has been called. Not calling this function is the same as
   * discarding the write operation.
   */
  virtual std::optional<TensorView> write_tensor() = 0;

  /**
   * @brief Commit the write operation.
   * @return A tl::expected indicating success or an error.
   *
   * @warning The user must have witten to the tensor returned by write_tensor
   * entirely before calling this function.
   *
   * @warning Calling this function invalidates the tensor provided by
   * write_tensor. The user must not use it after.
   */
  virtual void commit_write() = 0;

  /**
   * @brief Provide a tensor that the user can read from.
   * @return A tl::expected indicating success or an error.
   *
   * @note An empty queue is considered as a success therefore the user must
   * check if the TensorView returned is null.
   *
   * @note The tensor read will be accessible for writting only after
   * `commit_read` has been called. Not calling this function is the same as
   * discarding the read operation.
   */
  virtual std::optional<TensorView> read_tensor() = 0;

  /**
   * @brief Commit the read operation.
   * @return A tl::expected indicating success or an error.
   *
   * @warning Calling this function invalidates the tensor provided by
   * read_tensor. The user must not use it after.
   */
  virtual void commit_read() = 0;

  /**
   * @brief Returns the metadata associated with the accumulator.
   * @return A reference to the AccumulatorMeta.
   */
  const AccumulatorMeta &meta() const;

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

protected:
  AccumulatorMeta meta_; ///< Metadata defining input/output tensors.
  cudaStream_t stream_;  ///< CUDA stream associated with the accumulator.
};

/**
 * @brief Factory interface for creating accumulators.
 *
 * An AccumulatorFactory provides a mechanism for type-checking input tensors
 * and creating new Accumulators instances based on parameters and input
 * metadata.
 */
class AccumulatorFactory {
public:
  /**
   * @brief Default constructor.
   */
  AccumulatorFactory() = default;

  /**
   * @brief Virtual destructor.
   */
  virtual ~AccumulatorFactory() = default;

  // Disable copy and move operations.
  AccumulatorFactory(const AccumulatorFactory &) = delete;
  AccumulatorFactory &operator=(const AccumulatorFactory &) = delete;
  AccumulatorFactory(AccumulatorFactory &&) = delete;
  AccumulatorFactory &operator=(AccumulatorFactory &&) = delete;

  /**
   * @brief Checks whether the input metadata is valid for a specific
   * accumulator type.
   * @param imeta The input tensor metadata.
   * @param params Additional parameters in JSON format.
   * @return A tl::expected containing the produced AccumulatorMeta or an error.
   */
  virtual AccumulatorMeta type_check(const TensorMeta &imeta,
                                     const json &params) = 0;

  /**
   * @brief Creates an accumulator instance with the given parameters.
   * @param imeta The input tensor metadata.
   * @param params Additional parameters in JSON format.
   * @param stream The CUDA stream to associate with the accumulator.
   * @return A tl::expected containing a unique_ptr to an Accumulator or an
   * error.
   *
   * @note The stream is not owned by the accumulator and must be kept alive
   * until the accumulator is destroyed.
   */
  virtual std::unique_ptr<Accumulator>
  create(const TensorMeta &imeta, const json &params, cudaStream_t stream) = 0;
};

} // namespace dh