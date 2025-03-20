#pragma once

#include <cstdlib>
#include <glog/logging.h>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

namespace dh {

/**
 * @brief Enumeration representing memory locations of tensor data.
 */
enum class MemoryLocation {
  HOST,  ///< Memory located on the host (CPU).
  DEVICE ///< Memory located on the device (GPU).
};

/**
 * @brief Overload stream operator to print MemoryLocation.
 */
std::ostream &operator<<(std::ostream &os, MemoryLocation loc);

/**
 * @brief Enumeration representing different data types for tensors.
 */
enum class DataType {
  U8,  ///< Unsigned 8-bit integer.
  U16, ///< Unsigned 16-bit integer.
  F32, ///< 32-bit floating-point number.
  CF32 ///< Complex 32-bit floating-point number (2 * float32).
};

/**
 * @brief Returns the size in bytes of the given DataType.
 * @param data_type The DataType to query.
 * @return The size in bytes of the specified DataType.
 */
size_t size_of(DataType data_type);

/**
 * @brief Overload stream operator to print DataType.
 */
std::ostream &operator<<(std::ostream &os, DataType dtype);

/**
 * @brief Holds metadata for a tensor, including shape, strides, data type, and
 * memory location.
 */
class TensorMeta {
public:
  /**
   * @brief Constructs a TensorMeta with automatically computed row-major
   * strides.
   * @param data_type The data type of the tensor.
   * @param memory_location The memory location (HOST or DEVICE).
   * @param shape The dimensions of the tensor.
   */
  TensorMeta(DataType data_type, MemoryLocation memory_location,
             std::vector<size_t> shape);

  /**
   * @brief Constructs a TensorMeta with explicitly specified strides.
   * @param data_type The data type of the tensor.
   * @param memory_location The memory location (HOST or DEVICE).
   * @param shape The dimensions of the tensor.
   * @param strides The strides of the tensor.
   */
  TensorMeta(DataType data_type, MemoryLocation memory_location,
             std::vector<size_t> shape, std::vector<size_t> strides);

  /**
   * @brief Computes the total number of elements in the tensor.
   * @return The number of elements in the tensor.
   */
  size_t size() const;

  /**
   * @brief Computes the total memory size in bytes of the tensor.
   * @return The memory size in bytes.
   */
  size_t size_in_bytes() const;

  /**
   * @brief Gets the shape (dimensions) of the tensor.
   * @return A reference to the shape vector.
   */
  const std::vector<size_t> &shape() const;

  /**
   * @brief Gets the strides of the tensor.
   * @return A reference to the strides vector.
   */
  const std::vector<size_t> &strides() const;

  /**
   * @brief Gets the data type of the tensor elements.
   * @return The data type of the tensor.
   */
  DataType data_type() const;

  /**
   * @brief Gets the memory location of the tensor.
   * @return The memory location (HOST or DEVICE).
   */
  MemoryLocation memory_location() const;

  /**
   * @brief Overload stream operator to print TensorMeta.
   */
  friend std::ostream &operator<<(std::ostream &os, const TensorMeta &meta);

private:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  DataType data_type_;
  MemoryLocation memory_location_;
};

/**
 * @brief A non-owning view over tensor data.
 *
 * This class provides a view over an existing memory buffer, allowing access to
 * the tensor's metadata (shape, strides, data type, and memory location)
 * without managing the underlying memory.
 */
class TensorView {
public:
  /**
   * @brief Constructs a TensorView with a given data pointer and metadata.
   * @param data Pointer to the tensor data (CPU or GPU).
   * @param meta The tensor metadata (shape, strides, type, memory location).
   */
  TensorView(void *data, const TensorMeta &meta);

  /**
   * @brief Returns a mutable pointer to the tensor's data.
   * @return A pointer to the tensor's data.
   */
  void *data();

  /**
   * @brief Returns a const pointer to the tensor's data.
   * @return A const pointer to the tensor's data.
   */
  const void *data() const;

  /**
   * @brief Computes the total number of elements in the tensor.
   * @return The number of elements in the tensor.
   */
  size_t size() const;

  /**
   * @brief Computes the total memory size in bytes of the tensor.
   * @return The memory size in bytes.
   */
  size_t size_in_bytes() const;

  /**
   * @brief Gets the tensor metadata.
   * @return A reference to the TensorMeta structure.
   */
  const TensorMeta &meta() const;

  /**
   * @brief Gets the tensor shape.
   * @return A reference to the shape.
   */
  const std::vector<size_t> &shape() const;

  /**
   * @brief Gets the tensor strides.
   * @return A reference to the strides.
   */
  const std::vector<size_t> &strides() const;

  /**
   * @brief Gets the tensor data type.
   * @return A copy of the data type.
   */
  DataType data_type() const;

  /**
   * @brief Gets the tensor data location.
   * @return A copy of the data location.
   */
  MemoryLocation memory_location() const;

  /**
   * @brief Overload stream operator to print TensorView.
   */
  friend std::ostream &operator<<(std::ostream &os, const TensorView &view);

private:
  void *data_;             ///< Pointer to the tensor's data (CPU or GPU).
  const TensorMeta &meta_; ///< Metadata describing the tensor.
};

} // namespace dh
