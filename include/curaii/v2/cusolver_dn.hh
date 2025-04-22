// include/curaii/v2/cusolver_dn.hh
/**
 * @file include/curaii/v2/cusolver_dn.hh
 * @author Jules Guillou
 * @brief RAII wrappers and error‑checking macros for cuSOLVER‑DN.
 *
 * This header provides:
 *  - RAII wrapper for cusolverDnHandle_t.
 *  - RAII wrapper for cusolverDnParams_t.
 *
 * All symbols live in namespace curaii::cusolverdn.
 */
#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace curaii::cusolverdn {

/**
 * @class Handle
 * @brief RAII wrapper for cuSOLVER DN handles.
 *
 * Calls cusolverDnCreate() on construction and
 * cusolverDnDestroy() on destruction. Movable but not copyable.
 */
class Handle {
public:
  /**
   * @brief Construct a new Handle, creating a cuSOLVER DN context.
   * @throws curaii::cusolver::Error if cusolverDnCreate fails.
   */
  Handle();

  /**
   * @brief Take ownership of an existing handle.
   * @param raw A valid cusolverDnHandle_t.
   * @note handle must have been created by cusolverDnCreate().
   */
  explicit Handle(cusolverDnHandle_t raw) noexcept;

  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;

  /**
   * @brief Move‑construct, stealing the handle from @p other.
   * @param other The source Handle.
   */
  Handle(Handle &&other) noexcept;

  /**
   * @brief Move‑assign, destroying any existing handle and stealing from @p
   * other.
   * @param other The source Handle.
   * @return Reference to this.
   */
  Handle &operator=(Handle &&other) noexcept;

  /**
   * @brief Destroy the cuSOLVER DN handle if one is held.
   *
   * Calls cusolverDnDestroy() (non‑throwing variant) when @c handle_ !=
   * nullptr.
   */
  ~Handle() noexcept;

  /**
   * @brief Get the underlying raw cusolverDnHandle_t.
   * @return The native handle.
   */
  cusolverDnHandle_t get() const noexcept;

  /**
   * @brief Release ownership of the handle without destroying it.
   * @return The raw handle; this object becomes empty.
   */
  cusolverDnHandle_t release() noexcept;

  /**
   * @brief Replace the managed handle.
   *
   * Destroys the old handle if valid, then takes ownership of @p raw.
   * @param raw New handle (or nullptr to clear).
   */
  void reset(cusolverDnHandle_t raw = nullptr) noexcept;

  /**
   * @brief Check whether a valid handle is held.
   * @return true if @c handle_ != nullptr.
   */
  explicit operator bool() const noexcept;

private:
  cusolverDnHandle_t handle_{nullptr};
};

/**
 * @class Params
 * @brief RAII wrapper for cuSOLVER DN parameter objects.
 *
 * Calls cusolverDnCreateParams() on construction and
 * cusolverDnDestroyParams() on destruction. Movable but not copyable.
 */
class Params {
public:
  /**
   * @brief Construct a new Params object.
   * @throws curaii::cusolver::Error if cusolverDnCreateParams fails.
   */
  Params();

  /**
   * @brief Take ownership of an existing params object.
   * @param raw A valid cusolverDnParams_t.
   * @note Must have been created via cusolverDnCreateParams().
   */
  explicit Params(cusolverDnParams_t raw) noexcept;

  Params(const Params &) = delete;
  Params &operator=(const Params &) = delete;

  /**
   * @brief Move‑construct, stealing the params from @p other.
   * @param other The source Params.
   */
  Params(Params &&other) noexcept;

  /**
   * @brief Move‑assign, destroying any existing params and stealing from @p
   * other.
   * @param other The source Params.
   * @return Reference to this.
   */
  Params &operator=(Params &&other) noexcept;

  /**
   * @brief Destroy the cuSOLVER DN params if held.
   *
   * Calls cusolverDnDestroyParams() (non‑throwing variant) when @c params_ !=
   * nullptr.
   */
  ~Params() noexcept;

  /**
   * @brief Get the underlying raw cusolverDnParams_t.
   * @return The native params object.
   */
  cusolverDnParams_t get() const noexcept;

  /**
   * @brief Release ownership of the params without destroying it.
   * @return The raw params; this object becomes empty.
   */
  cusolverDnParams_t release() noexcept;

  /**
   * @brief Replace the managed params object.
   *
   * Destroys the old params if valid, then takes ownership of @p raw.
   * @param raw New params object (or nullptr to clear).
   */
  void reset(cusolverDnParams_t raw = nullptr) noexcept;

  /**
   * @brief Check whether a valid params object is held.
   * @return true if @c params_ != nullptr.
   */
  explicit operator bool() const noexcept;

private:
  cusolverDnParams_t params_{nullptr};
};

} // namespace curaii::cusolverdn
