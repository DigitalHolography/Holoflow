// Copyright 2026 Digital Holography Foundation
// SPDX-License-Identifier: Apache-2.0

#include "syncs/mtd_background.hh"
#include <cmath>
#include <stdexcept>

namespace holotask::syncs::mtd_detail {

// -------------------------------------------------------------------------------------------------
// Polynomial least-squares projection
// -------------------------------------------------------------------------------------------------

BackgroundProjection background_projection(std::span<const uint8_t> mask, int side,
                                           const RytovCorrectionSettings &s) {
  BackgroundProjection result;
  for (int y = 0; y < side; y += s.sample_step)
    for (int x = 0; x < side; x += s.sample_step)
      if (x >= s.background_margin && y >= s.background_margin && x < side - s.background_margin &&
          y < side - s.background_margin &&
          mask[static_cast<size_t>(y) * side + x] > s.mask_threshold)
        result.indexes.push_back(y * side + x);
  const size_t terms =
      static_cast<size_t>((s.polynomial_degree + 1) * (s.polynomial_degree + 2) / 2);
  const size_t samples = result.indexes.size();
  if (samples < terms)
    throw std::invalid_argument("MTD: insufficient background-mask samples for polynomial fit");

  std::vector<double> q(terms * samples), r(terms * terms, 0.0);
  size_t              term = 0;
  for (int px = 0; px <= s.polynomial_degree; ++px)
    for (int py = 0; py <= s.polynomial_degree - px; ++py, ++term)
      for (size_t i = 0; i < samples; ++i) {
        const double x        = 2.0 * (result.indexes[i] % side) / (side - 1) - 1.0;
        const double y        = 2.0 * (result.indexes[i] / side) / (side - 1) - 1.0;
        q[term * samples + i] = std::pow(x, px) * std::pow(y, py);
      }

  // Reorthogonalized modified Gram-Schmidt avoids the squared condition number
  // of normal equations. The normalized basis stays independent of pixel units.
  for (size_t column = 0; column < terms; ++column) {
    for (int pass = 0; pass < 2; ++pass)
      for (size_t previous = 0; previous < column; ++previous) {
        double dot = 0;
        for (size_t i = 0; i < samples; ++i)
          dot += q[previous * samples + i] * q[column * samples + i];
        r[previous * terms + column] += dot;
        for (size_t i = 0; i < samples; ++i)
          q[column * samples + i] -= dot * q[previous * samples + i];
      }
    double norm = 0;
    for (size_t i = 0; i < samples; ++i)
      norm += q[column * samples + i] * q[column * samples + i];
    norm = std::sqrt(norm);
    if (!std::isfinite(norm) || norm < 1e-10 * std::sqrt(static_cast<double>(samples)))
      throw std::invalid_argument("MTD: background mask produces a rank-deficient polynomial fit");
    r[column * terms + column] = norm;
    for (size_t i = 0; i < samples; ++i)
      q[column * samples + i] /= norm;
  }

  result.weights.resize(terms * samples);
  for (size_t sample = 0; sample < samples; ++sample)
    for (size_t row = terms; row-- > 0;) {
      double value = q[row * samples + sample];
      for (size_t col = row + 1; col < terms; ++col)
        value -= r[row * terms + col] * result.weights[col * samples + sample];
      result.weights[row * samples + sample] = value / r[row * terms + row];
    }
  return result;
}

} // namespace holotask::syncs::mtd_detail
