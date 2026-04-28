// Copyright 2026 Digital Holography Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "graph_builder_tracer.hh"

#include "holonp/abs.hh"
#include "holonp/add.hh"
#include "holonp/arange.hh"
#include "holonp/asarray.hh"
#include "holonp/ascontiguousarray.hh"
#include "holonp/concatenate.hh"
#include "holonp/conj.hh"
#include "holonp/copy.hh"
#include "holonp/divide.hh"
#include "holonp/empty.hh"
#include "holonp/equal.hh"
#include "holonp/fft.hh"
#include "holonp/fft2.hh"
#include "holonp/fftshift.hh"
#include "holonp/irfft2.hh"
#include "holonp/max.hh"
#include "holonp/mean.hh"
#include "holonp/meshgrid.hh"
#include "holonp/min.hh"
#include "holonp/multiply.hh"
#include "holonp/reshape.hh"
#include "holonp/rfft.hh"
#include "holonp/rfft2.hh"
#include "holonp/slice.hh"
#include "holonp/subtract.hh"
#include "holonp/transpose.hh"
#include "holonp/where.hh"
#include "holonp/zeros.hh"
#include "holotask/asyncs/batch_queue.hh"
#include "holotask/asyncs/slide_avg.hh"
#include "holotask/sinks/holofile.hh"
#include "holotask/sources/ametek_s710_euresys_coaxlink_octo.hh"
#include "holotask/sources/ametek_s711_euresys_coaxlink_qsfp+.hh"
#include "holotask/sources/fresnel_qin.hh"
#include "holotask/sources/fresnel_qout.hh"
#include "holotask/sources/holofile.hh"
#include "holotask/syncs/angular_spectrum.hh"
#include "holotask/syncs/conversion.hh"
#include "holotask/syncs/convolution.hh"
#include "holotask/syncs/correct_phase.hh"
#include "holotask/syncs/cross_correlation2.hh"
#include "holotask/syncs/cuda_stream_synchronize.hh"
#include "holotask/syncs/filter2d.hh"
#include "holotask/syncs/flatfield.hh"
#include "holotask/syncs/fresnel_diffraction.hh"
#include "holotask/syncs/mean_abs.hh"
#include "holotask/syncs/memcpy.hh"
#include "holotask/syncs/normalize.hh"
#include "holotask/syncs/pca.hh"
#include "holotask/syncs/pct_clip.hh"
#include "holotask/syncs/registration.hh"
#include "holotask/syncs/short_time_fresnel_diffraction.hh"
#include "holotask/syncs/unfold2d.hh"
#include "holotask/syncs/wrap2pi.hh"
#include "holotask/syncs/zernike.hh"
#include "holotask/syncs/zernike_phase.hh"
#include "tasks/sinks/display_tensor.hh"
#include "tasks/sinks/display_zernike_coefficients.hh"

namespace holovibes::pipeline {

// GraphBuilderTasks extends the tracer with strongly-typed wrappers for every
// task kind available in the registry. Each method registers one node in the
// graph and returns a traced output descriptor (or void for sink nodes).
class GraphBuilderTasks : public GraphBuilderTracer {
protected:
  using GraphBuilderTracer::GraphBuilderTracer;

  // clang-format off
  TDesc holofile_read(holotask::sources::HolofileSettings s);
  TDesc empty(holonp::EmptySettings s);
  TDesc zeros(holonp::ZerosSettings s);
  TDesc asarray(holonp::AsArraySettings s);
  TDesc arange(holonp::ArangeSettings s);
  TDesc ascontiguousarray(const TDesc &X, holonp::AsContiguousArraySettings s);
  TDesc copy(const TDesc &X, holonp::CopySettings s);
  TDesc memcpy(const TDesc &X, holotask::syncs::MemcpySettings s);
  TDesc batched_queue(const TDesc &X, holotask::asyncs::BatchQueueSettings s);
  TDesc convert(const TDesc &X, holotask::syncs::ConversionSettings s);
  TDesc pca(const TDesc &X, holotask::syncs::PcaSettings s);
  TDesc flatfield(const TDesc &X, holotask::syncs::FlatfieldSettings s);
  TDesc filter_2d(const TDesc &X, holotask::syncs::Filter2DSettings s);
  TDesc fresnel_diffraction(const TDesc &X, holotask::syncs::FresnelDiffractionSettings s);
  TDesc fresnel_qin(const TDesc &Z, holotask::sources::FresnelQinSettings s);
  TDesc fresnel_qout(const TDesc &Z, holotask::sources::FresnelQoutSettings s);
  TDesc short_time_fresnel_diffraction(const TDesc &X, holotask::syncs::ShortTimeFresnelDiffractionSettings s);
  TDesc unfold2d(const TDesc &X, holotask::syncs::Unfold2DSettings s);
  TDesc angular_spectrum(const TDesc &X, holotask::syncs::AngularSpectrumSettings s);
  TDesc cuda_stream_synchronize(const TDesc &X, holotask::syncs::CudaStreamSynchronizeSettings s);
  TDesc slide_avg(const TDesc &X, holotask::asyncs::SlidingAverageSettings s);
  void  xy_raw_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  void  xy_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  void  xz_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  void  yz_processed_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  void  shack_hartmann_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  void  shack_hartmann_xcorr_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  void  zernike_phase_display(const TDesc &X, tasks::sinks::DisplayTensorSettings s);
  void  zernike_coefficients_display(const TDesc &X, tasks::sinks::DisplayZernikeCoefficientsSettings s);
  void  holofile_write(const TDesc &X, holotask::sinks::HolofileSettings s);
  TDesc ametek_s710_euresys_coaxlink_octo(holotask::sources::AmetekS710EuresysCoaxlinkOctoSettings s);
  TDesc ametek_s711_euresys_coaxlink_qsfp_plus(holotask::sources::AmetekS711EuresysCoaxlinkQSFPSettings s);
  TDesc convolution(const TDesc &X, holotask::syncs::ConvolutionSettings s);
  TDesc correct_phase(const TDesc &X, const TDesc &PhaseMask, holotask::syncs::CorrectPhaseSettings s);
  TDesc pct_clip(const TDesc &X, holotask::syncs::PctClipSettings s);
  TDesc registration(const TDesc &X, holotask::syncs::RegistrationSettings s);
  TDesc wrap2pi(const TDesc &X, holotask::syncs::Wrap2PiSettings s);
  TDesc zernike(const TDesc &X, holotask::syncs::ZernikeSettings s);
  TDesc zernike_phase(const TDesc &X, holotask::syncs::ZernikePhaseSettings s);
  TDesc concatenate(std::span<const TDesc> Xs, holonp::ConcatenateSettings s);
  TDesc transpose(const TDesc &X, holonp::TransposeSettings s);
  TDesc add(const TDesc &A, const TDesc &B, holonp::AddSettings s);
  TDesc divide(const TDesc &A, const TDesc &B, holonp::DivideSettings s);
  TDesc multiply(const TDesc &A, const TDesc &B, holonp::MultiplySettings s);
  TDesc subtract(const TDesc &A, const TDesc &B, holonp::SubtractSettings s);
  TDesc equal(const TDesc &A, const TDesc &B, holonp::EqualSettings s);
  TDesc where(const TDesc &Cond, const TDesc &X, const TDesc &Y, holonp::WhereSettings s);
  TDesc rfft(const TDesc &X, holonp::RFFTSettings s);
  TDesc rfft2(const TDesc &X, holonp::RFFT2Settings s);
  TDesc irfft2(const TDesc &X, holonp::IRFFT2Settings s);
  TDesc cross_correlation2(const TDesc &Moving, const TDesc &Reference, holotask::syncs::CrossCorrelation2Settings s);
  TDesc slice(const TDesc &X, holonp::SliceSettings s);
  TDesc fft(const TDesc &X, holonp::FFTSettings s);
  TDesc fft2(const TDesc &X, holonp::FFT2Settings s);
  TDesc fftshift(const TDesc &X, holonp::FFTShiftSettings s);
  TDesc abs(const TDesc &X, holonp::AbsSettings s);
  TDesc mean(const TDesc &X, holonp::MeanSettings s);
  TDesc mean_abs(const TDesc &X, holotask::syncs::MeanAbsSettings s);
  TDesc min(const TDesc &X, holonp::MinSettings s);
  TDesc max(const TDesc &X, holonp::MaxSettings s);
  TDesc normalize(const TDesc &X, holotask::syncs::NormalizeSettings s);
  TDesc reshape(const TDesc &X, holonp::ReshapeSettings s);
  TDesc conj(const TDesc &X, holonp::ConjSettings s);
  // clang-format on
};

} // namespace holovibes::pipeline
