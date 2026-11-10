# Holo focus explorer

Implement the user-approved path in holoflow_romain: .holo reader -> propagation stack -> existing XY/XZ/YZ views. OCT is the requested UI umbrella; distinguish MTD diffraction tomography from holographic refocusing.

## Design and decisions
- Keep existing MTD reconstruction and ordinary playback. Add an OCT toggle and MTD/.holo selector in IMPORT.
- Read the actual binary dimensions. Inspect both current and legacy nested footers. Import wavelength, separate X/Y pixel pitches and saved propagation distance as editable estimates. State metadata conflicts; never infer retinal depth or repair pixel pitch from conflicting dimensions.
- One user-selected time frame generates an F32 [Z,Y,X] normalized intensity stack with configurable wavelength, pitches, Z endpoints/count, optional 2x padding and optional off-axis Fourier order isolation. Sensor propagation coordinates, not object-space dimensions.
- GPU implementation computes one forward FFT, reuses a 2D inverse FFT workspace and transfer calculation for each Z plane. Retain only output volume for navigation. Guard allocations and settings.
- Reuse a common GPU volume/slice cache for MTD and focus stacks. Preserve finite reconstruction and pause/resume behavior; slider/contrast updates do not rebuild the volume. Support rectangular volumes and independent spacing.
- PROCESSING holds optics/frame/stack fields and calibration summary. VIEW holds independent X/Y/Z sliders, physical Z label and fixed normalized intensity contrast. Persist both modes independently.

## Implementation steps
1. Add finite .holo reader option, calibration inspection, FocusStack task and reusable pipeline.
2. Extract common retained-volume navigation; add focus reconstruction session and manager routing.
3. Add OCT source controls, focus parameters, rectangular view controls, persistence, validation and mode lifecycle integration.
4. Document calibration evidence and usage. Run clang-format, configure and Release build; inspect diff.

## Verification and rulings
- Developer instruction prohibits adding/running tests without a user request. Use formatting, source review and the required build; no test suite or GUI interaction was requested.
- Native execution in the explicitly requested existing checkout. No commits or branch switching requested.
- Calibration evidence: sample binary header 512x512, 13,312 U8 frames; legacy footer 1024x512/8,192 frames. Saved wavelength 852nm, propagation 235mm, pixel pitch X/Y=20 (legacy micrometre convention). These are unverified, editable estimates.
- Review focus: rectangular axis sizes, finite source completion, wavelength/pitch units, stale volume on errors, signal recursion/restoration, allocations and fixed shared contrast.

## Completion record
- Implemented finite Holofile repeat=false, reusable focus graph and CUDA FocusStack task with normalization, rectangular dimensions, optional padding/order isolation and bounded output size.
- Extracted VolumeSession for GPU-only retained volume/slice display; existing MTD session and new FocusSession share it. Manager routes both finite volume modes and preserves cached navigation.
- Added OCT toggle/source selection, optics/stack controls, separate persistent settings, physical propagation labels and focus-specific validation/help/schema.
- Native review found inactive focus-view restoration bug; fixed independent restoration, saved last-valid source identity/dimensions and preserved settings while a source is temporarily unavailable. Reviewer inspected corrected code and reported no further actionable findings.
- No commits, tests or GUI interaction performed. Build and format evidence are reported in the task response.
