# OCT volume modes: Holovibes STFT and propagation stacks

## Open a recording
1. Enable **OCT mode** in IMPORT. The first and default source is **.holo (Holovibes / focus stack)**; select a raw little-endian U8/U16 recording.
2. In **OCT - HOLO VOLUME**, choose the reconstruction method. **Holovibes STFT volume** is the default and matches the supplied Holovibes reference settings. **Single-frame propagation stack** retains the original numerical refocusing mode.
3. Press **Start**. STFT continuously processes successive complete windows (512 frames by default), updating XY/XZ/YZ together. **Continuous playback (loop)** and **Playback FPS limit** are in the processing panel; the default limit is 20 output windows per second, subject to processing speed. Pause freezes playback; resume continues from the retained window. Disable continuous playback for a static window. Propagation mode remains a single-frame volume.
4. MTD remains available under **OCT mode -> MTD dataset (tomography)**. Ordinary playback remains available with OCT mode disabled. Stop before changing source.

The Auto Focus panel is hidden in OCT mode: its Shack-Hartmann/Zernike processing is not part of either OCT reconstruction graph. It returns when OCT is disabled. Optical parameters remain editable; file-loading errors appear below the controls.

## Holovibes STFT reference
The supplied screenshot uses Angular Spectrum, STFT with 512 frames/window/stride, wavelength 852 nm, spatial focus 2 mm, spatial filter radii 20 and 362 pixels, magnitude display, no FFT shift, one output accumulation, XY band start 224 with width 20, X/Y start 0 with widths 20, auto contrast and image-level renormalization.

These are **active settings supplied by the user**, not the stale settings in the file footer. The new preset uses the shown settings. Window start defaults to frame 0 (0-based). Playback advances by the window length, then loops to that configured start after the last complete window; an incomplete tail is skipped. The processing panel reports the currently displayed raw-frame range. Changing window start or reconstruction parameters rebuilds from that start; view controls preserve playback position. Temporal window length is configurable; one entire window is transformed with a rectangular, unnormalized complex FFT along time, as in Holovibes. Spatial propagation is applied to each frame, with inverse-FFT normalization and no spatial padding in this mode.

The reference's **Boundary = 86.5 mm** is a derived value, not an additional propagation distance. Holovibes computes it as `N * pixel_pitch^2 / wavelength`. At N=512 and wavelength=852 nm, a 12 um pixel pitch yields 86.5352 mm. The reference preset therefore uses **12 um**, explicitly distinct from the footer's saved 20 um estimate. This matches the reference software configuration; it does not independently establish camera calibration, acquisition geometry or physical depth.

### View controls
- **P** changes the start of the XY temporal band. XY averages magnitudes across the band.
- **X** changes the YZ/YP cut; **Y** changes the XZ/XP cut.
- X width, Y width and XY band width use Holovibes' inclusive extent: width 0 samples one plane; width 20 samples up to 21 planes. Ranges clip safely at the volume boundary.
- XZ/XP shows X horizontally and the temporal bin vertically. YZ/YP shows the temporal bin horizontally and Y vertically, matching Holovibes.
- Auto contrast uses each view's 0.5th/99.5th percentiles; the first two temporal bins are excluded when estimating cut contrast. Disable it to use shared manual limits.
- **Fit images to panels** fills the available dock area. Disable it to retain pixel aspect.

### Move slices directly in the views
Click a view to focus it, then press **Space** to unlock slice movement. Move the mouse over the image; press **Space** again to lock. Changing focus also locks the previous view. A small in-view hint shows the current state.

Red translucent bands show the selected planes and their inclusive averaging widths. Only the main XY view has two bands and moves X/Y. XZ/XP and YZ/YP each have a single band and move only the XY plane (Z/P), preserving X/Y. In Holovibes STFT mode, move vertically in XZ/XP or horizontally in YZ/YP; in propagation and MTD modes, move vertically in either cut. All indicators and parameter sliders remain linked. Mouse mapping follows the actual image rectangle when letterboxed or stretched. Cached slices remain browsable while playback is paused.

P is a temporal-frequency bin. No wavelength sweep/depth calibration was supplied, so these cuts are not labelled as anatomical depth or propagation distance. Spatial focus (2 mm) is a separate parameter. The pipeline does not infer depth from camera frame number.

## Single-frame propagation mode
Check wavelength, separate X/Y sensor pitches, frame index, propagation Z minimum/maximum and plane count. Frames are 0-based. X/Y/Z sliders browse YZ/XZ/XY. The physical Z label is sensor propagation distance, not anatomical eye depth. Optional 2x zero padding reduces wraparound; optional manual off-axis order isolation removes DC/conjugate contributions only if an isolated order has been correctly identified. Narrow filtering removes detail.

Output is propagated holographic intensity normalized by camera code range and FFT size, in relative display units. It is not a 3D refractive-index reconstruction or a coherence-gated OCT measurement. No object-space magnification is inferred. The geometry summary reports separate sampling distances.

## File metadata inspection
`I:/holofiles_2026/oct/20201023_155140_8b.holo` has a consistent binary header: 512 x 512 pixels, 13,312 frames, U8, version 5.

The legacy nested footer saves wavelength 852 nm, propagation about 235 mm, time method PCA/512 and unitless X/Y pitches of 20. The importer explicitly labels its legacy micrometre interpretation. The footer also reports 1024 x 512 pixels and 8,192 frames, conflicting with the binary header and with the active reference settings. The binary header controls reading; no pitch correction is inferred from this mismatch. Missing metadata fields retain existing editable parameter values. STFT reference parameters are independent from the propagation mode's metadata estimates.

## Implementation and limits
`build_focus_stack_pipeline` composes a finite Holofile source and the selected graph:
- STFT: device copy -> F32 conversion -> normalized AngularSpectrum/filter -> TemporalFFT -> magnitude F32 [P,Y,X].
- Propagation: one frame -> FocusStack -> intensity F32 [Z,Y,X]. One forward FFT and one reusable inverse-FFT workspace serve all propagation planes.

The worker schedules one STFT window at a time and yields between windows for pause and settings changes. Compiled graphs reuse device allocations, source readers, spatial/temporal FFT plans and retained volume storage across compatible windows. Only the three displayed planes are copied back to the GUI. The retained GPU display cache and VolumeSlice task support rectangular volumes, adjacent-plane means and transposed output. MTD keeps its existing default one-plane slices and its own optical reconstruction/contrast settings. Generic AngularSpectrum normalization is opt-in so existing consumers keep their prior amplitude scaling.

STFT requires even image width/height for the angular-spectrum FFT shift and rejects incomplete windows, invalid optics/filter ranges and complex intermediate volumes over 4 GiB. A 512x512x512 retained F32 volume takes 512 MiB; reconstruction temporarily uses additional raw/F32/complex FFT buffers and an output copy. Propagation defaults to 128 planes around saved propagation +/- 5 mm and limits F32 output to 4 GiB.

## Reference source and diagnosis
Inspected public Holovibes revision `de119d97661cc92883a18f23570f567432c17aca`:
- [Angular Spectrum and temporal transformation](https://github.com/DigitalHolography/Holovibes/blob/de119d97661cc92883a18f23570f567432c17aca/Backend/sources/compute/fourier_transform.cc)
- [STFT and XZ/YZ cut layout](https://github.com/DigitalHolography/Holovibes/blob/de119d97661cc92883a18f23570f567432c17aca/Backend/cuda_sources/stft.cu)
- [Boundary calculation](https://github.com/DigitalHolography/Holovibes/blob/de119d97661cc92883a18f23570f567432c17aca/Backend/sources/api/information_api.cc)
- [Inclusive XY magnitude averaging](https://github.com/DigitalHolography/Holovibes/blob/de119d97661cc92883a18f23570f567432c17aca/Backend/cuda_sources/tools_conversion.cu)

A separate NumPy diagnostic on the first 512 raw frames with the supplied settings reproduces the reference's XY structure and curved/symmetric bands in XZ/YP. Its preview is under build/reference_stft_diagnostic.png. This supports the diagnosis that the previous single-frame propagation path was the wrong computation for the reference. It does not certify native GUI output or pixel-exact equivalence to a live Holovibes window, whose frame offset is unknown.
