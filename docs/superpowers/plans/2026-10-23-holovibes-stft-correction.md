# Holovibes reference correction

User reports our .holo focus stack does not match Holovibes and supplied active settings screenshot. Root cause: implemented single-frame propagation versus reference batch512 complex temporal STFT after AngularSpectrum at2mm, spatialfilter20..362. Screenshot boundary86.5mm and sourceformula imply12um sensor pitch; stale footer had20um,235mm,PCA. Reference values override stale metadata only in the explicit STFT preset.

## Implementation
- Keep propagation as selectable method, default .holo to Holovibes STFT. Finite512framewindow, cachedGPUF32magnitudevolume.
- Reuse existing Memcpy/Conversion/AngularSpectrum; add normalized spatialIFFT opt-in (olddefaultunchanged) and genericleadingaxisTemporalFFT cuFFTnode.
- Extend VolumeSlice with backward-compatible count1 and transposefalse. Inclusive reference widths20mean21samples. XYtemporalbandmean224..244, X/Ywidthmeans. YZ temporalbinhorizontal.
- Add UI window/focus/pitch/filter settings and temporalbinlabels. Add per-view auto contrast and fitpanel. Preserve MTD, migrate prior focus settings.
- Diagnose raw data independently: NumPy first512frames reproduce curved/symmetricbands, preview build/reference_stft_diagnostic.png. This is data analysis, not execution/verification of the C++ implementation.
- Run clang-format and Release build. No tests or GUI automation requested; do not add/run tests. Request nativecode review per skill. No commits or unrelated changes.

## Review outcome
Independent reviewer found odd spatial dimensions incompatible with existing equal-quadrant AngularSpectrum shift; STFT now rejects them explicitly. Default512 dataset unaffected. Usage documentation was rewritten for STFT/propagation modes, reference source, width conventions and calibration differences.
