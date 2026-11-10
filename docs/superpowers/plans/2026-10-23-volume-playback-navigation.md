# Continuous .holo playback and linked slice navigation

User explicitly requests continuous 512-frame processing into XY/XZ/YZ video, linked visual slice indicators in all views, focus + Space unlock/move + Space relock, an in-view keybind hint, clean code and clang-format. Implement in holoflow_romain.

## Decisions
- Reuse the finite-window STFT graph and compiler's supported previous-output reuse. Keep its plans and allocations between windows; update source range through normal compilation. Add TemporalFFT update/rebind support and reuse retained volume storage. No manual mutations of compiler resources.
- Worker-thread single-shot QTimer schedules one window at a time, advances by window size (512 default), loops to selected start after last complete window, caps display rate at20fps by default and yields to parameter/pause events. Add Processing playback toggle/FPS and read-only current-window status. Reconstruction settings key excludes playback controls. View updates preserve playback position and its timer; pause/resume keep cached volume and position.
- Qt child overlay paints translucent red slice bands and keybind hint without disturbing OpenGL state. One image-rectangle calculation drives viewport, overlay geometry and mouse mapping (including letterboxing, panel fit, vertical flip and high DPI).
- Each focused view owns Space toggle; focus loss locks it. Mouse following works without dragging only in unlocked focused image area. Ignore key auto-repeat and consume Space shortcut override locally. Normal playback widgets do not intercept Space.
- XY changes X/Y and shows two bands. Per the user correction, each cut shows one band and changes only Z/P: vertical in XZ, horizontal in temporal YZ, vertical in other YZ layouts. Update paired controls once, keep other settings, coalesce worker updates. Inclusive averaging widths shown as bands. Bind overlays to same settings as sliders. Browse cached volume while paused without resuming video.

## Steps
1. Playback settings/UI, reusable FocusSession + TemporalFFT update + retained volume reuse.
2. Worker timer and lifecycle/status integration; view-only updates preserve cursor, cached-volume navigation also when paused.
3. Generic slice overlay/widget input API, central image mapping, linked MainWindow controls and guides.
4. Documentation, native code review, clang-format, Release build and diff check. No new dependencies, commits, test additions or test runs (developer explicitly prohibits tests unless requested). No GUI automation requested.

## Review focus
Source EOF/reset, plan reuse/error recovery, timer thread/lifetime and pause responsiveness, paired-axis updates, temporal YZ transpose, letterbox/flip/DPI mapping, Space focus/auto-repeat, stale overlay/image geometry and normal-mode shortcuts.

## Completion evidence
- Implemented worker-thread looping playback, read-only frame-range status, configurable FPS limit, compiler/FFT/storage reuse and paused cached-volume navigation.
- Added transparent red band overlays, linked coordinate controls, local Space shortcut handling, auto-repeat suppression and focus-loss locking, with in-view hints.
- Independent read-only code review found a missing timer/active-state cleanup on resume failure; fixed and re-reviewed with no remaining blockers.
- `cmake --preset msvc-multi`: succeeded (existing Qt deployment policy warning).
- `cmake --build --preset build-Release` through the VS developer-environment helper: succeeded, including rebuild after the review fix.
- `clang-format --dry-run --Werror` on all changed/new C++/CUDA sources and `git diff --check`: succeeded.
- No tests added or run; native GUI interaction has not been exercised during this task.
