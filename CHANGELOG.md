# Changelog

All notable changes to Holoflow are documented in this file.

The `v0.1.0-dev.6` through `v0.1.0-dev.8` releases were produced from the former
`refactor` branch. The current `main` release line diverged after
`v0.1.0-dev.5`; `v0.2.0-dev.1` establishes the new common release baseline.

## v0.2.0-dev.3 (2026-07-28)

### Feat

- show available application updates

### Perf

- render Zernike curves off the UI thread
- expand Zernike UI and curve worker profiling

## v0.2.0-dev.2 (2026-07-27)

### Fix

- support larger PCA depths
- package PCA runtime compilation assets

## v0.2.0-dev.1 (2026-07-27)

### Feat

- add padded angular spectrum propagation
- show Zernike history statistics
- manage visualization toggles from view menu
- add graph-laplacian Shack-Hartmann autofocus
- extend Zernike metrics displays
- add configurable Zernike history and pipeline pause
- add stride support in concatenate
- add nvtx task execution ranges
- add fft frequency range bin tool
- log defocus z propagation estimate
- add flatfield toggle and fft acceleration
- wire Shack-Hartmann iteration count
- use physical flatfield cutoff period
- add flatfield correction task
- refine clinical workspace controls
- redesign clinical UI shell
- enable angular spectrum propagation
- support real filter2d input
- wire 2d filter UI
- add optional holofile input fps limiting
- **holotask**: temp commit for camera handling
- **holovibes**: add validation tooltips
- **holovibes**: add pipeline settings validator
- add Arange and Argmax tests with Python oracle integration
- add batch processing support for PCA and process time analysis with a batch to reduce latency
- add CudaStreamSynchronize holotask and use it before device-to-host memcpy in Shack-Hartmann
- short time fresnel support for real input
- add Short-Time Fresnel Diffraction (STFD) task
- add M0/M1/M2 spectral moment view modes
- rename SHORT_TIME_FOURIER to RFFT and add FFT time transform option
- add Holoflow Agent Guide with detailed setup, build, test workflows, and coding standards
- add Holoflow Agent Guide with project setup, build, and testing workflows
- add grid region support in Zernike settings and update related processing in GraphBuilder_v2
- enhance Zernike execution with improved slope calculations and phase ramp adjustments
- add validation for subaperture grid size and autofocus subapertures in Zernike and GraphBuilder_v2
- extend Zernike support to include modes 7 to 10 and update related UI components
- enhance Fresnel diffraction and output processing with phase shift options
- update grabber preprocessor directives to use EGRABBER
- **view_widget**: update set_pct_radius to accept double values for improved precision
- **holofile**: add keep_cursor option to HolofileSettings and update JSON serialization
- add copyright headers to multiple source files
- **holovibes**: enable debug graph dumps and improve error handling in CUDA operations
- **manager**: enhance pipeline management with improved logging and event polling
- **holoflow**: added more profiling to compiler and implemented tasks update
- **auto_focus**: implement Zernike coefficient display and management in AutoFocusWidget
- **graph_builder**: add conditional handling for Zernike orders in phase calculation
- **auto_focus**: enhance AutoFocusWidget to support Zernike coefficient selection and update settings structure
- **auto_focus**: refactor AutoFocusWidget to support fixed configurations and visualization toggles
- **graph_builder**: update Zernike coefficients to support Noll indexes 2 to 6 in graph builder
- **zernike**: update Zernike coefficients to support Noll indexes 2, 3, and 4 in graph builder
- Implement CorrectPhase functionality and integrate into pipeline
- **holovibes**: displaying phase correctly
- **zernike_phase**: add ZernikePhase task implementation and integrate into graph builder
- **zernike**: add Zernike phase display widget and integrate into pipeline manager
- **zernike**: add Zernike task implementation and integrate into graph builder
- **ascontiguousarray, copy**: add AsContiguousArray and Copy classes with JSON serialization and factory methods
- **normalize**: add Normalize class and related functionality for normalization operations
- **fft2, transpose**: refactor FFT2 and Transpose classes for improved offset handling and memory management
- **graph_builder_v2**: remove dead code computation
- **mul**: enhance Mul class to support runtime type promotion and mixed-type operations
- **mean_abs**: implement MeanAbs class and related functionality for absolute mean computation
- **fresnel_diffraction**: add is_real parameter to FresnelDiffraction class and update related methods for real input support
- **holofile**: add input acquisition and output release methods; update constructor to include output tensor description
- **graph_builder_v2**: add pct_clip processing step to enhance image clipping functionality
- **graph_builder**: enhance data reshaping and processing methods in GraphBuilder_v2
- **holotask**: added real valued pca support
- **holonp**: add Argmax task and factory implementation
- **holovibes**: scaling should work now
- **holovibes**: started to add scaling to sh
- **holovibes**: added sh xcorr
- still working on sh. How am i gonna handle loops?
- implement add, div, and zeros operations with corresponding factories and settings
- made progress towards shack hartmann, need to rework topological sorting such that inplace node with side effects (assignment for example) run first. Need to check integer lattice partition
- **holoflow**: add strides to TDesc and JSON serialization functions
- **holonp**: added empty task
- **holovibes**: made progress on shack hartmann
- **holovibes**: made progress on shack hartmann view
- added mul and fresnel qin tasks
- **holonp**: added bunch more numpy nodes
- **holonp**: added abs and mean tasks to holonp
- **holonp**: added fft related tasks
- **holonp**: added slice copy task
- **holovibes**: added base of shack hartmann branch and transpose node
- **holovibes**: added dummy quiver widget
- **holoflow**: compiler should work with dag
- **holovibes**: added meshgrid node and started to modify compiler
- **holonp**: added arange task

### Fix

- write pipeline logs to user data directory
- **docs**: use portable snippet paths
- derive Zernike history sample timing
- support CUDA 13 MSVC builds
- added gauge in zernike measurement model
- preserve tensor stream after redocking
- display cumulative shack hartmann correction
- reuse fresnel tasks on distance updates
- reuse holofile sink on graph updates
- stabilize clinical display layout
- log scheduler task crashes before aborting
- validate export paths in UI
- handle invalid recording paths gracefully
- fixed invalid load batch size deduction
- corrected invalid output desc in concatenate
- update in_place field in CudaStreamSynchronizeFactory::infer to ensure input is transmitted
- replace fseek with _fseeki64 for correct seeking in Reader::seek function
- **holoflow**: fixed compiler duplicating nodes executions plans
- fixed whatever the bug was

### Refactor

- move PCA eigendecomposition to cuSolverDx
- now using docking for display area
- switched to negative condition for zernike
- split styling in seperate .qss file
- **holonp**: align API with numpy naming and move non-numpy ops to holotask
- **holonp**: hide shared fft helper internals
- **holovibes**: clean display sink task implementations
- **holotask**: clean short-time fresnel diffraction task
- **holonp**: clean correlation and window tasks
- **holonp**: clean fft task implementations
- **holonp**: clean mean reduction tasks
- **holonp**: clean arithmetic and reduction tasks
- **holonp**: split asarray family tasks
- **holonp**: clean shape and view transformation tasks
- **holonp**: clean binary broadcast and assign tasks
- **holonp**: clean basic generator and unary tasks
- **holotask**: clean async batch and sliding average tasks
- **holotask**: clean fresnel qin and qout sources
- **holotask**: clean legacy sync task implementations
- add pixel format mapping to configure grabber for improved clarity
- enhance grabber configuration with offsets and streamline parameter calculations
- AmetekS711EuresysCoaxlinkQSFP should own the output buffer
- optimize memory copy with multithreading and improve UI metrics polling interval
- reworked recording so its fast and non blocking. Still dirty
- cleaned camera sources
- cleaned holofile source
- cleaned fresnel diffraction and angular spectrum tasks, and move task declaration out of header
- improved fresnel diffraction code
- promote GraphBuilder_v2 to GraphBuilder, split into task and tracer layers
- clean up code formatting and improve readability in reshape and zernike modules
- remove logging of apply_lens_callback source code in apply_lens_lto function
- remove unused variable assignment in FresnelDiffractionFactory and update default parameter in GraphBuilder_v2
- **holovibes**: made graph builder v2 cleaner
- **zernike**: enhance comments for clarity and improve subaperture center calculations
- **zernike**: improve shift recovery and coefficient calculation, enhance code clarity
- **graph_builder_v2**: optimize batch processing and replace slide_avg with mean function
- **holotask**: reworked fresnel diffraction for multi axis
- **holovibes**: Who's the vector now?
- **assign**: update input/output variable assignments for clarity and consistency
- **holoflow**: re-enable subap processing in GraphBuilder_v2
- **holoflow**: made progress on storage refactor
- **holoflow**: enhance task graphviz
- **holoflow**: updated compiler
- **holoflow**: temp commit
- **holoflow**: replace raw pointers with unique_ptr for Storage management
- **holoflow**: forced the use of a method for accessing TDesc data
- **holoflow**: updated TDesc initialization and stride handling in infer functions
- **holoflow**: added constructor to TDesc

### Perf

- fuse Fresnel magnitude output
- profile Zernike plot rendering
- hoist pct clip roi setup
- fuse fft callbacks
- compute zernike phase on gpu
- removed useless reshape
- improved fresnel diffraction and its short time variant moving struct field to compile time constants during runtime compilation
- **holoflow**: reduced compilation times
- **holonp**: improved min and max perf, and updated nvtx usage in scheduer

## v0.1.0-dev.5 (2026-01-08)

### Fix

- typo
- add lib cpack

## v0.1.0-dev.4 (2026-01-08)

### Fix

- installer

## v0.1.0-dev.3 (2026-01-08)

### Feat

- add desktop shortcut

### Fix

- **CMake**: add rt lib of intel
- **holovibes**: added record back
- **holovibes**: added memcpy after stft to fix spatial consistency

## v0.1.0-dev.2 (2026-01-07)

### Feat

- **holovibes**: add buttons for spectrum view
- **holovibes**: processed spectrum in opengl view
- **holovibes**: started to add spectrum branches
- **holotask**: implemented extract_ranges and very primitive test for it
- hook for clang-format and fix the addlicence

### Fix

- **holovibes**: create export controls function, and change widget window name according to the image mode
- **holovibes**: close all window in cascade
- **README**: add specification for the QT Graphs package
- holovibes version

### Refactor

- **holovibes**: refactored graph builder to deduce graph from computations

## v0.1.0-dev.1 (2025-12-09)

### Feat

- project's version by a file and add 2 workflow
- **ci**: add ci
- **holovibes**: added ui params for auto-focus
- **holovibes**: disabled x and y spins if 3D cut is not enabled
- **holovibes**: prevent windows from being manually closed
- **holovibes**: reworked the view widget
- **holoflow**: nodespec now has debug option for prettier print
- **holovibes**: added display reticle
- **holovibes**: made export group toggleable
- **holovibes**: added raw_view on top of raw image
- **holovibes**: added camera config file qbox and fixed raw view
- **holovibes**: added support for the s711 phantom camera and raw view and processed record
- **holovibes**: :sparkles: import settings finalized
- **holovibes**: :rocket: Finally corrected registration not having a CAPITAL letter at the start.
- **installer**: add installer for the application
- **holovibes**: implemented footer support
- **rotate**: rotate node and add the rotation of the window yz for the 3D cuts
- **crop**: add the node crop
- **holoflow**: add logger binding to tasks and implement task logger creation
- **holovibes**: propagate pipeline error messages to GUI with signals
- integrate Ametek S710 Euresys Coaxlink Octo source with configuration handling
- add FindEgrabber module and include EGrabber in project
- **holovibes**: implemented registration
- **holovibes**: implemented filter2D
- **holovibes**: implemented update for convolution
- **holovibes**: implement raw recording functionality with start/stop controls and event handling
- **event**: integrate event handling in Holofile and Scheduler for event-driven notifications
- **holoflow_event**: add event handling system with router and bounded queue. Not integrated yet
- **holofile**: add Holofile sink task for recording
- **examples**: enable example builds and add SqrtTask implementation
- **holovibes**: fft convolution now uses callbacks
- convolution now uses fft and implemented divide
- **holofile**: implement Writer class for writing Holofile frames
- **holofile**: add JSON schema for Holofile settings
- **holovibes**: implement fixed aspect ratio handling for tensor display widget
- **holovibes**: added convolution
- **holovibes**: add metrics tracking and UI updates for input throughput
- **holovibes**: added metrics panel
- **holoflow**: added compile time profiling to compiler
- **holovibes**: implemented reshape
- **holovibes**: added debounce queue
- **holovibes**: Implemented sliding average
- **holovibes**: added GUI
- **holovibes**: added base of pipeline manager
- **holovibes**: improved example
- **holovibes**: updated example graph
- **holovibes**: added fresnel diffraction task
- **holovibes**: added fft shift task
- **holovibes**: added average task
- **holovibes**: added angular spectrum propagation task
- **holovibes**: added conversion task
- **holovibes**: added stft task
- **holovibes**: added pct clip task
- **holovibes**: added memcpy task
- **curaii**: added cusolver, cufft and nvrtc wrappers
- **curaii**: added cublas wrappers
- **holovibes**: added BatchQueue task
- **holovibes**: added display task and widget + fixed shceduler
- **holovibes**: added holofile source
- **holofile**: initial commit for holofile
- **holoflow**: made progress on compiler. Should be testable now.
- **holoflow**: made progress on compiler
- **holoflow**: started to work on compiler
- **holoflow**: first implementation of scheduler, not tested
- **holoflow**: started o work on scheduler
- **holoflow**: started to work on holoflow runtime
- **holoflow**: added factories for tasks
- **holoflow**: added base of task api
- **holoflow**: added declaration and definition of tensor and related types
- **curaii**: initial commit for curaii. Added minimal cuda wrappers
- **holoflow**: added core dtypes and added license headers
- initial commit

### Fix

- add win64 to the name of the installer
- ci command typo
- build failed because of the merge (Hein Cyril)
- change the shell to pwsh
- change the shell to bash for the release
- versionning for the release
- installer and add documentation about packaging
- **ci**: change the directory of the test exe
- **holovibes**: fixed start record button being active when not supposed to
- **holovibes**: fixed record skipping frames
- **holovibes**: disabled start record button if export is not active
- **holovibes**: fixed updated camera config file not being replaced by start/stop
- **holovibes**: temporary fix for raw_record event using wrong name
- **holovibes**: re-enabled recording being optional
- **holovibes**: fixed wrong includes
- **holovibes**: 3d cuts
- **manager**: withdraw useless node for the raw view
- **installer**: add the necessary lib when packaging
- **holovibes**: raw view no longer deformed if fresnel diffraction is active
- **holovibes**: fixed function decleration not present in header
- **holovibes**: removed hard coded footer data
- **compiler**: prevent duplicate logger initialization error
- **batchqueue**: change int to size_t to avoid overflow
- **ui**: change file dialog from open to save for export functionality
- **compiler**: improve output index validation in check_typing method
- **holoflow**: fixed compiler updating a node instead of creating a new one if name is the same
- **holovibes**: fixed headers includes
- **holovibes**: correct NVRTC argument flags for compilation
- **holovibes**: update metrics display to show 'N/A' for uninitialized values
- **holovibes**: fixed sliding average wrong size checking and restricted it to one-elements usage
- **holovibes**: fixed invalid debounce queue output size
- **holoflow**: fixed scheduler running sync nodes after async stopped
- **holoflow**: fixed todot using record instead of box
- **holovibes**: fixed small bugs with manager.cc
- **holovibes**: fixed nvrtc in fresnel diffraction
- fixed Qt6_DIR var lookup
- **curaii**: fixed invalid macro name and invalid namspace
- **holoflow**: moved runtime srcs out of core

### Refactor

- **holotask**: moved tasks to new library holotask
- **holovibes**: name of the camera s711 and the documentation
- architecture of the json schema and md schema
- **holovibes**: changed raw_record to record
- **holovibes**: segmented main_window.cc into widgets
- **holovibes**: split manager by creating a graph builder
- **holovibes**: removed boundary and renormalize from gui
- removed redundant function
- **holovibes**: refactored tasks namespaces
- **holovibes**: splitted tasks in source sinks syncs and asyncs
- **graph_spec**: improve label escaping and node edge formatting
- **holovibes**: add setup methods for menu bar, layout, display widgets, and pipeline manager
- **holovibes**: adjust widget sizing
- **holovibes**: switched to using QOpenGl for display
