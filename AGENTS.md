# AGENTS.md

## Scope

This repository is a C++/CUDA/Qt codebase for digital holography.

This file defines how coding agents should operate in this repository. Human contributors may also use it as a quick operational guide, but it is written primarily for agents.

## General approach

Be conservative by default.

Prefer small, targeted, low-risk changes over broad rewrites. Preserve existing structure and naming unless there is a clear reason to change them. Do not perform large refactors unless explicitly requested.

When modifying code:

- understand the local context first
- make the smallest reasonable patch
- keep style consistent with nearby code
- avoid incidental cleanup unrelated to the task
- document non-obvious behavior when needed

## Build workflow

Use these commands as the default build workflow:

```bash
cmake --preset msvc-multi
cmake --build --preset build-Release
````

Notes:

* Testing is not set up yet
* Do not invent or claim test coverage that does not exist
* Do not add ad hoc test commands to the documented workflow unless explicitly asked

## What agents should do by default

Agents should generally:

* make minimal patches
* keep changes narrowly scoped to the request
* ensure the project still builds when the task requires code changes
* update relevant documentation or comments when behavior changes
* match existing repository conventions

## What agents should avoid

Unless explicitly requested, do not:

* add new dependencies
* edit CI configuration
* perform large refactors

If a requested change appears to require one of the above, stop and state that clearly.

## Comment banner conventions

Use these banner styles consistently.

Main banner style:

```cpp
// -------------------------------------------------------------------------------------------------
// JSON serialization
// -------------------------------------------------------------------------------------------------
```

Sub-banner style:

```cpp
// ---- StreamModule: per-grabber DMA engine -------------------------------------------------------
```

Guidance:

* use the main banner style for major sections
* use the sub-banner style for smaller logical subsections
* keep banner titles short and descriptive
* do not introduce alternative banner styles

## Change discipline

Prefer:

* isolated edits
* minimal interface disturbance
* consistency with surrounding code

Avoid:

* opportunistic renaming
* moving code without clear benefit
* mixing formatting-only edits with functional changes
* unrelated cleanup in the same patch

## Communication

When summarizing work:

* state what changed
* mention whether the default build commands should be run
* explicitly note that testing is not set up

Do not claim tests passed unless actual tests were run and exist.
