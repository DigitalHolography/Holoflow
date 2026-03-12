# Inspection And Debugging

Holoflow has two useful graph visualization layers:

- source graph visualization from `holoflow::core::to_dot(const GraphSpec&)`
- compiled graph visualization from `holoflow::runtime::to_dot(const CompilerOutput&, Registry&)`

Use the first to debug what you asked for. Use the second to debug what the compiler decided to build.

## Source graph inspection

`GraphSpec` supports:

- JSON serialization with `to_json(...)`
- JSON deserialization with `from_json(...)`
- DOT export with `to_dot(...)`

### Example

```cpp
auto j   = holoflow::core::to_json(graph);
auto dot = holoflow::core::to_dot(graph);
```

The JSON format is:

```json
{
  "nodes": {
    "camera": {
      "type": "Camera",
      "params": { "fps": 100 }
    },
    "display": {
      "type": "Display",
      "params": {}
    }
  },
  "edges": [
    {
      "from": "camera",
      "to": "display",
      "out": 0,
      "in": 0
    }
  ]
}
```

## Compiled graph inspection

The compiled DOT export includes much more detail:

- node names and kinds
- input and output tensor IDs
- edge tensor descriptors
- storage alias information
- section clusters
- async producer/consumer split rendering

### Example

```cpp
auto compiled_dot = holoflow::runtime::to_dot(*compiled, registry);
```

This is usually the fastest way to inspect:

- tensor propagation
- in-place storage aliasing
- section partitioning
- stream assignment intent

## Compiler tracing

With `Compiler::Config` you can enable useful artifacts:

- `log_dir`: writes `compiler.log`
- `dump_dot_on_failure`: dumps DOT files on success/failure
- `enable_profiling`: emits trace events
- `trace_filename`: names the Chrome trace JSON file

A practical setup is:

```cpp
holoflow::runtime::Compiler::Config config;
config.log_dir = "logs";
config.enable_profiling = true;

holoflow::runtime::Compiler compiler(registry, config);
```

## Debugging compile failures

When compilation fails, debug in this order:

1. check that the registry contains every node kind
2. inspect duplicate node names
3. inspect whether multiple edges target the same input slot
4. inspect task `infer(...)` implementations for rejected tensor contracts
5. render the source and compiled DOT graphs
6. check whether ownership or in-place declarations conflict

## Debugging runtime failures

At runtime, focus on:

- task return codes
- incorrect ownership declarations
- incorrect CUDA stream assumptions
- lifetime of `CompilerOutput`
- polling async tasks that never transition out of `NotReady`

## Pseudo debugging workflow

```text
if compile fails:
  render GraphSpec to DOT
  inspect compiler log
  inspect failing node's infer() assumptions

if runtime behaves incorrectly:
  inspect compiled DOT
  verify tensor IDs and storage aliasing
  verify section partitioning
  verify task ownership hooks
  inspect scheduler metrics and emitted events
```

## What to inspect in a compiled DOT graph

Use this checklist:

- Does each node have the expected `in_tids` and `out_tids`?
- Do in-place tasks show outputs sharing the expected storage?
- Are async nodes attached to the intended producer and consumer sections?
- Do section boundaries align with where buffering or handoff is expected?
- Does a changed graph still reuse the nodes you expected to update?
