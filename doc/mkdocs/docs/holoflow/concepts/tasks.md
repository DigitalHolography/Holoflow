## Tasks

In Holoflow, a **task** is a unit of computation that transforms data. A task can be modeled as a function between tensors:

$$
f : \mathcal{T} \cup \{\varnothing\} \longrightarrow \mathcal{T}^*
$$

where:

* $\mathcal{T}$ is the set of all valid tensors.
* $\varnothing$ represents the absence of input.
* $\mathcal{T}^*$ denotes a finite sequence of output tensors (possibly empty).

Thus, a task consumes zero or one tensor and produces zero, one, or several tensors.

### Fixed Signature

Once a task is instantiated, its **signature is fixed**:

* **Input arity**: either zero or one, but always the same for that task instance.
* **Output arity**: the number of outputs is determined at instantiation and cannot vary at runtime.
* **Tensor descriptors** (shape, dtype, device, and other metadata) are also fixed. 
* **Ordering**: Each output position corresponds to a well-defined tensor descriptor.

Formally, each task instance defines a function

$$
f : D \longrightarrow C
$$

with

* $D \subseteq {\varnothing} \cup \mathcal{T}$ describing the admissible input tensor type/shape,
* $C \subseteq \mathcal{T}^k$ for a fixed $k \geq 0$ describing the tuple of outputs.

Tasks are splitted into two categories: **Synchronous** and **Asynchronous**. Synchronous tasks block the caller until completion, while asynchronous tasks return immediately and complete later. Both types can have **in-place** semantics, meaning they can modify their input tensor directly to produce outputs, and **owning** semantics, where they take responsibility for the input or output tensors allocations.

Tasks are instantiated via **task factories**. A task factory is a callable that exposes an inference method to determine the signature of the task it will create based on provided settings and its inputs. It also exposes a method to create a new instance of the task and
a method to update/recreate an instance by reusing resources when possible.

!!! note
    Although tasks can only have a single input, the API is based on multiple inputs. This design choice simplifies the implementation of tasks with zero inputs (by providing an empty input list) and allows for future extensions where tasks might accept multiple inputs.

## Synchronous Tasks
```cpp
struct SyncCtx {
  std::span<TView>   inputs;
  std::span<TView>   outputs;
  std::atomic<bool> *cancelled;
};

class ISyncTask : public ITask {
public:
  virtual ~ISyncTask() = default;
  [[nodiscard]] virtual OpResult execute(SyncCtx &ctx) = 0;
};
```

Creating a synchronous task is as simple as implementing the `ISyncTask` interface and defining the `execute` method. The method receives a context containing the input and output tensors as spans. The task can read from the input tensors and write to the output tensors. The method returns an `OpResult` indicating success or failure.

Long running tasks should periodically check the `cancelled` flag to see if they should abort early. If the task is cancelled, it should return `OpResult::Cancelled`.

### Example
The following example shows a simple synchronous task that computes the square root of each element in the input tensor and writes the result to the output tensor.

```cpp
class SqrtTask final : public ISyncTask {
public:
  [[nodiscard]] OpResult execute(SyncCtx &ctx) override {
    const TView in  = ctx.inputs[0];
    TView       out = ctx.outputs[0];

    const size_t n = in.size();
    auto *in_ptr   = in.data<float>();
    auto *out_ptr  = out.data<float>();

    for (size_t i = 0; i < n; ++i) {
      out_ptr[i] = std::sqrt(in_ptr[i]);
    }
    
    return OpResult::Ok;
  }
};
```
!!! note
    Inputs and outputs validation (arity, descriptors) is performed in the factory called by the holoflow compiler before execution. Tasks can assume that the inputs and outputs are valid by contract.

## Synchronous Factories
```cpp
struct InferResult {
  std::vector<TDesc>   input_descs;   
  std::vector<TDesc>   output_descs;  
  std::vector<InPlace> in_place;      
  std::vector<bool>    owned_inputs;  
  std::vector<bool>    owned_outputs; 
  TaskKind             kind;          
};

class ITaskFactory {
public:
  virtual ~ITaskFactory() = default;

  [[nodiscard]]
  virtual InferResult infer(std::span<const TDesc> input_descs,
                            const nlohmann::json  &jsettings) const = 0;
};
```