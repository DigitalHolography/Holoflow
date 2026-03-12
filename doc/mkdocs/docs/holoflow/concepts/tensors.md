# Tensors And Storage

Holoflow moves tensor *views*, not heavyweight tensor objects, through the runtime. The distinction is central to both performance and ownership semantics.

## `DType`

Supported scalar element types are:

- `U8`
- `U16`
- `F32`
- `CF32`

`holoflow::core::size_of(dtype)` returns the element size in bytes.

## `MemLoc`

A tensor lives either in:

- `MemLoc::Host`
- `MemLoc::Device`

This is part of the tensor contract. Factories are expected to validate and produce descriptors with the correct memory location instead of silently converting between host and device.

## `TDesc`

`TDesc` is the static descriptor of a tensor:

- `shape`
- `dtype`
- `mem_loc`
- `strides`
- `offset`

It answers three questions:

1. What is the logical array shape?
2. What is the scalar type?
3. Where and how is it laid out in memory?

### Default stride construction

When built from `(shape, dtype, mem_loc)`, `TDesc` creates contiguous row-major strides in bytes.

For example:

```cpp
holoflow::core::TDesc desc({480, 640}, holoflow::core::DType::F32,
                           holoflow::core::MemLoc::Device);
```

This is equivalent to a contiguous `480 x 640` float tensor on the GPU.

### Useful methods

- `rank()`: number of dimensions
- `num_elements()`: product of dimensions
- `num_bytes()`: total byte span from the first element to the last stride-covered region

## `Storage`

`Storage` is the raw allocation record:

- `mem_loc`
- `bytes`
- `ptr`

Multiple tensor views can point into the same `Storage`, possibly with different offsets or shapes.

## `TView`

`TView` is the runtime carrier of tensor data:

- `desc`
- `storage`

`TView::data()` returns `storage->ptr + desc.offset`.

This means a view combines:

- metadata from `TDesc`
- location of the underlying storage block
- optional slicing via `offset`

`TView` is non-owning.

## `Tensor`

`Tensor` is the owning wrapper used for direct allocations. It allocates either host or device memory from a `TDesc` and can expose a `TView` with `view()`.

In the runtime, compilation usually allocates `MemoryBlock` and `Storage` objects directly rather than storing a `Tensor` object per tensor ID.

## Tensor IDs vs storage IDs

The compiler distinguishes logical tensor identity from physical storage identity.

### Tensor ID (`tid`)

A tensor ID identifies a logical edge-produced value. Every node output gets a fresh tensor ID, and downstream input slots reference that tensor ID.

### Storage ID (`sid`)

A storage ID identifies the backing memory block. Several tensor IDs may share one storage ID:

- because of in-place execution
- because the same produced tensor is consumed by many downstream edges

### Example

```text
Node A out[0] -> tid 7 -> sid 4
Node B in[0]  -> tid 7 -> sid 4
Node B out[0] -> tid 8 -> sid 4   // in-place alias
```

`tid 7` and `tid 8` are different logical values in the plan, but they reuse the same underlying allocation.

## Why views instead of copies

The design avoids implicit data movement:

- scheduler updates spans of `TView`
- tasks read or write through those views
- ownership rules decide who may allocate or recycle the underlying storage

This keeps the runtime compatible with:

- GPU-heavy operators
- in-place transforms
- stateful async queues
- borrowed externally-managed buffers

## Pseudo algorithm: reading a view

```text
input: TView view

assert view.storage != null
base_ptr = view.storage.ptr
data_ptr = base_ptr + view.desc.offset
interpret bytes according to:
  - view.desc.dtype
  - view.desc.shape
  - view.desc.strides
```

## Practical guidance

- Prefer letting the compiler allocate scheduler-managed buffers unless your task truly needs private ownership.
- Use `offset` and `strides` only when your operator genuinely works with non-contiguous layouts.
- Treat `TView` as borrowed memory unless your task explicitly owns the slot through the factory contract.
