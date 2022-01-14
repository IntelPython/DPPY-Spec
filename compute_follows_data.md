# Compute follows data

Computation followed by data (CFD) means that the computation is performed on the same device on which the data resides.

## Semantics

CFD is unambiguous when different data resides on the same device 
(i.e. in SYCL context all arrays have the same queue), 
otherwise CFD is ambiguius and error should be raised.

```python
arLib_cfd.func(ar1, ar2, ar3, ...)  # computation is offloaded to the common queue
```

**NB**: Adopting CFD precludes all arrays from being host data only.

Data should have following attributes:
```
1. __sycl_usm_array_interface__
2. queue -> dpctl.SyclQueue
3. device -> dpctl.SyclDevice
4. usm_type -> "shared"|"device"|"host"
```

### Equivalent SYCL queues

Definition of **equivalent SYCL queues** is following.

Two SYCL queues are equivalent if they have the same:
1. SYCL context
2. SYCL device
3. queue properties

`dpctl` package will provide facility to check queue equivalency.

Q: What happens when queues are equivalent but not the same? In this case what queue will be used to submit the kernel?  
A: If two SYCL queues are equivalent, the kernel can be submitted to either one. The decision is arbitrary and is left to be decided by individual package implementation. 

### Rules

Rules for applying CFD semantics and selecting queue for offloading are following.

Assume we have example kernel function:

```python
@numba_dppy.kernel
def f(a, b, c):
    i = numba_dppy.get_global_id(0)
    c[i] = a[i] * b[i]
```

1. Users are not allowed to mix arrays with non-equivalent SYCL queues.

In following examples:
- `gpu_queue` and `gpu_queue_1` **are not** equivalent.
- `gpu_queue_1` and `gpu_queue_2` **are** equivalent.

```python
a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")

f(a, b, c)  # offloading to gpu_queue
```

```python
a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_1")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_1")

f(a, b, c)  # raising error as queues are not equivalent
```

```python
a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_1")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_2")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_2")

f(a, b, c)  # offloading to gpu_queue_1 or gpu_queue_2 depending on implementation
```

2. All unified shared memory types (USM-types) are accessible from device. Users can mix arrays with different USM-type as long as they were allocated
   using the same SYCL queue.

Considerations: Packages can warn about performance penalty (if there is any) when users mix USM-type.

```python
a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="host",   queue="gpu_queue")

f(a, b, c)  # offloading to gpu_queue even if USM-types are different
```

3. Packages should provide following helper functions to make data compatible for CFD.

```python
data.asarray(queue, usm_type="device")
# OR
pkg.asarray(data, queue, usm_type="device")
```

In following example `gpu_queue_1` and `gpu_queue_2` **are not** equivalent.

```python
a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_1")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")

f(a, b, c)  # raising error as queues are not equivalent
```

Users can explicitly convert data to make it CFD compatible.

```python
a_queue_2 = dpctl.tensor.asarray(a, queue="gpu_queue_2")  # potentially zero-copy

f(a_queue_2, b, c)  # offloading to gpu_queue_2
```

### NumPy arrays and numba-dppy kernels

1. Users are allowed to pass numpy.ndarray(s) to `@numba_dppy.kernel`. Users will have to specify the SYCL queue that should be used to copy the `numpy.ndarray` data and to submit the kernel. Numba_dppy provides `context_manager` to allow users to specify the SYCL queue.

Following examples will reuse kernel function from above.

```python
a = numpy.asarray([1, 2, 3, 4])
b = numpy.asarray([1, 2, 3, 4])
c = numpy.asarray([1, 2, 3, 4])

f(a, b, c)  # raising error as the queue to submit the kernel can not be determined
```

Users should use context manager to specify the SYCL queue.

```python
with numba_dppy.offload_to_sycl_device("gpu_queue"):
    f(a, b, c)  # offloading to gpu_queue
```

2. Users are not allowed to mix `numpy.ndarray` and `dpctl.tensor.usm_ndarray`.

```python
a = numpy.asarray([1, 2, 3, 4])
b = numpy.asarray([1, 2, 3, 4])
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")

f(a, b, c)  # raising error as the queue to submit the kernel can not be determined

with numba_dppy.offload_to_sycl_device("gpu_queue"):
    f(a, b, c)  # raising error as the queue to submit the kernel can not be determined
```

Although, the SYCL queue `gpu_queue_2` can be used to copy the `numpy.ndarray` data and
to submit the kernel, it breaks the CFD 
as inferring the SYCL queue can not be accomplished for data `a` and `b`.

```python
with numba_dppy.offload_to_sycl_device("gpu_queue_2"):
    f(a, b, c)  # raising error as inferring the SYCL queue can not be accomplished for data a and b
```

### Output type inference for @numba.njit

Precedence of USM-types: `device < shared < host`

```markdown
| USM-type | Device | Shared | Host   |
|----------|--------|--------|--------|
| Device   | Device | Device | Device |
| Shared   | Device | Shared | Shared |
| Host     | Device | Shared | Host   |
```
