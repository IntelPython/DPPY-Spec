## Compute follows data semantics

Computation follow data stipulates that computation must be off-loaded to device where data is resident.

```python
# Computation follow data is unambiguous when
# all data reside on the same device 
# (in SYCL context all arrays have the same queue), 
# otherwise an error is raised
arLib_cfd.func( ar1, ar2, ar3, ...) # computation is offloaded to the common queue
```

**NB**: Adopting computation follow data precludes all arrays from being host data only.

Data should have the following retreivable attribute:
```
1. __sycl_usm_array_interface__
2. queue -> dpctl.SyclQueue
3. device -> dpctl.SyclDevice
4. usm_type -> "shared"|"device"|"host"
```

### Rules
---
We will begin by defining **equivalent SYCL queues**.

```
# Two SYCL queues are equivalent if they have the same:
# 1. SYCL context
# 2. SYCL device
# 3. Same queue properties

# dpctl package will provide facility to check queue equivalency.
```

Q: What happens when Queues are equivalent but not the same? In which queue do we submit the kernel?

A: If two SYCL queues are equivalent, the kernel can be submitted to either one. The decision is arbitrary and is left to be decided by individual package implementor. 


1. Users are not allowed to mix arrays with non-equivalent SYCL queues.
```
@numba_dppy.kernel
def f(a, b, c):
    i = numba_dppy.get_global_id(0)
    c[i] = a[i] * b[i]

# gpu_queue and gpu_queue_1 are not equivalent.
# gpu_queue_1 and gpu_queue_2 are equivalent.

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")

# f() will be offloaded to "gpu_queue"
f(a, b, c)

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_1")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_1")

# This will result in error as the queue to submit the kernel can not be determined
f(a, b, c)


a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_1")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_2")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_2")

# f() can be offloaded to "gpu_queue_1" or "gpu_queue_2" depending on how a package has implemented the decision making.
f(a, b, c)
```

2. All usm-types are accessible from device. Users can mix arrays with different usm-type as long as they were allocated
   using the same SYCL queue.

   Considerations: Packages can warn about performance penalty (if there is any) when users mix usm-type.
```
@numba_dppy.kernel
def f(a, b, c):
    i = numba_dppy.get_global_id(0)
    c[i] = a[i] * b[i]

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="host",   queue="gpu_queue")

# f() will be offloaded to "gpu_queue" even though users have mixed usm type
f(a, b, c)
```
3. Packages need to provide the following functions to make data compatible for compute follows data:
```
data.asarray(queue, usm_type="device")
# OR
pkg.asarray(data, queue, usm_type="device")
```
For example:

```
@numba_dppy.kernel
def f(a, b, c):
    i = numba_dppy.get_global_id(0)
    c[i] = a[i] * b[i]

# gpu_queue_1 and gpu_queue_2 are not equivalent.

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_1")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")

# This will result in error as the queue to submit the kernel can not be determined
f(a, b, c)

# Users can use explicit conversion function to make data compatible
a_queue_2 = dpctl.tensor.asarray(a, queue="gpu_queue_2") # potentially zero-copy

# f() will be offloaded to "gpu_queue_2"
f(a_queue_2, b, c)
```

### Numpy.ndarray usage inside @numba_dppy.kernel
1. Users are allowed to pass numpy.ndarray(s) to `@numba_dppy.kernel`. Users will have to specify the SYCL queue that should be used to copy the `numpy.ndarray` data and to submit the kernel. Numba_dppy provides `context_manager` to allow users to specify the SYCL queue.

```
@numba_dppy.kernel
def f(a, b, c):
    i = numba_dppy.get_global_id(0)
    c[i] = a[i] * b[i]

a = numpy.asarray([1, 2, 3, 4])
b = numpy.asarray([1, 2, 3, 4])
c = numpy.asarray([1, 2, 3, 4])

# This will result in error as the queue to submit the kernel can not be determined
f(a, b, c)

# Users will have to use numba_dppy's context manager to specify the SYCL queue
# f() will be offloaded to "gpu_queue"
with numba_dppy.offload_to_sycl_device("gpu_queue"):
	f(a, b, c)
```

2. Users are not allowed to mix `numpy.ndarray` and `dpctl.tensor.usm_ndarray`.
```
@numba_dppy.kernel
def f(a, b, c):
    i = numba_dppy.get_global_id(0)
    c[i] = a[i] * b[i]

a = numpy.asarray([1, 2, 3, 4])
b = numpy.asarray([1, 2, 3, 4])
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")


# This will result in error as the queue to submit the kernel can not be determined
f(a, b, c)

# This will also result in error as the queue to submit the kernel can not be determined

with numba_dppy.offload_to_sycl_device("gpu_queue"):
	f(a, b, c)

# This will also result in error. Although, the SYCL queue "gpu_queue_2" can be used to copy
# the numpy.ndarray data and to submit the kernel, it breaks the compute follows data as 
# inferring the SYCL queue can not be accomplished for data a and b.

with numba_dppy.offload_to_sycl_device("gpu_queue_2"):
	f(a, b, c)
```


### Output type inference for @numba.njit
---
Precedence of usm-type: `device < shared < host`

```markdown
|        | Device | Shared | Host   |
|--------|--------|--------|--------|
| Device | Device | Device | Device |
| Shared | Device | Shared | Shared |
| Host   | Device | Shared | Host   |
```
