## Compute follows data semantics

Data should have the following retreivable attribute:
```
1. __sycl_usm_array_interface__
2. queue -> dpctl.SyclQueue
3. device -> dpctl.SyclDevice
4. usm_type -> "shared"|"device"|"host"
```

### Rules
---
1. Users are not allowed to mix arrays with different SYCL queues.
```
@numba_dppy.kernel
def f(a, b, c):
	d = c * b
	e = a + d
	return d + e

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")

# f() will be offloaded to "gpu_queue"
f(a, b, c)

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_1")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_2")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue_2")

# This will result in error as the queue to submit the kernel on can not be determined
f(a, b, c)
```
2. All usm-types are accessible from device. Users can mix arrays with different usm-type as long as they were allocated
   using the same SYCL queue.
   Considerations: Packages should warn about performance penalty when users mix usm-type. 
```
@numba_dppy.kernel
def f(a, b, c):
	d = c * b
	e = a + d
	return d + e

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="shared", queue="gpu_queue")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="host", queue="gpu_queue")

# f() will be offloaded to "gpu_queue" even though users have mixed usm type
f(a, b, c)
```
3. Packages need to provide the following functions to make data compatible for compute follows data:
```
data.as_array(queue, usm_type="device")
# OR
pkg.as_array(data, queue, usm_type="device")
```
E.g.

```
@numba_dppy.kernel
def f(a, b, c):
	d = c * b
	e = a + d
	return d + e

a = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_1")
b = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")

# This will result in error as the queue to submit the kernel on can not be determined
f(a, b, c)

# Users can use explicit conversion function to make data compatible
a_queue_2 = dpctl.as_array(a, queue="gpu_queue_2") # potentially zero-copy

# f() will be offloaded to "gpu_queue_2"
f(a_queue_2, b, c)
```

### Numpy.ndarray usage inside @numba_dppy.kernel
1. Users are allowed to pass numpy.ndarray(s) to `@numba_dppy.kernel`. Users will have to specify the SYCL queue that should be used to copy the `numpy.ndarray` data and to submit the kernel to. Numba_dppy provides `context_manager` to allow users to specify the SYCL queue.

```
def f(a, b, c):
	d = c * b
	e = a + d
	return d + e

a = numpy.as_array([1, 2, 3, 4])
b = numpy.as_array([1, 2, 3, 4])
c = numpy.as_array([1, 2, 3, 4])

# This will result in error as the queue to submit the kernel on can not be determined
f(a, b, c)

# Users will have to use numba_dppy's context manager to specofy the SYCL queue
# f() will be offloaded to "gpu_queue"
with numba_dppy.offload_to_sycl_device("gpu_queue"):
	f(a, b, c)

```

2. Users are not allowed to mix `numpy.ndarray` and `dpctl.tensor.usm_ndarray`.
```
def f(a, b, c):
	d = c * b
	e = a + d
	return d + e

a = numpy.as_array([1, 2, 3, 4])
b = numpy.as_array([1, 2, 3, 4])
c = dpctl.tensor.usm_ndarray([1, 2, 3, 4], type="device", queue="gpu_queue_2")


# This will result in error as the queue to submit the kernel on can not be determined
f(a, b, c)

# This will also result in error as the queue to submit the kernel on can not be determined
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
