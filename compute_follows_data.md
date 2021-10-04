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
data.to(queue, usm_type="device")
# OR
pkg.trasnfer_to(data, queue, usm_type="device")
```
E.g.

```
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
a_queue_2 = dpctl.transfer_to(a, queue="gpu_queue_1") # potentially zero-copy

# f() will be offloaded to "gpu_queue_2"
f(a_queue_2, b, c)
```

Output matrix:
device < shared < host
```markdown
|        | Device | Shared | Host   |
|--------|--------|--------|--------|
| Device | Device | Device | Device |
| Shared | Device | Shared | Shared |
| Host   | Device | Shared | Host   |
```
