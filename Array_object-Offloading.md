# Array object

### Attributes

* Array API

  ```python
  ary.dtype
  ary.device    # nature of this object is out of scope in Array-API
  ary.ndim
  ary.shape
  ary.size
  ary.T
  ```

* USMArray

  ```python
  ary.sycl_device   # to distinguish from ary.device
  ary.sycl_queue    # for uniformity with sycl_device
  ary.sycl_context
  ary.usm_type
  
  + array_API_attrs
  ```

```python
root_device=dpctl.SyclDevice("gpu")
root_device.device_selector_string -> "level_zero:gpu:0"
```

### Methods

https://data-apis.org/array-api/latest/design_topics/device_support.html#syntax-for-device-assignment

* Array API

  ```python
  ary.to_device(device)
  ```

  

# Array creation routines

NumPy: https://numpy.org/doc/stable/reference/routines.array-creation.html
Array-API: https://data-apis.org/array-api/latest/API_specification/creation_functions.html

```python
asarray(obj, /, *, dtype=None, device=None, copy=None)
    - obj can be a Python scalar, a (possibly nested) sequence of Python scalars, 
      or an object supporting DLPack or the Python buffer protocol (i.e. 
      memoryview(obj) works).
from_dlpack(x, /)

arange(start, /, *, stop=None, step=1, dtype=None, device=None)
empty(shape, /, *, dtype=None, device=None)
empty_like(x, /, *, dtype=None, device=None)
eye(N, /, *, M=None, k=0, dtype=None, device=None)
full(shape, fill_value, /, *, dtype=None, device=None)
full_like(x, fill_value, /, *, dtype=None, device=None)
linspace(start, stop, num, /, *, dtype=None, device=None, endpoint=True)
ones(shape, /, *, dtype=None, device=None)
ones_like(x, /, *, dtype=None, device=None)
zeros(shape, /, *, dtype=None, device=None)
zeros_like(x, /, *, dtype=None, device=None)
```

# SYCL ecosystem

### device keyword

Device keyword can be

- `None`  
- `"sycl_filter_string"`
- `dpctl.SyclDevice`

### Array creation routines

```Python
asarray(obj, /, *, 
        dtype=None, 
        device=None,         # use queue, not None, queue must be None, 
                             # or device inside must be compatible with this device
        copy=None, 
        usm_type='shared' 
)
```

```
# Supported device keyword values
device="cpu"
device=sycl_device -> q = dpctl.SyclQueue(sycl_device) # [X] disallowed for subdevices
device=sycl_queue -> q = sycl_queue                    # 
```

* For `obj` being another `usm_array`,  (zero-)copy is performed, otherwise a copy, possibly via host

```python
usm_array.asarray(   # copy shared array to a device array
    usm_shared_array,
    usm_type = 'device'
)

usm_array.asarray(    # copy as shared array to SYCL CPU device
    usm_shared_array_gpu,
    device='cpu'    
)
```

```
usm_array.copy_to_host(usm) 
	-> memoryview 
		-> numpy.ndarray
		
```

```
usm_array.asarray([
  numpy_array,
  usm_array
], usm_type='shared', device='gpu')
```

```
cpu_q=dpctl.SyclQueue("cpu")

device="cpu"
device=sycl_device -> q = dpctl.SyclQueue(sycl_device) # [X] disallowed for subdevices
device=sycl_queue -> q = sycl_queue                    # 

dpctl.SyclQueue(pycapsule)

-----
cpu_d=dpctl.SyclDevice("cpu")
d0, d1, d2 = cpu_d.create_subs_device(partition=(4,4,4))
ctx = dpctl.SyclContext((d0, d1, d2))
q0 = dpctl.SyclQueue(ctx, d0)
q1 = dpctl.SyclQueue(ctx, d1)

dpctl.SyclQueue( cap ) # 

import ipex
ipex.get_current_stream().get_capsule()
#
q_ref = new sycl::queue( q_ipex_in_stream )
PyCapsule_New( q_ref, delete_referce(o) )

# 
ipex.from_dlpack( dltensor_capsule )
```

```
usm_array_inst.device # 
     # queue, queue formats to relflect underlying device

usm_array_inst.device #
usm_array_inst.sycl_queue.sycl_device # 

daal4py._func(q, X_host, ...)
```

```
Torch(, device='cuda'
     ).__cuda_array_interface__
 # 
```



```
X0 = usm_array.asarray(
   obj, 
   device = "cpu"
   # device = q0
   # device=d0, # dpctl.SyclQueue(d0)
   # queue = q0
)

X1 = usm_array.asarray(
   obj, 
   #device=d0, # dpctl.SyclQueue(d0)
   queue=q1
)

# X0.context == X1.context
```

```
class Device:
   def __init__(self, queue=):
      self.queue=queue
   
```

```
daal4py.func( X_device, y_device )       # infers queue to offload to from arguments
daal4py.func( X_host, y_device )         # same
daal4py.func( X_host, y_host )           # uses host implementation
daal4py.func( X_host, y_host, queue=q )  # offloads to queue
```

----

# USMArray milestones

1. Object, with constructor, and copies: `copy_to_host`, `copy_from_host`, `copy_from_device` [2 weeks]
   1. Numba typing is independent
   2. dpnp is independent
   3. daal4py is independent
   4. populating usm_array functionality
      1. asarray
      2. Implement manipulation/extraction with support for strides
      3. Core Data-API excluding linear algebra subcomponent

```
obj.hasattr('__array_namespace__')
```

1. `dpnp` adopts `usm_array`.
2. 

---

```
torch.from_numpy( ndarray ) -> kCPU
torch.to_numpy( tensor.to('cpu') )   -> kCPU

ipex.from_dppy( usm_array ) -> xpu_tensor #
ipex.to_dppy( XPU_tensor ) -> usm_array #

ipex.from_dlpack( capsule ) -> tensor
ipex.to_dlpack( tensor ) -> capsule

tf.from_numpy
tf.to_numpy

itex.from_dppy( usm_array ) -> ...
itex.to_dppy( tf_obj )      -> usm_array 

itex.from_dlpack( capsule )
itex.to_dlpack( tf_obj )
```

```
( be, dty, rel_id ) -> device_id
 be, dty  (kSYCL_LZ_GPU, rel_id)
 
 DLDeviceType kCPU, kGPU, kGPU_PINNED, 
              kSYCL_OCL_CPU, kSYCL_OCL_GPU, 
```

---

```
usm_array.__sycl_usm_array_interface__
```

# Deadlines

* Code freeze for U2: May, 26
* Code free for NDA release: early-mid June

# 4/21/2021 meeting minutes

```
USMArray() -> usm_array_c_struct
```

```
func(sycl::queue & q, usm_array_struct A, .. B, event)

dpnp_func( void *p, n, ...)
```

`dpnp.sort(usm_array)`

```
b       *
i1, u1,
i2, u2,
i4, u4, * 
i8, u8, *
f2,
f4,     *
f8,     *
c8,     
c16
```

```
copy<typename T1, T2> strided_copy(q, A, B, events) 
```

```
USMArray.__set_item__(self, ind, B):    
```

```
func( obj ) -> obj  #
func( usm_array )   #
X[:2, :2]:          #
    compute_size   
    PyArray_New()
    
```

```
#
usm_array.var() -> xp.var(self)
```

