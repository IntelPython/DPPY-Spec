# Problems to be addressed (Goals)

[Helpful mental picture] By **offload** we mean submit jobs using **SYCL**. [ OpenMP is out of scope ]

1. Have expressive **semantics** to indicate offload target. Must be applicable to Python ecosystem.
   1. Where is the data 
   2. Where the computation happens
   3. What granularity are Python user controls (is offload specified in terms of devices, or queues)?
      Any distinction between power users/casual users? Addressing queues by device selector string should be allowed.
2. **Interoperability**. Data from `pkgA` must be understood by `pkgB` (including SYCL kernels of `pkgB` must be able to access data produced by `pkgA`, copy should be possible even if can be avoided).
   1. Python types of data containers are likely different (conversion is likely required).
   2. Accessibility of USM data from computation queue (dev + ctxt) must be ensured. [default device context in Sycl for root devices solves this for most use-cases]
   3. Zero-copy when possible (complexity of conversion is independent of data-size).
3. Package **developers** must be **enabled** for ecosystem to grow. 
4. Semantics should allow for **ease of use** with sensible defaults.
5. Python packages will always contain host-only Python code. Some will contain SYCL-powered code.

# Considerations

1. [Primary] Users must be always able to control where offloaded computation happens.
2. [Primary] Python packages should be free to implement how user exercises that choice, and what the default is without user's explicit input.
3. [Primary] Cross package experience. [ data-API attempts to address this ]
   1. (manual change of Python type of data object is likely required). 
      How much manual intervention is expected of user should minimized.
4. [Secondary] User should be able to offload computation on host data.
5. [Secondary] Should make it clear that the offload target only affects SYCL portion in the mix of host & SYCL code.
6. [Secondary] A solution with no global state entity in Python is preferred.

# Knobs to turn

**Computation queue**: where kernel is submitted
**Allocation queue**: device & context where USM data is allocated

When data associated with different queues need to be mixed, a copy to make array accessible for a given queue is needed. `array_accessible_by_queue_of_ary_other = ary.collocate(ary_other)`, or
``usm_array.asarray(ary, queue=ary_other.queue)``.

1. Universal knob: Every offload function is given a queue. (oneMKL)
   1. **Pros**: Unambiguous allows power users the fine-grain control over where computation takes place
   2. **Cons**: Forces Python user to think of `queue`. 
                 Not portable beyond SYCL.
                 Change function signature (e.g. when porting existing code from NumPy to dpnp)
2. Package manages default computation queue (TF, CuPy).
   New allocations are performed on computation queue.
   1. **Pros**: Offload target is explicit.
   2. **NB**: Need semantics to copy data to make it accessible to a given queue, like `asarray(ary, queue=ary_other.queue)`.
3. Package specific allocation queue, and Computation Follow Data paradigm infers the computation queue from data. (PyTorch).
   1. **Pros**: In likely scenario of single queue workload everything works.
                Computation queue is deduced.
   2. **Cons**: With multiple queues, deduction of computation queues can be ambiguous.
                 Suggested way to resolve this for power users needing precision is to make all 
                 arrays associated with the same queue, see `asarray`.
4. Global Queue Manager (`dpctl`): packages query for the queue to offload to/allocate with.
   1. Essentially equivalent to option 2.
   2. Package authors adopting controls as in option 2 can re-use `dpctl._SyclQueueManager` class.

# Addressable ecosystem

Of immediate interest is [PyData](https://pydata.org/) ecosystem grown around NumPy's `ndarray` as data structure

SYCL-powered array library needs to be developed, and should implement [data-API spec](https://data-apis.github.io/array-api/latest/), which is community driven standard co-developed with active participation of many PyData stake-holders.

SYCL-powered Python offload should work well with Python objects exposing their underlying host memory via [Python buffer protocol](https://docs.python.org/3/c-api/buffer.html).

Python offload controls should be able to control offload by Python calls to oneAPI performance libraries, e.g. oneMKL.

# Interoperability considerations

To interoperate Python packages need to be able to shared data.

For host memory, this usually means conversion of between package's respective Python data containers without explicit data copy (hence copy complexity is independent of the data-size).

Since USM data are SYCL context bound, packages must either use a common SYCL context, or be able to share a context between each other.

DPCPP was requested by deep learning extension developer teams to implement functionality to get a default single-device context per each SYCL root device. If `pkgA` and `pkgB` use that context offloading to the SYCL device selected using the same device selector, USM data allocated by `pkgA` will be accessible to `pkgB`. 

We recommend that USM data are shared using `__sycl_usm_array_interface__`, only that `syclobj` entry  in the interface dictionary can be a device identifier (either a triple of enums for `[backend, device_type, relative_id]`, or a single integer interpreted as a relative id for the backend and device type of the device selected by `sycl::default_selector{}`).

For power users who need to share data bound to other contexts, either non-default contexts, or contexts created for sub-devices, SYCL-powered Python ecosystem needs a way to share a context between packages. 

When sharing data in this case, the `__sycl_usm_array_interface__` must currently contain an instance of `dpctl.SyclContext` or `dpctl.SyclQueue`. It has been request that we accommodate passing of named Python capsule with a pointer to a SYCL object. 

`dpctl` should automate working with `__sycl_usm_array_interface__` as much as possible.

# Examples

## Package workflow: offloading to default queue

``````python
import pkg1 # Package implementing Knob_to_turn.Scenario_1
q = ....
A1 = allocate(q, (50, 10))
A2 = allocate(q, (10, 30))
A3 = pkg1.gemm(q, A1, A2)
```

```python
import pkg2 # Package implementing Knob_to_turn.Scenario_2

A1 = pkg2.allocate((50, 10)) # allocated on default queue
A2 = pkg2.allocate((10, 30))
A3 = pkg2.dot(A1, A2)   # offloaded to default queue

pkg2.set_default_queue("gpu:1") # change default queue
....
```

```python
import pkg3 # Package implementing Knob_to_turn.Scenario_3

A1 = pkg3.allocate((50, 10)) # allocated on default queue
A2 = pkg3.allocate((10, 30))
A3 = pkg2.dot(A1, A2)   # offloaded to queue inferred from data (default queue)
```

```python
import pkg4 # Package implementing Knob_to_turn.Scenario_4

A1 = pkg4.allocate((50, 10)) # allocated on manager's current queue
A2 = pkg4.allocate((10, 30))
A3 = pkg4.dot(A1, A2)   # offloaded to manager's current queue
```

## Package workflow: offloading to user specified queue

```python
import pkg1

alloc_q = ...
compute_q = ...

A1 = pkg1.allocate(alloc_q, shape1)
A2 = pkg1.allocate(alloc_q, shape2)

A3 = pkg1.compute(compute_q, A1, A2)
```

```python
import pkg2

# context manager is a syntactic sugar to set/restore
with pkg2.current("gpu:1"): # alloc device/queue
   A1 = pkg2.allocate(shape1)
   A2 = pkg2.allocate(shape2)

with pkg2.current("gpu:2"):
    A1c = A1.copy_to()
    A2c = A2.copy_to()
    A3 = pkg2.compute(A1c, A2c) # 
```

```python
import pkg3

# context manager is a syntactic sugar to set/restore
with pkg3.current("gpu:1"): # alloc device/queue
   A1 = pkg3.allocate(shape1)
   A2 = pkg3.allocate(shape2)

A1c = A1.copy_to("gpu:2")
A2c = A2.copy_to("gpu:2")
A3 = pkg3.compute(A1c, A2c)
```

```
import pkg4

... same as 2
```

## Interoperability

```
import pkgA
import pkgB

A = pkgA.allocate((50, 10)) # allocated on default queue

B = pkgB.func(A)  # for computational queue pkgB uses data in A must be accessible
    
```

```python
import pkg2
import pkg3

pkg2.set_default_queue("gpu:1")
pkg3.set_default_queue("gpu:1") # 

A4 = pkg3.allocate(....)

# context manager is a syntactic sugar to set/restore
with pkg2.current("gpu:1"):   # alloc device/queue for pkg2
   A1 = pkg2.allocate(shape1) # 
   A5 = pkg3.compute(A1)      # pkg3 works with its own default queue
   A6 = host_module.compute(A1, A5) # 
   A2 = pkg2.allocate(shape2)

with pkg2.current("gpu:2"):
    A1c = A1.copy_to()
    A2c = A2.copy_to()
    A3 = pkg2.compute(A1c, A2c) # 
```

```
cimport scipy.linalg.lapack_cython as lapack

lapack.func()
```

```
cimport scipy.linalg.oneAPIlapack_cython as lapack_oneAPI
cimport dpctl as c_dpctl

lapack = lapack_oneAPI(queue = c_dpctl.SyclQueue("gpu:1"))
lapack.svd(...)
```

```
import arLib1 # data-API conformant libraries
import arLib2

A1 = arLib1.empty(...)
A1c = arLib1.asarray(queue=q_used_by_arLib2)

A2 = arLib2.array( A1, device="dev" )
```

