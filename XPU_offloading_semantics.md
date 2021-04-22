# XPU Offloading semantics

**Offloading** means **executing kernels on SYCL queue** targeting XPU device (GPU/CPU/Accelerator).

## Data containers

USM allocated data is bound to SYCL context. Makes sense for them to carry SYCL queue.

```python
usm_array.queue -> dpctl.SyclQueue
usm_array.device -> dpctl.SyclDevice
usm_array.usm_type -> "shared"|"device"|"host"
```

Python users should be able to offload computation on host data, preferably without making an explicit copy.

## Discussion of "Computation Follow Data" in SYCL context

Computation follow data stipulates that computation must be off-loaded to device where data is resident.

```python
# Computation follow data is unambiguous when
# all data reside on the same device 
# (in SYCL context all arrays have the same queue), 
# otherwise an error is raised
arLib_cfd.func( ar1, ar2, ar3, ...) # computation is offloaded to the common queue
```

**NB**: Adopting computation follow data precludes all arrays from being host data only.

In SYCL, computations on USM data can take place when USM pointers are bound to the same **context** that is used in the queue. Hence `ar1` and `ar2` may have different queues, but must have the same context.

In multi-tile computation it is natural to create contexts encompassing multiple sub-devices, and want to perform computation  on tile-A on data allocated on tile-B.

```python
root_dev = dpctl.SyclDevice("gpu")
sub_devs = root_dev.create_sub_devices(partition="L2_cache")
ctx = dpctl.SyclContext(sub_devs)
qs = [ dpctl.SyclQueue(ctx, sd) for sd in sub_devs ]

# allocate memory on qs[1] and compute on it on qs[0]
```

Adopting this means walking away from computation follow data.

```python

arLib_cfd.func(ar1, ar2) # ar1.sycl_context == ar2.sycl_context
                         # ar1.queue != ar2.queue
                         # which device should computation per performed at?
 # HPC user might want **explicit** control to ensure load-balancing 
 # in distributed computations.
 
ar3 = ar2.copy_to(ar1.queue)
arLib_cfd.func(ar1, ar3) 

ar3 = usm_array.asarray(ar2, device=ar1.sycl_queue)
arLib_cfd.func(ar1, ar3)
```

## Explicit control for offloading

DL provide **explicit** control for where computation is offloaded to using concept of current device (would be current queue in SYCL context).

Wish-list:

- Control belongs to individual Python package, tailored to its need. (Pythonic, flexible)
- Global state is avoided
- [Requirements added] Explicit control
- Unified programming model
- Ease of use
- Combine GPU & CPU

```python
import dpnp # dpnp offloads to a default queue by default
dpnp.empty() # allocated on default queue

gpu1dpnp = dpnp.offload_to("gpu:1")

dpctl.set_global_queue("gpu")
d4py_gpu.linear_regression().compute(X) # 
dpctl.reset_global_queue()

dpctl.set_global_queue("gpu")
d4py_host.linear_regression().compute(X) # 
dpctl.reset_global_queue()


# for power users
arLib = dpnp.offload_to("gpu") # create Python module in which all functions
                               # operate on common queue, created during this call
                               # queue can not be changed.
                               # def add(*args):
                               #     return add_impl(module_q, args)
res1 = arLib.add( numpy_array, usm_array)
res2 = arLib.concat(usm_array_tileA, usm_array_tileB)

res2.__array_module__ -> arLib
```

```
import pkgA
import pkgB

gpuA = pkgA.offload_to("gpu") 
gpuB = pkgB.offload_to("gpu") 

gpuA.compute()
gpuB.compute()
```

```
resA = gpuA.compute()     #
resA.make_available(gpuB.get_queue())
recC = gpuB.compute(resA) # 
```

```
torch_ipex.from_usm_array(X) -> torch.Tensor.at('xpu')
```

