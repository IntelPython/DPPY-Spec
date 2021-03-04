# Abstract

This article surveys the state of the art of accelerator device programming
(mostly CUDA) for existing Python packages and the proposed Python data API
standard.

# Computation Paradigms

Device management refers to how a Python package selects a SYCL device to use
both for memory allocation and kernel submission. We first review device
management implementations for few important Python packages.

## Tensorflow

Tensorflow (TF) implements the notion of "current device" or a device that is
selected by default for execution of TF kernels. By default, if a GPU device
(CUDA) is available then the GPU is selected. Programmers have the option to
change the current device manually<sup>[1](#ref1)</sup>.

```python
# Place tensors on the CPU
with tf.device('/CPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Run on the GPU
c = tf.matmul(a, b)
print(c)
```

In the above example, `a` and `b` are created on a CPU, but since
the `matmul` kernel is invoked outside the `tf.device` context the kernel
executes on the current device that is the GPU.

---
**NOTE**

- Does **not** implement **computation follow data**.
- Current device is always selected
- Copy is done automatically.
---

## CuPy

CuPy also uses a notion of current or default device<sup>[2](#ref2)</sup>.
CuPy operations will use the current device to allocate the memory for arrays
and enqueue kernels. The current device can be set programmatically.
However, data allocated on one device cannot be directly used on another device.

```python

# on default device that is GPU 0
x_on_gpu0 = cp.array([1, 2, 3, 4, 5])

with cp.cuda.Device(1):
   y_on_gpu1 = cp.array([1, 2, 3, 4, 5])
   z_on_gpu0 = x_on_gpu0 * y_on_gpu1  # raises error
```
---
**NOTE**

- Does **not** implement **computation follow data**.
- Current device is always selected
- Explicit copy of data is needed across devices.
---

## PyTorch

PyTorch has semantics that are closest to the Data API
proposal<sup>[3](#ref3)</sup>. Data is always allocated on a specific device
and operations on that data is placed on the device where the data was
allocated. PyTorch too has a notion of
current or selected device, but that only influences the allocation of new
tensors, operations on tensors explicitly allocated on another device are
not influenced by the selected device. Data copy across devices needed explicit
`to()` and `copy_()` functions.

```python
cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

x = torch.tensor([1., 2.], device=cuda0)
# x.device is device(type='cuda', index=0)
y = torch.tensor([1., 2.]).cuda()
# y.device is device(type='cuda', index=0)

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.tensor([1., 2.], device=cuda)

    # transfers a tensor from CPU to GPU 1
    b = torch.tensor([1., 2.]).cuda()
    # a.device and b.device are device(type='cuda', index=1)

    # You can also use ``Tensor.to`` to transfer a tensor:
    b2 = torch.tensor([1., 2.]).to(device=cuda)
    # b.device and b2.device are device(type='cuda', index=1)

    c = a + b
    # c.device is device(type='cuda', index=1)

    z = x + y
    # z.device is device(type='cuda', index=0)

    # even within a context, you can specify the device
    # (or give a GPU index to the .cuda call)
    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)
    f = torch.randn(2).cuda(cuda2)
    # d.device, e.device, and f.device are all device(type='cuda', index=2)
```

---
**NOTE**

- Implements **computation follow data**.
---


## Data API Proposal

The data-api proposes array creation functions to have an explicit `device`
keyword to allocate arrays on a specific device and the output of an operation
to be on the same device if possible<sup>[4](#ref4)</sup>. However, the
proposed behavior is a recommendation only and the proposal leaves room for
possible deviations. A current or default device is not precluded by
data-api standard semantics, and it up to libraries to decide what the default
or global device selection strategy should be.

# Using Context Managers for Device Management

Using a default or global device is useful to alleviate the need to have
explicit `device` keyword arguments or `to()` or `copy_()` calls for each array
creation routine. The global or default device can be changed temporarily with a
Python context manager or globally. The pattern is found in several Python
libraries, but is not without flaws. The following excerpts are useful in
understanding some of the issues.


---
> A context manager for controlling the default device is present in most existing array libraries (NumPy being the exception). There are concerns with using a context manager however. A context manager can be tricky to use at a high level, since it may affect library code below function calls (non-local effects). See, e.g., [this PyTorch issue](https://github.com/pytorch/pytorch/issues/27878) for a discussion on a good context manager API.
>
>Adding a context manager may be considered in a future version of this API standard.

*pasted verbatim from* [\[4\]](#ref4)

---
>There is one side-effect of with statements: Their effects propagate down the call stack, even into library code. Consider the following code:
>
> ```python
> # In library.py
> @njit(parallel=False)
> def library_func(...):
>    ...
>
> # In user.py
> with dppl.device_context(gpu):
>    ...
>    library_func(...)
>    ...
> ```
>
>In this case, if the with statement is preferred, the library loses control of what's executed inside library_func . Raising an error is okay, IMO, but for this reason, we shouldn't prefer what the with statement says.
>
>Personally, I'd be okay with changing the default value of parallel to `parallel=unspecified`.

*pasted verbatim from dpctl #12* [\[5\]](#ref4)

---

# References

<a name="ref1">[1]</a> https://www.tensorflow.org/guide/gpu#manual_device_placement

<a name="ref2">[2]</a> https://docs.cupy.dev/en/stable/tutorial/basic.html#current-device

<a name="ref3">[3]</a> https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics

<a name="ref4">[4]</a> https://data-apis.github.io/array-api/latest/design_topics/device_support.html#semantics

<a name="ref5">[5]</a> https://github.com/IntelPython/dpctl/issues/12