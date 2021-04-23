# The DPPY Programming Model

## Array Creation

### Data API proposal

```python
ary.dtype
ary.device    # nature of this object is out of scope in Array-API
ary.ndim
ary.shape
ary.size
ary.T
```

### Array creation routines

* NumPy: https://numpy.org/doc/stable/reference/routines.array-creation.html
* Array-API: https://data-apis.org/array-api/latest/API_specification/creation_functions.html

```python
asarray(obj, /, *, dtype=None, device=None, copy=None)
"""
obj can be a Python scalar, a (possibly nested) sequence of Python scalars,
or an object supporting DLPack or the Python buffer protocol (i.e.
memoryview(obj) works).
"""
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

### USM array `asarray` prototype

```Python
asarray(
    obj, /, *,
    dtype=None,
    device=None,
    copy=None,
    usm_type='shared'
)
```

The `device` keyword can be

- `None`
- `"sycl_filter_string"`
- `dpctl.SyclDevice`
- `dpctl.SyclQueue`

```
# E.g.
device = "cpu"
device = dpctl.SyclQueue() # [X] disallowed for sub-devices
device = dpctl.SyclDevice()
```

## Compute follows data

A function executes where the input data is allocated and the output data is
generated on the same device. A prominent example of this design in PyTorch.

```Python
a = torch.tensor([1., 2.], device=cuda)
b = torch.tensor([1., 2.], device=cuda)
# the addition executes on the CUDA GPU device and the output tensor `c` is
# also allocated on the same device.
c = a + b
```

### Rules

```Python
def func(a, b):
    """ Both a and b need to be allocated on the same `device`. An exception
    is raised if the arguments are not on the same device.
    """
    pass
```

1. All arguments should be on the same `device` or `queue`.

```Python
a = usm_array(1024, device = q0)
b = usm_array(1024, device = q1)

func(a, b)  # Raise exception

```

2. A library is free to implement wrapper calls to "transfer" array ownership without explicit copy.

```Python
cpu_d = dpctl.SyclDevice("cpu")
d0, d1, d2 = cpu_d.create_subs_device(partition=(4,4,4))
ctx = dpctl.SyclContext((d0, d1, d2))
q0 = dpctl.SyclQueue(ctx, d0)
q1 = dpctl.SyclQueue(ctx, d1)

a = usm_array(1024, device = q0)
b = usm_array(1024, device = q1)

a = usm_array(a, device = q1)
func(a, b)
```

3. Host data needs to be explicitly copied over to adhere to compute follow data model.

```Python
a = numpy.ones(1024)
b = numpy.ones(1024)

a = usm_array(a, device = "gpu")
b = usm_array(b, device = "gpu")

func(a, b)
```
