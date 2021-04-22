# USM array data container specs

USM array represents a logic view into 1D contiguous array of USM memory to represent `r`-dimensional type-homogenous array with shape `(n[0], ..., n[r-1])`. Referencing an element of such an array requires specifying `r`-dimensional tuple of indexes `(i[0], ..., i[r-1]`) such that for all `0<=k<r` we have `0<=i[k] <n[k]`.

Location of the individual element of the array referenced by the index is determined by strides.

```
a[(i[0], ..., i[r-1])] == 
    buf_flat[ sum(strides[k]*i[k] for k in range(r) ) ]
```

Thus `strides[k]` describes the number of array elements we jump in 1D physical array when index `i[k]` increases by 1 while other indexes remain fixed.

The array container delegates ownership of memory to `dpctl.memory.MemoryUSM*` per `usm_type` of memory it represents. The Python object representing this memory is stored in `USMArray.base`.

```python
cdef public class USMArray [object PyUSMArrayObject, type PyUSMArrayType]:
    cdef int nd               # number of indexes needed to reference an element
    cdef char* data           # usm pointer representing location of element 
                              # with zero multi-index
    cdef Py_ssize_t *shape    # ranges of indexes, e.g. (2, 5, 1)
    cdef Py_ssize_t *strides  # Number of elements to jump with unit increase
                              # of each index value
    cdef int typenum          # enum representing array element data type 
    cdef int flags            # flags representing cached property of array
                              # USM_ARRAY_C_CONTIGUOUS, USM_ARRAY_F_CONTIGUOUS,
                              # USM_ARRAY_WRITEABLE
    cdef object base          # reference to object that owns the memory, the object 
                              # must be dpctl.memory.MemoryUSM*, ample enough 
                              # to accomodate all elements of the array
```

### Examples of USMArray instances

* 2D C-contiguous array (matrix) of shape (2, 3)

```text
nd: 2
shape: (2,3)
strides: (3, 1)

data: points to &a_flat[0]
buf_flat = {a[0,0], a[0,1], a[0,2], a[1, 0], a[1, 1], a[1, 2]}
strides[0]*i[0] + strides[1]*i[1]:
            {    0,       1,      2,       3,       4,       5 }
```

* 2D F-contiguous array (matrix) of shape (3, 2)

```text
nd: 2
shape: (2, 3)
strides: (1, 2)
data: points to &a_flat[0]
buf_flat = {a[0, 0], a[1,0], a[0, 1], a[1, 1], a[0, 2], a[1, 2]}
strides[0]*i[0] + strides[1]*i[1]:
             {    0,       1,      2,       3,       4,       5 }
```

* 2D non-contiguous array  of shape (2, 2), representing 2-2 submatrix `{{a[0,0], a[0, 1]}, {a[1,0], a[1,1]}}` in the C-contiguous layout (first example)

```text
nd: 2
shape: (2, 2)
strides: (3, 1)
data: points to &a_flat[0]
buf_flat: {a[0,0], a[0,1], *, a[1,0], a[1,1], *}
strides[0]*i[0] + strides[1]*i[1]:
           {    0,       1,       3,    4       }
```

* 1D non-contiguous array with negative strides `contig_vec[7::-2]`

```text
nd: 1
shape: (4, )
strides: (-2,)
data: points to &a_flat[7]
buf_flat = { *, contig_vec[1], 
           *, contig_vec[3], 
           *, contig_vec[5],
           *, contig_vec[7]  }
strides[0]*i[0] + strides[1]*i[1]:
         {     offset - 6, 
               offset - 4, 
               offset - 2, 
               offset - 0 }
  offset: number of elements from buffer left boundary to item with index (0,)
```

  

## Constructor

```python
USMArray(
    shape,                       # shape: tuple of integer
    dtype="|f8",                 # type_str for array element
                                 # np.dtype(type_str) must be 
                                 # subdtype of either integer or inexact 
    buffer='device',             # 'shared', 'host', 'device' to allocate new memory,
                                 # or existing _Memory object, or USMArray object
    strides=None,                # layout info: None of tuple of integers of 
                                 # same size of shape
    offset=0,                    # position of zero-multi-index in buffer
    order='C',                   # how to interpret strides=None
    buffer_ctor_kwargs = dict()  # additional keywords to pass to new memory constructor
                                 # supported keywords of `dpctl.memory.MemoryUSM*`
                                 # constructors: 'queue', 'alignment', 'copy'
)
```

### Examples

```python
USMArray((2,3), dtype='u2', buffer='device')
      # allocates 12 bytes of USM device memory for (2,3) array of 
      # unsigned shorts, C-contiguous
      # buffer has 12 elements
      # buf_flat = {a[0,0], a[0,1], a[0,2], a[1,0], a[1, 1], a[1,2]}
```

```python
USMArray((2,3), dtype='i8', buffer='shared', strides=(6, 1))
      # allocates 9*8 bytes of USM-shared memory for (2, 3) array of 
      # int64, strided
      # buffer has 9 elements
      # buf_flat = {a[0,0], a[0, 1], a[0, 2], __, __, __, a[1, 0], a[1, 1], a[1, 2]}
```

```python
USMArray((2, 2), dtype='u1', buffer='host', strides=(2, -1))
      # allocates 4 bytes of USM-host memory for (2,2) array of unsigned char
      # buffer has 4 elements
      # buf_flat = {a[0, 1],     a[0,0], a[1,1],     a[1,0]}
      #  addr:     { offset - 1, offset, offset + 1, offset + 2}
      # offset was auto-computed to be 1
```

```python
# object mem is MemoryUSMShared of 64 bytes
USMArray((4,), dtype='f8', buffer=mem, strides=(-2,), offset = 7)
     # map 4-elements of doubles to the buffer
     # buf_flat = {__, a[3], __, a[2], __, a[1], __, a[0]} 
     #   addr:    {    offset-6, offset-4, offset-2, offset }
```

```python
# passing USMArray in place of the buffer is equivalent to passing USMArray's usm_data
W = usm_array.USMArray((4, 2), dtype='i4', buffer='device', strides=(-5, -2))
Wiface = W.__sycl_usm_array_interface__
W2 = usm_array.USMArray(
         Wiface['shape'],           # same shape
         dtype=Wiface['typestr'],   # same type
         buffer=W,                  # reuse the USM-memory buffer underlying USMArray
         strides=Wiface['strides'], # use same strides
         offset=Wiface['offset']    # 
)
W2iface = W2.__sycl_usm_array_interface__
print(Wiface == W2iface) # prints True
```

#### Warning

Integrity of ``__sycl_usm_array_interface__`` dictionary (i.e. accessibility of all array elements) remains responsibility of the user. It is possible to craft `offset`, `strides` or `shape` that would result in access outside of the boundary of allocated USM data. Doing so may result in a crash, or a runtime-error.

#### Remark

`USMArray` constructor, much like `numpy.ndarray` constructor, is expected to be for the internal use, and is not expected to have external usage.