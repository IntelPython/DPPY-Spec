# USMArray::goals

1. Supports USM allocated data of all `sycl::usm::alloc` types ("shared", "device", "host").
2. Has features needed to accommodate zero-copy sharing of `malloc_device` memory allocated by DL frameworks.
3. Implements operations on USM data using precompiled **SYCL** kernels **only**.
   1. Necessity for `usm::alloc::device`.
   2. Implementation applies to all types of `usm::alloc`.
4. Near terms goal: data-container supporting structural operations (all that can be supported using SYCL alone, which is data-API [core](https://data-apis.org/array-api/latest/purpose_and_scope.html) with exception of linear_algebra). This allows package to be lean (small binary size).
5. Data container used by `dpnp`.

# USMArray::implications

1. `USMArray` can **not** be **subclass** of `numpy.ndarray`.
2. `USMArray` must be **strided** to accommodate Torch/TF tensors
3. `USMArray` must be converted to NumPy explicitly (zero-copy op. for shared/host USM).

# USMArray::PyData_Evolution

1. Implement data-API compliant array library and promote data-API adoption upstream
2. Allow for zero-copy conversion from `USMArray(type=Union["shared","host"])` to `numpy.ndarray`.
3. Enabling path to adoption is **not** a guarantee of adoption. 
   1. Points in our favor: CPU array computations become multi-threaded
   2. Code can run across XPUs

# Cons of `numpy.ndarray` sub-classing as a mainstream data container

1. NumPy documentation [recommends](https://numpy.org/devdocs/reference/arrays.classes.html) against it, so [subclassing](https://numpy.org/doc/stable/user/basics.subclassing.html) is likely to create friction.

2. Subclass is an implicit view of USM shared memory as host memory (goes against explicit is better than implicit [Zen](https://www.python.org/dev/peps/pep-0020/))

   1. Explicitly conversion is recommended by data-API and Python philosophy.

3. Technical issues:

   1. Not possible to properly write sub-classing of  NumPy array in Cython (c.f. [Cython/issue/799](https://github.com/cython/cython/issues/799)).

   2. Subclass must guarantee use of USM buffer (impossible to accomplish using Numpy'a public API functionality).

      ```python
      numpy.ndarray.__new__(subclass, ...) -> "creates subclass instance with malloc memory rather than usm_shared memory"
      ```

   3. Possible overhead of dispatching [?]

4. Sub-classing also goes against the grain of data-API approach (see [assumptions::dependencies](https://data-apis.org/array-api/latest/assumptions.html#dependencies)).

5. Rules out support for `usm::alloc::device`

