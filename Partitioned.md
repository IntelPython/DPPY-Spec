**Draft**

# Motivation
When operating in distributed memory systems a data container (such as tensors of data-frames) may be partitioned into several smaller chunks. An implementation of operations defined specifically for such a partitioned data container can easily make use of the specifics of the structure to provide good performance. Most prominently, making use of the data locality can be vital. For a consumer of such a partitioned container it will be inconvenient (if not impossible) to write specific code for any possible partitioned data container. On the other hand, like with the array interface, it would be much better to have a standard way of extracting the meta data about the partitioned nature of a given container.

The goal of the `__partitioned_interface__` protocol is to allow partitioned data containers to expose partitioning information to consumers so that unnecessary copying can be avoided as much as possible.

# Scope
Currently the focus is dense data structures with rectangular shapes, such as dense nd-arrays and DataFrames. The data structures compose their data-(index-)space into several rectangular partitions which form a regular, multi-dimensional grid.

While the interface is designed to be generic enough to cover any distribution backend, the currently considered backends are Ray, Dask and MPI-like (SPMD assigning ids to each process/rank/PE).

# Partitioned Interface
A conforming implementation of the partitioned-interface standard must provide and support a data structure object having a `__partitioned_interface__` method which returns a Python dictionary with the following fields:
* `shape`: a tuple defining the number of partitions per dimension of the container's global data-(index-)space.
* `partitions`: a dictionary mapping a position in the partition grid (as defined by `shape`) to a dictionary providing the partition object, the partition shape and locality information.
* `locals`: Only for SPMD/MPI-like: list of the positions of the locally owned partitions. The positions serve as lookup keys in the `partitions` dictionary. Must not be available if not SPMD/MPI-like.

In addition to the above required keys a container is encouraged to provide more information that could be potentially benefitial for consuming the distributed data structure. 

## `shape`
The shape of the partition grid must be of the same dimensionality as the underlying data-structure. `shape` provides the number of partitions along each dimension. Specifying `1` in a given dimension means the dimension is not cut.

## `partitions`
A dictionary mapping a position in the grid to information for each partition.
* All positions in the partition-grid as defined by `shape` must be present in the dictionary as a key.
* The position in the grid (key) is provided as a tuple with the dimensionality as the partition grid.

Each key/position maps to
* 'start': global start indices of the partition (same dimensionality as the underlying data-structure) given as tuple
* 'shape': shape of the partition (same dimensionality as the shape of the partition grid) given as tuple
* 'data': the actual data provided as ObjRef, Future, array or DF or...
* 'location': The location (or home) of the partition, given as ip-address or rank or...

A consumer must verify it supports the provided object types and locality information and should throw and exception if not.

In addition to the above required keys a container is encouraged to provide more information that could be potentially beneficial for consuming the distributed data structure. For example, a DataFrame structure might add the row and column labels for any given partition.

* `start`
  * The offset of the starting element in the partition from first element in the global index space of the underlying data-structure. It has the same dimensionality as the underlying data-structure and is given as a tuple. The stop indices are computed as the sum of `start` and `shape`.
* `shape`
  * The shape of the partition has the same dimensionality as the underlying data-structure and is given as tuple.
* `data`
  * All data/partitions must be of the same type.
  * The partition type is opaque to the `__partitioned_interface__`
    * The consumer needs to deal with it like it would in a non-partitioned case. For example, if the consumer can deal with array it could check for the existence of the array interface. Other conforming consumers could hard-code a very specific type check.
    * When the underlying backend supports it and for all non-SPMD backends partitions must be provided as references. This avoids unnecessary data movement.
      * Ray: ray.ObjectRef
      * Dask: dask.Future
  * For SPMD-MPI-like backends: partitions which are not locally available may be `None`. This is the recommended behavior unless the underlying backend supports references such as promises to avoid unnecessary data movement.
* `location`
  * The location information must include all necessary data to uniquely identify the location of the partition/data. The exact information depends on the underlying distribution system:
    * Ray: ip-address
    * Dask: worker-Id (name, ip, or ip:port)
    * SPMD/MPI-like frameworks such as MPI, SHMEM etc: rank

## `locals`
This is basically a short-cut for SPMD environments which allows processes/ranks to quickly extract the local partition. It saves processes from parsing the `partitions` dictionary for the local rank/address which is helpful when the number of ranks/processes/PEs is large.

## Examples
### 1d-data-structure (64 elements), 1d-partition-grid, 4 partitions on 4 nodes, blocked distribution, partitions are of type `Ray.ObjRef`, Ray
```python
__partitioned_interface__ = {
  ‘shape’: (4,),
  ‘partitions’: {
      (0,):  {
        'start': (0,),
        'shape': (16,),
        'data': ObjRef0,
        'location': ‘1.1.1.1’, }
      (1,): {
        'start': (16,),
        'shape': (16,),
        'data': ObjRef1,
        'location': ‘1.1.1.2’, }
      (2,): {
        'start': (32,),
        'shape': (16,),
        'data': ObjRef2,
        'location': ‘1.1.1.3’, }
      (3,): {
        'start': (48,),
        'shape': (16,),
        'data': ObjRef3,
        'location': ‘1.1.1.4’, }
  }
}
```
### 2d-structure (64 elements), 2d-partition-grid, 4 partitions on 2 nodes, block-cyclic distribution, partitions are of type `dask.Future`, dask
```python
__partitioned_interface__ = {
  ‘shape’: (2,2),
  ‘partitions’: {
      (1,1): {
        'start': (4, 4),
        'shape': (4, 4),
        'data': future0,
        'location': ‘Alice’, },
      (1,0):  {
        'start': (4, 0),
        'shape': (4, 4),
        'data': future1,
        'location': ‘1.1.1.2:55667’, },
      (0,1):  {
        'start': (0, 4),
        'shape': (4, 4),
        'data': future2,
        'location': ‘Alice’, },
      (0,0): {
        'start': (0,0),
        'shape': (4, 4),
        'data': future3,
        'location': ‘1.1.1.2:55667’, },
  }
}
```
### 2d-structure (64 elements), 1d-partition-grid, 4 partitions on 2 ranks, row-block-cyclic distribution, partitions are of type `pandas.DataFrame`, MPI
```python
__partitioned_interface__ = {
  ‘shape’: (4,1),
  ‘partitions’: {
      (0,0):  {
        'start': (0, 0),
        'shape': (2, 8),
        'data': df0,   # this is for rank 0, for rank 1 it'd be None
        'location': 0, },
      (1,0):  {
        'start': (16, 0),
        'shape': (2, 8),
        'data': None,   # this is for rank 0, for rank 1 it'd be df1
        'location': 1, },
      (2,0):  {
        'start': (32, 0),
        'shape': (2, 8),
        'data': df2,   # this is for rank 0, for rank 1 it'd be None
        'location': 0, },
      (3,0):  {
        'start': (48, 0),
        'shape': (2, 8),
        'data': None,   # this is for rank 0, for rank 1 it'd be df3
        'location': 1, },
  }
  'locals': [(0,0), (2,0)] # this is for rank 0, for rank 1 it'd be [(1,0), (3,0)]
}
```
