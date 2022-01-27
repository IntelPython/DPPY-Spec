**Draft**

# Motivation
When operating in distributed memory systems a data container (such as tensors or data-frames) needs to operate on several subsets of the data, each assigned to distinct (distributed) address spaces. An implementation of operations defined specifically for such a distributed data container will likely need to use specifics of the structure to provide good performance. Most prominently, making use of the data locality can be vital. For a consumer of such a partitioned container it will be inconvenient (if not impossible) to write specific code for any possible partitioned and/or distributed data container. On the other hand, like with the array API, it would be much better to have a standard way of extracting the meta data about the current partitioned and distributed state of a given container.

The goal of the `__partitioned__` protocol is to allow partitioned and distributed data containers to expose information to consumers so that unnecessary copying can be avoided as much as possible, ideally zero-copy.

# Scope
Currently the focus is dense data structures with rectangular shapes, such as dense nd-arrays and DataFrames. The data structures compose their data-(index-)space into several rectangular partitions which form a multi-dimensional grid.

While the protocol is designed to be generic enough to cover any distribution backend, the currently considered backends are Ray, Dask and MPI-like (SPMD assigning ids to each process/rank/PE).

# Non-Goals
This protocol accepts that multiple runtime systems for distributed computation exist which do not easily compose. A standard API for defining data- and control-flow on distributed systems would allow an even better composability. That is a much bigger effort and not a goal of this protocol.

Currently neither non-rectangular data structures, nor non-rectangular partitions nor irreglar partitions grids are considered.

# Partitioned Protocol
A conforming implementation of the partitioned protocol standard must provide and support a data structure object having a `__partitioned__` property which returns a Python dictionary with the following fields:
* `shape`: a tuple defining the number of elements for each dimension of the global data-(index-)space.
* `partition_tiling`: a tuple defining the number of partitions for each dimension of the container's global data-(index-)space.
* `partitions`: a dictionary mapping a position in the partition grid (as defined by `partition_tiling`) to a dictionary providing the partition object, the partition shape and locality information.
* `locals`: Only for SPMD/MPI-like implementations: list of the positions of the locally owned partitions. The positions serve as look-up keys in the `partitions` dictionary. Must not be available if not SPMD/MPI-like (such as when Ray- or Dask-based).
* `get`: A callable converting a sequence of handles into a sequence of data objects.

The `__partitioned__` dictionary must be pickle'able.

## `shape`
The `shape` field provides the dimensionality and sizes per dimension of the underlying data-structure.

## `partition_tiling`
The shape of the partition grid must be of the same dimensionality as the underlying data-structure. `partition_tiling` provides the number of partitions along each dimension. Specifying `1` in a given dimension means the dimension is not cut.

## `get`
A callable must return raw data object when called with a handle (or sequence of handles) provided in the `data` field of an entry in `partition`. Raw data objects are standard data structures: DataFrame or nd-array.

## `locals`
A short-cut for SPMD environments allows processes/ranks to quickly extract their local partition. It saves processes from parsing the `partitions` dictionary for the local rank/address which is helpful when the number of ranks/processes/PEs is large.

## `partitions`
A dictionary mapping a position in the grid to detailed information about the partition at the given position.
* All positions in the partition-grid as defined by `partition_tiling` must be present in the dictionary as a key.
* The position in the grid (key) is provided as a tuple with the same dimensionality as the partition grid.

Each key/position maps to
* 'start': The offset of the starting element in the partition from the first element in the global index space of the underlying data-structure, given as a tuple
* 'shape': Shape of the partition (same dimensionality as the shape of the partition grid) given as a tuple
* 'data': The actual data provided as ObjRef, Future, array or DF or...
* 'location': The location (or home) of the partition within the memory hierachy, provided as a tuple (IP, PID[, device])

A consumer must verify it supports the provided object types and locality information; it must throw an exception if not.

### `start`
* The offset of the starting element in the partition from first element in the global index space of the underlying data-structure. It has the same dimensionality as the underlying data-structure and is given as a tuple. The stop indices can be computed by the sum of `start` and `shape`.
### `shape`
* Number of elements in each dimension. The shape of the partition has the same dimensionality as the underlying data-structure and is given as tuple.
### `data`
* The actual data of the partition, potentially provided as a handle.
* All data/partitions must be of the same type.
* The actual partition type is opaque to the `__partitioned__`
  * The consumer needs to deal with it like it would in a non-partitioned case. For example, if the consumer can deal with array it could check for the existence of the array/dataframe APIs (https://github.com/data-apis/array-api, https://github.com/data-apis/dataframe-api). Other conforming consumers could hard-code a very specific type check.
  * Whenever possible the data should be provided as references. References are mandatory of non-SPMD backedns. This avoids unnecessary data movement.
    * Ray: ray.ObjectRef
    * Dask: dask.Future
  * It is recommended to access the actual data through the callable in the 'get' field of `__partitioned__`. This allows consumers to avoid checking handle types and container types.
* For SPMD-MPI-like backends: partitions which are not locally available may be `None`. This is the recommended behavior unless the underlying backend supports references (such as promises/futures) to avoid unnecessary data movement.
### `location`
* A sequence of locations where data can be accessed locally, e.g. without extra communication
* The location information includes all necessary data to uniquely identify the location of the partition/data within the memory hierachy. It is represented as a tuple: (IP, PID[, device])
  * IP: a string identifying the address of the node in the network
  * PID: an integer identifying the unique process ID of the process as assigned by the OS
  * device: optional string identifying the device; if present, assumes underlying object exposes `dlpack` and the value conforms to `dlpack`. Default: 'kDLCPU'.

## Examples
### 1d-data-structure (64 elements), 1d-partition-grid, 4 partitions on 4 nodes, blocked distribution, partitions are of type `Ray.ObjRef`, Ray
```python
__partitioned__ = {
  'shape': (64,),
  'partition_tiling': (4,),
  'partitions': {
      (0,):  {
        'start': (0,),
        'shape': (16,),
        'data': ObjRef0,
        'location': [('1.1.1.1’, 7755737)], }
      (1,): {
        'start': (16,),
        'shape': (16,),
        'data': ObjRef1,
        'location': [('1.1.1.2’, 7736336)], }
      (2,): {
        'start': (32,),
        'shape': (16,),
        'data': ObjRef2,
        'location': [('1.1.1.3’, 64763578)], }
      (3,): {
        'start': (48,),
        'shape': (16,),
        'data': ObjRef3,
        'location': [('1.1.1.4’, 117264534)], }
  },
  'get': lambda x: ray.get(x)
}
```
### 2d-structure (64 elements), 2d-partition-grid, 4 partitions on 2 nodes with 2 workers each, block-cyclic distribution, partitions are of type `dask.Future`, dask
```python
__partitioned__ = {
  'shape’: (8, 8),
  'partition_tiling’: (2, 2),
  'partitions’: {
      (1,1): {
        'start': (4, 4),
        'shape': (4, 4),
        'data': future0,
        'location': [('1.1.1.1', 77463764)], },
      (1,0):  {
        'start': (4, 0),
        'shape': (4, 4),
        'data': future1,
        'location': [('1.1.1.2', 9174756892)], },
      (0,1):  {
        'start': (0, 4),
        'shape': (4, 4),
        'data': future2,
        'location': [('1.1.1.1', 29384572)], },
      (0,0): {
        'start': (0,0),
        'shape': (4, 4),
        'data': future3,
        'location': [('1.1.1.2', 847236952)], },
  }
  'get': lambda x: distributed.get_client().gather(x)
}
```
### 2d-structure (64 elements), 1d-partition-grid, 4 partitions on 2 ranks, row-block-cyclic distribution, partitions are arrays allcocated on devices, MPI
```python
__partitioned__ = {
  'shape’: (8, 8),
  'partition_tiling’: (4, 1),
  'partitions’: {
      (0,0):  {
        'start': (0, 0),
        'shape': (2, 8),
        'data': ary0,   # this is for rank 0, for rank 1 it'd be None
        'location': [('1.1.1.1', 29384572, 'kDLOneAPI:0')], },
      (1,0):  {
        'start': (2, 0),
        'shape': (2, 8),
        'data': None,   # this is for rank 0, for rank 1 it'd be ary1
        'location': [('1.1.1.2', 575812655, 'kDLOneAPI:0')], },
      (2,0):  {
        'start': (4, 0),
        'shape': (2, 8),
        'data': ary2,   # this is for rank 0, for rank 1 it'd be None
        'location': [('1.1.1.1', 29384572, 'kDLOneAPI:1')], },
      (3,0):  {
        'start': (6, 0),
        'shape': (2, 8),
        'data': None,   # this is for rank 0, for rank 1 it'd be ary3
        'location': [('1.1.1.2', 575812655, 'kDLOneAPI:1')], },
  },
  'get': lambda x: x,
  'locals': [(0,0), (2,0)] # this is for rank 0, for rank 1 it'd be [(1,0), (3,0)]
}
```
