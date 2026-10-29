..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

.. Adapted from Apache TVM:
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/data_types.rst
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/tensors.rst

Types, Tensors, and Memory
==========================

This guide looks at how CUDA kernels represent values and manage their storage.
A tensor's dtype describes its elements, and its scope selects the memory
space. We will extend the add-one computation with vector accesses, shared
storage, and thread-private values, then create views of existing data.

Element Types and Vector Accesses
---------------------------------

Tensor elements and intermediate values can be integers, floating-point
values, or booleans. Common TIRx dtypes have the following representations
in generated CUDA source:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - TIRx dtype
     - Generated CUDA representation
   * - ``float32``
     - ``float``
   * - ``float16``
     - ``half``
   * - ``bfloat16``
     - ``nv_bfloat16``
   * - ``int32``
     - ``int``
   * - ``uint8``
     - ``uchar`` (an alias for ``unsigned char``)
   * - ``bool``
     - ``bool``
   * - ``float32x4``
     - ``float4``

A ``float32x4`` value contains four float32 components, called lanes.
CUDA C++ names this type ``float4``: each ``float`` is 32 bits, and ``4``
is the number of components.

We can extend the add-one kernel with vector accesses so each thread
reads and updates four elements as a group:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def add_one_vector(
        A: T.Tensor((256,), "float32", align=16),
        B: T.Tensor((256,), "float32", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        vector_value = A[T.Ramp(tx * 4, 1, 4)]
        B[T.Ramp(tx * 4, 1, 4)] = vector_value + 1

``T.Ramp(tx * 4, 1, 4)`` selects the four consecutive indices ``4*tx``,
``4*tx + 1``, ``4*tx + 2``, and ``4*tx + 3``. Its arguments specify the
starting index, stride, and lane count. Because ``A`` has float32 elements,
the read at these four positions produces ``vector_value`` with inferred
dtype ``float32x4``.

``vector_value + 1`` adds one to each lane. The store uses the same indices
to write the four results to the corresponding float32 elements in ``B``.

With 64 threads and four elements per thread, the launch covers all 256
elements. The ``align=16`` annotations declare 16-byte-aligned pointers;
each four-float group also starts at a 16-byte boundary, allowing aligned
vector loads and stores.

Declaring Tensors
-----------------

A parameter such as ``A: T.Tensor((128,), "float32")`` describes a tensor
passed by the caller. Inside a kernel, ``T.alloc_shared`` and
``T.alloc_local`` allocate scratch tensors. Both are conveniences for
``T.alloc_tensor``, whose tensor type specifies the shape, dtype, and scope.
The ``scope`` selects the memory space:

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Scope
     - Typical interface
     - Storage
   * - ``"global"``
     - Tensor parameters (default)
     - Device global memory
   * - ``"shared"``
     - ``T.alloc_shared``
     - Shared by threads in a block; size fixed at compile time
   * - ``"local"``
     - ``T.alloc_local``
     - Private to each thread
   * - ``"tmem"``
     - ``T.cuda.decl_tmem``
     - Blackwell tensor memory, reserved separately

When storage already exists, ``T.decl_tensor`` creates a tensor view over
its pointer. For example, these statements inside a device region allocate
one shared tensor and declare another view of the same elements:

.. code-block:: python

    shared = T.alloc_shared((128,), "float32")
    shared_view = T.decl_tensor(
        shared.data, ty_args=[T.Tensor((128,), "float32", scope="shared")]
    )

``shared.data`` is the base pointer for ``shared_view``. Both names refer
to the same storage, so a write through ``shared_view`` is also visible through
``shared``. The declaration does not allocate or copy data.

The tensor's ``layout`` determines how logical indices map to storage;
``align`` specifies data-pointer alignment in bytes. The
:doc:`Data Layout </programming_guides/data_layout>` guide covers explicit layouts, and the
:class:`TensorType API <tvm_tirx.tirx.TensorType>` lists the full parameters.

Shared Memory
-------------

Static Shared Memory
~~~~~~~~~~~~~~~~~~~~

``T.alloc_shared`` creates a shared tensor with a size known at compile
time. The following kernel lets each thread read the value staged by its
neighbor, wrapping the last thread back to the first:

.. code-block:: python

    @T.function
    def shared_shift(A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        shared = T.alloc_shared((128,), "float32")
        shared[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = shared[(tx + 1) % 128]

Each thread first writes one input element to ``shared``. The
``T.cuda.cta_sync()`` barrier waits until all threads in the block have
reached it, making those writes visible before the neighboring reads.
For example, thread 0 writes ``A[0]`` but reads the value that thread 1
stored from ``A[1]``.

Dynamic Shared Memory
~~~~~~~~~~~~~~~~~~~~~

With dynamic shared memory, the kernel launch specifies how many bytes to
reserve for each thread block. All threads in the block share this region.
Tensors with scope ``"shared.dyn"`` use storage within it, and multiple
tensors can occupy different byte ranges of the same region.

The ``SMEMPool`` helper manages these ranges and their total size while
TVMScript builds the kernel. The following version of ``shared_shift``
uses it for the shared tensor:

.. code-block:: python

    from tvm_tirx.backend.cuda.lang import SMEMPool

    @T.function
    def shared_shift_dynamic(
        A: T.Tensor((128,), "float32"), B: T.Tensor((128,), "float32")
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tx = T.cuda.thread_idx("x")
        pool = SMEMPool()
        shared = pool.alloc((128,), "float32")
        pool.commit()
        shared[tx] = A[tx]
        T.cuda.cta_sync()
        B[tx] = shared[(tx + 1) % 128]

The pool needs ``128 * 4 = 512`` bytes per block. ``pool.alloc`` creates
``shared`` as a tensor view into that region, and ``pool.commit()`` records
the byte count for the launcher. The tensor shape is fixed here; its
storage is requested through the launch configuration. The writes,
barrier, and neighbor reads then use this shared tensor.

Further ``pool.alloc`` calls place additional tensors within the same
region. ``pool.commit()`` should follow all allocations. The
:class:`SMEMPool API <tvm_tirx.backend.cuda.lang.SMEMPool>` covers alignment
and storage reuse.

Per-Thread Local Storage
------------------------

``T.alloc_local`` creates storage private to each thread. For example,
each of 64 threads can keep four input elements in a local array before
adding one and writing them back:

.. code-block:: python

    @T.function
    def add_one_local(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        values = T.alloc_local((4,), "float32")
        for k in T.unroll(4):
            values[k] = A[tx * 4 + k]
        for k in T.unroll(4):
            B[tx * 4 + k] = values[k] + 1

Every thread has its own ``values`` array. The CUDA compiler may place its
elements in registers when the accesses permit; local scope alone does not
guarantee register storage.

Mutable Scalars and ``let``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In TVMScript, an ordinary assignment such as ``base = tx * 4`` creates an
immutable binding whose type is inferred from the expression. With an
explicit type annotation, its form determines whether the local scalar
can be updated:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Declaration
     - Meaning
   * - ``total: T.float32 = 0.0``
     - Mutable float32 scalar, initialized to zero
   * - ``base: T.let[T.int32] = tx * 4``
     - Immutable value, with an explicit int32 type annotation

A running sum needs the mutable form to retain updates between loop
iterations. The following kernel sums four consecutive inputs per thread
and writes one output for each sum:

.. code-block:: python

    @T.function
    def sum_four(A: T.Tensor((256,), "float32"), B: T.Tensor((64,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tx = T.cuda.thread_idx("x")
        base: T.let[T.int32] = tx * 4
        total: T.float32 = 0.0
        for k in range(4):
            total += A[base + k]
        B[tx] = total

``total`` retains the result of each addition for the next iteration.
If thread 0 reads ``1, 2, 3, 4``, the running total is
``0 → 1 → 3 → 6 → 10``, and ``B[0]`` receives 10.

``base`` holds the group's starting index throughout the loop; the
``T.let[T.int32]`` annotation states its type explicitly.

TIRx represents ``total`` as a one-element local tensor, so each thread
has its own accumulator. The
:doc:`TIRx script API </reference/python/tirx.script>` also provides
``local_scalar``, ``shared_scalar``, and ``alloc_scalar`` for explicit
scalar allocation.

Tensor Views and Pointers
-------------------------

Tensor variables provide methods for accessing addresses and creating
views of existing storage. These operations are useful when an intrinsic
needs a pointer or a computation expects a different arrangement of the
same elements:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Expression
     - Meaning
   * - ``A.data``
     - The base data pointer
   * - ``A.ptr_to([i, j])``
     - A pointer to element ``A[i, j]``, with its element type and storage scope
   * - ``A.sub[i, :]``
     - A tensor region selecting row ``i``, for use as an instruction operand
   * - ``A.view(*shape)``
     - A view with a different shape
   * - ``A.permute(*dims)``
     - A view with the axes reordered
   * - ``A.rearrange(pattern, **sizes)``
     - A view that splits, reorders, or merges axes using a named pattern
   * - ``A.view("uint32")``
     - A view of the same bytes as 32-bit unsigned words

For a contiguous one-dimensional tensor ``A`` with 256 elements, we can
view the data as a matrix and transpose that view:

.. code-block:: python

    matrix = A.view(64, 4)
    transposed = matrix.permute(1, 0)

``matrix`` has shape ``(64, 4)`` and ``transposed`` has shape ``(4, 64)``.
``matrix[i, j]`` and ``transposed[j, i]`` both refer to ``A[i * 4 + j]``.

A tile instruction may expect separate axes for warps and lanes. We can
split the 64 rows into two groups of 32 with ``rearrange``:

.. code-block:: python

    grouped = matrix.rearrange("(warp lane) value -> warp lane value", warp=2, lane=32)

The left side describes the current shape: ``(warp lane)`` is one axis
of length ``2 * 32``, followed by the four-value axis. The right side
exposes all three axes, giving ``grouped`` shape ``(2, 32, 4)``.
``grouped[w, l, j]`` refers to ``matrix[32*w + l, j]``. The pattern names
the coordinates; the tensor's layout determines their physical placement.

``sub`` selects a region to pass to an operation. For example,
``matrix.sub[i, :]`` selects its four-element row ``i``. A tile load can
then read that region into registers, as shown in :doc:`/programming_guides/tile/data_movement`.

A dtype view changes the storage units used to access the bytes. Inside
a device region, eight float16 elements can be viewed as four uint32 words:

.. code-block:: python

    half_values = T.alloc_local((8,), "float16")
    packed_words = half_values.view("uint32")

``packed_words`` has shape ``(4,)``. Each word contains the bits of two
adjacent float16 elements. This is the representation used by instructions
that accept packed half values; a numeric conversion uses ``cvt`` instead,
as shown in :doc:`/programming_guides/tile/arithmetic`.

All these views and regions reuse the original storage. The
:class:`TensorType API <tvm_tirx.tirx.TensorType>` covers the available
methods and options, while :doc:`/programming_guides/data_layout` explains explicit storage
mappings, including layouts distributed across threads.

Tensor Memory
-------------

Blackwell tensor memory uses ``scope="tmem"`` and must be reserved and
released explicitly. ``T.cuda.decl_tmem`` declares a tensor at an allocated
tensor-memory address; access uses ``tcgen05`` instructions. The
:doc:`Blackwell Tensor Cores guide </programming_guides/tile/blackwell_tensor_cores>` shows the
allocation, synchronization, transfer, and release steps together. The
:func:`decl_tmem API <tvm_tirx.backend.cuda.script.decl_tmem>` describes its
arguments.
