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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/layout.rst

Axis Layouts for CUDA
=====================

In this chapter, we will use axis layouts to place tensor elements across
CUDA threads and memory. A layout maps a logical index to physical
coordinates: for a local tensor, these can identify the lane that owns
a value and its storage position within that thread.

For the mapping model and its hardware motivation, see `Data Layout and
Its Notation
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_data_layout/index.html>`_
in Modern GPU Programming for MLSys.

Mapping elements to lanes
-------------------------

A ``TileLayout`` maps logical coordinates to named physical axes.
``S[shape : strides]`` describes the placement of distinct elements; a
stride written as ``value @ axis`` contributes to that axis. An untagged
stride uses the linear memory axis ``m``.

For a tile of shape ``(32, 4)``, we can assign one row to each lane of a warp
and keep its four values in that thread's local storage:

.. code-block:: python

    from tvm_tirx.tirx.layout import TileLayout, S, R
    from tvm_tirx.backend.cuda.axis import laneid, warpid

    fragment_layout = TileLayout(S[(32, 4) : (1 @ laneid, 1)])

The first coordinate selects a lane; the second selects a local element
position. Thus element ``(i, j)`` belongs to lane ``i`` at local position
``j``. The meaning of ``m`` follows the tensor's storage scope: here it is a
thread-local position, while for a global or shared tensor it is a memory
offset. Storage strides are measured in elements.

We can inspect a specific element with ``apply``:

.. code-block:: python

    position = fragment_layout.apply(3, 2, shape=(32, 4))
    print(position["laneid"], position["m"])

The output is ``3 2``: lane 3 holds the value at its local position 2.
The `interactive layout demo
<https://mlc.ai/modern-gpu-programming-for-mlsys/_static/tirx-layout-demo/index.html>`_
lets you select logical elements and inspect their physical coordinates.

Attaching a layout to a tensor
------------------------------

The ``layout`` argument attaches this placement to a tensor. The following
add-one kernel launches one warp and gives each lane four values:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def add_one_layout(A: T.Tensor((32, 4), "float32"), B: T.Tensor((32, 4), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        fragment = T.alloc_local((32, 4), "float32", layout=fragment_layout)
        lane_values = fragment.view(4, layout="default")
        for j in T.unroll(4):
            lane_values[j] = A[lane, j] + 1
        for j in T.unroll(4):
            B[lane, j] = lane_values[j]

``fragment`` describes the logical tile across all 32 lanes. Its layout
assigns four elements to each thread. ``fragment.view(4, layout="default")``
exposes those four local positions as ``lane_values``, without allocating
new storage. Each thread reads row ``lane`` from ``A``, adds one through
this local view, and writes the values to the same row of ``B``.

Replication and offsets
-----------------------

A layout can also describe copies of each logical value or shift its
physical placement:

.. code-block:: python

    replicated = TileLayout(S[(32, 4) : (1 @ laneid, 1)] + R[2 : 1 @ warpid])
    shifted = TileLayout(S[(32, 4) : (1 @ laneid, 1)] + 4)

``R[2 : 1 @ warpid]`` places a copy of each value in the corresponding lane
of warps 0 and 1. The fixed offset ``+ 4`` shifts local positions from
``0..3`` to ``4..7``. These objects describe placement; the kernel or tile
operation specifies what to compute and move.

``apply`` returns the base physical coordinate, including the fixed
offset. It does not enumerate replicas; those remain in ``layout.replica``.
For the full shard, replica, and offset model, see
`Replication and Offset
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_data_layout/index.html#replication-and-offset>`_.
The :class:`TileLayout API <tvm_tirx.tirx.TileLayout>` describes the
constructor and layout operations.

Swizzled shared memory
----------------------

Shared-memory access patterns can concentrate on the same memory banks.
A swizzle permutes addresses to spread suitable access patterns across
banks. ``ComposeLayout`` applies that permutation after the underlying
``TileLayout`` mapping.

For example, a 128-byte swizzle for an ``(8, 64)`` float16 tile is:

.. code-block:: python

    from tvm_tirx.tirx.layout import ComposeLayout

    linear = TileLayout(S[(8, 64) : (64, 1)])
    swizzled = ComposeLayout(
        per_element=3, swizzle_len=3, atom_len=3, tile_layout=linear
    )

These parameters preserve contiguous groups of eight float16 elements
(16 bytes) while permuting their positions within the tile. For logical
element ``(1, 0)``, the linear layout gives element offset 64, while the
swizzled layout gives offset 72.

Inside a device region, attach the layout when allocating shared memory:

.. code-block:: python

    shared = T.alloc_shared((8, 64), "float16", layout=swizzled, align=1024)

The eight 128-byte rows form one 1024-byte swizzle period, matching the
allocation's alignment. Instructions such as TMA and MMA must interpret
this storage with the same swizzle arrangement. The book's
`Swizzle Layout
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_data_layout/index.html#swizzle-layout>`_
section explains bank conflicts and the XOR mapping with interactive
figures. The :class:`ComposeLayout API <tvm_tirx.tirx.ComposeLayout>` covers
its parameters.

Instruction-specific layouts
----------------------------

Tensor Core instructions require particular operand layouts. The CUDA
backend provides :func:`mma_shared_layout
<tvm_tirx.backend.cuda.layout.mma_shared_layout>` for shared operands and
:func:`tmem_mma_operand_layout
<tvm_tirx.backend.cuda.layout.tmem_mma_operand_layout>` for TMEM operands.
The :func:`tmem_datapath_layout API
<tvm_tirx.backend.cuda.layout.tmem_datapath_layout>` lists the datapaths,
shapes, and sub-slab options. The :doc:`tile operation guides
</cuda/tile_operations>` show how to prepare these operands and synchronize
their use.

For the hardware data paths behind these choices, see
`The Evolution of Tensor Core Data Layouts
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_layout_generations/index.html>`_.
