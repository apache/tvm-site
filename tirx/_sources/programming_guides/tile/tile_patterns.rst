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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/cuda_tile_patterns.rst

Tile Operands and Execution
===========================

A CUDA tile operation repeats a chosen computation over a tensor region.
``T.cuda.tile.apply`` uses the PTX instruction you specify, while
``T.cuda.tile.ewise`` applies a scalar TIRx function to each element. The
operand views describe which values each thread handles and how many times
it repeats that computation.

Reading an Operand's Shape
--------------------------

Suppose each of 64 threads processes twelve float32 values. A
``ld.global.v4.b32`` instruction loads four 32-bit words at a time, so
each thread needs three loads. We can describe this work with shape
``(2, 32, 3, 4)``:

.. code-block:: text

    (2, 32, 3, 4)
     └───┘  │  └─ atom: four words used by one instruction
      group └──── repeat: three instructions per thread

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Part
     - Meaning in this example
   * - Group
     - ``(2, 32)`` selects one of two warps and a lane within that warp.
   * - Repeat
     - ``(3,)`` repeats the chosen instruction three times for that thread.
   * - Atom
     - ``(4,)`` selects the four words transferred by ``.v4.b32``.

The instruction fixes the atom shape. The leading axes describe its
repetitions and the threads that execute them. The register layout below
maps the first two coordinates to ``warpid`` and ``laneid``, leaving
``3 * 4 = 12`` local positions per thread.

A Load, Compute, and Store Example
----------------------------------

The kernel uses this layout to load the values, add one to each, and
store the results. ``scope="cta"`` tells the tile calls to use the
leading ``(2, 32)`` axes for the threads in the block:

.. code-block:: python

    from tvm_tirx.script import tirx as T
    from tvm_tirx.tirx.layout import S, TileLayout
    from tvm_tirx.backend.cuda.axis import laneid, warpid

    register_layout = TileLayout(S[(2, 32, 3, 4) : (1 @ warpid, 1 @ laneid, 4, 1)])

    @T.function(private=True)
    def plus_one(value: T.float32) -> T.float32:
        return value + T.float32(1)

    @T.function
    def add_one_tile(
        A: T.Tensor((2, 32, 3, 4), "float32", align=16),
        B: T.Tensor((2, 32, 3, 4), "float32", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        registers = T.alloc_local((2, 32, 3, 4), "float32", layout=register_layout)
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, A, scope="cta")
        T.cuda.tile.ewise(plus_one, registers, registers, scope="cta")
        T.cuda.tile.apply(T.ptx["st.global.v4.b32"], B, registers, scope="cta")

The register layout assigns each thread twelve storage positions: three
groups of four. The load and store each transfer four 32-bit words per
instruction, so each thread issues three loads and three stores. Between
them, ``ewise`` applies ``plus_one`` to all twelve scalar values.

We can compile and check the result:

.. code-block:: python

    import torch
    import tvm_tirx

    compiled = tvm_tirx.compile(add_one_tile, target="cuda")
    a = torch.arange(768, dtype=torch.float32, device="cuda").reshape(2, 32, 3, 4)
    b = torch.empty_like(a)
    compiled(a, b)
    torch.testing.assert_close(b, a + 1)

Execution Scope and Scalar Functions
------------------------------------

The kernel's ``LaunchConfig`` launches 64 threads; ``scope="cta"``
describes how the tile operands map to those threads. It does not launch
additional threads or insert a barrier. At ``scope="thread"``, an operand
has no group prefix. Other scopes and warp-collective instructions use
the group shapes listed in the :ref:`coordinate reference
<cuda-tile-operand-coordinates>`.

``ewise`` uses one scalar value per invocation, so it has no trailing
instruction atom. For the same register view, ``(2, 32)`` still selects a
thread, while ``(3, 4)`` now specifies twelve scalar invocations.

Preparing Operand Views
-----------------------

The load's source and destination expose the same group and repeat
coordinates. Within each atom, the four global words must be consecutive
and 16-byte aligned; the destination words must belong to the selected
thread. These requirements come from the chosen instruction, and lowering
checks that the input and output views meet them.

An existing tensor may need a ``view`` or ``rearrange`` to expose these axes,
or a ``sub`` to select a region. These operations reuse its storage. They
can change the coordinates used to access values, but a transfer between
threads or memory spaces still requires an explicit load, store, or other
communication instruction. See :doc:`/programming_guides/types_tensors_memory` for tensor views and
:doc:`/programming_guides/data_layout` for their physical placement.

The dtype describes storage elements. In the example, one element is one
32-bit word. An instruction that packs two float16 values into one 32-bit
register instead needs an appropriate packed container; ``apply`` does not
pack values automatically. The :ref:`instruction reference
<cuda-tile-instruction-contracts>` lists the required operand order,
representation, alignment, and optional parameters for each family.

Synchronization Belongs to the Caller
-------------------------------------

The example uses thread-private registers, so no thread waits for another.
If a thread instead reads shared memory written by other threads, the
kernel must synchronize those accesses. ``apply`` and ``ewise`` do not add
that synchronization.

Asynchronous instructions also require their own completion protocol.
``cp.async`` uses copy groups and waits; TMA loads use transaction barriers;
``tcgen05`` operations have their own commit and wait instructions. The
corresponding :doc:`instruction guides <index>` show these calls
alongside the data movement. Selecting an execution scope does not perform
thread election or complete an asynchronous transfer.
