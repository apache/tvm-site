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

Choosing a Tile Operation
-------------------------

Both operations repeat a computation over the values described by the
operands. They differ in how you specify that computation:

.. list-table::
   :header-rows: 1
   :widths: 15 40 45

   * - Operation
     - You supply
     - Each repetition
   * - ``apply``
     - A PTX instruction, such as ``T.ptx["ld.global.v4.b32"]`` or
       ``T.ptx["add.rn.f32"]``.
     - Executes that instruction on the values required by its operand form.
   * - ``ewise``
     - A scalar TIRx function, such as ``plus_one(value)``.
     - Inlines the function for one element, potentially producing several
       instructions.

Use ``apply`` when you want a particular instruction, including its
rounding or other variants. It supports arithmetic and matrix instructions
as well as data movement. Use ``ewise`` to express a computation as a
scalar function over register elements.

A Load, Compute, and Store Example
----------------------------------

Consider an add-one kernel with 64 threads, each handling twelve float32
values. It makes three tile calls: ``apply`` to load the values into
registers, ``ewise`` to add one to each value, and ``apply`` to store the
results. The load and store use the ``.v4.b32`` instruction forms, which
transfer four 32-bit words at a time.

The tensors have shape ``(2, 32, 3, 4)``: two warps, 32 lanes per
warp, and three groups of four values per thread. The register layout
maps the first two axes to ``warpid`` and ``laneid``. ``scope="cta"``
tells the tile calls to use those leading axes for the threads in the block:

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
        A: T.Tensor((2, 32, 3, 4), T.float32, align=16),
        B: T.Tensor((2, 32, 3, 4), T.float32, align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        registers = T.alloc_local((2, 32, 3, 4), T.float32, layout=register_layout)
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

Reading an Operand's Shape
--------------------------

For ``apply``, operand axes describe the execution **group**, the
instruction **repeats**, and the instruction's **atom**. The atom is the
set of values used by one instruction; here it is four consecutive
32-bit words. The same register view has a different interpretation
for ``ewise``, which applies a scalar function to every element:

.. list-table::
   :header-rows: 1
   :widths: 25 20 25 30

   * - Operation in this kernel
     - Group
     - Repeat
     - Atom
   * - ``apply`` with ``ld/st.global.v4.b32``
     - ``(2, 32)``: warp and lane
     - ``(3,)``: three instructions per thread
     - ``(4,)``: four words per instruction
   * - ``ewise(plus_one, ...)``
     - ``(2, 32)``: warp and lane
     - ``(3, 4)``: twelve scalar applications per thread
     - One scalar element, with no trailing atom axis

The chosen instruction fixes ``apply``'s atom shape. The operand view
specifies how often to repeat it, and the layout maps the coordinates to
threads and storage. ``ewise`` uses every coordinate after the group
prefix for a scalar function application.

Execution Scope and Scalar Functions
------------------------------------

The kernel's ``LaunchConfig`` launches 64 threads; ``scope="cta"``
describes how the tile operands map to those threads. It does not launch
additional threads or insert a barrier. At ``scope="thread"``, an operand
has no group prefix. Other scopes and warp-collective instructions use
the group shapes listed in the :ref:`coordinate reference
<cuda-tile-operand-coordinates>`.

``ewise``'s function parameters and return type must match the input and
destination elements. In ``ewise(plus_one, registers, registers, ...)``,
the first ``registers`` is the destination and the second is the input;
the result replaces each input value in place. A function can contain
multiple scalar operations, as the SiLU example in
:doc:`arithmetic` demonstrates.

Using apply for Arithmetic
--------------------------

The add-one computation can also use an explicitly selected PTX
instruction. In the kernel above, replace the ``ewise`` call with:

.. code-block:: python

    scalars = registers.view(2, 32, 3, 4, 1)
    T.cuda.tile.apply(
        T.ptx["add.rn.f32"], scalars, scalars, T.float32(1), scope="cta"
    )

``add.rn.f32`` uses a single-register atom, represented by the trailing
``1``. The ``(3, 4)`` repeat axes issue twelve additions per thread,
reusing the scalar constant ``1``. The view reuses the same register
storage. Here ``apply`` selects the addition instruction and its rounding
mode directly; ``ewise`` expresses the computation through ``plus_one``.

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
