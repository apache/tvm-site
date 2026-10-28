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

.. Adapted from the Apache TVM CUDA tile instruction guides.

.. _cuda-tile-instruction-contracts:

CUDA tile instruction reference
===============================

This reference specifies the operands, parameters, storage requirements, and
lowering behavior of CUDA tile instructions. For a worked introduction to
operand views, see :doc:`/cuda/tile_patterns`. The :doc:`instruction guides
</cuda/tile_operations>` provide complete kernels and completion protocols.

.. contents:: Instruction families
   :local:
   :depth: 1

.. _cuda-tile-operand-coordinates:

Operand coordinates and storage
-------------------------------

Tensor operands use ``(*group, *repeat, *atom)`` coordinates. The selected
instruction determines its atom suffix. Repeats are positive static
extents, expanded in row-major order. Each repeat emits one selected native
instruction; address preparation may add instructions such as descriptor
arithmetic or ``mapa``. Counts refer to tile lowering before later compiler
optimizations. ``ewise`` instead evaluates one scalar function per element.

Execution groups
~~~~~~~~~~~~~~~~

For a thread instruction, ``scope="thread"`` has no group prefix;
``"warp"`` uses ``(lane,)``; ``"warpgroup"`` uses
``(warp_in_group, lane)``; and ``"cta"`` uses ``(warp_in_cta, lane)``.
Extents come from the active execution scope, including explicitly guarded
subranges. The register layout must assign each value to that thread.

A warp-collective instruction consumes the lane axis inside its atom.
Its group prefix is empty at warp scope and has one warp axis at
warpgroup or CTA scope. All 32 lanes must participate.

Cluster scope requires an explicit launch cluster. Its CTA coordinates
precede the warp/lane groups in z/y/x order, omitting undeclared leading
axes. A one-dimensional cluster therefore has ``(cta_x, warp, lane)``
groups for a thread instruction. Local tensor ownership uses
``cbx/cby/cbz`` for these CTA coordinates. Execution scope does not elect
an issuing thread.

Regions and units
~~~~~~~~~~~~~~~~~

A tensor argument denotes its full region. For a region beginning at
``origin``, physical coordinates come from applying the tensor layout to
``origin + group/repeat/atom`` coordinates. Region starts participate in
bounds and alignment checks. Starts may be symbolic when those properties
can be proved; extents must be positive static integers. Group and repeat
shapes agree across operands except for the documented MMA M/N/K sharing.

Storage strides and offsets count elements of the declared dtype.
Descriptor strides and alignment count bytes. A b32 container is one
storage element even when it holds two b16 values. Register offsets name
storage positions; they do not specify final machine register numbers.
An allocation must cover the layout's physical span, including gaps and
any composed swizzle.

Views reuse the tensor's storage. Lowering checks the supplied physical
image; it does not select a different opcode, width, matrix count,
transpose mode, packing, or layout permutation to make it valid.

Parameters and completion
~~~~~~~~~~~~~~~~~~~~~~~~~

``apply(instruction, *operands, scope="thread", **parameters)`` accepts the
family-specific keywords below. Opcode qualifiers belong in
``T.ptx[opcode]``. Static parameters are compile-time values; runtime
parameters are ordinary IR expressions. Unknown instruction keywords are
rejected. The generic IR construction keywords ``attrs``, ``ty``,
``ty_args``, and ``loc`` are separate from hardware parameters.

Predication is available only where listed. ``pred`` gates instruction
execution. For supported register results, ``preserve_dst=True`` retains
an initialized destination when the predicate is false; otherwise inactive
results are undefined. A skipped asynchronous issue performs no completion
accounting. In particular, ``cp.async`` zero fill still issues an operation,
whereas ``pred=False`` suppresses it.

The caller supplies packing, shared-memory staging, synchronization,
barrier initialization and accounting, thread election, and pipeline
scheduling. The individual instruction's completion rules still apply.

.. _cuda-tile-contract-registers:

Register instructions
---------------------

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :doc:`/cuda/tile_patterns`. It does not elect an issuing thread.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional.
   * - ``preserve_dst``
     - static bool; default False
     - For a predicated register result, retain the initialized destination when pred is false; otherwise the inactive result is undefined.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

Each operand uses the scalar PTX register type or its equal-width bit container. A
packed f16x2/bf16x2 value occupies one b32 container; f32x2 occupies one b64 container.
The final tensor axis counts native registers, not the packed scalar elements. Rounding,
saturation, signedness and packing are selected in the opcode.

Legality conditions
~~~~~~~~~~~~~~~~~~~

All tensor operands are local registers owned by the issuing thread. Their group/repeat
shapes agree; the final extent equals that operand's native register count (one for a
scalar operand). Scalar expressions are allowed for single-register read inputs and
broadcast across repeats. Destination slots must not overlap. Register storage strides
are in container elements; they need not be one. The selected PTX form validates operand
types.

Code generation
~~~~~~~~~~~~~~~

For each group and row-major repeat coordinate, bind each tensor's final register axis
in increasing order and emit one selected instruction. A scalar input is reused. No
packing, memory staging or synchronization is inserted. The number of expanded
instructions per issuing thread is the product of repeat extents; downstream compiler
optimizations may simplify unused or constant results.

.. _cuda-tile-contract-mov:

Register moves
~~~~~~~~~~~~~~

Operand order: ``dst, src; or the explicit packing/unpacking register groups``.

``mov`` copies register bits or assembles/disassembles a native register group. It
performs no numeric conversion and cannot change which thread owns a value. Use
:ref:`numeric conversion <cuda-tile-contract-cvt>` for conversion and :doc:`/cuda/tile_ops/data_movement` for memory.

The scalar form ends both tensor operands in ``(1,)``. All earlier axes after the
execution-group prefix are repeats. A scalar source expression can broadcast, for
example to fill a register tile. Vector-group mov forms instead use the group length
declared by that PTX form; each input/output word remains an explicit operand slot.

For a predicated output that must keep its old value, initialize it and
pass ``preserve_dst=True``. A suppressed move does not zero-fill its
output. A dtype view can expose existing packed bits without a move;
assembling separately stored values requires an explicit packing form
with matching register groups and carrier dtypes.

.. _cuda-tile-contract-add-sub-mul:

Addition, subtraction, and multiplication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operand order: ``dst, lhs, rhs``.

The opcode chooses the numeric type, rounding, saturation and packing mode. The operand
order is ``dst, lhs, rhs``. For a scalar instruction, each tensor ends in a singleton
register-group axis; for example ``(8,1)`` at thread scope means eight scalar
instructions. A read-only scalar expression may replace an input tensor and broadcasts
over all eight repeats.

Native packed arithmetic
^^^^^^^^^^^^^^^^^^^^^^^^

Packed arithmetic still has one register **carrier** per operand:

.. list-table::
   :header-rows: 1

   * - Example opcode
     - Carrier dtype
     - Values per carrier
   * - ``add.rn.f32``
     - float32
     - One float32
   * - ``add.rn.f16x2``
     - uint32 (raw bits)
     - Two float16
   * - ``mul.rn.bf16x2``
     - uint32 (raw bits)
     - Two bfloat16
   * - ``add.rn.f32x2``
     - uint64 (raw bits)
     - Two float32

Not every qualifier is valid for every type or architecture. PTX form validation checks
the spelling and the compiler checks the target. Packing is a caller decision, not the
consequence of an even tensor extent.

Subtraction and multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sub`` preserves the written operand order, including a scalar on the left. ``mul``
uses the selected floating-point form or the integer ``lo``/``hi``/ ``wide`` form and
its declared result type. Different result widths require the matching destination
dtype, not a silent reinterpretation.

Tensor operands share repeat extents and must have the correct thread owner. An input
broadcast is a scalar or an explicit zero-stride view. Output storage cannot overlap
across repeats. ``pred`` and ``preserve_dst`` follow the register instruction contract
described in :ref:`cuda-tile-contract-registers`.

.. _cuda-tile-contract-div-rcp-sqrt:

Division, reciprocal, and square root
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operand order: ``dst, numerator, denominator; or dst, src``.

Select accuracy, rounding and flush-to-zero behavior in the PTX opcode. These choices
change arithmetic semantics; the tile layer does not replace a division with
multiplication by an approximate reciprocal.

.. list-table::
   :header-rows: 1

   * - Example
     - Operand order
     - Atom
   * - ``div.rn.f32``
     - ``dst, numerator, denominator``
     - One float32 per operand
   * - ``rcp.approx.ftz.f32``
     - ``dst, src``
     - One float32 per operand
   * - ``sqrt.rn.f32``
     - ``dst, src``
     - One float32 per operand

Tensor operands end in ``(1,)``, preceded by execution-group and repeat axes. Scalar
read-only inputs broadcast over those repeats. A constant numerator is allowed; its
position is not exchanged with the denominator.

.. _cuda-tile-contract-min-max:

Minimum and maximum
~~~~~~~~~~~~~~~~~~~

Operand order: ``dst, a, b[, c]``.

``min`` and ``max`` apply the native instruction independently at every register repeat.
They do not perform a warp, block, or tensor reduction. Choose NaN, sign,
absolute-value, packed-type and other supported semantics through the instruction
spelling.

Each scalar tensor operand ends in ``(1,)``. Binary and three-input forms are
distinguished by the explicitly supplied operands, not by a layout search. ``max.f32``
with ``dst,a,b,c`` uses the registered three-input form; ``dst,a,b`` uses its binary
form. The selected target must support that form. Packed f16x2/bf16x2 variants use b32
carriers as in :ref:`cuda-tile-contract-add-sub-mul`.

.. _cuda-tile-contract-fma:

Fused multiply-add
~~~~~~~~~~~~~~~~~~

Operand order: ``dst, a, b, c``.

``T.cuda.tile.apply(T.ptx["fma.rn.f32"], dst, a, b, c)`` computes the selected fused
multiply-add for each repeat. ``a*b+c`` is one native FMA atom, including its explicit
rounding qualifier. It is not a request for two independently rounded multiply/add
instructions.

Scalar tensor operands have suffix ``(1,)``. Read-only scalar operands broadcast over
repeats. Packed f16x2/bf16x2 forms use uint32 carriers; packed f32x2 uses uint64. A
packed constant must encode both values. The opcode and storage representation stay
explicit even if a contiguous layout would permit several other forms.

.. _cuda-tile-contract-cvt:

Numeric conversion
~~~~~~~~~~~~~~~~~~

Operand order: ``dst, src[, src1, ...]``.

``cvt`` performs a numeric conversion selected by the opcode. A dtype view only
reinterprets bits; it is used here to expose the native carrier layout, not to perform
the conversion itself. Destination and source widths, operand group lengths, rounding,
saturation and packing come from the PTX form.

Scalar conversion uses tensor suffix ``(1,)`` on both sides. Packed ``cvt.rn.f16x2.f32``
has three operands: one b32 destination, a float32 source for its **high** halfword, and
a float32 source for its **low** halfword. All three have suffix ``(1,)``. The pair is
not an inferred vectorization of a scalar cast.

For ``cvt.rn.f16.f32`` and ``cvt.f32.f16``, expose the half-precision
operand as a uint16 bit container. The reverse of a packed
``cvt.rn.f16x2.f32`` uses two scalar ``cvt.f32.f16`` instructions, one per
halfword; there is no ``cvt.f32x2.f16x2`` form in this API. A uint16 view of
the packed word exposes those halfwords without converting them.

.. _cuda-tile-contract-ex2-lg2:

Base-two exponential and logarithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operand order: ``dst, src``.

These PTX instructions calculate base-two exponential and logarithm. They are not
natural exp/log. Select the explicit approximation and subnormal policy, for example
``ex2.approx.ftz.f32`` or ``lg2.approx.ftz.f32``. The tile layer forwards this spelling
without selecting a replacement.

Both tensor operands use local registers and end in ``(1,)``; each preceding repeat
selects another scalar input/output. Output repeats cannot alias. A scalar read-only
input broadcasts over all repeats.

.. _cuda-tile-contract-abs-neg:

Absolute value and negation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Operand order: ``dst, src``.

``abs`` and ``neg`` operate independently on local register values. Their opcode selects
the numeric type and applicable qualifiers. Operand order is ``dst,src``; each scalar
tensor operand ends in ``(1,)`` after the group and repeat axes. An ordinary dtype view
does not change the numeric type of the requested operation.

NaN, signed-zero, and integer-boundary behavior follow the selected PTX
form. Register views may select separated slots but cannot change which
thread owns a value.

.. _cuda-tile-contract-ld-st:

Memory loads and stores
-----------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_registers, src_memory for ld; dst_memory, src_registers for st``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.
   * - ``cache_policy``
     - runtime uint64; required with .L2::cache_hint
     - Native cache-policy operand; absent on forms without that operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional.
   * - ``preserve_dst``
     - static bool; default False
     - For a predicated register result, retain the initialized destination when pred is false; otherwise the inactive result is undefined.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

Both operands use scalar storage elements with the opcode's bit width. Vector length V
is taken from .v2/.v4 or another supported vector qualifier; scalar forms use V=1.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

Use ``T.cuda.tile.apply`` to repeat a chosen memory instruction. The operand order is
always destination first: a load takes ``registers, memory`` and a store takes ``memory,
registers``. For example::

    T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], R, A)
    T.cuda.tile.apply(T.ptx["st.shared.v4.b32"], S, R)

Both calls transfer four 32-bit words per repeat. Global-to-shared copying through
registers consists of these two explicit calls and caller-allocated register storage. To
issue a direct asynchronous transfer, use :doc:`/cuda/tile_ops/async_copies`.

Contract and instruction selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Operands have shape ``(*group,*repeat,word)``. The final extent is the vector length in
the opcode, or one for a scalar instruction. Both operands use scalar elements with the
opcode's storage bit width. Each memory atom must be consecutive and aligned to the
whole vector's byte size. Its start can vary across repeats. Local register slots can be
separated or permuted, but each must belong to the issuing thread; loaded destinations
cannot overlap.

The supported address-space spellings are ``global``, ``shared``, ``shared::cta``, and
``local``. The pointer side of ``ld.local``/``st.local`` is addressable local memory,
distinct from the register operands. Legal types, vector lengths, and qualifiers come
from the selected PTX form.

Cache operators such as ``.nc`` and ``.cg`` and prefetch/eviction qualifiers are part of
the opcode. A form with a cache-policy operand takes ``cache_policy=...``. ``pred``
gates execution. A predicated load can use ``preserve_dst=True`` to keep an already
initialized destination when the predicate is false. Lowering does not reduce the width
when alignment fails.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Both regions end in (V,). Memory elements within an atom are consecutive, and the atom
start is aligned to V*element_bytes. The written address space must match storage. Local
register slots belong to the issuing thread and loaded destinations do not overlap. A
region start participates in every address and alignment check.

Code generation
~~~~~~~~~~~~~~~

For each repeat, take the memory pointer at word zero and bind the V register slots in
word order. Emit one selected ld/st, optionally with cache_policy and pred. Repeat axes
unroll; group axes bind hardware IDs. Ordinary ld/st performs no asynchronous completion
protocol.

For a complete kernel, see :doc:`/cuda/tile_ops/data_movement`.

.. _cuda-tile-contract-ldmatrix-stmatrix:

Matrix loads and stores
-----------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_registers, src_shared for ldmatrix; dst_shared, src_registers for
stmatrix``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

Both operands use scalar b32 containers (uint32/int32/float32 as bits). Each packs two
b16 matrix elements. x1/x2/x4 selects N=1/2/4 b32 outputs per lane. Two halves of one
output are fixed packing; a free register stride separates distinct b32 outputs.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

The tile patterns support ``m8n8.b16`` with ``.x1``, ``.x2``, or ``.x4`` and optional
``.trans``. Select the exact form in the opcode::

    T.cuda.tile.apply(
        T.ptx["ldmatrix.sync.aligned.m8n8.x2.shared.b16"],
        registers, shared_words, scope="warp",
    )

``stmatrix`` reverses the operand order: ``shared_words, registers``. The implementation
repeats the specified warp instruction, without picking a matrix count or transpose mode
from the layout.

Address providers and result owners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Both regions end in ``(matrix,row,word)=(count,8,4)`` and use scalar 32-bit containers.
A container holds two adjacent 16-bit values. Use a dtype view to expose packed
half/bfloat16 storage as b32 words; this is not a conversion.

* Each shared row has four consecutive words and a 16-byte-aligned start.
  Different rows may have gaps and may be interleaved with other matrices.
* Lane ``8*matrix+row`` provides that row's shared address. For x1/x2,
  the remaining address providers repeat valid row addresses.
* Register word ``(matrix,row,word)`` belongs to lane ``4*row+word``.
  Each lane receives ``count`` words. Their register offsets need not be
  adjacent, and need not have the same order as another tensor declaration.

Thus, the lane supplying a shared address generally differs from the lanes receiving the
values from that row. The contract checks both maps. All 32 lanes must participate,
including those whose addresses are duplicates.

Parameterized layouts
^^^^^^^^^^^^^^^^^^^^^

For one warp and one repeat axis, all storage strides below are in b32 words::

    coordinates: (repeat, matrix, row, word)
    smem: (K,N,8,4):(B,   M,   S,        1)
    reg:  (K,N,8,4):(P@m, R@m, 4@laneid, 1@laneid)

B/M/S separate repeats, matrices and shared rows; P/R separate repeats and outputs in
each thread's register storage. They are read from the layout, not supplied as keywords.
S=16 means 64-byte row pitch; R=2 means every second b32 output slot. These slots do not
specify machine register numbers. The more general row-address contract is::

    byte_address(t,i,j,w) = row_address(t,i,j) + 4*w

Each row_address must be 16-byte aligned; it need not have constant stride across j. A
nonzero region start is included before applying the layout. For .trans the b16 image
satisfies::

    register_image[i,j,2*w+h] = shared_matrix[i,2*w+h,j]

Without .trans, the right-hand side is shared_matrix[i,j,2*w+h].

Legality conditions
~~~~~~~~~~~~~~~~~~~

Both regions end in (N,8,4), after matching group/repeat axes. Each shared row has four
consecutive words and a 16-byte-aligned start. Different rows have independent starts.
Register word (i,j,w) belongs to lane 4*j+w; loaded slots are distinct. All 32 lanes
participate; scope="warp" has no group prefix, and wider scopes add warp groups. No
opcode-specific keyword, including pred, is accepted.

Code generation
~~~~~~~~~~~~~~~

For each repeat t, every lane l executes the following expansion::

    address = &smem[t, (l // 8) % N, l % 8, 0]
    registers = [reg[t, i, l // 4, l % 4] for i in range(N)]
    emit_selected_instruction(registers, address)  # ld; reversed for st

The first 8*N lanes supply the N matrices' row addresses; unused address providers
repeat valid addresses. Each repeat emits one warp-wide .xN instruction. Initialization
and synchronization are caller operations. .trans changes the b16 semantic image, as
specified in the operand specification.

Transpose and stores
~~~~~~~~~~~~~~~~~~~~

``.trans`` transposes each 8x8 matrix of **16-bit elements**, not a matrix of b32 words.
Register image element ``(row,2*word+half)`` comes from shared element
``(2*word+half,row)``. The register word's lane owner remains ``4*row+word``. Reorganize
operands with this image in mind; toggling ``.trans`` is a semantic change, not a layout
optimization hint.

``stmatrix`` uses the corresponding register image as input and each provider's shared
row address as output. Its supported architecture is checked by the PTX compiler.
Synchronize before a load consumes another thread's shared stores, and after a store
before other threads consume or reuse that shared storage. No fence or barrier is
inserted by the pattern.

An insufficiently aligned row, wrong register owner, overlapping loaded registers, or a
partially active warp is rejected. Gaps **between** rows or register slots are legal;
gaps **within** a shared 16-byte row are not.

For a complete kernel, see :doc:`/cuda/tile_ops/warp_matrix`.

.. _cuda-tile-contract-cp-async:

Asynchronous global-to-shared copies
------------------------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_shared, src_global``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.
   * - ``chunk_bytes``
     - static int; required
     - Bytes per atom: 4, 8 or 16, subject to the chosen PTX form (.cg uses 16).
   * - ``src_size``
     - native byte-count operand; default None
     - Copy this prefix and zero-fill the remainder; mutually exclusive with ignore_src. The selected PTX form determines immediate restrictions.
   * - ``ignore_src``
     - runtime bool; default None
     - True zero-fills the destination; mutually exclusive with src_size.
   * - ``cache_policy``
     - runtime uint64; required with .L2::cache_hint
     - Native cache-policy operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

Each operand ends in chunk_bytes*8/storage_bits scalar elements. Source and destination
may use different equal-total-size storage representations; this is a byte copy.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

``cp.async`` copies a short contiguous chunk from global to shared memory without
passing the payload through registers. Each issuing thread owns its async-copy group.
Select the cache/prefetch qualifiers in the opcode and the copy width with
``chunk_bytes``::

    T.cuda.tile.apply(
        T.ptx["cp.async.cg.shared.global"], dst, src,
        chunk_bytes=16, scope="cta",
    )

Operand contract
^^^^^^^^^^^^^^^^

The operand order is ``dst_shared, src_global``. At thread scope each region has shape
``(*repeat,element)``. The final axis occupies exactly ``chunk_bytes`` bytes and is
contiguous and naturally aligned on both sides. For float16 with ``chunk_bytes=16``,
that axis has eight elements. An explicit b32 view of the same bytes has four elements.

.. list-table::
   :header-rows: 1

   * - Field
     - Meaning
   * - ``.ca`` / ``.cg``
     - Select in the opcode. ``.ca`` supports 4/8/16 bytes; ``.cg`` requires 16.
   * - ``chunk_bytes``
     - Required constant 4, 8 or 16; no width is inferred from the layout.
   * - ``.L2::64B`` / ``.L2::128B`` / ``.L2::256B``
     - Optional prefetch qualifier in the opcode, not a larger copy size.
   * - ``cache_policy``
     - Runtime operand for an opcode containing ``.L2::cache_hint``.
   * - ``scope``
     - Thread by default. CTA scope adds explicit ``(warp,lane)`` group axes.

The same byte count is copied when source and destination use different storage element
widths, provided each side's atom contains that many whole contiguous bytes. This is a
bit copy, not a conversion.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Source is global; destination is shared/shared.dyn. Each atom is contiguous and aligned
to chunk_bytes. The chunk contains whole elements on both sides. Group/repeat shapes
agree. Source bounds must describe a valid region even when zero-fill operands are
supplied.

Code generation
~~~~~~~~~~~~~~~

At each repeat, emit the selected cp.async with destination/source atom pointers,
chunk_bytes and the explicit optional zero-fill/cache operands. No commit_group,
wait_group, barrier or thread election is inserted. pred suppresses the issue;
ignore_src/src_size instead issue a copy with zero fill.

Predication is different from zero fill
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pred=False`` suppresses the instruction; it does not write zeros to shared memory. To
copy a partial chunk and zero the remaining bytes, supply ``src_size`` in the range
``0..chunk_bytes``. To zero the entire chunk, use ``src_size=0`` or ``ignore_src=True``.
``src_size`` and ``ignore_src`` are alternative operand forms and cannot both be
supplied.

For example, inside a kernel where ``tid`` selects one 16-byte row::

    T.cuda.tile.apply(
        T.ptx["cp.async.cg.shared.global.L2::128B"],
        shared.sub[tid, :], A.sub[tid, :], chunk_bytes=16,
        src_size=T.if_then_else(tid < valid_rows, T.uint32(16), T.uint32(0)),
    )

Here both tensors have uint8 rows of length 16. The final allocated row is still part of
the declared source view: a zero-fill operand does not relax the tile pattern's
bounds/shape proof.

Completion and errors
~~~~~~~~~~~~~~~~~~~~~

Each thread commits and waits for its own issues. Use a CTA barrier before other threads
read the resulting shared tile. These calls are not inserted by ``apply``. A
shared-to-global request, a noncontiguous chunk, an unaligned base, or a missing group
axis fails its contract; lowering never substitutes a synchronous copy or a narrower
instruction.

For a complete kernel, see :doc:`/cuda/tile_ops/async_copies`.

.. _cuda-tile-contract-cp-async-bulk:

Asynchronous cluster shared copies
----------------------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_shared, src_shared``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.
   * - ``chunk_bytes``
     - static positive int; required
     - Bytes per atom; divisible by 16.
   * - ``mbar``
     - runtime pointer; required
     - Local shared address of the completion barrier in the destination CTA image.
   * - ``remote_cta_id``
     - runtime integer; required
     - Destination CTA rank within the cluster.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

Each operand ends in chunk_bytes*8/storage_bits scalar elements. Source, destination and
mbar denote local shared addresses before remote address mapping.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

This pattern issues
``cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes``. It copies
one contiguous shared-memory chunk to another CTA in the launch cluster and completes
transaction bytes on a barrier in the destination CTA. For global tensor maps, use
:doc:`/cuda/tile_ops/tma` instead.

The pointer contract
^^^^^^^^^^^^^^^^^^^^

``apply`` takes ``dst_shared, src_shared`` followed by explicit ``chunk_bytes``,
``mbar`` and ``remote_cta_id``. Source and destination regions use coordinates
``(*group,*repeat,element)``. The final axis covers exactly ``chunk_bytes``, a positive
multiple of 16; its bytes are consecutive and its origin is 16-byte aligned. Both
shared/shared.dyn storage are accepted.

All three addresses supplied by the caller are **local** shared addresses:

* ``src`` names storage in the issuing CTA.
* ``dst`` names the local counterpart of the destination allocation. Lowering
  maps its address to ``remote_cta_id`` with ``mapa`` for each repeat.
* ``mbar`` names the local counterpart of the completion barrier. Lowering
  maps it to the same remote CTA once before the repeat loop.

Do not pass an already mapped destination or barrier. Both CTAs must have corresponding
live shared allocations at those offsets. ``remote_cta_id`` is the rank within the
cluster, not an absolute block index. ``pred`` gates the copy instruction and must agree
with the caller's transaction accounting.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Both regions use shared/shared.dyn storage, contain whole elements and contiguous
chunks, and have 16-byte-aligned starts. Group/repeat shapes agree. The caller supplies
a valid cluster rank and matching launch/completion protocol.

Code generation
~~~~~~~~~~~~~~~

Map mbar to remote_cta_id once per tile call. In each repeat, map the destination atom
pointer with mapa and emit one cp.async.bulk using the local source, mapped
destination/barrier and chunk_bytes. The core instruction count equals the repeat
product; address preparation includes mapa. Allocation, barrier
initialization/transaction accounting and cluster synchronization belong to the caller.

Repeated chunks and synchronization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A float16 view ``(8,128):(256,1)`` with ``chunk_bytes=256`` represents eight separate
contiguous chunks with gaps between them. It emits eight bulk copies. Set expected
transaction bytes to ``8*256``. The source and destination may have different outer
strides, provided each matching 128-element chunk is contiguous and aligned on its own
side.

Do not flatten across a physical gap and request one larger chunk: it would describe
different bytes and is rejected. There is no automatic split or fallback to a scalar
loop. Cluster launch configuration, issuer election, barrier initialization,
expected-byte accounting, proxy fences, phase progression and storage lifetime are
caller responsibilities. Ordinary ``cp.async.commit_group``/``wait_group`` is not this
completion mechanism.

The following example provides the complete cluster handshake.

.. _cuda-tile-cluster-copy-example:

Complete cluster copy
~~~~~~~~~~~~~~~~~~~~~

This example requires thread-block cluster support. Two single-thread CTAs
keep the sender and receiver roles explicit. Both initialize their local
barrier and synchronize before CTA 0 sends 256 bytes to CTA 1:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def cluster_copy(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=2, block=1, cluster=2))
        rank = T.cuda.cluster_cta_id("x")
        src = T.alloc_shared((64,), "float32", layout="default", align=16)
        dst = T.alloc_shared((64,), "float32", layout="default", align=16)
        mbar = T.alloc_shared((1,), "uint64", align=8)
        remote_addresses = T.alloc_local((2,), "uint64", layout="default")
        T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.mbarrier_init.release.cluster()
        T.cuda.cluster_sync()
        if rank == 0:
            for i in range(64):
                src[i] = A[i]
            T.ptx.fence.proxy.async_.shared__cta()
            T.ptx.mapa.u64(remote_addresses[0], mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mapa.u64(remote_addresses[1], dst.ptr_to([0]), T.uint32(1))
            T.ptx["cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"](
                T.cast(remote_addresses[1], ty="uint32"),
                src.ptr_to([0]), T.uint32(256),
                T.cast(remote_addresses[0], ty="uint32"),
            )
        else:
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(256))
            T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
            for i in range(64):
                B[i] = dst[i]
        T.cuda.cluster_sync()

The raw PTX path maps the sender's local ``dst`` and ``mbar`` addresses
to their counterparts in rank 1 before issuing the copy. The receiver
expects ``64 * 4 = 256`` bytes and waits for phase zero. The final cluster
barrier keeps both CTAs' shared allocations alive until consumption
finishes.

.. code-block:: python

    import torch
    import tvm_tirx

    a = torch.arange(64, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    tvm_tirx.compile(cluster_copy, target="cuda")(a, b)
    torch.testing.assert_close(b, a)

.. _cuda-tile-contract-cp-async-bulk-tensor:

Tensor memory accelerator transfers
-----------------------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``shared, global_region_or_tensormap for load;
global_region_or_tensormap, shared for store/reduction``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.
   * - ``box``
     - static tuple[int]; required
     - One extent per opcode rank, outer-to-inner, each 1..256.
   * - ``coords``
     - tuple of runtime integers; required
     - Starting coordinates relative to the descriptor view, outer-to-inner.
   * - ``coord_steps``
     - static nested integer tuples; default ()
     - One rank-length increment vector for each group/repeat axis; must be supplied when these axes exist.
   * - ``format``
     - static dtype str; default operand dtype
     - Descriptor representation; may be tf32 over float32 storage.
   * - ``element_strides``
     - static tuple[int]; default all ones
     - Per-dimension sampling steps, each 1..8.
   * - ``swizzle``
     - static int; default 0
     - 0/1/2/3 means none/32B/64B/128B; must match the shared image and an external descriptor.
   * - ``interleave``
     - static int; default 0
     - 0/1/2 means none/16B/32B; changes the canonical image.
   * - ``l2_promotion``
     - static int; default 0
     - Descriptor L2 promotion mode, 0..3.
   * - ``oob_fill``
     - static int; default 0
     - 0: zero; 1: NaN for supported floating-point formats.
   * - ``gather4``
     - four runtime integers; required for gather4/scatter4
     - The four explicit row coordinates.
   * - ``mbar``
     - runtime pointer; required for loads
     - Completion barrier pointer in the opcode's shared address space.
   * - ``cta_mask``
     - runtime integer; required for multicast
     - Native multicast mask, uint16/uint32 according to the opcode.
   * - ``cache_policy``
     - runtime uint64; required with cache hint
     - Native cache-policy operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

The descriptor format and interleave mode define the storage representation.
Non-interleaved shared suffix is ceil(box/element_strides); gather4/scatter4 uses
(4,box[1]). Interleaved and padded FP4 images are specified below.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

TMA transfers a descriptor-defined global box to or from a shared-memory image. Select
direction, rank, tile/gather/scatter mode, multicast, cache qualifiers, and any
reduction operator in the opcode. ``apply`` encodes the specified descriptor and issues
that instruction at explicit coordinates; it does not plan a decomposition of an
arbitrary tensor layout.

Operands and descriptor fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a load the operands are ``shared, global``; for a store/reduction they are ``global,
shared``. The global operand is either a rectangular global tensor region or an ordinary
pointer to a caller-encoded TensorMap. In the latter case the caller is responsible for
matching its fields to the tile contract; lowering does not inspect or select opaque
descriptors.

Without interleave, the global region has exactly the opcode rank, unit innermost
stride, and suitably aligned outer byte strides. Its dimensions and outer strides can be
runtime values and are passed to the host descriptor encoder. Known invalid fields are
rejected rather than silently promoted, folded, transposed or reinterpreted as another
dtype.

The shared suffix is the box image: ``ceil(box[i]/element_strides[i])`` per dimension,
compact before swizzling. Every atom's unswizzled origin is 128-byte aligned, even for
swizzle zero; a swizzled backing base is aligned to the whole swizzle period. Swizzled
subregions use their inverse-swizzled origins so the hardware applies the phase once.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Global tensor rank matches opcode rank, plus the explicit channel-slice axis for
interleave. Shared group/repeat extents are static and its image matches the descriptor.
Global dimensions/strides may be runtime values checked by the host encoder. Every
unswizzled shared origin is 128B aligned; swizzled allocations align to their period. An
opaque TensorMap cannot be inspected: its declared fields are the caller's contract.

Code generation
~~~~~~~~~~~~~~~

For a global tensor region, create the host TensorMap from its dimensions/strides and
explicit fields. For each group/repeat, add coord_steps to coords, reverse coordinate
order for PTX, undo the shared origin swizzle once, and emit one selected native
instruction. gather4/scatter4 substitutes the explicit row coordinates. Host descriptor
setup is separate from device repeat loops; commit/wait, barrier accounting and election
remain caller code.

Stores, reductions, and completion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The corresponding store spelling is
``cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group``. Reduction uses, for
example, ``cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.bulk_group``. The
coordinate/box contract is the same, with global destination first. Each thread that writes shared memory executes
``T.ptx.fence.proxy.async_.shared__cta()``; a CTA barrier orders these
writes and fences before the issuing thread starts the transfer. That
thread then calls ``T.ptx.cp.async_.bulk.commit_group()`` and
``T.ptx.cp.async_.bulk.wait_group(0)``. Initialize a reduction destination to the
intended starting value. No mbarrier operand is used for these stores.

Loads use their supplied mbarrier, with explicit initialization, expected transaction
bytes, and phase wait. Multicast adds the explicit CTA mask and the corresponding
cluster/barrier setup. The tile call elects no thread and inserts neither waits nor
descriptor prefetch. A desired prefetch is an ordinary ``T.ptx.prefetch.tensormap``
call.

Gather4 and scatter4
~~~~~~~~~~~~~~~~~~~~

These forms use a 2D descriptor with ``box=(1,width)`` and a shared image ``(4,width)``.
The four row coordinates are ordinary runtime operands in ``gather4``; they need not be
consecutive. For a prepared ``shared`` of shape ``(4,32)`` and a matching barrier, an
issue can be written as::

    T.cuda.tile.apply(
        T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"],
        shared, A, box=(1,32), coords=(0,0),
        gather4=(row,row+3,row+7,row+11), mbar=mbar.ptr_to([0]),
    )

For uint32 this is one 512-byte gather4 issue. Scatter4 uses global destination first
and the matching store opcode/completion protocol. Gather/scatter coordinate selection
is caller logic.

Interleave and packed FP4
~~~~~~~~~~~~~~~~~~~~~~~~~

With ``interleave=1`` or ``2``, a rank-3..5 opcode takes global axes
``(N,C_slices,*spatial,Ci)``. ``Ci`` holds one 16B/32B slice and is implicit in the
descriptor rank. Its elements are contiguous and the innermost spatial stride is one
slice. Interleave 2 requires swizzle 1 and 32-byte global alignment. One issue selects
**one** C slice, even if a larger C box field is supplied; channel iteration requires an
explicit repeat and C step.

For example, uint16/interleave-2 with ``box=(1,1,8)`` has shared atom ``(1,1,8,16)``.
Add a leading repeat of three and ``coord_steps=((0,1,0),)`` to issue three successive
channel slices. Gather4/scatter4 do not use interleave. On B200, 16B interleave with
swizzle 1/2/3 can trap, including when issued through raw PTX.

``format="float4_e2m1fn"`` selects the load-only ``16U4_ALIGN16B`` format. The inner
global dimension is a multiple of 128, the inner box is 128, and global origins/outer
strides are 32-byte aligned. Each sixteen nibbles occupy the low eight bytes of a
sixteen-byte shared block. The remaining eight bytes are undefined padding; barrier
accounting counts payload bytes, not this padding. Swizzle 0/3 and gather4 use this
explicit packed image.

For a complete kernel, see :doc:`/cuda/tile_ops/tma`.

.. _cuda-tile-contract-mma-sync:

Warp matrix multiply-accumulate
-------------------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``D, A, B, C``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

D/C use f32 accumulator elements. A/B use b32 containers packing two f16/bf16 values.
The supported tile forms are m16n8k8 and m16n8k16 with f32 accumulation; opcode fixes
the formats and row/column interpretation.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

The warp tile pattern supports the explicit
``mma.sync.aligned.m16n8k{8,16}.row.col.f32.{f16,bf16}.{f16,bf16}.f32`` forms registered
by the PTX dialect. Each issue multiplies fixed register fragments and accumulates into
four float32 registers per lane. The tile layer does not select K=8 versus K=16, change
input type, load shared memory, or transpose an operand implicitly.

Operand coordinates and ownership
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The call takes ``D,A,B,C``. After optional execution-group and batch axes, the canonical
shapes are:

.. list-table::
   :header-rows: 1

   * - Operand
     - Shape
     - Storage
   * - D / tensor C
     - ``(M_repeat,N_repeat,16,8)``
     - float32
   * - A
     - ``(M_repeat,K_repeat,16,k/2)``
     - b32 words containing two adjacent f16/bf16 K values
   * - B
     - ``(N_repeat,K_repeat,8,k/2)``
     - The same packed representation

B's logical atom axes are ``(N,K/2)``, even though the mathematical product uses B
transposed. For lane ``l``, native register ``r`` corresponds to:

.. list-table::
   :header-rows: 1

   * - Operand
     - Atom coordinate
     - Registers per lane
   * - A
     - ``(l//4+8*(r%2), l%4+4*(r//2))``
     - k/4
   * - B
     - ``(l//4, l%4+4*r)``
     - k/8
   * - D / C
     - ``(l//4+8*(r//2), 2*(l%4)+r%2)``
     - 4

Local offsets can be strided; lane ownership must match these coordinates. All 32 lanes
execute the warp instruction. Warpgroup/CTA scope prepends a warp group axis; that axis
is not an additional matrix dimension.

Legality conditions
~~~~~~~~~~~~~~~~~~~

All tensor operands are local, with the fixed fragment ownership below. D/A/B and tensor
C have matching group/batch and M/N/K repeat extents as applicable. C may be an explicit
scalar broadcast. Writable D slots are nonoverlapping. All 32 warp lanes participate. No
instruction-specific keyword, including pred, is accepted.

Code generation
~~~~~~~~~~~~~~~

Expand batch, M_repeat, N_repeat, K_repeat in that order. Bind each native fragment
using the selected shape's lane/register coordinates. Feed C on K_repeat=0 and the
previous D thereafter, and emit one selected mma.sync. No alpha/beta arithmetic,
implicit fragment conversion or memory staging is inserted.

Accumulation, transpose and alpha/beta
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Repeat order is batch, M, N, then K. C can be a matching register tensor or an explicit
scalar broadcast. Only the first K issue reads the original C; later issues read D. This
is the documented recurrence, rather than an arbitrary traversal inferred from layouts.

To transpose a logical input, prepare a view whose canonical A/B coordinates refer to
the intended values while preserving the required lane map. A view alone cannot exchange
values between lanes; an incompatible physical owner requires explicit preparation.
``alpha``/``beta`` scaling likewise belongs before or after MMA at the call site, with
the intended rounding order.

The tile call inserts no shared loads, ``ldmatrix``, communication, or barrier. Use
:doc:`/cuda/tile_ops/warp_matrix` to prepare fragments when that native shared/register image
matches your operands. Sparse and other MMA shapes require their own explicit pattern;
this page does not claim those forms.

For a complete kernel, see :doc:`/cuda/tile_ops/warp_matrix`.

.. _cuda-tile-contract-tcgen05-ld-st:

TMEM loads and stores
---------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``registers, tmem for ld; tmem, registers for st; split forms use two
TMEM operands``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.
   * - ``imm_half_splitoff``
     - static int; required only for 16x32bx2
     - Native address immediate from the first TMEM half-access to the second; validated by the selected PTX form.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

Register operands are b32 words. Ordinary TMEM operands are 32-bit cells; pack/unpack
forms expose the documented extra halfword axis with 16-bit elements. The shape and xN
determine counts; storage dtype does not select packing.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

These warp instructions transfer a fixed image between TMEM and registers. The opcode
specifies the row/bit shape, repetition qualifier ``.xN``, and optional pack/unpack
mode. The tile repeats that exact instruction over caller-supplied views. A load takes
``registers,tmem``; a store takes ``tmem,registers``. Both require all 32 lanes of each
participating warp.

Native images
^^^^^^^^^^^^^

For the ordinary b32 forms both operands end in ``(row,column)``. Let ``l`` be the lane
ID and ``r`` a register index within one instruction:

.. list-table::
   :header-rows: 1

   * - Shape
     - Atom extents for xN
     - Registers per lane
     - Register r contains coordinate
   * - ``32x32b``
     - ``(32,N)``
     - N
     - ``(l,r)``
   * - ``16x64b``
     - ``(16,2N)``
     - N
     - ``(l//4+8*(l%2), l//2%2+2*r)``
   * - ``16x128b``
     - ``(16,4N)``
     - 2N
     - ``(l//4+8*(r%2), l%4+4*(r//2))``
   * - ``16x256b``
     - ``(16,8N)``
     - 4N
     - ``(l//4+8*(r//2%2), r%2+2*(l%4)+8*(r//4))``

TMEM rows and columns are consecutive physical lanes/cells inside each atom. Each warp
can access its own 32-lane slab; a 16-row instruction may select either half explicitly.
Warp scope therefore requires a statically selected warp. Warpgroup/CTA scope exposes
the warp as a group axis. Register slots may be separated as long as their owners match
this table.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Regions have the specified native images and matching group/repeat prefixes. Register
ownership follows the per-shape lane/register mapping. TMEM rows align to the atom row
count and belong to the executing warp slab. Packed origins select a cell's low half.
Split regions share a backing allocation and their addresses differ by
imm_half_splitoff. No pred keyword is accepted.

Code generation
~~~~~~~~~~~~~~~

For each repeat, enumerate the selected shape's native register coordinates, bind those
slots, and encode the warp-relative TMEM address. Emit one selected tcgen05.ld/st,
adding the split immediate where applicable. Thread synchronization, tcgen05.wait and
producer/consumer fences are explicit caller operations.

Pack/unpack is a different TMEM image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``tcgen05.ld...pack::16b.b32`` and ``tcgen05.st...unpack::16b.b32`` keep the ordinary
b32 register image. The TMEM operand instead has suffix ``(row,column,half)`` with
``half=2`` and a 16-bit storage type. The two halves of register word ``[row,column]``
correspond to the **low 16 bits of two different TMEM cells**, at columns ``2*column``
and ``2*column+1``.

In 16-bit layout units, the column and half strides are respectively ``4 @ TCol`` and
``2 @ TCol``. Strides 2 and 1 would instead select both halves of one dense cell
and would not satisfy the instruction. Hardware performs the pack/unpack; the tile emits
no software conversion. The upper 16 bits left by an unpack are not part of its defined
16-bit payload.

Split accesses: 16x32bx2
~~~~~~~~~~~~~~~~~~~~~~~~

This form uses register suffix ``(access,row,column)=(2,16,N)``; lane ``l`` owns
``(l//16,l%16,r)``. Supply two separate TMEM regions and the explicit native immediate
``imm_half_splitoff``. Each TMEM suffix is ``(16,N)`` or ``(16,N,2)`` for packing. For
example, for an already prepared warpgroup view ``mem`` with shape ``(4,32,16)``::

    T.cuda.tile.apply(
        T.ptx["tcgen05.ld.sync.aligned.16x32bx2.x4.b32"],
        registers, mem[:, :16, :4], mem[:, 16:32, 8:12],
        imm_half_splitoff=(16 << 16) + 8, scope="warpgroup",
    )

Both halves share the same TMEM backing allocation. The second encoded address equals
the first plus the immediate; no extra sixteen-row shift is implied by lane 16. For
stores, operand order is ``mem_low,mem_high,registers``. One instruction handles both
accesses for each repeat.

Completion and producer dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Call ``tcgen05.wait::ld`` before consuming loaded registers and ``tcgen05.wait::st``
before relying on stores. For an MMA/cp producer, complete its commit/mbarrier protocol
and required thread-sync fence before issuing a TMEM load; ``wait::ld`` does not
complete a preceding MMA. Allocation, deallocation, fences and synchronization remain
explicit.

For TMEM allocation, MMA completion, and result readback in one kernel, see
:doc:`/cuda/tile_ops/blackwell_tensor_cores`.

.. _cuda-tile-contract-tcgen05-cp:

Shared-to-TMEM copies
---------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``dst_tmem, src_shared``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

Both regions use scalar 32-bit storage words. Opcode shape gives (rows,words); multicast
determines destination replicas. Decompression uses the same b32 footprint but
explicitly changes the packed payload semantics.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

``tcgen05.cp`` copies a descriptor-defined shared image into TMEM. It is issued by a
single thread, not collectively by the warps whose TMEM lanes receive the data. Select
shape, CTA group, multicast and optional decompression directly in the opcode.

TMEM shape and replicas
^^^^^^^^^^^^^^^^^^^^^^^

Operands are ``dst_tmem, src_shared``, both with suffix ``(row,b32_word)`` and 32-bit
storage elements. An earlier axis is an explicit repeat.

.. list-table::
   :header-rows: 1

   * - Shape / multicast
     - Atom extents
     - TMEM lanes
   * - ``128x256b``
     - ``(128,8)``
     - Consecutive 128 lanes
   * - ``128x128b``
     - ``(128,4)``
     - Consecutive 128 lanes
   * - ``4x256b``
     - ``(4,8)``
     - 0, 32, 64, 96 relative to the origin
   * - ``64x128b.warpx2::02_13``
     - ``(64,4)``
     - 64 consecutive lanes, replicated 64 lanes apart
   * - ``64x128b.warpx2::01_23``
     - ``(64,4)``
     - Logical row r maps to ``64*(r//32)+r%32``, replicated 32 lanes apart
   * - ``32x128b.warpx4``
     - ``(32,4)``
     - 32 consecutive lanes, four replicas 32 lanes apart

The destination layout must state these replicas explicitly; the pattern does not infer
multicast. Columns are consecutive b32 cells with a 128-bit aligned origin. The whole
footprint, including replicas, must stay inside the 128-lane, 512-column TMEM address
space. A CTA-group-2 opcode makes both CTAs copy from their own corresponding shared
storage into their own TMEM. The launch, allocation and commit must use the matching
CTA-group protocol.

Shared descriptor address walk
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shared layout determines internal byte strides LDO/SDO and swizzle mode 0/1/2/3.
These are not call parameters. Before swizzling, row ``r`` starts at ``(r//8)*SDO +
(r%8)*(16 << swizzle)`` bytes relative to the descriptor origin. Without swizzling, word
``w`` adds ``(w//4)*LDO + (w%4)*4``. With swizzling, words are contiguous and add
``4*w``; hardware ignores LDO in that case.

Descriptor strides are nonnegative 16-byte multiples below 2^18; origins are 16-byte
aligned. Swizzled bases require 256/512/1024-byte period alignment for modes 1/2/3. The
complete physical walk is checked, including the origin's swizzle phase. A 4-row logical
region still needs a backing storage block large enough for the descriptor's eight-row
organization.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Both regions end in the selected (rows,words). Shared must match one of the descriptor
address walks in the operand specification. TMEM row/column mapping and replicas match the opcode exactly.
Column origin is 128-bit aligned and the full image fits 128 lanes and 512 columns.
Shared origins are 16B aligned, descriptor strides are nonnegative 16B multiples below
2^18, and swizzled allocations align to their full period.

Code generation
~~~~~~~~~~~~~~~

Extract LDO/SDO/swizzle from the shared region using
:ref:`cuda-matrix-descriptor-extraction`. For each repeat, encode the shared atom
origin, compute the TMEM address from TLane/TCol, and emit one selected tcgen05.cp. A
single elected thread issues it; allocation, proxy fencing, commit, wait and
deallocation remain explicit caller code.

Native FP4/FP6 decompression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Append ``.b8x16.b4x16_p64`` or ``.b8x16.b6x16_p32`` to the opcode. The b32 footprint
above stays unchanged: each 16-byte shared block contains sixteen packed values followed
by eight bytes (FP4) or four bytes (FP6) of padding. The destination contains sixteen
expanded bytes. Four-bit values gain two low zero bits; six-bit values gain two high
zero bits. A 256-bit row contains two such blocks. The pattern emits one native
decompression instruction per repeat, with no software unpacking loop.

For instance, a base opcode can be explicitly replaced with
``tcgen05.cp.cta_group::1.128x128b.b8x16.b4x16_p64`` **when the source data has been
prepared in that packed format**. The reference then compares the expanded values, not
the original words or padding. A dtype name alone neither prepares the payload nor
chooses this qualifier.

Allocation, shared proxy fencing, election, commit, wait and deallocation are caller
code. ``pred`` gates the cp but does not adjust a surrounding completion protocol.

The :doc:`Blackwell Tensor Core guide </cuda/tile_ops/blackwell_tensor_cores>`
compares this completion protocol with TMEM loads and stores.

.. _cuda-tile-contract-tcgen05-mma:

Tensor Core matrix multiply-accumulate
--------------------------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.apply(T.ptx[opcode], *operands, scope="thread", **parameters)``

Operand order: ``D, A, B; plus SFA, SFB for block_scale``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.
   * - ``instruction_shape``
     - static (M,N,K); required
     - Native instruction shape; M/N are totals for the CTA group.
   * - ``a_major, b_major``
     - static "k" or "mn"; default "k"
     - Instruction operand major order; TMEM A requires k.
   * - ``a_format, b_format``
     - static dtype str; default respective tensor dtype
     - Numeric representation, including explicit tf32 over float32.
   * - ``neg_a, neg_b``
     - static bool; default False
     - Instruction descriptor negation bits, subject to the selected kind.
   * - ``accum``
     - runtime bool; default False
     - Use existing D on the first K repeat; subsequent K repeats accumulate.
   * - ``sat_d``
     - static bool; default False; dense forms only
     - Instruction descriptor saturation bit.
   * - ``disable_output_lane``
     - runtime integer tuple; default all zeros; dense non-.ws forms
     - Four mask words per CTA.
   * - ``zero_column_mask``
     - runtime integer; default 0; dense .ws forms
     - Native zero-column mask descriptor operand.
   * - ``pred``
     - runtime bool; default None
     - Execute the instruction when true; None is unconditional. A skipped asynchronous issue performs no completion accounting.

Only the parameters listed here are instruction parameters. Opcode qualifiers are
written in ``T.ptx[opcode]``.

Data representation
~~~~~~~~~~~~~~~~~~~

D/A/B retain their documented native TMEM/shared representations. D uses b32 cells even
with 16-bit accumulators; shared values use their storage bit width. Format overrides
affect numeric interpretation, not storage layout. SFA/SFB are 8-bit scale formats in
their native TMEM image.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

Choose dense, weight-stationary ``.ws``, or ``.block_scale`` in the PTX opcode, together
with CTA group and kind. Supply ``instruction_shape=(M,N,K)`` explicitly. M is the total
for the CTA group, so M=128 with ``cta_group::2`` gives 64 local M rows per CTA. The
instruction is issued by an explicitly elected thread; group scope does not elect one.

Operands and repeat axes
^^^^^^^^^^^^^^^^^^^^^^^^

Dense/.ws calls take ``D,A,B``. Block scale adds ``SFA,SFB``. After any execution-group
and batch prefix, the canonical shapes are:

.. list-table::
   :header-rows: 1

   * - Operand
     - Shape, with g = CTA group
     - Storage
   * - D
     - ``(M_repeat,N_repeat,M/g,N)``
     - TMEM
   * - A
     - ``(M_repeat,K_repeat,M/g,K)``
     - Shared or TMEM
   * - B
     - ``(N_repeat,K_repeat,N/g,K)``
     - Shared
   * - SFA
     - ``(M_repeat,K_repeat,M/g,scale_vec)``
     - TMEM
   * - SFB
     - ``(N_repeat,K_repeat,N,scale_vec)``
     - TMEM, including full N in each CTA of a group-2 operation

M64 .ws with TMEM A adds an atom bank axis: A ends in ``(2,64,K)`` rather than
``(64,K)``. D and A use the selected native TMEM datapath; the pure
``tmem_mma_operand_layout`` helper constructs that layout from explicit shape/type/mode
parameters. It does not choose the MMA opcode.

D stores one accumulator per b32 TMEM cell. Float16 accumulation uses only the low half,
hence a TCol stride of two in 16-bit element units. A in TMEM is packed according to its
input format instead. ``a_format``/``b_format`` default to tensor dtype; supply
``"tf32"`` explicitly for TF32 instructions whose storage dtype is float32.

Shared descriptors are byte-stride contracts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The shared layouts determine each operand's LDO/SDO/swizzle. No descriptor tuple is
passed. ``a_major``/``b_major`` are explicitly ``"k"`` or ``"mn"``. For swizzle modes
0..3, a descriptor row is ``16 << swizzle`` bytes and groups contain eight rows. K-major
rows enumerate M/N; MN-major rows enumerate K. The next row group and next contiguous
chunk use:

.. list-table::
   :header-rows: 1

   * - Mode
     - Row-group stride
     - Contiguous-chunk stride
   * - K-major
     - SDO
     - LDO; swizzled instruction atoms fit within one row
   * - MN-major, swizzled
     - SDO
     - LDO
   * - MN-major, unswizzled
     - LDO
     - SDO

The final physical address is the selected swizzle of this byte walk. The origin is
inverse-swizzled before encoding so it is not permuted twice. Descriptor strides are
16-byte multiples below 2^18, origins are 16-byte aligned, and backing allocations align
to the full swizzle period. Fields are extracted from fixed canonical coordinates and
the full address walk is checked.

Descriptor mode 4 is the separate 128B-span/32B-atomicity mode for 32-bit MN-major
operands. It groups **four** K rows of 128 bytes per SDO and uses LDO for the next
128-byte M/N chunk. Its permutation is ``offset ^ ((offset >> 2) & 96)``, with a
512-byte period. It is not an alternative row pitch for ordinary mode 3, and is not this
pattern's K-major mode. TF32 MN-major requires this mode; other MN-major element widths
use modes 0..3.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Instruction kind, shape, formats, major order, CTA group and scale vector must agree.
Shared operands match an extracted descriptor; TMEM operands match the native datapath
and replica map. All group/batch/repeat extents are static. Descriptor strides fit 16B
units below 2^18; origins and backing allocations meet descriptor alignment. M64 .ws
A-in-TMEM has the explicit bank axis described in the operand specification.

Code generation
~~~~~~~~~~~~~~~

Expand batch, M_repeat, N_repeat, K_repeat in that order. Read D at (m,n), A at (m,k), B
at (n,k), and the corresponding scale images. Extract each shared descriptor; encode
instruction_shape, formats, major and flags into the instruction descriptor. Emit one
selected MMA per repeat tuple. Its accumulation predicate is accum OR k_repeat!=0. No
implicit K decomposition, staging, election, commit or wait is inserted.

Weight-stationary and block scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``.ws`` is selected in the opcode, not inferred from packed C. M64 .ws uses Layout E
with an explicit bank dimension for TMEM A; M128 .ws uses Layout D. The current pattern
requires CTA group 1 and N=64/128/256 for .ws. Splitting an M64 N128 operation into two
N64 issues changes each instruction's bank boundary from 64 to 32 columns. Preserve the
intended result with explicit B preparation and a matching C view; cutting two
consecutive logical N slices is insufficient. ``test_tcgen05_mma_ws_split_n.py`` gives
the complete numerical example with distinct data in the two A banks.

Block-scaled opcodes specify ``.kind`` and ``.scale_vec::1X/2X/4X``. Prepare SFA/SFB in
TMEM: scale row r uses lane ``r%32`` and byte-column ``4*(r//32)+sf``, with four
replicas spaced 32 lanes apart. Scale-factor IDs are encoded from each tensor's own
byte-column origin. Supplying scales neither allocates them nor changes the explicit
input kind/scale-vector mode.

``neg_a``, ``neg_b``, ``sat_d``, dense ``disable_output_lane`` masks,
``zero_column_mask`` for .ws, and ``pred`` are explicit parameters where applicable. The
tile does not initialize barriers, elect a thread, perform TMA, prepare scale factors,
commit or wait. Keep the shared/TMEM allocations alive until their consumers complete.

For a complete kernel, see :doc:`/cuda/tile_ops/blackwell_tensor_cores`.

.. _cuda-tile-contract-ewise:

Scalar function application
---------------------------

Call parameters
~~~~~~~~~~~~~~~

``T.cuda.tile.ewise(function, dst, *inputs, scope="thread")``

Operand order: ``function, dst, *inputs``.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type / default
     - Meaning
   * - ``scope``
     - static str; default "thread"
     - Execution scope; group axes follow :ref:`cuda-tile-operand-coordinates`. It does not elect an issuing thread.

The scalar function defines the computation. ``scope`` is the only tile keyword.

Data representation
~~~~~~~~~~~~~~~~~~~

function is a typed scalar TIRx Function; its parameter and return types exactly match
input and destination scalars. The atom is one scalar element, with no native
register-group suffix.

Operand specification
~~~~~~~~~~~~~~~~~~~~~

``T.cuda.tile.ewise(function, dst, *inputs, scope="thread")`` applies a typed scalar
TIRx function independently to register elements. This is the one CUDA tile operation
whose atom is a caller-defined expression rather than one PTX instruction. It can
represent exp, SiLU, scale/bias, casts, or another scalar combination. Use
:ref:`cuda-tile-contract-add-sub-mul`, :ref:`cuda-tile-contract-fma`, or :ref:`numeric conversion <cuda-tile-contract-cvt>` when the native instruction and its
qualifiers are the contract.

The scalar atom
^^^^^^^^^^^^^^^

Tensor operands have coordinates ``(*group,*repeat)``. There is no trailing PTX
register-group axis: every coordinate after the group prefix selects one scalar
invocation. Inputs and destination have matching extents and belong to the executing
thread. Read-only zero-stride input views express tensor broadcasts; a destination must
not overlap itself.

Scalar arguments broadcast to every invocation. Their types must match the function
parameters, and its result type must match the destination dtype. The function supports
scalar bindings and returns, including an if/else with a scalar return on both paths. It
does not express reductions, memory access algorithms, or inter-thread communication.

Legality conditions
~~~~~~~~~~~~~~~~~~~

Tensor operands are local and have identical group/repeat shapes. Scalar inputs
broadcast; tensor broadcasting is an explicit zero-stride view. Destinations belong to
the issuing thread and do not overlap. The function contains scalar bindings and a
scalar return on every path, including supported conditional returns. No
instruction-specific keyword is accepted.

Code generation
~~~~~~~~~~~~~~~

Inline the scalar function at each repeat coordinate and store its result to the
corresponding destination slot. There is one scalar function evaluation per element, not
a guaranteed single PTX instruction. Numeric conversion, packed arithmetic and
load/store staging must be written explicitly by the caller.

For a complete kernel, see :doc:`/cuda/tile_ops/arithmetic`.

.. _cuda-matrix-descriptor-extraction:

Shared matrix descriptors
-------------------------

``tcgen05.cp`` and ``tcgen05.mma`` obtain shared descriptors from their canonical
operand views. LDO/SDO are internal byte strides; swizzle is an internal hardware
encoding. None is an additional tile call parameter. The opcode and MMA major order
remain explicit user choices.

Extraction first tries the standard swizzle represented by ComposeLayout, then the
remaining legal modes in ascending order. cp permits modes 0..3; MMA permits 0..3 except
that 32-bit MN-major requires mode 4. For each mode, inverse-swizzle the bit addresses
and obtain row-group and column-chunk strides from fixed boundary coordinates. The
entire address walk must match, including every group/repeat and the region's nonzero
origin. This finite encoding check does not permute logical coordinates or select
another opcode.

An LDO unused by the atom is encoded as 16 bytes; an unused SDO is zero. Extraction
never probes outside the requested atom to determine those fields. A small region may
not expose all swizzle bits; physical equivalence over the complete access footprint is
sufficient. Prefer the layout's stated mode when valid; otherwise choose the first fully
matching legal mode. Symbolic origins remain address expressions; descriptor strides
must be compile-time constants.

See :ref:`cuda-tile-contract-tcgen05-cp` and
:ref:`cuda-tile-contract-tcgen05-mma` for address formulas, field roles, and alignment. The descriptor encodes the unswizzled origin so hardware
applies its phase exactly once. Base-offset and absolute-LDO modes are outside the
current tile contract.

Implementation and extension
----------------------------

``python/tvm_tirx/backend/cuda/tile/patterns/`` has one module per fixed instruction family.
``common.py`` evaluates canonical coordinates, validates group/repeat shapes and
physical storage, and emits repeat loops. ``descriptors.py`` extracts and verifies
shared matrix descriptor fields.
To add a pattern:

1. Define its canonical operand suffixes, ownership and physical address walk,
   all semantic qualifiers, and required caller preparation in this reference.
2. Resolve the exact instruction through ``ptx.instruction_forms``. Reuse the
   existing PTX table for legality and native operand emission.
3. Use ``Context`` and ``Region`` to check the supplied views at those fixed
   coordinates. Reject mismatches; do not search for another layout or opcode.
4. Emit one selected instruction inside ``Context.finish``. Put algorithms,
   packing, synchronization and thread election at the call site.
5. Add positive/negative contracts, script/JSON roundtrips, exact PTX spelling
   and issue-count checks, and GPU results reconstructed independently from
   the documented hardware mapping.

``apply`` constructs an ``ir.Call`` to ``tirx.cuda.tile.apply``. Its three
arguments hold the instruction name, a tuple of operands, and a tuple of
named parameters. The execution scope is stored in ``ScopeAttrs``; static
and runtime descriptor parameters remain in the call arguments. ``ewise``
stores its typed scalar body as ``LambdaExpr``, so the serialized IR does
not depend on a Python callable.

Region extents are positive static integers. Starts may be symbolic when their bounds
and alignment can be proved. General physical-equivalence discovery is outside the
contract. Gapped allocation covers the layout's physical span, including the address
range introduced by a composed swizzle.
