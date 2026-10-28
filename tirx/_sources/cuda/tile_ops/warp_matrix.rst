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

Warp Matrix Instructions
========================

In this chapter, we will multiply two float16 matrices with a warp of
32 threads. ``ldmatrix`` loads matrix fragments from shared memory into
registers, and ``mma.sync`` multiplies those fragments with float32
accumulation. The example computes ``Out = A @ B.T`` for A of shape
``(16, 16)`` and B of shape ``(8, 16)`` on an SM80-or-later GPU.

From shared memory to a matrix product
--------------------------------------

All 32 lanes cooperate on one ``16 x 8`` output tile. The data follows
this path:

.. code-block:: text

    float16 inputs -> shared memory -> ldmatrix -> register fragments
                                                       |
                                                   mma.sync
                                                       |
                                      float32 registers -> Out

Each lane holds part of a matrix, called its register fragment. In this
example, the fragments have the following sizes:

.. list-table::
   :header-rows: 1

   * - Registers in each lane
     - Contents
     - Complete matrix across the warp
   * - ``ar``: four uint32 words
     - Eight float16 elements, packed in pairs
     - A: ``(16, 16)``
   * - ``br``: two uint32 words
     - Four float16 elements, packed in pairs
     - B: ``(8, 16)``
   * - ``dr``: four float32 values
     - Four output elements
     - Out: ``(16, 8)``

The instruction fixes which elements belong to each lane. The
``ldmatrix`` views use coordinates ``(matrix, row, packed_word)``: each
``8 x 8`` half matrix has eight rows of four words. A contains four such
matrices and B contains two, giving view shapes ``(4, 8, 4)`` and
``(2, 8, 4)``. The load layouts place their words in ``ar`` and ``br``.

The ``*_mma_layout`` views describe those same registers with MMA operand
coordinates. For example, A's ``(1, 1, 16, 8)`` view has one M repeat,
one K repeat, 16 rows, and eight packed words per row. These views let
both instructions use the same fragments without moving values between
lanes. The
:ref:`warp MMA reference <cuda-tile-contract-mma-sync>` gives the fixed
lane mappings represented by these layouts.

.. code-block:: python

    from tvm_tirx.script import tirx as T
    from tvm_tirx.backend.cuda.axis import laneid
    from tvm_tirx.tirx.layout import S, TileLayout

    a_load_layout = TileLayout(S[(4, 8, 4) : (1, 4 @ laneid, 1 @ laneid)])
    b_load_layout = TileLayout(S[(2, 8, 4) : (1, 4 @ laneid, 1 @ laneid)])
    a_mma_layout = TileLayout(S[(1, 1, 2, 8, 2, 4) : (4, 4, 1, 4 @ laneid, 2, 1 @ laneid)])
    b_mma_layout = TileLayout(S[(1, 1, 8, 2, 4) : (2, 2, 4 @ laneid, 1, 1 @ laneid)])
    d_mma_layout = TileLayout(S[(1, 1, 2, 8, 4, 2) : (4, 4, 2, 4 @ laneid, 1 @ laneid, 1)])


    @T.function
    def warp_matmul(
        A: T.Tensor((16, 16), "float16"),
        B: T.Tensor((8, 16), "float16"),
        Out: T.Tensor((16, 8), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        a = T.alloc_shared((16, 16), "float16", layout="default", align=16)
        b = T.alloc_shared((8, 16), "float16", layout="default", align=16)
        for q in range(8):
            index = lane + 32 * q
            a[index // 16, index % 16] = A[index // 16, index % 16]
        for q in range(4):
            index = lane + 32 * q
            b[index // 16, index % 16] = B[index // 16, index % 16]
        T.cuda.cta_sync()
        a_words = a.view("uint32").rearrange(
            "(group row) (chunk word) -> (chunk group) row word", group=2, chunk=2, word=4
        )
        b_words = b.view("uint32").rearrange("row (chunk word) -> chunk row word", chunk=2)
        ar = T.alloc_local((4,), "uint32", layout="default")
        br = T.alloc_local((2,), "uint32", layout="default")
        dr = T.alloc_local((4,), "float32", layout="default")
        T.cuda.tile.apply(
            T.ptx["ldmatrix.sync.aligned.m8n8.x4.shared.b16"],
            ar.view(4, 8, 4, layout=a_load_layout), a_words, scope="warp",
        )
        T.cuda.tile.apply(
            T.ptx["ldmatrix.sync.aligned.m8n8.x2.shared.b16"],
            br.view(2, 8, 4, layout=b_load_layout), b_words, scope="warp",
        )
        T.cuda.tile.apply(
            T.ptx["mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"],
            dr.view(1, 1, 16, 8, layout=d_mma_layout),
            ar.view(1, 1, 16, 8, layout=a_mma_layout),
            br.view(1, 1, 8, 8, layout=b_mma_layout),
            T.float32(0), scope="warp",
        )
        for r in range(4):
            Out[lane // 4 + 8 * (r // 2), 2 * (lane % 4) + r % 2] = dr[r]

Each thread first copies part of A and B to shared memory. The CTA barrier
ensures those stores are visible before ``ldmatrix`` reads them. The
``uint32`` views pack two float16 elements into each instruction operand;
they preserve the bits rather than convert the values to integers.

``ldmatrix.x4`` loads the four A matrices, and ``ldmatrix.x2`` loads the
two B matrices. Both instructions require all 32 lanes to participate.

The MMA call takes ``D, A, B, C`` in that order. Its views include repeat
axes of length one before the matrix dimensions. The selected
``m16n8k16`` instruction computes the whole product in one issue, with
``T.float32(0)`` as the initial C value. Each lane receives four output
elements in ``dr``; the final loop writes them to their matrix positions.
For lane 0, ``dr[0]`` through ``dr[3]`` belong to ``Out[0, 0]``,
``Out[0, 1]``, ``Out[8, 0]``, and ``Out[8, 1]``. The other lanes use the
same coordinate formula to cover the remaining output elements.

We can compile the kernel and compare its result with a float32 matrix
product:

.. code-block:: python

    import torch
    import tvm_tirx

    compiled = tvm_tirx.compile(warp_matmul, target="cuda")
    a = torch.randn((16, 16), dtype=torch.float16, device="cuda")
    b = torch.randn((8, 16), dtype=torch.float16, device="cuda")
    out = torch.empty((16, 8), dtype=torch.float32, device="cuda")
    compiled(a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float().T, rtol=2e-4, atol=2e-4)

Other fragment shapes and transfers
-----------------------------------

A longer K dimension needs multiple MMA issues. The tile interface
represents these with an explicit K-repeat axis in A and B. The first
issue reads C, and later issues accumulate into the preceding D. The
:ref:`warp MMA reference <cuda-tile-contract-mma-sync>` gives the repeat
shapes, lane mappings, and supported float16/bfloat16 forms.

``ldmatrix`` also supports ``.x1`` and ``.trans`` variants. The transpose
acts on each matrix of 16-bit elements. ``stmatrix`` transfers a register
fragment to shared memory on targets that support it, with the shared
destination as its first argument. The :ref:`matrix-transfer reference
<cuda-tile-contract-ldmatrix-stmatrix>` lists the required layouts and
alignment. These tile calls do not insert barriers: synchronize before
reading another thread's shared stores, or before other threads consume
or reuse a matrix written to shared memory.

For the hardware layout diagrams behind these mappings, see
`The Evolution of Tensor Core Data Layouts
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_layout_generations/index.html>`_
in Modern GPU Programming for MLSys.
