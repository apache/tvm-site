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

This guide shows how to multiply two float16 matrices with a warp of 32
threads. ``ldmatrix`` loads matrix fragments from shared memory into registers,
and ``mma.sync`` multiplies those fragments with float32 accumulation. The
example computes ``Out = A @ B.T`` for A of shape ``(16, 16)`` and B of shape
``(8, 16)`` on an SM80-or-later GPU.

From Shared Memory to a Matrix Product
--------------------------------------

All 32 lanes cooperate on one ``16 x 8`` output tile. The data follows
this path:

.. raw:: html

   <iframe data-tirx-matrix-dataflow
           src="https://tvm.apache.org/images/docs/tirx/warp_matrix.html"
           title="Warp matrix multiplication: shared memory, register fragments, and output"
           style="display: block; width: 100%; height: 600px; border: 0;"
           loading="lazy"></iframe>
   <script>
     // Keep both matrix diagrams sized and themed with their documentation pages.
     (() => {
       const dark = matchMedia('(prefers-color-scheme: dark)');
       document.querySelectorAll('iframe[data-tirx-matrix-dataflow]').forEach((frame) => {
         if (frame.dataset.ready) return;
         frame.dataset.ready = 'true';
         const origin = new URL(frame.src, document.baseURI).origin;
         const syncTheme = () => {
           const selected = document.documentElement.dataset.theme;
           const theme = ['light', 'dark'].includes(selected) ? selected : (dark.matches ? 'dark' : 'light');
           frame.contentWindow.postMessage({ type: 'tirx-matrix-dataflow:theme', theme }, origin);
         };
         addEventListener('message', (event) => {
           if (event.source !== frame.contentWindow || event.origin !== origin) return;
           const data = event.data;
           if (data?.type !== 'tirx-matrix-dataflow:height') return;
           if (!Number.isFinite(data.height) || data.height < 100 || data.height > 3000) return;
           frame.style.height = Math.ceil(data.height) + 'px';
         });
         new MutationObserver(syncTheme).observe(document.documentElement, {
           attributes: true, attributeFilter: ['data-theme']
         });
         dark.addEventListener('change', syncTheme);
         frame.addEventListener('load', syncTheme);
         syncTheme();
       });
     })();
   </script>

A register fragment is the part of a matrix held in one lane's registers.
The kernel below uses ``ar`` and ``br`` for the input fragments and ``dr``
for the output fragment:

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
        A: T.Tensor((16, 16), T.float16),
        B: T.Tensor((8, 16), T.float16),
        Out: T.Tensor((16, 8), T.float32),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        a = T.alloc_shared((16, 16), T.float16, layout="default", align=16)
        b = T.alloc_shared((8, 16), T.float16, layout="default", align=16)
        for q in range(8):
            index = lane + 32 * q
            a[index // 16, index % 16] = A[index // 16, index % 16]
        for q in range(4):
            index = lane + 32 * q
            b[index // 16, index % 16] = B[index // 16, index % 16]
        T.cuda.cta_sync()
        a_words = a.view(T.uint32.dtype).rearrange(
            "(group row) (chunk word) -> (chunk group) row word", group=2, chunk=2, word=4
        )
        b_words = b.view(T.uint32.dtype).rearrange("row (chunk word) -> chunk row word", chunk=2)
        ar = T.alloc_local((4,), T.uint32, layout="default")
        br = T.alloc_local((2,), T.uint32, layout="default")
        dr = T.alloc_local((4,), T.float32, layout="default")
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

The threads copy ``A`` and ``B`` into shared-memory buffers ``a`` and ``b``,
then synchronize before ``ldmatrix`` reads them.

The ``uint32`` views reinterpret pairs of float16 values as 32-bit words.
``rearrange`` expresses the shared-memory views in
``(matrix, row, packed_word)`` coordinates for ``ldmatrix``.
All 32 lanes participate in ``ldmatrix.x4`` to load A into ``ar`` and
``ldmatrix.x2`` to load B into ``br``.

The load and MMA layouts describe the same register fragments in each
instruction's coordinates. This lets ``mma.sync`` reuse the loaded values
without moving them between lanes. The :ref:`warp MMA reference
<cuda-tile-contract-mma-sync>` gives the fixed lane mappings.

The MMA call takes ``D, A, B, C``, with ``T.float32(0)`` as the initial C
value. The ``m16n8k16`` instruction computes the product in one issue,
leaving four float32 results per lane in ``dr``. The final loop stores
them at their corresponding positions in ``Out``.

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

Other Fragment Shapes and Transfers
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
