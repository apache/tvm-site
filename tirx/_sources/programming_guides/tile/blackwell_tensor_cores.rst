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

Blackwell Tensor Cores
======================

This guide shows how to multiply float16 matrices with ``tcgen05.mma`` and
retrieve its float32 result from Tensor Memory (TMEM). The example targets
``sm_100a`` and requires a GPU with that instruction support, such as a B200.
TMEM holds the accumulator while the Tensor Core reads the input tiles from
shared memory.

The Accumulator's Lifetime
--------------------------

The example computes ``Out = A @ B.T`` with A of shape ``(128, 64)``,
B of shape ``(64, 64)``, and Out of shape ``(128, 64)``. One CTA of 128
threads performs these stages:

.. raw:: html

   <iframe data-tirx-matrix-dataflow
           src="https://tvm.apache.org/images/docs/tirx/blackwell_tensor_cores.html"
           title="Blackwell matrix multiplication: TMEM allocation, MMA completion, and output"
           style="display: block; width: 100%; height: 740px; border: 0;"
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

The kernel uses ``d`` for the TMEM accumulator and ``frag`` for the output
values held in each thread's registers:

.. code-block:: python

    from tvm_tirx.script import tirx as T
    from tvm_tirx.backend.cuda.axis import tid_in_wg
    from tvm_tirx.backend.cuda.layout import mma_shared_layout, tmem_mma_operand_layout
    from tvm_tirx.tirx.layout import S, TileLayout

    a_layout = mma_shared_layout(T.float16, 3, (128, 64))
    b_layout = mma_shared_layout(T.float16, 3, (64, 64))
    d_layout = tmem_mma_operand_layout("D", (128, 64), T.float32, M=128, cta_group=1)
    register_layout = TileLayout(S[(128, 64) : (1 @ tid_in_wg, 1)])


    @T.function
    def blackwell_matmul(
        A: T.Tensor((128, 64), T.float16),
        B: T.Tensor((64, 64), T.float16),
        Out: T.Tensor((128, 64), T.float32),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        a = T.alloc_shared((128, 64), T.float16, layout=a_layout, align=1024)
        b = T.alloc_shared((64, 64), T.float16, layout=b_layout, align=1024)
        address = T.alloc_shared((1,), T.uint32, layout="default")
        mbar = T.alloc_shared((1,), T.uint64, layout="default", align=8)
        if warp == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(address), T.uint32(64)
            )
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
        for j in range(64):
            a[tid, j] = A[tid, j]
            if tid < 64:
                b[tid, j] = B[tid, j]
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        d = T.cuda.decl_tmem(
            address[0], ty_args=[T.Tensor((128, 64), T.float32, scope="tmem", layout=d_layout)]
        )
        av = a.rearrange("m (kr k) -> kr m k", k=16).view(1, 4, 128, 16)
        bv = b.rearrange("n (kr k) -> kr n k", k=16).view(1, 4, 64, 16)
        if tid == 0:
            T.cuda.tile.apply(
                T.ptx["tcgen05.mma.cta_group::1.kind::f16"],
                d.view(1, 1, 128, 64),
                av,
                bv,
                instruction_shape=(128, 64, 16),
                accum=False,
            )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        T.ptx.tcgen05.fence__after_thread_sync()
        frag = T.alloc_local((128, 64), T.float32, layout=register_layout)
        T.cuda.tile.apply(
            T.ptx["tcgen05.ld.sync.aligned.32x32b.x64.b32"],
            frag.view(4, 32, 64),
            d.view(4, 32, 64),
            scope="warpgroup",
        )
        T.ptx.tcgen05.wait__ld.sync.aligned()
        local = frag.view(64, layout="default")
        for j in range(64):
            Out[tid, j] = local[j]
        T.cuda.cta_sync()
        if warp == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(address[0], T.uint32(64))

All 32 threads of warp 0 collectively allocate 64 TMEM columns for the
``128 x 64`` float32 accumulator. The allocation writes its base address
to ``address[0]``.

The inputs are staged in shared-memory buffers ``a`` and ``b``.
``mma_shared_layout`` mode ``3`` selects a 128-byte swizzle with
1024-byte base alignment. Each thread executes ``fence.proxy.async``
after its copies; the CTA barrier then ensures all inputs are ready
for MMA and the TMEM address is visible to all threads.
``T.cuda.decl_tmem`` binds that allocation to tensor ``d``.

The views ``av`` and ``bv`` split K=64 into four K=16 slices, so thread 0's
single ``T.cuda.tile.apply`` call expands into four MMA instructions.
With ``accum=False``, the first instruction writes its product into ``d``;
the next three add their products to it.

Thread 0 calls ``tcgen05.commit`` to signal ``mbar`` once all four MMA
instructions complete, matching the barrier's arrival count of 1.
All threads wait for phase 0, then execute
``tcgen05.fence::after_thread_sync`` before loading the result from TMEM.

``register_layout`` assigns one output row of 64 float32 values to each
thread. All four warps use ``tcgen05.ld`` to load their rows into ``frag``.
``tcgen05.wait::ld`` waits for these loads to complete before each thread
writes its row to ``Out``.

The final CTA barrier ensures every warp has finished using TMEM.
Warp 0 then collectively relinquishes the CTA's allocation permit and
deallocates the 64 columns.

On a supported GPU, we can compare the result with a PyTorch float32
matrix product:

.. code-block:: python

    import torch
    import tvm_tirx

    compiled = tvm_tirx.compile(blackwell_matmul, target={"kind": "cuda", "arch": "sm_100a"})
    a = torch.randn((128, 64), dtype=torch.float16, device="cuda")
    b = torch.randn((64, 64), dtype=torch.float16, device="cuda")
    out = torch.empty((128, 64), dtype=torch.float32, device="cuda")
    compiled(a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float().T, rtol=2e-4, atol=2e-4)

Other TMEM Transfers
--------------------

The accumulator above needs no initial fill because the first MMA ignores
its previous contents. Other computations may need to transfer existing
values into TMEM or update them from registers. The available operations
have different participants and completion rules:

.. list-table::
   :header-rows: 1

   * - Operation
     - Transfer
     - Participation and completion
   * - ``tcgen05.ld``
     - TMEM to registers
     - A whole warp issues the load, then uses ``tcgen05.wait::ld``
       before consuming the registers.
   * - ``tcgen05.st``
     - Registers to TMEM
     - A whole warp issues the store, then uses ``tcgen05.wait::st``
       before a dependent access or reuse of the affected TMEM.
   * - ``tcgen05.cp``
     - Shared memory to TMEM
     - One elected thread issues the copy and commits it to an
       ``mbarrier``; consumers wait and use the thread-sync fence.

For ``tcgen05.mma`` and ``tcgen05.cp``, a commit signals completion through
a barrier arrival. :doc:`TMA loads <tma>` instead track completion using
an expected-byte count. The barrier setup must match the operation being
tracked.

``tcgen05.cp`` is useful for operands that must reside in TMEM, such as
the scale factors of a block-scaled MMA. Its shared source must be ready
for asynchronous access, just as the MMA inputs above must be. The
:ref:`shared-to-TMEM reference <cuda-tile-contract-tcgen05-cp>` describes
copy shapes, replication, and optional decompression. The
:ref:`TMEM load/store reference <cuda-tile-contract-tcgen05-ld-st>` covers
register layouts, packing, and transfer sizes.

The :ref:`MMA reference <cuda-tile-contract-tcgen05-mma>` specifies other
instruction shapes, CTA-pair operation, weight-stationary forms, and
block scaling. For diagrams of the accumulator and its memory accesses,
see `Blackwell Tensor Core
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tensor_cores/index.html>`_
and `Tensor Memory
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tmem/index.html>`_
in Modern GPU Programming for MLSys.
