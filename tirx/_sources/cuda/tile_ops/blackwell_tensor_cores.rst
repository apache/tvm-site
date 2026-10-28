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

Blackwell Tensor Core Operations
================================

In this chapter, we will multiply float16 matrices with ``tcgen05.mma``
and retrieve its float32 result from Tensor Memory (TMEM). The example
targets ``sm_100a`` and requires a GPU with that instruction support,
such as a B200. TMEM holds the accumulator while the Tensor Core reads
the input tiles from shared memory.

The accumulator's lifetime
--------------------------

The example computes ``Out = A @ B.T`` with A of shape ``(128, 64)`` and
B of shape ``(64, 64)``. One CTA of 128 threads performs these stages:

.. code-block:: text

    allocate TMEM -> stage A/B in shared memory -> issue MMA
                                                    |
                                    commit + wait for completion
                                                    |
    release TMEM <- write Out <- load TMEM into registers + wait

Different stages require different participants. The conditions in the
kernel select them explicitly:

.. list-table::
   :header-rows: 1

   * - Participants
     - Operations
   * - ``warp == 0``: all 32 threads of warp 0
     - Allocate TMEM, relinquish the allocation permit, and deallocate TMEM.
   * - ``tid == 0``: thread 0
     - Initialize the barrier, issue the MMA instructions, and commit them.
   * - All four warps
     - Stage inputs, synchronize, load the completed result from TMEM, and write Out.

``mma_shared_layout`` constructs the swizzled shared layouts for A and B.
Mode 3 selects a 128-byte swizzle; its 1024-byte period determines the
shared allocation alignment used below.
``tmem_mma_operand_layout`` maps the float32 accumulator to TMEM lanes
and columns for the chosen instruction. The ``128 x 64`` accumulator
uses 64 columns, so the allocation and deallocation both specify 64.

.. code-block:: python

    from tvm_tirx.script import tirx as T
    from tvm_tirx.backend.cuda.axis import tid_in_wg
    from tvm_tirx.backend.cuda.layout import mma_shared_layout, tmem_mma_operand_layout
    from tvm_tirx.tirx.layout import S, TileLayout

    a_layout = mma_shared_layout("float16", 3, (128, 64))
    b_layout = mma_shared_layout("float16", 3, (64, 64))
    d_layout = tmem_mma_operand_layout("D", (128, 64), "float32", M=128, cta_group=1)
    register_layout = TileLayout(S[(128, 64) : (1 @ tid_in_wg, 1)])


    @T.function
    def blackwell_matmul(
        A: T.Tensor((128, 64), "float16"),
        B: T.Tensor((64, 64), "float16"),
        Out: T.Tensor((128, 64), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
        tid = T.cuda.thread_idx("x")
        warp = T.cuda.warp_id()
        a = T.alloc_shared((128, 64), "float16", layout=a_layout, align=1024)
        b = T.alloc_shared((64, 64), "float16", layout=b_layout, align=1024)
        address = T.alloc_shared((1,), "uint32", layout="default")
        mbar = T.alloc_shared((1,), "uint64", layout="default", align=8)
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
            address[0], ty_args=[T.Tensor((128, 64), "float32", scope="tmem", layout=d_layout)]
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
        frag = T.alloc_local((128, 64), "float32", layout=register_layout)
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

The allocation writes a TMEM base address to the shared ``address`` slot.
After the CTA barrier publishes the address, ``decl_tmem`` gives the allocated
storage a tensor shape and layout. It does not perform another allocation.

Each thread stages input values in shared memory and executes a proxy
fence. The following CTA barrier lets the issuing thread proceed only
after all writers have finished this handoff to the asynchronous reader.
The A/B views split K=64 into four K=16 slices. For example,
``av`` has shape ``(1, 4, 128, 16)``: one M tile, four K repeats, and a
``128 x 16`` A slice per issue. B likewise has one N tile and four K
repeats. Consequently, the tile call issues four MMA instructions.
``accum=False`` starts a fresh result on the first slice; subsequent
slices accumulate into D.

The elected thread commits these operations to ``mbar``, initialized with
one expected arrival. All threads wait for phase 0 and execute
``tcgen05.fence::after_thread_sync`` before reading TMEM. This completion
protocol uses a barrier arrival, rather than the expected-byte count
used by :doc:`TMA copies <tma>`.

The register fragment assigns one output row to each thread. Its
``(4, 32, 64)`` view lets each of the four warps load its 32 TMEM lanes.
``tcgen05.wait::ld`` then makes the loaded registers ready for the output
stores. It waits for these loads; the preceding barrier and fence are
what establish that MMA has finished. The final CTA barrier keeps TMEM
alive until every warp has finished using it, after which warp 0 gives
up further allocations and releases the columns.

On a supported GPU, we can check the product as follows:

.. code-block:: python

    import torch
    from tvm_tirx import tirx

    compiled = tirx.compile(blackwell_matmul, target={"kind": "cuda", "arch": "sm_100a"})
    a = torch.randn((128, 64), dtype=torch.float16, device="cuda")
    b = torch.randn((64, 64), dtype=torch.float16, device="cuda")
    out = torch.empty((128, 64), dtype=torch.float32, device="cuda")
    compiled(a, b, out)
    torch.testing.assert_close(out, a.float() @ b.float().T, rtol=2e-4, atol=2e-4)

Other TMEM transfers
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
