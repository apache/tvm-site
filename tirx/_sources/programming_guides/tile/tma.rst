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
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/cp_async_bulk_tensor.rst

TMA Transfers
=============

This guide shows how to use the Tensor Memory Accelerator (TMA) to move
multidimensional tiles between global and shared memory. One thread issues the
transfer using a TensorMap descriptor, which records the global shape, strides,
element format, and tile configuration. TIRx prepares this descriptor in the
host launcher from the tensor operand and explicit ``box`` parameters.

TMA requires a Hopper-or-later GPU. The `Async Data Movement: TMA
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_tma/index.html>`_
chapter explains its hardware role and layout support in detail.

A Tile Load, Computation, and Store
-----------------------------------

The following kernel loads an ``(8, 32)`` float32 tile, adds one to each
element, and stores it back to global memory. Lane 0 issues the transfers;
all 32 lanes update one column each:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def tma_add_one(
        A: T.Tensor((8, 32), "float32", align=16),
        B: T.Tensor((8, 32), "float32", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane = T.cuda.lane_id()
        shared = T.alloc_shared((8, 32), "float32", layout="default", align=128)
        mbar = T.alloc_shared((1,), "uint64", align=8)
        if lane == 0:
            T.ptx.mbarrier.init.shared.b64(mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(mbar.ptr_to([0]), T.uint32(1024))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx["cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"],
                shared, A, box=(8, 32), coords=(0, 0), mbar=mbar.ptr_to([0]),
            )
        T.cuda.mbarrier_wait(mbar.ptr_to([0]), 0)
        for row in range(8):
            shared[row, lane] = shared[row, lane] + 1
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if lane == 0:
            T.cuda.tile.apply(
                T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"],
                B, shared, box=(8, 32), coords=(0, 0),
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group(0)
        T.cuda.cta_sync()

``box=(8, 32)`` selects the tile dimensions and ``coords=(0, 0)`` selects
its starting position in the global tensor, both in outer-to-inner order.
The shared tensor has the same tile shape. ``layout="default"`` gives it
contiguous row-major storage, and ``align=128`` aligns its base. The load
takes ``shared, A``; the store reverses the direction with ``B, shared``.

Load Completion
---------------

The load reports completion through ``mbar``, an eight-byte barrier object
in shared memory. Its initialization expects one arrival per phase.
``arrive.expect_tx`` performs that arrival and registers the tile's
``8 * 32 * 4 = 1024`` transaction bytes. The initial fence and CTA barrier
publish the initialized barrier before lane 0 issues the transfer.

The TMA engine completes those transaction bytes as it fills shared memory.
``mbarrier_wait(..., 0)`` waits for the initial phase to finish, so each
lane can then read and update its column. A phase completes when both its
expected arrivals and transaction bytes have been satisfied. The
`mbarrier chapter
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_async_barriers/index.html>`_
explains phases and reuse across pipeline stages.

Store Completion
----------------

After the updates, each thread executes the proxy fence to make its
shared-memory writes visible to the asynchronous transfer. The CTA barrier
then ensures all writers have finished their writes and fences before
lane 0 issues the store.

A TMA store uses a bulk async group. Lane 0 commits its store with
``bulk.commit_group()`` and waits for completion with
``bulk.wait_group(0)``. The final CTA barrier holds the other threads until
that wait finishes, allowing the shared buffer to be reused. These bulk
groups are distinct from the ordinary copy groups used by :doc:`async_copies`.

Compile and Run
---------------

The host launcher needs tensor-map encoding support, available with an
LLVM-enabled build or the CUDA C++ host backend. With LLVM enabled, the
kernel can be compiled and checked as follows:

.. code-block:: python

    import torch
    import tvm_tirx

    a = torch.arange(256, dtype=torch.float32, device="cuda").reshape(8, 32)
    b = torch.empty_like(a)
    tvm_tirx.compile(tma_add_one, target="cuda")(a, b)
    torch.testing.assert_close(b, a + 1)

The :ref:`TMA instruction reference <cuda-tile-contract-cp-async-bulk-tensor>`
covers repeated boxes, coordinate steps, swizzles, multicast, reductions,
and packed formats. For a complete multistage schedule, see
`Pipelining GEMM with TMA
<https://mlc.ai/modern-gpu-programming-for-mlsys/chapter_gemm_async/index.html>`_.
