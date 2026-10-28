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
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/cp_async.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/cp_async_bulk.rst

Asynchronous Copies
===================

In this chapter, we will copy data asynchronously and wait before using
it. A kernel can issue a transfer, perform independent work, then wait for
completion. The copy instruction and its completion calls are separate
parts of that sequence.

A global-to-shared copy
-----------------------

``cp.async`` copies a short contiguous chunk directly from global to
shared memory. On an Ampere-or-later GPU, we can use it for the row
exchange from :doc:`data_movement`. Each thread copies one four-float row,
then reads the next thread's row:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def async_read_next_row(
        A: T.Tensor((64, 4), "float32", align=16),
        B: T.Tensor((64, 4), "float32", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tid = T.cuda.thread_idx("x")
        shared = T.alloc_shared((64, 4), "float32", align=16)
        T.cuda.tile.apply(
            T.ptx["cp.async.cg.shared.global"],
            shared.sub[tid, :], A.sub[tid, :], chunk_bytes=16,
        )
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
        T.cuda.cta_sync()
        for j in range(4):
            B[tid, j] = shared[(tid + 1) % 64, j]

The source and destination views each contain four float32 elements,
which occupy the requested 16 bytes. Each thread issues one copy. The
following calls complete the handoff:

1. ``commit_group()`` collects that thread's previously uncommitted copies
   into a copy group.
2. ``wait_group(0)`` waits for all of its committed groups to complete.
3. ``cta_sync()`` lets the threads read data copied by their neighbors.

Independent computation can run between the commit and wait. A CTA
barrier alone does not wait for outstanding ``cp.async`` transfers.

We can check that the output has the same row shift as the synchronous
version:

.. code-block:: python

    import torch
    from tvm_tirx import tirx

    a = torch.arange(256, dtype=torch.float32, device="cuda").reshape(64, 4)
    b = torch.empty_like(a)
    tirx.compile(async_read_next_row, target="cuda")(a, b)
    torch.testing.assert_close(b, torch.roll(a, shifts=-1, dims=0))

A false ``pred`` skips the copy and leaves the destination unchanged.
``src_size`` instead copies a byte prefix and zero-fills the rest, while
``ignore_src=True`` zero-fills the whole chunk. The declared source region
must still satisfy the bounds checks. The :ref:`cp.async reference
<cuda-tile-contract-cp-async>` lists supported widths and optional operands.

Copies between CTAs
-------------------

A thread-block cluster can also transfer shared memory between its CTAs.
The ``cp.async.bulk.shared::cluster.shared::cta`` form uses a transaction
barrier in the destination CTA. Both CTAs must have matching live shared
allocations and initialize their barriers before the sender issues a copy.

The sender passes its local source address and the destination's remote
shared and barrier addresses to the copy instruction. For a 256-byte chunk,
the receiver registers 256 expected transaction bytes and waits for its barrier phase before
reading the destination. Both CTAs synchronize before either exits or
reuses the participating storage.

This path uses the destination's transaction barrier rather than ordinary
``cp.async`` copy groups. The :ref:`complete cluster example
<cuda-tile-cluster-copy-example>` uses raw PTX to show address mapping,
proxy fences, expected-byte accounting, and the receiver's wait. TMA's
descriptor-based global/shared transfers are covered separately in :doc:`tma`.
