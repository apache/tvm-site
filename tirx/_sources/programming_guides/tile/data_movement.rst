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
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/ld_st.rst
   https://tvm.apache.org/docs/_sources/tirx/tile_ops/mov.rst

Data Movement
=============

This guide shows how to move tensor data between global memory, shared memory,
and thread-local registers. The instruction determines the source and
destination spaces, while tensor views select the values to transfer. All three
instruction families take the destination first:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Instruction
     - Operands
     - Data movement
   * - ``ld``
     - ``registers, memory``
     - Memory to the executing thread's registers
   * - ``st``
     - ``memory, registers``
     - Registers to memory
   * - ``mov``
     - ``destination, source``
     - Register values within the same thread

A Shared-Memory Exchange
------------------------

A block can use shared memory to exchange data between threads. In this
example, each of 64 threads writes its four-element row, waits for the
other threads, then reads the next thread's row:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def read_next_row(
        A: T.Tensor((64, 4), "float32", align=16),
        B: T.Tensor((64, 4), "float32", align=16),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=64))
        tid = T.cuda.thread_idx("x")
        registers = T.alloc_local((4,), "float32", layout="default")
        shared = T.alloc_shared((64, 4), "float32", align=16)
        T.cuda.tile.apply(T.ptx["ld.global.v4.b32"], registers, A.sub[tid, :])
        T.cuda.tile.apply(T.ptx["st.shared.v4.b32"], shared.sub[tid, :], registers)
        T.cuda.cta_sync()
        T.cuda.tile.apply(
            T.ptx["ld.shared.v4.b32"], registers, shared.sub[(tid + 1) % 64, :]
        )
        T.cuda.tile.apply(T.ptx["st.global.v4.b32"], B.sub[tid, :], registers)

``A.sub[tid, :]`` selects a four-element row without reading it. The first
``apply`` performs the read; the next writes those values to shared
memory. The barrier makes every thread's write visible before the shared
load. Thread 63 reads row 0, so the output shifts the rows cyclically.

Each ``.v4.b32`` instruction transfers four 32-bit words. The row views and
register tensor have shape ``(4,)``: one four-word atom with no repeat
axes. The calls use the default thread scope, so each thread transfers its
own row. The 16-byte alignment and four-element row stride keep every
memory atom aligned.

We can compile the kernel and compare its result with a row shift:

.. code-block:: python

    import torch
    import tvm_tirx

    compiled = tvm_tirx.compile(read_next_row, target="cuda")
    a = torch.arange(256, dtype=torch.float32, device="cuda").reshape(64, 4)
    b = torch.empty_like(a)
    compiled(a, b)
    torch.testing.assert_close(b, torch.roll(a, shifts=-1, dims=0))

The two global/shared transfers pass through the caller's register tensor.
For a direct asynchronous global-to-shared transfer, see :doc:`async_copies`,
which includes the required completion calls.

Register Copies and Initialization
----------------------------------

Inside this kernel, we can save loaded values before reusing ``registers``
for another read. Insert the following statements after its global load:

.. code-block:: python

    saved = T.alloc_local((4, 1), "float32", layout="default")
    T.cuda.tile.apply(T.ptx["mov.b32"], saved, registers.view(4, 1))

The view exposes four repeats, each ending in a single-register atom
``(1,)``. The call copies the four words into ``saved``. Later operations
can use that copy after ``registers`` has been overwritten.

A scalar source broadcasts to every repeat. For example, the following
call initializes all four saved values to zero:

.. code-block:: python

    T.cuda.tile.apply(T.ptx["mov.b32"], saved, T.float32(0))

``mov`` preserves the bits within the same thread. A numeric type change
uses ``cvt``, covered in :doc:`arithmetic`. Transfers between threads use
shared memory or another communication instruction.

Cache qualifiers, predication, and destination preservation are described
in the :ref:`memory instruction reference <cuda-tile-contract-ld-st>` and
:ref:`register move reference <cuda-tile-contract-mov>`.
