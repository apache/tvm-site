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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/threads_sync.rst

Calling CUDA and PTX
====================

This guide shows how to use three interfaces for CUDA programming in TVMScript.
Backend intrinsics under ``T.cuda`` provide common operations, such as
synchronization and warp shuffles. ``T.ptx`` lets us specify individual PTX
instructions. ``T.cuda.func_call`` embeds CUDA C++ source so a kernel can call
a custom ``__device__`` function.

Calling Backend Intrinsics
--------------------------

The backend provides helpers for thread synchronization, data movement,
reductions, and matrix operations:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Interface
     - Purpose
   * - ``T.cuda.cta_sync()``
     - Synchronize the threads in a block.
   * - ``T.cuda.warp_sync()``
     - Synchronize a full warp.
   * - ``T.cuda.warpgroup_sync(8)``
     - Synchronize a 128-thread warpgroup using named barrier 8.
   * - ``T.cuda.__shfl_xor_sync(mask, value, lane_mask, width)``
     - Read a value from another lane selected by XOR.
   * - ``T.cuda.cta_sum(value, num_warps, scratch)``
     - Reduce a value across a block using a shared scratch pointer.

For example, a warp can sum its 32 input values with
``T.cuda.__shfl_xor_sync``:

.. code-block:: python

    from tvm_tirx.script import tirx as T

    @T.function
    def warp_reduce(A: T.Tensor((32,), "float32", align=16)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32))
        lane_id = T.cuda.lane_id()
        v = T.alloc_local((1,), "float32")
        i = T.alloc_local((1,), "int32")
        v[0] = A[lane_id]
        i[0] = 16
        while i[0] >= 1:
            v[0] += T.cuda.__shfl_xor_sync(0xFFFFFFFF, v[0], i[0], 32)
            i[0] = i[0] // 2
        A[lane_id] = v[0]

All lanes participate, as selected by ``0xFFFFFFFF``. The XOR distances
16, 8, 4, 2, and 1 combine all 32 values; every lane receives the sum.
The device helper calls CUDA's ``__shfl_xor_sync``:

.. code-block:: c++

    v_ptr[0] = v_ptr[0] + __shfl_xor_sync(0xFFFFFFFF, v_ptr[0], i_ptr[0], 32);

The :doc:`TMA Transfers guide </programming_guides/tile/tma>` shows a complete barrier
initialization, arrival, and phase wait around an asynchronous transfer.

.. _cuda-direct-ptx:

Calling PTX Instructions
------------------------

``T.ptx`` selects individual PTX instructions. PTX names its destination
registers explicitly, so the destination is an argument to the call.
This kernel uses ``add.rn.f32`` to add two float32 values in each thread;
``rn`` selects round-to-nearest-even:

.. code-block:: python

    @T.function
    def add_ptx(
        A: T.Tensor((32,), T.float32),
        B: T.Tensor((32,), T.float32),
        C: T.Tensor((32,), T.float32),
    ):
        with T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=32)):
            tid = T.cuda.thread_idx("x")
            result = T.local_scalar("float32")
            T.ptx.add.rn.f32(result, A[tid], B[tid])
            C[tid] = result

``result`` is the writable destination. The PTX call puts the sum
in that register, and the next statement stores it in ``C``. The call
itself has no return value. A CUDA helper such as ``__shfl_xor_sync``
can instead return an expression, as in the preceding reduction.

The indexed spelling ``T.ptx["add.rn.f32"]`` constructs the same call.
Attribute chains escape Python keywords with a trailing underscore,
as in ``T.ptx.ld.global_.b32``; PTX's ``::`` becomes ``__``, as in
``T.ptx.mbarrier.init.shared__cta.b64``. Indexed names use the exact PTX
spelling, which is convenient for longer instruction names.

The :doc:`PTX reference </reference/python/backend.cuda.ptx>` covers
predication and special operands. The :doc:`PTX TableGen guide
</dev/cuda/ptx_tablegen>` explains how instruction definitions generate these
interfaces and their CUDA helpers.

Calling CUDA C++ Device Functions
---------------------------------

A kernel can also reuse a CUDA C++ device function by including its source
and calling it through ``T.cuda.func_call``. The example below defines a
small ReLU function, which returns the input when it is positive and zero
otherwise:

.. code-block:: python

    SRC = r"""
    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    """


    @T.function
    def relu_cuda(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=256))
        tx = T.cuda.thread_idx("x")
        B[tx] = T.cuda.func_call("my_relu", A[tx], SRC, ty="float32")

The generated CUDA source includes ``my_relu`` and calls it for each
element:

.. code-block:: c++

    __device__ __forceinline__ float my_relu(float x) { return x > 0.f ? x : 0.f; }
    // ...
    B_ptr[tx] = my_relu(A_ptr[tx]);
