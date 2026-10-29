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
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/functions.rst
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/control_flow.rst
   https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/parser_utils.rst

Language Basics
===============

This guide shows how to define CUDA functions, configure their launches, and
assign work to threads. We will use the GPU add-one program from :doc:`Quick
Start </get_started/quick_start>`, then extend it with a runtime parameter.

A CUDA Function
---------------

The ``@T.function`` decorator constructs a ``tirx.Function`` from a
TVMScript definition. The add-one function takes two tensors with the same
symbolic length ``n``:

.. code-block:: python

    import tvm_tirx
    from tvm_tirx.script import tirx as T

    n = T.dynamic("n")

    @T.function
    def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
        i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
        if i < n:
            B[i] = A[i] + 1

``A`` is the input tensor, and ``B`` holds the result. The body describes
the launch, the index assigned to each thread, and the guarded tensor store.

Parameters
----------

The parameter annotations specify the types of arguments a kernel accepts:

.. list-table::
   :header-rows: 1
   :widths: 34 36 30

   * - Annotation
     - Meaning
     - Argument
   * - ``T.Tensor(shape, dtype)``
     - A tensor with the declared shape and dtype
     - A tensor on the expected device
   * - ``T.int32``, ``T.float32``, etc.
     - A runtime scalar
     - A Python ``int`` or ``float``
   * - ``T.handle``
     - An opaque handle
     - A handle value
   * - ``T.constexpr`` (``@T.jit``)
     - A compile-time parameter
     - Passed to ``.specialize(...)``

We can extend ``add_one`` so each call chooses the amount to add.
The ``increment: T.float32`` parameter controls that amount:

.. code-block:: python

    @T.function
    def add_scalar(
        A: T.Tensor((n,), T.float32),
        B: T.Tensor((n,), T.float32),
        increment: T.float32,
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
        i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
        if i < n:
            B[i] = A[i] + increment

We can compile this version and choose the increment when we call it:

.. code-block:: python

    import torch

    a = torch.arange(4, dtype=torch.float32, device="cuda")
    b = torch.empty_like(a)
    add_scalar_compiled = tvm_tirx.compile(add_scalar, target="cuda")
    add_scalar_compiled(a, b, 0.5)
    torch.testing.assert_close(b, a + 0.5)

``a`` and ``b`` are CUDA PyTorch tensors, while ``0.5`` is a Python float
passed directly as ``increment``. The compiled function derives ``n``
from the tensor shapes, so the call needs no separate length argument.

.. _cuda-symbolic-shapes:

Symbolic Shapes
---------------

A kernel may need to handle tensors of different sizes. A symbolic shape
uses symbols for dimensions whose values are determined at runtime. In
``add_one``, ``T.dynamic`` declares the symbol ``n`` for the tensor length:

.. code-block:: python

    n = T.dynamic("n")

Both ``A`` and ``B`` use ``T.Tensor((n,), T.float32)``, so their lengths
must match. The compiled function gets the value of ``n`` from the input
tensor's shape when it is called. We can use the same compiled function
with inputs of length 100 and 200:

.. code-block:: python

    add_one_compiled = tvm_tirx.compile(add_one, target="cuda")
    for length in (100, 200):
        a = torch.arange(length, dtype=torch.float32, device="cuda")
        b = torch.empty_like(a)
        add_one_compiled(a, b)
        torch.testing.assert_close(b, a + 1)

The first call uses an input of shape ``(100,)``, so ``n`` is 100.
The second uses shape ``(200,)``, so ``n`` is 200. Each value determines
the launch's block count and the ``i < n`` bounds check in ``add_one``.
The caller passes only ``a`` and ``b``; the compiled function derives
``n`` from their shapes without recompilation.

Launch Configuration
--------------------

Each call to ``add_one`` needs enough GPU threads for its current length
``n``. CUDA organizes threads into blocks. The launch configuration sets
the number of blocks (``grid``) and the number of threads in each block
(``block``).

In TVMScript, ``T.device_entry`` marks the start of the GPU computation.
Its ``launch`` argument sets that configuration. The corresponding
statements in ``add_one`` are:

.. code-block:: python

    T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
    i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
    if i < n:
        B[i] = A[i] + 1

``block=256`` gives each block 256 threads. ``T.ceildiv(n, 256)`` rounds
``n / 256`` up to provide at least ``n`` threads. For an input of 300 elements, this
launches two blocks, or 512 threads in total.

``T.cuda.block_idx("x")`` gives the block index.
``T.cuda.thread_idx("x")`` gives the thread index within that block. The index
formula assigns a distinct element position ``i`` to each thread. The
``if i < n`` condition lets the first 300 threads write their results and
skips the remaining threads in this example.

In ``add_one``, the statements after ``T.device_entry(...)`` form the GPU
computation. The same section can also be written inside a
``with T.device_entry(...):`` block. :ref:`tvmscript-region-syntax`
explains both forms.

``grid`` and ``block`` accept an integer or up to three dimensions.
``LaunchConfig`` also supports cluster and stream settings, while
``KernelAttributes`` sets compile-time kernel properties such as launch
bounds. The :doc:`CUDA launch API </reference/python/backend.cuda.launch>`
covers these options; the
:doc:`CUDA script API </reference/python/backend.cuda.script>` lists the
coordinate helpers for blocks, threads, warps, and lanes.
