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

Defining and launching kernels
==============================

In this chapter, we will define CUDA functions, configure their launches,
and assign work to threads. We will use the GPU add-one program from
:doc:`Quick Start </get_started/quick_start>`, then extend it with a runtime
parameter, a loop, and compile-time specialization.

A CUDA function
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
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,)))
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
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,)))
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

Symbolic shapes
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

Launch configuration
--------------------

Each call to ``add_one`` needs enough GPU threads for its current length
``n``. CUDA organizes threads into blocks. The launch configuration sets
the number of blocks (``grid``) and the number of threads in each block
(``block``).

In TVMScript, ``T.device_entry`` marks the start of the GPU computation.
Its ``launch`` argument sets that configuration. The corresponding
statements in ``add_one`` are:

.. code-block:: python

    T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,)))
    i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
    if i < n:
        B[i] = A[i] + 1

``block=(256,)`` gives each block 256 threads. The grid rounds ``n / 256``
up to provide at least ``n`` threads. For an input of 300 elements, this
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

.. _cuda-kernel-control-flow:

Conditions and loops
--------------------

A thread can process several elements with a loop. This version assigns
four consecutive elements to each thread, using 64 threads per block:

.. code-block:: python

    @T.function
    def add_one_four(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(64,)))
        base = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x") * 4
        for offset in T.unroll(4):
            i = base + offset
            if i < n:
                B[i] = A[i] + 1

The launch creates 64 threads per block, and the loop assigns four
consecutive elements to each thread, covering up to ``64 * 4 = 256``
elements. A regular ``range(4)`` would process the four offsets
sequentially within each thread. The example uses ``T.unroll(4)`` to ask
the compiler to expand the loop into four copies of its body, one for
each offset.

The ``if i < n`` condition checks each element individually. A thread
whose four-element group crosses the end of the tensor therefore writes
only the valid positions.

The :doc:`TIRx script API </reference/python/tirx.script>` also describes
loop forms for vectorized operations (``T.vectorized``) and
multidimensional iteration (``T.grid``).

Compile-time specialization
---------------------------

A symbolic dimension varies at runtime. When a size should be fixed before
compilation, ``@T.jit`` lets us declare it as ``T.constexpr`` and set it
through ``.specialize(...)``. Parsing happens at specialization time, and
the result is an ordinary ``tirx.Function``.

The following definition specializes the add-one function for a fixed
length. Put ``from __future__ import annotations`` at the beginning of the
Python file so the annotation can refer to ``N`` before specialization:

.. code-block:: python

    from __future__ import annotations
    from tvm_tirx.script import tirx as T

    @T.jit
    def add_one_fixed(
        A: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        *,
        N: T.constexpr,
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=((N + 255) // 256,), block=(256,)))
        i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
        if i < N:
            B[i] = A[i] + 1

    kernel = add_one_fixed.specialize(N=256)

``kernel`` has length 256 fixed in its parameter types and body. It can be
compiled with ``tvm_tirx.compile`` and called with its two tensor arguments;
``N`` was fixed during specialization. In contrast, ``@T.function`` parses
immediately, and the symbolic ``n`` in ``add_one`` remains a runtime dimension. The
:doc:`JIT API </reference/python/tirx.script.jit>` covers optional
parameters and specialization rules.

.. _cuda-inline-functions:

Reusable code fragments
-----------------------

A kernel can use small functions to reuse common pieces of code. For
example, the bounds check and store in ``add_one_four`` handle one element:

.. code-block:: python

    if i < n:
        B[i] = A[i] + 1

We can put these two statements in a function marked with ``@T.inline``:

.. code-block:: python

    @T.inline
    def add_one_if_valid(A, B, i, n):
        if i < n:
            B[i] = A[i] + 1

Each loop iteration in ``add_one_four`` can then call this function to
process its element:

.. code-block:: python

    for offset in T.unroll(4):
        i = base + offset
        add_one_if_valid(A, B, i, n)

During parsing, TVMScript expands ``add_one_if_valid(...)`` into its two
statements. The loop above is therefore equivalent to:

.. code-block:: python

    for offset in T.unroll(4):
        i = base + offset
        if i < n:
            B[i] = A[i] + 1

Other kernels can call ``add_one_if_valid`` in the same way to reuse the
update. The GPU executes the expanded checks and stores directly. The
:func:`inline API <tvm_tirx.tirx.script.inline>` describes its options.
