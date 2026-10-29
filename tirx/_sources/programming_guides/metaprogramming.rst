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

Metaprogramming
===============

.. note::

   All the code in this guide is in
   `examples/programming_guides/metaprogramming.py
   <https://github.com/apache/tvm-tirx/blob/main/examples/programming_guides/metaprogramming.py>`_.

This guide shows how to use Python to generate, reuse, and specialize TIRx
kernels. TIRx functions are written in Python, so ordinary Python code runs
before compilation and decides what each kernel contains: which dtypes and
sizes it uses, which helper code it includes, and which values are fixed.

The examples use the following imports:

.. literalinclude:: ../../examples/programming_guides/metaprogramming.py
   :language: python
   :start-after: # [imports.begin]
   :end-at: from tvm_tirx.script import tirx as T

Parameterizing Kernels with Captured Values
-------------------------------------------

A Python function can build kernels. The inner ``@T.function`` can read any
value from the enclosing Python scope, such as a shape variable, a dtype, a
constant, or a launch size. Each call to the outer function produces a new
kernel with those values filled in:

.. literalinclude:: ../../examples/programming_guides/metaprogramming.py
   :language: python
   :start-after: # [generator.begin]
   :end-before: # [generator.end]

The parser reads ``n``, ``dtype``, ``value``, and ``block`` from the enclosing
scope when it parses ``add``. In ``add_two_f16``, the tensors have element type
``T.float16``, the constant becomes ``T.float16(2.0)``, and each block has 128
threads. ``n`` stays a symbolic dimension, so the kernel still accepts tensors
of any length. The script prints:

.. code-block:: text

    tensor([ 2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.], device='cuda:0',
           dtype=torch.float16)

Calling ``make_add(T.float32, 1)`` instead produces a separate float32 kernel
from the same definition. The parser needs the function's source code, so
define kernel generators in a Python file rather than in an interactive
session that has no source to read.

.. _cuda-inline-functions:

Reusable Code Fragments
-----------------------

Kernels often repeat the same few statements. A function marked with
``@T.inline`` holds such a fragment, and a kernel calls it like a regular
function. In the following kernel, each thread handles two neighboring
elements and calls ``add_one_if_valid`` once for each:

.. literalinclude:: ../../examples/programming_guides/metaprogramming.py
   :language: python
   :start-after: # [inline.begin]
   :end-before: # [inline.end]

During parsing, TVMScript expands each ``add_one_if_valid(...)`` call into the
helper's statements, with the arguments substituted. The body of
``add_one_pairs`` is therefore equivalent to:

.. code-block:: python

    i = (T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")) * 2
    if i < n:
        B[i] = A[i] + 1
    if i + 1 < n:
        B[i + 1] = A[i + 1] + 1

No function call remains in the compiled kernel, and other kernels can call
``add_one_if_valid`` in the same way.

Compile-Time Specialization
---------------------------

A symbolic dimension such as ``n`` varies at runtime. When a size should
instead be fixed before compilation, ``@T.jit`` lets us declare it as a
``T.constexpr`` parameter and set it later through ``.specialize(...)``:

.. literalinclude:: ../../examples/programming_guides/metaprogramming.py
   :language: python
   :start-after: # [jit.begin]
   :end-before: # [jit.end]

Unlike ``@T.function``, which parses its definition immediately, ``@T.jit``
waits until ``.specialize`` supplies the constexpr values. ``add_one_256`` is
an ordinary TIRx function with 256 fixed in its parameter types, launch
configuration, and bounds check, and it compiles with ``tvm_tirx.compile`` like
any other function. The annotations refer to ``N`` before it has a value, which
is why the file starts with ``from __future__ import annotations``.
