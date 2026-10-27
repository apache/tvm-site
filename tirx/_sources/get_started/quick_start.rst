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

Quick Start
===========

In this chapter, we'll write and run a small TIRx program using TVMScript,
which lets us describe tensor computations with Python syntax. We'll also
inspect the IR that TVMScript creates from the code. The examples use
PyTorch tensors; the :doc:`/dev/source_build` guide explains how to set up
TIRx.

Our example computes ``B[i] = A[i] + 1``, where ``A`` is the input tensor
and ``B`` is the output. Both tensors have the same dynamic length ``n``,
so the compiled program can handle different input lengths without
recompilation.

We'll show two versions of this program: one for the CPU and one for an
NVIDIA GPU. We'll first compile and run the CPU version using LLVM.

.. literalinclude:: examples/add_one_cpu.py
   :language: python
   :start-after: # begin program
   :end-before: # end program

``@T.function`` turns the ``add_one`` definition into IR. The code
in ``add_one`` loops over the input and writes each result to ``B``.
``tirx.compile`` compiles it for the CPU. The compiled function accepts
PyTorch tensors directly, and ``torch.testing.assert_close`` checks its
output.

The same ``add_one`` object lets us inspect the program's IR. We can get
the type of parameter ``A`` through ``params[0].ty``, then read its symbolic
length from ``shape[0]``:

.. code-block:: python

   input_type = add_one.params[0].ty
   length = input_type.shape[0]

The :doc:`IR Walkthrough </ir/overview>` explains these IR objects and how
to work with them.

We can also run this computation on an NVIDIA GPU. In addition to LLVM,
it still needs CUDA support, an NVIDIA driver, the CUDA toolkit, and
``cuda-bindings`` to run.

.. literalinclude:: examples/add_one_cuda.py
   :language: python
   :start-after: # begin program
   :end-before: # end program

``T.device_entry`` marks the GPU computation and configures blocks of 256
threads. Each thread handles one element, so the block count is rounded
up to cover ``n`` elements. The ``if i < n`` check keeps extra threads in
the last block from accessing the tensors. We compile the function for
CUDA with ``tirx.compile`` and run it with input and output tensors on
``"cuda:0"``, then check the result with PyTorch as in the CPU example.

The complete examples are available as
:download:`add_one_cpu.py <examples/add_one_cpu.py>` and
:download:`add_one_cuda.py <examples/add_one_cuda.py>`. You can run them with
``python add_one_cpu.py`` and ``python add_one_cuda.py``, respectively.
