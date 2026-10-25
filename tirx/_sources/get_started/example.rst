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

Your First TIRx Program
=======================

In this chapter, we'll write and run a small TIRx program using TVMScript,
TIRx's Python-based notation for IR, then look at the IR it builds. Our example
adds one to each element of a tensor: ``B[i] = A[i] + 1``.

``A`` is the input tensor, and ``B`` stores the result. Both tensors have the
same dynamic length ``n``. We can call the compiled program with different
lengths without recompiling.

With an LLVM-enabled TIRx build, you can download
:download:`add_one_cpu.py <examples/add_one_cpu.py>` and run ``python add_one_cpu.py``.
Here's the complete program:

.. literalinclude:: examples/add_one_cpu.py
   :language: python
   :start-after: # begin program
   :end-before: # end program

Here, ``@T.function`` turns the ``add_one`` definition into IR. The code
in ``add_one`` loops over the input and writes each result to ``B``.
``tirx.compile`` compiles it for the CPU, and the NumPy assertion checks
its output. We'll explore the IR objects behind these constructs in the
:doc:`IR Walkthrough </ir/overview>`.

We can also run this computation on an NVIDIA GPU. In addition to LLVM,
this example needs CUDA support, an NVIDIA driver, the CUDA toolkit, and
``cuda-bindings``. You can download
:download:`add_one_cuda.py <examples/add_one_cuda.py>` and run ``python add_one_cuda.py``.
Here's the complete GPU program:

.. literalinclude:: examples/add_one_cuda.py
   :language: python
   :start-after: # begin program
   :end-before: # end program

Here, ``T.device_entry`` marks the GPU computation. Each thread handles one
element, and the launch uses enough blocks of 256 threads to cover ``n``
elements. The ``if i < n`` check skips any extra threads. ``tirx.compile``
compiles it for CUDA, and ``cuda(0)`` selects the GPU for the input and output
tensors. The NumPy assertion checks the result in the same way as the CPU example.
