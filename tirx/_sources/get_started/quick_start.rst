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

.. note::

   All the code in this guide is under
   `examples/quick_start <https://github.com/apache/tvm-tirx/tree/main/examples/quick_start>`_
   in the repository.

This guide walks through a small ``add_one`` program that computes
``B = A + 1`` on CPU and CUDA with TIRx, a minimal, open ML compiler
foundation:

- **Programs in Python**: tensor programs are written as Python functions and
  compiled directly.
- **Transparent program structure**: every element of a program, from
  functions to statements and types, is accessible from Python and other
  languages, so developers and agents can inspect and transform it.
- **Built for extension**: all compiler APIs are open, so developers and
  agents can quickly iterate on extensions in separate packages that link to
  TIRx dynamically, without rebuilding one monolithic bundle.

This guide shows the first two in action and gives a first taste of the third.

.. admonition:: Prerequisite
   :class: hint

   - Python 3.10 or newer
   - PyTorch, to create and check tensors
   - CUDA 13 or newer environment, for the CUDA example
   - TIRx, installed with:

     .. code-block:: bash

        pip install --upgrade --force-reinstall apache-tvm-tirx
        # add the cuda-bindings dependency:
        # pip install --upgrade --force-reinstall "apache-tvm-tirx[cuda]"


Write a Simple ``add_one``
--------------------------

A TIRx program is an ordinary Python function marked with ``@T.function``.
Here ``add_one`` takes an input tensor ``A`` and an output tensor ``B``, each
holding ``n`` float32 values. We create ``n`` with ``T.dynamic`` instead of
picking a fixed number, so the same program works for tensors of any length.

.. tabs::

   .. group-tab:: CPU

      .. literalinclude:: ../../examples/quick_start/add_one_cpu.py
         :language: python
         :start-after: # [function.begin]
         :end-at: B[i] = A[i] + 1

   .. group-tab:: CUDA

      .. literalinclude:: ../../examples/quick_start/add_one_cuda.py
         :language: python
         :start-after: # [function.begin]
         :end-at: B[i] = A[i] + 1

The CPU version is a loop that walks over all ``n`` elements. The CUDA
version spreads the work across many GPU threads instead: ``T.device_entry``
starts the GPU code and launches it in blocks of 256 threads, with enough
blocks to cover all ``n`` elements. Each thread works out the element ``i`` it
owns and updates only that one. The last block may have more threads than
elements left, so the ``if i < n`` check keeps those extra threads from going
past the end of the tensors.

Compile and Run
---------------

With the program written, we compile it for a target, ``"llvm"`` for the CPU
or ``"cuda"`` for an NVIDIA GPU, using ``tvm_tirx.compile``. The result is a
compiled program that we call like a function, passing PyTorch tensors
directly, so we create the inputs with ``torch`` and check the output with
``torch.testing.assert_close``:

.. tabs::

   .. group-tab:: CPU

      .. literalinclude:: ../../examples/quick_start/add_one_cpu.py
         :language: python
         :start-after: # [run.begin]
         :end-before: # [run.end]

   .. group-tab:: CUDA

      .. literalinclude:: ../../examples/quick_start/add_one_cuda.py
         :language: python
         :start-after: # [run.begin]
         :end-before: # [run.end]

The script prints the result, each input value plus one (the CUDA version also
shows ``device='cuda:0'``):

.. code-block:: text

   tensor([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10.])


Inspect and Transform the Program
---------------------------------

Besides compiling and running ``add_one``, we can also look inside it. The
``add_one`` returned by ``@T.function`` is a tree of program elements,
commonly called an intermediate representation (IR). To see how the program
is structured, we can use ``format_tree`` to display its IR as a tree:

.. code-block:: python

   import tvm_tirx

   print(tvm_tirx.ir.format_tree(add_one))

.. code-block:: text

   Function
   ├── params[0]: A = Var(ty=TensorType(shape=[n], dtype=float32))
   ├── params[1]: B = Var(ty=TensorType(shape=[n], dtype=float32))
   └── body: SeqStmt
       └── seq[0]: For(begin=IntImm(value=0), end=n)
           ├── var: i = Var(ty=int32)
           └── body: SeqStmt
               └── seq[0]: TensorStore(dest=B, indices=[i])
                   └── value: Add
                       ├── a: TensorLoad(source=A, indices=[i])
                       └── b: FloatImm(value=1.0)

Every element is accessible from Python and other languages, so we can walk
this tree directly in Python:

.. literalinclude:: ../../examples/quick_start/inspect_and_transform.py
   :language: python
   :start-after: # [inspect.begin]
   :end-before: # [inspect.end]

Named values in the program, such as the parameters ``A`` and ``B``, the length
``n``, and the loop index ``i``, are variables represented by
:py:class:`~tvm_tirx.ir.Var` objects. Note that the constant ``1`` was
converted to ``T.float32(1.0)`` to match the element type.

These elements are minimal `TVM-FFI <https://tvm.apache.org/ffi/>`_ objects,
which share one representation across languages. On top of them, TIRx provides
reusable APIs to build analyses and transformations. For example, we can
specialize ``add_one`` for tensors of exactly 128 elements by replacing every
use of ``n`` with ``128``:

.. literalinclude:: ../../examples/quick_start/inspect_and_transform.py
   :language: python
   :start-after: # [transform.begin]
   :end-before: # [transform.end]

.. code-block:: python

   @T.function
   def add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
       for i in range(128):
           B[i] = A[i] + T.float32(1.0)

``tvm_ffi.structural_map`` walks every element of the program and calls our
function on each ``Var``, replacing it with whatever we return.
The result, ``add_one_specialized``, is a new program specialized to tensors of
128 elements. We can compile it with ``tvm_tirx.compile`` like any other
program, while the original ``add_one`` stays unchanged.

.. seealso::

   :doc:`IR Overview </ir/overview>` for a more in-depth introduction to the
   main IR objects and their roles in the overall program structure.


Export and Load the Compiled Program
------------------------------------

So far, we have compiled and run ``add_one`` in the same Python process. In
practice, we often want to compile a program once and run it somewhere else:
in another process, on another machine, or inside an application that is not
written in Python.

We support this by exporting the compiled program as a shared library. Under
the hood, ``tvm_tirx.compile`` returns an ``Executable``. It holds the compiled
module, which contains the function ``add_one``; calling the executable, as we
did earlier, calls that function. The executable's ``export_library`` method
writes this module to a shared library:

.. tabs::

   .. group-tab:: CPU

      .. literalinclude:: ../../examples/quick_start/add_one_cpu.py
         :language: python
         :start-after: # [export.begin]
         :end-before: # [export.end]

   .. group-tab:: CUDA

      .. literalinclude:: ../../examples/quick_start/add_one_cuda.py
         :language: python
         :start-after: # [export.begin]
         :end-before: # [export.end]

The exported library is a regular TVM-FFI module, so any language or framework
that TVM-FFI supports, such as C++, Rust, and JAX, can load and call it.
Running it does not need the compiler, only the minimal TVM runtime libraries,
``libtvm_runtime.so`` and ``libtvm_runtime_cuda.so`` (for CUDA), each under
1 MB.

Load in Python
~~~~~~~~~~~~~~

Later, for example in a new Python process, ``tvm_ffi.load_module`` loads the
module back. Importing ``tvm_tirx`` loads the runtime libraries the module
needs. The loaded module exposes its functions by name, so we call ``add_one``
as an attribute of the module:

.. tabs::

   .. group-tab:: CPU

      .. literalinclude:: ../../examples/quick_start/load_add_one_cpu.py
         :language: python
         :start-after: # [load.begin]
         :end-before: # [load.end]

   .. group-tab:: CUDA

      .. literalinclude:: ../../examples/quick_start/load_add_one_cuda.py
         :language: python
         :start-after: # [load.begin]
         :end-before: # [load.end]

Load from C++
~~~~~~~~~~~~~

The following C++ program loads the CPU build, ``add_one_cpu.so``, and calls
``add_one`` through the TVM-FFI C++ API. TIRx kernels require 64-byte aligned
tensor data for performance, so the program uses the runtime tensor allocator
to provide the required alignment on each platform:

.. literalinclude:: ../../examples/quick_start/load_cpp.cc
   :language: cpp
   :start-after: // [main.begin]
   :end-before: // [main.end]

To build it, we link against only the runtime libraries, which
``tvm-tirx-config`` finds for us, so the program runs without the compiler.
Run it from the directory that contains ``add_one_cpu.so``:

.. code-block:: bash

   g++ -O2 load_cpp.cc \
       $(tvm-tirx-config --cxxflags --ldflags --libs runtime) \
       -Wl,-rpath,$(tvm-tirx-config --libdir) \
       -Wl,-rpath,$(tvm-ffi-config --libdir) \
       -o load_cpp
   ./load_cpp

The same program can also be built with CMake. The ``CMakeLists.txt`` in
``examples/quick_start`` finds the TIRx and TVM-FFI CMake packages with
``python -m tvm_tirx.config --cmakedir`` and ``python -m tvm_ffi.config
--cmakedir``, then links ``load_cpp`` against ``tvm_runtime::shared`` and
``tvm_ffi::shared``:

.. code-block:: bash

   cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build --config RelWithDebInfo
   build/load_cpp


Troubleshooting
---------------

If compiling for CUDA fails with ``ValueError: CUDA compilation requires an
explicit backend_config arch or Target arch when no CUDA device is available``
even though the machine has a GPU, TIRx could not find the CUDA runtime library
``libcudart.so.13``. Install and import a CUDA 13 build of PyTorch, which ships
this library, or install the CUDA 13 toolkit and add its ``lib64`` directory to
``LD_LIBRARY_PATH``.


Further Reading
---------------

- :doc:`/ir/overview`: the IR objects in depth and how they fit together.
- :doc:`Programming Guides </programming_guides/index>`: writing CUDA
  kernels with TIRx.
- :doc:`Python API </reference/python/index>`: the full Python API reference.
- :doc:`/dev/source_build`: building TIRx from source.
- `TVM-FFI documentation <https://tvm.apache.org/ffi/>`_: the FFI and ABI
  foundation TIRx is built upon.
