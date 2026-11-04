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

Getting Started with Rust Extensions
====================================

.. note::

   All the code in this tutorial is in the repository, as the Python package
   `examples/extension <https://github.com/apache/tvm-tirx/tree/main/examples/extension>`_.

TIRx is designed around the idea that "everything is a plugin": compiler
features can be provided by separately built libraries, using the same IR and
registration interfaces as built-in features. This tutorial shows how to write
such a library in Rust.

Case Study: Complex Numbers
---------------------------

To show "everything is a plugin" in practice, we use complex numbers as an
example. All complex-number support comes from an extension that loads into
TIRx without modifying its core library.

Our goal is that, once the extension is loaded, complex types and operators are
written in TVMScript just like built-in ones, and functions that use them
compile with TIRx. For example, the following function multiplies two complex
tensors element by element. ``A`` and ``B`` hold the real and imaginary parts
of the first tensor, ``C`` and ``D`` those of the second. We will compile and
run it at the end of the tutorial.

.. literalinclude:: ../../examples/extension/run_example.py
   :language: python
   :start-after: # begin kernel
   :end-before: # end kernel

``my_ext`` is the extension's Python package, which holds the builders of its
type and operators; :doc:`using_ir_extensions` shows how it provides them.
``z``, ``w``, and ``product`` are complex values that exist only in the IR.
The function's inputs and outputs are ordinary ``float32`` tensors, so no
runtime storage format for complex numbers is needed.

Code generation has no complex type, so by the time the function reaches the
backend, every complex value must be a pair of ``float32`` values. Expanding
complex multiplication as

.. math::

   (a + bi)(c + di) = (ac - bd) + (ad + bc)i

gives the loop we want the compiler to produce:

.. code-block:: python

   for i in range(N):
       z_re: T.let[T.float32] = A[i]
       z_im: T.let[T.float32] = B[i]
       w_re: T.let[T.float32] = C[i]
       w_im: T.let[T.float32] = D[i]
       product_re: T.let[T.float32] = z_re * w_re - z_im * w_im
       product_im: T.let[T.float32] = z_re * w_im + z_im * w_re
       R[i] = product_re
       Im[i] = product_im

``z``, ``w``, and ``product`` have each become a pair of variables for the real
and imaginary parts, and only scalar arithmetic remains.

To support complex numbers as a custom type, we need:

1. **A type.** For TIRx to recognize complex values, a complex type, the type
   of ``z``, ``w``, and ``product``.
2. **Operators.** For TIRx to compute on complex values, the operators
   ``complex``, ``cmul``, ``real``, and ``imag``, which build complex values,
   multiply them, and read their parts.
3. **A lowering pass.** For the target to compile the function, ``LowerComplex``,
   which lowers the complex type and operators to the scalar arithmetic above.

.. note::

   An extension can also add passes that optimize computations on its types,
   for example to simplify a product with a known real factor. They are written
   the same way as the lowering pass, so the example shows only the latter.

The rest of this page sets up the extension's Python package.
:doc:`defining_ir_extensions` then defines the type, operators, and pass, and
:doc:`using_ir_extensions` compiles and runs the function.

Setting Up the Project
----------------------

An extension can be shipped as a Python package that contains its Rust library
(or written in pure Python, like the version shown beside Rust in
:doc:`defining_ir_extensions`), so it can be installed and imported like any
other package. The example package, ``my-ext``, has this layout:

.. code-block:: text

   examples/extension/
   ├── Cargo.toml        # the Rust library, which depends on tvm-tirx
   ├── src/lib.rs        # the type, operators, and pass
   ├── CMakeLists.txt    # builds the library with cargo and installs it into my_ext
   ├── pyproject.toml    # the Python package, built with scikit-build-core
   ├── python/my_ext/    # loads the library and registers a compilation pipeline
   └── run_example.py    # compiles and runs a function that uses the extension

The Rust library is an ordinary crate that builds a shared library. It depends
on the `tvm-tirx <https://crates.io/crates/tvm-tirx>`_ crate, which re-exports
TVM FFI as ``tvm_tirx::tvm_ffi``, and also lists ``tvm-ffi`` itself, because
the code that ``#[dispatch]`` generates refers to the crate by that name:

.. literalinclude:: ../../examples/extension/Cargo.toml
   :language: toml
   :start-at: [lib]

First build and install TIRx from this checkout. With Rust 1.95 or newer,
install the extension from the repository root in the same Python environment:

.. code-block:: bash

   uv pip install scikit-build-core
   uv pip install --no-build-isolation --no-deps ./examples/extension

The example uses this checkout's Rust crate and pinned TVM FFI sources.
The package depends on ``apache-tvm-tirx`` at run time; disabling build
isolation keeps its native build aligned with these sources instead of
installing a released package in a temporary build environment.

Next, :doc:`defining_ir_extensions` walks through the extension's Rust library.
