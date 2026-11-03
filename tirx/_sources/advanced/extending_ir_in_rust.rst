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
registration interfaces as built-in features. This chapter puts that idea into practice
with a complex-number extension that loads into TIRx without modifying its
core library.

This tutorial shows how to extend TIRx from Rust: define new IR types and
operators, write a pass that lowers them, and use them from Python like
built-in features.

On this page, we will first look at why extensions are written in Rust, then
get ready for the rest of the tutorial: a Python package for the extension and
the example program it will compile.

Why Rust
--------

TIRx is built on a stable C++ library for its IR, analysis, transformations,
and code generation. In most cases, you work with it from Python: tensor
programs are written as Python functions in TVMScript, then compiled and run
directly.

With agents, compiler development iterates quickly. Along the way, a project may
need new types and operators to describe its own concepts, such as a new number
format or a hardware feature, and passes that lower them to what the backend
supports. To support this, TIRx provides the ``tvm-tirx`` crate, with which you
define new types, operators, and passes in Rust on the same IR objects as C++
and Python.

With the ``tvm-tirx`` crate, users can extend TIRx in separate Rust libraries.
Each extension builds and loads on its own, without changing the C++ library, so
the C++ core stays stable while the extension iterates quickly. Once loaded, its
types and operators can be written in TVMScript and its passes can run in Python
compilation pipelines, just like built-in ones.

Setting Up the Project
----------------------

An extension is shipped as a Python package that contains its Rust library, so
it is installed and imported like any other package. The example package,
``my-ext``, has this layout:

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

Case Study: Complex Numbers
---------------------------

To show how to define new types, operators, and passes, we will use complex
numbers as a running example throughout this tutorial: a complex type, operators
that build and multiply complex values, and a pass that expands complex
multiplication into real arithmetic:

.. math::

   (a + bi)(c + di) = (ac - bd) + (ad + bc)i.

The following function multiplies two tensors of complex numbers element by
element. ``A`` and ``B`` hold the real and imaginary parts of the first operand,
``C`` and ``D`` those of the second:

.. literalinclude:: ../../examples/extension/run_example.py
   :language: python
   :start-after: # begin kernel
   :end-before: # end kernel

``my_ext`` is the extension's Python package, which holds the builders of its
type and operators; :doc:`using_ir_extensions` shows how it provides them.
``z``, ``w``, and ``product`` are complex values that exist only in the IR.
The function's inputs and outputs are ordinary ``float32`` tensors, so no
runtime storage format for complex numbers is needed.

Next, :doc:`defining_ir_extensions` defines the type, operators, and pass this
function uses, and :doc:`using_ir_extensions` compiles and runs it.
