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

Using and Compiling the Extension
=================================

In this chapter, we look at the Python package that loads the library defined
in :doc:`defining_ir_extensions`, then use it to compile the complex
multiplication function and check the result against NumPy.

The Python Package
------------------

In :doc:`defining_ir_extensions`, the type and operators were registered in
static init blocks, which run as soon as the library loads. To write them in
TVMScript, Python also needs a module that holds their builders. Definitions
under ``tirx.`` go to ``T``, but an extension namespace such as ``my_ext`` is
published only into a module that the extension declares. The package's
`__init__.py <https://github.com/apache/tvm-tirx/blob/main/examples/extension_guides/rust/my_ext/__init__.py>`_
therefore declares itself as that module before it loads the library:

.. literalinclude:: ../../examples/extension_guides/rust/my_ext/__init__.py
   :language: python
   :start-after: # begin load
   :end-before: # end load

The three lines do the following:

- ``_init_exposed_op_api("my_ext", __name__, wrapper=auto_emit_void_call)``
  declares the module ``__name__`` as the destination of the operators whose
  names start with ``my_ext.``, and publishes each of them there without the
  prefix, as ``my_ext.cmul``. ``auto_emit_void_call`` gives the builders the
  TVMScript behavior of the ``T`` builders.
- ``_init_exposed_type_api("my_ext", __name__)`` does the same for the types
  whose keys start with ``my_ext.``, so ``my_ext.ComplexType`` is published as
  ``my_ext.Complex``.
- ``load_lib_module("my-ext", "my_ext", extra_lib_paths=[...])`` finds the
  library ``my_ext`` (``libmy_ext.so`` on Linux) of the package ``my-ext``,
  here next to ``__init__.py``, and loads it, which runs its static init
  blocks. It returns the library as a module, whose exported functions, such as
  ``lower_complex``, Python calls.

Both declarations come before the library loads, so its registrations reach
``my_ext`` as they happen. The two ``_init_exposed_*_api`` functions are
internal to ``tvm_tirx.ir`` for now. To parse printed IR that uses the
extension, pass the package as well, as in
``extra_vars={"T": T, "my_ext": my_ext}``.

The registered type and operators let the IR express complex values;
``LowerComplex`` turns them into a form the backend can compile. The default
pipeline does not know about the pass, so the package registers a pipeline,
also named ``my_ext``, that runs it first:

.. literalinclude:: ../../examples/extension_guides/rust/my_ext/__init__.py
   :language: python
   :start-after: # begin pipeline
   :end-before: # end pipeline

``lower_complex`` returns the pass that the library exports.

Compile the Function
--------------------

`run_example.py <https://github.com/apache/tvm-tirx/blob/main/examples/extension_guides/run_example.py>`_
imports ``my_ext``, which loads the library and registers the pipeline, and
compiles the function from :doc:`extending_ir_in_rust` with it:

.. literalinclude:: ../../examples/extension_guides/run_example.py
   :language: python
   :start-after: # begin compile
   :end-before: # end compile

Compiling with the default pipeline instead fails in LLVM code generation,
which has no complex type.

Check the Result
----------------

A successful compilation shows that ``LowerComplex`` left no complex values for
the backend. To check that its rewrite also computes the right numbers, the
script runs the compiled function on random ``float32`` inputs and compares
both outputs with NumPy's complex multiplication:

.. literalinclude:: ../../examples/extension_guides/run_example.py
   :language: python
   :start-after: # begin comparison
   :end-before: # end comparison

With the package installed as in :doc:`extending_ir_in_rust`, run the script:

.. code-block:: bash

   python examples/extension_guides/run_example.py

A successful run prints:

.. code-block:: text

   Complex multiplication matches NumPy.

``tests/python/ir/test_op_rust.py`` runs the script too, and also checks
printing and parsing, invalid calls, chained products, and that lowering leaves
no complex types or operators.

Next Steps
----------

Complex addition or conjugation needs one more operator and one more rewrite
rule in ``LowerComplex``, registered and written the same way. The example
keeps complex values inside a function: parameters and outputs are ordinary
tensors, and complex numbers have no runtime storage format or calling
convention. See the :doc:`Rust API reference <../reference/rust/index>` for
the full Rust API.
