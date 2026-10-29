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
:download:`__init__.py <../../examples/extension/python/my_ext/__init__.py>`
therefore declares itself as that module before it loads the library:

.. literalinclude:: ../../examples/extension/python/my_ext/__init__.py
   :language: python
   :start-after: # begin load
   :end-before: # end load

The two ``_init_exposed_*_api`` calls, internal to ``tvm_tirx.ir`` for now,
publish the operators and the type into ``my_ext``; ``auto_emit_void_call``
gives its builders the TVMScript behavior of the ``T`` builders.
``load_lib_module`` finds the library that the package installs next to
``__init__.py``. To parse printed IR that uses the extension, pass the package
as well, as in ``extra_vars={"T": T, "my_ext": my_ext}``.

The registered type and operators let the IR express complex values;
``LowerComplex`` turns them into a form the backend can compile. The default
pipeline does not know about the pass, so the package registers a pipeline,
also named ``my_ext``, that runs it first:

.. literalinclude:: ../../examples/extension/python/my_ext/__init__.py
   :language: python
   :start-after: # begin pipeline
   :end-before: # end pipeline

``lower_complex`` returns the pass that the library exports.

Compile the Function
--------------------

:download:`run_example.py <../../examples/extension/run_example.py>` imports
``my_ext``, which loads the library and registers the pipeline, and compiles
the function from :doc:`extending_ir_in_rust` with it:

.. literalinclude:: ../../examples/extension/run_example.py
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

.. literalinclude:: ../../examples/extension/run_example.py
   :language: python
   :start-after: # begin comparison
   :end-before: # end comparison

With the package installed as in :doc:`extending_ir_in_rust`, run the script:

.. code-block:: bash

   uv run --no-sync python examples/extension/run_example.py

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
arrays, and complex numbers have no runtime storage format or calling
convention. See the :doc:`Rust API reference <../reference/rust/index>` for
the full Rust API.
