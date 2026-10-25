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

.. _ir-construction-and-printing:

Printing and Parsing
====================

In this chapter, we will introduce TVMScript and the interfaces for printing
and parsing IR. TVMScript provides a Python-based notation for program
definitions, while the printer and parser convert between IR objects and
source text. These interfaces allow generated programs to be inspected,
saved, and reconstructed. Compiler extensions can reuse them with custom
builder namespaces and printer hooks.

The ``@T.function`` decorator parses a Python function definition into a
``tirx.Function``. The parser reads the definition's source from its Python
file; the numerical computation runs when the compiled function is called.

The example program defines ``bias_sigmoid`` inside a Python factory:

.. literalinclude:: examples/tvmscript_sigmoid.py
   :language: python
   :start-at: from __future__ import annotations
   :end-at:     return bias_sigmoid

The factory creates ``n`` with ``T.dynamic`` before the function definition
and reuses it in the tensor shapes and loop bound. The decorated definition
records the loops, arithmetic, and indexing as IR.

For printing and parsing, we reuse ``module`` from
:doc:`functions_and_modules`.

The ``script()`` method returns TVMScript source for an IR object, so
generated loops and expressions can be inspected before lowering. The
130-channel specialization can be printed with:

.. code-block:: python

   function = module["bias_sigmoid_130"]
   source = function.script()
   print(source)

The printed loop structure shows the iteration over samples and channels:

.. code-block:: text

   for i in range(T.int64(0), T.int64(2) * n):
       for j in range(T.int64(0), T.int64(130)):
           ...

The full source includes the bias addition, sigmoid expression, and output
store. It keeps ``n`` symbolic so the tensor shapes and outer loop share
the same symbol when parsed again.

Smaller objects can be inspected too. ``function.params[0].ty.script()``
prints the input tensor type. ``function.show()`` displays highlighted source.
The :meth:`~tvm_tirx.tirx.Function.show` reference describes line numbers and
object annotations.
These interfaces also work on functions produced by a compiler pass.

The :doc:`TVMScript parser </reference/python/script.parser>` converts source
text back into IR through ``from_source``. A printed specialization can
therefore be reconstructed and checked before compilation:

.. code-block:: python

   from tvm_tirx.script import from_source

   restored_function = from_source(source, extra_vars={"T": T})
   ir.assert_structural_equal(function, restored_function)

The default source comments out its imports, so ``extra_vars`` supplies the
``T`` namespace from ``tvm_tirx.script.tirx``. The result is a ``tirx.Function``
that can be inspected or passed to ``tirx.compile``.

A self-contained source file needs its imports. With
``ir.comment_imports=False``, the :doc:`printer </reference/python/script.printer>`
includes them in the generated source. The helper below saves and reloads a
whole module with those imports:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :pyobject: save_program

``output_path`` is a ``pathlib.Path``. The saved file contains the module's
TVMScript definition and its imports, so ``from_source`` can reconstruct the
module from that file. The structural check verifies that the reconstructed
module has the same IR structure as the original. The returned module can
then be inspected, transformed, or compiled with ``tirx.compile``.
