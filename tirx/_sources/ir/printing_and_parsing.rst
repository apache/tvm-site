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

Printing renders IR as readable TVMScript, and parsing constructs IR from
Python source. Compiler extensions can reuse these interfaces to generate
programs, inspect the resulting IR, and reconstruct programs from source.
Extensions can customize how source builds IR through builder namespaces
and how IR is rendered through printer hooks.

For example, this function increments every element of a float32 tensor with
shape ``(2*n, 128)``. Save the code in a Python file so the decorator can inspect
its source:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin tvmscript example
   :end-before: # end tvmscript example

``@T.function`` parses the definition into a ``tirx.Function``. The future
import postpones annotation evaluation until the parser reads the definition,
so ``n`` in the tensor annotation can resolve to the first IR parameter. The
tensor's shape and the loop bound refer to that same variable, keeping ``2*n``
symbolic in both places. The loops and indexing construct IR; the tensor
computation runs when the compiled function is called.

Script helpers let generated functions reuse parts of a computation.
An ``@T.inline`` helper can return an expression, while ``@T.macro`` inserts
statements into the function being built. These helpers separate the element
addition from the tensor write:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin script helpers
   :end-before: # end script helpers

``increment`` constructs the addition, and ``increment_at`` writes its result
back to the tensor. ``ir.const(1, value.ty)`` gives the constant the element's
type, so the helpers also work with other element types.

An ordinary Python function can choose a shape and dtype, construct a
``@T.function`` definition, and return the resulting ``tirx.Function``.
Here, ``make_increment`` builds a function with the requested column count
and element type:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin program factory
   :end-before: # end program factory

``width`` and ``dtype`` are Python values read while the factory runs. ``wide``
has a fixed second dimension of 128 and float32 elements; ``narrow`` has a second
dimension of 64 and float16 elements. Their first dimension remains ``2*n``,
where ``n`` is an IR parameter supplied when the compiled function is called.
The width check and function construction run while ``make_increment``
executes; the generated loops describe the later tensor computation.
Generating IR with Python logic in this way is metaprogramming.

Printing lets you inspect what this Python code generated:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin print generated function
   :end-before: # end print generated function

``script()`` returns source text. The printed ``wide`` function contains these
loops, using the selected width and element type:

.. code-block:: text

   for i in range(T.int64(0), T.int64(2) * n):
       for j in range(T.int64(0), T.int64(128)):
           A[i, j] = A[i, j] + T.float32(1.0)

The helper calls have expanded into an addition and a tensor write. The source
also declares ``n`` with ``T.dynamic``, which creates a symbolic ``ir.Var`` for
reuse in the signature and tensor shape. You can also print the result of a
pass or smaller parts of a program:
``wide.params[1].ty.script()`` renders its tensor type, and
``wide.body[0].script()`` renders its outer loop.

For interactive inspection, ``wide.show()`` displays highlighted source.
See :meth:`~tvm_tirx.tirx.Function.show` for options such as line numbers and
underlining IR objects.

``from_source`` parses a source string into a new IR object. Names used in
the text must be supplied by its imports or the ``extra_vars`` dictionary.
The printed source refers to ``T``, but its import is a comment by default,
so this example supplies ``T`` through ``extra_vars``:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin printing and parsing
   :end-before: # end printing and parsing

The structural comparison checks that the reconstructed program has the same
operations, types, and variable bindings. Its printed syntax can differ from
the original: for example, ``T.grid`` becomes nested loops.

The same process works for modules. ``@I.ir_module`` turns a class containing
function definitions into an ``ir.IRModule``. Here it collects both generated
variants, which can then be printed and parsed together:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin module round trip
   :end-before: # end module round trip

To include import statements in the printed text, set ``ir.comment_imports``
to False. The parser can then resolve ``I`` and ``T`` from those imports,
without ``extra_vars``. This is useful when saving the source for another
process:

.. literalinclude:: examples/printing_and_parsing.py
   :language: python
   :start-after: # begin standalone source
   :end-before: # end standalone source

Download the :download:`complete example <examples/printing_and_parsing.py>`
for runnable code. See the :doc:`parser </reference/python/script.parser>` and
:doc:`printer </reference/python/script.printer>` API references for details.
