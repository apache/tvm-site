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

Every TIRx program, at any stage of compilation, can be written in
TVMScript, an embedded Python syntax. The printer turns IR into TVMScript,
and the parser turns TVMScript back into IR. This lets one syntax serve
three roles:

- **DSL frontend**: developers and agents write tensor programs directly
  in TVMScript.
- **IR constructor for tests**: a test writes its input and expected
  output as TVMScript, then compares them structurally with the result of
  a pass.
- **Diagnostic printer**: when compilation fails, the printer shows the
  program as it stands at that stage, in the same syntax the user wrote.

.. note::

   The runnable code in this chapter is under
   `examples/ir_walkthrough <https://github.com/apache/tvm-tirx/tree/main/examples/ir_walkthrough>`_
   in the repository.

Printing and Parsing
--------------------

We will use a CPU function that adds one to each tensor element:

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-after: # [function.begin]
   :end-before: # [function.end]

The ``script()`` method prints any IR object as TVMScript:

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-after: # [print.begin]
   :end-before: # [print.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + T.float32(1.0)

The printer writes out everything the IR records. The constant ``1``
becomes ``T.float32(1.0)``, its value with the type the addition uses. The
symbolic dimension ``n`` is declared in ``[n]`` after the function name,
using the type-parameter syntax introduced in Python 3.12. Each name in
the brackets declares a symbolic variable as if by ``T.dynamic``, but
scoped to this function, so the printed function is self-contained. On
Python 3.12 and newer, we can also write this form ourselves. A symbol with a dtype other than ``int32`` carries an
annotation, as in ``[m: T.int64]``. The same method works on smaller
objects. For example, ``add_one.params[0].ty.script()`` returns
``T.Tensor((n,), T.float32)``.

A module prints the same way, as a class marked with ``@T.ir_module`` with
one ``@T.function`` per member:

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-after: # [print_module.begin]
   :end-before: # [print_module.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   n = T.dynamic("n", dtype="int32")
   @T.ir_module
   class Module:
       @T.function
       def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
           for i in range(n):
               B[i] = A[i] + T.float32(1.0)

Functions in a module can share symbols, so ``n`` is declared once before
the class instead of in ``[n]``.

The printed output is TVMScript that ``@T.function`` and ``@T.ir_module``
accept. On Python 3.12 and newer, a printed function placed in a Python
file, here ``add_one_printed.py`` with its import enabled, builds the same
IR again:

.. literalinclude:: ../../examples/ir_walkthrough/add_one_printed.py
   :language: python
   :start-after: # [printed.begin]
   :end-before: # [printed.end]

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-after: # [parse.begin]
   :end-before: # [parse.end]

Printing and then parsing gives back the same IR. This is what lets a test
write its expected IR as TVMScript and compare it with the output of a
pass. To store or transfer IR, use ``ir.save_json`` and ``ir.load_json``,
which serialize the IR objects directly.

How the Parser Works
--------------------

The parser is a thin layer over an IR builder, the Python API that
constructs IR. The builder works with frames: entering a function, loop,
or region opens a frame, and the statements built inside it become its
body. Parsing is transpilation: the parser reads the Python syntax tree of
a TVMScript definition, rewrites each syntax form into the matching
builder call, then compiles and runs the result as an ordinary Python
function in the definition's original scope.

For example, this definition:

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-at: @T.function
   :end-at: B[i] = A[i] + 1

becomes this builder program:

.. code-block:: python

   with _I0.IRBuilder() as _builder0:
       with T.function_() as _fn0:

           def _declare0():
               T.func_name_('add_one')
               A = T.arg_('A', T.Tensor((n,), T.float32))
               B = T.arg_('B', T.Tensor((n,), T.float32))
           _declare0()

           def _build0():
               A, B = _fn0.params
               with T.for_(T.range_(n), names='i') as i:
                   T.setitem_(value=A[i] + 1, target=B, key=i)
           _build0()
   _result0 = _fn0.function
   _result0.__name__ = 'add_one'
   T.check_well_formed_(_result0)

Each syntax form maps to builder calls: the definition becomes a
``function_`` frame, parameter annotations become ``arg_`` calls with
their types, the loop becomes a ``for_`` frame,
and the store becomes ``setitem_``. The signature is declared before the
body is built, and the finished function is checked for well-formedness.
Expressions such as ``A[i] + 1`` stay as written: ``A`` is the IR
variable returned by ``arg_``, so Python's ``[]`` and ``+`` build the IR
expression directly.
Because the program runs as normal Python, ``n`` and other names come from
the surrounding scope. The real output also passes source positions to
each call, which is how parsed IR gets its ``loc``.

Because the builder program runs as ordinary Python, the parser supports
metaprogramming out of the box. A TVMScript function can read any Python
value from its enclosing scope, and a Python function can build a new
TVMScript function on each call:

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-after: # [metaprogramming.begin]
   :end-before: # [metaprogramming.end]

``add_two`` adds ``2`` to ``int32`` tensors, and ``add_three`` adds
``T.float32(3.0)`` to ``float32`` tensors. The parser needs no template
language for this: ``value`` and ``dtype`` are evaluated like any other
Python name when the builder program runs.

The builder is also a public API, so we can construct the same function
directly, without the parser:

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-after: # [builder.begin]
   :end-before: # [builder.end]

``builder.get()`` returns a ``tirx.Function`` that is structurally equal
to ``add_one``. Hand-written builder code uses the user-facing API, such as
``T.serial`` and ``T.assign``, rather than the protocol hooks the parser
emits. Writing builder code directly is sometimes helpful because
it stays unambiguously in the Python domain. A plain Python ``for`` or
``if`` runs while the IR is built, and IR control flow comes only from
explicit frames such as ``T.serial`` and ``T.if_`` with ``T.then_``. Developers and agents
therefore know exactly what each line executes. To record source
locations, wrap the construction in ``builder.line_trace_scope()``, and
each statement then records its source line in ``loc``.

.. _tvmscript-region-syntax:

Regions are built with frames in the same way. A region operator such as
``T.device_entry`` opens a frame, and the statements inside its ``with``
block become the region's body:

.. code-block:: python

   with T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256)):
       i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
       if i < n:
           B[i] = A[i] + 1

As a shorthand, a region call written as a plain statement opens the same
frame for the rest of its enclosing block, so ``T.device_entry(...)``
followed by the same statements at the same indentation produces the same
``ir.RegionStmt``.

We value **predictability** over syntactic sugar. Each syntax form has one
fixed meaning: an annotation is a type, an assignment binds a variable, a
subscript store writes a tensor, and a call is an ordinary Python call.
Types such as ``T.Tensor`` and operators such as ``T.cuda.thread_idx`` are
plain functions that return IR, so the parser has no rules specific to
them.

Extending the Syntax
--------------------

Because the parser only rewrites syntax into builder calls, extending
TVMScript rarely touches the parser. Operators and types registered by an
extension package are plain functions that return IR, so they work in
TVMScript as soon as the package is imported. For example, an operator
registered under an extension namespace ``my_ext`` is written
``my_ext.cmul(a, b)``, like any built-in operator.

As an advanced topic, a new language variant can also be added if
necessary. The builder hooks that the parser emits for each syntax form,
such as ``bind_``, ``setitem_``, and ``for_``, form the *parser protocol*,
defined in ``tvm_tirx.script.ir_builder.parser_protocol``. A variant
implements these hooks on its own namespace and registers it with the
parser, and the same parser then builds its IR.

Diagnostics
-----------

When compilation fails, the most useful view is usually the program as it
stands at that stage. Calling ``script()`` on the module at that point
shows it in the same syntax used to write it.

The printer can also point at specific objects. ``obj_to_underline`` marks
objects in the output, and ``obj_to_annotate`` attaches a message to them:

.. literalinclude:: ../../examples/ir_walkthrough/printing_and_parsing.py
   :language: python
   :start-after: # [highlight.begin]
   :end-before: # [highlight.end]

The output ends with:

.. code-block:: text

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + T.float32(1.0)  # result dtype: float32
                  ^^^^^^^^^^^^^^^^^^^^^

``show()`` accepts the same options and displays the result with syntax
highlighting.
