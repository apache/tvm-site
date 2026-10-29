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

In this chapter, we will introduce printing and parsing with TVMScript,
TIRx's Python-based syntax for tensor programs.
The printer turns IR objects into readable TVMScript code, and the parser
constructs IR objects from that code. We will use these interfaces to
inspect the add-one function, save and load programs, and examine a
compiler pass's changes.

Printing IR
-----------

We will use a CPU function that adds one to each tensor element. The
TVMScript decorator ``@T.function`` parses its definition into a
``tirx.Function`` IR object:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

The ``script()`` method returns the function's TVMScript source:

.. code-block:: python

   source = add_one.script()
   print(source)

The output makes the symbolic dimension and constant's type explicit:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + T.float32(1.0)

The original code declares ``n`` with ``T.dynamic``. The printer places
that declaration in ``[n]`` after the function name, so the printed
definition includes the symbol needed by its tensor shapes and loop
bound. ``T.float32(1.0)`` records the float32 constant used by the addition.
The printed form represents the same IR as the original definition.

The same method works on modules and smaller IR objects. For example,
``add_one.params[0].ty.script()`` returns ``T.Tensor((n,), T.float32)``
for the input tensor type.

.. _parsing-source:

Parsing TVMScript
-----------------

The :doc:`TVMScript parser </reference/python/script.parser>` provides
``from_source`` to construct IR from a string containing TVMScript code.
We can use it with the add-one definition:

.. code-block:: python

   from tvm_tirx.script import from_source

   source_text = """
   from tvm_tirx.script import tirx as T

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + T.float32(1.0)
   """
   parsed_function = from_source(source_text)
   print(type(parsed_function).__name__)
   print(parsed_function.script())

The output is:

.. code-block:: text

   Function
   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + T.float32(1.0)

``parsed_function`` is a ``tirx.Function`` containing the parameter
variables, loop, and tensor store. It can be inspected or compiled
through the same interfaces as ``add_one``.

The import inside ``source_text`` tells the parser what ``T`` refers to.
Our first call to ``add_one.script()`` returned ``source`` with that
import commented out. To parse this string, we can bind the name ``T``
to the existing namespace through ``extra_vars``:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   restored_function = from_source(source, extra_vars={"T": T})
   ir.assert_structural_equal(add_one, restored_function)

The printed ``[n]`` declares the symbolic dimension, so it needs no
external binding.

Saving and Loading a Module
---------------------------

An IR module can be saved as TVMScript for later inspection or reuse.
We can store the CPU function in a module and write its source to a file.
The file can include its own imports, so a parser can read it in another
process. ``ir.comment_imports=False`` enables those imports:

.. code-block:: python

   from pathlib import Path

   module = ir.IRModule({"add_one": add_one})
   module_source = module.script(extra_config={"ir.comment_imports": False})
   output_path = Path("add_one_module.py")
   output_path.write_text(module_source)

We can read the saved source and reconstruct the module:

.. code-block:: python

   restored_module = from_source(output_path.read_text())
   ir.assert_structural_equal(module, restored_module)

The parser resolves the namespace names from the imports, so this call
needs no ``extra_vars``. ``restored_module`` is an ``ir.IRModule``
with the same function definitions as ``module``.

Inspecting IR
-------------

The printer can also help us locate a selected computation in the source
or see how a compiler pass changes the program.

.. _highlighting-nodes:

Highlighting IR Objects
~~~~~~~~~~~~~~~~~~~~~~~

An analysis may need to identify an expression in a diagnostic. We can
reach the addition in ``B[i] = A[i] + 1`` through the function's body,
then ask the printer to underline it and attach a message about its type:

.. code-block:: python

   loop = add_one.body[0]
   store = loop.body[0]
   addition = store.value
   marked_source = add_one.script(
       obj_to_underline=[addition],
       obj_to_annotate={addition: "result dtype: float32"},
       print_line_numbers=True,
       num_context_lines=1,
   )
   print(marked_source)

The marked lines include:

.. code-block:: text

   7     for i in range(n):
   8         B[i] = A[i] + T.float32(1.0)  # result dtype: float32
                    ^^^^^^^^^^^^^^^^^^^^^

``obj_to_underline`` selects the expression to mark, and ``obj_to_annotate``
attaches the message. The line numbers refer to this printed source.
``num_context_lines=1`` keeps one surrounding line, so the diagnostic
shows the loop containing the addition.

For interactive inspection,
:meth:`add_one.show() <tvm_tirx.tirx.Function.show>` displays highlighted
source directly and accepts the same options.

Inspecting a Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The printer also helps us check what a compiler pass changes. Consider
this version of add-one with a bounds check inside the loop:

.. code-block:: python

   @T.function
   def guarded_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           if i < n:
               B[i] = A[i] + 1

The loop already guarantees ``i < n``, so this extra branch is redundant.
The ``StmtSimplify`` pass can remove it. We can print the module before
and after applying that pass:

.. code-block:: python

   from tvm_tirx import ir, tirx

   before_simplify = ir.IRModule({"guarded_add_one": guarded_add_one})
   print(before_simplify.script())
   after_simplify = tirx.transform.StmtSimplify()(before_simplify)
   print(after_simplify.script())

The second output contains the simplified loop:

.. code-block:: python

   for i in range(n):
       B[i] = A[i] + T.float32(1.0)

Building Program Variants
-------------------------

A source generator may produce functions with the same structure but
different constants. ``from_source`` can parse the same template with
different ``extra_vars`` to build these variants. This template replaces
the add-one constant with ``increment``:

.. code-block:: python

   variant_source = """
   @T.function
   def add_constant[n](A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + increment
   """

The same text can produce versions that add one or two:

.. code-block:: python

   add_one_variant = from_source(
       variant_source, extra_vars={"T": T, "increment": 1.0}
   )
   add_two_variant = from_source(
       variant_source, extra_vars={"T": T, "increment": 2.0}
   )
   print(add_two_variant.script())

The parser turns the ``increment`` value passed through ``extra_vars``
into a constant. The addition in ``add_two_variant`` is therefore
``A[i] + T.float32(2.0)``; both functions still take only ``A`` and ``B``
as parameters.

.. _user-ir-source-locations:

Source Locations
----------------

IR nodes can carry source locations to help diagnostics point back to
the original code. The ``loc`` field stores this information, and
:class:`ir.SourceLoc <tvm_tirx.ir.SourceLoc>` records the source file
and range.

The parameter ``A`` is declared on this line of the CPU program:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: def add_one(
   :end-at: def add_one(
   :lineno-match:

The parameter's ``loc`` points back to this declaration. We can read
``start_line`` to report its line number:

.. code-block:: python

   parameter = add_one.params[0]
   location = parameter.loc
   line = location.start_line
   print(f"Parameter {parameter.name} is declared on line {line}.")

The output is:

.. code-block:: text

   Parameter A is declared on line 29.

``source_name.name`` also records the path to the CPU program's source
file:

.. code-block:: text

   parameter: ir.Var             A
   └── loc: ir.SourceLoc
       ├── source_name: ir.SourceName
       │   └── name              path to the source file
       └── start_line            line of A's declaration

Source locations are also useful when parsing fails. The following
source is missing the type annotation for parameter ``A``. We give the
text a filename so the diagnostic can identify it:

.. code-block:: python

   invalid_source = """@T.function
   def invalid_add_one(A):
       T.evaluate(0)
   """
   try:
       from_source(
           invalid_source,
           extra_vars={"T": T},
           filename="generated_add_one.py",
       )
   except SyntaxError as error:
       print(error)

The output is:

.. code-block:: text

   Parameter 'A' requires an annotation (generated_add_one.py, line 2)

The message identifies the supplied source text and the line with the
invalid parameter. Successfully parsed nodes record the filename and
their positions in ``loc``.

.. _tvmscript-region-syntax:

Region Syntax
-------------

The parser also determines which statements belong to a region. A CUDA
function uses ``T.device_entry`` to mark the statements that execute on
the GPU. In the following function, the region starts at that line:

.. literalinclude:: ../../examples/quick_start/add_one_cuda.py
   :language: python
   :start-at: @T.function
   :end-at:         B[i] = A[i] + 1

For this form, TVMScript takes all following statements in the same
enclosing block as the region's body. The enclosing block here is the
function body: the region extends from ``T.device_entry(...)`` to the
end of the function. It contains the assignment to ``i`` and the entire
``if`` branch, even though those statements have the same indentation
as ``T.device_entry(...)``.

An explicit ``with`` block shows the same region through indentation:

.. code-block:: python

   @T.function
   def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       with T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256)):
           i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
           if i < n:
               B[i] = A[i] + 1

Both forms produce the same ``ir.RegionStmt``. The :doc:`statements`
chapter explains the IR structure inside that region.
