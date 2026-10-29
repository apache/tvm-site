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
the Python-based syntax used to describe tensor programs in Quick Start.
The printer turns IR objects into readable TVMScript code, and the parser
constructs IR objects from that code. We will use these interfaces to
inspect the add-one function, save and load programs, and examine a
compiler pass's changes.

Printing IR
-----------

In the CPU example, the TVMScript decorator ``@T.function`` parses the
function definition into a ``tirx.Function`` IR object:

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
The earlier ``source`` returned by ``add_one.script()`` has this import
commented out. In that case, ``extra_vars`` supplies the existing ``T``
namespace:

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
To let that file supply its own namespaces when parsed, we include the
imports with ``ir.comment_imports=False``:

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

The imports in the file supply the namespaces needed by the parser, so
this call needs no ``extra_vars``. ``restored_module`` is an ``ir.IRModule``
with the same function definitions as ``module``.

Inspecting IR
-------------

The printer can also help us locate a selected computation in the source
or see how a compiler pass changes the program.

.. _highlighting-nodes:

Highlighting IR Objects
~~~~~~~~~~~~~~~~~~~~~~~

The function's body lets us reach the addition in ``B[i] = A[i] + 1``.
To find this expression in the printed program, we can ask the printer to
underline it:

.. code-block:: python

   loop = add_one.body[0]
   store = loop.body[0]
   addition = store.value
   marked_source = add_one.script(obj_to_underline=[addition])
   print(marked_source)

The marked lines include:

.. code-block:: text

   B[i] = A[i] + T.float32(1.0)
          ^^^^^^^^^^^^^^^^^^^^^

``obj_to_underline`` selects the IR objects to mark. For a diagnostic,
``print_line_numbers=True`` adds line numbers, ``obj_to_annotate`` can
attach messages to selected objects, and ``num_context_lines`` limits the
surrounding source.

For interactive inspection,
:meth:`add_one.show() <tvm_tirx.tirx.Function.show>` displays highlighted
source directly and accepts the same options.

Inspecting a Transformation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The printer also helps us check what a compiler pass changes. We can
extend the CPU example with a bounds check inside the loop:

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

For functions that differ only in a constant, ``from_source`` can parse
the same template with different ``extra_vars``. This template replaces
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

The parser turns the ``increment`` value supplied through ``extra_vars``
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

The ``filename`` argument labels generated source in parser diagnostics.
The following example removes the required type annotation from ``A``
in the template:

.. code-block:: python

   invalid_source = variant_source.replace("A: T.Tensor((n,), T.float32)", "A")
   try:
       from_source(
           invalid_source,
           extra_vars={"T": T, "increment": 1.0},
           filename="generated_add_constant.py",
       )
   except SyntaxError as error:
       print(error)

The output is:

.. code-block:: text

   Parameter 'A' requires an annotation (generated_add_constant.py, line 3)

The program comes from ``invalid_source``; ``filename`` supplies the
name shown in the message, and successfully parsed nodes record it in
their ``loc`` fields.

.. _tvmscript-region-syntax:

Region Syntax
-------------

The parser also determines which statements belong to a region. In the
GPU example, TVMScript recognizes ``T.device_entry`` as a region
construct. Its standalone form starts the device region:

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
       with T.device_entry(launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,))):
           i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
           if i < n:
               B[i] = A[i] + 1

Both forms produce the same ``ir.RegionStmt``. The :doc:`statements`
chapter explains the IR structure inside that region.
