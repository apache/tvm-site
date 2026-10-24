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

.. _statements-and-control-flow:

Statements
==========

Statements describe variable bindings, tensor writes, and control flow.
Extensions can construct and rewrite program bodies with the existing
statement nodes. Custom operators can use ``ir.RegionStmt`` to define how
a group of statements executes.

.. _statements-and-sequences:

Statement Nodes
---------------

``ir.Stmt`` is the base class for all statement nodes, including sequences,
branches, and loops. It defines the following field:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``loc``
     - ``ir.Location``
     - Source information for diagnostics. Defaults to ``UnknownLoc`` and does
       not affect structural comparison.

Subclasses inherit ``loc`` and add fields for operands or nested statements.
The shared statement classes are:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Represents
     - Additional fields
   * - ``ir.Bind``
     - A variable definition
     - ``var``, ``value``
   * - ``ir.TensorStore``
     - A tensor element write; the destination type must support stores
     - ``dest``, ``indices``, ``value``
   * - ``ir.Evaluate``
     - Expression evaluation with its result discarded
     - ``value``
   * - ``ir.AssertStmt``
     - A program assertion
     - ``condition``, ``error_kind``, ``message_parts``
   * - ``ir.Return``
     - A function return
     - ``value``
   * - ``ir.Break``
     - Exit the current loop
     - None
   * - ``ir.Continue``
     - Start the next loop iteration
     - None
   * - ``ir.SeqStmt``
     - A sequence of statements in execution order
     - ``seq``
   * - ``ir.If``
     - Select a branch
     - ``condition``, ``then_case``, ``else_case``
   * - ``ir.For``
     - Repeat a body over an integer range
     - ``var``, ``begin``, ``end``, ``step``, ``kind``, ``body``, ``annotations``
   * - ``ir.While``
     - Repeat while a condition is true
     - ``condition``, ``body``
   * - ``ir.RegionStmt``
     - An operator-defined statement with a nested body
     - ``op``, ``args``, ``attrs``, ``body_params``, ``body``, ``result_vars``

Statement Sequences
-------------------

A return that uses a bound variable must follow its definition.
``ir.SeqStmt`` records the execution order of such statements.
For example, consider a function that adds one to its input:

.. code-block:: python

   def increment(x):
       result = x + 1
       return result

The following code constructs that function body as IR objects:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin statements
   :end-before: # end statements
   :prepend: from tvm_tirx import ir, tirx

The Python variable ``added`` holds the expression for ``x + 1``. ``ir.Var``
creates the IR variable ``result``, and ``ir.Bind(result, added)`` defines it
with that expression as its value. ``added.ty`` gives ``result`` the addition's
``int32`` type. ``ir.Return(result)`` refers to the same variable, and
``SeqStmt`` places the definition before the return.

Nested sequences are flattened, and a sequence introduces no scope of its
own. Variables defined by ``Bind`` are available to subsequent statements in
the enclosing scope.

Control Flow and Scope
----------------------

Control-flow statement nodes include ``ir.If`` for branching, ``ir.For`` and
``ir.While`` for loops, ``ir.Break`` and ``ir.Continue`` for loop control, and
``ir.Return`` for returning from a function.

The statements inside a branch or loop form its body. Branch bodies are
stored in ``If.then_case`` and ``If.else_case``, while loop bodies are stored
in ``For.body`` and ``While.body``. Each body is a ``SeqStmt`` and can contain
further branches and loops.

For example, this branch returns the incremented value only for positive
inputs. Both bodies are normalized to ``SeqStmt``:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin branch
   :end-before: # end branch

``If`` and ``While`` use boolean conditions. ``If`` requires an explicit
``else_case`` argument; pass None when there is no else branch.

A loop instead repeats a body. This loop evaluates each index from zero through
three:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin loop
   :end-before: # end loop

``For`` takes an exclusive ``end`` and explicit ``step`` and ``kind`` arguments.
Its variable, bounds, and step must have matching scalar integer types after
conversion.

The ``kind`` argument controls how loop iterations are executed.
``ForKind.DEFAULT`` executes them in order, one after another.
See :class:`~tvm_tirx.ir.ForKind` for the available execution kinds.

``Break`` exits a loop, and ``Continue`` starts its next iteration.

Variable scope determines where a definition can be used. The loop defines ``i``
inside its body; it cannot be used after the loop. Branch-local definitions
similarly stay inside their branch. Each variable object has one definition,
and its uses refer to that object.

``ir.RegionStmt`` lets an operator define how a nested body is executed without
introducing a new statement class. Its ``op`` field refers to the registered
``ir.Op`` that defines this behavior. Inputs in ``args`` and ``attrs``
are evaluated in the enclosing scope. ``body_params`` defines variables
visible within ``body``;
``result_vars`` defines variables available to subsequent statements in the
enclosing sequence. See :class:`~tvm_tirx.ir.RegionStmt` for the operator
contract.

To check these definition and scope rules, the body needs the context of a
complete function or module. :doc:`functions_and_modules` supplies that context;
``tirx.analysis.check_well_formed`` then checks variable and tensor uses and loop
control. Invalid uses raise ``RuntimeError`` by default; ``assert_mode=False``
returns False instead.

The :download:`statements_and_functions.py <examples/statements_and_functions.py>`
example completes and runs the increment function. See
:doc:`/reference/python/ir` for statement constructors and
:doc:`/reference/python/tirx.analysis` for verification APIs.
