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

In this chapter, we will introduce the statements that organize a
function's execution. Statements can be used to describe value bindings, tensor
writes, and control flow. Statements take several forms, including sequences,
branches, and loops. Sequences record execution order, while branches and
loops determine which statements execute and how often. Compiler passes can
use these structures to construct and transform programs, including bodies
defined by custom operators.

.. _statements-and-sequences:

Statement Classes
-----------------

``ir.Stmt`` is the base class for all statements, from bindings and tensor
writes to sequences, branches, and loops:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Class
     - Represents
     - Fields
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

One of the most common statement types is ``ir.SeqStmt``, which groups
statements in execution order in its ``seq`` field.
A sequence can serve as a function body or as the body of a branch or loop.

In our sigmoid example, each iteration binds the biased input before storing
the sigmoid result. Reusing the tensor variables and indices from
:ref:`user-ir-var` and the helper from :ref:`user-ir-call`, we can build this
sequence:

.. code-block:: python

   biased = ir.Var("biased", "float32")
   biased_value = X[i, j] + bias[j]
   result = sigmoid(biased)

   body = ir.SeqStmt(
       [
           ir.Bind(biased, biased_value),
           ir.TensorStore(Y, [i, j], result),
       ]
   )

``ir.Bind`` defines ``biased`` with the value of ``X[i, j] + bias[j]``.
The following ``ir.TensorStore`` evaluates the sigmoid expression ``result``
using that binding and writes the value to ``Y[i, j]``.
The sequence forms the body of the channel loop, which we examine next.

Control Flow and Scope
----------------------

Control-flow statements determine which parts of a program execute, how
many times they execute, and when execution leaves a loop or function.
TIRx provides ``ir.If`` for branching, ``ir.For`` and ``ir.While`` for loops,
``ir.Break`` and ``ir.Continue`` for loop control, and ``ir.Return`` for
returning from a function.

The statements inside a branch or loop form its body. Branch bodies are
stored in ``If.then_case`` and ``If.else_case``, while loop bodies are stored
in ``For.body`` and ``While.body``. Each body is a ``SeqStmt`` and can contain
further branches and loops.

An ``ir.For`` specifies a loop variable, begin and end bounds, a step, and an
execution kind. The end bound is exclusive. ``ForKind.DEFAULT`` executes
iterations in order; the :class:`~tvm_tirx.ir.ForKind` API reference describes
the other execution kinds.

The program processes one channel at a time. The surrounding loops cover
samples and channels, keeping each tensor access within its bounds:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin loop
   :end-before: # end loop
   :dedent: 4

``batch_loop`` advances ``i`` by one sample. ``channel_loop`` advances ``j``
by one channel, executing ``body`` at each step. Both loops use int64
variables, bounds, and steps. The channel loop stops before ``end``, the
channel count, so ``j`` ranges from 0 to ``channels - 1``.

Scope determines where a definition can be used. ``ir.SeqStmt`` groups
statements without introducing an additional scope. In the example, the
sequence ensures that ``biased`` is bound before the ``TensorStore`` uses it,
while the enclosing channel loop limits its visibility to that loop body.
The loop variables ``i`` and ``j`` are likewise visible only within their
loop bodies.

Some operators introduce their own statement bodies.
:class:`ir.RegionStmt <tvm_tirx.ir.RegionStmt>` represents these using a
registered operator, whose definition specifies the body's parameters.

These scope rules also apply to programs built through the IR API. After
construction, :func:`tirx.analysis.check_well_formed <tvm_tirx.tirx.analysis.check_well_formed>`
can check a function or module for invalid variable and tensor uses and
misplaced loop-control statements.
