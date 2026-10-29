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

In this chapter, we will introduce statements, which describe the actions
a function performs and their execution order. A statement can bind an
intermediate value to a variable, write a result to a tensor, or control
execution through a branch or loop. These statements form the function's
body, which compiler passes can inspect and rewrite through the IR
interfaces.

We will start with this CPU function, which adds one to every element
of ``A`` and writes the results to ``B``:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

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

A function body often contains several statements that run one after
another. ``ir.SeqStmt`` stores these statements in execution order in
its ``seq`` field. The statements inside a loop or branch are also
stored in a ``SeqStmt``.

The loop in ``add_one`` writes one output element in each iteration:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at:     for i in range(n):
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

The indented assignment ``B[i] = A[i] + 1`` becomes an
``ir.TensorStore`` statement describing a write to a tensor element.
The loop body is a ``SeqStmt`` containing this statement. We can also
construct this body through the IR API, using the parameter variables
from ``add_one``:

.. code-block:: python

   from tvm_tirx import ir

   A, B = add_one.params
   n = A.ty.shape[0]
   i = ir.Var("i", "int32")
   store = ir.TensorStore(B, [i], A[i] + 1)
   body = ir.SeqStmt([store])

``ir.TensorStore`` takes the destination tensor ``B``, the index list
``[i]``, and the value expression ``A[i] + 1``. These arguments populate
its ``dest``, ``indices``, and ``value`` fields. ``ir.SeqStmt([store])``
holds this store statement as its first entry:

.. code-block:: text

   body: ir.SeqStmt
   └── seq[0]: ir.TensorStore                B[i] = A[i] + 1
       ├── dest: ir.Var                     B
       ├── indices[0]: ir.Var               i
       └── value: ir.prim.Add                A[i] + 1

When the compiled function runs, the store computes ``A[i] + 1`` and
writes the result to element ``i`` of ``B``. We can retrieve this
statement with ``body[0]``, which is shorthand for ``body.seq[0]``.

A small extension of the CPU loop names the addition's result before
writing it to the output:

.. code-block:: python

   for i in range(n):
       value = A[i] + 1
       B[i] = value

The assignment to ``value`` becomes an ``ir.Bind`` statement: its
``var`` field holds the variable, and its ``value`` field holds the
addition expression. The following ``TensorStore`` uses that variable
as the value to write. We can construct these two statements and put
them in a sequence:

.. code-block:: python

   value = ir.Var("value", "float32")
   binding = ir.Bind(value, A[i] + 1)
   store_from_value = ir.TensorStore(B, [i], value)
   sequence = ir.SeqStmt([binding, store_from_value])

.. code-block:: text

   sequence: ir.SeqStmt
   ├── seq[0]: ir.Bind                      value = A[i] + 1
   │   ├── var: ir.Var                      value
   │   └── value: ir.prim.Add               A[i] + 1
   └── seq[1]: ir.TensorStore               B[i] = value
       └── value → same ir.Var              value

When the function runs, the binding computes the addition and defines
``value``, then the store writes this value to ``B[i]``.
The next section places the original single-store ``body`` inside the
CPU loop.

.. _control-flow-and-scope:

Control Flow
------------

Control-flow statements determine which parts of a program execute and
how often. A loop repeats a body of statements, while a branch selects
which body runs.

For loops
~~~~~~~~~

``add_one`` visits every element with a loop:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at:     for i in range(n):
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

The loop becomes an ``ir.For``. Its ``var`` field identifies the loop
variable ``i``, and its ``body`` field holds the ``SeqStmt`` containing
the tensor store. We can construct this loop using ``i`` and ``body``
from the previous section:

.. code-block:: python

   loop = ir.For(i, 0, n, 1, ir.ForKind.DEFAULT, body)

The arguments ``0``, ``n``, and ``1`` set the ``begin``, ``end``, and
``step`` fields. ``ForKind.DEFAULT`` selects execution in iteration
order. On each iteration, the loop executes the statements in ``body``:

.. code-block:: text

   loop: ir.For
   ├── var: ir.Var                           i
   ├── begin: ir.IntImm                      0
   ├── end: ir.Var                           n
   ├── step: ir.IntImm                       1
   ├── kind: ir.ForKind.DEFAULT
   └── body: ir.SeqStmt
       └── seq[0]: ir.TensorStore            B[i] = A[i] + 1

The end bound is exclusive, so ``i`` runs from 0 through ``n - 1``.
Each iteration executes the store in ``body`` with that value of ``i``.
The loop defines ``i`` for use inside ``For.body``, which is its scope.
The ``For`` establishes this scope, and its ``SeqStmt`` body records
the order of the statements within it. The
:class:`~tvm_tirx.ir.ForKind` API reference describes the other execution
kinds.

Branches
~~~~~~~~

A branch can choose between two results. This CPU function applies
ReLU: positive input values pass through, and all other values become
zero:

.. code-block:: python

   @T.function
   def relu(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           if A[i] > 0:
               B[i] = A[i]
           else:
               B[i] = 0.0

TVMScript represents the branch as an ``ir.If``. Its ``condition`` is
the boolean comparison ``A[i] > 0``. Its ``then_case`` and ``else_case``
each hold a ``SeqStmt`` containing one store. Only the sequence selected
by the condition runs for each element.

A bounds check uses the same kind of statement. A GPU version of add-one
distributes the elements across threads, and each thread checks whether
it has a valid element to process. These statements belong inside its
function body:

.. literalinclude:: ../../examples/quick_start/add_one_cuda.py
   :language: python
   :start-at:     T.device_entry(
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

The ``if i < n`` becomes an ``ir.If``. Its ``condition`` field holds
the comparison, an ``ir.prim.LT`` expression. Its ``then_case`` field
holds a ``SeqStmt`` containing the tensor store:

.. code-block:: text

   ir.If
   ├── condition: ir.prim.LT                 i < n
   ├── then_case: ir.SeqStmt
   │   └── seq[0]: ir.TensorStore            B[i] = A[i] + 1
   └── else_case: None

The condition produces a boolean. When it is true, the thread executes
the store in ``then_case``. An else branch has its own ``SeqStmt`` in
``else_case``. This example has no else branch, so ``else_case`` is
``None`` and threads whose indices are outside the tensor perform no
tensor access. These branch bodies can also contain further loops and
branches.

The ``T.device_entry(...)`` line groups the statements that follow it in
the same function body into a device region. An
:class:`ir.RegionStmt <tvm_tirx.ir.RegionStmt>` represents this region in
the IR. Its ``op`` identifies ``tirx.device_entry``, and its ``body`` is
a ``SeqStmt`` containing the index definition followed by the branch:

.. code-block:: text

   ir.RegionStmt
   ├── op → ir.Op                          tirx.device_entry
   └── body: ir.SeqStmt
       ├── seq[0]: ir.Bind                 i = ...
       └── seq[1]: ir.If                   if i < n
           └── then_case: ir.SeqStmt
               └── seq[0]: ir.TensorStore  B[i] = A[i] + 1

The ``Bind`` defines ``i`` before the ``If`` uses it. The variable is
available to the condition and tensor store within the region. The
:ref:`Printing and Parsing <tvmscript-region-syntax>` chapter explains
the grouping rule for this syntax and the equivalent ``with`` form.

While loops
~~~~~~~~~~~

A ``while`` loop tests a condition before each iteration. We can express
the same add-one computation with an index that advances until it reaches
the tensor length:

.. code-block:: python

   @T.function
   def add_one_while(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       i: T.int32 = 0
       while i < n:
           B[i] = A[i] + 1
           i += 1

``i: T.int32 = 0`` creates a mutable scalar, so ``i += 1`` updates the
index for the next test. The loop stops when ``i < n`` becomes false.
Without that update, the loop would keep processing the same element.

TVMScript represents the loop as an ``ir.While``. Its ``condition``
holds the comparison, and its ``body`` contains both the output store
and the index update:

.. code-block:: text

   ir.While
   ├── condition: ir.prim.LT                 i < n
   └── body: ir.SeqStmt
       ├── seq[0]: ir.TensorStore            B[i] = A[i] + 1
       └── seq[1]: ir.TensorStore            i += 1

The mutable scalar uses a one-element local tensor in IR. The second
store writes its next value, which the condition reads before the next
iteration.

Loop exits
~~~~~~~~~~

Some loops can finish before reaching their bound. This function finds
the index of the first positive integer. It leaves ``B[0]`` at ``-1``
if no such element exists:

.. code-block:: python

   @T.function
   def first_positive(A: T.Tensor((n,), T.int32), B: T.Tensor((1,), T.int32)):
       B[0] = -1
       for i in range(n):
           if A[i] <= 0:
               continue
           B[0] = i
           break

``continue`` starts the next iteration when ``A[i]`` is nonpositive. If
the element is positive, the store records its index and ``break`` ends
the loop. These statements become
``ir.Continue`` and ``ir.Break`` inside their enclosing loop; neither
has operands or a nested body.

Assertions and returns
~~~~~~~~~~~~~~~~~~~~~~

A function can check a precondition before computing its result. This
function adds one to the first input element and returns the value.
The assertion ensures that the tensor has an element to read:

.. code-block:: python

   @T.function
   def first_plus_one(A: T.Tensor((n,), T.float32)) -> T.float32:
       assert n > 0, "the input must contain at least one element"
       return A[0] + 1

The ``assert`` becomes an ``ir.AssertStmt``. Its ``condition`` holds
``n > 0``, its ``error_kind`` names ``RuntimeError``, and its
``message_parts`` contains the error message. An empty input raises that
error before the tensor read. Otherwise, execution reaches the
``ir.Return``, whose ``value`` is the addition expression:

.. code-block:: text

   first_plus_one.body: ir.SeqStmt
   ├── seq[0]: ir.AssertStmt                 assert n > 0, ...
   └── seq[1]: ir.Return
       └── value: ir.prim.Add                A[0] + 1

The return annotation ``-> T.float32`` declares the result type. For an
input starting with ``3.5``, the function returns ``4.5`` directly to its
caller. It needs no output tensor parameter.
