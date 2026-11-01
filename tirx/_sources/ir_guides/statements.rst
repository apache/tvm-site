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

This guide introduces **statements**, the IR nodes that describe
actions: binding a value to a variable, writing to a tensor, repeating
work in a loop, or choosing between branches. TIRx provides a generic
set of statements that appear in most high-level kernel programs.
Statements run in order, and they contain expressions: a loop holds
its bounds, and a store holds the value it writes.

A function's ``body`` is a tree of statements. This add-one function
loops over its input and writes each result to ``B``:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + 1

   print(ir.format_tree(add_one.body))

The output is:

.. code-block:: text

   SeqStmt
   └── seq[0]: For(begin=IntImm(value=0), end=n)
       ├── var: i = Var(ty=int32)
       └── body: SeqStmt
           └── seq[0]: TensorStore(dest=B, indices=[i])
               └── value: Add
                   ├── a: TensorLoad(source=A, indices=[i])
                   └── b: FloatImm(value=1.0)

The body is a sequence holding one ``For`` loop. The loop's body is
another sequence holding one ``TensorStore``, which writes the
expression ``A[i] + 1`` to ``B[i]``.

``ir.Stmt`` is the base class of all statements. The following table
lists the statement classes, and the sections below introduce them in
turn, ending with regions, the main place for extensions:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Class
     - Represents
     - Fields
   * - ``ir.SeqStmt``
     - Statements that run in order
     - ``seq``
   * - ``ir.Bind``
     - A variable definition
     - ``var``, ``value``
   * - ``ir.TensorStore``
     - A write to a tensor element
     - ``dest``, ``indices``, ``value``
   * - ``ir.Evaluate``
     - An expression run for its effect
     - ``value``
   * - ``ir.For``
     - A loop over an integer range
     - ``var``, ``begin``, ``end``, ``step``, ``kind``, ``body``,
       ``annotations``
   * - ``ir.While``
     - A loop that runs while a condition holds
     - ``condition``, ``body``
   * - ``ir.If``
     - A choice between two sequences
     - ``condition``, ``then_case``, ``else_case``
   * - ``ir.Continue``
     - Skip to the next iteration of the innermost loop
     - None
   * - ``ir.Break``
     - Leave the innermost loop
     - None
   * - ``ir.AssertStmt``
     - A run-time check that raises an error
     - ``condition``, ``error_kind``, ``message_parts``
   * - ``ir.Return``
     - A function return
     - ``value``
   * - ``ir.RegionStmt``
     - An operator-defined statement with a nested body
     - ``op``, ``args``, ``attrs``, ``body_params``, ``body``,
       ``result_vars``

Sequences, Bindings, and Stores
-------------------------------

``ir.SeqStmt`` holds statements that run one after another, in its
``seq`` field. A function body, a loop body, and each branch of an
``if`` are all sequences.

``ir.Bind`` defines a variable: its ``var`` field holds the new
variable, and its ``value`` field holds the expression that computes
it. ``ir.TensorStore`` writes ``value`` to the element of ``dest``
selected by ``indices``. In this add-one loop, the sum is bound to a
name before it is written:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           value = A[i] + 1
           B[i] = value

   loop = add_one.body[0]
   print(ir.format_tree(loop.body))

``add_one.body[0]`` is shorthand for ``add_one.body.seq[0]``, the first
statement in the sequence. The output is:

.. code-block:: text

   SeqStmt
   ├── seq[0]: Bind
   │   ├── var: value = Var(ty=float32)
   │   └── value: Add
   │       ├── a: TensorLoad(source=A, indices=[i])
   │       └── b: FloatImm(value=1.0)
   └── seq[1]: TensorStore(dest=B, indices=[i])
       └── value: value

The ``Bind`` defines ``value``, and the ``TensorStore`` that follows it
uses it. A variable defined by a statement in a sequence is visible to
the statements after it in the same sequence, including any nested
inside them, but not outside that sequence. Each variable is defined
only once in a function.

``ir.Evaluate`` runs an expression and discards its result. Its most
common use is to hold a ``Call`` made only for its effect.
``T.cuda.cta_sync()`` is such a call: it makes each thread in a GPU
block wait until all threads in the block reach it:

.. code-block:: python

   @T.function
   def sync_block(A: T.Tensor((128,), T.float32)):
       T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=128))
       T.cuda.cta_sync()

   region = sync_block.body[0]
   print(ir.format_tree(region.body[0], hide=["attrs", "ty_args"]))

The output is:

.. code-block:: text

   Evaluate
   └── value: Call(op=tirx.cuda.cta_sync)
       └── args: []

The barrier produces no value; ``Evaluate`` turns the call into a
statement.

Loops
-----

``ir.For`` repeats its ``body`` once for each value of its loop
variable ``var``, from ``begin`` up to but not including ``end``, in
steps of ``step``. The loop defines ``var``, which is visible only
inside ``body``:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + 1

   print(ir.format_tree(add_one.body[0]))

The output is:

.. code-block:: text

   For(begin=IntImm(value=0), end=n)
   ├── var: i = Var(ty=int32)
   └── body: SeqStmt
       └── seq[0]: TensorStore(dest=B, indices=[i])
           └── value: Add
               ├── a: TensorLoad(source=A, indices=[i])
               └── b: FloatImm(value=1.0)

The tree omits ``step`` and ``kind`` because they have their default
values: a step of one, and iterations that run in order. Other values
of ``kind`` (``ir.ForKind``) ask for the iterations to be run in
parallel, vectorized, or unrolled.

``ir.While`` repeats its ``body`` as long as its ``condition`` is
true. The condition is checked before each iteration. This loop halves
a value until it is at most one:

.. code-block:: python

   @T.function
   def halve_until_small(X: T.Tensor((1,), T.float32)):
       while X[0] > 1.0:
           X[0] = X[0] * 0.5

   print(ir.format_tree(halve_until_small.body[0]))

The output is:

.. code-block:: text

   While
   ├── condition: GT
   │   ├── a: TensorLoad(source=X, indices=[IntImm(value=0)])
   │   └── b: FloatImm(value=1.0)
   └── body: SeqStmt
       └── seq[0]: TensorStore(dest=X, indices=[IntImm(value=0)])
           └── value: Mul
               ├── a: TensorLoad(source=X, indices=[IntImm(value=0)])
               └── b: FloatImm(value=0.5)

Branches
--------

``ir.If`` runs ``then_case`` when its ``condition`` is true, and
``else_case`` otherwise. Both cases are sequences, and ``else_case``
is ``None`` when the branch has no ``else``. This loop applies ReLU,
keeping positive values and replacing the rest with zero:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def relu(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           if A[i] > 0:
               B[i] = A[i]
           else:
               B[i] = 0.0

   loop = relu.body[0]
   print(ir.format_tree(loop.body[0]))

The output is:

.. code-block:: text

   If
   ├── condition: GT
   │   ├── a: TensorLoad(source=A, indices=[i])
   │   └── b: FloatImm(value=0.0)
   ├── then_case: SeqStmt
   │   └── seq[0]: TensorStore(dest=B, indices=[i])
   │       └── value: TensorLoad(source=A, indices=[i])
   └── else_case: SeqStmt
       └── seq[0]: TensorStore(dest=B, indices=[i])
           └── value: FloatImm(value=0.0)

The condition is a boolean expression, here the comparison
``A[i] > 0``. For each element, only one of the two stores runs.

Loop Exits
----------

``ir.Continue`` skips to the next iteration of the innermost enclosing
loop, and ``ir.Break`` leaves that loop. This loop records the index of
the first positive element of ``A``:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def first_positive(A: T.Tensor((n,), T.int32), B: T.Tensor((1,), T.int32)):
       B[0] = -1
       for i in range(n):
           if A[i] <= 0:
               continue
           B[0] = i
           break

   loop = first_positive.body[1]
   print(ir.format_tree(loop.body))

The output is:

.. code-block:: text

   SeqStmt
   ├── seq[0]: If(else_case=None)
   │   ├── condition: LE
   │   │   ├── a: TensorLoad(source=A, indices=[i])
   │   │   └── b: IntImm(value=0)
   │   └── then_case: SeqStmt
   │       └── seq[0]: Continue()
   ├── seq[1]: TensorStore(dest=B, indices=[IntImm(value=0)])
   │   └── value: i
   └── seq[2]: Break()

``Continue`` and ``Break`` have no fields.

Assertions and Returns
----------------------

``ir.AssertStmt`` checks a ``condition`` at run time and raises an
error when it is false. ``ir.Return`` ends the function and gives its
``value`` to the caller. This function checks that its input is not
empty, then returns the first element plus one:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def first_plus_one(A: T.Tensor((n,), T.float32)) -> T.float32:
       assert n > 0, "the input must contain at least one element"
       return A[0] + 1

   print(ir.format_tree(first_plus_one.body))

The output is:

.. code-block:: text

   SeqStmt
   ├── seq[0]: AssertStmt
   │   ├── condition: GT
   │   │   ├── a: n
   │   │   └── b: IntImm(value=0)
   │   ├── error_kind: StringImm(value='RuntimeError')
   │   └── message_parts[0]: StringImm(value='the input must contain at least one element')
   └── seq[1]: Return
       └── value: Add
           ├── a: TensorLoad(source=A, indices=[IntImm(value=0)])
           └── b: FloatImm(value=1.0)

The assertion's ``error_kind`` names the error it raises, and its
``message_parts`` hold the message as a list of string fragments. The
fragments are joined only when the assertion fails, through the TVM FFI
C API. Many assertions can therefore share common fragments, such as
an argument name or a shared phrase, which keeps error messages
compact in the compiled code. Calling the function with an empty
tensor fails with this message before the tensor read. Otherwise, the
function returns ``A[0] + 1``.

Regions
-------

Regions are the main place for extensions. ``ir.RegionStmt`` is a
statement whose meaning is defined by an operator rather than by a
dedicated node class. A library registers a region operator to add a
new kind of statement, and the IR represents every such statement in
the same form:

- ``op`` is the operator that defines the region's meaning.
- ``args`` and ``attrs`` are the operator's operands and attributes,
  evaluated in the enclosing scope.
- ``body`` is the sequence of statements inside the region.
- ``body_params`` are variables the region defines on entry, visible
  only inside ``body``.
- ``result_vars`` are variables the region defines for the statements
  that follow it.

This GPU version of add-one places its computation in a device region,
where each thread handles one element:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def gpu_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
       i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
       if i < n:
           B[i] = A[i] + 1

   region = gpu_add_one.body[0]
   print(ir.format_tree(
       region, hide=["attrs", "args", "body_params", "result_vars", "value"]))

The output is:

.. code-block:: text

   RegionStmt(op=tirx.device_entry)
   └── body: SeqStmt
       ├── seq[0]: Bind
       │   └── var: i = Var(ty=int32)
       └── seq[1]: If(else_case=None)
           ├── condition: LT
           │   ├── a: i
           │   └── b: n
           └── then_case: SeqStmt
               └── seq[0]: TensorStore(dest=B, indices=[i])

The ``tirx.device_entry`` operator marks its body as code that runs on
the GPU, and its operands carry the launch configuration. The view
hides those operands and the value expressions to focus on the
statements inside the region.

.. note::

   Regions and calls are the two ways operators extend TIRx: a call
   applies an operator as an expression, and a region applies one as a
   statement.
   :doc:`operators_calls_and_regions` covers how operators are registered and
   how calls and regions use them.
   :ref:`Region Syntax <tvmscript-region-syntax>` explains how TVMScript
   writes a region, and :doc:`../advanced/defining_ir_extensions` walks
   through a complete extension with its own type, operators, and
   lowering pass.
