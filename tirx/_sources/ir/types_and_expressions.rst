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

.. _expressions-and-types:

Types and Expressions
=====================

In this chapter, we will introduce types and expressions and explain how
they work together in the IR. Types describe properties such as dtypes,
tensor shapes, and function signatures, while expressions represent values
and computations. Compiler passes can work with these objects directly to
inspect or rewrite a program without executing it.

.. _user-ir-type:

Types
-----

``ir.Type`` is the base class for all IR types, including primitive types
for scalar and vector values, tensor types, and function types:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Describes
     - Fields
   * - ``ir.PrimType``
     - A scalar or vector dtype
     - ``dtype``
   * - ``ir.PtrType``
     - A pointer's element type and memory scope
     - ``element_type``, ``scope``
   * - ``ir.TupleType``
     - Ordered tuple field types
     - ``fields``
   * - ``ir.FuncType``
     - Function argument and return types
     - ``arg_types``, ``ret_type``
   * - ``tirx.TensorType``
     - A tensor's shape, element type, and storage properties
     - ``shape``, ``dtype``, ``scope``, ``data_alignment``, ``layout``

Next, we will look at two common types in tensor programs in more detail:
``ir.PrimType`` for scalar and vector values, and ``tirx.TensorType`` for tensors.

.. _user-ir-prim-type:

PrimType
~~~~~~~~

``ir.PrimType`` is a widely used type in TIRx IR, which describes integer,
floating-point, and boolean values, either as scalars or vectors. Its ``dtype``
records the element kind, bit width, and number of lanes.

In the sigmoid program, primitive types describe both shape symbols and
computed values. The dynamic shape symbol ``n`` is declared as:

.. literalinclude:: examples/tvmscript_sigmoid.py
   :language: python
   :start-at:     n = T.dynamic
   :end-at:     n = T.dynamic
   :dedent: 4

where ``n`` is a 64-bit integer symbol with type ``ir.PrimType("int64")``.

The sigmoid calculation uses floating-point values:

.. literalinclude:: examples/tvmscript_sigmoid.py
   :language: python
   :start-at:                 biased =
   :end-at:                 Y[i, j] =
   :dedent: 16

In this example, ``X`` and ``bias`` are declared as tensors with float32
elements, so ``X[i, j]`` and ``bias[j]`` have that type. ``T.float32(1)`` supplies
float32 constants for the formula. The arithmetic operations and ``T.exp``
preserve this type, so ``biased`` and the sigmoid result are automatically
inferred as float32 during expression construction. In the IR, this type is
represented by ``ir.PrimType("float32")``.
You can also pass an expression to a ``PrimType`` instance to cast it to that type.

To create a variable of a specific type, you can call the corresponding
``PrimType`` instance without arguments:

.. code-block:: python

   f32 = ir.PrimType("float32")
   x = f32()

``PrimType`` also provides methods to inspect a type, like ``is_scalar()``,
``matches_code(...)``, and ``matches_element_type(...)``. See the
:class:`~tvm_tirx.ir.PrimType` API reference for more details.

.. _user-tensor-type:

TensorType
~~~~~~~~~~

Most computations in TIRx involve tensors. To define a tensor type, we need
two important properties: its element type and shape.
:class:`tirx.TensorType <tvm_tirx.tirx.TensorType>` stores these in its ``dtype``
and ``shape`` fields, respectively. The element type is a ``PrimType``, and
each dimension is an integer expression.

In the sigmoid program, the input ``X`` is declared as:

.. literalinclude:: examples/tvmscript_sigmoid.py
   :language: python
   :start-at:         X: T.Tensor
   :end-at:         X: T.Tensor
   :dedent: 8

where ``X`` is the parameter name and the part after ``:`` is its *type
annotation*, which tells TVMScript what kind of value the parameter accepts.
This annotation becomes a ``tirx.TensorType`` in the IR: ``shape`` describes
``2*n`` rows and ``channels`` columns, and ``dtype`` is
``ir.PrimType("float32")``.

**Symbolic shapes.** A dimension such as ``2*n`` can vary between calls.
In this example, ``channels`` is fixed at 130 when the program is constructed,
while ``n`` is determined from the input's row count at runtime. An input
with shape ``(16, 130)`` gives ``n=8``. A later call can use a different
even row count without recompiling the function. Both fixed shapes such as
``(16, 130)`` and symbolic shapes such as ``(2*n, 130)`` use the same
``TensorType`` representation.

Expressions
-----------

``ir.Expr`` is the base class for all expressions, which represent program
values and the operations that produce them. Each expression's ``ty`` field
records its result type:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``ty``
     - ``ir.Type``
     - The type of value represented by the expression.

Expressions include variables, constants, arithmetic, tensor reads, and calls.
These classes store their values or operands in the following fields:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Classes (examples)
     - Represents
     - Additional fields
   * - ``ir.Var``
     - A variable
     - ``name``
   * - ``ir.IntImm``, ``ir.FloatImm``, ``ir.StringImm``
     - Literal constants
     - ``value``
   * - ``ir.prim.Add``, ``ir.prim.Sub``, ``ir.prim.Mul``
     - Binary arithmetic
     - ``a``, ``b``
   * - ``ir.prim.EQ``, ``ir.prim.LT``
     - Comparisons
     - ``a``, ``b``
   * - ``ir.Op``
     - A registered operator
     - ``name``, ``doc``, ``args_info``
   * - ``ir.Call``
     - A call to an operator or function
     - ``op``, ``args``, ``attrs``, ``ty_args``
   * - ``ir.TensorLoad``
     - A tensor element read
     - ``source``, ``indices``

In our sigmoid example, the input parameter ``X`` is represented by an
``ir.Var`` expression. We can inspect its type through the ``program`` module
(introduced later in :doc:`functions_and_modules`):

.. literalinclude:: examples/tvmscript_sigmoid.py
   :language: python
   :start-at:     function = program["bias_sigmoid"]
   :end-at:     location = row_extent.loc
   :dedent: 4

The ``ir.prim.Mul`` expression for ``2 * n`` is one example of a primitive operation.
Next, we'll look at how primitive operations represent the arithmetic in the
sigmoid formula.

.. _user-ir-prim-operations:

Primitive Operations
~~~~~~~~~~~~~~~~~~~~

Primitive operations give the compiler a common way to represent and reason
about basic computations. The ``ir.prim`` namespace provides expressions such
as ``Add`` for arithmetic and ``LE`` for comparisons. These expressions have
defined semantics, explicit operands, and result types, allowing compiler
passes to analyze computations and perform optimizations such as constant
folding and algebraic simplification. See the
:doc:`ir.prim </reference/python/ir.prim>` API reference for the available
expression classes and construction helpers.

TIRx overloads Python operators such as ``+`` and ``/`` to construct IR
expressions. In our program, we use them as follows:

.. literalinclude:: examples/tvmscript_sigmoid.py
   :language: python
   :start-at:                 biased =
   :end-at:                 Y[i, j] =
   :dedent: 16

In this code, the ``+`` and ``/`` operations become
``ir.prim.Add`` and ``ir.prim.Div`` expressions, respectively.
Each binary expression has two operand fields: ``a`` for the expression on its
left and ``b`` for the expression on its right.

**Bias addition.** The ``+`` in the definition of ``biased`` becomes an
``ir.prim.Add``. Its ``a`` field points to the ``X[i, j]`` tensor read, and
its ``b`` field points to the ``bias[j]`` tensor read. The addition has this
structure:

.. code-block:: text

   ir.prim.Add                 X[i, j] + bias[j]
   ├── a: ir.TensorLoad         X[i, j]
   └── b: ir.TensorLoad         bias[j]

Each tensor read produces one float32 value, and the addition combines
the input element with its channel's bias.

**Sigmoid.** In the final line, the ``+`` in
``T.float32(1) + T.exp(-biased)`` becomes an ``ir.prim.Add``. The ``/`` becomes
an ``ir.prim.Div`` whose ``a`` operand supplies the numerator, ``1``, and
whose ``b`` operand is the addition that computes the denominator.
The exponential ``T.exp(-biased)`` is represented by an ``ir.Call``, which
we will discuss in the :ref:`user-ir-call` section.

.. _user-ir-const:

Primitive Expressions
~~~~~~~~~~~~~~~~~~~~~

A primitive expression is an expression whose ``ty`` is an ``ir.PrimType``.
The expression may be a variable, a tensor element read, an arithmetic
calculation, or a call. Its result type tells us whether it produces an
integer, a floating-point value, a boolean, or a vector of these values.
The float32 tensor reads and computations in our sigmoid example are primitive
expressions.

**Type inference.** Result types are usually determined when expressions are
constructed and stored in their ``ty`` fields. Overloaded arithmetic operators
such as ``+`` match operand types, promoting compatible types and inserting
casts where needed. Comparisons produce boolean results:

.. code-block:: python

   from tvm_tirx import ir

   x = ir.Var("x", "float32")
   (x + 1).ty                   # ir.PrimType("float32")
   (x > 0).ty                   # ir.PrimType("bool")
   ir.Call("prim.exp", [x]).ty  # ir.PrimType("float32")

For ``ir.Call``, omitting ``ty`` uses the operator's registered result-type
rule or the callee's function signature. ``ir.reinfer_type(call)`` recomputes
the result type without modifying ``call.ty`` or validating the call.
See :ref:`types-and-validation` for inference rules and explicit validation.

**Index expressions.** The loop ``for i in range(2 * n)`` uses ``2*n`` to
determine how many rows to process. The multiplication's ``ty`` is
``ir.PrimType("int64")``. The loop indices ``i`` and ``j`` also have this
type and select one row and one channel, respectively.

.. _user-ir-var:

Var
~~~

Variables can be function parameters, loop indices, or intermediate values
defined by bindings. ``ir.Var`` can be used to represent these values together
with their names and types.

We can declare the variables in our sigmoid example using the IR API:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin variables
   :end-before: # end variables
   :dedent: 4

``X``, ``bias``, and ``Y`` have tensor types. ``i`` selects a sample, and ``j``
selects a channel. These indices have int64 types. ``biased`` holds one
float32 value after adding the channel's bias to the input element.

.. _user-ir-call:

Call
~~~~

After defining a function (introduced later in :doc:`functions_and_modules`),
we can call it with arguments. :doc:`ir.Call <operators_and_calls>` represents
such a call in the IR. Its ``op`` field (an ``ir.Op`` for a registered operator)
identifies the function or operator being called, and ``args`` holds the
argument expressions. Using an unregistered operator name in ``ir.Op.get``
or ``ir.Call`` raises ``AttributeError``.

In sigmoid, ``1 / (1 + exp(-x))``, the exponential is a call inside a larger
arithmetic expression:

.. code-block:: python

   from tvm_tirx import ir, tirx
   from tvm_tirx.script import from_source
   from tvm_tirx.script import tirx as T

   exp_op = ir.Op.get("prim.exp")

   def sigmoid(x):
       one = ir.const(1, x.ty)
       exp_neg_x = ir.Call(exp_op, [-x])
       return one / (one + exp_neg_x)

``exp_op`` is the registered ``ir.Op`` for the exponential operation.
``ir.Call(exp_op, [-x])`` constructs a call with the negated input as its
argument. The sum ``one + exp_neg_x`` forms the denominator; the division
produces sigmoid's result.
The helper preserves the input's primitive type, float32 in this example.

We can construct IR by calling a Python helper directly. Here,
``sigmoid(biased)`` runs the helper above, which builds an ``ir.Call`` for the
exponential and returns the complete sigmoid expression:

.. code-block:: python

   biased_value = X[i, j] + bias[j]
   result = sigmoid(biased)

``biased_value`` is an ``ir.prim.Add`` containing two tensor reads. The program
binds this value to ``biased`` before using it in the sigmoid calculation.
``result`` is an ``ir.prim.Div`` whose denominator contains another addition
and the exponential call. The program stores this result in ``Y[i, j]``.

When the compiled program runs, each loop iteration reads the input and bias
values, computes the activation, and writes it to ``Y``. The biased values stay
local to that iteration, so bias addition and sigmoid share one traversal
without an intermediate tensor.
