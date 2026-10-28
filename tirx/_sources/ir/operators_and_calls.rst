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

Operators and Calls
===================

In this chapter, we will introduce operators and calls. A registered
operator provides the shared argument requirements, type rules, and effect
information for a computation. A call refers to an operator or function
and supplies argument expressions for that use. We will examine how
the IR records these definitions and calls, infers result types, and
checks arguments.

Operators
---------

An operator identifies a computation, such as sigmoid or a CUDA
thread-index lookup. :class:`ir.Op <tvm_tirx.ir.Op>` represents a
registered operator and records its name, argument descriptions, and
attributes. The operator's registration also provides rules for
result-type inference and argument validation.

The GPU example uses the registered operator ``tirx.cuda.thread_idx`` to
obtain a thread coordinate. ``ir.Op.get`` retrieves its ``ir.Op`` object,
and ``op.args_info`` describes its parameters. We can extract their names:

.. code-block:: python

   from tvm_tirx import ir
   op = ir.Op.get("tirx.cuda.thread_idx")
   argument_names = [argument.name for argument in op.args_info]
   print(argument_names)

The output is:

.. code-block:: text

   ['axis']

The list contains one parameter name, ``axis``. In the GPU example,
``T.cuda.thread_idx("x")`` passes the string ``"x"`` for this parameter.
This selects the x axis, so the call reads the current thread's x-axis
coordinate within its block.

Calls
-----

:class:`ir.Call <tvm_tirx.ir.Call>` represents one use of an operator or
function with specific argument expressions. Its ``op`` field references
the expression being called, known as the callee, and ``args`` holds the
arguments for this use. The main fields describe what is called, what
is passed to it, and what type of value it produces:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``op``
     - The callee: a registered operator or another function-valued expression.
   * - ``args``
     - Value arguments in calling order.
   * - ``ty_args``
     - Types supplied as arguments to the callee, empty by default.
   * - ``attrs``
     - Operator-specific configuration or annotations for this call,
       stored as an ``Attrs`` object, or ``None``.
   * - ``ty``
     - The result type of this call, inferred or explicitly supplied.

The thread-index call refers to the registered ``tirx.cuda.thread_idx``
operator. We can construct it directly through the IR API:

.. code-block:: python

   op = ir.Op.get("tirx.cuda.thread_idx")
   x_call = ir.Call(op, [ir.StringImm("x")])

``ir.Op.get`` retrieves the shared operator definition. ``ir.Call``
creates an expression whose ``op`` refers to that definition and whose
``args`` contains the axis argument. Its ``ty`` records the int32 result:

.. code-block:: text

   x_call: ir.Call
   ├── op → shared ir.Op                tirx.cuda.thread_idx
   ├── args[0]: ir.StringImm             "x"
   └── ty: ir.PrimType                   int32

``op.args_info[0].name`` is the parameter name ``"axis"``;
``x_call.args[0]`` supplies its argument value ``"x"``. The call describes
reading the current thread's x-axis coordinate when the compiled GPU
program runs.

For this registered operator, ``ir.Call`` infers ``ty`` from the
operator's type rule. An explicit ``ty`` supplies the result type directly
and skips inference.

The constructor also accepts the registered name directly:
``ir.Call("tirx.cuda.thread_idx", [ir.StringImm("x")])``.

The program uses the x axis. As an additional example, we can construct
a call for the y axis with the same operator:

.. code-block:: python

   y_call = ir.Call(op, [ir.StringImm("y")])

Both ``x_call.op`` and ``y_call.op`` reference the shared ``op``. Each
call has its own ``args`` array, containing ``"x"`` or ``"y"``, and
both calls produce an int32 coordinate:

.. code-block:: text

   x_call.op ──┐
               ├──> op: ir.Op  tirx.cuda.thread_idx
   y_call.op ──┘

A call's result can be an operand in a larger expression. In the GPU
example, two calls contribute to the element index ``i``:

.. literalinclude:: ../get_started/examples/add_one_cuda.py
   :language: python
   :start-at:     i = T.cuda.block_idx
   :end-at:     i = T.cuda.block_idx
   :dedent: 4

``T.cuda.block_idx("x")`` supplies the block coordinate, and
``T.cuda.thread_idx("x")`` supplies the thread's coordinate within that
block. Each helper constructs an ``ir.Call``. The multiplication and
addition use the calls' results as operands:

.. code-block:: text

   ir.prim.Add
   ├── a: ir.prim.Mul
   │   ├── a: ir.Call                   T.cuda.block_idx("x")
   │   └── b: ir.IntImm                 256
   └── b: ir.Call                       T.cuda.thread_idx("x")

Expression Arguments
~~~~~~~~~~~~~~~~~~~~

A call can also receive a computation as an argument.
As an extension of the scalar add-one expression, we can apply sigmoid to
``scalar_value + 1``. The variable ``scalar_value`` has type
``ir.PrimType("float32")``:

.. code-block:: python

   scalar_value = ir.Var("scalar_value", "float32")
   scalar_plus_one = scalar_value + 1
   activated_plus_one = ir.prim.sigmoid(scalar_plus_one)

``ir.prim.sigmoid(...)`` constructs and returns an ``ir.Call`` object.
``activated_plus_one`` refers to this call, which is also a primitive
expression because its result type is ``ir.PrimType("float32")``.
We can construct the same call explicitly:

.. code-block:: python

   sigmoid_op = ir.Op.get("prim.sigmoid")
   activated_plus_one = ir.Call(sigmoid_op, [scalar_plus_one])

In both forms, the call's ``op`` refers to the registered ``prim.sigmoid``
operator, and ``args[0]`` refers to the addition expression ``scalar_plus_one``:

.. code-block:: text

   activated_plus_one: ir.Call
   ├── op → shared ir.Op                prim.sigmoid
   ├── args[0]: ir.prim.Add              scalar_value + 1
   │   ├── a: ir.Var                    scalar_value
   │   └── b: ir.FloatImm               1.0
   └── ty: ir.PrimType                   float32

When the compiled program runs, sigmoid receives the result of the
addition. This expression therefore computes
``1 / (1 + exp(-(scalar_value + 1)))`` and produces a float32 value.

Function Calls
--------------

The callee can also be a function defined in IR. A function stored in an
``IRModule`` is referenced through an ``ir.GlobalVar``. For example, a
small helper can put the scalar add-one computation in a function:

.. code-block:: python

   from tvm_tirx.script import tirx as T

   @T.function(private=True)
   def add_one_scalar(value: T.float32) -> T.float32:
       return value + 1

``private=True`` makes this an internal helper. We can place it in a
module and construct a call that passes ``scalar_value`` as its argument:

.. code-block:: python

   call_module = ir.IRModule({"add_one_scalar": add_one_scalar})
   callee = call_module.get_global_var("add_one_scalar")
   function_call = ir.Call(callee, [scalar_value], ty=add_one_scalar.ret_type)

``callee`` is the module's global reference to ``add_one_scalar``.
``function_call.op`` refers to this ``GlobalVar``, and
``function_call.args[0]`` supplies the value for the helper's parameter
``value``. A function-valued callee with a ``FuncType`` supplies a return
type for inference. Here, ``callee.ty`` is ``MissingType``: the reference
itself does not carry the function's signature. We therefore supply the
return type from the definition through ``ty``. The call holds the
reference and argument; the module holds the function's definition.
Other function-valued IR expressions can also serve as callees.

Calls in Function Bodies
------------------------

A call contributes to a function's computation through the expression or
statement that uses it. We can extend the CPU example to apply sigmoid
to each add-one result:

.. code-block:: python

   n = T.dynamic("n")

   @T.function
   def add_one_sigmoid(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = T.sigmoid(A[i] + 1)

The TVMScript helper ``T.sigmoid`` builds a call to ``prim.sigmoid``.
The tensor store uses that call as its ``value`` expression:

.. code-block:: python

   activation_store = add_one_sigmoid.body[0].body[0]
   activation_call = activation_store.value

.. code-block:: text

   activation_store: ir.TensorStore
   ├── dest: ir.Var                     B
   └── value: ir.Call                   prim.sigmoid
       └── args[0]: ir.prim.Add         A[i] + 1

When the function runs, the addition supplies the sigmoid input and the
store writes its result to ``B[i]``.

Calls for Their Effects
~~~~~~~~~~~~~~~~~~~~~~~

Some registered operators perform a memory write themselves. A common
example is a masked store, which writes only selected vector lanes. A
vector version of add-one can use masked reads and writes to process
four elements at a time, including a final group with fewer than four
elements:

.. code-block:: python

   @T.function
   def masked_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for base in range(0, n, 4):
           indices = T.Ramp(base, 1, 4)
           values = T.masked_load(A, indices, indices < n)
           T.masked_store(B, values + 1, indices, indices < n)

``indices`` contains four consecutive positions starting at ``base``.
The mask ``indices < n`` enables only positions within the tensor for
both the read and the write. For example, if ``n = 10``, valid indices
range from 0 to 9. The final group contains indices 8, 9, 10, and 11, so
the mask enables only the first two lanes.

``T.masked_store`` builds a call to ``tirx.masked_store``, which performs
the write and returns ``void``. The TVMScript helper places this call in
an ``ir.Evaluate`` statement inside the loop body:

.. code-block:: python

   memory_loop = masked_add_one.body[0]
   write_statement = memory_loop.body[2]
   write_call = write_statement.value

.. code-block:: text

   memory_loop.body: ir.SeqStmt
   ├── seq[0]: ir.Bind                  indices
   ├── seq[1]: ir.Bind                  values
   └── seq[2]: ir.Evaluate
       └── value: ir.Call               tirx.masked_store
           └── ty: ir.PrimType          void

Through the IR API, ``ir.Evaluate(write_call)`` constructs the
corresponding statement. Even with a ``void`` result type, the call is
an expression; ``Evaluate`` gives it a place in the statement sequence.

The operators' effect attributes explain why these two calls need
different treatment when their results are unused:

.. code-block:: python

   activation_effect = activation_call.op.get_attr("TCallEffectKind")
   write_effect = write_call.op.get_attr("TCallEffectKind")

``activation_effect`` is ``ir.CallEffectKind.Pure``: the sigmoid
operation has no memory side effects. ``write_effect`` is
``ir.CallEffectKind.UpdateState``. Compiler passes cannot discard this
write merely because its result is unused; it updates the output tensor.

Type Arguments and Attributes
-----------------------------

Some operators need a type as an input to describe the value they
produce. For example, a tensor allocation needs the shape and element
type of the storage to create. This additional example describes a
four-element float32 tensor in local storage:

.. code-block:: python

   from tvm_tirx import tirx

   temporary_type = tirx.TensorType((4,), "float32", scope="local")
   allocation_call = ir.Call(
       "tirx.alloc_tensor", [], ty_args=[temporary_type], attrs={}
   )

``ty_args[0]`` supplies the tensor type to the allocation operator. Its
type rule uses this argument as the result type, so ``allocation_call.ty``
is ``temporary_type``. A call can therefore produce a tensor as well as a
scalar or vector value.
A surrounding ``ir.Bind`` can name the allocated tensor for subsequent
reads and writes, as described in :doc:`statements`.

The ``attrs`` argument supplies annotations for an individual call. A
dictionary becomes an ``ir.DictAttrs`` object; the operator determines
which annotations it supports and how they are used. These annotations
belong to this call, while the attributes on ``ir.Op`` describe
properties shared by every call to that operator.

.. _types-and-validation:

.. _type-inference-and-validation:

Call Types and Validation
-------------------------

The earlier examples inferred a call's result type or supplied it
explicitly. During IR construction, we may also need a call whose result
type is not known yet. ``ty=ir.Type.missing()`` records this state and
defers inference. The same scalar sigmoid input, ``scalar_plus_one``,
lets us compare all three choices:

.. code-block:: python

   sigmoid_op = ir.Op.get("prim.sigmoid")
   inferred_call = ir.Call(sigmoid_op, [scalar_plus_one])
   typed_call = ir.Call(sigmoid_op, [scalar_plus_one], ty="float32")
   untyped_call = ir.Call(sigmoid_op, [scalar_plus_one], ty=ir.Type.missing())

``inferred_call`` and ``typed_call`` both store a float32 result type.
``untyped_call`` explicitly supplies ``MissingType``, so its result type
remains undetermined.

The same operator also accepts a four-lane float32 vector. We can leave
this call's result type undetermined, then infer it from the vector
argument:

.. code-block:: python

   vector_value = ir.Var("vector_value", "float32x4")
   untyped_sigmoid = ir.Call(sigmoid_op, [vector_value], ty=ir.Type.missing())

With the vector argument in place,
:func:`ir.reinfer_type <tvm_tirx.ir.reinfer_type>` applies the sigmoid
operator's type rule and returns the result type. We can use it to
construct a new call with that type:

.. code-block:: python

   inferred = ir.reinfer_type(untyped_sigmoid)
   rebuilt_call = ir.Call(sigmoid_op, untyped_sigmoid.args, ty=inferred)

``ir.reinfer_type`` returns ``ir.PrimType("float32x4")`` without changing
``untyped_sigmoid.ty``, which remains ``MissingType``. ``rebuilt_call``
stores the returned type.

The sigmoid operator also requires one input argument. ``rebuilt_call``
passes ``vector_value`` as that input. The following call leaves it out:

.. code-block:: python

   invalid_sigmoid = ir.Call(sigmoid_op, [], ty="float32x4")

``ty="float32x4"`` supplies the result type directly, so the constructor
skips inference. It still creates the incomplete ``ir.Call`` object
because argument checks do not run automatically. We can use
:meth:`Call.validate() <tvm_tirx.ir.Call.validate>` to check whether the
arguments satisfy the operator's requirements:

.. code-block:: python

   invalid_sigmoid.validate()

``invalid_sigmoid.validate()`` raises ``TypeError`` for the missing input.
``untyped_sigmoid.validate()`` succeeds because the call supplies
``vector_value``. Its ``ty`` remains ``MissingType`` because validation
checks the arguments without inferring the result type.
