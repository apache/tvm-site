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
and records the argument expressions for one use. We will examine how
the IR records these definitions and calls, infers result types, and
checks arguments.

Operators
---------

An operator identifies a computation, such as sigmoid or a CUDA
thread-index lookup. :class:`ir.Op <tvm_tirx.ir.Op>` represents a
registered operator and records its name, argument descriptions, and
attributes. The operator's registration also provides rules for
result-type inference and argument validation.

A CUDA kernel can read a thread's x-axis coordinate with
``T.cuda.thread_idx("x")``. The registered operator behind this helper is
``tirx.cuda.thread_idx``. We can retrieve its ``ir.Op`` object and inspect
the names of the parameters it accepts:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   op = ir.Op.get("tirx.cuda.thread_idx")
   argument_names = [argument.name for argument in op.args_info]
   print(argument_names)

The output is:

.. code-block:: text

   ['axis']

The list contains one parameter name, ``axis``.
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
     - Types passed as arguments to the callee, empty by default.
   * - ``attrs``
     - Operator-specific configuration or annotations for this call,
       stored as an ``Attrs`` object, or ``None``.
   * - ``ty``
     - The result type of this call, inferred or explicitly specified.

The thread-index call refers to the registered ``tirx.cuda.thread_idx``
operator. We can construct it directly through the IR API:

.. code-block:: python

   op = ir.Op.get("tirx.cuda.thread_idx")
   x_call = ir.Call(op, [ir.StringImm("x")])

``ir.Op.get`` retrieves the shared operator definition. ``ir.Call``
creates an expression whose ``op`` refers to that definition and whose
``args`` contains the axis argument. Its ``ty`` records the int32 result:

.. code-block:: python

   print(ir.format_tree(
       x_call,
       expand=[(ir.Call, "ty")],
       hide=["attrs", "ty_args"],
   ))

The output is:

.. code-block:: text

   Call(op=tirx.cuda.thread_idx)
   ├── ty: PrimType(dtype=int32)
   └── args[0]: StringImm(value='x')

``op.args_info[0].name`` is the parameter name ``"axis"``;
``x_call.args[0]`` holds its argument value ``"x"``. The call describes
reading the current thread's x-axis coordinate when the compiled GPU
program runs.

For this registered operator, ``ir.Call`` infers ``ty`` from the
operator's type rule. If we pass ``ty`` to the constructor, it uses that
result type and skips inference.

The constructor also accepts the registered name directly:
``ir.Call("tirx.cuda.thread_idx", [ir.StringImm("x")])``.

The call above selects the x axis. We can construct a call for the
y axis with the same operator:

.. code-block:: python

   y_call = ir.Call(op, [ir.StringImm("y")])

Both ``x_call.op`` and ``y_call.op`` reference the shared ``op``. Each
call has its own ``args`` array, containing ``"x"`` or ``"y"``, and
both calls produce an int32 coordinate. We can check that their operator
references identify the same object:

.. code-block:: python

   print(x_call.op.same_as(y_call.op))

The output is:

.. code-block:: text

   True

A call's result can be an operand in a larger expression. A kernel with
256 threads per block can combine the block and thread coordinates to
find the tensor element assigned to a thread:

.. code-block:: python

   n = T.dynamic("n")

   @T.function
   def gpu_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       T.device_entry(launch=T.cuda.LaunchConfig(grid=T.ceildiv(n, 256), block=256))
       i = T.cuda.block_idx("x") * 256 + T.cuda.thread_idx("x")
       if i < n:
           B[i] = A[i] + 1

The block-index call produces the block coordinate, and the thread-index
call produces the coordinate within that block. Each helper constructs
an ``ir.Call``. The multiplication and addition use the calls' results
as operands:

.. code-block:: python

   device_region = gpu_add_one.body.seq[0]
   index_binding = device_region.body.seq[0]  # The definition of i.
   print(ir.format_tree(index_binding.value, hide=["attrs", "ty_args"]))

The output is:

.. code-block:: text

   Add
   ├── a: Mul
   │   ├── a: Call(op=tirx.cuda.block_idx)
   │   │   └── args[0]: StringImm(value='x')
   │   └── b: IntImm(value=256)
   └── b: Call(op=tirx.cuda.thread_idx)
       └── args[0]: StringImm(value='x')

Expression Arguments
~~~~~~~~~~~~~~~~~~~~

A call can also receive a computation as an argument. We can add one to
a float32 scalar and pass the result to sigmoid:

.. code-block:: python

   scalar_value = ir.Var("scalar_value", "float32")
   scalar_plus_one = scalar_value + 1
   activated_plus_one = ir.prim.sigmoid(scalar_plus_one)

``activated_plus_one`` is the ``ir.Call`` returned by the
``ir.prim.sigmoid`` helper. Its result type is ``ir.PrimType("float32")``,
so it is also a primitive expression. We can construct the same call
explicitly:

.. code-block:: python

   sigmoid_op = ir.Op.get("prim.sigmoid")
   activated_plus_one = ir.Call(sigmoid_op, [scalar_plus_one])

In both forms, the call's ``op`` refers to the registered ``prim.sigmoid``
operator, and ``args[0]`` refers to the addition expression ``scalar_plus_one``:

.. code-block:: python

   print(ir.format_tree(
       activated_plus_one,
       expand=[(ir.Call, "ty")],
       hide=["attrs", "ty_args"],
   ))

The output is:

.. code-block:: text

   Call(op=prim.sigmoid)
   ├── ty: PrimType(dtype=float32)
   └── args[0]: Add
       ├── a: scalar_value
       └── b: FloatImm(value=1.0)

When the compiled program runs, sigmoid receives the result of the
addition. This expression therefore computes
``1 / (1 + exp(-(scalar_value + 1)))`` and produces a float32 value.

Function Calls
--------------

The callee can also be a function defined in IR. An ``IRModule`` stores
named function definitions, and an ``ir.GlobalVar`` identifies one of
those definitions. For example, a small helper can put the scalar
add-one computation in a function:

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
``function_call.args[0]`` holds the value passed to the helper's parameter
``value``. The call records this reference and argument; the module holds
the function's definition.

In this example, ``callee.ty`` is ``MissingType``, so the reference does
not provide a signature for result-type inference. We pass the return
type from the definition through ``ty`` instead. Other function-valued
IR expressions can also serve as callees. When their ``ty`` is a
``FuncType``, the constructor can infer the result from its return type.

Calls in Function Bodies
------------------------

A call contributes to a function's computation through the expression or
statement that uses it. This CPU function applies sigmoid to each
input element after adding one:

.. code-block:: python

   n = T.dynamic("n")

   @T.function
   def add_one_sigmoid(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = T.sigmoid(A[i] + 1)

The TVMScript helper ``T.sigmoid`` builds a call to ``prim.sigmoid``.
The tensor store uses that call as its ``value`` expression:

.. code-block:: python

   activation_store = add_one_sigmoid.body.seq[0].body.seq[0]
   activation_call = activation_store.value
   print(ir.format_tree(activation_store, hide=["attrs", "ty_args"]))

The output is:

.. code-block:: text

   TensorStore(dest=B, indices=[i])
   └── value: Call(op=prim.sigmoid)
       └── args[0]: Add
           ├── a: TensorLoad(source=A, indices=[i])
           └── b: FloatImm(value=1.0)

When the function runs, the addition produces the sigmoid input and the
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

   memory_loop = masked_add_one.body.seq[0]
   write_statement = memory_loop.body.seq[2]
   write_call = write_statement.value
   print(ir.format_tree(
       write_statement,
       hide=["args", "attrs", "ty_args"],
   ))

The output is:

.. code-block:: text

   Evaluate
   └── value: Call(op=tirx.masked_store)

The loop body first binds ``indices`` and ``values``. Its third
statement, ``write_statement``, evaluates the masked-store call.
This view hides the call's arguments to focus on the ``Evaluate`` wrapper.
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

A call's value arguments do not always determine its result type.
A tensor allocation, for example, has no input tensor from which to
infer a shape or dtype. We pass the required tensor type through
``ty_args``. This call describes four float32 elements in local storage:

.. code-block:: python

   from tvm_tirx import tirx

   temporary_type = tirx.TensorType((4,), "float32", scope="local")
   allocation_call = ir.Call(
       "tirx.alloc_tensor", [], ty_args=[temporary_type],
       attrs={"tirx.volatile": True},
   )

``ty_args[0]`` gives the tensor type to the allocation operator. Its
type rule uses this argument as the result type, so ``allocation_call.ty``
is ``temporary_type``. A call can therefore produce a tensor as well as a
scalar or vector value.

The ``attrs`` dictionary configures this allocation. It becomes an
``ir.DictAttrs`` object, and ``"tirx.volatile": True`` requests volatile
accesses: code generation must preserve the loads and stores to this
storage. These attributes belong to this call; the attributes on ``ir.Op``
describe the operation shared by all its calls.

We can give the allocated tensor a name with a binding:

.. code-block:: python

   temporary = ir.Var("temporary", temporary_type)
   allocation_binding = ir.Bind(temporary, allocation_call)

The binding's ``value`` is the allocation call, and its ``var`` is
``temporary``. A function body can place this binding before statements
that read or write the temporary tensor.

.. _types-and-validation:

.. _type-inference-and-validation:

Call Types and Validation
-------------------------

A call needs a result type that matches its inputs. For sigmoid, a scalar
float32 input produces float32, and a four-lane input produces float32x4.
The constructor normally infers this type. An explicit ``ty`` skips
inference, and ``ty=ir.Type.missing()`` leaves the result undetermined.
With the scalar addition defined above, the three choices are:

.. code-block:: python

   sigmoid_op = ir.Op.get("prim.sigmoid")
   inferred_call = ir.Call(sigmoid_op, [scalar_plus_one])
   typed_call = ir.Call(sigmoid_op, [scalar_plus_one], ty="float32")
   untyped_call = ir.Call(sigmoid_op, [scalar_plus_one], ty=ir.Type.missing())

``inferred_call`` and ``typed_call`` both store a float32 result type.
``untyped_call`` explicitly sets ``MissingType``, so its result type
remains undetermined.

Now suppose we are rebuilding the scalar call to work on four values at
once. We can first set its vector argument and leave the result type
undetermined until we run the operator's inference rule:

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

``ty="float32x4"`` sets the result type directly, so the constructor
skips inference. It still creates the incomplete ``ir.Call`` object
because argument checks do not run automatically. We can use
:meth:`Call.validate() <tvm_tirx.ir.Call.validate>` to check whether the
arguments satisfy the operator's requirements:

.. code-block:: python

   invalid_sigmoid.validate()

``invalid_sigmoid.validate()`` raises ``TypeError`` for the missing input.
``untyped_sigmoid.validate()`` succeeds because the call includes
``vector_value``. Its ``ty`` remains ``MissingType`` because validation
checks the arguments without inferring the result type.
