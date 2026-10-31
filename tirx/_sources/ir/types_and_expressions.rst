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

An **expression** is an IR node that represents a value: a variable, a
constant, an arithmetic operation, a tensor read, or a call. Expressions
nest to form computations. In ``A[i] + 1``, the addition node holds the
tensor read ``A[i]`` and the constant ``1`` as its operands.

Every expression has a **type** attached to it in its ``ty`` field. The
type describes the value the expression produces: a float32 scalar, a
four-lane float32 vector, a tensor with shape ``(n,)``, and so on.

.. code-block:: python

   from tvm_tirx import ir
   x = ir.Var("x", "float32")
   y = x + 1
   print(ir.format_tree(y, expand=[(ir.Expr, "ty")]))

The output is:

.. code-block:: text

   Add
   ├── ty: PrimType(dtype=float32)
   ├── a: x
   └── b: FloatImm(value=1.0)
       └── ty: PrimType(dtype=float32)

``y`` is an ``ir.prim.Add`` node. Its operands are the variable ``x`` and
the constant ``1``, and each node carries its own ``ty``. The variable was
given type float32 when it was created; the constant and the addition
derive theirs from it.

``ir.Expr`` is the base class of all expressions, and ``ir.Type`` is the
base class of all types. The rest of this chapter introduces the main
type classes, then the main expression classes, and finally how types
can depend on expressions. It uses this add-one function as the running
example:

.. literalinclude:: ../../examples/quick_start/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

.. _user-ir-type:

Types
-----

``ir.Type`` is the base class of all types. A type describes a value:
its dtype, its shape, or the values it is built from.

``ir.PrimType`` describes a scalar or vector value by its dtype: the
element kind, the bit width, and the number of lanes. ``float32`` is a
single 32-bit floating-point value, and ``float32x4`` holds four of
them in one vector. In TVMScript, ``T.float32`` is the object
``ir.PrimType("float32")``.

``tirx.TensorType`` describes a whole tensor, and the next subsection
covers it in detail.

The main type classes are:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Describes
     - Fields
   * - ``ir.PrimType``
     - A scalar or vector value
     - ``dtype``
   * - ``tirx.TensorType``
     - A tensor's shape, element type, and storage
     - ``shape``, ``dtype``, ``scope``, ``data_alignment``, ``layout``
   * - ``ir.TupleType``
     - An ordered group of values
     - ``fields``
   * - ``ir.PtrType``
     - A memory address and the element type it points to
     - ``element_type``, ``scope``
   * - ``ir.FuncType``
     - A function's argument and result types
     - ``arg_types``, ``ret_type``

.. _user-tensor-type:

TensorType
~~~~~~~~~~

``tirx.TensorType`` describes a whole tensor: its shape, its element
type, and how it is stored. ``T.Tensor`` builds one:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   tile_type = T.Tensor((16, 4), T.float16)
   print(ir.format_tree(tile_type))

The output is:

.. code-block:: text

   TensorType
   ├── shape[0]: IntImm(value=16)
   ├── shape[1]: IntImm(value=4)
   └── dtype: PrimType(dtype=float16)

``shape`` lists one expression per dimension; here both are integer
constants. ``dtype`` is the ``PrimType`` of each element.

A tensor type also records where its elements are stored. ``scope`` is
the memory space that holds them, such as ``"global"`` (the default),
``"shared"``, or ``"local"``:

.. code-block:: python

   from tvm_tirx.script import tirx as T

   shared_type = T.Tensor((16, 4), T.float16, scope="shared")
   print(repr(shared_type.scope))

The output is:

.. code-block:: text

   MemoryScope('shared')

We can optionally supply a tensor's ``layout``, which changes how its
elements are placed in storage. The
:doc:`Data Layout </programming_guides/data_layout>` guide covers
layouts.

.. _user-ir-expr:

Expressions
-----------

Each expression class adds its own fields alongside ``ty``. When an
expression is constructed, its ``ty`` is computed from its inputs, so a
finished expression always carries a concrete type. While an expression
is still being built and its type is not yet known, ``ty`` holds the
placeholder ``ir.MissingType``.

The following table lists the expression classes and the fields each
adds, and the subsections below introduce them in turn:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Class
     - Represents
     - Fields
   * - ``ir.Var``
     - A named value
     - ``name``
   * - ``ir.IntImm``, ``ir.FloatImm``, ``ir.StringImm``
     - Literal constants
     - ``value``
   * - ``ir.prim`` nodes, such as ``ir.prim.Add``
     - Primitive computations, listed in Primitive Expressions
     - Operands, such as ``a`` and ``b``
   * - ``ir.TensorLoad``
     - A tensor element read
     - ``source``, ``indices``
   * - ``ir.Tuple``
     - A group of values
     - ``fields``
   * - ``ir.TupleGetItem``
     - One field of a tuple
     - ``tuple_value``, ``index``
   * - ``ir.Call``
     - A call to an operator or function
     - ``op``, ``args``, ``ty_args``, ``attrs``
   * - ``ir.GlobalVar``
     - A reference to a function in a module
     - ``name_hint``

.. _user-ir-var:

Var
~~~

``ir.Var`` represents a named value. It has two fields: ``name``, a
readable label, and ``ty``, the type of the value it stands for.

In the add-one function, the parameters ``A`` and ``B`` are variables.
Each one's ``ty`` is the annotation written in the signature:

.. code-block:: python

   A, B = add_one.params
   print(ir.format_tree(A, expand=[(ir.Var, "ty")]))

The output is:

.. code-block:: text

   A = Var
   └── ty: TensorType
       ├── shape[0]: n = Var
       │   └── ty: PrimType(dtype=int32)
       └── dtype: PrimType(dtype=float32)

``T.Tensor((n,), T.float32)`` becomes a ``tirx.TensorType``. Its
``shape`` holds the dimension expressions and its ``dtype`` holds the
element type, ``ir.PrimType("float32")``.

The function has two other kinds of variables. The symbolic dimension
``n`` comes from ``n = T.dynamic("n")``. It is an int32 ``Var`` that
stands for a size known only when the function is called, and it appears
in ``A``'s shape above. The loop index ``i`` is defined by
``for i in range(n)``. It is an int32 ``Var`` that the loop body uses to
read ``A[i]`` and write ``B[i]``.

In TIRx canonical form, each ``Var`` is defined at most once in a
function. A variable is defined where it is introduced, such as a
function parameter or a loop index, and it can then be used any number
of times. Two loops cannot share the same ``Var`` object as their index,
even when neither loop is nested in the other; each loop needs its own
``Var``, even if the names are the same. This rule keeps analysis
simple: every use of a variable refers to exactly one definition.

.. _user-ir-const:

Constants
~~~~~~~~~

``ir.IntImm``, ``ir.FloatImm``, and ``ir.StringImm`` hold literal values
in their ``value`` field. A constant's ``ty`` records its dtype:

.. code-block:: python

   from tvm_tirx import ir

   print(ir.format_tree(ir.IntImm("int64", 3), expand=[(ir.Expr, "ty")]))

The output is:

.. code-block:: text

   IntImm(value=3)
   └── ty: PrimType(dtype=int64)

A Python number used in an operation becomes a constant with the dtype
of the other operand, so ``x + 1`` with a float32 ``x`` contains a
float32 ``FloatImm``.

.. _user-ir-prim-expr:

Primitive Expressions
~~~~~~~~~~~~~~~~~~~~~

A primitive expression is an expression whose ``ty`` is an
``ir.PrimType``. These expressions usually compute an index or an
element value, and such computations occur in nearly every ML compiler
scenario: tensor offsets, loop bounds, tiling and vectorization, and the
scalar math inside each kernel. TIRx therefore gives them dedicated node
classes in the ``ir.prim`` namespace, such as ``ir.prim.Add``,
``ir.prim.LT``, and ``ir.prim.Cast``, rather than representing them as
generic calls. The IR can refer to these common patterns directly, and
analyses and transformations can recognize, reason about, and simplify
them by their structure. Python operators on expressions construct
these nodes.

The following table lists the primitive expression node classes in
``ir.prim``:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Category
     - Node classes
     - TVMScript syntax
   * - Arithmetic
     - ``Add``, ``Sub``, ``Mul``, ``Div``, ``Mod``, ``FloorDiv``,
       ``FloorMod``, ``Min``, ``Max``
     - ``+``, ``-``, ``*``, ``/`` (float), ``//``, ``%`` (int)
   * - Comparison
     - ``EQ``, ``NE``, ``LT``, ``LE``, ``GT``, ``GE``
     - ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``
   * - Logical
     - ``And``, ``Or``, ``Not``
     - ``and``, ``or``, ``not``
   * - Bitwise
     - ``BitwiseAnd``, ``BitwiseOr``, ``BitwiseXor``, ``BitwiseNot``,
       ``LShift``, ``RShift``
     - ``&``, ``|``, ``^``, ``~``, ``<<``, ``>>``
   * - Conversion
     - ``Cast``
     - —
   * - Selection
     - ``Select``
     - —
   * - Vector
     - ``Broadcast``, ``Ramp``, ``Shuffle``
     - —
   * - Binding
     - ``Let``
     - —

**Index expressions.** Index calculations combine integer variables and
constants, such as a block offset plus a thread index. A binary node
stores its operands in ``a`` and ``b``, and nodes nest to form larger
computations.

.. code-block:: python

   from tvm_tirx import ir

   bx = ir.Var("bx", "int32")
   tx = ir.Var("tx", "int32")
   print(ir.format_tree(bx * 256 + tx))

The output is:

.. code-block:: text

   Add
   ├── a: Mul
   │   ├── a: bx
   │   └── b: IntImm(value=256)
   └── b: tx

**Value computations.** Element values can combine operands of different
dtypes. The constructor determines the result type from its operands and
inserts an explicit ``ir.prim.Cast`` when they differ, so the conversion
is visible to later passes and code generation.

.. code-block:: python

   i = ir.Var("i", "int32")
   x = ir.Var("x", "float32")
   print(ir.format_tree(i + x, expand=[(ir.Expr, "ty")]))

The output is:

.. code-block:: text

   Add
   ├── ty: PrimType(dtype=float32)
   ├── a: Cast
   │   ├── ty: PrimType(dtype=float32)
   │   └── value: i
   └── b: x

**Vector expressions.** A ``PrimType`` can hold several lanes, such as
``float32x4`` for four float32 values, and arithmetic applies lane by
lane. When one operand is a scalar, the constructor inserts an
``ir.prim.Broadcast`` that repeats it across the lanes.

.. code-block:: python

   v = ir.Var("v", "float32x4")
   print(ir.format_tree(v + 1, expand=[(ir.Expr, "ty")]))

The output is:

.. code-block:: text

   Add
   ├── ty: PrimType(dtype=float32x4)
   ├── a: v
   └── b: Broadcast
       ├── ty: PrimType(dtype=float32x4)
       ├── value: FloatImm(value=1.0)
       │   └── ty: PrimType(dtype=float32)
       └── lanes: IntImm(value=4)
           └── ty: PrimType(dtype=int32)

Operations without dedicated node classes, such as ``ir.prim.exp`` and
``ir.prim.sigmoid``, construct an ``ir.Call`` to a registered operator.
Their results are still primitive expressions.

The :doc:`ir.prim </reference/python/ir.prim>` API reference documents
each node class, its construction helpers, and the available math
functions.

.. _user-ir-tensor-read:

Tensor Reads
~~~~~~~~~~~~

Indexing a tensor variable reads one element. ``A[i]`` constructs an
``ir.TensorLoad`` whose ``source`` is the tensor and whose ``indices``
select the element. Its ``ty`` is the tensor's element type, so reading
from a ``TensorType`` value produces a ``PrimType`` value:

.. code-block:: python

   A, B = add_one.params
   i = ir.Var("i", "int32")
   print(ir.format_tree(A[i], expand=[(ir.Expr, "ty")]))

The output is:

.. code-block:: text

   TensorLoad(source=A, indices=[i])
   └── ty: PrimType(dtype=float32)

Low-level code sometimes needs the tensor's memory address instead.
``A.data`` is a call to ``tirx.tensor_data_ptr`` whose type is an
``ir.PtrType`` with element type float32.

.. _user-ir-tuple:

Tuple
~~~~~

``ir.Tuple`` groups several expressions into one value, for example to
return more than one result. Its ``fields`` hold the expressions, and
its type is an ``ir.TupleType`` whose ``fields`` hold their types in the
same order:

.. code-block:: python

   from tvm_tirx import ir

   i = ir.Var("i", "int32")
   x = ir.Var("x", "float32")
   pair = ir.Tuple([i, x])
   print(ir.format_tree(pair, expand=[(ir.Expr, "ty")]))

The output is:

.. code-block:: text

   Tuple
   ├── ty: TupleType
   │   ├── fields[0]: PrimType(dtype=int32)
   │   └── fields[1]: PrimType(dtype=float32)
   ├── fields[0]: i
   └── fields[1]: x

``ir.TupleGetItem(pair, 1)`` selects one field by position. It records
the tuple in ``tuple_value`` and the position in ``index``, and its type
is the type of the selected field, ``PrimType("float32")``.

.. _user-ir-call:

Call
~~~~

``ir.Call`` expresses a call to another function or to a specialized
operation. Its ``op`` field is the callee, and ``args`` holds the
argument expressions. A specialized operation is a registered operator,
represented by ``ir.Op`` and looked up by name. Like every expression, a
call has a ``ty``, computed from its arguments when the call is
constructed.

For example, sigmoid is a registered operator rather than a dedicated
``ir.prim`` node. A call to it takes a float32 argument and produces a
float32 result:

.. code-block:: python

   from tvm_tirx import ir

   x = ir.Var("x", "float32")
   sigmoid_op = ir.Op.get("prim.sigmoid")
   y = ir.Call(sigmoid_op, [x + 1])
   print(ir.format_tree(y, expand=[(ir.Expr, "ty")],
                        hide=["attrs", "ty_args"]))

The output is:

.. code-block:: text

   Call(op=prim.sigmoid)
   ├── ty: PrimType(dtype=float32)
   └── args[0]: Add
       ├── ty: PrimType(dtype=float32)
       ├── a: x
       └── b: FloatImm(value=1.0)
           └── ty: PrimType(dtype=float32)

**Type arguments and attributes.** Some calls need inputs that are not
value expressions. ``ty_args`` passes types to the callee, and ``attrs``
holds configuration specific to this call. A tensor allocation has no
input from which to infer its shape, so it receives its tensor type in
``ty_args``:

.. code-block:: python

   from tvm_tirx.script import tirx as T

   @T.function
   def scratch(A: T.Tensor((4,), T.float32)):
       tmp = T.alloc_local((4,), T.float32)
       for i in range(4):
           tmp[i] = A[i] + 1

   alloc = scratch.body[0].value
   print(alloc.op.name)
   print(alloc.ty_args[0])

The output is:

.. code-block:: text

   tirx.alloc_tensor
   T.Tensor((4,), T.float32, scope="local")

The allocation's type rule uses this type argument as the call's result
type, so the call produces a tensor. This allocation needs no extra
configuration, so its ``attrs`` is empty.

**Function calls.** The callee can also be another function in the
same module. Here ``main`` calls the helper ``add_one_scalar`` on each
element:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   @T.ir_module
   class Module:
       @T.function(private=True)
       def add_one_scalar(value: T.float32) -> T.float32:
           return value + 1

       @T.function
       def main(A: T.Tensor((4,), T.float32), B: T.Tensor((4,), T.float32)):
           for i in range(4):
               B[i] = Module.add_one_scalar(A[i])

   store = Module["main"].body[0].body[0]
   print(ir.format_tree(store.value, expand=[(ir.Expr, "ty")],
                        hide=["attrs", "ty_args"]))

The output is:

.. code-block:: text

   Call(op=GlobalVar(ty=FuncType(arg_types=[float32], ret_type=float32), name_hint='add_one_scalar'))
   ├── ty: PrimType(dtype=float32)
   └── args[0]: TensorLoad(source=A, indices=[i])
       └── ty: PrimType(dtype=float32)

The call's ``op`` is an ``ir.GlobalVar``, the module's name for the
helper, explained in :ref:`Global Variables <ir-global-var>`. Its type is
the helper's ``FuncType``, and the call takes its result type from that
signature.

.. note::

   Operators are the main extension point of TIRx.
   :doc:`operators_calls_and_regions` covers how operators are registered and
   how their attributes define type rules, effects, and lowering.

.. _user-ir-dependent-type:

Dependent Types
---------------

A type can contain expressions. The shape of a ``tirx.TensorType`` is a
list of expressions, so a tensor's dimensions can be symbolic values
rather than fixed numbers:

.. code-block:: python

   from tvm_tirx import ir
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")
   matrix_type = T.Tensor((n, n * 2), T.float32)
   print(ir.format_tree(matrix_type))

The output is:

.. code-block:: text

   TensorType
   ├── shape[0]: n = Var(ty=int32)
   ├── shape[1]: Mul
   │   ├── a: n
   │   └── b: IntImm(value=2)
   └── dtype: PrimType(dtype=float32)

The first dimension is the int32 variable ``n``, and the second is the
expression ``n * 2``. A type whose fields refer to expressions is called
a dependent type: the concrete shape depends on the value of ``n``.

In a function signature, dependent types relate parameters to each
other. In the add-one function below, both tensors have type
``T.Tensor((n,), T.float32)``:

.. code-block:: python

   n = T.dynamic("n")

   @T.function
   def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           B[i] = A[i] + 1

Both types refer to the same variable ``n``. The first appearance of
``n``, in ``A``'s type, defines it; the appearance in ``B``'s type is a
use. When the function is
called, ``n`` takes the length of ``A``, and ``B`` must have the same
length. The loop body can also use ``n`` as an ordinary int32 value,
here as the loop bound.

TIRx analysis and transformation utilities are designed with dependent
types in mind. When a transformation replaces a variable, it updates
every type that refers to that variable, not only the expressions. For
example, replacing ``n`` with ``16`` in the add-one function changes the
loop bound to ``16`` and also changes the types of ``A`` and ``B`` to
``T.Tensor((16,), T.float32)``. Types and expressions therefore stay
consistent after each rewrite.

Customization
-------------

The type and expression system is open to extension. A library can
define a new type, such as a complex number type with float32 real and
imaginary parts, and register operators that construct and consume
values of that type. Each operator provides a type rule, so a call to it
gets the new type in its ``ty`` just like a built-in expression.

TIRx provides a common foundation for all IR nodes, so new types and
the calls that use them support structural traversal, transformation,
equivalence checking, printing, and parsing out of the box. A library
adds only what is specific to its extension, such as passes that lower
the new operations to what the backend supports.

.. note::

   :doc:`../advanced/defining_ir_extensions` walks through a complete
   extension that defines a complex type, its operators, and a lowering
   pass.
