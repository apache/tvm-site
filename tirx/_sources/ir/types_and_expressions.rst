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

In this chapter, we will introduce types and expressions. Expressions
represent values and computations, and their types describe properties
such as dtypes, tensor shapes, and function signatures. The add-one
examples will show how to inspect these objects and construct expressions
directly with the IR API.

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

:class:`ir.PrimType <tvm_tirx.ir.PrimType>` describes integer,
floating-point, and boolean values, either as scalars or vectors. Its
``dtype`` records the element kind, bit width, and number of lanes.

In the program's tensor annotations, ``T.float32`` is an
``ir.PrimType("float32")`` object.

The program declares its dynamic length as:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at: n = T.dynamic

where ``n`` is a 32-bit integer symbol with type ``ir.PrimType("int32")``.
The loop index ``i`` has the same primitive type.

We can express the addition ``A[i] + 1`` directly through the IR API,
with a scalar variable standing for the input element:

.. code-block:: python

   from tvm_tirx import ir
   scalar_value = ir.Var("scalar_value", "float32")
   scalar_plus_one = scalar_value + 1

``scalar_value`` has type ``ir.PrimType("float32")``, so it represents a single
32-bit floating-point value. The result ``scalar_plus_one`` has the
same type, just like ``A[i] + 1`` in the program.

.. _user-ir-vector-expression:

``PrimType`` also supports vector values. We can extend the same addition
to four float32 values at once:

.. code-block:: python

   vector_value = ir.Var("vector_value", "float32x4")
   vector_plus_one = vector_value + 1

The type of ``vector_value`` is ``ir.PrimType("float32x4")``. The ``x4`` suffix
specifies four float32 lanes within one vector value. The addition adds
one to each lane, and ``vector_plus_one`` has the same vector type.

.. _user-tensor-type:

TensorType
~~~~~~~~~~

:class:`tirx.TensorType <tvm_tirx.tirx.TensorType>` is an IR type object
that describes a whole tensor. It records the tensor's dimensions and
the type of each element.

The program declares its input and output as:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: def add_one(
   :end-at: def add_one(

``T.Tensor((n,), T.float32)`` is the TVMScript notation for a tensor type;
the parser represents it as a ``tirx.TensorType`` object. Its ``shape``
field contains one dimension, ``n``, and its ``dtype`` field is
``ir.PrimType("float32")``:

.. code-block:: text

   tirx.TensorType                  T.Tensor((n,), T.float32)
   ├── shape[0]: ir.Var              n
   └── dtype: ir.PrimType            float32

The dimension ``n`` is symbolic: it is not fixed to a particular number
when the function is compiled. Each call determines ``n`` from the length
of ``A``. Since ``B`` uses the same symbol in its type, its length must
match ``A``. The loop then uses ``n`` as its bound.

Expressions
-----------

``ir.Expr`` is the base class for expressions, which represent program
values and computations. All expression classes inherit its ``ty`` field,
which records the type of value they represent:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``ty``
     - ``ir.Type``
     - The type of value represented by the expression.

We can inspect the types of ``scalar_value`` and ``scalar_plus_one``
through ``ty``. Both are primitive types, so their ``dtype`` fields give
the dtype names:

.. code-block:: python

   print(scalar_value.ty.dtype)
   print(scalar_plus_one.ty.dtype)

The output is:

.. code-block:: text

   float32
   float32

The variable's type was supplied by the ``"float32"`` argument in
``ir.Var("scalar_value", "float32")``. The addition's constructor
determines its result type from its operands; this process is called
type inference.

An expression under construction can also have an ``ir.MissingType``
in ``ty`` while its type is undetermined. The
:ref:`Call Types and Validation <types-and-validation>` section shows
how a missing result type can be inferred later.

Each expression class adds fields for its value or operands, alongside
the shared ``ty`` field:

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

.. _user-ir-const:

Primitive Expressions
~~~~~~~~~~~~~~~~~~~~~

A primitive expression is an expression whose ``ty`` is an ``ir.PrimType``.
This term groups expressions by their result type. Variables, tensor
reads, arithmetic nodes, and calls can all belong to this group when
they produce primitive scalar or vector values.

**Scalar expressions.** The CPU example uses the int32 expressions ``n``
and ``i`` for the loop bound and index. The tensor read ``A[i]`` and the
addition ``A[i] + 1`` both produce a single float32 value. In the GPU
example, ``i < n`` produces a boolean that decides whether a thread
writes its result. These scalar result types are stored as ``PrimType``
objects in ``ty``.

``A`` itself represents the whole tensor and has a ``TensorType``.
The expression ``A[i]`` reads element ``i`` of this tensor. TVMScript
represents the read as an ``ir.TensorLoad`` node. Its ``source`` field
refers to ``A``, and ``indices`` contains ``i``, selecting the element
to read:

.. code-block:: text

   ir.TensorLoad                    A[i]
   ├── ty: ir.PrimType               float32
   ├── source: ir.Var                A
   │   └── ty: tirx.TensorType       (n,) of float32
   └── indices[0]: ir.Var            i

**Vector expressions.** A primitive expression can also produce a vector
value. The earlier add-one result ``vector_plus_one`` has type
``ir.PrimType("float32x4")``, representing four float32 lanes within
one value.

A call whose result type is a ``PrimType`` also belongs to this category.
For example, the GPU expression ``T.cuda.thread_idx("x")`` is an
``ir.Call`` with result type ``ir.PrimType("int32")``.

.. _user-ir-prim-operations:

Primitive Operations
~~~~~~~~~~~~~~~~~~~~

The previous section grouped expressions by result type. The operations
below describe the computations that produce those values. The
``ir.prim`` namespace provides node classes and Python helpers for
arithmetic, comparisons, boolean logic, casts, and other primitive
computations:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Python expression
     - IR node constructed
   * - ``scalar_value + 1``
     - ``ir.prim.Add``
   * - ``ir.prim.sigmoid(scalar_value)``
     - ``ir.Call`` to ``prim.sigmoid``

Python operators such as ``+``, ``*``, and ``<`` construct arithmetic
and comparison nodes. The Python helper ``ir.prim.sigmoid`` constructs
a call to a registered operator; the returned IR node is an ``ir.Call``.
The :ref:`Call <user-ir-call>` section below examines its fields.
The :doc:`ir.prim </reference/python/ir.prim>` API reference lists the
available classes and construction helpers.

In the CPU example, ``A[i] + 1`` becomes an ``ir.prim.Add``. A binary node
stores its left operand in ``a`` and its right operand in ``b``:

.. code-block:: text

   ir.prim.Add                  A[i] + 1
   ├── a: ir.TensorLoad          A[i]
   └── b: ir.FloatImm            1.0

The arithmetic constructor derives the result type from its operands.
Since ``A[i]`` is float32, it converts the source literal ``1`` to the
float32 constant shown as ``ir.FloatImm``. The result is also float32.

These constructors also support vector operands. In ``vector_value + 1``,
the scalar constant is broadcast to all four lanes, producing the
four-lane expression ``vector_plus_one``.

The GPU example combines multiplication, addition, and comparison to
compute and check each thread's element index:

.. literalinclude:: ../get_started/examples/add_one_cuda.py
   :language: python
   :start-at:     i = T.cuda.block_idx
   :end-at:         B[i] = A[i] + 1
   :dedent: 4

The GPU launch uses 256 threads per block. The multiplication by 256
becomes an ``ir.prim.Mul``, and the sum becomes an ``ir.prim.Add``.
The condition ``i < n`` is an ``ir.prim.LT`` whose ``a`` and ``b``
operands are the index and tensor length:

.. code-block:: text

   ir.prim.LT                       i < n
   ├── a: ir.Var                    i
   └── b: ir.Var                    n

The comparison checks whether the thread has a valid element to process.

.. _user-ir-var:

Var
~~~

``ir.Var`` represents a named value with a type. Variables can be
function parameters, loop indices, or intermediate values defined by
bindings.

The tensor parameters ``A`` and ``B`` in the program are represented by
``ir.Var`` objects. The PyTorch tensors ``a`` and ``b`` are passed to these
parameters when the compiled function is called. In the CPU example, we
can retrieve the parameter variables from ``add_one.params`` and inspect
their types through ``ty``:

.. code-block:: python

   A, B = add_one.params
   input_type = A.ty
   output_type = B.ty

Each parameter is a ``Var`` with its own ``name`` and ``ty`` fields.
For ``A``, these are:

.. code-block:: text

   A: ir.Var
   ├── name: "A"
   └── ty: tirx.TensorType           (n,) of float32

TVMScript creates these parameter variables when it parses the function.
We can also create variables directly with ``ir.Var(name, ty)``. The
scalar add-one example in PrimType uses this constructor:

.. code-block:: python

   from tvm_tirx import ir
   scalar_value = ir.Var("scalar_value", "float32")
   scalar_plus_one = scalar_value + 1

``ir.Var("scalar_value", "float32")`` creates a variable whose ``name``
is ``"scalar_value"`` and whose ``ty`` is ``ir.PrimType("float32")``.

.. _user-ir-call:

Call
~~~~

:class:`ir.Call <tvm_tirx.ir.Call>` represents a call to an operator or
function. Its ``op`` field refers to the operator or function expression
being called, and ``args`` holds the argument expressions. A registered
operator is represented by ``ir.Op``.

In the GPU example, the index expression contains calls to
``T.cuda.block_idx("x")`` and ``T.cuda.thread_idx("x")``.
These TVMScript helpers construct calls to registered CUDA operators.
We can construct the thread-index call directly through the IR API:

.. code-block:: python

   from tvm_tirx import ir

   op = ir.Op.get("tirx.cuda.thread_idx")
   x_call = ir.Call(op, [ir.StringImm("x")])

``op`` is the ``ir.Op`` object for ``tirx.cuda.thread_idx``. The call's
``op`` field refers to this object, and its ``args`` array contains the
string expression ``"x"``. Its ``ty`` is ``ir.PrimType("int32")``:

.. code-block:: text

   x_call: ir.Call
   ├── op → ir.Op                   tirx.cuda.thread_idx
   ├── args[0]: ir.StringImm         "x"
   └── ty: ir.PrimType               int32

The ``"x"`` argument selects the x axis. When the compiled GPU function
runs, this call reads the current thread's x-axis coordinate within its
block. The index arithmetic uses the coordinates returned by the
thread-index and block-index calls to compute ``i``.

A common extension of add-one is an activation applied to its result.
We can pass the earlier addition expression ``scalar_plus_one`` to the
built-in ``prim.sigmoid`` operator:

.. code-block:: python

   sigmoid_op = ir.Op.get("prim.sigmoid")
   activated_plus_one = ir.Call(sigmoid_op, [scalar_plus_one])

The call's argument is now the addition expression:

.. code-block:: text

   activated_plus_one: ir.Call
   ├── op → ir.Op                   prim.sigmoid
   ├── args[0]: ir.prim.Add          scalar_value + 1
   └── ty: ir.PrimType               float32

``activated_plus_one.args[0]`` refers to ``scalar_plus_one``, so sigmoid
receives the addition's result. The expression computes
``1 / (1 + exp(-(scalar_value + 1)))``. The operator's type rule gives
the call the same type as its input, ``ir.PrimType("float32")``. This
call is therefore also a primitive expression. The Python helper
``ir.prim.sigmoid(scalar_plus_one)`` constructs the same call.

The :doc:`operators_and_calls` chapter also covers calls to IR functions
and explains how registered operators provide result-type inference
and argument validation.

.. _user-ir-source-locations:

Source Locations
----------------

IR nodes can carry source locations to help diagnostics point back to
the original code. The ``loc`` field stores this information, and
:class:`ir.SourceLoc <tvm_tirx.ir.SourceLoc>` records the source file
and range.

The parameter ``A`` is declared on this line of the CPU program:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
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
   print(f"Parameter {parameter.name} is declared on line {line} of the CPU program.")

The output is:

.. code-block:: text

   Parameter A is declared on line 29 of the CPU program.

``source_name.name`` also records the path to the CPU program's source
file:

.. code-block:: text

   parameter: ir.Var                 A
   └── loc: ir.SourceLoc
       ├── source_name: ir.SourceName
       │   └── name                  path to the CPU program's source file
       └── start_line                line of A's declaration
