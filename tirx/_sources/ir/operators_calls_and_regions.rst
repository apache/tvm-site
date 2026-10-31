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


Operators, Calls, and Regions
=============================

This guide introduces **operators**, the main extension point of TIRx.
An operator, ``ir.Op``, names a computation such as sigmoid or a
thread-index read, and carries attributes that tell the compiler how to
treat it. A call applies an operator as an expression, and a region
applies one as a statement. New computations and new kinds of
statements are added by registering operators, without new IR node
classes.

This GPU add-one function uses both forms:

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

   print(ir.format_tree(
       gpu_add_one.body[0],
       hide=["attrs", "args", "ty_args", "body_params", "result_vars",
             "var", "then_case"]))

The output is:

.. code-block:: text

   RegionStmt(op=tirx.device_entry)
   └── body: SeqStmt
       ├── seq[0]: Bind
       │   └── value: Add
       │       ├── a: Mul
       │       │   ├── a: Call(op=tirx.cuda.block_idx)
       │       │   └── b: IntImm(value=256)
       │       └── b: Call(op=tirx.cuda.thread_idx)
       └── seq[1]: If(else_case=None)
           └── condition: LT
               ├── a: i
               └── b: n

The region applies the ``tirx.device_entry`` operator, which marks its
body as GPU code. Inside it, two calls apply the ``tirx.cuda.block_idx``
and ``tirx.cuda.thread_idx`` operators to read the current block and
thread coordinates. The view hides operands and other fields to show
only where operators appear.

Operators
---------

``ir.Op`` is a registered operator, looked up by name with
``ir.Op.get``. Its ``name`` identifies it, and its ``args_info``
describes the arguments it accepts:

.. code-block:: python

   from tvm_tirx import ir

   op = ir.Op.get("prim.sigmoid")
   print(op.name)
   for arg in op.args_info:
       print(arg.name, "-", arg.doc)

The output is:

.. code-block:: text

   prim.sigmoid
   x - The input value.

Each operator is a single shared object. A call stores the operator in
its ``op`` field and its own operands in ``args``, so every call to
sigmoid refers to this same ``op``. When a call is constructed, the
operator's type rule computes the call's ``ty`` from its arguments.

.. note::

   :doc:`types_and_expressions` covers the fields of a call, including
   ``ty_args``, ``attrs``, and calls to functions.

Operator Attributes
-------------------

An operator's behavior comes from its **attributes**: named values
attached to the operator, such as its type rule, its side effects, or
how to lower it for a target. Attributes are stored by name. All values
for one name form a map from operators to values, which
``ir.Op.get_attr_map`` returns. ``op.get_attr(name)`` reads one
operator's value. For example, ``TCallEffectKind`` records whether a
call only computes a value or also changes state:

.. code-block:: python

   from tvm_tirx import ir

   effect = ir.Op.get_attr_map("TCallEffectKind")
   for name in ["prim.sigmoid", "tirx.masked_store"]:
       kind = ir.CallEffectKind(effect[ir.Op.get(name)])
       print(name, kind.name)

The output is:

.. code-block:: text

   prim.sigmoid Pure
   tirx.masked_store UpdateState

Sigmoid is ``Pure``, so a pass may remove a call whose result is
unused. A masked store is ``UpdateState``: it writes memory, so it must
be kept.

Attributes are extensible. Any library can introduce a new attribute
name and set it on any operator, built-in or new, for example
``ir.Op.get("prim.sigmoid").set_attr("my_pass.Cost", 4)``. Passes find
operators through attributes, never through a list of operator names.
To lower calls for a target, TIRx looks each operator up under keys
built from the target name, such as ``cuda.FLowerIntrinsic``, and falls
back to ``default`` keys such as ``default.FLegalize``. Extension therefore works in both
directions: a new operator works with existing passes once it sets the
attributes those passes read, and a new pass can define its own
attribute and set it on existing operators.

The common attributes are:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Attribute
     - Meaning
   * - ``FInferType``
     - Computes a call's result type from its arguments.
   * - ``TFixedReturnType``
     - A result type that does not depend on the arguments.
   * - ``TCallEffectKind``
     - Whether a call is pure, reads state, or updates state.
   * - ``<target>.FLowerIntrinsic``
     - Lowers a call to code for one target, such as ``llvm`` or
       ``cuda``.
   * - ``<target>.FLegalize``
     - Rewrites a call into simpler operations, for one target or with
       ``default`` for all targets.
   * - ``TVectorizable``
     - Whether a call can be applied lane by lane to vectors.
   * - ``TGlobalSymbol``
     - The name of the external function the call maps to.
   * - ``FRegionGetBodyParams``, ``FRegionValidate``
     - Make the operator a region operator: define the body's
       parameters and check the region's operands.

Defining an Operator
--------------------

``ir.Op.register`` adds a new operator. The operator starts with no
signature and no attributes, so the definition sets both. This operator
squares a value:

.. code-block:: python

   from tvm_tirx import ir

   op = ir.Op.register("my_ext.square", "Square a value.")
   op.set_signature(args=[("x", "The value to square.")])
   op.set_attr("TCallEffectKind", ir.CallEffectKind.Pure.value)
   op.set_attr("FInferType", lambda call: call.args[0].ty)

The signature declares one argument. ``TCallEffectKind`` marks the
operator as pure, and ``FInferType`` gives a call the type of its
argument. A call to the new operator now gets its type from that rule:

.. code-block:: python

   v = ir.Var("v", "float32x4")
   squared = ir.Call(op, [v])
   print(ir.format_tree(squared, expand=[(ir.Expr, "ty")],
                        hide=["attrs", "ty_args"]))

The output is:

.. code-block:: text

   Call(op=my_ext.square)
   ├── ty: PrimType(dtype=float32x4)
   └── args[0]: v

The signature also lets ``Call.validate()`` reject a call with the
wrong number of arguments: ``ir.Call(op, [], ty="float32").validate()``
raises ``TypeError``.

Once its signature and attributes are set, ``op.expose()`` publishes
the operator as Python helpers in the namespaces that serve its name
prefix. For example, an operator named ``prim.<name>`` becomes available
as ``ir.prim.<name>`` and, in TVMScript, as ``T.<name>``.

.. note::

   To compile calls to a new operator, it also needs lowering
   attributes, such as ``default.FLegalize``, that rewrite it into
   operations a backend supports.
   :doc:`../advanced/defining_ir_extensions` walks through a complete
   extension with its own type, operators, and lowering pass.

Regions
-------

A region applies an operator to a nested body of statements. An
operator becomes a region operator by setting two attributes:
``FRegionGetBodyParams`` defines the variables its body receives, and
``FRegionValidate`` checks its operands. ``ir.RegionStmt`` records one
use of such an operator: ``op``, the operands ``args`` and ``attrs``,
the ``body``, and any variables the region defines in ``body_params``
and ``result_vars``.

``tirx.device_entry`` is a region operator that enters a GPU kernel. Its
operands carry the kernel's launch configuration:

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
   print(ir.format_tree(region, hide=["body"]))

The output is:

.. code-block:: text

   RegionStmt(op=tirx.device_entry)
   ├── attrs: DictAttrs
   │   └── dict: Map
   │       ├── 'cuda.kernel_attrs': {}
   │       ├── 'cuda.launch_fields'[0]: 'grid.x'
   │       ├── 'cuda.launch_fields'[1]: 'grid.y'
   │       ├── 'cuda.launch_fields'[2]: 'grid.z'
   │       ├── 'cuda.launch_fields'[3]: 'block.x'
   │       ├── 'cuda.launch_fields'[4]: 'block.y'
   │       └── 'cuda.launch_fields'[5]: 'block.z'
   ├── args[0]: FloorDiv
   │   ├── a: Sub
   │   │   ├── a: Add
   │   │   │   ├── a: n
   │   │   │   └── b: IntImm(value=256)
   │   │   └── b: IntImm(value=1)
   │   └── b: IntImm(value=256)
   ├── args[1]: IntImm(value=1)
   ├── args[2]: IntImm(value=1)
   ├── args[3]: IntImm(value=256)
   ├── args[4]: IntImm(value=1)
   ├── args[5]: IntImm(value=1)
   ├── body_params: []
   └── result_vars: []

The ``args`` are the six launch values, and ``cuda.launch_fields`` in
``attrs`` names each one: the grid has ``ceildiv(n, 256)`` blocks along
x, and each block has 256 threads. The region's body, hidden here, runs
on the GPU with that configuration. ``device_entry`` defines no
variables, so ``body_params`` and ``result_vars`` are empty.
