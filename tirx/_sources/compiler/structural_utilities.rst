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

Traversing and Transforming IR
==============================

Compiler passes often need to find variable uses, check a property of a
subtree, or rewrite an expression. TVM FFI's structural operations follow
the child relationships defined by IR types so these tasks can be written
as callbacks. A callback, also called a handler here, is a Python function
that runs when the traversal reaches a matching value.

The table below summarizes the four APIs and their roles:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Task
     - Automatic recursion
     - Callback-controlled recursion
   * - Inspect values
     - ``tvm_ffi.structural_walk``
     - ``tvm_ffi.structural_visit``
   * - Replace values
     - ``tvm_ffi.structural_map``
     - ``tvm_ffi.structural_mutate``

These operations inspect and rewrite IR without executing the programs it
describes. Walk and map handle recursion, letting callbacks focus on
inspecting or replacing nodes. Visit and mutate give a matching callback
control over recursive calls, so it can choose which children to process,
change their order, and manage state before and after each call.

We will use the expression ``(x + 1) * (y + 2)`` to compare these workflows.
Each of the four API sections creates its own variables and expression,
so it can be run independently. Within a section, run the examples in
order: later examples reuse the objects and helper functions defined
earlier.

Structural Walk
---------------

``tvm_ffi.structural_walk`` automatically traverses IR. Its callbacks can
collect nodes, check properties, or stop the traversal when a matching value
is found. The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_walk(
       root, callbacks=(), with_def_region_kind=(), order="post")

Walking an Expression
~~~~~~~~~~~~~~~~~~~~~

``root`` is the value to traverse, such as an IR expression, a function,
or an FFI container. A ``(Type, handler)`` pair in ``callbacks`` specifies
the type to match and a function to call for each matching value. The
following example uses a callback to print the variable names in
``(x + 1) * (y + 2)``:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

   def on_var(var):
       print(var.name)

   tvm_ffi.structural_walk(root=expr, callbacks=(ir.Var, on_var))

The output is:

.. code-block:: text

   x
   y

To see how the walker reaches these variables, we can print the
expression's IR structure with
:func:`ir.format_tree <tvm_tirx.ir.format_tree>`:

.. code-block:: python

   print(ir.format_tree(expr))

The output is:

.. code-block:: text

   Mul
   ├── a: Add
   │   ├── a: x
   │   └── b: IntImm(value=1)
   └── b: Add
       ├── a: y
       └── b: IntImm(value=2)

The root ``Mul`` represents multiplication. Binary arithmetic nodes such
as ``Mul`` and ``Add`` store their left operand in ``a`` and their right
operand in ``b``. For example, in the ``Add`` representing ``x + 1``,
``a`` is the variable ``x`` and ``b`` is the integer constant ``1``.

.. _ir-default-structural-traversal:

The walker descends from ``Mul`` through the two ``Add`` nodes to their
operands without needing callbacks for those arithmetic nodes. Each
IR node type defines which children to visit, and the walker follows these
rules automatically. This is the default structural traversal.
``callbacks=(ir.Var, on_var)`` determines where ``on_var`` runs during that
traversal: only ``x`` and ``y`` match ``ir.Var``, so the walker calls
``on_var(x)`` and ``on_var(y)`` to print their names.

Choosing Callbacks
~~~~~~~~~~~~~~~~~~

The following table lists the accepted callback forms, using ``on_var`` from
the example and Python's built-in ``print`` function:

.. list-table::
   :header-rows: 1
   :widths: 48 52

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, on_var)``
     - Variables are passed to ``on_var``.
   * - ``((ir.prim.Add, ir.prim.Mul), print)``
     - Additions and multiplications are passed to ``print``, one node per call.
   * - ``[(ir.Var, on_var), (ir.IntImm, print)]``
     - A Python list of two ``(Type, handler)`` pairs. Variables are passed to
       ``on_var``, and integer IR constants are passed to ``print``.
   * - ``((ir.Var, on_var), (ir.IntImm, print))``
     - The same two pairs in a Python tuple, with the same matching order.
   * - ``print``
     - Every visited value is passed to ``print``, with no type filter.
   * - ``()``
     - The empty tuple selects no handlers. The walker still visits
       child nodes. This is the default when ``callbacks`` is omitted.

A class in ``callbacks`` also matches objects of its subclasses. Since
``ir.Var`` is a subclass of ``ir.Expr``, both pairs below match ``x``:

.. code-block:: python

   selected = []
   tvm_ffi.structural_walk(
       root=x,
       callbacks=[
           (ir.Var, lambda var: selected.append("variable")),
           (ir.Expr, lambda node: selected.append("expression")),
       ],
   )
   print(selected)

The output is ``['variable']``. The walker checks the pairs in order and
calls only the first matching handler. If the ``ir.Expr`` pair came first,
the expression handler would run instead.
Returning ``None`` continues traversal. It does not try the next handler.

Traversal Order
~~~~~~~~~~~~~~~

``order`` controls when a matching handler runs. With ``order="post"``
(the default), it runs after the node's children have been visited. With
``order="pre"``, it runs before them. The corresponding enum values are
``tvm_ffi.WalkOrder.POSTORDER`` and ``tvm_ffi.WalkOrder.PREORDER``.

We can compare the two orders by recording a short label for each visited
expression. The ``describe`` function below creates these labels, and
``ir.Expr`` matches variables, integer constants, and arithmetic nodes:

.. code-block:: python

   def describe(node):
       if isinstance(node, ir.Var):
           return node.name
       if isinstance(node, ir.IntImm):
           return str(int(node))
       return type(node).__name__

   for order in ("pre", "post"):
       trace = []
       tvm_ffi.structural_walk(
           expr,
           (ir.Expr, lambda node: trace.append(describe(node))),
           order=order,
       )
       print(order, trace)

The output is:

.. code-block:: text

   pre ['Mul', 'Add', 'x', '1', 'Add', 'y', '2']
   post ['x', '1', 'Add', 'y', '2', 'Add', 'Mul']

The pre-order trace records ``Mul`` before its ``Add`` children. The
post-order trace records it after them. In both cases, each binary
expression's left operand ``a`` is visited before its right operand ``b``.

Skipping and Stopping
~~~~~~~~~~~~~~~~~~~~~

A callback can also control whether traversal continues, skips children,
or stops entirely. Its return value determines what happens next:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Callback returns
     - Effect
   * - ``None`` or ``tvm_ffi.WalkResult.ADVANCE``
     - The walk continues.
   * - ``tvm_ffi.WalkResult.SKIP``
     - In pre-order, this value's children are omitted, and traversal continues
       elsewhere.
   * - ``tvm_ffi.VisitInterrupt(payload)``
     - The whole walk stops and returns the interrupt to the caller.

``structural_walk`` returns ``None`` on completion or ``VisitInterrupt`` on
early exit. The interrupt's ``value`` field holds the payload from the
callback.

A pre-order callback can return ``WalkResult.SKIP`` to omit the current
node's children. We can skip the children of ``x + 1`` while continuing
through the rest of the expression. ``node.a.same_as(x)`` checks whether
an addition's left operand is the original variable ``x``, using object
identity rather than its printed name:

.. code-block:: python

   trace = []

   def skip_left_add(node):
       trace.append(describe(node))
       if isinstance(node, ir.prim.Add) and node.a.same_as(x):
           return tvm_ffi.WalkResult.SKIP
       return None

   tvm_ffi.structural_walk(expr, (ir.Expr, skip_left_add), order="pre")
   print(trace)

The output is:

.. code-block:: text

   ['Mul', 'Add', 'Add', 'y', '2']

The callback records the first ``Add`` before returning ``SKIP``, so that
node appears in the trace while its children, ``x`` and ``1``, do not.
Traversal continues through the right subtree. Skipping children requires
pre-order traversal, because a post-order callback runs after the children
have already been visited.

A search can stop as soon as it finds a match. The following callback
returns ``VisitInterrupt(var)`` when it reaches ``y``:

.. code-block:: python

   def find_y(var):
       if var.same_as(y):
           return tvm_ffi.VisitInterrupt(var)
       return None

   found = tvm_ffi.structural_walk(expr, (ir.Var, find_y))
   if found is not None:
       print(found.value.name)

Returning ``VisitInterrupt(var)`` stops the walk and makes that interrupt
the return value of ``structural_walk``. Here, ``found.value`` is the
matching variable ``y``, so the output is ``y``. The walk stops before
visiting the remaining constant ``2``.

.. _ir-ffi-containers:

Containers and Shared Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A walk can also start from a collection of IR nodes. We use TVM FFI's data
structures here: ``tvm_ffi.Array`` for sequences and ``tvm_ffi.Map`` for
key-value mappings. The following example collects variables from an array
stored inside a map:

.. code-block:: python

   container_nodes = tvm_ffi.Array([x, y])
   container_groups = tvm_ffi.Map({"inputs": container_nodes})
   container_names = []

   def collect_container_var(var):
       container_names.append(var.name)

   tvm_ffi.structural_walk(
       root=container_groups, callbacks=(ir.Var, collect_container_var)
   )
   print(container_names)

The output is:

.. code-block:: text

   ['x', 'y']

The walk follows the map's ``"inputs"`` value to the array, then visits
``x`` and ``y``. The string key ``"inputs"`` is skipped.

The same rules apply to ``tvm_ffi.List`` and ``tvm_ffi.Dict``.

An expression can use the same IR object in multiple positions. The following
example checks whether the walker visits that object once or separately at
each position:

.. code-block:: python

   shared = x + 1
   twice = ir.prim.Mul(shared, shared)
   additions = []
   tvm_ffi.structural_walk(twice, (ir.prim.Add, additions.append))
   print("Add visits:", len(additions))
   print("Same object:", additions[0].same_as(additions[1]))

The output is:

.. code-block:: text

   Add visits: 2
   Same object: True

``shared = x + 1`` creates one ``Add`` object. ``twice`` uses it as both
operands of a multiplication, representing ``(x + 1) * (x + 1)``. The walker
reaches that object through each operand, so ``additions.append`` records
it twice even though both list entries refer to the same object.

.. _ir-definition-regions:

Definitions and Uses
~~~~~~~~~~~~~~~~~~~~

Repeated visits to a variable can have different roles: one position may
define it, while another uses it. An analysis that checks for unused
variables needs to distinguish these roles. For example, a Python
calculation might use a temporary like this:

.. code-block:: python

   tmp = (x + 1) * (y + 2)
   result = tmp + tmp

The first line defines ``tmp``. The second line uses it twice.
``ir.prim.Let(var, value, body)`` describes this pattern in IR: compute
``value``, name its result ``var``, then compute ``body`` using that
variable. The result of ``body`` is the result of the whole ``Let``
expression. Using our earlier ``expr`` for the intermediate calculation:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(
       tmp,        # var: the temporary variable
       expr,       # value: (x + 1) * (y + 2)
       tmp + tmp,  # body: use the temporary variable twice
   )

``ir.Var`` creates the variable object. The ``Let`` node makes ``tmp``
available within ``body``. Creating these IR objects does not execute the
calculation.

There are three occurrences of the same ``tmp`` object: the definition in
``Let.var`` and the two uses in ``Let.body``. Counting all three visits
would overcount the uses. An ordinary callback receives only the variable,
so it cannot distinguish these roles from the object alone.

To count only the uses, register the callback through
``with_def_region_kind``. The walker then calls ``handler(var, kind)``,
where ``kind`` is a ``tvm_ffi.DefRegionKind`` value describing the role at
that position. In this ``Let``, ``DEF_SIMPLE`` marks the definition of
``tmp`` in ``var``, and ``NONE`` marks each use in ``body``. The callback
below focuses on ``tmp`` and records only its uses:

.. code-block:: python

   tmp_uses = []

   def record_tmp(var, kind):
       if var.same_as(tmp):
           print(var.name, kind.name)
           if kind == tvm_ffi.DefRegionKind.NONE:
               tmp_uses.append(var)

   tvm_ffi.structural_walk(
       scoped_expr,
       with_def_region_kind=(ir.Var, record_tmp),
   )
   print("tmp uses:", len(tmp_uses))

The output is:

.. code-block:: text

   tmp DEF_SIMPLE
   tmp NONE
   tmp NONE
   tmp uses: 2

The identity check excludes ``x`` and ``y`` from the report. The callback
prints all three visits to ``tmp`` but adds only the two ``NONE`` visits
to ``tmp_uses``. A zero use count would let a checker flag an unused
temporary.

.. _ir-pattern-definition-regions:

Defining ``tmp`` introduces only that variable, whose type is ``int32``.
A tensor's type can itself contain variables describing its shape. A
tensor parameter can introduce these variables along with the tensor,
as in this function that returns its input length:

.. code-block:: python

   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def tensor_length(A: T.Tensor((n,), T.float32)) -> T.int32:
       return n

``T.dynamic("n")`` creates a symbolic length. The annotation describes
``A`` as a one-dimensional float32 tensor of that length. For example, if
the input has shape ``(100,)``, ``n`` represents 100 for that call. The
parameter declaration introduces both the tensor variable ``A`` and its
length ``n``, and ``return n`` reads that length.

A ``DEF_PATTERN`` definition introduces the variable and also treats
variables encountered in its type as definitions. We can use
``with_def_region_kind`` again to see the roles in the parameter
declaration and the return statement:

.. code-block:: python

   def show_parameter_role(var, kind):
       print(var.name, kind.name)

   tvm_ffi.structural_walk(
       tensor_length,
       with_def_region_kind=(ir.Var, show_parameter_role),
   )

The output is:

.. code-block:: text

   n DEF_PATTERN
   A DEF_PATTERN
   n NONE

The first two lines come from the parameter declaration: both ``A`` and
the ``n`` in its shape are definitions. The default post-order traversal
reports ``n`` in the type before ``A`` itself. The last line comes from
``return n``, where ``n`` is used.

A local tensor variable ``B`` can instead have a type that refers to a
length ``n`` defined earlier. Its binding uses the ``DEF_SIMPLE`` rule:
only ``B`` is defined, while the ``n`` in its type is a use and receives
``NONE``.
The two kinds let an analysis distinguish a shape variable introduced by
a definition from a dependency on an existing variable. Each IR node's
traversal rules determine which kind applies. Function parameters are one
example of where the IR uses ``DEF_PATTERN``.

If a handler in ``callbacks`` also matches the variable, it runs instead
of the handler in ``with_def_region_kind``. Register the variable handler
only through ``with_def_region_kind`` when it needs ``kind``.

Structural Map
--------------

A walk can tell us which variables occur in an expression. To replace
those variables, we can use ``tvm_ffi.structural_map``, whose callbacks
return replacements for matching nodes. When a child changes, the mapper
builds its parent with the updated child and returns the rewritten root.
A callback can therefore replace a variable without constructing the
whole expression that contains it. The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_map(
       root, callbacks=(), with_def_region_kind=(), order="post")

Replacing Variables
~~~~~~~~~~~~~~~~~~~

Suppose we know the values of ``x`` and ``y`` in ``(x + 1) * (y + 2)``.
We can replace these variables with IR constants to obtain an expression
specialized for those values:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

   def specialize_var(var):
       if var.same_as(x):
           return ir.const(8, var.ty)
       if var.same_as(y):
           return ir.const(3, var.ty)
       return var

   bound = tvm_ffi.structural_map(root=expr, callbacks=(ir.Var, specialize_var))
   print(bound)
   print(expr)

The output is:

.. code-block:: text

   T.Add(8, 1) * T.Add(3, 2)
   (x + 1) * (y + 2)

The callback uses ``same_as`` to identify the original variables.
``ir.const(8, var.ty)`` creates the constant 8 with the variable's type,
and ``return var`` keeps any other variable unchanged.

The mapper keeps ``expr`` unchanged by constructing new ``Add`` nodes for
``8 + 1`` and ``3 + 2``, then a ``Mul`` using those additions. It returns
the new expression as ``bound``. The arithmetic operations are still IR
nodes: replacing the variables does not itself compute ``8 + 1`` or
``3 + 2``, which is why the result prints those additions as ``T.Add``.

For an expression field, a callback must return an IR expression. Return
the input expression to keep it unchanged. ``None`` is not a valid
replacement.

Choosing Callbacks
~~~~~~~~~~~~~~~~~~

The mapper still processes a node's children when no handler matches that
node. This is why ``(ir.Var, specialize_var)`` alone is sufficient for the
variable replacement above. Default recursion also processes elements of
FFI sequences and values of FFI mappings.

The following table summarizes callback forms. A Python list or tuple
can combine several ``(Type, handler)`` pairs:

.. list-table::
   :header-rows: 1
   :widths: 52 48

   * - ``callbacks`` argument
     - Meaning
   * - ``(ir.Var, specialize_var)``
     - Run the variable replacement rule.
   * - ``((ir.prim.Add, ir.prim.Mul), lambda node: node)``
     - Additions and multiplications share one handler.
   * - ``[(ir.Var, specialize_var), (ir.Expr, lambda node: node)]``
     - Variables use the first handler. Other expressions use the second.
   * - ``lambda value: value``
     - One function matches every reached value and returns it.
   * - ``()``
     - No callbacks run. Default mapping still processes children.

Classes match their subclasses, and only the first matching handler runs.
In the list above, ``ir.Var`` must come before ``ir.Expr`` for variable
replacement to run. Reversing them would send variables to the lambda,
which returns them unchanged.

Mapping Order
~~~~~~~~~~~~~

After replacing the variables, ``bound`` represents
``(8 + 1) * (3 + 2)``. We can reduce it to 45 by computing operations whose
operands are constants. This is constant folding. To finish in one
traversal, we need the additions to become 9 and 5 before computing the
multiplication. The following rule handles both kinds of operation:

.. code-block:: python

   def fold_arithmetic(node):
       if not isinstance(node.a, ir.IntImm) or not isinstance(node.b, ir.IntImm):
           return node
       lhs, rhs = int(node.a), int(node.b)
       if isinstance(node, ir.prim.Add):
           value = lhs + rhs
       else:
           value = lhs * rhs
       return ir.const(value, node.ty)

   arithmetic_types = (ir.prim.Add, ir.prim.Mul)
   folded = tvm_ffi.structural_map(
       bound, callbacks=(arithmetic_types, fold_arithmetic), order="post"
   )
   folded_pre = tvm_ffi.structural_map(
       bound, callbacks=(arithmetic_types, fold_arithmetic), order="pre"
   )
   print("post:", folded)
   print("pre:", folded_pre)

The output is:

.. code-block:: text

   post: 45
   pre: T.Mul(9, 5)

With ``order="post"`` (the default), the mapper processes children before
calling their parent's handler. The two additions become 9 and 5, so the
multiplication's handler receives constant operands and returns 45.

With ``order="pre"``, the multiplication's handler runs first. Its operands
are still additions, so it returns the node unchanged. The mapper then
processes the returned node's children, folding the additions to 9 and 5.
It does not call the multiplication's handler again afterward.

The ``arithmetic_types`` filter ensures that ``fold_arithmetic`` receives
only additions and multiplications, both of which have ``a`` and ``b``
operands. ``int(node.a)`` reads a constant's Python integer value, and
``ir.const(value, node.ty)`` converts the result back into typed IR.

The order can also be written as ``tvm_ffi.WalkOrder.POSTORDER`` or
``tvm_ffi.WalkOrder.PREORDER``. In either order, the mapper applies a handler
once at each position. A post-order replacement is final and is not
traversed again. Reaching a result that needs no further rewrites may
require another map or a separate simplification pass.

Definitions and Uses
~~~~~~~~~~~~~~~~~~~~

In Structural Walk, we used ``with_def_region_kind`` to count variable
uses. A map callback receives the same ``(var, kind)`` information but
returns a replacement. We can use it to rename a function parameter and a
loop variable, updating their uses throughout the function:

.. code-block:: python

   from tvm_tirx.script import tirx as T

   @T.function
   def example(n: T.int32):
       for i in range(n):
           T.evaluate(i + n)

We will rename ``n`` to ``n_new`` and ``i`` to ``i_new``. Each definition
and its uses must refer to the same new variable object. The callback
below records each old variable and its replacement in a Python dictionary,
``replacements``. At each use, it retrieves the same replacement. The
parameter ``n`` has kind ``DEF_PATTERN``, the loop variable ``i`` has kind
``DEF_SIMPLE``, and their uses have kind ``NONE``:

.. code-block:: python

   replacements = {}

   def rename(var, kind):
       if kind != tvm_ffi.DefRegionKind.NONE:
           replacements[var] = ir.Var(var.name + "_new", var.ty)
       return replacements.get(var, var)

   renamed = tvm_ffi.structural_map(
       example,
       with_def_region_kind=(ir.Var, rename),
       order="pre",
   )
   loop = renamed.body.seq[0]       # The function's first statement is the loop.
   evaluation = loop.body.seq[0]    # The loop's first statement is T.evaluate.
   addition = evaluation.value     # Its argument is i + n.
   print(renamed.params[0].name, loop.var.name)
   print(loop.end.same_as(renamed.params[0]))
   print(addition.a.same_as(loop.var))
   print(addition.b.same_as(renamed.params[0]))

The output is:

.. code-block:: text

   n_new i_new
   True
   True
   True

With ``order="pre"``, the mapper reaches the definitions of ``n`` and
``i`` before their uses in this function. The callback records each
replacement at its definition and returns that same object at each use.
The ``body.seq`` fields contain the function's and loop's statement
sequences, which let us inspect the rewritten loop and its addition.
The three ``True`` results confirm that the loop bound refers to
``n_new`` and that the addition refers to ``i_new`` and ``n_new``.
Creating a new variable at each use would break these connections, even
if the names matched. The mapper builds the rewritten function and leaves
``example`` unchanged.

Shared Objects
~~~~~~~~~~~~~~

A multiplication can use the same addition as both operands:

.. code-block:: python

   shared = x + 1
   twice = ir.prim.Mul(shared, shared)

``shared`` is one ``Add`` object representing ``x + 1``. Both ``twice.a``
and ``twice.b`` refer to it, so the multiplication represents
``(x + 1) * (x + 1)``.

The mapper reaches this object through each operand and calls its handler
twice. We can see what happens when each call creates a new addition,
changing the constant from 1 to 2:

.. code-block:: python

   mapped_additions = []

   def rewrite_add(node):
       mapped_additions.append(node)
       return ir.prim.Add(node.a, ir.const(2, node.ty))

   mapped_twice = tvm_ffi.structural_map(twice, callbacks=(ir.prim.Add, rewrite_add))
   print(mapped_twice)
   print("Add callbacks:", len(mapped_additions))
   print("Input operands are the same object:", twice.a.same_as(twice.b))
   print("Output operands are the same object:", mapped_twice.a.same_as(mapped_twice.b))

The output is:

.. code-block:: text

   (x + 2) * (x + 2)
   Add callbacks: 2
   Input operands are the same object: True
   Output operands are the same object: False

The input ``twice`` uses ``shared`` for both operands, so its identity
check is ``True``. In the output ``mapped_twice``, both operands represent
``x + 2``, but ``rewrite_add`` created a distinct ``Add`` for each operand.
The output identity check is therefore ``False``.

Each call to ``rewrite_add`` creates a new ``Add``, and the mapper does not
cache the callback's results. To keep both operands pointing to the same
new addition, we can use a dictionary as we did for variable replacements.
The callback records a replacement on its first visit and returns that
same object on later visits to the original addition:

.. code-block:: python

   add_replacements = {}

   def rewrite_add_once(node):
       if node not in add_replacements:
           add_replacements[node] = ir.prim.Add(node.a, ir.const(2, node.ty))
       return add_replacements[node]

   shared_result = tvm_ffi.structural_map(
       twice, callbacks=(ir.prim.Add, rewrite_add_once)
   )
   print(shared_result.a.same_as(shared_result.b))

The output is ``True``. Both callback calls receive the same original
``Add``, so the second retrieves the replacement created by the first.
The two operands of ``shared_result`` therefore share one new addition.

Copies and Storage Reuse
~~~~~~~~~~~~~~~~~~~~~~~~

The earlier rewrite returned ``mapped_twice`` while leaving ``twice``
unchanged. We can print ``twice`` to check that it still contains
``x + 1``, then inspect the left addition in ``mapped_twice`` to check
whether it still uses the original variable ``x``:

.. code-block:: python

   print(twice)

   left_add = mapped_twice.a   # The left addition: x + 2.
   left_operand = left_add.a  # Its left operand: x.
   print(left_operand.same_as(x))

The output is:

.. code-block:: text

   (x + 1) * (x + 1)
   True

The first line confirms that ``twice`` is unchanged. The ``True`` result
shows that the rewritten addition reuses the original ``x`` object.
In an ordinary call, default mapping preserves the input through
copy-on-write: it copies changed nodes and reuses unchanged children.

If the original IR is no longer needed, ``root._move()`` allows an
additional optimization. It transfers ownership to the mapper, allowing
default mapping to update nodes in place when they are not shared with
other references. Callbacks still return replacements and must not modify
their input objects in place.

.. _structural-visitor:

Structural Visit
----------------

With ``tvm_ffi.structural_walk``, the walker handles recursion, so callbacks
do not need to visit children themselves. With
``tvm_ffi.structural_visit``, a matching callback decides which children
to visit and in what order. If it returns without making recursive calls,
those children are not visited. The callback can also update analysis
state before and after a child visit. The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_visit(root, callbacks=())

Choosing Children
~~~~~~~~~~~~~~~~~

Suppose an analysis needs the variables in just the left operand of
``(x + 1) * (y + 2)``. A visitor callback can make that choice at the
multiplication, before visiting either operand. The following handler
continues into ``a`` and leaves ``b`` unvisited:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)
   left_variables = []

   def visit_left(node, visitor):
       return visitor.visit(node.a)

   def record_var(var, visitor):
       left_variables.append(var.name)

   tvm_ffi.structural_visit(
       root=expr,
       callbacks=[(ir.prim.Mul, visit_left), (ir.Var, record_var)],
   )
   print(left_variables)

The output is ``['x']``. The multiplication matches ``visit_left``, which
receives the node and a ``tvm_ffi.StructuralVisitor`` created for this
traversal. ``node.a`` is the left operand, ``x + 1``. Calling
``visitor.visit(node.a)`` processes it with the same callbacks.

The addition has no matching handler, so the visitor follows its children
automatically and passes ``x`` to ``record_var``. The multiplication's
right operand, ``node.b``, is never visited, so ``y`` is absent from the
result. A matched callback chooses all further visits from its node.
Returning ``None`` without a recursive call finishes that callback without
visiting its children.

``callbacks`` accepts a single ``(Type, handler)`` pair, a list or tuple
of pairs, grouped classes, or a function matching every reached value.
Classes match subclasses, and only the first matching handler runs.
With no matching handler, including when ``callbacks=()``, the visitor
uses the node's default child traversal.

.. _visiting-default-children:

Using the Default Traversal
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here, "default" means the child traversal defined by the current node's
IR type. For a multiplication, that rule visits ``a`` before ``b``.
``visitor.default_visit(node)`` follows this rule without calling the
current node's handler again. At each child, a matching handler still
decides whether to visit its own children. Calling ``visitor.visit(node)``
on the current multiplication would select its handler again.

Inside the multiplication's handler, the choices are:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Within the handler
     - Effect
   * - ``visitor.visit(node.a)``
     - Visit the left operand, ``x + 1``. The right operand is not visited
       automatically.
   * - ``visitor.default_visit(node)``
     - Visit ``x + 1`` followed by ``y + 2``. Each operand is processed using
       the same callback rules.
   * - ``return None`` without recursive calls
     - Return from the current handler without visiting either operand.

Structural Walk's ``order`` option selects whether a handler runs before
or after its children. A visitor callback can do work at both points in
the same call: first before ``default_visit``, then after the child visits
are complete. We can record both pre-order and post-order traces this way.
The handler below calls ``default_visit`` for every expression, so it
visits all nodes in ``expr``:

.. code-block:: python

   def describe(node):
       if isinstance(node, ir.Var):
           return node.name
       if isinstance(node, ir.IntImm):
           return str(int(node))
       return type(node).__name__

   before_children = []
   after_children = []

   def visit_expr(node, visitor):
       before_children.append(describe(node))
       interrupted = visitor.default_visit(node)
       if interrupted is not None:
           return interrupted
       after_children.append(describe(node))
       return None

   tvm_ffi.structural_visit(expr, callbacks=(ir.Expr, visit_expr))
   print(before_children)
   print(after_children)

The output is:

.. code-block:: text

   ['Mul', 'Add', 'x', '1', 'Add', 'y', '2']
   ['x', '1', 'Add', 'y', '2', 'Add', 'Mul']

Each callback records its node, visits the children, then records the node
again. The first list therefore follows pre-order and the second follows
post-order. The placement of recursive calls determines the order, so
``structural_visit`` has no ``order`` argument.

Both visitor methods return ``None`` when visiting finishes, or a
``VisitInterrupt`` when a child requests an early exit. The check after
``default_visit`` returns that interrupt immediately, before appending to
``after_children``.

Choosing Child Order
~~~~~~~~~~~~~~~~~~~~

Suppose a search should try the right branch first. Structural Walk's
pre-order and post-order options change when a handler runs relative to
its children, but keep the default left-to-right operand order. A visitor
callback can choose that order explicitly. We can visit ``b`` before ``a``
for both additions and multiplications, then collect variable names to
check the order:

.. code-block:: python

   reversed_variables = []

   def visit_right_first(node, visitor):
       for child in (node.b, node.a):
           interrupted = visitor.visit(child)
           if interrupted is not None:
               return interrupted
       return None

   def record_variable(var, visitor):
       reversed_variables.append(var.name)

   tvm_ffi.structural_visit(
       expr,
       callbacks=[
           ((ir.prim.Add, ir.prim.Mul), visit_right_first),
           (ir.Var, record_variable),
       ],
   )
   print(reversed_variables)

The output is ``['y', 'x']``. The multiplication visits ``y + 2`` before
``x + 1``. Each addition also visits its constant before its variable.
The loop continues to the next child only if the previous visit returned
``None``.

Tracking the Current Path
~~~~~~~~~~~~~~~~~~~~~~~~~

When reporting a variable occurrence, an analysis may need the path from
the root to that variable. A visitor callback can maintain this path by
adding its node's label before visiting children and removing it
afterward. Each child then sees the path for its own branch:

.. code-block:: python

   path = []
   variable_paths = []

   def visit_with_path(node, visitor):
       path.append(describe(node))
       try:
           if isinstance(node, ir.Var):
               variable_paths.append("/".join(path))
           return visitor.default_visit(node)
       finally:
           path.pop()

   tvm_ffi.structural_visit(expr, callbacks=(ir.Expr, visit_with_path))
   print(variable_paths)
   print(path)

The output is:

.. code-block:: text

   ['Mul/Add/x', 'Mul/Add/y']
   []

Before visiting children, the callback adds its node's label to ``path``.
Each recursive child call sees that label along with its earlier ancestors.
On return, ``finally`` removes the label. This prevents entries from the
left branch from remaining in the right branch's path. The push and pop
surround the same recursive call, keeping the path active for exactly
that subtree. Cleanup also runs if a child interrupts or raises an error.
The final empty list confirms that the path has been restored after the
whole visit.

.. _stopping-a-visit:

Handling Early Exit
~~~~~~~~~~~~~~~~~~~~

Structural Walk carries an interrupt from a callback back to its caller
as part of its automatic traversal. When a visitor callback makes
recursive calls itself, it must check each child's result and return any
interrupt before visiting another child. The checks in
``visit_right_first`` do this. We can test them by searching for ``y``
with that traversal and recording which variables are reached:

.. code-block:: python

   searched_variables = []

   def visit_until_y(var, visitor):
       searched_variables.append(var.name)
       if var.same_as(y):
           return tvm_ffi.VisitInterrupt(var)
       return None

   visitor_found = tvm_ffi.structural_visit(
       expr,
       callbacks=[
           ((ir.prim.Add, ir.prim.Mul), visit_right_first),
           (ir.Var, visit_until_y),
       ],
   )
   print(visitor_found.value.name)
   print("Variables visited:", searched_variables)

The output is:

.. code-block:: text

   y
   Variables visited: ['y']

The right branch reaches ``y`` first. ``visit_until_y`` returns an
interrupt containing that variable, and each ``visit_right_first`` call
returns it before trying its next child. The interrupt passes through
the addition and multiplication to ``structural_visit``, which returns it
as ``visitor_found``. The visit never reaches ``x`` in the left branch,
so only ``y`` appears in ``searched_variables``.

Ignoring a child's return value would let its parent callback continue
running after the child requests an early exit. When the whole visit
finishes normally, ``structural_visit`` returns ``None``.

.. _definitions-and-uses-2:

Preserving Definition Roles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A binding node's default traversal also marks which fields define
variables. When a visitor callback chooses those fields manually, it must
preserve these roles. Otherwise, a definition could be reported as a use.
We can check this with a ``Let`` that defines ``tmp`` and uses it twice.
First, leave the ``Let`` traversal to its default rule and record each
variable's role through ``visitor.def_region_kind()``:

.. code-block:: python

   tmp = ir.Var("tmp", "int32")
   scoped_expr = ir.prim.Let(tmp, expr, tmp + tmp)
   variable_contexts = []

   def record_context(var, visitor):
       variable_contexts.append((var.name, visitor.def_region_kind().name))
       return visitor.default_visit(var)

   tvm_ffi.structural_visit(scoped_expr, callbacks=(ir.Var, record_context))
   for name, kind in variable_contexts:
       print(name, kind)

The output is:

.. code-block:: text

   tmp DEF_SIMPLE
   x NONE
   y NONE
   tmp NONE
   tmp NONE

The default ``Let`` traversal visits ``Let.var`` under ``DEF_SIMPLE`` and
visits ``Let.value`` and ``Let.body`` outside that definition region.
The temporary therefore appears once as a definition and twice as a use.
The variable handler calls ``default_visit(var)`` to also follow any
children in the variable's type under the appropriate roles.

Now we can take over the ``Let`` traversal with our own handler. Because
the default ``Let`` rule will no longer run, calling ``visit(node.var)``
directly would report the definition as an ordinary use. The handler
enters ``DEF_SIMPLE`` explicitly for that visit, then visits ``value`` and
``body`` outside the definition region:

.. code-block:: python

   def visit_let(node, visitor):
       interrupted = visitor.with_def_region_kind(
           tvm_ffi.DefRegionKind.DEF_SIMPLE,
           lambda: visitor.visit(node.var),
       )
       if interrupted is not None:
           return interrupted
       interrupted = visitor.visit(node.value)
       if interrupted is not None:
           return interrupted
       return visitor.visit(node.body)

   variable_contexts.clear()
   tvm_ffi.structural_visit(
       scoped_expr,
       callbacks=[(ir.prim.Let, visit_let), (ir.Var, record_context)],
   )
   for name, kind in variable_contexts:
       print(name, kind)

This prints the same five lines, confirming that the custom traversal
preserves the default definition roles. ``with_def_region_kind(kind, fn)``
calls ``fn()`` under the requested definition kind and restores the
previous kind afterward. An enclosing ``DEF_PATTERN`` propagates through
nested regions, as described in
:ref:`the pattern-definition example <ir-pattern-definition-regions>`.

.. _structural-mutator:

Structural Mutate
-----------------

Structural Map handles child rewrites and rebuilds parents automatically.
With ``tvm_ffi.structural_mutate``, a matching callback takes responsibility
for both steps. It chooses which children to rewrite, makes the recursive
calls, and returns a result containing the child replacements it wants to
keep. It can also maintain state during those calls, as a visitor does.
The API has the following signature:

.. code-block:: text

   tvm_ffi.structural_mutate(root, callbacks=())

Rewriting Selected Children
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Suppose we want to replace variables only in the left operand of
``(x + 1) * (y + 2)``. The multiplication handler can select that operand
for rewriting and keep the right operand as it is. It then builds the
result using the rewritten left operand and the original right operand:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir

   x = ir.Var("x", "int32")
   y = ir.Var("y", "int32")
   expr = (x + 1) * (y + 2)

   def replace_var(var, mutator):
       if var.same_as(x):
           return ir.const(8, var.ty)
       if var.same_as(y):
           return ir.const(3, var.ty)
       return var

   def mutate_left(node, mutator):
       left = mutator.mutate(node.a)
       if left.same_as(node.a):
           return node
       return ir.prim.Mul(left, node.b, loc=node.loc)

   partial = tvm_ffi.structural_mutate(
       root=expr,
       callbacks=[(ir.prim.Mul, mutate_left), (ir.Var, replace_var)],
   )
   print(partial)
   print(partial.b.same_as(expr.b))
   print(expr)

The output is:

.. code-block:: text

   T.Add(8, 1) * (y + 2)
   True
   (x + 1) * (y + 2)

Each handler receives ``(value, mutator)``, where ``mutator`` is a
``tvm_ffi.StructuralMutator`` created for this rewrite. The multiplication
handler calls ``mutator.mutate(node.a)`` to process ``x + 1`` with the same
callbacks. The addition has no matching handler, so default mutation
processes its children and ``replace_var`` changes ``x`` to the constant 8.

The recursive call returns the rewritten left operand. ``mutate_left``
puts it in a new multiplication and keeps ``node.b`` as the right operand.
``loc=node.loc`` retains the original multiplication's source location.
Because this handler takes over the multiplication's rewrite, it must
return a node containing the child result. Its identity check lets it
return the original multiplication when the left operand is unchanged.

Since the handler never calls ``mutate`` on the right operand, ``y`` stays
a variable. The ``True`` identity check shows that the result shares this
unchanged subtree with ``expr``. Printing ``expr`` confirms that the input
still contains both variables.

Callback selection follows the same rules as in Structural Map: classes match
subclasses, and only the first matching handler runs. A list or tuple
combines rules, grouped classes can share a handler, and a function passed
on its own matches every reached value. Nodes with no matching handler
follow default mutation, including when ``callbacks=()``.

Returning Replacements
~~~~~~~~~~~~~~~~~~~~~~

In the ordinary calls used here, rewriting a child does not edit its
original parent. ``mutate_left`` must include the rewritten child in the
``Mul`` it returns. Returning the original ``node`` instead would discard
that child replacement. A handler can also return a new expression
without processing its children, for example when a rule replaces an
entire subtree.

Inside a multiplication handler, recursive calls and return values have
the following effects:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Within the handler
     - Effect
   * - ``mutator.mutate(node.a)``
     - Rewrite only the left operand. Use the returned child when building
       the parent result.
   * - ``mutator.default_mutate(node)``
     - For a multiplication, rewrite both operands using the same callbacks
       and return a multiplication containing their results.
   * - ``return node`` without recursive calls
     - Keep the node and its children without processing them.
   * - ``return replacement``
     - Use the replacement as the result at this position.

Calling ``mutate(node)`` on the current node would select its handler
again. ``default_mutate(node)`` bypasses that callback for the current
node and processes its children. For an expression field, the handler
must return a compatible IR expression, either the original or a
replacement. ``None``, ``WalkResult.SKIP``, and ``VisitInterrupt`` are not
control results for expression mutation.

Once a handler returns a replacement, the mutator accepts that result
without automatically processing it again. We can check this by replacing
``x`` with a new addition and recording any integer constants visited by
another handler:

.. code-block:: python

   new_literals = []

   def replace_symbol(var, mutator):
       return ir.prim.Add(ir.const(10, var.ty), ir.const(20, var.ty))

   def observe_literal(value, mutator):
       new_literals.append(int(value))
       return value

   replacement = tvm_ffi.structural_mutate(
       x,
       callbacks=[(ir.Var, replace_symbol), (ir.IntImm, observe_literal)],
   )
   print(replacement)
   print(new_literals)

The output is:

.. code-block:: text

   T.Add(10, 20)
   []

The variable handler returns a new addition containing 10 and 20. The
mutator accepts it without visiting its children, so neither constant
reaches ``observe_literal``. If we want the introduced constants to pass
through our rules too, the variable handler can explicitly process the
new addition's children before returning it:

.. code-block:: python

   new_literals.clear()

   def replace_and_process(var, mutator):
       introduced = replace_symbol(var, mutator)
       return mutator.default_mutate(introduced)

   processed = tvm_ffi.structural_mutate(
       x,
       callbacks=[(ir.Var, replace_and_process), (ir.IntImm, observe_literal)],
   )
   print(processed)
   print(new_literals)

The output is:

.. code-block:: text

   T.Add(10, 20)
   [10, 20]

The constants are now recorded, but the addition remains unchanged
because these callbacks do not compute its result. The callback chooses
whether its replacement's children are processed, separately from
constructing and returning that replacement.

Rewriting Children Before Parents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Our first rewrite produced ``T.Add(8, 1) * (y + 2)``. We can also compute
the left addition, while keeping the same rule that leaves the right
operand untouched. An addition handler first rewrites its children, then
checks whether their results are constants it can add together:

.. code-block:: python

   def mutate_and_fold_add(node, mutator):
       rewritten = mutator.default_mutate(node)
       if isinstance(rewritten.a, ir.IntImm) and isinstance(rewritten.b, ir.IntImm):
           value = int(rewritten.a) + int(rewritten.b)
           return ir.const(value, rewritten.ty)
       return rewritten

   folded_left = tvm_ffi.structural_mutate(
       expr,
       callbacks=[
           (ir.prim.Mul, mutate_left),
           (ir.prim.Add, mutate_and_fold_add),
           (ir.Var, replace_var),
       ],
   )
   print(folded_left)
   print(folded_left.b.same_as(expr.b))
   print(expr)

The output is:

.. code-block:: text

   9 * (y + 2)
   True
   (x + 1) * (y + 2)

The multiplication still calls only ``mutator.mutate(node.a)``. Inside
that branch, ``default_mutate`` replaces ``x`` with 8 before the addition
handler computes ``8 + 1``. The resulting constant 9 returns to
``mutate_left``, which puts it in the new multiplication. The right branch
is never processed and remains shared with the original expression.

Each handler determines how to use its children's results. The
multiplication selects one child, while the addition uses default
recursion and computes a result afterward. These recursive calls establish
the processing order, so ``structural_mutate`` has no ``order`` argument.

.. _renaming-bound-variables:

Tracking Variable Replacements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In Structural Map, we kept a Python dictionary to connect renamed
definitions with their uses. A mutation callback can use the variable
substitution table owned by its mutator. The same table is available
throughout its recursive calls, so a replacement recorded at a definition
can be retrieved when rewriting a use. We can apply this to a function
parameter and loop variable:

.. code-block:: python

   from tvm_tirx.script import tirx as T

   @T.function
   def example(n: T.int32):
       for i in range(n):
           T.evaluate(i + n)

   def rename_bound(var, mutator):
       replacement = mutator.var_remap_get(var)
       if replacement is not None:
           return replacement
       if mutator.def_region_kind() != tvm_ffi.DefRegionKind.NONE:
           replacement = ir.Var(var.name + "_new", var.ty)
           mutator.var_remap_set(var, replacement)
           return replacement
       return var

   renamed_function = tvm_ffi.structural_mutate(
       example, callbacks=(ir.Var, rename_bound)
   )
   loop = renamed_function.body.seq[0]  # The function's first statement is the loop.
   evaluation = loop.body.seq[0]        # The loop's first statement is T.evaluate.
   addition = evaluation.value         # Its argument is i + n.
   print(renamed_function.params[0].name, loop.var.name)
   print(loop.end.same_as(renamed_function.params[0]))
   print(addition.a.same_as(loop.var))
   print(addition.b.same_as(renamed_function.params[0]))
   print(example.params[0].name)

The output is:

.. code-block:: text

   n_new i_new
   True
   True
   True
   n

``var_remap_get(var)`` retrieves a recorded replacement or returns
``None``. At each definition, the callback creates a new variable and
records it with ``var_remap_set(var, replacement)``. The default function
and loop traversal identify their definitions through
``def_region_kind()``. Later uses retrieve the recorded variables, keeping
the loop bound and addition connected to the new definitions. The final
line confirms that the original function still has its parameter ``n``.

The callback must perform the lookup explicitly because it handles the
variable itself. Default variable mutation does not run after this handler
returns. This table records variable substitutions, not the results of
every arithmetic rewrite.

If a callback takes over a binding node and selects its fields manually,
it must preserve their definition roles. It can use
``mutator.with_def_region_kind(kind, fn)`` to run ``fn()`` under the
appropriate kind, just as the visitor does. The previous kind is restored
afterward, and an enclosing ``DEF_PATTERN`` continues to propagate through
nested regions.

Copies and Storage Reuse
~~~~~~~~~~~~~~~~~~~~~~~~

The selective rewrite also shows which parts of the result can be reused.
The multiplication handler constructs a new parent but keeps the original
right operand. Default mutation similarly copies changed nodes and reuses
unchanged children when preserving the input. Returning a rewritten value
does not require copying the whole tree.

If the input is no longer needed, ``root._move()`` transfers ownership and
allows default mutation to reuse storage where it is not shared. A custom
mutation handler can also choose an implementation based on ownership by
accepting a third argument:
``handler(value, mutator, inplace_mode)``. The mode is an integer that
can be compared with ``tvm_ffi.InplaceMode.DISALLOW`` or
``tvm_ffi.InplaceMode.ALLOW``. ``DISALLOW`` requires preserving the input,
and ``ALLOW`` permits reusing storage for the current value. This
permission is determined for each callback invocation, so passing
``_move()`` does not make every node available for reuse. The examples here
use ordinary references and return replacement values from two-argument
handlers.

The TVM FFI
`structural operations guide <https://github.com/apache/tvm-ffi/blob/e9fdc58fa4bf2e60ed701dacac48a01e41a3dc1f/docs/concepts/structural_eq_hash.rst#structural-walk-and-map>`_
covers ownership options, custom object hooks, and native implementations.
For analysis that also tracks variable ranges and constraints, continue to
:doc:`symbolic_integer_analysis`. :doc:`writing_passes` explains how to
package a rewrite as a compiler pass.
