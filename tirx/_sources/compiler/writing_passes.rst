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

Writing and Debugging Passes
============================

Suppose we want a version of an add-one function for tensors of length 128.
We will write a pass that fixes its symbolic length, then combine it with
statement simplification. A compiler pass packages an IR transformation so
the pass manager can run it as part of a pipeline. As we build this pipeline,
we will check its result and inspect the intermediate programs.

Starting Program
----------------

The following function adds one to each input element. ``T.dynamic("n")``
creates a symbolic length, so the tensor shapes and loop bound use the same
variable ``n``:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir, tirx
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")

   @T.function
   def guarded_add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
       for i in range(n):
           if i < n:
               B[i] = A[i] + 1

The bounds check is deliberately redundant: ``range(n)`` already guarantees
``i < n``. This gives us two changes to make. Our pass will replace ``n``
with 128 in the shapes and body. The existing ``StmtSimplify`` pass will
remove the unnecessary check. The resulting function will accept tensors
of length 128 rather than a length determined at runtime.

Passes run on an ``ir.IRModule``, which holds named functions. We place our
function in a module before transforming it:

.. code-block:: python

   before = ir.IRModule({"guarded_add_one": guarded_add_one})

Writing a Function Pass
-----------------------

The replacement rule matches the original variable ``n`` and returns an
integer constant in its place. We can use ``structural_map`` from
:doc:`structural_utilities` to apply this rule throughout a function.
The :py:func:`~tvm_tirx.tirx.transform.function_pass` decorator turns the
rewrite into a pass:

.. code-block:: python

   def specialize_dimension(dimension, extent):
       replacement = ir.const(extent, dimension.ty)

       @tirx.transform.function_pass(opt_level=0, name="SpecializeDimension")
       def transform(func, mod, ctx):
           def replace_dimension(var):
               if var.same_as(dimension):
                   return replacement
               return var

           return tvm_ffi.structural_map(
               func, (ir.Var, replace_dimension), order="pre"
           )

       return transform

The inner callback ``transform`` receives a function, its module, and the
active pass context. This rewrite uses only ``func`` and returns the
rewritten function. The pass manager calls it for each TIRx function in
the module. ``same_as`` selects the original dimension object, so unrelated
variables with the printed name ``n`` stay unchanged.

``specialize_dimension`` creates a pass configured for a particular
dimension and extent. Calling that pass with a module runs the rewrite:

.. code-block:: python

   specialize = specialize_dimension(n, 128)
   specialized = specialize(before)
   print(int(specialized["guarded_add_one"].params[0].ty.shape[0]))

The output is ``128``. ``params[0]`` is parameter ``A``, and
``ty.shape[0]`` is its first dimension. The rewrite also changes ``B``'s
shape, the loop bound, and the condition to use 128. It leaves the original
module ``before`` unchanged.

Running a Pipeline
------------------

The specialized function still contains ``if i < 128``. We can follow our
pass with ``StmtSimplify`` to remove that check. ``Sequential`` runs the
passes in the order listed:

.. code-block:: python

   simplify = tirx.transform.StmtSimplify()
   pipeline = ir.transform.Sequential([specialize, simplify])

   with ir.transform.PassContext(opt_level=2):
       after = pipeline(before)

   print(after["guarded_add_one"].script())

The function now has fixed shapes and no bounds check:

.. code-block:: text

   @T.function
   def guarded_add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
       for i in range(128):
           B[i] = A[i] + T.float32(1.0)

``PassContext`` sets execution options for the pipeline. Its optimization
level of 2 allows our level-0 pass and the built-in simplifier to run.
``Sequential`` specifies the transformations and their order.

Checking the Result
--------------------

First, check that the transformed module is well formed. For example,
each variable use must have a valid definition, and a tensor load's result
type must agree with its source and indices:

.. code-block:: python

   assert tirx.analysis.check_well_formed(after)

We also want to check that it has the structure we intended. We can write
the expected module directly in TVMScript, then compare it with the result:

.. code-block:: python

   @T.ir_module
   class Expected:
       @T.function
       def guarded_add_one(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
           for i in range(128):
               B[i] = A[i] + 1

   ir.assert_structural_equal(after, Expected)

This comparison checks both fixed tensor shapes and the loop with a direct
store in its body. Neither well-formedness nor structural comparison proves
that an arbitrary rewrite preserves computation. Here, specialization
restricts the accepted length to 128, and removing the guard is valid because
the loop already ensures ``i < 128``.

Inspecting a Pass's Input and Output
------------------------------------

If the structural comparison fails, we need to find which stage produced
the unexpected IR. A pass instrument can print the module immediately
before and after a selected pass. To inspect statement simplification,
use its ``info.name`` in both name lists:

.. code-block:: python

   from tvm_tirx.ir.transform.instrument import PassPrintingInstrument

   printing = PassPrintingInstrument(
       print_before_pass_names=[simplify.info.name],
       print_after_pass_names=[simplify.info.name],
   )
   with ir.transform.PassContext(opt_level=2, instruments=[printing]):
       inspected = pipeline(before)

The output has two labeled programs, ``Print IR before: tirx.StmtSimplify``
and ``Print IR after: tirx.StmtSimplify``. Both have length 128. The first
still contains ``if i < 128``, and the second has the direct store. This
shows the boundary between the two transformations: specialization has
already fixed the length when the simplifier receives the module.

The instrument observes the pipeline without changing it. To inspect our
own pass instead, use ``specialize.info.name`` in the lists.

The :doc:`pass API </reference/python/ir.transform>` and
:doc:`TIRx transformation API </reference/python/tirx.transform>` cover
other pass forms and built-in transformations. The
:doc:`instrument API </reference/python/ir.transform.instrument>` also
provides tools for dumping every stage and measuring pass timings.
