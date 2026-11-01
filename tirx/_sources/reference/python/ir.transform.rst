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

tvm_tirx.ir.transform
=====================

.. automodule:: tvm_tirx.ir.transform
   :no-members:

.. currentmodule:: tvm_tirx.ir.transform

A transformation in TIRx is any function that takes a module and returns a
new one. We can write it as an ordinary Python function and call it
directly. Passes are an optional, thin layer on top: a pass wraps a
transformation with a name and an optimization level, and runs it under a
shared :py:class:`PassContext`. This lets us compose passes into a pipeline
with :py:class:`Sequential`, select which passes run through the context's
options, and observe each step with pass instruments. A pass only describes
how a transformation is invoked; inside, different passes can work in
completely different ways.

The example below wraps a function as a pass with
:py:func:`~tvm_tirx.tirx.transform.function_pass`, which applies it to every
function in the module. ``specialize`` assumes that the tensors have 64
elements. A built-in simplification pass then removes the ``if i < 100``
check, which the specialized loop always satisfies. A
:py:class:`~tvm_tirx.ir.transform.instrument.PassPrintingInstrument` prints
the module before the first pass and after the last one:

.. code-block:: python

   import tvm_ffi
   from tvm_tirx import ir, tirx
   from tvm_tirx.ir.transform import PassContext, Sequential
   from tvm_tirx.ir.transform.instrument import PassPrintingInstrument
   from tvm_tirx.script import ir as I
   from tvm_tirx.script import tirx as T

   n = T.dynamic("n")


   @I.ir_module
   class Module:
       @T.function
       def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
           for i in range(n):
               if i < 100:
                   B[i] = A[i] + 1


   def replace_n(var):
       return ir.const(64, var.ty) if var.same_as(n) else var


   @tirx.transform.function_pass(opt_level=0)
   def specialize(func, mod, ctx):
       return tvm_ffi.structural_map(func, (ir.Var, replace_n))


   pipeline = Sequential([specialize, tirx.transform.StmtSimplify()])
   printer = PassPrintingInstrument(["specialize"], ["tirx.StmtSimplify"])
   with PassContext(opt_level=2, instruments=[printer]):
       result = pipeline(Module)

The output is:

.. code-block:: text

   Print IR before: specialize
   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   n = T.dynamic("n", dtype="int32")
   @T.ir_module
   class Module:
       @T.function
       def add_one(A: T.Tensor((n,), T.float32), B: T.Tensor((n,), T.float32)):
           for i in range(n):
               if i < 100:
                   B[i] = A[i] + T.float32(1.0)


   Print IR after: tirx.StmtSimplify
   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.ir_module
   class Module:
       @T.function
       def add_one(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
           for i in range(64):
               B[i] = A[i] + T.float32(1.0)

``Module`` itself is unchanged, and ``result`` holds the transformed module.
Inside a :py:class:`Sequential`, a pass runs only when its optimization level
is at most the context's ``opt_level`` or its name is in ``required_pass``,
and never when its name is in ``disabled_pass``. Calling a pass directly, as
in ``specialize(Module)``, always runs it.

Classes
-------

.. autosummary::
   :toctree: generated/

   ModulePass
   Pass
   PassContext
   PassInfo
   PassInstrument
   Sequential

Functions
---------

.. autosummary::
   :toctree: generated/

   PrintIR
   module_pass
   pass_instrument

Namespaces
----------

.. toctree::
   :maxdepth: 1

   ir.transform.instrument
