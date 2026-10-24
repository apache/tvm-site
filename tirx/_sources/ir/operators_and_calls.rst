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

Operators describe computations, while calls apply them to particular inputs.
Custom operators can be registered alongside built-in ones, with their own
argument requirements, type rules, and effects. Their calls use the existing
IR representation, so extensions can build analyses, transformations, and
lowering passes around the same objects and interfaces.

Operators
---------

``ir.Op`` is an expression that identifies a registered operator. The registry
stores shared operator definitions, which can be retrieved by name with
``ir.Op.get(name)``. An operator's ``args_info`` describes its required value
arguments. A validator is a registered function that checks whether a call's
arguments meet the operator's requirements.

Operator attributes describe properties shared by all calls to that operator.
The ``TCallEffectKind`` attribute records whether a call only computes a value
or can read or change program state.

``prim.exp`` computes the exponential of one input. It declares an argument
named ``x`` and has the effect kind ``Pure``: it computes a value without
reading or changing external state. Inspect these properties on the registered
object:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin operator lookup
   :end-before: # end operator lookup
   :prepend: from tvm_tirx import ir

Here, ``argument_names`` is ``["x"]``, and ``effect`` equals
``ir.CallEffectKind.Pure``. See :class:`~tvm_tirx.ir.Op` for the registration
and attribute APIs.

Calls
-----

``ir.Call`` represents an invocation of an operator or function. The expression
being called is its *callee*. A call inherits ``ty`` and ``loc`` from
``ir.Expr`` and adds these fields:

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
     - Explicit type arguments, empty by default.
   * - ``attrs``
     - Attributes for this invocation, or None. A dictionary becomes
       ``ir.DictAttrs``.

``ty_args`` supplies types as inputs to the operation; ``ty`` describes its
result. Call attributes belong to one invocation, while the operator's
registration supplies the rules used to interpret them.

For example, this call applies the registered exponential operator to a float32
constant:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin call contract
   :end-before: # end call contract

This constructs the IR for ``exp(1.0)``; its result type is
``ir.PrimType("float32")``. The computation is performed when a program
containing the call is compiled and executed.

A function-valued variable can also be the callee. Here, ``f`` has a signature
that takes one float32 value and returns a float32 value:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin function call type
   :end-before: # end function call type

``function_call`` represents ``f(1.0)``. The variable supplies a signature;
the example does not provide an implementation of ``f``.

.. _types-and-validation:

Type Inference and Validation
-----------------------------

To avoid repeating type information at each call, ``ir.Call`` can infer its
result type from the operator's registered rule or the callee's function
signature. Omitting ``ty`` asks ``ir.Call`` to infer it; passing an explicit
type stores that type instead. ``MissingType`` marks a result whose type has
not yet been determined.

Validation checks whether the arguments meet the operator's requirements.
It is separate from construction, so tools can assemble a call before
checking its arguments with ``call.validate()``. Type inference can also
reject inputs while computing the result type.

For ``prim.exp``, an explicit result type lets the constructor record a call
with no arguments. Validation then rejects the argument count:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin validation failure
   :end-before: # end validation failure
   :append: invalid.validate()  # raises TypeError: expected one argument

When constructing or rewriting IR, a call can temporarily carry a missing
result type. ``ir.reinfer_type`` derives the type from its current inputs:

.. literalinclude:: examples/operators_and_calls.py
   :language: python
   :start-after: # begin provisional call
   :end-before: # end provisional call

Here, ``inferred`` is ``ir.PrimType("float32")``. The function returns a type;
it leaves ``provisional.ty`` unchanged. See :class:`~tvm_tirx.ir.Call` and
:func:`~tvm_tirx.ir.reinfer_type` for the inference rules, and
:meth:`~tvm_tirx.ir.Call.validate` for the validation contract.

Register New Operators
----------------------

Registering an operator gives a custom computation an argument and result-type
contract that calls and passes can share. New operators can be added from a
Rust extension without modifying the C++ core.

Use a ``register_ops!`` block with one ``OpDef`` chain per operator. Each chain
supplies the operator's name, argument requirements, type-inference rule, and
attributes.
The operator joins the same registry as the built-in operators, so Python,
C++, and Rust code can retrieve it.

For example, this block registers the interface of ``my_ext.clamp``, an
operation for clamping an integer index to ``[0, bound)``. A call has the type
of its index, and ``validate()`` rejects arguments that are not integer
expressions:

.. code-block:: rust

   use tvm_tirx::ir::op_attr::CALL_EFFECT_KIND;
   use tvm_tirx::ir::sig::{self, IntExpr};
   use tvm_tirx::ir::{CallEffectKind, CallObj, OpDef, Type};
   use tvm_tirx::tvm_ffi::Result;

   fn index_type(call: &CallObj) -> Result<Type> {
       Ok(call.args.get(0)?.ty.clone())
   }

   tvm_tirx::register_ops! {
       OpDef::new("my_ext.clamp", "Clamp an index to the range `[0, bound)`.")
           .signature((
               sig::arg::<IntExpr>("index", "The index."),
               sig::arg::<IntExpr>("bound", "The exclusive upper bound."),
           ))
           .set_infer_type(index_type)
           .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
   }

Here, ``.signature`` generates a validator from the ``IntExpr`` argument
constraints. ``.set_infer_type`` registers ``index_type``, which returns the
first argument's type. ``.set_attr`` records that the operation is pure.

Once the extension is loaded and the operator is registered, Python can
retrieve it with ``ir.Op.get("my_ext.clamp")`` and construct calls in the same
way as for a built-in operator.

To make these calls executable, supply a lowering that replaces them with
existing IR or implement the operation in a backend.

See :doc:`/reference/python/ir` for the complete operator and call APIs. The
:download:`operators_and_calls.py <examples/operators_and_calls.py>` example includes runnable
calls, re-inference, and validation failures.
