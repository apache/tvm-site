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

Defining Types, Operators, and Passes
=====================================

In this chapter, we walk through the Rust library of the package set up in
:doc:`extending_ir_in_rust`: a complex type, four operators, and a pass that
lowers them to scalar arithmetic. The excerpts come from
:download:`src/lib.rs <../../examples/extension/src/lib.rs>`, which also contains
the validation and traversal helpers they use.

Defining a Type
---------------

``ComplexType`` has one field, ``elem``, the scalar type shared by the real and
imaginary parts:

.. literalinclude:: ../../examples/extension/src/lib.rs
   :language: rust
   :start-after: // begin type
   :end-before: // end type

As in C++, the type is defined in two parts: ``ComplexTypeObj`` is the object
that holds the fields, and ``ComplexType`` is a reference-counted reference to
it, which IR nodes store and functions pass around. ``base: TypeObj`` makes the
object an IR type, and ``rs_class`` registers its fields for reflection, through
which Python and IR JSON loading construct the type as they construct a C++ one.

``def_object_ref!(pub ComplexType, Type, ComplexTypeObj)`` defines the
reference. Its arguments are:

- ``pub ComplexType``: the name and visibility of the reference.
- ``Type``: the reference to the parent object, ``TypeObj``. A ``ComplexType``
  converts into a ``Type`` with ``into()`` wherever a ``Type`` is expected, such
  as the result of type inference.
- ``ComplexTypeObj``: the object it refers to. A ``ComplexTypeObj`` converts
  into a ``ComplexType`` with ``into()``, and a ``ComplexType`` dereferences to
  it, so ``ty.elem`` reads the field.

``infer_complex`` in the next section uses both conversions: it builds a
``ComplexTypeObj``, turns it into a ``ComplexType``, and returns it as a
``Type``.

The ``static_init_block!`` runs ``register_type`` when the library loads, as a
``TVM_FFI_STATIC_INIT_BLOCK`` does for C++, and Python then sees the type as
``my_ext.Complex``.

A ``ComplexType`` records the scalar type of the two parts, and a
``complex(re, im)`` call supplies their values as its operands. ``elem`` can be
any scalar float type, such as ``float32``, ``float64`` or ``bfloat16``; this
tutorial uses ``float32``.

Defining Operators
------------------

To support operations on the new type, we need to define new operators. In our
case, complex arithmetic needs four: one builds a complex value from two
scalars, one multiplies complex values, and two read the parts back as scalars
so that results can be stored in ordinary ``float32`` buffers. Like the type,
they are registered under the ``my_ext`` namespace:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Operator
     - Meaning
     - Result type (``float32`` parts)
   * - ``complex(re, im)``
     - Construct a complex value
     - ``my_ext.Complex(T.float32)``
   * - ``cmul(z, w)``
     - Multiply two complex values
     - ``my_ext.Complex(T.float32)``
   * - ``real(z)``
     - Extract the real part
     - ``T.float32``
   * - ``imag(z)``
     - Extract the imaginary part
     - ``T.float32``

The constructor is registered as an ``OpDef`` chain in a
``static_init_block!``. Each step updates the operator registry, and the chain
completes when its statement ends:

.. literalinclude:: ../../examples/extension/src/lib.rs
   :language: rust
   :start-after: // begin constructor
   :end-before: // end constructor

``signature`` declares the arguments of the operator and of its Python and
TVMScript builder. ``set_infer_type`` attaches a named Rust function that
checks the arguments and computes the result type:

.. literalinclude:: ../../examples/extension/src/lib.rs
   :language: rust
   :start-after: // begin inference
   :end-before: // end inference

``set_validator`` repeats these checks for a call whose result type is given
explicitly rather than inferred. ``Pure`` marks the operator as computing a
value without writing memory.

``infer_complex`` requires both parts to be scalar floats of the same type, and
converts neither. The other three operators follow the same pattern. ``cmul``
requires two complex arguments of the same type and returns it; ``real`` and
``imag`` require one and return its ``elem``. Passing an ordinary scalar to ``cmul`` therefore fails
with ``TypeError: my_ext.cmul expects a complex value``.

Writing a Pass
--------------

Although the new type and operators can be used like built-in ones when writing
programs, the existing passes know nothing about them: they can neither
optimize them further nor lower them to what the backend supports. We therefore
add a pass of our own to handle them. In our case, ``LowerComplex`` applies
three rewrite rules:

.. code-block:: text

   cmul(complex(a, b), complex(c, d))  ->  complex(a*c - b*d, a*d + b*c)
   real(complex(a, b))                 ->  a
   imag(complex(a, b))                 ->  b

A complex local variable ``z = complex(a, b)`` is split into two scalar
variables, ``z_re = a`` and ``z_im = b``, and later uses of ``z`` resolve to
that pair. This lets the rules apply to the step-by-step function from
:doc:`extending_ir_in_rust`, whose loop body becomes:

.. code-block:: python

   z_re: T.let[T.float32] = A[i]
   z_im: T.let[T.float32] = B[i]
   w_re: T.let[T.float32] = C[i]
   w_im: T.let[T.float32] = D[i]
   product_re: T.let[T.float32] = z_re * w_re - z_im * w_im
   product_im: T.let[T.float32] = z_re * w_im + z_im * w_re
   R[i] = product_re
   Im[i] = product_im

Only scalar arithmetic remains. The pass refers to the four operators through
handles:

.. literalinclude:: ../../examples/extension/src/lib.rs
   :language: rust
   :start-after: // begin handles
   :end-before: // end handles

A handle names a registered operator and looks it up on first use.
``CMUL.is(&call.op)`` tests whether a call's ``op`` is ``my_ext.cmul``, and
``COMPLEX.get()?`` returns ``my_ext.complex`` to build a new call. ``Lowering``
implements these rewrites with them:

.. literalinclude:: ../../examples/extension/src/lib.rs
   :language: rust
   :start-after: // begin lowering
   :end-before: // end lowering
   :prepend: #[derive(Default)]
      struct Lowering {
          bindings: Vec<(Var, Var, Var)>,
      }
      impl Lowering {
          fn parts(&self, expr: &Expr) -> Result<(Expr, Expr)> { ... }
      }
      #[dispatch(map)]
      impl Lowering {
          fn map_sequence(&mut self, seq: SeqStmt) -> SeqStmt { ... }
   :append: }

``bindings`` records each complex variable with its two scalar replacements,
and ``parts`` reads the real and imaginary parts of an operand, either a
``complex(re, im)`` call or a variable already split. ``#[dispatch(map)]``
sends each node to the ``map_`` method of its type, and leaves other nodes
unchanged: ``map_call`` applies the three rules, ``map_bind`` splits a complex
binding, and ``map_sequence`` flattens the sequence that a split binding leaves
inside another.

The pass is built from an ordinary named function:

.. literalinclude:: ../../examples/extension/src/lib.rs
   :language: rust
   :start-after: // begin function callback
   :end-before: // end function callback

It receives a function, its module, and the pass context, and returns the
rewritten function; ``Some`` replaces the original. The first ``structural_map``
walks the function with ``Lowering``. ``PostOrder`` visits children before their
parent, so the operands of ``cmul`` are lowered before ``cmul`` itself. The
second ``structural_map`` only checks the result. It takes a tuple of three
functions, and calls each on the nodes of its parameter type:

.. code-block:: rust

   /// Reject a complex type, wherever it appears: on a variable, a parameter or an expression.
   fn check_type(ty: Type) -> Result<Type> { ... }

   /// Reject an expression whose type is complex.
   fn check_expr(expr: Expr) -> Result<Expr> { ... }

   /// Reject a call to one of the four complex operators that the rewrite did not consume.
   fn check_call_lowered(call: Call) -> Result<Call> { ... }

Each returns its node unchanged and raises a ``ValueError`` on an unsupported
use, such as a complex function parameter, so the pass reports such a use
before code generation.

Finally, ``FunctionPass::new`` wraps the function as a pass, and the factory is
exported so that Python can call it as ``lib.lower_complex()``:

.. literalinclude:: ../../examples/extension/src/lib.rs
   :language: rust
   :start-after: // begin pass
   :end-before: // end pass

Next, :doc:`using_ir_extensions` loads the library and compiles the function.
