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

In this chapter, we define the extension set up in :doc:`extending_ir_in_rust`:
a complex type, four operators, and a pass that lowers them to scalar
arithmetic. Each step is shown in Rust, from the library
`src/lib.rs <https://github.com/apache/tvm-tirx/blob/main/examples/extension/src/lib.rs>`_,
and in Python, from the module
`my_ext_py/my_ext.py <https://github.com/apache/tvm-tirx/blob/main/examples/extension/my_ext_py/my_ext.py>`_.
Both files also contain the validation and traversal helpers that the excerpts
use.

The two versions define the same names, so the example program runs with
either. The Python version is a single module that needs no build, which makes
it a quick way to try out an extension before writing it in Rust. To run the
example with it, put its directory on ``PYTHONPATH``:

.. code-block:: bash

   PYTHONPATH=examples/extension/my_ext_py python examples/extension/run_example.py

.. note::

   **Why is Rust preferred for extensions?**

   Python is the quickest way to try out an extension, but it is slow on large
   programs.

   Rust is fast and memory safe, which makes it a good fit for agents exploring
   new types, operators, and passes.

Defining a Type
---------------

To work with complex numbers directly in the IR, we first introduce a type for
them. ``ComplexType`` has one field, ``dtype``, the scalar type shared by the
real and imaginary parts, as in ``TensorType``:

.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../examples/extension/src/lib.rs
         :language: rust
         :start-after: // begin type
         :end-before: // end type

   .. group-tab:: Python

      .. literalinclude:: ../../examples/extension/my_ext_py/my_ext.py
         :language: python
         :start-after: # begin type
         :end-before: # end type

.. note::

   **What does def_object_ref! do?**

   ``def_object_ref!(ObjRef, Parent, Obj)`` defines ``ObjRef``, the reference
   type of the object type ``Obj``, as the C++ macro
   ``TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE`` does. An ``ObjRef`` holds
   a reference-counted pointer to an ``Obj``, and the macro gives it these
   conversions:

   - an ``Obj`` converts into an ``ObjRef`` with ``into()``;
   - an ``ObjRef`` dereferences to its ``Obj``, so its fields read directly;
   - an ``ObjRef`` converts into ``Parent``, the reference to the parent
     object, with ``into()``.

   A visibility such as ``pub`` before ``ObjRef`` applies to the reference.
   Here, ``def_object_ref!(pub ComplexType, Type, ComplexTypeObj)`` defines the
   public reference ``ComplexType``: ``ty.dtype`` reads the field of a
   ``ComplexType``, and a ``ComplexType`` converts into a ``Type`` wherever one
   is expected, such as the result of type inference.

In Rust, as in C++, the type is defined in two parts: ``ComplexTypeObj`` is the
object that holds the fields, and ``ComplexType`` is a reference-counted
reference to it, which IR nodes store and functions pass around.
``base: TypeObj`` makes the object an IR type, and ``rs_object`` registers its
fields for reflection, through which Python and IR JSON loading construct the
type as they construct a C++ one. ``def_object_ref!`` defines the reference.

The ``static_init_block!`` registers the type when the library loads.

.. note::

   **What does static_init_block! do?**

   ``static_init_block! { ... }`` runs its body when the program or library that
   holds it loads, as a C++ ``TVM_FFI_STATIC_INIT_BLOCK`` does. The body
   registers what the library defines, such as types, operators, and global
   functions, and Python sees these registrations once it loads the library.
   The body is the body of a function that returns ``Result<()>``, so ``?``
   stops it at its first error.

   Here, ``register_type::<ComplexType>()`` registers ``ComplexType`` as an IR
   type, so that Python constructs it from its fields and IR JSON holds it, as
   for a C++ type.

In Python, one class is enough. ``py_class`` from ``tvm_ffi.dataclasses``
registers it as an IR object: ``my_ext.ComplexType`` is its type key,
``ir.Type`` its parent, and the annotated ``dtype`` its field. The class gets a
constructor, ``ComplexType(dtype)``, and its field takes part in structural
equality, printing, and IR JSON, as the fields of a C++ type do. The global
function ``ir.expose_type`` then publishes the type, as ``register_type`` does
in Rust.

Either way, Python sees the type as ``my_ext.Complex``. A ``ComplexType``
records the scalar type of the two parts, and a ``complex(re, im)`` call
supplies their values as its operands. ``dtype`` can be any scalar float type,
such as ``float32``, ``float64`` or ``bfloat16``; this tutorial uses
``float32``.

Defining Operators
------------------

To support operations on the new type, we need to define new operators. In our
case, complex arithmetic needs four: one builds a complex value from two
scalars, one multiplies complex values, and two read the parts back as scalars
so that results can be stored in ordinary ``float32`` tensors. Like the type,
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

Each operator has a signature, a type inference function, and the effect kind
``Pure``, which marks it as a pure function that does not interact with any
external state. The ``complex`` operator is registered as follows:

.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../examples/extension/src/lib.rs
         :language: rust
         :start-after: // begin constructor
         :end-before: // end constructor

   .. group-tab:: Python

      .. literalinclude:: ../../examples/extension/my_ext_py/my_ext.py
         :language: python
         :start-after: # begin constructor
         :end-before: # end constructor

In Rust, the operator is an ``OpDef`` chain in a ``static_init_block!``: each
step updates the operator registry, and the chain completes when its statement
ends. In Python, ``ir.Op.register`` adds the operator to the registry and
returns it, and the helper ``_define`` sets its signature and attributes;
``expose`` publishes its builder, ``my_ext.complex``, once the definition is
complete. The module keeps the returned operator as the global ``COMPLEX``.

.. note::

   A ``static_init_block!`` can hold any registrations, types and operators
   alike. The example keeps the type and the operators in separate blocks, but
   one block can register both:

   .. code-block:: rust

      tvm_ffi::static_init_block! {
          register_type::<ComplexType>()?;
          OpDef::new("my_ext.complex", "Construct the complex number re + im * i.")
              .signature(...)
              .set_infer_type(infer_complex);
      }

In both, the signature declares the arguments of the operator and of its Python
and TVMScript builder, and the type inference function checks the arguments
and computes the result type:

.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../examples/extension/src/lib.rs
         :language: rust
         :start-after: // begin inference
         :end-before: // end inference

   .. group-tab:: Python

      .. literalinclude:: ../../examples/extension/my_ext_py/my_ext.py
         :language: python
         :pyobject: _infer_complex

The function computes the type of a ``complex(re, im)`` call: a
``ComplexType`` whose ``dtype`` is the type of the two parts. It requires both
parts to be scalar floats of the same type, and converts neither. The Rust
version uses both conversions of the reference: it builds a
``ComplexTypeObj``, turns it into a ``ComplexType``, and returns it as a
``Type``; the Python version calls the class. The other three operators follow
the same pattern. ``cmul`` requires two complex arguments of the same type and
returns it; ``real`` and ``imag`` require one and return its ``dtype``. Passing
an ordinary scalar to ``cmul`` therefore fails with
``TypeError: my_ext.cmul expects a complex value``.

An operator's validator checks a call whose result type is given explicitly
rather than inferred. In Rust, ``set_validator`` attaches one that repeats the
checks of type inference. A Python operator has the validator that its
signature generates, which checks the number of arguments; validators that also
check types are native functions. The Python version therefore checks the types
of such calls in its pass, before lowering them.

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

Only scalar arithmetic remains. To apply the rules, the pass tests whether a
call's ``op`` is one of the four operators. The Rust pass looks them up by name,
as C++ does with ``Op::Get``. The operators are registered when the library
loads, and ``Op::get`` returns the registry's single object for a name.
``call.op.same_as(&Op::get("my_ext.cmul")?)`` therefore tests whether a call's
``op`` is ``my_ext.cmul`` by comparing pointers. The Python pass compares with
the operators that ``Op.register`` returned, as in ``call.op.same_as(CMUL)``.
The rewrites are:

.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../examples/extension/src/lib.rs
         :language: rust
         :start-after: // begin lowering
         :end-before: // end lowering
         :prepend: struct Lowering {
                bindings: Vec<(Var, Var, Var)>,
            }
            impl Lowering {
                fn parts(&self, expr: &Expr) -> Result<(Expr, Expr)> { ... }
            }
            #[dispatch(map, (AnyView<'_>,), Result<Any>)]
            impl Lowering {
                fn map_sequence(&mut self, seq: SeqStmt) -> Result<Any> { ... }
         :append: }

   .. group-tab:: Python

      .. literalinclude:: ../../examples/extension/my_ext_py/my_ext.py
         :language: python
         :pyobject: _Lowering

``bindings`` records each complex variable with its two scalar replacements,
and ``parts`` reads the real and imaginary parts of an operand, either a
``complex(re, im)`` call or a variable already split. ``map_call`` first
validates a call to one of the operators, then applies the three rules;
``map_bind`` splits a complex binding, and ``map_sequence`` flattens the
sequence that a split binding leaves inside another. In Rust,
``#[dispatch(map, (AnyView<'_>,), Result<Any>)]`` sends each node to the
``map_`` method of its type. In Python, the next excerpt pairs each method with
the type of node it takes.

.. seealso::

   :ref:`rust-dispatch` and :ref:`rust-structural-utilities` in
   :doc:`../ir_guides/working_with_ir_in_rust`.

The pass is built from an ordinary named function:

.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../examples/extension/src/lib.rs
         :language: rust
         :start-after: // begin function callback
         :end-before: // end function callback

   .. group-tab:: Python

      .. literalinclude:: ../../examples/extension/my_ext_py/my_ext.py
         :language: python
         :pyobject: _lower_function

It receives a function, its module, and the pass context, and returns the
rewritten function; in Rust, ``Some`` replaces the original. The first
``structural_map`` walks the function with the rewrites. It visits children
before their parent, ``WalkOrder::PostOrder`` in Rust and the default order in
Python, so the operands of ``cmul`` are lowered before ``cmul`` itself. In
Python, ``structural_map`` takes a list of ``(type, callback)`` entries and
calls, on each node, the first callback whose type matches it.

The second ``structural_map`` only checks the result. It takes three functions,
a tuple in Rust and a list of entries in Python, and calls each on the nodes of
its type:

.. tabs::

   .. group-tab:: Rust

      .. code-block:: rust

         /// Reject a complex type, wherever it appears: on a variable, a parameter or an expression.
         fn check_type(ty: Type) -> Result<Type> { ... }

         /// Reject an expression whose type is complex.
         fn check_expr(expr: Expr) -> Result<Expr> { ... }

         /// Reject a call to one of the four complex operators that the rewrite did not consume.
         fn check_call_lowered(call: Call) -> Result<Call> { ... }

   .. group-tab:: Python

      .. code-block:: python

         def _check_type(ty):
             """Reject a complex type, wherever it appears: on a variable, a parameter or an expression."""

         def _check_expr(expr):
             """Reject an expression whose type is complex."""

         def _check_call_lowered(call):
             """Reject a call to one of the four complex operators that the rewrite did not consume."""

Each returns its node unchanged and raises a ``ValueError`` on an unsupported
use, such as a complex function parameter, so the pass reports such a use
before code generation.

Finally, the function becomes a pass:

.. tabs::

   .. group-tab:: Rust

      .. literalinclude:: ../../examples/extension/src/lib.rs
         :language: rust
         :start-after: // begin pass
         :end-before: // end pass

   .. group-tab:: Python

      .. literalinclude:: ../../examples/extension/my_ext_py/my_ext.py
         :language: python
         :pyobject: lower_complex

In Rust, ``FunctionPass::new`` wraps the function as a pass, and
``tvm_ffi_dll_export_typed_func!`` exports ``lower_complex``, which builds it,
so that the Python package can call it from the loaded library. In Python,
``tirx.transform.function_pass`` wraps the function, and the module defines
``lower_complex`` directly.

Next, :doc:`using_ir_extensions` loads the library and compiles the function.
