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

Functions and Modules
=====================

Functions define computations through parameters, result types, and statement
bodies, while modules collect named function definitions. Extensions can build
programs with custom types and operators using these same structures, and
passes can inspect or replace definitions through the existing interfaces.

Functions
---------

``ir.BaseFunc`` is the shared base class for function definitions and is itself
an expression. It inherits ``ty`` and ``loc`` from ``ir.Expr`` and adds a field
shared by function definitions:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``attrs``
     - ``ir.DictAttrs``
     - Named function properties.

``tirx.Function`` adds these fields:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``params``
     - Parameter variables in calling order.
   * - ``ret_type``
     - The declared result type; an omitted type becomes an empty ``TupleType``.
   * - ``body``
     - A ``SeqStmt``, or None for a declaration without an implementation.

For example, this function takes an int32 input and returns that value plus
one:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin function construction
   :end-before: # end function construction
   :prepend: from tvm_tirx import ir, tirx

``value`` is defined as a parameter and used by the expression in the return
statement. The function's ``ty`` is ``ir.FuncType([value.ty], value.ty)``:
construction derives this signature from the parameter types and ``ret_type``.
The return type is declared explicitly; it is not inferred from the body.
Passing None for ``body`` creates a declaration with a signature but no
implementation.

Modules
-------

``ir.IRModule`` maps ``ir.GlobalVar`` references to ``ir.BaseFunc`` definitions.
Calls can use these references to identify a function, allowing a pass to
replace its definition while retaining the reference. Place the increment
function in a module under the name ``main``:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin module construction
   :end-before: # end module construction

``entry`` is the reference; ``function`` contains the parameters and body.
``module[entry]`` and ``module["main"]`` retrieve that same definition.
``global_symbol`` sets the exported function name used by code generation.

Use the stored reference to construct a call representing ``main(3)``:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin global call
   :end-before: # end global call

This creates a call expression with an int32 result type. The signature lives
on ``function.ty``; insertion does not copy it to ``entry.ty``, so the call
supplies its result type explicitly. Reuse ``entry`` or retrieve it with
``module.get_global_var("main")`` when referring to this definition.

To update a definition while keeping the original module, clone the module
and replace the definition in the clone. This example adds an attribute to
the cloned module's ``main`` function:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin module update
   :end-before: # end module update

``with_attr`` returns a new function, and assignment updates ``variant``'s
function map. The original ``module["main"]`` keeps its attributes. Use
:meth:`~tvm_tirx.tirx.Function.with_body` to replace a function body, and see
:class:`~tvm_tirx.ir.IRModule` for lookup, copying, and update APIs.

A module also carries metadata: ``attrs`` stores named attributes, and
``global_infos`` stores named arrays of ``ir.GlobalInfo``. Save a module as JSON
to reconstruct its IR later:

.. literalinclude:: examples/statements_and_functions.py
   :language: python
   :start-after: # begin module serialization
   :end-before: # end module serialization

Structural comparison checks the representation, including corresponding bound
variables. Loading requires the graph's object types to be registered in the
receiving process.

See :class:`~tvm_tirx.tirx.Function` and :class:`~tvm_tirx.ir.IRModule` for the
complete APIs. The :download:`statements_and_functions.py <examples/statements_and_functions.py>`
example checks scopes and updates, compiles with LLVM, and verifies that an
input of three returns four.
