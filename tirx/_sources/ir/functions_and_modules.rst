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

In this chapter, we will introduce functions and modules as the structures
that organize IR programs. A function combines parameters, a result type,
and a statement body; a module collects named function definitions. Compiler
passes use these structures to inspect a function's interface and
implementation, locate definitions through global references, and replace
an implementation while preserving references to it.

Functions
---------

For tensor programs, use ``@T.function`` to define functions in TVMScript,
as in our sigmoid example. The decorator produces a
:class:`tirx.Function <tvm_tirx.tirx.Function>` IR object. To help users better
understand the IR, we also show how to construct the same representation
directly through the IR API.

``ir.BaseFunc`` is the base class for all function definitions and is also
an expression. Its ``attrs`` field holds named function properties:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Field
     - Value type
     - Meaning
   * - ``attrs``
     - ``ir.DictAttrs``
     - Named function properties.

:class:`tirx.Function <tvm_tirx.tirx.Function>` extends ``ir.BaseFunc`` with
parameters, a result type, and a statement body:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``params``
     - Parameter variables in calling order.
   * - ``ret_type``
     - The declared result type.
   * - ``body``
     - A ``SeqStmt``, or None for a declaration without an implementation.

The parameter and result types form the function's signature, available as
an ``ir.FuncType`` through ``ty``. The ``global_symbol`` entry in ``attrs``
specifies the function's exported name for code generation. For public
functions, ``@T.function`` sets it to the Python function name by default.
The IR API example below sets it explicitly with ``with_attr``.

The program defines a function with input, bias, and output tensors.
Its IR definition combines these three parameters with the batch and
channel loops:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin function construction
   :end-before: # end function construction
   :dedent: 4

``batch_loop`` contains a channel loop that processes one element at a time.
The function writes activated values into ``Y``; ``ir.TupleType([])`` declares
that it returns no value.
The input and output types have shape ``(2*n, channels)``. The symbol ``n``,
whose value the compiled function derives from the tensor shapes at runtime,
is used in these types and the loop bound.

Modules
-------

:class:`ir.IRModule <tvm_tirx.ir.IRModule>` organizes named functions into a
program and lets us look up their definitions. An
:class:`ir.GlobalVar <tvm_tirx.ir.GlobalVar>` is an expression that refers to a
function stored in the module. This differs from the :ref:`ir.Var <user-ir-var>`
objects used for parameters and local values in our example. The module
associates each function reference with its ``ir.BaseFunc`` definition, such
as a ``tirx.Function``.

In the bias + sigmoid example, one module keeps two model specializations.
The same factory builds kernels for 128-channel and 130-channel outputs,
each with its own exported name:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin module construction
   :end-before: # end module construction

Both accept tensors with a runtime row count of ``2*n``. Each specialization
uses a scalar loop over its channels. Their tensor types record the different
widths.

Use ``module.get_global_var(name)`` to retrieve the ``ir.GlobalVar`` reference
for an existing function:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin module lookup
   :end-before: # end module lookup

``entry`` is the function's ``ir.GlobalVar`` reference; ``module[entry]`` and
``module["bias_sigmoid_130"]`` retrieve its current definition. A compiler pass
can replace that definition while keeping ``entry``. If the replacement
preserves the function's calling interface, calls using this reference can
remain unchanged.

For example, a compiler can keep the original expanded sigmoid and build
another module that uses the named ``prim.sigmoid`` operator. The factory's
``activation`` argument selects this representation:

.. literalinclude:: examples/sigmoid.py
   :language: python
   :start-after: # begin module update
   :end-before: # end module update

``variant`` keeps the 128-channel definition and replaces the 130-channel
definition with one that calls ``prim.sigmoid``. The replacement keeps the
same tensor interface and exported name, so ``variant[entry]`` now retrieves
the new implementation through the existing global reference. The original
``module`` still contains the expanded sigmoid expression.
