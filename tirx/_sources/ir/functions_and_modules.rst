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
implementation and locate definitions through global references.

Functions
---------

:class:`ir.BaseFunc <tvm_tirx.ir.BaseFunc>` is the common base class for
IR function definitions. It provides the ``attrs`` field for named
function properties. :class:`tirx.Function <tvm_tirx.tirx.Function>` is
its subclass for tensor programs, adding parameters, a return type, and
a statement body:

.. code-block:: text

   ir.BaseFunc                     attrs
   └── tirx.Function               params, ret_type, body

The main fields of ``tirx.Function`` are:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``params``
     - Parameter variables in calling order.
   * - ``ret_type``
     - The declared return type.
   * - ``body``
     - A ``SeqStmt``, or ``None`` for a declaration without an implementation.
   * - ``attrs``
     - An ``ir.DictAttrs`` containing named function properties.

In TVMScript, ``@T.function`` produces this IR object from a function
definition. The CPU example is defined as:

.. literalinclude:: ../get_started/examples/add_one_cpu.py
   :language: python
   :start-at: from tvm_tirx.script import tirx as T
   :end-at:         B[i] = A[i] + 1

The name ``add_one`` refers to the resulting ``tirx.Function``. Its
parameters are the variables ``A`` and ``B``, and its body contains the
loop. The loop has its own body containing the tensor store:

.. code-block:: text

   add_one: tirx.Function
   ├── params[0]: ir.Var                     A
   ├── params[1]: ir.Var                     B
   ├── ret_type: ir.TupleType                no return value
   └── body: ir.SeqStmt
       └── seq[0]: ir.For                    for i in range(n)
           └── body: ir.SeqStmt
               └── seq[0]: ir.TensorStore    B[i] = A[i] + 1

``add_one.ty`` stores an ``ir.FuncType`` describing the function's
signature: its parameter types and return type. We can inspect these
types directly:

.. code-block:: python

   signature = add_one.ty
   parameter_types = signature.arg_types
   return_type = signature.ret_type

``parameter_types`` contains the tensor types of ``A`` and ``B`` in
calling order. The entries in ``add_one.params`` are the variables
themselves. ``return_type`` is the same type stored in
``add_one.ret_type``: an empty tuple type, ``ir.TupleType([])``, indicating
no return value. This function writes its results into ``B``.

A function can also return a value explicitly. The ``add_one_scalar``
helper from :doc:`operators_and_calls` returns a single float32 value:

.. code-block:: python

   @T.function(private=True)
   def add_one_scalar(value: T.float32) -> T.float32:
       return value + 1

The return annotation ``-> T.float32`` sets its ``ret_type`` to
``ir.PrimType("float32")``, and the ``return`` becomes an ``ir.Return``
statement in its body. The result of ``add_one`` goes into an output
parameter; the result of ``add_one_scalar`` is a return value.

So far, ``@T.function`` has built a function object for us from source
code. Compiler code can also build this kind of object directly with
``tirx.Function``. To focus on the function constructor, we can reuse
the parameters and loop already stored in ``add_one``:

.. code-block:: python

   from tvm_tirx import ir, tirx

   A, B = add_one.params
   loop = add_one.body[0]

``A`` and ``B`` are the IR parameter variables, and ``loop`` is the
``ir.For`` shown in the tree above. We can pass these objects to the
constructor to create another function with the same computation:

.. code-block:: python

   ir_function = tirx.Function([A, B], loop, ret_type=ir.TupleType([]))

The new function takes ``A`` and ``B`` as parameters and runs ``loop``
as its body. The constructor wraps this single statement in a ``SeqStmt``.
``ret_type=ir.TupleType([])`` specifies that the function returns no value;
its loop writes the result to ``B``. A function that returns a value needs
its return type supplied through ``ret_type``; the constructor does not
infer it from the body.

We will put ``ir_function`` in a module and compile it below. To make the
compiled function available under the name ``"add_one"``, we set its
``global_symbol`` attribute:

.. code-block:: python

   ir_function = ir_function.with_attr("global_symbol", "add_one")

``with_attr``, inherited from ``ir.BaseFunc``, returns a new function with
this entry in ``attrs``. For public functions written in TVMScript,
``@T.function`` sets
``global_symbol`` to the Python function name automatically.

Modules
-------

A program can contain several functions that call one another.
:class:`ir.IRModule <tvm_tirx.ir.IRModule>` collects their definitions so
compiler passes can find the functions being called and transform the
program as a whole. We can put the function we just constructed,
``ir_function``, in a module under the name ``"add_one"``:

.. code-block:: python

   module = ir.IRModule({"add_one": ir_function})

The module's ``functions`` field maps global references to function
definitions. Each key is an :class:`ir.GlobalVar <tvm_tirx.ir.GlobalVar>`,
and each value is a function definition whose class inherits
``ir.BaseFunc``. Here, that value is our ``tirx.Function`` object,
``ir_function``. The constructor creates a ``GlobalVar`` named
``"add_one"`` from the string key and associates it with this function:

.. code-block:: text

   module: ir.IRModule
   └── functions
       └── key: ir.GlobalVar ("add_one") → value: tirx.Function (ir_function)

The ``GlobalVar`` serves as a reference; the function body belongs to
the associated ``tirx.Function``. We can retrieve the reference by name,
then use it to look up the definition:

.. code-block:: python

   add_one_ref = module.get_global_var("add_one")
   function = module[add_one_ref]

The first line returns the ``GlobalVar`` key, and the second returns
the ``tirx.Function`` stored under it. ``module["add_one"]`` retrieves
the same definition directly by name. A call can use ``add_one_ref`` as
its ``op`` to identify this function, as in the function-call example in
:doc:`operators_and_calls`.

The name ``"add_one"`` used in this lookup is the function's name within
the module. The function's ``attrs["global_symbol"]`` specifies its
exported name for code generation. Both names are ``"add_one"`` in this
example, but they serve these separate purposes.

``tvm_tirx.compile`` accepts a module as well as a standalone function. We
can compile ``module`` with the same LLVM target used in Quick Start,
then create CPU tensors and call its exported ``add_one`` function:

.. code-block:: python

   import torch

   import tvm_tirx

   compiled = tvm_tirx.compile(module, target="llvm")
   a = torch.arange(10, dtype=torch.float32)
   b = torch.empty(a.shape, dtype=a.dtype)
   compiled["add_one"](a, b)

``compiled["add_one"]`` retrieves the exported function. The call reads
``a`` and writes ``a + 1`` into ``b``, just as in Quick Start.
