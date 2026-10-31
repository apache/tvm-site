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

A function describes one piece of computation: its parameters and types
say how it is called, and its body says what it does. In many cases, a
program needs multiple functions that interact with each other, such as a
host function that launches device kernels. A module collects these
functions under their names, so calls can refer to one another and passes
can transform the whole program.

In TVMScript, a module is written as a class marked with ``@T.ir_module``,
with one ``@T.function`` per member:

.. literalinclude:: ../../examples/ir_walkthrough/functions_and_modules.py
   :language: python
   :start-after: # [module.begin]
   :end-before: # [module.end]

``add_one`` and ``add_two`` add one and two to each element of ``A``.
The rest of this chapter looks at the functions and then at how the
module ties them together.

.. note::

   The runnable code in this chapter is under
   `examples/ir_walkthrough <https://github.com/apache/tvm-tirx/tree/main/examples/ir_walkthrough>`_
   in the repository.

Functions
---------

A ``tirx.Function`` has three fields of its own:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - What it holds
   * - ``params``
     - The parameter variables, in calling order. ``add_one`` has ``A`` and
       ``B``.
   * - ``ret_type``
     - The type of the returned value. ``add_one`` returns nothing, an
       empty tuple type, and writes its result into ``B`` instead. A
       function that returns a value declares it with ``->``, as in
       ``def f(x: T.float32) -> T.float32``.
   * - ``body``
     - The statements that run when the function is called.

It also has the fields every ``ir.BaseFunc`` shares: ``attrs``, described
below; ``ty``, the function's signature, built from its parameter and
return types; and ``loc``, its source location.

Attributes
~~~~~~~~~~

``attrs`` allows us to store additional attribute information about a
function. For example, ``global_symbol`` locks in the symbol name of a
public function: ``add_one`` is exported as ``"add_one"`` in the compiled
library. ``@T.function`` sets it from the Python name. A function
marked ``@T.function(private=True)`` has no ``global_symbol`` and stays
internal to the module.

In TVMScript, ``T.func_attr`` adds attributes in the function body, and
the printer writes them back the same way:

.. literalinclude:: ../../examples/ir_walkthrough/functions_and_modules.py
   :language: python
   :start-after: # [func_attr.begin]
   :end-before: # [func_attr.end]
   :dedent: 4

From Python, ``with_attr`` returns a new function with an attribute added
and leaves the original unchanged:

.. literalinclude:: ../../examples/ir_walkthrough/functions_and_modules.py
   :language: python
   :start-after: # [with_attr.begin]
   :end-before: # [with_attr.end]

Modules
-------

A module has three main fields:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Field
     - What it holds
   * - ``functions``
     - The module's functions, keyed by ``GlobalVar``.
   * - ``attrs``
     - Additional attribute information about the whole module, in the same
       form as a function's ``attrs``.
   * - ``global_infos``
     - Reserved for special use cases that need named objects shared
       across functions. Most modules leave it empty.

``functions`` is keyed by ``GlobalVar``, which the next section explains.
The module also keeps an index from names to ``GlobalVar`` keys, which
``Module["add_one"]`` uses to look up a function by name.

.. _ir-global-var:

Global Variables
----------------

``ir.GlobalVar`` is an expression, listed in the
:ref:`expression table <user-ir-expr>`, that refers to a function in a
module. The module's ``functions`` map is keyed by ``GlobalVar``: we can
get the reference by name and look up its definition:

.. literalinclude:: ../../examples/ir_walkthrough/functions_and_modules.py
   :language: python
   :start-after: # [global_var.begin]
   :end-before: # [global_var.end]

Keeping the reference separate from the definition lets IR refer to a
function without containing its body. For example, a ``Call`` whose
``op`` is a ``GlobalVar`` calls that function, and a pass can rewrite or
replace the definition without touching the places that refer to it.
In practice, references between functions mostly connect host and device
functions, such as a host function and the kernels it launches.

The ``GlobalVar``'s name is the function's name within the module. It is
separate from ``global_symbol``, the name a public function is exported
under. For ``add_one``, both are ``"add_one"``; a private function has a
module name but no exported symbol.
