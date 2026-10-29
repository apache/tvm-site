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

Control Flow
============

.. note::

   All the code in this guide is in
   `examples/programming_guides/control_flow.py
   <https://github.com/apache/tvm-tirx/blob/main/examples/programming_guides/control_flow.py>`_.

This guide shows how to write conditions and loops in TIRx kernels. TIRx uses
Python's own control flow syntax: ``if``, ``elif``, ``else``, ``for``,
``while``, ``break``, and ``continue`` all mean what they mean in Python, and
each thread executes them independently with its own values.

The examples use the following imports and a symbolic length ``n``:

.. literalinclude:: ../../examples/programming_guides/control_flow.py
   :language: python
   :start-after: # [imports.begin]
   :end-before: # [imports.end]

``if`` Statements
-----------------

An ``if`` statement runs its body only on the threads whose condition holds.
The following kernel implements ReLU, which keeps positive values and replaces
the rest with zero:

.. literalinclude:: ../../examples/programming_guides/control_flow.py
   :language: python
   :start-after: # [if.begin]
   :end-before: # [if.end]

The outer ``if i < n`` is the usual bounds check: the last block may have more
threads than elements left, and the check keeps those threads from reading or
writing past the end of the tensors. Inside it, ``if`` and ``else`` choose
between two stores for each element, and ``elif`` can add more branches, just
as in Python. A branch-free ``T.max(A[i], 0.0)`` would also compute ReLU; the
``if`` form shows how threads can take different paths.

``for`` Loops
-------------

A ``for`` loop over ``range(...)`` runs its body once per iteration, in order,
within each thread. ``T.unroll(...)`` iterates in the same way but asks the
compiler to expand the loop into one copy of its body per iteration. The
following kernel uses both to sum each row of an ``n x 64`` matrix, one row
per thread:

.. literalinclude:: ../../examples/programming_guides/control_flow.py
   :language: python
   :start-after: # [for.begin]
   :end-before: # [for.end]

The outer ``range(16)`` loop stays a loop in the generated code, while the
inner ``T.unroll(4)`` loop becomes four consecutive additions. Unrolling suits
short loops with a fixed trip count, where it removes the loop overhead and
lets the compiler schedule the statements together. ``total`` is a mutable
local scalar, declared with a type annotation so that it keeps its value
across iterations.

For nested loops over several dimensions, ``T.grid`` iterates over all index
combinations, like nested ``range`` loops. The following kernel transposes a
64 x 64 matrix, with each thread moving a 4 x 4 tile:

.. literalinclude:: ../../examples/programming_guides/control_flow.py
   :language: python
   :start-after: # [grid.begin]
   :end-before: # [grid.end]

``for r, c in T.grid(4, 4)`` visits the 16 pairs ``(r, c)`` row by row, which
is the same as ``for r in range(4)`` with ``for c in range(4)`` nested inside.

``while`` Loops
---------------

A ``while`` loop repeats its body as long as its condition holds, which suits
loops whose trip count depends on the data. The following kernel finds, for
each query, the position where it would be inserted into a sorted tensor of 64
values, using binary search:

.. literalinclude:: ../../examples/programming_guides/control_flow.py
   :language: python
   :start-after: # [while.begin]
   :end-before: # [while.end]

Each thread narrows its own range ``[lo, hi)`` until it is empty, so different
threads may run different numbers of iterations. ``lo`` and ``hi`` are mutable
local scalars, so they keep their updates from one iteration to the next.

``break`` and ``continue``
--------------------------

``break`` exits the innermost loop, and ``continue`` skips to its next
iteration, just as in Python. Both work in ``for`` and ``while`` loops, and
each thread decides on its own when to break or continue.

Early Thread Exit
-----------------

``T.gpu_thread_return()`` ends the current thread early. A common use is the
bounds check, written as an early exit instead of wrapping the rest of the
kernel in an ``if``:

.. literalinclude:: ../../examples/programming_guides/control_flow.py
   :language: python
   :start-after: # [return.begin]
   :end-before: # [return.end]

Threads past the end of the tensors exit at ``T.gpu_thread_return()``, and the
remaining threads continue with the store. A plain Python ``return`` is
different: it returns from the whole function and requires a value, so it
cannot end a single GPU thread.
