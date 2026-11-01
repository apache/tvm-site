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


Symbolic Integer Analysis
=========================

Tensor programs compute indices from symbolic values such as a tensor length
``n``, whose actual value is known only at runtime. Many analyses and
transformations still need to reason about these indices at compile time: to
simplify an index calculation, or to prove that an access stays in bounds so
that a runtime check can be removed. TIRx provides a set of lightweight
symbolic integer analysis utilities to help us with this reasoning and
simplification:

- **Simplification**: an :py:class:`~tvm_tirx.sym.Analyzer` rewrites an
  integer expression into a simpler equivalent form.
- **Proofs**: the analyzer checks whether a condition, such as
  ``i // 4 < n``, always holds.
- **Facts**: a :py:class:`~tvm_tirx.sym.ConstraintContext` records what is
  known at a point in the program, such as a loop's range or a branch
  condition, and the analyzer uses those facts in its simplifications and
  proofs.

To see these utilities at work, consider a variant of ``add_one`` that works
on a two-dimensional tensor. A single loop visits all ``n * 4`` elements, and
the indices ``i // 4`` and ``i % 4`` select the row and column of each one:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [add_one_2d.begin]
   :end-before: # [add_one_2d.end]

Before generating code, a compiler typically flattens each tensor to one
dimension. The function keeps its interface, while the body accesses
one-dimensional views of the same storage, declared with
:py:func:`T.decl_tensor <tvm_tirx.tirx.script.decl_tensor>`. Element
``(r, c)`` of an ``(n, 4)`` tensor sits at position ``r * 4 + c`` in such a
view, so the program becomes:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [add_one_1d.begin]
   :end-before: # [add_one_1d.end]

The flattened index computes a division and a remainder for every element,
yet it always equals ``i``. Recognizing this, and many similar facts about
index expressions, is what symbolic integer analysis is for.

.. note::

   The runnable code in this chapter is under
   `examples/ir_guides <https://github.com/apache/tvm-tirx/tree/main/examples/ir_guides>`_
   in the repository.

Simplify and Prove
------------------

An ``Analyzer`` answers questions about integer expressions. Let's start with
the index of the flattened program. We read it from the store in
``add_one_1d`` and ask the analyzer to simplify it:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [simplify.begin]
   :end-before: # [simplify.end]

The output is:

.. code-block:: text

   i

The analyzer recognizes that taking the row ``i // 4``, multiplying it by the
row length, and adding the column ``i % 4`` gives back ``i``. This holds for
any ``i``, so the analyzer needs no further information, and a pass can
replace the index with ``i`` and drop the division and the remainder.

Not every question can be answered so directly. In ``add_one_2d``, the row
index ``i // 4`` must stay below ``n`` for the access to be valid.
:py:meth:`can_prove <tvm_tirx.sym.Analyzer.can_prove>` checks whether a
condition always holds:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [prove.begin]
   :end-before: # [prove.end]

The output is:

.. code-block:: text

   False

A ``False`` result means the analyzer could not establish the condition, not
that the condition is false. Here the analyzer knows nothing about ``i``. As
far as it can tell, ``i`` could be any integer. The bound depends on the
loop, and the analyzer needs that information as a fact.

Constraint Context
------------------

A ``ConstraintContext`` holds the facts known at a point in the program.
Inside the loop, ``i`` lies in ``[0, n * 4)``. We record that range and
synchronize the analyzer with the context:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [loop_range.begin]
   :end-before: # [loop_range.end]

The output is:

.. code-block:: text

   False

Still not proved. When ``n`` is zero or less, the range ``[0, n * 4)`` is empty and
the loop never runs. Inside the loop body, though, we know the loop is
running, so its extent is positive: ``n * 4 > 0``. With that fact, the proof
succeeds:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [loop_runs.begin]
   :end-before: # [loop_runs.end]

The output is:

.. code-block:: text

   True

Facts also come from branches. Inside ``if i % 4 < 2``, the condition holds,
so it becomes a fact. We add it to a snapshot of the loop context, so the
original context stays unchanged:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [branch.begin]
   :end-before: # [branch.end]

The output is:

.. code-block:: text

   True
   False

Inside the branch, the analyzer proves ``i % 4 < 3``. Synchronizing with
``loop_context`` again models leaving the branch, and the fact no longer
applies. An analyzer uses only the facts of the context it was last
synchronized with, which lets the same analyzer serve different points in a
program.

Structural Walk and Map with Constraint Context
-----------------------------------------------

Building contexts by hand works for one query, but a pass over a whole
function would have to rebuild the facts at every node. The structural
utilities can collect them for us.
:py:func:`tirx.structural_walk_with_constraint_context <tvm_tirx.tirx.structural.structural_walk_with_constraint_context>`
and
:py:func:`tirx.structural_map_with_constraint_context <tvm_tirx.tirx.structural.structural_map_with_constraint_context>`
work like walk and map, and also pass the facts valid at each node to our
callback.

Compilers often insert bounds checks to stay safe, such as the
``if i // 4 < n`` below. Here the loop already guarantees the condition, so
the check only costs time on every iteration:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [add_one_guarded.begin]
   :end-before: # [add_one_guarded.end]

A walk can check whether each condition is provable from the facts at that
point:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [walk.begin]
   :end-before: # [walk.end]

The output is:

.. code-block:: text

   [True]

The traversal collected the loop range and the fact that the loop runs,
``n * 4 > 0``, the same facts we added by hand. The callback sees the facts
that hold at the ``if`` itself, so the branch's own condition is not a fact
there.

A map with the same callback structure removes the check by returning the
branch body in place of the ``if``:

.. literalinclude:: ../../examples/ir_guides/symbolic_integer_analysis.py
   :language: python
   :start-after: # [map.begin]
   :end-before: # [map.end]

The output is:

.. code-block:: python

   from __future__ import annotations

   # from tvm_tirx.script import tirx as T

   @T.function
   def add_one_guarded[n](A: T.Tensor((n, 4), T.float32), B: T.Tensor((n, 4), T.float32)):
       for i in range(n * 4):
           B[i // 4, i % 4] = A[i // 4, i % 4] + T.float32(1.0)

Visit and mutate variants,
:py:func:`structural_visit_with_constraint_context <tvm_tirx.tirx.structural.structural_visit_with_constraint_context>`
and
:py:func:`structural_mutate_with_constraint_context <tvm_tirx.tirx.structural.structural_mutate_with_constraint_context>`,
provide the same facts to callbacks that control their own traversal.

A transformation like ``remove_redundant_guard`` is an ordinary function, and
we can call it directly. When it helps, we can also wrap it as a pass with
:py:func:`~tvm_tirx.tirx.transform.function_pass` and compose it with other
passes, as described in the :doc:`pass API </reference/python/ir.transform>`.

Beyond simplification and proofs, ``tvm_tirx.sym`` also provides bounds,
divisibility, iteration-map, and region analyses, documented in the
:doc:`tvm_tirx.sym reference </reference/python/sym>`.
