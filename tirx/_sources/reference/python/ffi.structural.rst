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

tvm_ffi.structural
==================

TIRx builds on TVM-FFI and supports it as a first-class foundation. Every
TIRx IR node is a TVM-FFI object that is shared across languages, so TIRx
uses the native TVM-FFI utilities for walking and rewriting IR directly. We
list these utilities here for completeness. See the `TVM-FFI documentation
<https://tvm.apache.org/ffi/concepts/structural_eq_hash.html>`_ for more
details.

.. seealso:: :doc:`tvm_tirx.tirx.structural <tirx>` provides variants of
   these utilities that also track constraint information, such as loop
   ranges and branch conditions, at each node.

.. currentmodule:: tvm_ffi

Functions
---------

.. autosummary::
   :toctree: generated/

   structural_walk
   structural_map
   structural_visit
   structural_mutate
   structural_equal
   structural_hash
   get_first_structural_mismatch

Classes
-------

.. autosummary::
   :toctree: generated/

   StructuralVisitor
   StructuralMutator
   StructuralKey
   WalkResult
   WalkOrder
   VisitInterrupt
   DefRegionKind
   InplaceMode

.. _ffi-structural-hooks:

Structural Hooks
----------------

An object type tells these utilities how to handle it through structural
hooks, which it can register from C++ or Python:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Hook
     - Purpose
   * - ``__s_visit__``
     - Visits the object's children and marks which fields define
       variables.
   * - ``__s_mutate__``
     - Rewrites the object's children.
   * - ``__s_maybe_inplace_mutate__``
     - Optional. Reuses the object's storage when it is safe to do so.
   * - ``__s_equal__`` and ``__s_hash__``
     - Define structural equality and hashing.

Every TIRx IR node registers compiled hooks; other objects fall back to
their reflected fields.

TIRx marks loop and ``Bind`` variables as ``DEF_SIMPLE``; it marks function
parameters, including the variables in their types, as ``DEF_PATTERN``.

See `Custom Visit and Mutation Hooks
<https://tvm.apache.org/ffi/concepts/structural_eq_hash.html#custom-visit-and-mutation-hooks>`_
in the TVM-FFI documentation for how to write hooks.
