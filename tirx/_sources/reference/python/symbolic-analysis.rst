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

Symbolic Analysis
=================

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.sym
------------

.. currentmodule:: tvm_tirx.sym

.. autosummary::
   :toctree: generated/

   Analyzer
   CompareResult
   ConstIntBound
   IntSet
   IntervalSet
   IterMapExpr
   IterMark
   IterSplitExpr
   IterSumExpr
   ModularSet
   ProofStrength
   allow_uint_as_index
   deduce_bound
   detect_clip_bound
   detect_iter_map
   detect_linear_equation
   estimate_region_lower_bound
   estimate_region_strict_bound
   estimate_region_upper_bound
   inverse_affine_iter_map
   iter_map_simplify
   normalize_iter_map_to_expr
   normalize_to_iter_sum
   subspace_divide


tvm_tirx.sym.analyzer
---------------------

.. currentmodule:: tvm_tirx.sym.analyzer

.. autosummary::
   :toctree: generated/

   ConstraintScope

* :obj:`Analyzer <tvm_tirx.sym.Analyzer>`
* :obj:`CompareResult <tvm_tirx.sym.CompareResult>`
* :obj:`ConstIntBound <tvm_tirx.sym.ConstIntBound>`
* :obj:`ModularSet <tvm_tirx.sym.ModularSet>`
* :obj:`ProofStrength <tvm_tirx.sym.ProofStrength>`
* :obj:`allow_uint_as_index <tvm_tirx.sym.allow_uint_as_index>`

tvm_tirx.sym.int_set
--------------------

.. currentmodule:: tvm_tirx.sym.int_set

.. autosummary::
   :toctree: generated/

   neg_inf
   pos_inf
   union_lower_bound

* :obj:`IntSet <tvm_tirx.sym.IntSet>`
* :obj:`IntervalSet <tvm_tirx.sym.IntervalSet>`
* :obj:`estimate_region_lower_bound <tvm_tirx.sym.estimate_region_lower_bound>`
* :obj:`estimate_region_strict_bound <tvm_tirx.sym.estimate_region_strict_bound>`
* :obj:`estimate_region_upper_bound <tvm_tirx.sym.estimate_region_upper_bound>`

tvm_tirx.sym.iter_affine_map
----------------------------

.. currentmodule:: tvm_tirx.sym.iter_affine_map

.. autosummary::
   :toctree: generated/

   IterMapLevel
   IterMapResult

* :obj:`IterMapExpr <tvm_tirx.sym.IterMapExpr>`
* :obj:`IterMark <tvm_tirx.sym.IterMark>`
* :obj:`IterSplitExpr <tvm_tirx.sym.IterSplitExpr>`
* :obj:`IterSumExpr <tvm_tirx.sym.IterSumExpr>`
* :obj:`detect_iter_map <tvm_tirx.sym.detect_iter_map>`
* :obj:`inverse_affine_iter_map <tvm_tirx.sym.inverse_affine_iter_map>`
* :obj:`iter_map_simplify <tvm_tirx.sym.iter_map_simplify>`
* :obj:`normalize_iter_map_to_expr <tvm_tirx.sym.normalize_iter_map_to_expr>`
* :obj:`normalize_to_iter_sum <tvm_tirx.sym.normalize_to_iter_sum>`
* :obj:`subspace_divide <tvm_tirx.sym.subspace_divide>`

tvm_tirx.sym.pattern
--------------------

.. currentmodule:: tvm_tirx.sym.pattern

* :obj:`detect_clip_bound <tvm_tirx.sym.detect_clip_bound>`
* :obj:`detect_linear_equation <tvm_tirx.sym.detect_linear_equation>`

tvm_tirx.sym.utils
------------------

.. currentmodule:: tvm_tirx.sym.utils

.. autosummary::
   :toctree: generated/

   get_const_int
