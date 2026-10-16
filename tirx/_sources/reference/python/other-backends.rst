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

Other Backends
==============

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.backend.adreno
-----------------------

.. currentmodule:: tvm_tirx.backend.adreno

.. autosummary::
   :toctree: generated/

   register_backend


tvm_tirx.backend.hexagon
------------------------

.. currentmodule:: tvm_tirx.backend.hexagon

.. autosummary::
   :toctree: generated/

   register_backend


tvm_tirx.backend.hexagon.linker
-------------------------------

.. currentmodule:: tvm_tirx.backend.hexagon.linker

.. autosummary::
   :toctree: generated/

   hexagon_link
   link_shared
   register_linker


tvm_tirx.backend.metal
----------------------

.. currentmodule:: tvm_tirx.backend.metal

.. autosummary::
   :toctree: generated/

   register_backend
   script_namespace
   script_namespaces


tvm_tirx.backend.metal.op
-------------------------

.. currentmodule:: tvm_tirx.backend.metal.op

.. autosummary::
   :toctree: generated/

   cooperative_tensor_fill
   cooperative_tensor_load
   cooperative_tensor_multiply_accumulate
   cooperative_tensor_store
   make_filled_simdgroup_matrix
   simd_shuffle
   simd_shuffle_down
   simd_shuffle_up
   simdgroup_load
   simdgroup_multiply_accumulate
   simdgroup_store


tvm_tirx.backend.metal.script
-----------------------------

.. currentmodule:: tvm_tirx.backend.metal.script

.. autosummary::
   :toctree: generated/

   MetalNamespace


tvm_tirx.backend.opencl
-----------------------

.. currentmodule:: tvm_tirx.backend.opencl

.. autosummary::
   :toctree: generated/

   nd_mem_alloc_with_scope
   register_backend
   texture2d_load
   texture2d_store


tvm_tirx.backend.opencl.script
------------------------------

.. currentmodule:: tvm_tirx.backend.opencl.script

.. autosummary::
   :toctree: generated/

   nd_mem_alloc_with_scope
   texture2d_load
   texture2d_store


tvm_tirx.backend.rocm
---------------------

.. currentmodule:: tvm_tirx.backend.rocm

.. autosummary::
   :toctree: generated/

   register_backend


tvm_tirx.backend.trn
--------------------

.. currentmodule:: tvm_tirx.backend.trn

.. autosummary::
   :toctree: generated/

   register_backend
   script_namespace
   script_namespaces


tvm_tirx.backend.trn.axis
-------------------------

.. currentmodule:: tvm_tirx.backend.trn.axis

.. autosummary::
   :toctree: generated/

   register


tvm_tirx.backend.trn.layout
---------------------------

.. currentmodule:: tvm_tirx.backend.trn.layout

.. autosummary::
   :toctree: generated/

   is_trainium_layout


tvm_tirx.backend.trn.thread_scope
---------------------------------

.. currentmodule:: tvm_tirx.backend.trn.thread_scope

.. autosummary::
   :toctree: generated/

   ThreadScope


tvm_tirx.backend.trn.op
-----------------------

.. currentmodule:: tvm_tirx.backend.trn.op

.. autosummary::
   :toctree: generated/

   nki_activation
   nki_activation_reduce
   nki_affine_select
   nki_identity
   nki_load
   nki_matmul
   nki_memset
   nki_reciprocal
   nki_scalar_tensor_scalar
   nki_scalar_tensor_tensor
   nki_store
   nki_tensor_copy
   nki_tensorreduce
   nki_tensorscalar
   nki_tensorscalar_reduce
   nki_tensortensor


tvm_tirx.backend.trn.pipeline
-----------------------------

.. currentmodule:: tvm_tirx.backend.trn.pipeline

.. autosummary::
   :toctree: generated/

   finalize_device_passes_trn
   trn_pipeline


tvm_tirx.backend.trn.script
---------------------------

.. currentmodule:: tvm_tirx.backend.trn.script

.. autosummary::
   :toctree: generated/

   NKINamespace


tvm_tirx.backend.trn.transform
------------------------------

.. currentmodule:: tvm_tirx.backend.trn.transform

.. autosummary::
   :toctree: generated/

   LowerTIRx
   LowerTrainiumLayout
   TrnNaiveAllocator
   TrnPrivateTensorAlloc


tvm_tirx.backend.vulkan
-----------------------

.. currentmodule:: tvm_tirx.backend.vulkan

.. autosummary::
   :toctree: generated/

   register_backend


tvm_tirx.backend.webgpu
-----------------------

.. currentmodule:: tvm_tirx.backend.webgpu

.. autosummary::
   :toctree: generated/

   register_backend
