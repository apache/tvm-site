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

CUDA Script Namespaces
======================

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.backend.cuda.script.iket
---------------------------------

.. currentmodule:: tvm_tirx.backend.cuda.script.iket

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   mark
   official_event
   range_end
   range_pop
   range_push
   range_start
   sentinel_token


tvm_tirx.backend.cuda.script.tcgen05
------------------------------------

.. currentmodule:: tvm_tirx.backend.cuda.script.tcgen05

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   encode_instr_descriptor
   encode_instr_descriptor_block_scaled
   encode_matrix_descriptor


tvm_tirx.backend.cuda.script.tile
---------------------------------

.. currentmodule:: tvm_tirx.backend.cuda.script.tile

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   apply
   ewise


tvm_tirx.backend.cuda.script.wgmma
----------------------------------

.. currentmodule:: tvm_tirx.backend.cuda.script.wgmma

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   encode_matrix_descriptor
   noop_barrier


tvm_tirx.tirx.script.nvshmem
----------------------------

.. currentmodule:: tvm_tirx.tirx.script.nvshmem

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   barrier_all
   fence
   getmem_nbi
   my_pe
   n_pes
   putmem_nbi
   putmem_signal_nbi
   quiet
   signal_op
   wait_until


tvm_tirx.tirx.script.nvshmem.getmem_nbi
---------------------------------------

.. currentmodule:: tvm_tirx.tirx.script.nvshmem.getmem_nbi

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   block
   warp


tvm_tirx.tirx.script.nvshmem.putmem_nbi
---------------------------------------

.. currentmodule:: tvm_tirx.tirx.script.nvshmem.putmem_nbi

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   block
   warp


tvm_tirx.tirx.script.nvshmem.putmem_signal_nbi
----------------------------------------------

.. currentmodule:: tvm_tirx.tirx.script.nvshmem.putmem_signal_nbi

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   block
   warp


tvm_tirx.tirx.script.ptx
------------------------

.. currentmodule:: tvm_tirx.tirx.script.ptx

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   abs
   activemask
   add
   addr
   and
   applypriority
   atom
   bar
   barrier
   bfe
   bfi
   bfind
   bmsk
   brev
   clmad
   clusterlaunchcontrol
   clz
   cnot
   copysign
   cos
   cp
   createpolicy
   cvt
   cvt_pack
   cvta
   discard
   div
   dp2a
   dp4a
   elect_sync
   ex2
   fabric
   fence
   fma
   fns
   getctarank
   griddepcontrol
   isspacep
   ld
   ldmatrix
   ldu
   lg2
   lop3
   mad
   mad24
   mapa
   match
   max
   mbarrier
   min
   mma
   mov
   movmatrix
   mul
   mul24
   multimem_cp
   multimem_ld_reduce
   multimem_red
   multimem_red_async
   multimem_st
   multimem_st_async
   neg
   not
   or
   popc
   pred
   prefetch
   prefetchu
   prmt
   rcp
   red
   red_async
   redux_sync
   rem
   rsqrt
   sad
   selp
   set
   setmaxnreg
   setp
   shf
   shfl_sync
   shl
   shr
   sin
   slct
   spcompress
   spdecompress
   sqrt
   st
   st_async
   st_bulk
   stmatrix
   sub
   szext
   tanh
   tcgen05
   tensormap_cp_fenceproxy
   tensormap_replace
   testp
   vote_sync
   wgmma
   xor


tvm_tirx.tirx.script.ptx_legacy
-------------------------------

.. currentmodule:: tvm_tirx.tirx.script.ptx_legacy

.. autosummary::
   :toctree: generated/
   :template: autosummary/callable.rst

   ldmatrix
   mma
