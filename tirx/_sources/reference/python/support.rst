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

Support
=======

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.support
----------------

.. currentmodule:: tvm_tirx.support

.. autosummary::
   :toctree: generated/

   describe
   libinfo


tvm_tirx.support.cc
-------------------

.. currentmodule:: tvm_tirx.support.cc

.. autosummary::
   :toctree: generated/

   create_executable
   create_shared
   create_staticlib
   cross_compiler
   get_cc
   get_global_symbol_section_map
   get_target_by_dump_machine


tvm_tirx.support.clang
----------------------

.. currentmodule:: tvm_tirx.support.clang

.. autosummary::
   :toctree: generated/

   create_llvm
   find_clang


tvm_tirx.support.ndk
--------------------

.. currentmodule:: tvm_tirx.support.ndk

.. autosummary::
   :toctree: generated/

   create_shared
   create_staticlib
   get_global_symbol_section_map


tvm_tirx.support.nvcc
---------------------

.. currentmodule:: tvm_tirx.support.nvcc

.. autosummary::
   :toctree: generated/

   callback_libdevice_path
   compile_cuda
   find_cuda_path
   find_libdevice_path
   find_nvshmem_paths
   get_cuda_version
   get_target_compute_version
   have_bf16
   have_cudagraph
   have_fp16
   have_fp4
   have_fp8
   have_int8
   have_tensorcore
   parse_compute_version
   tvm_callback_cuda_compile


tvm_tirx.support.popen_pool
---------------------------

.. currentmodule:: tvm_tirx.support.popen_pool

.. autosummary::
   :toctree: generated/

   MapResult
   PopenPoolExecutor
   PopenWorker
   StatusKind
   kill_child_processes


tvm_tirx.support.rocm
---------------------

.. currentmodule:: tvm_tirx.support.rocm

.. autosummary::
   :toctree: generated/

   callback_rocm_bitcode_path
   callback_rocm_link
   find_lld
   find_rocm_path
   get_rocm_arch
   have_matrixcore
   parse_compute_version
   rocm_link


tvm_tirx.support.tar
--------------------

.. currentmodule:: tvm_tirx.support.tar

.. autosummary::
   :toctree: generated/

   normalize_file_list_by_unpacking_tars
   tar
   untar


tvm_tirx.support.utils
----------------------

.. currentmodule:: tvm_tirx.support.utils

.. autosummary::
   :toctree: generated/

   DirectoryCreatedPastAtExit
   FileLock
   TempDirectory
   filelock
   is_source_path
   tempdir
   which


tvm_tirx.support.xcode
----------------------

.. currentmodule:: tvm_tirx.support.xcode

.. autosummary::
   :toctree: generated/

   compile_coreml
   compile_metal
   create_dylib
   xcrun
