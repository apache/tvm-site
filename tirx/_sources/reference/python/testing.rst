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

Testing
=======

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.testing
----------------

.. currentmodule:: tvm_tirx.testing

.. autosummary::
   :toctree: generated/

   assert_allclose
   device_enabled
   echo
   generate_random_array
   identity_after
   main
   np_dtype_from_str
   object_use_count
   parameter
   run_with_gpu_lock
   terminate_self
   test_raise_error


tvm_tirx.testing.env
--------------------

.. currentmodule:: tvm_tirx.testing.env

.. autosummary::
   :toctree: generated/

   build_flag_enabled
   cuda_arch
   has_adreno_opencl
   has_cpu_feature
   has_cublas
   has_cuda
   has_cuda_arch
   has_cuda_compute
   has_cudagraph
   has_cudnn
   has_gpu
   has_hexagon
   has_hexagon_toolchain
   has_hipblas
   has_llvm
   has_llvm_min_version
   has_matrixcore
   has_metal
   has_multi_gpu
   has_nccl
   has_nvcc_version
   has_nvptx
   has_nvshmem
   has_opencl
   has_rocm
   has_vulkan
   has_x86_avx512
   has_x86_vnni


tvm_tirx.testing.locking
------------------------

.. currentmodule:: tvm_tirx.testing.locking

* :obj:`run_with_gpu_lock <tvm_tirx.testing.run_with_gpu_lock>`

tvm_tirx.testing.utils
----------------------

.. currentmodule:: tvm_tirx.testing.utils

* :obj:`assert_allclose <tvm_tirx.testing.assert_allclose>`
* :obj:`device_enabled <tvm_tirx.testing.device_enabled>`
* :obj:`generate_random_array <tvm_tirx.testing.generate_random_array>`
* :obj:`identity_after <tvm_tirx.testing.identity_after>`
* :obj:`main <tvm_tirx.testing.main>`
* :obj:`np_dtype_from_str <tvm_tirx.testing.np_dtype_from_str>`
* :obj:`parameter <tvm_tirx.testing.parameter>`
* :obj:`terminate_self <tvm_tirx.testing.terminate_self>`
