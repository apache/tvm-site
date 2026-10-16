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

Runtime and Top-Level Utilities
===============================

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx
--------

.. currentmodule:: tvm_tirx

.. autosummary::
   :toctree: generated/

   get_global_func
   hexagon
   register_global_func
   register_object

* :obj:`DataType <tvm_tirx.runtime.DataType>`
* :obj:`DataTypeCode <tvm_tirx.runtime.DataTypeCode>`
* :obj:`IRModule <tvm_tirx.ir.IRModule>`
* :obj:`Object <tvm_tirx.runtime.Object>`
* :obj:`cpu <tvm_tirx.runtime.cpu>`
* :obj:`cuda <tvm_tirx.runtime.cuda>`
* :obj:`device <tvm_tirx.runtime.device>`
* :obj:`device_from_target <tvm_tirx.target.device_from_target>`
* :obj:`ext_dev <tvm_tirx.runtime.ext_dev>`
* :obj:`metal <tvm_tirx.runtime.metal>`
* :obj:`opencl <tvm_tirx.runtime.opencl>`
* :obj:`rocm <tvm_tirx.runtime.rocm>`
* :obj:`vpi <tvm_tirx.runtime.vpi>`
* :obj:`vulkan <tvm_tirx.runtime.vulkan>`

tvm_tirx.runtime
----------------

.. currentmodule:: tvm_tirx.runtime

.. autosummary::
   :toctree: generated/

   DataType
   DataTypeCode
   Device
   MemoryScope
   Module
   Object
   ShapeTuple
   Tensor
   convert
   cpu
   cuda
   device
   empty
   enabled
   ext_dev
   from_dlpack
   load_module
   metal
   num_threads
   opencl
   rocm
   system_lib
   tensor
   vpi
   vulkan


tvm_tirx.runtime.memory_scope
-----------------------------

.. currentmodule:: tvm_tirx.runtime.memory_scope

* :obj:`MemoryScope <tvm_tirx.runtime.MemoryScope>`

tvm_tirx.config
---------------

.. currentmodule:: tvm_tirx.config

.. autosummary::
   :toctree: generated/

   configuration


tvm_tirx.libinfo
----------------

.. currentmodule:: tvm_tirx.libinfo

.. autosummary::
   :toctree: generated/

   configuration
   find_include_path
   find_libtvm_runtime
   package_lib_paths
