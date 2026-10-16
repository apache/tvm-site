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

Target and Code Generation
==========================

.. contents:: Table of Contents
   :local:
   :depth: 1

tvm_tirx.target
---------------

.. currentmodule:: tvm_tirx.target

.. autosummary::
   :toctree: generated/

   Target
   TargetKind
   device_from_target
   list_tags
   register_tag


tvm_tirx.target.target
----------------------

.. currentmodule:: tvm_tirx.target.target

.. autosummary::
   :toctree: generated/

   TargetFeatures

* :obj:`Target <tvm_tirx.target.Target>`
* :obj:`TargetKind <tvm_tirx.target.TargetKind>`
* :obj:`device_from_target <tvm_tirx.target.device_from_target>`

tvm_tirx.target.tag
-------------------

.. currentmodule:: tvm_tirx.target.tag

* :obj:`list_tags <tvm_tirx.target.list_tags>`
* :obj:`register_tag <tvm_tirx.target.register_tag>`

tvm_tirx.target.detect_target
-----------------------------

.. currentmodule:: tvm_tirx.target.detect_target

.. autosummary::
   :toctree: generated/

   detect_target_from_device
   register_device_target_detector


tvm_tirx.codegen
----------------

.. currentmodule:: tvm_tirx.codegen

.. autosummary::
   :toctree: generated/

   Executable
   build_module
   load_static_library
   target_has_features


tvm_tirx.codegen.llvm
---------------------

.. currentmodule:: tvm_tirx.codegen.llvm

.. autosummary::
   :toctree: generated/

   cpu_has_features
   get_cpu_archlist
   get_cpu_features
   get_intrinsic_name
   get_system_cpu
   get_system_triple
   get_system_x86_vendor
   get_targets
   get_vector_width
   is_valid_cpu
   lookup_intrinsic_id
   version_major
