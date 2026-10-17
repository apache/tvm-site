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

tvm_tirx
========

.. automodule:: tvm_tirx
   :no-members:

.. currentmodule:: tvm_tirx

Classes
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``DataType``
     - :obj:`tvm_tirx.runtime.DataType`
   * - ``DataTypeCode``
     - :obj:`tvm_tirx.runtime.DataTypeCode`
   * - ``IRModule``
     - :obj:`tvm_tirx.ir.IRModule`
   * - ``Object``
     - :obj:`tvm_tirx.runtime.Object`

Functions
---------

.. autosummary::
   :toctree: generated/

   get_global_func
   hexagon
   register_global_func
   register_object

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``cpu``
     - :obj:`tvm_tirx.runtime.cpu`
   * - ``cuda``
     - :obj:`tvm_tirx.runtime.cuda`
   * - ``device``
     - :obj:`tvm_tirx.runtime.device`
   * - ``device_from_target``
     - :obj:`tvm_tirx.target.device_from_target`
   * - ``ext_dev``
     - :obj:`tvm_tirx.runtime.ext_dev`
   * - ``metal``
     - :obj:`tvm_tirx.runtime.metal`
   * - ``opencl``
     - :obj:`tvm_tirx.runtime.opencl`
   * - ``rocm``
     - :obj:`tvm_tirx.runtime.rocm`
   * - ``vpi``
     - :obj:`tvm_tirx.runtime.vpi`
   * - ``vulkan``
     - :obj:`tvm_tirx.runtime.vulkan`

Namespaces
----------

.. toctree::
   :maxdepth: 2

   runtime
   ir
   ir.transform
   script
   sym
   target
   tirx
   tirx.transform
   codegen
   backend
   support
   testing

Additional namespaces
---------------------

.. toctree::
   :maxdepth: 1

   config
   libinfo
