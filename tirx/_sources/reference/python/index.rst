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

Python API
==========

.. automodule:: tvm_tirx
   :no-members:

.. currentmodule:: tvm_tirx

Namespaces
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Namespace
     - Purpose
   * - :doc:`runtime <runtime>`
     - Tensors, devices, data types, and runtime modules.
   * - :doc:`ir <ir>`
     - Shared expressions, types, statements, and IR modules.
   * - :doc:`ir.prim <ir.prim>`
     - Primitive expressions and scalar operators.
   * - :doc:`ir.transform <ir.transform>`
     - Pass infrastructure and instrumentation.
   * - :doc:`script <script>`
     - Python syntax registration for IR dialects.
   * - :doc:`script.parser <script.parser>`
     - Translate Python scripts into IR builder operations.
   * - :doc:`script.ir_builder <script.ir_builder>`
     - Shared IR builder contexts and statements.
   * - :doc:`sym <sym>`
     - Symbolic arithmetic, analysis, and simplification.
   * - :doc:`target <target>`
     - Compilation targets and device selection.
   * - :doc:`tirx <tirx>`
     - Tensor IR construction and compilation.
   * - :doc:`tirx.op <tirx.op>`
     - Tensor IR operators and intrinsic calls.
   * - :doc:`tirx.analysis <tirx.analysis>`
     - Tensor IR analysis and verification.
   * - :doc:`tirx.transform <tirx.transform>`
     - Tensor IR transformation passes.
   * - :doc:`tirx.layout <tirx.layout>`
     - Tensor layouts and logical-to-physical index mappings.
   * - :doc:`tirx.tile <tirx.tile>`
     - Tile operations, views, and dispatch.
   * - :doc:`tirx.script <tirx.script>`
     - Tensor IR syntax and JIT compilation.
   * - :doc:`codegen <codegen>`
     - Shared code generation and LLVM helpers.
   * - :doc:`backend <backend>`
     - Load and register device backends.
   * - :doc:`backend.cuda <backend.cuda>`
     - CUDA backend configuration and registration.
   * - :doc:`backend.cuda.op <backend.cuda.op>`
     - CUDA intrinsic builders.
   * - :doc:`backend.cuda.script <backend.cuda.script>`
     - CUDA kernel authoring namespace.
   * - :doc:`backend.cuda.layout <backend.cuda.layout>`
     - CUDA shared-memory and tensor-memory layouts.
   * - :doc:`backend.cuda.lang <backend.cuda.lang>`
     - CUDA pipelines, barriers, storage pools, and scheduling.
   * - :doc:`backend.cuda.transform <backend.cuda.transform>`
     - CUDA-specific transformation passes.
   * - :doc:`backend.rocm <backend.rocm>`
     - ROCm backend registration.
   * - :doc:`backend.metal <backend.metal>`
     - Metal backend and script namespaces.
   * - :doc:`backend.metal.op <backend.metal.op>`
     - Metal intrinsic builders.
   * - :doc:`backend.opencl <backend.opencl>`
     - OpenCL backend and script namespaces.
   * - :doc:`backend.adreno <backend.adreno>`
     - Adreno backend registration.
   * - :doc:`backend.hexagon <backend.hexagon>`
     - Hexagon backend and linker integration.
   * - :doc:`backend.vulkan <backend.vulkan>`
     - Vulkan backend registration.
   * - :doc:`backend.webgpu <backend.webgpu>`
     - WebGPU backend registration.
   * - :doc:`backend.trn <backend.trn>`
     - Trainium backend and script namespaces.
   * - :doc:`backend.trn.op <backend.trn.op>`
     - Trainium NKI intrinsic wrappers.
   * - :doc:`backend.trn.transform <backend.trn.transform>`
     - Trainium-specific lowering passes.
   * - :doc:`support <support>`
     - Build tools and system utilities.
   * - :doc:`testing <testing>`
     - Test fixtures and numerical checks.

.. toctree::
   :hidden:
   :maxdepth: 1
   :titlesonly:

   runtime <runtime>
   ir <ir>
   ir.prim <ir.prim>
   ir.transform <ir.transform>
   script <script>
   script.parser <script.parser>
   script.ir_builder <script.ir_builder>
   sym <sym>
   target <target>
   tirx <tirx>
   tirx.op <tirx.op>
   tirx.analysis <tirx.analysis>
   tirx.transform <tirx.transform>
   tirx.layout <tirx.layout>
   tirx.tile <tirx.tile>
   tirx.script <tirx.script>
   codegen <codegen>
   backend <backend>
   backend.cuda <backend.cuda>
   backend.cuda.op <backend.cuda.op>
   backend.cuda.script <backend.cuda.script>
   backend.cuda.layout <backend.cuda.layout>
   backend.cuda.lang <backend.cuda.lang>
   backend.cuda.transform <backend.cuda.transform>
   backend.rocm <backend.rocm>
   backend.metal <backend.metal>
   backend.metal.op <backend.metal.op>
   backend.opencl <backend.opencl>
   backend.adreno <backend.adreno>
   backend.hexagon <backend.hexagon>
   backend.vulkan <backend.vulkan>
   backend.webgpu <backend.webgpu>
   backend.trn <backend.trn>
   backend.trn.op <backend.trn.op>
   backend.trn.transform <backend.trn.transform>
   support <support>
   testing <testing>

Classes
-------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``DataTypeCode``
     - :obj:`tvm_tirx.runtime.DataTypeCode`
   * - ``IRModule``
     - :obj:`tvm_tirx.ir.IRModule`

Functions
---------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Alias
     - Canonical API
   * - ``cpu``
     - :obj:`tvm_tirx.runtime.cpu`
   * - ``cuda``
     - :obj:`tvm_tirx.runtime.cuda`
   * - ``opencl``
     - :obj:`tvm_tirx.runtime.opencl`
   * - ``vulkan``
     - :obj:`tvm_tirx.runtime.vulkan`
   * - ``metal``
     - :obj:`tvm_tirx.runtime.metal`
   * - ``vpi``
     - :obj:`tvm_tirx.runtime.vpi`
   * - ``rocm``
     - :obj:`tvm_tirx.runtime.rocm`
   * - ``ext_dev``
     - :obj:`tvm_tirx.runtime.ext_dev`
   * - ``device_from_target``
     - :obj:`tvm_tirx.target.device_from_target`
