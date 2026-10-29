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

CUDA Backend Development
========================

This guide shows how to find the source files involved in extending or
debugging the CUDA backend. We will follow ``T.cuda.thread_idx("x")`` from the
:doc:`GPU add-one example </get_started/quick_start>` to its IR operator and
CUDA output, then locate the compilation and runtime entry points.

The source tree
---------------

CUDA support has three main parts:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Directory
     - Responsibility
   * - `include/tvm/backend/cuda/ <https://github.com/apache/tvm-tirx/tree/main/include/tvm/backend/cuda>`__
     - Public C++ operator declarations, attributes, and backend contracts.
   * - `src/backend/cuda/ <https://github.com/apache/tvm-tirx/tree/main/src/backend/cuda>`__
     - Native operators, lowering passes, code generation, and runtime support.
   * - `python/tvm_tirx/backend/cuda/ <https://github.com/apache/tvm-tirx/tree/main/python/tvm_tirx/backend/cuda>`__
     - Script namespaces, PTX instruction tables, tile patterns, compiler configuration, and profiling orchestration.

Trace a CUDA operation
-----------------------

The add-one kernel uses ``T.cuda.thread_idx("x")`` to get each thread's
position within its block. This call connects three parts of the backend:

.. code-block:: text

   TVMScript: T.cuda.thread_idx("x")
       -> IR call: tirx.cuda.thread_idx, argument "x", result type int32
       -> CUDA C++: ((int)threadIdx.x)

The script namespace is installed by ``register_backend()`` in
`python/tvm_tirx/backend/cuda/__init__.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/__init__.py>`__.
`python/tvm_tirx/backend/cuda/script.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/script.py>`__
exposes the registered CUDA operators under ``T.cuda``.

The operator itself is registered in
`src/backend/cuda/op/target_builtin.cc
<https://github.com/apache/tvm-tirx/blob/main/src/backend/cuda/op/target_builtin.cc>`__.
Its definition requires an axis string and infers an int32 result.
TVMScript constructs an ``ir.Call`` referring to that operator, with
``"x"`` as its argument.

``CodeGenCUDA::Dispatch_`` in
`src/backend/cuda/codegen/codegen_cuda.cc
<https://github.com/apache/tvm-tirx/blob/main/src/backend/cuda/codegen/codegen_cuda.cc>`__
recognizes the operator and emits ``((int)threadIdx.x)``. These locations
let us trace the script interface, argument rules, and generated expression
separately when investigating an incorrect result.

PTX instruction calls use Python code-generation callbacks. The CUDA
source generator looks them up through ``tirx.intrinsics.cuda.get_codegen``,
registered in
`python/tvm_tirx/backend/cuda/codegen/registry.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/codegen/registry.py>`__.
The :doc:`PTX TableGen guide <ptx_tablegen>` follows that path for
``add.rn.f32``.

Compilation and launch
-----------------------

The shared pipeline in
`python/tvm_tirx/tirx/compilation_pipeline.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/tirx/compilation_pipeline.py>`__
lowers tensor accesses and separates device computation from host code.
``SplitHostDevice`` extracts a device region into a kernel and leaves a
launch in the host function. ``MakePackedAPI`` prepares the host entry
point that accepts the input and output tensors.

Once the CUDA source has been generated,
`python/tvm_tirx/backend/cuda/compiler.py
<https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/compiler.py>`__
compiles it with NVCC or NVRTC. The runtime implementation in
`src/backend/cuda/runtime/cuda_module.cc
<https://github.com/apache/tvm-tirx/blob/main/src/backend/cuda/runtime/cuda_module.cc>`__
loads the device module and launches kernels with their grid, block, and
shared-memory settings. The :doc:`Compiling and Inspecting guide
</programming_guides/compiling>` shows how to inspect the generated source.

Where to make a change
----------------------

The following locations provide starting points for common changes:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Task
     - Starting point
   * - Add a PTX instruction form
     - `python/tvm_tirx/backend/cuda/ptx/table.py <https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/ptx/table.py>`__ and `tests/python/tirx/codegen/test_ptx_dialect.py <https://github.com/apache/tvm-tirx/blob/main/tests/python/tirx/codegen/test_ptx_dialect.py>`__.
   * - Add a tensor instruction pattern
     - `python/tvm_tirx/backend/cuda/tile/patterns/ <https://github.com/apache/tvm-tirx/tree/main/python/tvm_tirx/backend/cuda/tile/patterns>`__ and `tests/python/tirx/tile/cuda/ <https://github.com/apache/tvm-tirx/tree/main/tests/python/tirx/tile/cuda>`__.
   * - Change source compilation
     - `python/tvm_tirx/backend/cuda/compiler.py <https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/compiler.py>`__, `python/tvm_tirx/backend/cuda/backend_config.py <https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/backend_config.py>`__, and `python/tvm_tirx/support/nvcc.py <https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/support/nvcc.py>`__.
   * - Change CUDA axis semantics
     - `src/backend/cuda/layout/axis.cc <https://github.com/apache/tvm-tirx/blob/main/src/backend/cuda/layout/axis.cc>`__ and `python/tvm_tirx/backend/cuda/axis.py <https://github.com/apache/tvm-tirx/blob/main/python/tvm_tirx/backend/cuda/axis.py>`__.
   * - Change device loading or launching
     - `src/backend/cuda/runtime/ <https://github.com/apache/tvm-tirx/tree/main/src/backend/cuda/runtime>`__ and `src/backend/cuda/launch/ <https://github.com/apache/tvm-tirx/tree/main/src/backend/cuda/launch>`__.
   * - Change profiling annotation lowering
     - ``LowerIket`` in `src/backend/cuda/transform/lower_iket.cc <https://github.com/apache/tvm-tirx/blob/main/src/backend/cuda/transform/lower_iket.cc>`__.

The :doc:`CUDA backend API </reference/python/backend.cuda>` documents
the public interfaces in these modules.

.. toctree::
   :hidden:
   :maxdepth: 1

   ptx_tablegen
