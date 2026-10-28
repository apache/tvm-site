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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/native_basics/cuda/compiling.rst

Compiling and inspecting
========================

In this chapter, we will compile a CUDA function with a chosen target and
compiler options, then inspect its IR and generated code. The examples reuse
``add_one`` from the GPU program in :doc:`Quick Start </get_started/quick_start>`.

Compiling for CUDA
------------------

A ``Target`` specifies the GPU architecture and device compiler settings.
For example, we can compile ``add_one`` with fast math enabled:

.. code-block:: python

    import tvm_tirx
    from tvm_tirx.script import tirx as T
    from tvm_tirx.target import Target

    target = Target({
        "kind": "cuda",
        "arch": "sm_75",
        "backend_config": {"cuda": {"nvrtc": ["--use_fast_math"]}},
    })
    add_one_compiled = tvm_tirx.compile(add_one, target=target)

``arch`` selects the GPU architecture; ``sm_75`` denotes compute capability
7.5. Set it to the architecture of the GPU that will run the program. On a
machine with CUDA support and an available GPU, ``Target("cuda")`` can detect
the architecture instead. An offline build needs an explicit architecture.

TIRx uses NVRTC by default. The ``nvrtc`` list passes individual compiler
arguments, such as ``--use_fast_math``, to that compiler. These target settings
apply to the function's device code. The returned executable accepts the
input and output tensors just as in Quick Start. The
:class:`BackendConfig API <tvm_tirx.backend.cuda.BackendConfig>` describes
other compiler settings.

Overriding options for a kernel
-------------------------------

A kernel can override the target's settings through its ``device_entry``.
For example, to add source line information to the add-one kernel, replace
its ``T.device_entry`` statement with:

.. code-block:: python

    T.device_entry(
        launch=T.cuda.LaunchConfig(grid=((n + 255) // 256,), block=(256,)),
        backend_config={"cuda": {
            "nvrtc": ["--use_fast_math", "--generate-line-info"],
        }},
    )

After redefining ``add_one`` with this entry, compile it again:

.. code-block:: python

    add_one_compiled = tvm_tirx.compile(add_one, target=target)

Device-entry settings take precedence over target settings. Compiler argument
lists replace the inherited list in full, so this example includes
``--use_fast_math`` again alongside the new flag. ``launch`` controls the
thread configuration; ``backend_config`` controls compilation.

Inspecting IR and CUDA
----------------------

The function and compiled executable expose different stages of the program.
``add_one.script()`` returns the IR as TVMScript, which lets us check the
parameter types, index expression, and bounds check:

.. code-block:: python

    print(add_one.script())

``add_one.show()`` displays the same representation directly. For printing
options, see :doc:`/ir/printing_and_parsing`.

The compiled executable contains a host launcher and an imported device
module. For this single-kernel program, the first import contains the CUDA
code:

.. code-block:: python

    device_module = add_one_compiled.mod.imports[0]
    print(device_module.inspect_source("cuda"))

This output shows the lowered kernel, including its data pointers,
``blockIdx.x`` and ``threadIdx.x`` accesses, and the condition guarding the
output store. It is useful for checking how the authored computation was
translated to CUDA.

To check which compiler settings were applied, inspect the same module's
resolved configuration:

.. code-block:: python

    print(device_module.inspect_source("backend_config"))

For a build using the override above, the ``nvrtc`` list includes both
``--use_fast_math`` and ``--generate-line-info``.
