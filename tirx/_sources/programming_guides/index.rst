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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/overview.rst

Programming Overview
====================

This page provides a high-level overview of TIRx programming: the design
philosophy behind it, the programming model, and the guides in this section.

Design Philosophy
-----------------

Kernel DSLs are most effective when they choose the right boundary between the
programmer and the machine. For mature kernels on mature hardware, that
boundary can be high-level: the compiler hides thread assignment, memory
movement, layout details, and instruction selection behind compact tensor or
tile abstractions, and this works well for established kernel patterns.

At the frontier, the same boundary is under more pressure. New instructions,
memory spaces, cooperation patterns, and kernel algorithms often appear before
a compiler has enough built-in machinery to automate them well. At the same
time, kernel programming is shifting toward agents, which work best with
direct, predictable behavior: code that maps closely to what the hardware
does, rather than to decisions a compiler makes behind the scenes.

TIRx chooses a lower and more explicit boundary. It is a generic language built
for extension, and its core primitives stay direct and close to the native
level of the hardware. The parts of a kernel that frequently require expert
control, such as pipeline structure, synchronization, role assignment, memory
placement, and backend intrinsics, are written directly in hardware-native
source code.

On top of these primitives, TIRx provides a predictable convenience layer for
the recurring tile-level structure through three constructs: execution scope,
tensor layout, and tile operation dispatch. The layer is optional, and any part
of a kernel can drop down to the native primitives.

The result is a language that can grow with the hardware. A new feature can
first be used directly as a backend intrinsic, and later become a reusable tile
operation once the pattern stabilizes. This is the core design philosophy
behind TIRx: keep the foundation small and explicit, and let developers,
agents, and the backend library extend it as new accelerator generations
arrive.

The Programming Model
---------------------

A TIRx program reads as a structured native kernel: loops, branches, tensors,
synchronization, pipeline state, backend intrinsics, and hardware roles are
written directly. Tile operations appear exactly where a repeated
hardware-level operation should become reusable and dispatchable.

Authoring Layers and IR
~~~~~~~~~~~~~~~~~~~~~~~

TIRx programs use one TVMScript namespace alias:

.. code-block:: python

   from tvm_tirx.script import tirx as T

The namespace exposes three authoring layers. They share the same TIRx
program, but enter the compiler at different levels:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Authoring layer
     - Syntax
     - Representation and lowering
   * - Core language
     - ``T.*``
     - Creates TIRx statements, expressions, tensors, and control flow.
   * - Backend operations
     - ``T.cuda.*`` and ``T.ptx.*``
     - Creates backend calls directly; these calls bypass tile operation
       dispatch.
   * - Tile operations
     - ``T.cuda.tile.*``
     - Creates a tile call that is replaced by a target-specific
       implementation during tile operation dispatch.

``tvm_tirx.script.tirx`` is the construction syntax for the TIRx IR, not a
separate IR. Tile operation dispatch and layout and scope lowering replace
high-level TIRx nodes with lower-level TIRx statements and calls:

.. code-block:: text

   T.* ───────────────────────────▶ TIRx statements and expressions ──┐
   T.cuda.* / T.ptx.* ────────────▶ backend calls ────────────────────┤
   T.cuda.tile.* ─▶ tile call ─▶ tile operation dispatch ─────────────┤
                                                                      ▼
                                        layout and scope lowering ─▶ codegen

The model rests on three constructs:

- **Execution scope** describes which threads take part in an operation and at
  what level, such as a single thread, a warp, or a whole thread block.
  Ordinary control flow selects the hardware roles that enter a region.
- **Tensor layout** describes how a logical tensor maps to physical storage:
  which memory it lives in and how its elements are spread across threads and
  per-thread storage. Tile operation dispatch reads these declarations to
  choose an implementation.
- **Tile operation dispatch** selects an implementation for each tile
  operation according to the operation, the execution scope, the operand
  layouts, and the target, then generates the loops and addressing that apply
  it across the whole tile.

Across a full kernel, orchestration stays in ordinary source code, and the
recurring hardware operations appear as tile operations.

Guides in This Section
----------------------

- :doc:`language_basics` covers defining and launching kernels.
- :doc:`control_flow` covers conditions, loops, and early thread exit.
- :doc:`types_tensors_memory` covers element types, tensors, and memory
  spaces.
- :doc:`metaprogramming` covers kernels parameterized by captured values,
  reusable code fragments, and compile-time specialization.
- :doc:`cuda_ptx_interfaces` covers calling backend intrinsics, such as
  synchronization and warp shuffles, PTX instructions, and CUDA C++ device
  functions directly.
- :doc:`data_layout` covers how tensor elements map to threads and memory.
- :doc:`tile/index` covers tile operations and the instruction guides.
- :doc:`compiling` covers compiler options and inspecting generated code.
- :doc:`iket_profiling` covers profiling kernels with IKET.
