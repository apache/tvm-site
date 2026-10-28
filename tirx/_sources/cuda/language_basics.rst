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

.. Adapted from Apache TVM: https://tvm.apache.org/docs/_sources/tirx/native_basics.rst

CUDA Language Basics
====================

In this chapter, we will introduce the CUDA interfaces for writing and running
TIRx kernels. We will explore how to express computations with explicit control
over GPU threads and memory, then compile the kernels and inspect their IR and
generated CUDA code.

* :doc:`language/functions` covers function parameters, symbolic shapes,
  launch configuration, control flow, specialization, and reusable fragments.
* :doc:`language/tensors` covers element types, vector accesses, shared and
  per-thread storage, mutable scalars, and tensor views.
* :doc:`language/threads_sync` covers three ways to use CUDA functionality:
  backend intrinsics, PTX instructions, and custom CUDA C++ device functions.
* :doc:`language/compiling` explains compiler configuration and how to
  inspect the IR and generated CUDA.

.. toctree::
   :hidden:
   :maxdepth: 1

   language/functions
   language/tensors
   language/threads_sync
   language/compiling
