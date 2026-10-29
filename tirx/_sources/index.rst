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

Apache TVM TIRx Documentation
=============================

Welcome to the documentation for TVM TIRx. You can get started by reading the
Get Started section, then reading through the other sections.

Installation
------------

To try out TVM TIRx, install it via pip or uv:

.. code-block:: bash

   pip install --upgrade --force-reinstall apache-tvm-tirx
   # add the cuda-bindings dependency:
   # pip install --upgrade --force-reinstall "apache-tvm-tirx[cuda]"

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Get Started
   :titlesonly:

   get_started/quick_start

.. toctree::
   :maxdepth: 1
   :caption: IR Walkthrough
   :titlesonly:

   ir/overview
   ir/types_and_expressions
   ir/statements
   ir/operators_and_calls
   ir/functions_and_modules
   ir/printing_and_parsing

.. toctree::
   :maxdepth: 1
   :caption: Programming Guides
   :titlesonly:

   programming_guides/index
   programming_guides/language_basics
   programming_guides/control_flow
   programming_guides/types_tensors_memory
   programming_guides/metaprogramming
   programming_guides/cuda_ptx_interfaces
   programming_guides/data_layout
   programming_guides/tile/index
   programming_guides/compiling
   programming_guides/iket_profiling

.. toctree::
   :maxdepth: 1
   :caption: Compiler Internals
   :titlesonly:

   compiler/symbolic_integer_analysis

.. toctree::
   :maxdepth: 1
   :caption: Reference

   Python API <reference/python/index>
   C++ API <reference/cpp/index>
   Rust API <reference/rust/index>

.. toctree::
   :maxdepth: 1
   :caption: Developer Manual

   dev/source_build
   dev/doc_build
   dev/cuda/index
   dev/ci_cd
   dev/release_process
