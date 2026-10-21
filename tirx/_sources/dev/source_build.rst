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

Build from Source
=================

This guide provides instructions on how to build the TIRx package from source.

.. admonition:: Prerequisite
   :class: tip

   - Python: 3.10 or newer.
   - Compiler: a C++20-capable GCC, Clang, or MSVC toolchain.
   - Build tools: Git, CMake 3.26+, Ninja, and uv.
   - LLVM: development libraries and ``llvm-config`` from LLVM 15 or newer.

   On most platforms, you can get LLVM and the build dependencies through
   `Miniforge <https://github.com/conda-forge/miniforge>`__:

   .. code-block:: bash

      mamba create -n tirx-dev -c conda-forge python=3.12 llvmdev "cmake>=3.26" ninja git uv
      mamba activate tirx-dev
      llvm-config --version

   Use this mamba environment for all commands. Keep ``uv run`` in the same
   environment without resynchronizing the packages installed by ``uv pip``:

   .. code-block:: bash

      export UV_PROJECT_ENVIRONMENT="$CONDA_PREFIX"
      export UV_NO_SYNC=1

Build the Python Package
------------------------

Download the source via:

.. code-block:: bash

   git clone --recursive https://github.com/apache/tvm-tirx
   cd tvm-tirx

.. note::

   Always clone with ``--recursive`` to pull submodules. If you already cloned
   without it, run:

   .. code-block:: bash

      git submodule update --init --recursive

Build the Python package with scikit-build-core, which drives CMake to compile
the C++ libraries:

.. code-block:: bash

   uv pip install --force-reinstall --verbose -e .

The ``--force-reinstall`` flag forces a rebuild, and ``-e`` (editable) means
Python-only changes are reflected immediately without rebuilding.

**CMake flags** can be exported through ``CMAKE_ARGS`` before running the
install command. For example, to switch off the CUDA runtime (which is
automatically switched on when available):

.. code-block:: bash

   export CMAKE_ARGS="-DTVM_TIRX_USE_CUDA=OFF"

Available CMake options (see `CMakeLists.txt
<https://github.com/apache/tvm-tirx/blob/main/CMakeLists.txt>`__) include:

- ``TVM_TIRX_USE_LLVM`` -- Enable LLVM, or select an ``llvm-config`` command (default: ``ON``).
- ``TVM_TIRX_USE_CUDA`` -- Enable the CUDA runtime (default: ``AUTO``).
- ``TVM_TIRX_USE_METAL`` -- Enable the Metal runtime on macOS (default: ``OFF``).
- ``TVM_TIRX_USE_Z3`` -- Enable the Z3 solver (default: ``AUTO``).
- ``TVM_TIRX_BUILD_TESTS`` -- Build C++ tests (default: ``OFF``).

Verify the installation:

.. code-block:: bash

   uv run python -c "import tvm_tirx; print(tvm_tirx.__version__)"
   uv run tvm-tirx-config -h

.. tip::

   Use ``tvm-tirx-config`` to query include and link flags for external C++ projects:

   .. code-block:: bash

      tvm-tirx-config --includedir
      tvm-tirx-config --libfiles

   On Linux/macOS, query linker flags for the compiler or runtime libraries:

   .. code-block:: bash

      tvm-tirx-config --libs compiler      # Compiler and its dependencies
      tvm-tirx-config --libs runtime       # Core runtime
      tvm-tirx-config --libs runtime cuda  # Core and CUDA runtimes (CUDA build)
      tvm-tirx-config --ldflags            # Library search paths

Build Wheels with Static LLVM
-----------------------------

When shipping wheels, link LLVM statically so users do not need the same shared
LLVM library installed. Install its development dependencies and export the
static-link setting before building the wheel:

.. code-block:: bash

   mamba install -c conda-forge zlib zstd-static libxml2-devel
   export CMAKE_ARGS="-DTVM_TIRX_USE_LLVM='llvm-config --link-static' -DZLIB_USE_STATIC_LIBS=ON"

See :doc:`ci_cd` for wheel builds and the publication toolchain configuration.

Troubleshooting
---------------

- **Rebuilding after C++ changes**: re-run ``uv pip install --force-reinstall -e .``.
  Editable installs automatically reflect only Python changes.
- **Submodules missing**: run ``git submodule update --init --recursive`` from
  the repository root.

.. seealso::

   - :doc:`ci_cd`: Reproduce linters, unit tests, and wheel builds locally.
