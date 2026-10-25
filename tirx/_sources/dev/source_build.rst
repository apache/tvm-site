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
``uv sync`` creates the project ``.venv`` and installs the package in editable mode.

.. admonition:: Prerequisite
   :class: tip

   - Python: 3.10 or newer.
   - Compiler: a C++20-capable GCC, Clang, or MSVC toolchain.
   - Build tools: Git and uv.

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
the C++ libraries and supplies CMake/Ninja when needed in the isolated build:

.. code-block:: bash

   uv sync --no-default-groups

The editable install reflects Python-only changes immediately without rebuilding.

**CMake flags** can be exported through ``CMAKE_ARGS`` before running the
install command. For example, to switch off the CUDA runtime (which is
automatically switched on when available):

.. code-block:: bash

   export CMAKE_ARGS="-DTVM_TIRX_USE_CUDA=OFF"

Available CMake options (see `CMakeLists.txt
<https://github.com/apache/tvm-tirx/blob/main/CMakeLists.txt>`__) include:

- ``TVM_TIRX_USE_LLVM`` -- Prefer the build Python's static LLVM SDK, or select an
  explicit ``LLVM_DIR`` / ``llvm-config`` command (default: ``ON``). ``OFF`` disables LLVM.
- ``TVM_TIRX_USE_CUDA`` -- Enable the CUDA runtime (default: ``AUTO``).
- ``TVM_TIRX_USE_METAL`` -- Enable the Metal runtime on macOS (default: ``OFF``).
- ``TVM_TIRX_USE_Z3`` -- Enable the Z3 solver (default: ``AUTO``).
- ``TVM_TIRX_USE_RUST`` -- Build the operators written in Rust (``rust/``) with cargo
  (default: ``AUTO``, which needs a cargo as new as the crate's ``rust-version`` and the crate's
  dependencies, from crates.io or cargo's cache).
- ``TVM_TIRX_BUILD_TESTS`` -- Build C++ tests (default: ``OFF``).

Verify the installation:

.. code-block:: bash

   uv run --no-sync python -c "import tvm_tirx; print(tvm_tirx.__version__)"
   uv run --no-sync tvm-tirx-config -h

.. tip::

   Use ``tvm-tirx-config`` to query include and link flags for external C++ projects:

   .. code-block:: bash

      uv run --no-sync tvm-tirx-config --includedir
      uv run --no-sync tvm-tirx-config --libfiles

   On Linux/macOS, query linker flags for the compiler or runtime libraries:

   .. code-block:: bash

      uv run --no-sync tvm-tirx-config --libs compiler      # Compiler and its dependencies
      uv run --no-sync tvm-tirx-config --libs runtime       # Core runtime
      uv run --no-sync tvm-tirx-config --libs runtime cuda  # Core and CUDA runtimes (CUDA build)
      uv run --no-sync tvm-tirx-config --ldflags            # Library search paths

Build with CMake
----------------

Install only the build tools and SDKs, without installing the Python package,
then configure and build:

.. code-block:: bash

   uv sync --only-group build
   uv run --no-sync cmake -S . -B build_cpp -G Ninja -DCMAKE_BUILD_TYPE=Release
   uv run --no-sync cmake --build build_cpp --parallel

Build Wheels with Customized LLVM
---------------------------------

By default, builds use the pinned prebuilt LLVM/Z3 static SDK packages from
PyPI specified in ``pyproject.toml``; no manual SDK setup is needed.

An explicit ``LLVM_DIR`` or ``TVM_TIRX_USE_LLVM=/path/to/llvm-config`` selects
an external LLVM installation. Supply its development libraries and transitive
dependencies before building.

.. note::

   When building wheels for release, we recommend linking LLVM statically into
   the wheel to avoid conflicts with LLVM libraries used by other packages.
   Pass ``--link-static`` to ``llvm-config`` to enable this.

.. code-block:: bash

   CMAKE_ARGS="-DTVM_TIRX_USE_LLVM='/path/to/llvm-config --link-static'" uv build --wheel

For macOS wheels using the default SDK, set ``MACOSX_DEPLOYMENT_TARGET=14.0``.
See :doc:`ci_cd` for the publication toolchain configuration.

Troubleshooting
---------------

- **Rebuilding after C++ changes**: re-run ``uv sync --no-default-groups --reinstall-package apache-tvm-tirx``.
  Editable installs automatically reflect only Python changes.
- **Submodules missing**: run ``git submodule update --init --recursive`` from
  the repository root.

.. seealso::

   - :doc:`ci_cd`: Reproduce linters, unit tests, and wheel builds locally.
