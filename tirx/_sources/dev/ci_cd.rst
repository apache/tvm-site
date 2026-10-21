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

Reproduce CI/CD
===============

This guide explains how to reproduce CI checks and tests locally,
and how wheel builds and releases work. Workflows are defined under
`.github/workflows/ <https://github.com/apache/tvm-tirx/tree/main/.github/workflows>`__
and `ci/jenkins/ <https://github.com/apache/tvm-tirx/tree/main/ci/jenkins>`__.
To build the project from source, see :doc:`source_build`.

Linters
-------

Pre-commit
~~~~~~~~~~

The project uses `pre-commit <https://pre-commit.com/>`__ to run linters and
formatters. All hooks are defined in `.pre-commit-config.yaml
<https://github.com/apache/tvm-tirx/blob/main/.pre-commit-config.yaml>`__.
Install and register the hooks so they run automatically before each commit:

.. code-block:: bash

   uv tool install pre-commit
   pre-commit install

You can also run hooks manually:

.. code-block:: bash

   # Run all hooks on every file
   pre-commit run --all-files

   # Run only on staged files
   pre-commit run

   # Run a single hook in isolation
   pre-commit run ruff-check --all-files
   pre-commit run clang-format --all-files

The main tools are:

- **Python** -- ``ruff`` checks and formats Python code.
- **C/C++** -- ``clang-format`` formats native code.
- **YAML** -- ``yamllint`` checks syntax and style.
- **TOML** -- ``taplo`` formats configuration files.
- **Repository checks** -- ASF headers, allowed file types, whitespace, and
  generated CUDA launch files.

Hooks may fix files in place. Review the changes, stage them, and run again.
Use ``pre-commit clean`` if a stale hook environment causes problems.

C++ Tests
---------

Configure, build, and run locally:

.. code-block:: bash

   cmake . -B build_test -G Ninja -DTVM_TIRX_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug
   cmake --build build_test --config Debug --target tirx_cpptest
   ctest -V -C Debug --test-dir build_test --output-on-failure

CMake uses an installed GoogleTest package or fetches it when needed.
On Windows, run from a Visual Studio Developer Command Prompt.

Python Tests
------------

Reproduce locally with:

.. code-block:: bash

   # Install the editable package and test dependencies
   uv pip install --reinstall --verbose --group test -e .

   # Run the CPU tests
   uv run pytest -vvs tests/python -m "not gpu"

Use ``-m gpu`` for device tests with the matching backend, SDK, and hardware.
Pass a test file or directory to focus on the component you changed.

Build Python Wheels
-------------------

CI builds wheels using `cibuildwheel <https://cibuildwheel.pypa.io/>`__ on
Linux (x86_64, aarch64), Windows (AMD64), and macOS (arm64). Wheel configuration
lives in ``[tool.cibuildwheel]`` in ``pyproject.toml``. Configure the platform's
LLVM toolchain as in the `wheel build action
<https://github.com/apache/tvm-tirx/blob/main/.github/actions/build-wheel-for-publish/action.yml>`__;
Linux builds also require Docker and the LLVM prefix mounted into the build image.

To build wheels locally:

.. code-block:: bash

   uv tool install cibuildwheel
   cibuildwheel --output-dir dist

TIRx uses TVM-FFI for bindings, making it agnostic to Python versions.
The project sets ``wheel.py-api = "py3"``, so its
``py3-none-<platform>`` wheel works across supported Python 3 versions
without rebuilding TIRx for each minor version. The package requires Python
3.10 or newer (classifiers list 3.10--3.13), with compatible TVM-FFI and other
dependencies. OS, CPU architecture, and native runtime requirements still apply.

.. seealso::

   - :doc:`source_build`: Static LLVM linkage for distributing wheels.
   - :doc:`release_process`: Publishing wheels and creating release artifacts.
