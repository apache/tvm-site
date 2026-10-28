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

Build This Doc Site
===================

This guide walks through building the TIRx documentation locally.
Building the docs requires the Python package to be installed first; see
:doc:`source_build` for instructions.

.. admonition:: Prerequisite
   :class: hint

   - Use the project ``.venv``, as in :doc:`source_build`.
   - Run commands from the repository root.
   - Install the documentation tools and editable package with
     ``uv sync --no-default-groups --group docs``.
   - Optional: install ``Doxygen`` if you plan to generate the C++ API
     reference (see :ref:`build-cpp-docs`).

Interactive Build (Auto-Reload)
-------------------------------

Rebuild and serve the documentation locally with live reload:

.. code-block:: bash

   uv run --no-sync sphinx-autobuild docs docs/_build/html \
     --ignore docs/reference/python/generated --ignore docs/reference/cpp/generated

Open ``http://127.0.0.1:8000`` after the initial build completes.

One-Off Build
-------------

Generate the HTML documentation once, with warnings treated as errors:

.. code-block:: bash

   uv run --no-sync python -m sphinx -b html -W docs docs/_build/html

Open ``docs/_build/html/index.html`` to browse the result.

.. _build-cpp-docs:

Build with C++ Docs
-------------------

Generating the C++ reference takes longer and requires Doxygen:

.. code-block:: bash

   brew install doxygen        # macOS
   sudo apt install doxygen    # Linux

Set ``BUILD_CPP_DOCS=1`` on the desired build command to enable the extra step:

.. code-block:: bash

   # Interactive build with auto-rebuild on C++ header changes
   BUILD_CPP_DOCS=1 uv run --no-sync sphinx-autobuild docs docs/_build/html \
     --ignore docs/reference/python/generated --ignore docs/reference/cpp/generated \
     --watch include

   # One-off build
   BUILD_CPP_DOCS=1 uv run --no-sync python -m sphinx -b html -W docs docs/_build/html

.. _build-rust-docs:

Build with Rust Docs
--------------------

The Rust API reference is the ``cargo doc`` output of the ``rust/`` crate, copied
under ``reference/rust/generated`` of the HTML output. Use Rust 1.85 or newer to avoid
the `Rust 1.84 rustdoc crash <https://github.com/rust-lang/rust/issues/134715>`_
when documenting ``tvm-ffi``. The crate's build script runs
``tvm-tirx-config`` to find ``libtvm_tirx_compiler``, so
build the package first; ``uv run`` puts the helper on ``PATH``. Set
``BUILD_RUST_DOCS=1`` to enable the extra step:

.. code-block:: bash

   # Interactive build with auto-rebuild on crate changes
   BUILD_RUST_DOCS=1 uv run --no-sync sphinx-autobuild docs docs/_build/html \
     --ignore docs/reference/python/generated --ignore docs/reference/cpp/generated \
     --watch rust/src

   # One-off build
   BUILD_RUST_DOCS=1 uv run --no-sync python -m sphinx -b html -W docs docs/_build/html

Rustdoc warnings, such as a broken intra-doc link, fail the build; set
``RUSTDOCFLAGS`` to override the ``-D warnings`` default.

Cleanup
-------

Remove generated artifacts when they are no longer needed:

.. code-block:: bash

   rm -rf docs/_build/
   rm -rf docs/reference/python/generated
   rm -rf docs/reference/cpp/generated
   rm -rf rust/target/doc
