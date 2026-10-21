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

   - Use the activated mamba environment from :doc:`source_build`.
   - Run commands from the repository root.
   - Install the documentation tools with ``uv pip install --group docs``.

Interactive Build (Auto-Reload)
-------------------------------

Rebuild and serve the documentation locally with live reload:

.. code-block:: bash

   uv run sphinx-autobuild docs docs/_build/html --ignore docs/reference/python/generated

Open ``http://127.0.0.1:8000`` after the initial build completes.

One-Off Build
-------------

Generate the HTML documentation once, with warnings treated as errors:

.. code-block:: bash

   uv run python -m sphinx -b html -W docs docs/_build/html

Open ``docs/_build/html/index.html`` to browse the result.

Cleanup
-------

Remove generated artifacts when they are no longer needed:

.. code-block:: bash

   rm -rf docs/_build/
   rm -rf docs/reference/python/generated
