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

tvm_tirx.backend.cuda.iket
==========================

.. automodule:: tvm_tirx.backend.cuda.iket
   :no-members:

.. currentmodule:: tvm_tirx.backend.cuda.iket

Classes
-------

.. autosummary::
   :toctree: generated/

   IketProfileError
   IketProfileResult
   IketProfiler

Functions
---------

.. autosummary::
   :toctree: generated/

   validate_official_environment
   profile
   run

Device annotations
------------------

``IketProfiler`` provides checkpoints and two range forms inside TVMScript:

.. list-table::
   :header-rows: 1

   * - Method
     - Meaning
   * - ``mark(name)``
     - Record a named checkpoint.
   * - ``range_push(name)`` / ``range_pop()``
     - Bracket nested ranges in stack order.
   * - ``range_start(name)`` / ``range_end(token)``
     - Connect a start and end through the returned uint32 token.

Capture and artifacts
---------------------

``run(workload, ...)`` replays the active Python script or module under
the official IKET tracker and capture processes. Put GPU initialization,
compilation, and launches inside the callable, and invoke ``run`` from
the script's ``__main__`` block. For a separate driver, ``profile`` takes
a replayable command such as ``[sys.executable, "profile_workload.py"]``;
that command compiles and runs its instrumented kernels without starting
another capture.

``output_dir`` selects the published artifact directory. The ``postprocess``
option accepts ``"json"``, ``"perfetto"``, ``"html"``, ``"all"``, or
``"none"``. Use ``clobber=True`` to replace an existing output directory.
The function signatures list timeout and capture-limit options.

``IketProfileResult.trace_json``, ``perfetto``, and ``html`` return the
single artifact of each requested format. For captures producing multiple
artifacts, use ``json_traces``, ``perfetto_traces``, and ``html_reports``
instead. The :doc:`IKET Profiling guide </cuda/iket_profiling>` contains a
complete annotation and capture example.
