.. Licensed to the Apache Software Foundation (ASF) under one
.. or more contributor license agreements.  See the NOTICE file
.. distributed with this work for additional information
.. regarding copyright ownership.  The ASF licenses this file
.. to you under the Apache License, Version 2.0 (the
.. "License"); you may not use this file except in compliance
.. with the License.  You may obtain a copy of the License at
..
..   http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing,
.. software distributed under the License is distributed on an
.. "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
.. KIND, either express or implied.  See the License for the
.. specific language governing permissions and limitations
.. under the License.

Quick Start
===========

This guide walks through shipping a minimal ``add_one`` function that computes
``y = x + 1`` in C++ and CUDA.
TVM-FFI's Open ABI and FFI make it possible to **ship one library** for multiple frameworks and languages.
We can build a single shared library that works across:

- **ML frameworks**, e.g. PyTorch, JAX, NumPy, CuPy, etc., and
- **languages**, e.g. C++, Python, Rust, etc.
- **Python ABI versions**, e.g. ship one wheel to support multiple Python versions, including free-threaded Python.

.. admonition:: Prerequisite
   :class: hint
   :name: prerequisite

   - Python: 3.9 or newer
   - Compiler: C++17-capable toolchain (GCC/Clang/MSVC)
   - Optional ML frameworks for testing: NumPy, PyTorch, JAX, CuPy
   - CUDA: Any modern version (if you want to try the CUDA part)
   - TVM-FFI installed via

     .. code-block:: bash

        pip install --reinstall --upgrade apache-tvm-ffi


Write a Simple ``add_one``
--------------------------

.. _sec-cpp-source-code:

Source Code
~~~~~~~~~~~

Suppose we implement a C++ function ``AddOne`` that performs elementwise ``y = x + 1`` for a 1-D ``float32`` vector. The source code (C++, CUDA) is:

.. tabs::

  .. group-tab:: C++

    .. code-block:: cpp
      :emphasize-lines: 8, 17

      // File: add_one_cpu.cc
      #include <tvm/ffi/container/tensor.h>
      #include <tvm/ffi/function.h>

      namespace tvm_ffi_example_cpp {

      /*! \brief Perform vector add one: y = x + 1 (1-D float32) */
      void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
        int64_t n = x.shape()[0];
        float* x_data = static_cast<float *>(x.data_ptr());
        float* y_data = static_cast<float *>(y.data_ptr());
        for (int64_t i = 0; i < n; ++i) {
          y_data[i] = x_data[i] + 1;
        }
      }

      TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cpu, tvm_ffi_example_cpp::AddOne);
      }


  .. group-tab:: CUDA

    .. code-block:: cpp
      :emphasize-lines: 15, 22, 26

      // File: main.cu
      #include <tvm/ffi/container/tensor.h>
      #include <tvm/ffi/extra/c_env_api.h>
      #include <tvm/ffi/function.h>

      namespace tvm_ffi_example_cuda {

      __global__ void AddOneKernel(float* x, float* y, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
          y[idx] = x[idx] + 1;
        }
      }

      void AddOne(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
        int64_t n = x.shape()[0];
        float* x_data = static_cast<float *>(x.data_ptr());
        float* y_data = static_cast<float *>(y.data_ptr());
        int64_t threads = 256;
        int64_t blocks = (n + threads - 1) / threads;
        cudaStream_t stream = static_cast<cudaStream_t>(
          TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
        AddOneKernel<<<blocks, threads, 0, stream>>>(x_data, y_data, n);
      }

      TVM_FFI_DLL_EXPORT_TYPED_FUNC(add_one_cuda, tvm_ffi_example_cuda::AddOne);
      }



The macro :c:macro:`TVM_FFI_DLL_EXPORT_TYPED_FUNC` exports the C++ function ``AddOne``
as a TVM FFI compatible symbol with the name ``add_one_cpu`` or ``add_one_cuda`` in the resulting library.

The class :cpp:class:`tvm::ffi::TensorView` allows zero-copy interop with tensors from different ML frameworks:

- NumPy, CuPy,
- PyTorch, JAX, or
- any array type that supports the standard `DLPack protocol <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html>`_.

Finally, :cpp:func:`TVMFFIEnvGetStream` can be used in the CUDA code to launch a kernel on the caller's stream.

.. _sec-cpp-compile-with-tvm-ffi:

Compile with TVM-FFI
~~~~~~~~~~~~~~~~~~~~

**Raw command.** We can use the following minimal commands to compile the source code:

.. tabs::

  .. group-tab:: C++

    .. code-block:: bash

      g++ -shared -O3 add_one_cpu.cc                   \
          -fPIC -fvisibility=hidden             \
          `tvm-ffi-config --cxxflags`           \
          `tvm-ffi-config --ldflags`            \
          `tvm-ffi-config --libs`               \
          -o add_one_cpu.so

  .. group-tab:: CUDA

    .. code-block:: bash

      nvcc -shared -O3 add_one_cuda.cu                  \
        --compiler-options -fPIC                \
        --compiler-options -fvisibility=hidden  \
        `tvm-ffi-config --cxxflags`             \
        `tvm-ffi-config --ldflags`              \
        `tvm-ffi-config --libs`                 \
        -o add_one_cuda.so

This step produces a shared library ``add_one_cpu.so`` and ``add_one_cuda.so`` that can be used across languages and frameworks.

**CMake.** As the preferred approach for building across platforms,
CMake relies on the CMake package ``tvm_ffi``, which can be found via ``tvm-ffi-config --cmakedir``.

.. tabs::

  .. group-tab:: C++

    .. code-block:: cmake

      find_package(Python COMPONENTS Interpreter REQUIRED)
      # Run `tvm_ffi.config --cmakedir` to find tvm-ffi targets
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE tvm_ffi_ROOT
      )
      find_package(tvm_ffi CONFIG REQUIRED)
      # Create C++ target `add_one_cpu`
      add_library(add_one_cpu SHARED add_one_cpu.cc)
      target_link_libraries(add_one_cpu PRIVATE tvm_ffi_header)
      target_link_libraries(add_one_cpu PRIVATE tvm_ffi_shared)
      # show as add_one_cpu.so
      set_target_properties(add_one_cpu PROPERTIES PREFIX "" SUFFIX ".so")

  .. group-tab:: CUDA

    .. code-block:: cmake

      find_package(Python COMPONENTS Interpreter REQUIRED)
      # Run `tvm_ffi.config --cmakedir` to find tvm-ffi targets
      execute_process(
        COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE tvm_ffi_ROOT
      )
      find_package(tvm_ffi CONFIG REQUIRED)
      # Create C++ target `add_one_cuda`
      enable_language(CUDA)
      add_library(add_one_cuda SHARED add_one_cuda.cu)
      target_link_libraries(add_one_cuda PRIVATE tvm_ffi_header)
      target_link_libraries(add_one_cuda PRIVATE tvm_ffi_shared)
      # show as add_one_cuda.so
      set_target_properties(add_one_cuda PROPERTIES PREFIX "" SUFFIX ".so")

.. hint::

   For a single-file C++/CUDA project, a convenient method :py:func:`tvm_ffi.cpp.load_inline`
   is provided to minimize boilerplate code in compilation, linking, and loading.

The resulting ``add_one_cpu.so`` and ``add_one_cuda.so`` are minimal libraries that are agnostic to:

- Python version/ABI, because it is pure C++ and not compiled or linked against Python
- C++ ABI, because TVM-FFI interacts with the artifact only via stable C APIs
- Languages, which can be C++, Rust or Python.

.. _sec-use-across-framework:

Ship Across ML Frameworks
-------------------------

TVM-FFI's Python package provides :py:func:`tvm_ffi.load_module`, which can load either
the ``add_one_cpu.so`` or ``add_one_cuda.so`` into :py:class:`tvm_ffi.Module`.

.. code-block:: python

   import tvm_ffi
   mod  : tvm_ffi.Module   = tvm_ffi.load_module("add_one_cpu.so")
   func : tvm_ffi.Function = mod.add_one_cpu

``mod.add_one_cpu`` retrieves a callable :py:class:`tvm_ffi.Function` that accepts tensors from host frameworks
directly, which can be zero-copy incorporated into all popular ML frameworks. This process is done seamlessly
without any boilerplate code and with extremely low latency.
We can then use these functions in the following ways:


.. tab-set::

    .. tab-item:: PyTorch

        .. code-block:: python

          import torch
          # cpu also works by changing the module to add_one_cpu.so and device to "cpu"
          mod = tvm_ffi.load_module("add_one_cuda.so")
          device = "cuda"
          x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32, device=device)
          y = torch.empty_like(x)
          mod.add_one_cuda(x, y)
          print(y)


    .. tab-item:: JAX

        Support via `jax-tvm-ffi <https://github.com/nvidia/jax-tvm-ffi>`_

        .. code-block:: python

          import jax
          import jax.numpy as jnp
          import jax_tvm_ffi
          import tvm_ffi

          mod = tvm_ffi.load_module("add_one_cuda.so")

          # Register the function with JAX
          jax_tvm_ffi.register_ffi_target("add_one_cuda", mod.add_one_cuda, platform="cuda")
          x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
          y = jax.ffi.ffi_call(
              "add_one_cuda",
              jax.ShapeDtypeStruct(x.shape, x.dtype),
              vmap_method="broadcast_all",
          )(x)
          print(y)

    .. tab-item:: NumPy (CPU)

        .. code-block:: python

          import numpy as np

          mod = tvm_ffi.load_module("add_one_cpu.so")
          x = np.array([1, 2, 3, 4, 5], dtype=np.float32)
          y = np.empty_like(x)
          mod.add_one_cpu(x, y)
          print(y)

    .. tab-item:: CuPy (CUDA)

        .. code-block:: python

          import cupy as cp

          mod = tvm_ffi.load_module("add_one_cuda.so")
          x = cp.array([1, 2, 3, 4, 5], dtype=cp.float32)
          y = cp.empty_like(x)
          mod.add_one_cuda(x, y)
          print(y)


Ship Across Languages
---------------------

TVM-FFI's core loading mechanism is ABI stable and works across language boundaries.
A single artifact can be loaded in every language TVM-FFI supports,
without having to recompile different artifacts targeting different ABIs or languages.


Python
~~~~~~

As shown in the :ref:`previous section<sec-use-across-framework>`, :py:func:`tvm_ffi.load_module` loads a language-
and framework-independent ``add_one_cpu.so`` or ``add_one_cuda.so`` and can be used to incorporate it into all Python
array frameworks that implement the standard `DLPack protocol <https://data-apis.org/array-api/2024.12/design_topics/data_interchange.html>`_.

C++
~~~

TVM-FFI's C++ API :cpp:func:`tvm::ffi::Module::LoadFromFile` loads ``add_one_cpu.so`` or ``add_one_cuda.so`` and
can be used directly in C/C++ with no Python dependency.

.. code-block:: cpp

  // File: run_example.cc
  #include <tvm/ffi/container/tensor.h>
  #include <tvm/ffi/extra/module.h>

  namespace ffi = tvm::ffi;
  struct CPUNDAlloc {
    void AllocData(DLTensor* tensor) { tensor->data = malloc(ffi::GetDataSize(*tensor)); }
    void FreeData(DLTensor* tensor) { free(tensor->data); }
  };

  inline ffi::Tensor Empty(ffi::Shape shape, DLDataType dtype, DLDevice device) {
    return ffi::Tensor::FromNDAlloc(CPUNDAlloc(), shape, dtype, device);
  }

  int main() {
    // load the module
    ffi::Module mod = ffi::Module::LoadFromFile("add_one_cpu.so");

    // create an Tensor, alternatively, one can directly pass in a DLTensor*
    ffi::Tensor x = Empty({5}, DLDataType({kDLFloat, 32, 1}), DLDevice({kDLCPU, 0}));
    for (int i = 0; i < 5; ++i) {
      reinterpret_cast<float*>(x.data_ptr())[i] = static_cast<float>(i);
    }

    ffi::Function add_one_cpu = mod->GetFunction("add_one_cpu").value();
    add_one_cpu(x, x);

    std::cout << "x after add_one_cpu(x, x)" << std::endl;
    for (int i = 0; i < 5; ++i) {
      std::cout << reinterpret_cast<float*>(x.data_ptr())[i] << " ";
    }
    std::cout << std::endl;
    return 0;
  }

Compile it with:

.. code-block:: bash

    g++ -fvisibility=hidden -O3               \
        run_example.cc                        \
        `tvm-ffi-config --cxxflags`           \
        `tvm-ffi-config --ldflags`            \
        `tvm-ffi-config --libs`               \
        -Wl,-rpath,`tvm-ffi-config --libdir`  \
        -o run_example

    ./run_example

.. hint::

  Sometimes it may be desirable to directly bundle the exported module into the same binary as the main program.
  In such cases, we can use :cpp:func:`tvm::ffi::Function::FromExternC` to create a
  :cpp:class:`tvm::ffi::Function` from the exported symbol, or directly use
  :cpp:func:`tvm::ffi::Function::InvokeExternC` to invoke the function. This feature can be useful
  when the exported module is generated by another DSL compiler matching the ABI.

  .. code-block:: cpp

      // File: test_bundle.cc, link with libmain.o
      #include <tvm/ffi/function.h>
      #include <tvm/ffi/container/tensor.h>

      // declare reference to the exported symbol
      extern "C" int __tvm_ffi_add_one(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);

      namespace ffi = tvm::ffi;

      int bundle_add_one(ffi::TensorView x, ffi::TensorView y) {
        void* closure_handle = nullptr;
        ffi::Function::InvokeExternC(closure_handle, __tvm_ffi_add_one, x, y);
        return 0;
      }

Rust
~~~~

TVM-FFI's Rust API ``tvm_ffi::Module::load_from_file`` loads ``add_one_cpu.so`` or ``add_one_cuda.so`` and
then retrieves a function ``add_one_cpu`` or ``add_one_cuda`` from it.
This procedure is identical to those in C++ and Python:

.. code-block:: rust

    fn run_add_one(x: &Tensor, y: &Tensor) -> Result<()> {
        let module = tvm_ffi::Module::load_from_file("add_one_cpu.so")?;
        let fn = module.get_function("add_one_cpu")?;
        let typed_fn = into_typed_fn!(fn, Fn(&Tensor, &Tensor) -> Result<()>);
        typed_fn(x, y)?;
        Ok(())
    }


.. hint::

    We can also use the Rust API to target the TVM FFI ABI. This means we can use Rust to write the function
    implementation and export to Python/C++ in the same fashion.


Troubleshooting
---------------

- ``OSError: cannot open shared object file``: Add an rpath (Linux/macOS) or ensure the DLL is on ``PATH`` (Windows). Example run-path: ``-Wl,-rpath,`tvm-ffi-config --libdir```.
- ``undefined symbol: __tvm_ffi_add_one``: Ensure you used ``TVM_FFI_DLL_EXPORT_TYPED_FUNC`` and compiled with default symbol visibility (``-fvisibility=hidden`` is fine; the macro ensures export).
- ``CUDA error: invalid device function``: Rebuild with the correct ``-arch=sm_XX`` for your GPU, or include multiple ``-gencode`` entries.
