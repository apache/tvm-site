
.. _program_listing_file_tvm_runtime_tensor.h:

Program Listing for File tensor.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_runtime_tensor.h>` (``tvm/runtime/tensor.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   /*
    * Licensed to the Apache Software Foundation (ASF) under one
    * or more contributor license agreements.  See the NOTICE file
    * distributed with this work for additional information
    * regarding copyright ownership.  The ASF licenses this file
    * to you under the Apache License, Version 2.0 (the
    * "License"); you may not use this file except in compliance
    * with the License.  You may obtain a copy of the License at
    *
    *   http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing,
    * software distributed under the License is distributed on an
    * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    * KIND, either express or implied.  See the License for the
    * specific language governing permissions and limitations
    * under the License.
    */
   
   #ifndef TVM_RUNTIME_TENSOR_H_
   #define TVM_RUNTIME_TENSOR_H_
   
   #include <tvm/ffi/cast.h>
   #include <tvm/ffi/container/shape.h>
   #include <tvm/ffi/container/tensor.h>
   #include <tvm/ffi/dtype.h>
   #include <tvm/ffi/optional.h>
   #include <tvm/ffi/string.h>
   #include <tvm/runtime/base.h>
   #include <tvm/runtime/device_api.h>
   #include <tvm/support/io.h>
   #include <tvm/support/serializer.h>
   
   #include <algorithm>
   #include <atomic>
   #include <functional>
   #include <limits>
   #include <mutex>
   #include <unordered_map>
   #include <utility>
   #include <vector>
   
   namespace tvm {
   namespace runtime {
   
   TVM_RUNTIME_DLL ffi::Tensor Empty(ffi::Shape shape, DLDataType dtype, Device dev,
                                     ffi::Optional<MemoryScope> mem_scope = std::nullopt);
   
   TVM_RUNTIME_DLL void CopyFromTo(ffi::TensorView source, ffi::TensorView destination,
                                   TVMStreamHandle stream = nullptr);
   
   TVM_RUNTIME_DLL ffi::Tensor CopyTo(ffi::TensorView source, Device dev,
                                      ffi::Optional<MemoryScope> mem_scope = std::nullopt);
   
   TVM_RUNTIME_DLL ffi::Tensor CreateView(ffi::Tensor source, ffi::Shape shape, DLDataType dtype,
                                          uint64_t relative_byte_offset = 0);
   
   TVM_RUNTIME_DLL void CopyFromBytes(ffi::TensorView destination, const void* source_bytes,
                                      size_t nbytes, TVMStreamHandle stream = nullptr);
   
   TVM_RUNTIME_DLL void CopyToBytes(ffi::TensorView source, void* destination_bytes, size_t nbytes,
                                    TVMStreamHandle stream = nullptr);
   
   constexpr uint64_t kTVMTensorMagic = 0xDD5E40F096B4A13F;
   
   TVM_RUNTIME_DLL void SaveTensor(ffi::TensorView source, support::Stream* strm);
   
   TVM_RUNTIME_DLL ffi::Tensor LoadTensor(support::Stream* strm);
   
   class TensorMemoryPool;
   
   TVM_RUNTIME_DLL ffi::ObjectPtr<TensorMemoryPool> CreateTensorMemoryPool(DLDevice device);
   
   class TensorMemoryPool : public ffi::Object {
    public:
     explicit TensorMemoryPool(size_t page_size = 4096) : page_size_(page_size), used_memory_(0) {
       TVM_FFI_CHECK_GT(page_size_, 0, ValueError) << "Pool page size must be positive";
       TVM_FFI_CHECK_LE(page_size_, static_cast<size_t>(std::numeric_limits<int64_t>::max()),
                        ValueError);
     }
   
     virtual ~TensorMemoryPool() { Clear(); }
   
     virtual ffi::Tensor Empty(ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                               ffi::Optional<MemoryScope> mem_scope = std::nullopt) {
       std::lock_guard<std::recursive_mutex> lock(mu_);
       if (mem_scope.has_value() && *mem_scope != MemoryScope::Global()) {
         return DeviceAPI::Get(device)->Empty(shape, dtype, device, mem_scope);
       }
       size_t nbytes = ffi::GetDataSize(shape.Product(), dtype);
       TVM_FFI_CHECK_LE(std::max<size_t>(nbytes, 1),
                        static_cast<size_t>(std::numeric_limits<int64_t>::max()) - (page_size_ - 1),
                        ValueError)
           << "Rounded tensor allocation size overflows";
       size_t size = ((std::max<size_t>(nbytes, 1) + page_size_ - 1) / page_size_) * page_size_;
       size_t alignment = kAllocAlignment;
       size_t element_bytes = (static_cast<size_t>(dtype.bits) * dtype.lanes + 7) / 8;
       while (alignment < element_bytes) alignment *= 2;
   
       ffi::Tensor backing;
       auto it = memory_pool_.find(size);
       if (it != memory_pool_.end()) {
         auto& pool = it->second;
         bool addressable = DeviceAPI::Get(device)->SupportsDevicePtrArithmeticOnHost();
         auto candidate = std::find_if(pool.begin(), pool.end(), [&](const ffi::Tensor& tensor) {
           return tensor.device().device_type == device.device_type &&
                  tensor.device().device_id == device.device_id &&
                  (!addressable || reinterpret_cast<uintptr_t>(tensor.data_ptr()) % alignment == 0);
         });
         if (candidate != pool.end()) {
           backing = std::move(*candidate);
           pool.erase(candidate);
         }
       }
       if (!backing.defined()) {
         // Collective backends cannot retry by unilaterally releasing cached buffers.
         // Any allocation-pressure retry belongs to the non-collective adapter.
         backing = AllocBacking(size, alignment, device);
         used_memory_.fetch_add(size, std::memory_order_relaxed);
       }
       Lease lease{ffi::GetObjectPtr<TensorMemoryPool>(this), std::move(backing)};
       try {
         return ffi::Tensor::FromNDAlloc(lease, shape, dtype, device);
       } catch (...) {
         // Tensor metadata allocation may fail before the owning tensor exists.
         Recycle(std::move(lease.backing));
         throw;
       }
     }
   
     virtual ffi::Tensor CreateView(ffi::Tensor backing, ffi::Shape shape, DLDataType dtype,
                                    uint64_t byte_offset = 0,
                                    ffi::Optional<MemoryScope> mem_scope = std::nullopt) {
       auto view = runtime::CreateView(std::move(backing), std::move(shape), dtype, byte_offset);
       if (view.device().device_type == kDLHexagon &&
           (!mem_scope.has_value() || *mem_scope == MemoryScope::Global())) {
         // Hexagon kernels receive the adjusted pointer rather than a byte offset.
         auto* tensor = const_cast<DLTensor*>(view.GetDLTensorPtr());
         if (tensor->byte_offset != 0) {
           tensor->data = static_cast<char*>(tensor->data) + tensor->byte_offset;
         }
         tensor->byte_offset = 0;
       }
       return view;
     }
   
     void Clear() {
       std::lock_guard<std::recursive_mutex> lock(mu_);
       for (const auto& [size, pool] : memory_pool_) {
         used_memory_.fetch_sub(size * pool.size(), std::memory_order_relaxed);
       }
       memory_pool_.clear();
     }
   
     size_t UsedMemory() const { return used_memory_.load(std::memory_order_relaxed); }
   
     static constexpr bool _type_mutable = true;
     TVM_FFI_DECLARE_OBJECT_INFO("runtime.TensorMemoryPool", TensorMemoryPool, ffi::Object);
   
    protected:
     virtual ffi::Tensor AllocBacking(size_t nbytes, size_t alignment, DLDevice device) {
       auto allocate = [&]() {
         auto* api = DeviceAPI::Get(device);
         DLDataType bytes{kDLUInt, 8, 1};
         if (alignment <= kAllocAlignment) {
           return api->Empty({static_cast<int64_t>(nbytes)}, bytes, device);
         }
         // Preserve the backend's dtype alignment contract for wide vector elements.
         TVM_FFI_CHECK_LE(alignment, std::numeric_limits<uint16_t>::max(), ValueError);
         auto owner = api->Empty({static_cast<int64_t>((nbytes + alignment - 1) / alignment)},
                                 DLDataType{kDLUInt, 8, static_cast<uint16_t>(alignment)}, device);
         return runtime::CreateView(std::move(owner), {static_cast<int64_t>(nbytes)}, bytes);
       };
       try {
         return allocate();
       } catch (const ffi::Error&) {
         // This default allocator is non-collective; release cached memory and retry.
         Clear();
         return allocate();
       }
     }
   
    private:
     struct Lease {
       ffi::ObjectPtr<TensorMemoryPool> pool;
       ffi::Tensor backing;
   
       void AllocData(DLTensor* tensor) { tensor->data = backing.data_ptr(); }
       void FreeData(DLTensor*) { pool->Recycle(std::move(backing)); }
     };
   
     void Recycle(ffi::Tensor backing) {
       std::lock_guard<std::recursive_mutex> lock(mu_);
       size_t size = ffi::GetDataSize(backing);
       try {
         memory_pool_[size].push_back(std::move(backing));
       } catch (const std::bad_alloc&) {
         // A cache insertion failure must not escape tensor destruction.
         used_memory_.fetch_sub(size, std::memory_order_relaxed);
       }
     }
   
     size_t page_size_;
     std::atomic<size_t> used_memory_;
     std::unordered_map<size_t, std::vector<ffi::Tensor>> memory_pool_;
     std::recursive_mutex mu_;
   };
   
   }  // namespace runtime
   }  // namespace tvm
   
   namespace std {
   template <>
   struct hash<tvm::Device> {
     std::size_t operator()(const tvm::Device& dev) const {
       return ((dev.device_id << 8) | dev.device_type);
     }
   };
   
   template <>
   struct equal_to<tvm::Device> {
     bool operator()(const tvm::Device& lhs, const tvm::Device& rhs) const {
       return (lhs.device_type == rhs.device_type && lhs.device_id == rhs.device_id);
     }
   };
   }  // namespace std
   
   #endif  // TVM_RUNTIME_TENSOR_H_
