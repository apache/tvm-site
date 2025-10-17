
.. _program_listing_file_tvm_ffi_container_tensor.h:

Program Listing for File tensor.h
=================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_container_tensor.h>` (``tvm/ffi/container/tensor.h``)

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
   
   #ifndef TVM_FFI_CONTAINER_TENSOR_H_
   #define TVM_FFI_CONTAINER_TENSOR_H_
   
   #include <tvm/ffi/container/shape.h>
   #include <tvm/ffi/dtype.h>
   #include <tvm/ffi/error.h>
   #include <tvm/ffi/type_traits.h>
   
   #include <atomic>
   #include <memory>
   #include <string>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   class Tensor;
   
   inline bool IsDirectAddressDevice(const DLDevice& device) {
     return device.device_type <= kDLCUDAHost || device.device_type == kDLCUDAManaged ||
            device.device_type == kDLROCM || device.device_type == kDLROCMHost;
   }
   
   inline bool IsContiguous(const DLTensor& arr) {
     if (arr.strides == nullptr) return true;
     int64_t expected_stride = 1;
     for (int32_t i = arr.ndim; i != 0; --i) {
       int32_t k = i - 1;
       if (arr.shape[k] == 1) {
         // Skip stride check if shape[k] is 1, where the dimension is contiguous
         // regardless of the value of stride.
         //
         // For example, PyTorch will normalize stride to 1 if shape is 1 when exporting
         // to DLPack.
         // More context: https://github.com/pytorch/pytorch/pull/83158
         continue;
       }
       if (arr.strides[k] != expected_stride) return false;
       expected_stride *= arr.shape[k];
     }
     return true;
   }
   
   inline bool IsAligned(const DLTensor& arr, size_t alignment) {
     if (IsDirectAddressDevice(arr.device)) {
       return (reinterpret_cast<size_t>(static_cast<char*>(arr.data) + arr.byte_offset) % alignment ==
               0);
     } else {
       return arr.byte_offset % alignment == 0;
     }
   }
   
   inline size_t GetDataSize(size_t numel, DLDataType dtype) {
     // compatible handling sub-byte uint1(bool), which usually stored as uint8_t
     // TODO(tqchen): revisit and switch to kDLBool
     if (dtype.code == kDLUInt && dtype.bits == 1 && dtype.lanes == 1) {
       return numel;
     }
     // for other sub-byte types, packing is preferred
     return (numel * dtype.bits * dtype.lanes + 7) / 8;
   }
   
   inline size_t GetDataSize(const DLTensor& arr) {
     size_t size = 1;
     for (int i = 0; i < arr.ndim; ++i) {
       size *= static_cast<size_t>(arr.shape[i]);
     }
     return GetDataSize(size, arr.dtype);
   }
   
   class TensorObj : public Object, public DLTensor {
    public:
     static constexpr const uint32_t _type_index = TypeIndex::kTVMFFITensor;
     TVM_FFI_DECLARE_OBJECT_INFO_STATIC(StaticTypeKey::kTVMFFITensor, TensorObj, Object);
   
     DLManagedTensor* ToDLPack() const {
       TensorObj* self = const_cast<TensorObj*>(this);
       DLManagedTensor* ret = new DLManagedTensor();
       ret->dl_tensor = *static_cast<DLTensor*>(self);
       ret->manager_ctx = self;
       ret->deleter = DLManagedTensorDeleter<DLManagedTensor>;
       details::ObjectUnsafe::IncRefObjectHandle(self);
       return ret;
     }
   
     DLManagedTensorVersioned* ToDLPackVersioned() const {
       TensorObj* self = const_cast<TensorObj*>(this);
       DLManagedTensorVersioned* ret = new DLManagedTensorVersioned();
       ret->version.major = DLPACK_MAJOR_VERSION;
       ret->version.minor = DLPACK_MINOR_VERSION;
       ret->dl_tensor = *static_cast<DLTensor*>(self);
       ret->manager_ctx = self;
       ret->deleter = DLManagedTensorDeleter<DLManagedTensorVersioned>;
       details::ObjectUnsafe::IncRefObjectHandle(self);
       return ret;
     }
   
    protected:
     template <typename TDLManagedTensor>
     static void DLManagedTensorDeleter(TDLManagedTensor* tensor) {
       TensorObj* obj = static_cast<TensorObj*>(tensor->manager_ctx);
       details::ObjectUnsafe::DecRefObjectHandle(obj);
       delete tensor;
     }
   
     friend class Tensor;
   };
   
   namespace details {
   template <typename TNDAlloc>
   class TensorObjFromNDAlloc : public TensorObj {
    public:
     using Self = TensorObjFromNDAlloc<TNDAlloc>;
   
     template <typename... ExtraArgs>
     TensorObjFromNDAlloc(TNDAlloc alloc, ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                          ExtraArgs&&... extra_args)
         : alloc_(alloc) {
       this->device = device;
       this->ndim = static_cast<int>(shape.size());
       this->dtype = dtype;
       this->byte_offset = 0;
       // inplace alloc shape and strides after data structure
       this->shape = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(this) + sizeof(Self));
       this->strides = this->shape + shape.size();
       std::copy(shape.begin(), shape.end(), this->shape);
       details::FillStridesFromShape(shape, this->strides);
       // call allocator to alloc data
       alloc_.AllocData(static_cast<DLTensor*>(this), std::forward<ExtraArgs>(extra_args)...);
     }
   
     ~TensorObjFromNDAlloc() { alloc_.FreeData(static_cast<DLTensor*>(this)); }
   
    private:
     TNDAlloc alloc_;
   };
   
   template <typename TDLPackManagedTensor>
   class TensorObjFromDLPack : public TensorObj {
    public:
     using Self = TensorObjFromDLPack<TDLPackManagedTensor>;
   
     explicit TensorObjFromDLPack(TDLPackManagedTensor* tensor, bool extra_strides_at_tail)
         : tensor_(tensor) {
       *static_cast<DLTensor*>(this) = tensor_->dl_tensor;
       if (extra_strides_at_tail) {
         this->strides = reinterpret_cast<int64_t*>(reinterpret_cast<char*>(this) + sizeof(Self));
         details::FillStridesFromShape(ShapeView(tensor_->dl_tensor.shape, tensor_->dl_tensor.ndim),
                                       this->strides);
       }
     }
   
     ~TensorObjFromDLPack() {
       // run DLPack deleter if needed.
       if (tensor_->deleter != nullptr) {
         (*tensor_->deleter)(tensor_);
       }
     }
   
    private:
     TDLPackManagedTensor* tensor_;
   };
   }  // namespace details
   
   class Tensor : public ObjectRef {
    public:
     Tensor() = default;
     explicit Tensor(::tvm::ffi::ObjectPtr<TensorObj> n) : ObjectRef(std::move(n)) {}
     explicit Tensor(::tvm::ffi::UnsafeInit tag) : ObjectRef(tag) {}
     TVM_FFI_DEFINE_DEFAULT_COPY_MOVE_AND_ASSIGN(Tensor)
   
     void* data_ptr() const { return get()->data; }
   
     DLDevice device() const { return get()->device; }
   
     int32_t ndim() const { return get()->ndim; }
   
     DLDataType dtype() const { return get()->dtype; }
   
     ShapeView shape() const {
       const TensorObj* obj = get();
       return tvm::ffi::ShapeView(obj->shape, obj->ndim);
     }
   
     ShapeView strides() const {
       const TensorObj* obj = get();
       TVM_FFI_ICHECK(obj->strides != nullptr || obj->ndim == 0);
       return ShapeView(obj->strides, obj->ndim);
     }
   
     int64_t size(size_t idx) const { return get()->shape[idx]; }
   
     int64_t stride(size_t idx) const { return get()->strides[idx]; }
   
     int64_t numel() const { return this->shape().Product(); }
     uint64_t byte_offset() const { return get()->byte_offset; }
     bool IsContiguous() const { return tvm::ffi::IsContiguous(*get()); }
     bool IsAligned(size_t alignment) const { return tvm::ffi::IsAligned(*get(), alignment); }
     template <typename TNDAlloc, typename... ExtraArgs>
     static Tensor FromNDAlloc(TNDAlloc alloc, ffi::ShapeView shape, DLDataType dtype, DLDevice device,
                               ExtraArgs&&... extra_args) {
       // inplace alloc shape and strides after data structure (as a result why multiply 2)
       size_t num_extra_i64_at_tail = shape.size() * 2;
       return Tensor(make_inplace_array_object<details::TensorObjFromNDAlloc<TNDAlloc>, int64_t>(
           num_extra_i64_at_tail, alloc, shape, dtype, device,
           std::forward<ExtraArgs>(extra_args)...));
     }
     static Tensor FromEnvAlloc(int (*env_alloc)(DLTensor*, TVMFFIObjectHandle*), ffi::ShapeView shape,
                                DLDataType dtype, DLDevice device) {
       TVMFFIObjectHandle out;
       DLTensor prototype{};
       prototype.device = device;
       prototype.dtype = dtype;
       prototype.shape = const_cast<int64_t*>(shape.data());
       prototype.ndim = static_cast<int>(shape.size());
       TVM_FFI_CHECK_SAFE_CALL(env_alloc(&prototype, &out));
       return Tensor(
           details::ObjectUnsafe::ObjectPtrFromOwned<TensorObj>(static_cast<TVMFFIObject*>(out)));
     }
     static Tensor FromDLPack(DLManagedTensor* tensor, size_t require_alignment = 0,
                              bool require_contiguous = false) {
       if (require_alignment != 0 && !ffi::IsAligned(tensor->dl_tensor, require_alignment)) {
         TVM_FFI_THROW(RuntimeError) << "FromDLPack: Data is not aligned to " << require_alignment
                                     << " bytes.";
       }
       if (require_contiguous && !ffi::IsContiguous(tensor->dl_tensor)) {
         TVM_FFI_THROW(RuntimeError) << "FromDLPack: Tensor is not contiguous.";
       }
       if (tensor->dl_tensor.strides != nullptr || tensor->dl_tensor.ndim == 0) {
         return Tensor(make_object<details::TensorObjFromDLPack<DLManagedTensor>>(
             tensor, /*extra_strides_at_tail=*/false));
       } else {
         return Tensor(
             make_inplace_array_object<details::TensorObjFromDLPack<DLManagedTensor>, int64_t>(
                 tensor->dl_tensor.ndim, tensor, /*extra_strides_at_tail=*/true));
       }
     }
   
     static Tensor FromDLPackVersioned(DLManagedTensorVersioned* tensor, size_t require_alignment = 0,
                                       bool require_contiguous = false) {
       if (require_alignment != 0 && !ffi::IsAligned(tensor->dl_tensor, require_alignment)) {
         TVM_FFI_THROW(RuntimeError) << "FromDLPack: Data is not aligned to " << require_alignment
                                     << " bytes.";
       }
       if (require_contiguous && !ffi::IsContiguous(tensor->dl_tensor)) {
         TVM_FFI_THROW(RuntimeError) << "FromDLPack: Tensor is not contiguous.";
       }
       if (tensor->flags & DLPACK_FLAG_BITMASK_IS_SUBBYTE_TYPE_PADDED) {
         TVM_FFI_THROW(RuntimeError) << "Subbyte type padded is not yet supported";
       }
       if (tensor->dl_tensor.strides != nullptr || tensor->dl_tensor.ndim == 0) {
         return Tensor(make_object<details::TensorObjFromDLPack<DLManagedTensorVersioned>>(
             tensor, /*extra_strides_at_tail=*/false));
       } else {
         return Tensor(
             make_inplace_array_object<details::TensorObjFromDLPack<DLManagedTensorVersioned>,
                                       int64_t>(tensor->dl_tensor.ndim, tensor,
                                                /*extra_strides_at_tail=*/true));
       }
     }
   
     DLManagedTensor* ToDLPack() const { return get_mutable()->ToDLPack(); }
   
     DLManagedTensorVersioned* ToDLPackVersioned() const { return get_mutable()->ToDLPackVersioned(); }
     const DLTensor* GetDLTensorPtr() const { return get(); }
     [[maybe_unused]] static constexpr bool _type_is_nullable = true;
     using ContainerType = TensorObj;
   
    protected:
     const TensorObj* get() const { return static_cast<const TensorObj*>(ObjectRef::get()); }
     TensorObj* get_mutable() const { return const_cast<TensorObj*>(get()); }
   };
   
   class TensorView {
    public:
     TensorView(const Tensor& tensor) {  // NOLINT(*)
       TVM_FFI_ICHECK(tensor.defined());
       tensor_ = *tensor.GetDLTensorPtr();
     }  // NOLINT(*)
     TensorView(const DLTensor* tensor) {  // NOLINT(*)
       TVM_FFI_ICHECK(tensor != nullptr);
       tensor_ = *tensor;
     }
     TensorView(const TensorView& tensor) = default;
     TensorView(TensorView&& tensor) = default;
     TensorView& operator=(const TensorView& tensor) = default;
     TensorView& operator=(TensorView&& tensor) = default;
     TensorView& operator=(const Tensor& tensor) {
       TVM_FFI_ICHECK(tensor.defined());
       tensor_ = *tensor.GetDLTensorPtr();
       return *this;
     }
   
     // explicitly delete move constructor
     TensorView(Tensor&& tensor) = delete;  // NOLINT(*)
     // delete move assignment operator from owned tensor
     TensorView& operator=(Tensor&& tensor) = delete;
     void* data_ptr() const { return tensor_.data; }
     DLDevice device() const { return tensor_.device; }
     int32_t ndim() const { return tensor_.ndim; }
     DLDataType dtype() const { return tensor_.dtype; }
     ShapeView shape() const { return ShapeView(tensor_.shape, tensor_.ndim); }
   
     int64_t numel() const { return this->shape().Product(); }
   
     ShapeView strides() const {
       TVM_FFI_ICHECK(tensor_.strides != nullptr || tensor_.ndim == 0);
       return ShapeView(tensor_.strides, tensor_.ndim);
     }
   
     int64_t size(size_t idx) const { return tensor_.shape[idx]; }
   
     int64_t stride(size_t idx) const { return tensor_.strides[idx]; }
   
     uint64_t byte_offset() const { return tensor_.byte_offset; }
   
     bool IsContiguous() const { return tvm::ffi::IsContiguous(tensor_); }
   
    private:
     DLTensor tensor_;
     template <typename, typename>
     friend struct TypeTraits;
   };
   
   inline size_t GetDataSize(const Tensor& tensor) {
     return GetDataSize(tensor.numel(), tensor.dtype());
   }
   
   inline size_t GetDataSize(const TensorView& tensor) {
     return GetDataSize(tensor.numel(), tensor.dtype());
   }
   
   // TensorView type, allow implicit casting from DLTensor*
   // NOTE: we deliberately do not support MoveToAny and MoveFromAny since it does not retain ownership
   template <>
   struct TypeTraits<TensorView> : public TypeTraitsBase {
     static constexpr bool storage_enabled = false;
     static constexpr int32_t field_static_type_index = TypeIndex::kTVMFFIDLTensorPtr;
   
     TVM_FFI_INLINE static void CopyToAnyView(const TensorView& src, TVMFFIAny* result) {
       result->type_index = TypeIndex::kTVMFFIDLTensorPtr;
       result->zero_padding = 0;
       TVM_FFI_CLEAR_PTR_PADDING_IN_FFI_ANY(result);
       result->v_ptr = const_cast<DLTensor*>(&(src.tensor_));
     }
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       return src->type_index == TypeIndex::kTVMFFIDLTensorPtr;
     }
   
     TVM_FFI_INLINE static TensorView CopyFromAnyViewAfterCheck(const TVMFFIAny* src) {
       return TensorView(static_cast<DLTensor*>(src->v_ptr));
     }
   
     TVM_FFI_INLINE static std::optional<TensorView> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index == TypeIndex::kTVMFFIDLTensorPtr) {
         return TensorView(static_cast<DLTensor*>(src->v_ptr));
       } else if (src->type_index == TypeIndex::kTVMFFITensor) {
         return TensorView(TVMFFITensorGetDLTensorPtr(src->v_obj));
       }
       return std::nullopt;
     }
   
     TVM_FFI_INLINE static std::string TypeStr() { return StaticTypeKey::kTVMFFIDLTensorPtr; }
     TVM_FFI_INLINE static std::string TypeSchema() {
       return R"({"type":")" + std::string(StaticTypeKey::kTVMFFIDLTensorPtr) + R"("})";
     }
   };
   
   }  // namespace ffi
   }  // namespace tvm
   
   #endif  // TVM_FFI_CONTAINER_TENSOR_H_
