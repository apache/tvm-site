
.. _program_listing_file_tvm_ffi_container_map_base.h:

Program Listing for File map_base.h
===================================

|exhale_lsh| :ref:`Return to documentation for file <file_tvm_ffi_container_map_base.h>` (``tvm/ffi/container/map_base.h``)

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
   
   #ifndef TVM_FFI_CONTAINER_MAP_BASE_H_
   #define TVM_FFI_CONTAINER_MAP_BASE_H_
   
   #include <tvm/ffi/any.h>
   #include <tvm/ffi/container/container_details.h>
   #include <tvm/ffi/memory.h>
   #include <tvm/ffi/object.h>
   #include <tvm/ffi/optional.h>
   
   #include <algorithm>
   #include <limits>
   #include <string>
   #include <type_traits>
   #include <utility>
   
   namespace tvm {
   namespace ffi {
   
   #if TVM_FFI_DEBUG_WITH_ABI_CHANGE
   #define TVM_FFI_MAP_FAIL_IF_CHANGED() \
     TVM_FFI_ICHECK(state_marker == self->state_marker) << "Concurrent modification of the Map";
   #else
   #define TVM_FFI_MAP_FAIL_IF_CHANGED()
   #endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
   
   class MapBaseObj : public Object {
    public:
     using key_type = Any;
     using mapped_type = Any;
     using KVType = std::pair<Any, Any>;
   
     struct KVRawStorageType {
       TVMFFIAny first;
       TVMFFIAny second;
     };
   
     class iterator;
   
     static_assert(std::is_standard_layout_v<KVType>, "KVType is not standard layout");
     static_assert(sizeof(KVType) == 32, "sizeof(KVType) incorrect");
   
     size_t size() const { return size_; }
     size_t count(const key_type& key) const;
     const mapped_type& at(const key_type& key) const;
     mapped_type& at(const key_type& key);
     iterator begin() const;
     iterator end() const;
     iterator find(const key_type& key) const;
     void erase(const iterator& position);
     void erase(const key_type& key) { erase(find(key)); }
   
     void clear();
   
     class iterator {
      public:
       using iterator_category = std::forward_iterator_tag;
       using difference_type = int64_t;
       using value_type = KVType;
       using pointer = KVType*;
       using reference = KVType&;
   #if TVM_FFI_DEBUG_WITH_ABI_CHANGE
       iterator() : state_marker(0), index(0), self(nullptr) {}
   #else
       iterator() : index(0), self(nullptr) {}
   #endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
       bool operator==(const iterator& other) const {
         TVM_FFI_MAP_FAIL_IF_CHANGED()
         return index == other.index && self == other.self;
       }
       bool operator!=(const iterator& other) const { return !(*this == other); }
       pointer operator->() const;
       reference operator*() const {
         TVM_FFI_MAP_FAIL_IF_CHANGED()
         return *((*this).operator->());
       }
       iterator& operator++();
       iterator& operator--();
       iterator operator++(int) {
         TVM_FFI_MAP_FAIL_IF_CHANGED()
         iterator copy = *this;
         ++(*this);
         return copy;
       }
       iterator operator--(int) {
         TVM_FFI_MAP_FAIL_IF_CHANGED()
         iterator copy = *this;
         --(*this);
         return copy;
       }
   
      protected:
   #if TVM_FFI_DEBUG_WITH_ABI_CHANGE
       uint64_t state_marker;
       iterator(uint64_t index, const MapBaseObj* self)
           : state_marker(self->state_marker), index(index), self(self) {}
   
   #else
       iterator(uint64_t index, const MapBaseObj* self) : index(index), self(self) {}
   #endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
       uint64_t index;
       const MapBaseObj* self;
   
       friend class DenseMapBaseObj;
       friend class SmallMapBaseObj;
     };
   
    protected:
   #if TVM_FFI_DEBUG_WITH_ABI_CHANGE
     uint64_t state_marker;
   #endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
     inline void InplaceSwitchTo(ObjectPtr<Object>&& other);
     template <typename MapObjType>
     static inline ObjectPtr<Object> Empty();
     template <typename MapObjType, typename IterType>
     static inline ObjectPtr<Object> CreateFromRange(IterType first, IterType last);
     template <typename MapObjType>
     static inline ObjectPtr<Object> InsertMaybeReHash(KVType&& kv, const ObjectPtr<Object>& map);
     template <typename MapObjType>
     static inline ObjectPtr<Object> CopyFrom(MapBaseObj* from);
   
     void* data_;
     uint64_t size_;
     uint64_t slots_;
     static constexpr uint64_t kSmallTagMask = static_cast<uint64_t>(1) << 63;
     bool IsSmallMap() const { return (slots_ & kSmallTagMask) != 0ull; }
     void (*data_deleter_)(void*) = nullptr;
     // Reference class
     friend class SmallMapBaseObj;
     friend class DenseMapBaseObj;
   
     template <typename, typename, typename>
     friend class Dict;
   
     template <typename, typename>
     friend struct TypeTraits;
   };
   
   class DenseMapBaseObj : public MapBaseObj {
    private:
     static constexpr int kBlockCap = 16;
     static constexpr double kMaxLoadFactor = 0.99;
     static constexpr uint8_t kEmptySlot = static_cast<uint8_t>(0b11111111);
     static constexpr uint8_t kProtectedSlot = static_cast<uint8_t>(0b11111110);
     static constexpr int kNumJumpDists = 126;
     static constexpr uint64_t kInvalidIndex = std::numeric_limits<uint64_t>::max();
     struct ListNode;
     struct ItemType {
       KVType data;
       uint64_t prev = kInvalidIndex;
       uint64_t next = kInvalidIndex;
   
       explicit ItemType(KVType&& data) : data(std::move(data)) {}
       explicit ItemType(key_type key, mapped_type value) : data(std::move(key), std::move(value)) {}
     };
     struct Block {
       uint8_t bytes[kBlockCap + kBlockCap * sizeof(ItemType)];
     };
     static_assert(sizeof(Block) == kBlockCap * (sizeof(ItemType) + 1), "sizeof(Block) incorrect");
     static_assert(std::is_standard_layout_v<Block>, "Block is not standard layout");
   
     static void BlockDeleter(void* data) { delete[] static_cast<Block*>(data); }
   
    public:
     using MapBaseObj::iterator;
   
     uint64_t NumSlots() const { return slots_; }
   
     ~DenseMapBaseObj() { this->Reset(); }
     size_t count(const key_type& key) const { return !Search(key).IsNone(); }
     const mapped_type& at(const key_type& key) const { return At(key); }
     mapped_type& at(const key_type& key) { return At(key); }
     iterator find(const key_type& key) const {
       ListNode node = Search(key);
       return node.IsNone() ? end() : iterator(node.index, this);
     }
     void erase(const iterator& position) {
       uint64_t index = position.index;
       if (position.self != nullptr && index <= this->NumSlots()) {
         Erase(ListNode(index, this));
       }
     }
     iterator begin() const { return iterator(iter_list_head_, this); }
     iterator end() const { return iterator(kInvalidIndex, this); }
   
     void clear() {
       uint64_t n_blocks = CalcNumBlocks(this->NumSlots());
       for (uint64_t bi = 0; bi < n_blocks; ++bi) {
         uint8_t* meta_ptr = GetBlock(bi)->bytes;
         ItemType* data_ptr = reinterpret_cast<ItemType*>(GetBlock(bi)->bytes + kBlockCap);
         for (int j = 0; j < kBlockCap; ++j, ++meta_ptr, ++data_ptr) {
           uint8_t& meta = *meta_ptr;
           if (meta != kProtectedSlot && meta != kEmptySlot) {
             meta = kEmptySlot;
             data_ptr->ItemType::~ItemType();
           }
         }
       }
       size_ = 0;
       iter_list_head_ = kInvalidIndex;
       iter_list_tail_ = kInvalidIndex;
     }
   
    private:
     Block* GetBlock(size_t index) const { return static_cast<Block*>(data_) + index; }
     void IterListUnlink(ListNode node) {
       // update head and tail of iterator list if needed
       if (node.Item().prev == kInvalidIndex) {
         iter_list_head_ = node.Item().next;
       } else {
         ListNode prev_node(node.Item().prev, this);
         prev_node.Item().next = node.Item().next;
       }
       if (node.Item().next == kInvalidIndex) {
         iter_list_tail_ = node.Item().prev;
       } else {
         ListNode next_node(node.Item().next, this);
         next_node.Item().prev = node.Item().prev;
       }
     }
     void IterListPushBack(ListNode node) {
       node.Item().prev = iter_list_tail_;
       node.Item().next = kInvalidIndex;
       if (iter_list_tail_ != kInvalidIndex) {
         ListNode prev_node(iter_list_tail_, this);
         prev_node.Item().next = node.index;
       }
       if (iter_list_head_ == kInvalidIndex) {
         iter_list_head_ = node.index;
       }
       iter_list_tail_ = node.index;
     }
     void IterListReplaceNodeBy(ListNode src, ListNode dst) {
       // set link correctly on the dst
       dst.Item().prev = src.Item().prev;
       dst.Item().next = src.Item().next;
       // update prev and next of dst
       if (dst.Item().prev == kInvalidIndex) {
         iter_list_head_ = dst.index;
       } else {
         ListNode prev_node(dst.Item().prev, this);
         prev_node.Item().next = dst.index;
       }
       if (dst.Item().next == kInvalidIndex) {
         iter_list_tail_ = dst.index;
       } else {
         ListNode next_node(dst.Item().next, this);
         next_node.Item().prev = dst.index;
       }
     }
     ListNode Search(const key_type& key) const {
       if (this->size_ == 0) {
         return ListNode();
       }
       for (ListNode iter = GetListHead(AnyHash()(key)); !iter.IsNone(); iter.MoveToNext(this)) {
         if (AnyEqual()(key, iter.Key())) {
           return iter;
         }
       }
       return ListNode();
     }
     mapped_type& At(const key_type& key) const {
       ListNode iter = Search(key);
       if (iter.IsNone()) {
         TVM_FFI_THROW(KeyError) << "key is not in Map";
       }
       return iter.Val();
     }
     bool TryInsert(const key_type& key, ListNode* result) {
       if (slots_ == 0) {
         return false;
       }
       // required that `iter` to be the head of a linked list through which we can iterator
       ListNode iter = IndexFromHash(AnyHash()(key));
       // `iter` can be: 1) empty; 2) body of an irrelevant list; 3) head of the relevant list
       // Case 1: empty
       if (iter.IsEmpty()) {
         iter.NewHead(ItemType(key, Any(nullptr)));
         this->size_ += 1;
         *result = iter;
         return true;
       }
       // Case 2: body of an irrelevant list
       if (!iter.IsHead()) {
         // we move the elements around and construct the single-element linked list
         return IsFull() ? false : TrySpareListHead(iter, key, result);
       }
       // Case 3: head of the relevant list
       // we iterate through the linked list until the end
       // make sure `iter` is the previous element of `next`
       ListNode next = iter;
       do {
         // find equal item, do not insert
         if (AnyEqual()(key, next.Key())) {
           // we plan to take next, so we need to unlink it from iterator list
           IterListUnlink(next);
           *result = next;
           return true;
         }
         // make sure `iter` is the previous element of `next`
         iter = next;
       } while (next.MoveToNext(this));
       // `iter` is the tail of the linked list
       // always check capacity before insertion
       if (IsFull()) {
         return false;
       }
       // find the next empty slot
       uint8_t jump;
       if (!iter.GetNextEmpty(this, &jump, result)) {
         return false;
       }
       result->NewTail(ItemType(key, Any(nullptr)));
       // link `iter` to `empty`, and move forward
       iter.SetJump(jump);
       this->size_ += 1;
       return true;
     }
     bool TrySpareListHead(ListNode target, const key_type& key, ListNode* result) {
       // `target` is not the head of the linked list
       // move the original item of `target` (if any)
       // and construct new item on the position `target`
       // To make `target` empty, we
       // 1) find `w` the previous element of `target` in the linked list
       // 2) copy the linked list starting from `r = target`
       // 3) paste them after `w`
       // read from the linked list after `r`
       ListNode r = target;
       // write to the tail of `w`
       ListNode w = target.FindPrev(this);
       // after `target` is moved, we disallow writing to the slot
       bool is_first = true;
       uint8_t r_meta, jump;
       ListNode empty;
       do {
         // `jump` describes how `w` is jumped to `empty`
         // rehash if there is no empty space after `w`
         if (!w.GetNextEmpty(this, &jump, &empty)) {
           return false;
         }
         // move `r` to `empty`
         // first move the data over
         empty.NewTail(ItemType(std::move(r.Data())));
         // then move link list chain of r to empty
         // this needs to happen after NewTail so empty's prev/next get updated
         IterListReplaceNodeBy(r, empty);
         // explicit call destructor to destroy the item in `r`
         r.DestructData();
         // clear the metadata of `r`
         r_meta = r.Meta();
         if (is_first) {
           is_first = false;
           r.SetProtected();
         } else {
           r.SetEmpty();
         }
         // link `w` to `empty`, and move forward
         w.SetJump(jump);
         w = empty;
         // move `r` forward as well
       } while (r.MoveToNext(this, r_meta));
       // finally we have done moving the linked list
       // fill data_ into `target`
       target.NewHead(ItemType(key, Any(nullptr)));
       this->size_ += 1;
       *result = target;
       return true;
     }
     void Erase(const ListNode& iter) {
       this->size_ -= 1;
       if (!iter.HasNext()) {
         // `iter` is the last
         if (!iter.IsHead()) {
           // cut the link if there is any
           iter.FindPrev(this).SetJump(0);
         }
         // unlink the node from iterator list
         IterListUnlink(iter);
         // IMPORTANT: must explicit call destructor `iter` to avoid memory leak
         // This is because we need to recycle iter's data
         iter.DestructData();
         // set the meta data to be empty
         iter.SetEmpty();
       } else {
         ListNode last = iter, prev = iter;
         for (last.MoveToNext(this); last.HasNext(); prev = last, last.MoveToNext(this)) {
         }
         // needs to first unlink iter from the list
         IterListUnlink(iter);
         // move data from last to iter
         iter.Data() = std::move(last.Data());
         // Move link chain of iter to last as we stores last node to the new iter loc.
         IterListReplaceNodeBy(last, iter);
         // IMPORTANT: must explicit call destructor `last` to avoid memory leak
         // likely we don't need this in this particular case because Any move behavior
         // keep it here to be safe so code do not depend on specific move behavior of KVType
         last.DestructData();
         // set the meta data to be empty
         last.SetEmpty();
         prev.SetJump(0);
       }
     }
     void Reset() {
       clear();
       ReleaseMemory();
     }
     void InplaceSwitchTo(ObjectPtr<Object>&& other) {
       this->Reset();
       MapBaseObj* other_map = static_cast<MapBaseObj*>(other.get());
       // to simplify implementation, inplace switch from another dense map
       // since all current switch usecases are pushing elements to map
       // so we don't need to handle small -> dense switch
       TVM_FFI_ICHECK(!other_map->IsSmallMap());
       DenseMapBaseObj* other_dense_map = static_cast<DenseMapBaseObj*>(other_map);
       DenseMapBaseObj* this_dense_map = this;
       this_dense_map->size_ = other_dense_map->size_;
       this_dense_map->slots_ = other_dense_map->slots_;
       this_dense_map->data_ = other_dense_map->data_;
       this_dense_map->data_deleter_ = other_dense_map->data_deleter_;
       this_dense_map->fib_shift_ = other_dense_map->fib_shift_;
       this_dense_map->iter_list_head_ = other_dense_map->iter_list_head_;
       this_dense_map->iter_list_tail_ = other_dense_map->iter_list_tail_;
       other_dense_map->data_ = nullptr;
       other_dense_map->data_deleter_ = nullptr;
       other_dense_map->size_ = 0;
       other_dense_map->slots_ = 0;
     }
     void ReleaseMemory() {
       if (data_ != nullptr) {
         TVM_FFI_ICHECK(data_deleter_ != nullptr);
         data_deleter_(data_);
       }
       data_ = nullptr;
       data_deleter_ = nullptr;
       slots_ = 0;
       size_ = 0;
       fib_shift_ = 63;
     }
     template <typename MapObjType>
     static ObjectPtr<Object> Empty(uint32_t fib_shift, uint64_t n_slots) {
       // Ensure even slot count (power-of-two expected by callers; this guard
       // makes the method robust if a non-even value slips through).
       ObjectPtr<DenseMapBaseObj> p = make_object<DenseMapBaseObj>();
       uint64_t n_blocks = CalcNumBlocks(n_slots);
       Block* block = new Block[n_blocks];
       p->data_ = block;
       // assign block deleter so even if we take re-alloc data
       // in another shared-lib that may have different malloc/free behavior
       // it will still be safe.
       p->data_deleter_ = BlockDeleter;
       p->SetSlotsAndDenseLayoutTag(n_slots);
       p->size_ = 0;
       p->fib_shift_ = fib_shift;
       p->iter_list_head_ = kInvalidIndex;
       p->iter_list_tail_ = kInvalidIndex;
       // manually set the type index to specialized subclass
       // data layout is ensured to be the same except for the type index
       static_assert(std::is_base_of_v<MapBaseObj, MapObjType>);
       static_assert(sizeof(MapObjType) == sizeof(MapBaseObj));
       p->header_.type_index = MapObjType::RuntimeTypeIndex();
       for (uint64_t i = 0; i < n_blocks; ++i, ++block) {
         std::fill(block->bytes, block->bytes + kBlockCap, kEmptySlot);
       }
       return p;
     }
     template <typename MapObjType>
     static ObjectPtr<Object> CopyFrom(DenseMapBaseObj* from) {
       ObjectPtr<DenseMapBaseObj> p = make_object<DenseMapBaseObj>();
       uint64_t n_blocks = CalcNumBlocks(from->NumSlots());
       p->data_ = new Block[n_blocks];
       // assign block deleter so even if we take re-alloc data
       // in another shared-lib that may have different malloc/free behavior
       // it will still be safe.
       p->data_deleter_ = BlockDeleter;
       p->SetSlotsAndDenseLayoutTag(from->NumSlots());
       p->size_ = from->size_;
       p->fib_shift_ = from->fib_shift_;
       p->iter_list_head_ = from->iter_list_head_;
       p->iter_list_tail_ = from->iter_list_tail_;
       // manually set the type index to specialized subclass
       // data layout is ensured to be the same except for the type index
       static_assert(std::is_base_of_v<MapBaseObj, MapObjType>);
       static_assert(sizeof(MapObjType) == sizeof(MapBaseObj));
       p->header_.type_index = MapObjType::RuntimeTypeIndex();
       for (uint64_t bi = 0; bi < n_blocks; ++bi) {
         uint8_t* meta_ptr_from = from->GetBlock(bi)->bytes;
         ItemType* data_ptr_from = reinterpret_cast<ItemType*>(from->GetBlock(bi)->bytes + kBlockCap);
         uint8_t* meta_ptr_to = p->GetBlock(bi)->bytes;
         ItemType* data_ptr_to = reinterpret_cast<ItemType*>(p->GetBlock(bi)->bytes + kBlockCap);
         for (int j = 0; j < kBlockCap;
              ++j, ++meta_ptr_from, ++data_ptr_from, ++meta_ptr_to, ++data_ptr_to) {
           uint8_t& meta = *meta_ptr_to = *meta_ptr_from;
           TVM_FFI_ICHECK(meta != kProtectedSlot);
           if (meta != kEmptySlot) {
             new (data_ptr_to) ItemType(*data_ptr_from);
           }
         }
       }
       return p;
     }
     template <typename MapObjType>
     static ObjectPtr<Object> InsertMaybeReHash(KVType&& kv, const ObjectPtr<Object>& map) {
       DenseMapBaseObj* map_node = static_cast<DenseMapBaseObj*>(map.get());
       ListNode iter;
       // Try to insert. If succeed, we simply return
       if (map_node->TryInsert(kv.first, &iter)) {
         iter.Val() = std::move(kv.second);
         // update the iter list relation
         map_node->IterListPushBack(iter);
         return ObjectPtr<Object>(nullptr);
       }
       TVM_FFI_ICHECK(!map_node->IsSmallMap());
       // Otherwise, start rehash
       ObjectPtr<Object> p = Empty<MapObjType>(map_node->fib_shift_ - 1, map_node->NumSlots() * 2);
   
       // need to insert in the same order as the original map
       for (uint64_t index = map_node->iter_list_head_; index != kInvalidIndex;) {
         ListNode node(index, map_node);
         // now try move src_data into the new map, note that src may still not
         // be fully consumed into the call, but destructor will be called.
         ObjectPtr<Object> rehashed = InsertMaybeReHash<MapObjType>(std::move(node.Data()), p);
         if (rehashed != nullptr) p = std::move(rehashed);
         // Important, needs to explicit call destructor in case move did remove
         // node's internal item
         index = node.Item().next;
         // IMPORTANT: must explicit call destructor `node` to avoid memory leak
         // We must call node.DestructData() here.
         // This is because std::move() arguments in IterMaybeReHash may or may not
         // explicitly move out the node.Data()
         // Remove this call will cause memory leak very likely.
         node.DestructData();
       }
       {
         ObjectPtr<Object> rehashed = InsertMaybeReHash<MapObjType>(std::move(kv), p);
         if (rehashed != nullptr) p = std::move(rehashed);
       }
       map_node->ReleaseMemory();
       return p;
     }
     bool IsFull() const {  // NOLINTNEXTLINE(bugprone-narrowing-conversions)
       return (size_ + 1) > static_cast<uint64_t>(NumSlots()) * kMaxLoadFactor;
     }
     uint64_t IncItr(uint64_t index) const {
       // keep at the end of iterator
       if (index == kInvalidIndex) {
         return index;
       }
       ListNode node(index, this);
       return node.Item().next;
     }
     uint64_t DecItr(uint64_t index) const {
       // this is the end iterator, we need to return tail.
       if (index == kInvalidIndex) {
         return iter_list_tail_;
       }
       // circle around the iterator list, which is OK
       ListNode node(index, this);
       return node.Item().prev;
     }
     KVType* DeRefItr(uint64_t index) const { return &ListNode(index, this).Data(); }
     ListNode IndexFromHash(uint64_t hash_value) const {
       return ListNode(FibHash(hash_value, fib_shift_), this);
     }
     ListNode GetListHead(uint64_t hash_value) const {
       ListNode node = IndexFromHash(hash_value);
       return node.IsHead() ? node : ListNode();
     }
     static uint64_t CalcNumBlocks(uint64_t n_slots) { return (n_slots + kBlockCap - 1) / kBlockCap; }
     static void CalcTableSize(uint64_t cap, uint32_t* fib_shift, uint64_t* n_slots) {
       uint32_t shift = 64;
       uint64_t slots = 1;
       for (uint64_t c = cap; c; c >>= 1) {
         shift -= 1;
         slots <<= 1;
       }
       TVM_FFI_ICHECK_GT(slots, cap);
       if (slots < cap * 2) {
         *fib_shift = shift - 1;
         *n_slots = slots << 1;
       } else {
         *fib_shift = shift;
         *n_slots = slots;
       }
     }
     static uint64_t FibHash(uint64_t hash_value, uint32_t fib_shift) {
       constexpr uint64_t coeff = 11400714819323198485ull;
       return (coeff * hash_value) >> fib_shift;
     }
     struct ListNode {
       ListNode() : index(0), block(nullptr) {}
       ListNode(uint64_t index, const DenseMapBaseObj* self)
           : index(index), block(self->GetBlock(index / kBlockCap)) {}
       uint8_t& Meta() const { return *(block->bytes + index % kBlockCap); }
       ItemType& Item() const {
         return *(reinterpret_cast<ItemType*>(block->bytes + kBlockCap +
                                              (index % kBlockCap) * sizeof(ItemType)));
       }
       KVType& Data() const { return Item().data; }
       key_type& Key() const { return Data().first; }
       mapped_type& Val() const { return Data().second; }
       bool IsHead() const { return (Meta() & 0b10000000) == 0b00000000; }
       bool IsNone() const { return block == nullptr; }
       bool IsEmpty() const { return Meta() == kEmptySlot; }
       bool IsProtected() const { return Meta() == kProtectedSlot; }
       void SetEmpty() const { Meta() = kEmptySlot; }
       void DestructData() const {
         // explicit call destructor to destroy the item
         // Favor this over ~KVType as MSVC may not support ~KVType (need the original name)
         (&Data())->first.Any::~Any();
         (&Data())->second.Any::~Any();
       }
       void SetProtected() const { Meta() = kProtectedSlot; }
       void SetJump(uint8_t jump) const { (Meta() &= 0b10000000) |= jump; }
       void NewHead(ItemType v) const {
         Meta() = 0b00000000;
         new (&Item()) ItemType(std::move(v));
       }
       void NewTail(ItemType v) const {
         Meta() = 0b10000000;
         new (&Item()) ItemType(std::move(v));
       }
   
       bool HasNext() const { return NextProbeLocation(Meta() & 0b01111111) != 0; }
       bool MoveToNext(const DenseMapBaseObj* self, uint8_t meta) {
         uint64_t offset = NextProbeLocation(meta & 0b01111111);
         if (offset == 0) {
           index = 0;
           block = nullptr;
           return false;
         }
         // the probing will go to next position and round back to stay within the
         // correct range of the slots
         index = (index + offset) % self->NumSlots();
         block = self->GetBlock(index / kBlockCap);
         return true;
       }
       bool MoveToNext(const DenseMapBaseObj* self) { return MoveToNext(self, Meta()); }
       ListNode FindPrev(const DenseMapBaseObj* self) const {
         // start from the head of the linked list, which must exist
         ListNode next = self->IndexFromHash(AnyHash()(Key()));
         // `prev` is always the previous item of `next`
         ListNode prev = next;
         for (next.MoveToNext(self); index != next.index; prev = next, next.MoveToNext(self)) {
         }
         return prev;
       }
       bool GetNextEmpty(const DenseMapBaseObj* self, uint8_t* jump, ListNode* result) const {
         for (uint8_t idx = 1; idx < kNumJumpDists; ++idx) {
           // the probing will go to next position and round back to stay within the
           // correct range of the slots
           ListNode candidate((index + NextProbeLocation(idx)) % self->NumSlots(), self);
           if (candidate.IsEmpty()) {
             *jump = idx;
             *result = candidate;
             return true;
           }
         }
         return false;
       }
       uint64_t index;
       Block* block;
     };
   
    protected:
     uint32_t fib_shift_;
     uint64_t iter_list_head_ = kInvalidIndex;
     uint64_t iter_list_tail_ = kInvalidIndex;
   
     static uint64_t NextProbeLocation(size_t index) {
       /* clang-format off */
       static const uint64_t kNextProbeLocation[kNumJumpDists] {
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         // Quadratic probing with triangle numbers. See also:
         // 1) https://en.wikipedia.org/wiki/Quadratic_probing
         // 2) https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n/
         // 3) https://github.com/skarupke/flat_hash_map
         21, 28, 36, 45, 55, 66, 78, 91, 105, 120,
         136, 153, 171, 190, 210, 231, 253, 276, 300, 325,
         351, 378, 406, 435, 465, 496, 528, 561, 595, 630,
         666, 703, 741, 780, 820, 861, 903, 946, 990, 1035,
         1081, 1128, 1176, 1225, 1275, 1326, 1378, 1431, 1485, 1540,
         1596, 1653, 1711, 1770, 1830, 1891, 1953, 2016, 2080, 2145,
         2211, 2278, 2346, 2415, 2485, 2556, 2628,
         // larger triangle numbers
         8515, 19110, 42778, 96141, 216153,
         486591, 1092981, 2458653, 5532801, 12442566,
         27993903, 62983476, 141717030, 318844378, 717352503,
         1614057336, 3631522476, 8170957530, 18384510628, 41364789378,
         93070452520, 209408356380, 471168559170, 1060128894105, 2385289465695,
         5366898840628, 12075518705635, 27169915244790, 61132312065111, 137547689707000,
         309482283181501, 696335127828753, 1566753995631385, 3525196511162271, 7931691992677701,
         17846306936293605, 40154190677507445, 90346928918121501, 203280589587557251,
         457381325854679626, 1029107982097042876, 2315492959180353330, 5209859154120846435,
       };
       /* clang-format on */
       return kNextProbeLocation[index];
     }
     friend class MapBaseObj;
     friend class SmallMapBaseObj;
   
    private:
     void SetSlotsAndDenseLayoutTag(uint64_t n) {
       TVM_FFI_ICHECK(((n & kSmallTagMask) == 0ull)) << "DenseMap expects MSB clear";
       slots_ = n;
     }
   };
   
   class SmallMapBaseObj : public MapBaseObj {
    private:
     static constexpr uint64_t kInitSize = 2;
     static constexpr uint64_t kMaxSize = 4;
   
    public:
     using MapBaseObj::iterator;
     using MapBaseObj::KVType;
   
     // Return the number of usable slots for Small layout (mask off tag).
     uint64_t NumSlots() const { return slots_ & ~kSmallTagMask; }
   
     ~SmallMapBaseObj() {
       // in destructor, need to check if the map was inplace switched to a dense map.
       MapBaseObj* base_map = static_cast<MapBaseObj*>(this);
       if (base_map->IsSmallMap()) {
         this->Reset();
       } else {
         // this map was inplace switched to a dense map.
         DenseMapBaseObj* this_dense_map = static_cast<DenseMapBaseObj*>(base_map);
         this_dense_map->Reset();
       }
     }
     void clear() {
       KVType* begin = static_cast<KVType*>(data_);
       for (uint64_t index = 0; index < size_; ++index) {
         // call destructor to destroy the item in `begin + index`
         // Explicit call Any::~Any() to destroy the Any object
         // Favor this over ~KVType as MSVC may not support ~KVType (need the original name)
         (begin + index)->first.Any::~Any();
         (begin + index)->second.Any::~Any();
       }
       size_ = 0;
     }
     size_t count(const key_type& key) const { return find(key).index < size_; }
     const mapped_type& at(const key_type& key) const {
       iterator itr = find(key);
       if (itr.index >= size_) {
         TVM_FFI_THROW(KeyError) << "key is not in Map";
       }
       return itr->second;
     }
     mapped_type& at(const key_type& key) {
       iterator itr = find(key);
       if (itr.index >= size_) {
         TVM_FFI_THROW(KeyError) << "key is not in Map";
       }
       return itr->second;
     }
     iterator begin() const { return iterator(0, this); }
     iterator end() const { return iterator(size_, this); }
     iterator find(const key_type& key) const {
       KVType* ptr = static_cast<KVType*>(data_);
       for (uint64_t i = 0; i < size_; ++i, ++ptr) {
         if (AnyEqual()(ptr->first, key)) {
           return iterator(i, this);
         }
       }
       return iterator(size_, this);
     }
     void erase(const iterator& position) { Erase(position.index); }
   
    private:
     void SetSlotsAndSmallLayoutTag(uint64_t n) { slots_ = (n & ~kSmallTagMask) | kSmallTagMask; }
     /*
      * \brief Reset original small Storage so it can be ready for switch.
      */
     void Reset() {
       this->clear();
       if (data_deleter_ != nullptr) {
         data_deleter_(data_);
         // after deletion we have zero slots
         slots_ = 0;
         data_ = nullptr;
         data_deleter_ = nullptr;
       }
     }
     static void InplaceSmallMapDeleterFromData(void* data) {
       details::ObjectUnsafe::ObjectPtrFromOwned<SmallMapBaseObj>(
           reinterpret_cast<Object*>(reinterpret_cast<char*>(data) - sizeof(SmallMapBaseObj)))
           .reset();
     }
     void InplaceSwitchTo(ObjectPtr<Object>&& other) {
       // invariant this map have not been inplace switched to a dense map
       TVM_FFI_ICHECK(this->IsSmallMap());
       this->Reset();
       MapBaseObj* other_map = static_cast<MapBaseObj*>(other.get());
       if (other_map->IsSmallMap()) {
         SmallMapBaseObj* other_small_map = static_cast<SmallMapBaseObj*>(other_map);
         SmallMapBaseObj* this_small_map = this;
         this_small_map->size_ = other_small_map->size_;
         this_small_map->slots_ = other_small_map->slots_;
         this_small_map->data_ = other_small_map->data_;
         if (other_small_map->data_deleter_ != nullptr) {
           this_small_map->data_deleter_ = other_small_map->data_deleter_;
           other_small_map->data_deleter_ = nullptr;
         } else {
           // we switch to the inplace deleter from the data
           this_small_map->data_deleter_ = InplaceSmallMapDeleterFromData;
           // move out other from the ptr so the deletion can only be triggered
           // via InplaceSmallMapDeleterFromData
           details::ObjectUnsafe::MoveObjectPtrToTVMFFIObjectPtr(std::move(other));
         }
         other_small_map->data_ = nullptr;
         other_small_map->size_ = 0;
         other_small_map->slots_ = 0;
       } else {
         // Reinterpret this SmallMapBaseObj's memory as DenseMapBaseObj to write fields at the
         // correct offsets. This is raw memory manipulation: SmallMapBaseObj's allocation is
         // guaranteed large enough (see static_assert in Empty()), and all member access
         // compiles to fixed-offset stores with no virtual dispatch involved.
         // The destructor will also cross check and apply the correct deletion.
         // As a result, we can inplace switch the container storage to dense map
         DenseMapBaseObj* other_dense_map = static_cast<DenseMapBaseObj*>(other_map);
         DenseMapBaseObj* this_dense_map = reinterpret_cast<DenseMapBaseObj*>(this);
         this_dense_map->size_ = other_dense_map->size_;
         this_dense_map->slots_ = other_dense_map->slots_;
         this_dense_map->data_ = other_dense_map->data_;
         this_dense_map->data_deleter_ = other_dense_map->data_deleter_;
         this_dense_map->fib_shift_ = other_dense_map->fib_shift_;
         this_dense_map->iter_list_head_ = other_dense_map->iter_list_head_;
         this_dense_map->iter_list_tail_ = other_dense_map->iter_list_tail_;
         other_dense_map->data_ = nullptr;
         other_dense_map->data_deleter_ = nullptr;
         other_dense_map->size_ = 0;
         other_dense_map->slots_ = 0;
       }
     }
     void Erase(const uint64_t index) {
       if (index >= size_) {
         return;
       }
       KVType* begin = static_cast<KVType*>(data_);
       // call destructor to destroy the item in `begin + index`
       // Explicit call Any::~Any() to destroy the Any object
       // Favor this over ~KVType as MSVC may not support ~KVType (need the original name)
       (begin + index)->first.Any::~Any();
       (begin + index)->second.Any::~Any();
       // IMPORTANT: We do direct raw memmove to bring later items to the current position
       // to preserve the order of insertion.
       // This works because direct memory copy preserves the Any's move semantics.
       if (index + 1 < size_) {
         std::memmove(reinterpret_cast<char*>(begin + index),
                      reinterpret_cast<char*>(begin + index + 1),
                      (size_ - index - 1) * sizeof(KVType));
       }
       size_ -= 1;
     }
     template <typename MapObjType>
     static ObjectPtr<Object> Empty(uint64_t n = kInitSize) {
       // We always allocate a SmallMapObj to be large enough so it can inplace switch to DenseMapObj
       static_assert(alignof(SmallMapBaseObj) % alignof(KVType) == 0);
       static_assert(sizeof(SmallMapBaseObj) + kInitSize * sizeof(KVType) >= sizeof(DenseMapBaseObj));
       n = std::max(n, static_cast<uint64_t>(kInitSize));
       // allocate a SmallMapBaseObj with enough space to inplace store n elements and switch to
       // DenseMapBaseObj
       ObjectPtr<SmallMapBaseObj> p = ffi::make_inplace_array_object<SmallMapBaseObj, KVType>(n);
       // data_ is after the SmallMapBaseObj header
       p->data_ = reinterpret_cast<char*>(p.get()) + sizeof(SmallMapBaseObj);
       p->size_ = 0;
       p->SetSlotsAndSmallLayoutTag(n);
       // manually set the type index to specialized subclass
       // data layout is ensured to be the same except for the type index
       static_assert(std::is_base_of_v<MapBaseObj, MapObjType>);
       static_assert(sizeof(MapObjType) == sizeof(MapBaseObj));
       p->header_.type_index = MapObjType::RuntimeTypeIndex();
       return p;
     }
     template <typename MapObjType, typename IterType>
     static ObjectPtr<Object> CreateFromRange(uint64_t n, IterType first, IterType last) {
       ObjectPtr<Object> p = Empty<MapObjType>(n);
       SmallMapBaseObj* small_map_base_obj = static_cast<SmallMapBaseObj*>(p.get());
       KVType* ptr = static_cast<KVType*>(small_map_base_obj->data_);
       for (; first != last; ++first, ++small_map_base_obj->size_) {
         new (ptr++) KVType(*first);
       }
       return p;
     }
     template <typename MapObjType>
     static ObjectPtr<Object> CopyFrom(SmallMapBaseObj* from) {
       KVType* first = static_cast<KVType*>(from->data_);
       KVType* last = first + from->size_;
       return CreateFromRange<MapObjType>(from->size_, first, last);
     }
     template <typename MapObjType>
     static ObjectPtr<Object> InsertMaybeReHash(KVType&& kv, const ObjectPtr<Object>& map) {
       SmallMapBaseObj* map_node = static_cast<SmallMapBaseObj*>(map.get());
       iterator itr = map_node->find(kv.first);
       if (itr.index < map_node->size_) {
         itr->second = kv.second;
         return ObjectPtr<Object>(nullptr);
       }
       if (map_node->size_ < map_node->NumSlots()) {
         KVType* ptr = static_cast<KVType*>(map_node->data_) + map_node->size_;
         new (ptr) KVType(std::move(kv));
         ++map_node->size_;
         return ObjectPtr<Object>(nullptr);
       }
       uint64_t next_size = std::max(map_node->NumSlots() * 2, kInitSize);
       next_size = std::min(next_size, kMaxSize);
       TVM_FFI_ICHECK_GT(next_size, map_node->NumSlots());
       ObjectPtr<Object> new_map =
           CreateFromRange<MapObjType>(next_size, map_node->begin(), map_node->end());
       ObjectPtr<Object> rehashed = InsertMaybeReHash<MapObjType>(std::move(kv), new_map);
       if (rehashed != nullptr) new_map = std::move(rehashed);
       return new_map;
     }
     uint64_t IncItr(uint64_t index) const { return index + 1 < size_ ? index + 1 : size_; }
     uint64_t DecItr(uint64_t index) const { return index > 0 ? index - 1 : size_; }
     KVType* DeRefItr(uint64_t index) const { return static_cast<KVType*>(data_) + index; }
   
    protected:
     friend class MapBaseObj;
     friend class DenseMapBaseObj;
   };
   
   #define TVM_FFI_DISPATCH_MAP(base, var, body)   \
     {                                             \
       using TSmall = SmallMapBaseObj*;            \
       using TDense = DenseMapBaseObj*;            \
       if ((base)->IsSmallMap()) {                 \
         TSmall var = static_cast<TSmall>((base)); \
         body;                                     \
       } else {                                    \
         TDense var = static_cast<TDense>((base)); \
         body;                                     \
       }                                           \
     }
   
   #define TVM_FFI_DISPATCH_MAP_CONST(base, var, body) \
     {                                                 \
       using TSmall = const SmallMapBaseObj*;          \
       using TDense = const DenseMapBaseObj*;          \
       if ((base)->IsSmallMap()) {                     \
         TSmall var = static_cast<TSmall>((base));     \
         body;                                         \
       } else {                                        \
         TDense var = static_cast<TDense>((base));     \
         body;                                         \
       }                                               \
     }
   
   inline MapBaseObj::iterator::pointer MapBaseObj::iterator::operator->() const {
     TVM_FFI_MAP_FAIL_IF_CHANGED()
     TVM_FFI_DISPATCH_MAP_CONST(self, p, { return p->DeRefItr(index); });
   }
   
   inline MapBaseObj::iterator& MapBaseObj::iterator::operator++() {
     TVM_FFI_MAP_FAIL_IF_CHANGED()
     TVM_FFI_DISPATCH_MAP_CONST(self, p, {
       index = p->IncItr(index);
       return *this;
     });
   }
   
   inline MapBaseObj::iterator& MapBaseObj::iterator::operator--() {
     TVM_FFI_MAP_FAIL_IF_CHANGED()
     TVM_FFI_DISPATCH_MAP_CONST(self, p, {
       index = p->DecItr(index);
       return *this;
     });
   }
   
   inline size_t MapBaseObj::count(const key_type& key) const {
     TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->count(key); });
   }
   
   inline const MapBaseObj::mapped_type& MapBaseObj::at(const MapBaseObj::key_type& key) const {
     TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->at(key); });
   }
   
   inline MapBaseObj::mapped_type& MapBaseObj::at(const MapBaseObj::key_type& key) {
     TVM_FFI_DISPATCH_MAP(this, p, { return p->at(key); });
   }
   
   inline MapBaseObj::iterator MapBaseObj::begin() const {
     TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->begin(); });
   }
   
   inline MapBaseObj::iterator MapBaseObj::end() const {
     TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->end(); });
   }
   
   inline MapBaseObj::iterator MapBaseObj::find(const MapBaseObj::key_type& key) const {
     TVM_FFI_DISPATCH_MAP_CONST(this, p, { return p->find(key); });
   }
   
   inline void MapBaseObj::erase(const MapBaseObj::iterator& position) {
     TVM_FFI_DISPATCH_MAP(this, p, { p->erase(position); });
   }
   
   inline void MapBaseObj::clear() {
     TVM_FFI_DISPATCH_MAP(this, p, { p->clear(); });
   }
   
   inline void MapBaseObj::InplaceSwitchTo(ObjectPtr<Object>&& other) {
     TVM_FFI_DISPATCH_MAP(this, p, { p->InplaceSwitchTo(std::move(other)); });
   }
   
   #undef TVM_FFI_DISPATCH_MAP
   #undef TVM_FFI_DISPATCH_MAP_CONST
   
   template <typename MapObjType>
   inline ObjectPtr<Object> MapBaseObj::Empty() {
     return SmallMapBaseObj::Empty<MapObjType>();
   }
   
   template <typename MapObjType>
   inline ObjectPtr<Object> MapBaseObj::CopyFrom(MapBaseObj* from) {
     if (from->IsSmallMap()) {
       return SmallMapBaseObj::CopyFrom<MapObjType>(static_cast<SmallMapBaseObj*>(from));
     } else {
       return DenseMapBaseObj::CopyFrom<MapObjType>(static_cast<DenseMapBaseObj*>(from));
     }
   }
   
   template <typename MapObjType, typename IterType>
   inline ObjectPtr<Object> MapBaseObj::CreateFromRange(IterType first, IterType last) {
     int64_t _cap = std::distance(first, last);
     if (_cap < 0) {
       return SmallMapBaseObj::Empty<MapObjType>();
     }
     uint64_t cap = static_cast<uint64_t>(_cap);
     if (cap < SmallMapBaseObj::kMaxSize) {
       if (cap < 2) {
         return SmallMapBaseObj::CreateFromRange<MapObjType>(cap, first, last);
       }
       // need to insert to avoid duplicate keys
       ObjectPtr<Object> obj = SmallMapBaseObj::Empty<MapObjType>(cap);
       for (; first != last; ++first) {
         KVType kv(*first);
         ObjectPtr<Object> rehashed =
             SmallMapBaseObj::InsertMaybeReHash<MapObjType>(std::move(kv), obj);
         if (rehashed != nullptr) obj = std::move(rehashed);
       }
       return obj;
     } else {
       uint32_t fib_shift;
       uint64_t n_slots;
       DenseMapBaseObj::CalcTableSize(cap, &fib_shift, &n_slots);
       ObjectPtr<Object> obj = DenseMapBaseObj::Empty<MapObjType>(fib_shift, n_slots);
       for (; first != last; ++first) {
         KVType kv(*first);
         ObjectPtr<Object> rehashed =
             DenseMapBaseObj::InsertMaybeReHash<MapObjType>(std::move(kv), obj);
         if (rehashed != nullptr) obj = std::move(rehashed);
       }
       return obj;
     }
   }
   
   template <typename MapObjType>
   inline ObjectPtr<Object> MapBaseObj::InsertMaybeReHash(KVType&& kv, const ObjectPtr<Object>& map) {
     MapBaseObj* base = static_cast<MapBaseObj*>(map.get());
   #if TVM_FFI_DEBUG_WITH_ABI_CHANGE
     base->state_marker++;
   #endif  // TVM_FFI_DEBUG_WITH_ABI_CHANGE
     if (base->IsSmallMap()) {
       SmallMapBaseObj* sm = static_cast<SmallMapBaseObj*>(base);
       if (sm->NumSlots() < SmallMapBaseObj::kMaxSize) {
         return SmallMapBaseObj::InsertMaybeReHash<MapObjType>(std::move(kv), map);
       } else if (sm->NumSlots() == SmallMapBaseObj::kMaxSize) {
         if (base->size_ < sm->NumSlots()) {
           return SmallMapBaseObj::InsertMaybeReHash<MapObjType>(std::move(kv), map);
         } else {
           ObjectPtr<Object> new_map =
               MapBaseObj::CreateFromRange<MapObjType>(base->begin(), base->end());
           ObjectPtr<Object> rehashed =
               DenseMapBaseObj::InsertMaybeReHash<MapObjType>(std::move(kv), new_map);
           if (rehashed != nullptr) new_map = std::move(rehashed);
           return new_map;
         }
       }
     } else {
       return DenseMapBaseObj::InsertMaybeReHash<MapObjType>(std::move(kv), map);
     }
     return ObjectPtr<Object>(nullptr);
   }
   
   
   template <>
   inline ObjectPtr<MapBaseObj> make_object<>() = delete;
   
   template <typename Derived, typename MapRef, typename K, typename V>
   struct MapTypeTraitsBase : public ObjectRefTypeTraitsBase<MapRef> {
     using Base = ObjectRefTypeTraitsBase<MapRef>;
     using Base::CopyFromAnyViewAfterCheck;
   
     TVM_FFI_INLINE static bool CheckAnyStrict(const TVMFFIAny* src) {
       if (src->type_index != Derived::kPrimaryTypeIndex) return false;
       if constexpr (std::is_same_v<K, Any> && std::is_same_v<V, Any>) {
         return true;
       } else {
         const MapBaseObj* n = reinterpret_cast<const MapBaseObj*>(src->v_obj);
         for (const auto& kv : *n) {
           if constexpr (!std::is_same_v<K, Any>) {
             if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first)) return false;
           }
           if constexpr (!std::is_same_v<V, Any>) {
             if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second)) return false;
           }
         }
         return true;
       }
     }
   
     TVM_FFI_INLINE static std::string GetMismatchTypeInfo(const TVMFFIAny* src) {
       if (src->type_index != Derived::kPrimaryTypeIndex &&
           src->type_index != Derived::kOtherTypeIndex) {
         return TypeTraitsBase::GetMismatchTypeInfo(src);
       }
       if constexpr (!std::is_same_v<K, Any> || !std::is_same_v<V, Any>) {
         const MapBaseObj* n = reinterpret_cast<const MapBaseObj*>(src->v_obj);
         for (const auto& kv : *n) {
           if constexpr (!std::is_same_v<K, Any>) {
             if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first) &&
                 !kv.first.try_cast<K>().has_value()) {
               return std::string(Derived::kTypeName) + "[some key is " +
                      details::AnyUnsafe::GetMismatchTypeInfo<K>(kv.first) + ", V]";
             }
           }
           if constexpr (!std::is_same_v<V, Any>) {
             if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second) &&
                 !kv.second.try_cast<V>().has_value()) {
               return std::string(Derived::kTypeName) + "[K, some value is " +
                      details::AnyUnsafe::GetMismatchTypeInfo<V>(kv.second) + "]";
             }
           }
         }
       }
       TVM_FFI_THROW(InternalError) << "Cannot reach here";
       TVM_FFI_UNREACHABLE();
     }
   
     TVM_FFI_INLINE static std::optional<MapRef> TryCastFromAnyView(const TVMFFIAny* src) {
       if (src->type_index != Derived::kPrimaryTypeIndex &&
           src->type_index != Derived::kOtherTypeIndex) {
         return std::nullopt;
       }
       const MapBaseObj* n = reinterpret_cast<const MapBaseObj*>(src->v_obj);
       if constexpr (!std::is_same_v<K, Any> || !std::is_same_v<V, Any>) {
         bool storage_check = [&]() {
           for (const auto& kv : *n) {
             if constexpr (!std::is_same_v<K, Any>) {
               if (!details::AnyUnsafe::CheckAnyStrict<K>(kv.first)) return false;
             }
             if constexpr (!std::is_same_v<V, Any>) {
               if (!details::AnyUnsafe::CheckAnyStrict<V>(kv.second)) return false;
             }
           }
           return true;
         }();
         // fast path: if storage check passes and type is primary, return directly.
         if (storage_check && src->type_index == Derived::kPrimaryTypeIndex) {
           return CopyFromAnyViewAfterCheck(src);
         }
         // slow path: create a new map and convert each key-value pair.
         MapRef ret;
         for (const auto& kv : *n) {
           auto k = kv.first.try_cast<K>();
           auto v = kv.second.try_cast<V>();
           if (!k.has_value() || !v.has_value()) return std::nullopt;
           ret.Set(*std::move(k), *std::move(v));
         }
         return ret;
       } else {
         if (src->type_index == Derived::kPrimaryTypeIndex) {
           return CopyFromAnyViewAfterCheck(src);
         }
         // cross-type conversion for Any,Any: create new MapRef, copy all entries.
         MapRef ret;
         for (const auto& kv : *n) {
           ret.Set(kv.first, kv.second);
         }
         return ret;
       }
     }
   
     TVM_FFI_INLINE static std::string TypeStr() {
       return std::string(Derived::kTypeName) + "<" + details::Type2Str<K>::v() + ", " +
              details::Type2Str<V>::v() + ">";
     }
   
    private:
     MapTypeTraitsBase() = default;
     friend Derived;
   };
   
   }  // namespace ffi
   }  // namespace tvm
   #endif  // TVM_FFI_CONTAINER_MAP_BASE_H_
