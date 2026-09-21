/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MULTI_STREAM_L_2_MEM_BLOCK_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MULTI_STREAM_L_2_MEM_BLOCK_H_
#include <cstdint>
#include <vector>
#include <list>
#include "exe_graph/runtime/gert_mem_block.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "mif.h"
#include "block_alloc_type.h"
namespace gert {
namespace memory {

/**
 * 支持多流的Memory Block，对于同一块内存，对应唯一的本实例。当发生birth回收时，本实例同步可以被回收。
 */
class MultiStreamMemBlock : public GertMemBlock {
 public:
  MultiStreamMemBlock();
  ~MultiStreamMemBlock() override;
  MultiStreamMemBlock(const MultiStreamMemBlock &) = delete;
  MultiStreamMemBlock &operator=(const MultiStreamMemBlock &) = delete;

  /**
   * 调用本接口前，需要保证GertMemBlock对应的内存已经完成了birth recycle
   * @param birth_stream
   * @param stream_ids_to_allocators
   * @param block
   * @return
   */
  ge::graphStatus ReInit(GertAllocator *birth_allocator, ge::MemBlock *block, BlockAllocType alloc_type);
  void Free(int64_t stream_id) override;

  void *GetAddr() override {
    if (SECUREC_UNLIKELY(mem_block_ == nullptr)) {
      return nullptr;
    }
    return mem_block_->GetAddr();
  }

  size_t GetSize() const {
    if (SECUREC_UNLIKELY(mem_block_ == nullptr)) {
      return 0U;
    }
    return mem_block_->GetSize();
  }
  void SetSize(const size_t mem_size) {
    if (SECUREC_LIKELY(mem_block_ != nullptr)) {
      mem_block_->SetSize(mem_size);
    }
  }

  size_t AddCount(int64_t stream_id) {
    if (SECUREC_UNLIKELY(!IsStreamIdValid(stream_id))) {
      return 0U;
    }
    return ++stream_ids_to_occupy_count_[stream_id];
  }
  size_t SubCount(int64_t stream_id) {
    if (SECUREC_UNLIKELY(!IsStreamIdValid(stream_id))) {
      return 0U;
    }
    return --stream_ids_to_occupy_count_[stream_id];
  }
  size_t GetCount(int64_t stream_id) const {
    if (SECUREC_UNLIKELY(!IsStreamIdValid(stream_id))) {
      return 0U;
    }
    return stream_ids_to_occupy_count_[stream_id];
  }

  const ge::MemBlock *GetMemBlock() const {
    return mem_block_;
  }

  ge::MemBlock *GetMemBlock() {
    return mem_block_;
  }

  void SetMemBlock(ge::MemBlock *block) {
    mem_block_ = block;
  }

  int64_t GetBirthStreamId() const {
    return birth_allocator_->GetStreamId();
  }

  int64_t GetVersion() const {
    return version_;
  }

  ge::graphStatus NewAccessStream(int64_t src_stream, int64_t dst_stream);

  /**
   * 从 src_stream向dst_stream同步回收状态
   * @param src_stream
   * @param dst_stream
   * @return
   */
  ge::graphStatus SyncLocalRecycleStatus(int64_t src_stream, int64_t dst_stream);
  ge::graphStatus SyncAllLocalRecycleStatus(int64_t dst_stream);

  const GertAllocator *GetBirthAllocator() const {
    return birth_allocator_;
  }
  GertAllocator *GetBirthAllocator() {
    return birth_allocator_;
  }

  const BlockAllocType &GetAllocType() const {
    return alloc_type_;
  }

 private:
  bool IsStreamIdValid(int64_t stream_id) const {
    return static_cast<size_t>(stream_id) < stream_ids_to_occupy_count_.size();
  }

  int64_t PlusVersion() {
    return ++version_;
  }

  /**
   * 调用本函数前，确认`birth_allocator_`已经正确初始化
   */
  void ReInitMifs() {
    occupy_mif_.ReInit(birth_allocator_->GetStreamNum());
    stream_ids_to_occupy_count_.clear();
    stream_ids_to_occupy_count_.resize(birth_allocator_->GetStreamNum(), 0U);
  }
  /**
   * 危险函数，确保本block对应的内存已经完成了birth回收，才可以调用该函数
   */
  void ClearMemBlock() {
    mem_block_ = nullptr;
  }

 private:
  ge::MemBlock *mem_block_;
  int64_t version_;
  GertAllocator *birth_allocator_;
  BlockAllocType alloc_type_;
  Mif occupy_mif_;
  ge::SmallVector<size_t, kDefaultMaxStreamNum> stream_ids_to_occupy_count_;
  friend class MultiStreamMemBlockHelper;
};
}  // namespace memory
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MULTI_STREAM_L_2_MEM_BLOCK_H_
