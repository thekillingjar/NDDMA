/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_SINGLE_STREAM_L_2_ALLOCATOR_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_SINGLE_STREAM_L_2_ALLOCATOR_H_
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "caching_mem_allocator.h"
#include "ti_block_allocator.h"

namespace gert {
namespace memory {
class SingleStreamL2Allocator : public GertAllocator {
 public:
  explicit SingleStreamL2Allocator(CachingMemAllocator *caching_mem_allocator = nullptr)
      : SingleStreamL2Allocator(kOnDeviceHbm, caching_mem_allocator) {}
  SingleStreamL2Allocator(TensorPlacement placement, ge::Allocator *caching_mem_allocator);

  ~SingleStreamL2Allocator() noexcept override;
  GertTensorData MallocTensorData(size_t size) override;
  GertMemBlock *Malloc(size_t size) override;
  TensorData MallocTensorDataFromL1(size_t size) override;
  void Free(GertMemBlock *gert_mem_block) override;
  ge::graphStatus FreeAt(int64_t stream_id, GertMemBlock *block) override;
  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override;
  ge::Allocator *GetL1Allocator();
  int64_t GetStreamNum() override;
  ge::graphStatus ShareFromTensorData(const TensorData &td, GertTensorData &gtd) override;
  void BirthRecycle(MultiStreamMemBlock *block);

 protected:
  ge::Allocator *l1_allocator_;
  MultiStreamMemBlockPool ms_block_pool_;
  TiGtdAllocator ti_allocator_;
};
}  // namespace memory
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_SINGLE_STREAM_L_2_ALLOCATOR_H_
