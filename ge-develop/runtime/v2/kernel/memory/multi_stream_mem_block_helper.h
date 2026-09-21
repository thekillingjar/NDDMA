/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MULTI_STREAM_MEM_BLOCK_HELPER_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MULTI_STREAM_MEM_BLOCK_HELPER_H_
#include "multi_stream_mem_block.h"
namespace gert {
namespace memory {
/**
 * 该类使用了MultiStreamMemBlock的私有函数/成员，因此除了联系紧密的类（例如allocator），禁止其他类使用该类
 */
class MultiStreamMemBlockHelper {
 public:
  static int64_t PlusVersion(MultiStreamMemBlock *block) {
    return block->PlusVersion();
  }
  static void ClearMemBlock(MultiStreamMemBlock *block) {
    block->ClearMemBlock();
  }
  static void ReInitMifs(MultiStreamMemBlock *block) {
    block->ReInitMifs();
  }
  static bool CanGlobalRecycle(MultiStreamMemBlock *block, int64_t stream_id) {
    return !block->occupy_mif_.IsAnySet(stream_id);
  }
  /**
   * 在`maintained_stream_id`视角，`stream_id`是否在占用`block`
   * @param block
   * @param maintained_stream_id
   * @param stream_id
   * @return
   */
  static bool IsOccupied(const MultiStreamMemBlock *block, int64_t maintained_stream_id, int64_t stream_id) {
    return block->occupy_mif_.IsSet(maintained_stream_id, stream_id);
  }
  static void UpdateLocalRecycleMif(MultiStreamMemBlock *block, int64_t stream_id) {
    block->occupy_mif_.Clear(stream_id, stream_id);
  }
};
}
}

#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_MULTI_STREAM_MEM_BLOCK_HELPER_H_
