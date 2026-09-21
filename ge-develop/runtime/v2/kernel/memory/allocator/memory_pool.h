/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_MEMORY_POOL_H
#define AIR_CXX_MEMORY_POOL_H

#include "ge/ge_allocator.h"
#include "ge_common/ge_api_types.h"
#include "kernel/memory/type/mem_size.h"
#include "kernel/memory/device/device_allocator.h"

namespace gert {
class MemoryPool {
 public:
  explicit MemoryPool(DeviceMemAllocator &allocator) :device_mem_allocator_(allocator) {}
  virtual ~MemoryPool() = default;
  virtual ge::MemBlock *Alloc(ge::Allocator &allocator, const MemSize size) = 0;
  virtual void Free(ge::MemBlock *block) = 0;

  /**
   * 将空闲内存归还给DeviceMemAllocator
   * @return
   */
  virtual void Recycle() = 0;
  virtual ge::Status Finalize(bool no_log) = 0;

  /**
   * 从内存复用的block中获取原始的block
   * @param block 内存复用的block
   * @return 如果block没有被切分则返回原始的block，否则转换失败返回nullptr
   */
  virtual ge::MemBlock *ConvertToRootBlock(ge::MemBlock *block) = 0;

  virtual const std::string &GetId() const = 0;

  virtual void PrintDetails(const int32_t level) = 0;

 protected:
  DeviceMemAllocator &device_mem_allocator_;
};
}  // namespace gert

#endif  // AIR_CXX_MEMORY_POOL_H
