/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_BLOCK_ALLOC_TYPE_H_
#define AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_BLOCK_ALLOC_TYPE_H_
#include <cstdint>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif

namespace gert {
namespace memory {
class BlockAllocType {
 public:
  enum MainType {
    kTensorDataWrapped,   // 外部申请，作为图的输入，从Data传入的block | 内部申请，作为图的输出
    kNorm,  // 内部申请
    kEnd
  };

  BlockAllocType() : BlockAllocType(kEnd) {}
  BlockAllocType(MainType main_type) : BlockAllocType(main_type, 0) {}

  BlockAllocType(MainType main_type, uint8_t sub_type)
      : main_type_{static_cast<uint8_t>(main_type)}, sub_type_{sub_type}, reserved_{0} {}

  MainType GetType() const {
    return static_cast<MainType>(main_type_);
  }
  void SetMainType(MainType mt) {
    main_type_ = static_cast<uint8_t>(mt);
  }

  uint8_t GetSubType() const {
    return sub_type_;
  }
  void SetSubType(uint8_t st) {
    sub_type_ = st;
  }

  void SetType(MainType mt, uint8_t st) {
    SetMainType(mt);
    SetSubType(st);
  }

 private:
  uint8_t main_type_;
  uint8_t sub_type_;
  uint8_t reserved_[6];
};
}  // namespace memory
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_KERNEL_MEMORY_BLOCK_ALLOC_TYPE_H_
