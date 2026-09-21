/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/mem_allocator.h"
#include "exe_graph/runtime/tensor_data.h"
#include "exe_graph/runtime/allocator.h"
#include "common/checker.h"
#include "runtime/gert_api.h"

namespace gert {
ge::Allocator *Allocators::GetAllocator(const TensorPlacement &placement, const size_t &usage) {
  if ((placement < kTensorPlacementEnd) && (usage < static_cast<size_t>(AllocatorUsage::kEnd))) {
    return allocators[static_cast<size_t>(placement)][usage].get();
  } else {
    GELOGE(ge::FAILED, "unsupport placement %d or unsupport usage %zu", placement, usage);
    return nullptr;
  }
}
ge::graphStatus Allocators::SetAllocator(const TensorPlacement &placement, const size_t &usage,
                                         std::shared_ptr<ge::Allocator> &allocator) {
  GE_ASSERT_NOTNULL(allocator);
  if ((placement < kTensorPlacementEnd) && (usage < static_cast<size_t>(AllocatorUsage::kEnd))) {
    for (size_t i = 0U; i < static_cast<size_t>(AllocatorUsage::kEnd); ++i) {
      allocators[static_cast<size_t>(placement)][i] = allocator;
    }
    return ge::SUCCESS;
  } else {
    GELOGE(ge::FAILED,
           "Unsupportd placement %zu or unsupportd usage %zu, Only support placemen[%zu~%zu)",
           static_cast<size_t>(placement), usage, static_cast<size_t>(kOnDeviceHbm),
           static_cast<size_t>(AllocatorUsage::kEnd));
    return ge::FAILED;
  }
}
} // namespace gert
