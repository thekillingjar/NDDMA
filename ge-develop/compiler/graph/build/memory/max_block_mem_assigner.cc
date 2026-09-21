/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/build/memory/max_block_mem_assigner.h"
#include <algorithm>
#include "common/checker.h"

namespace ge {
Status MaxBlockMemAssigner::GetMemoryRanges(std::vector<int64_t> &ranges) {
  std::vector<int64_t> all_memory_size;

  GE_ASSERT_SUCCESS(GetOutAndWorkSpaceMem(all_memory_size));

  auto it = std::max_element(std::begin(all_memory_size), std::end(all_memory_size));
  if (it != std::end(all_memory_size)) {
    ranges.emplace_back(*it);
  }
  return SUCCESS;
}
}  // namespace ge
