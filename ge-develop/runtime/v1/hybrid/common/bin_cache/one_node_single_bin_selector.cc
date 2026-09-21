/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "one_node_single_bin_selector.h"

namespace ge {
// for non-fuzz-compile
NodeCompileCacheItem *OneNodeSingleBinSelector::SelectBin(const NodePtr &node, const GEThreadLocalContext *ge_context,
                                                          std::vector<domi::TaskDef> &task_defs) {
  (void)node;
  (void)ge_context;
  (void)task_defs;
  return &cci_;
}

Status OneNodeSingleBinSelector::Initialize() {
  return SUCCESS;
}

REGISTER_BIN_SELECTOR(fuzz_compile::kOneNodeSingleBinMode, OneNodeSingleBinSelector);
}  // namespace ge
