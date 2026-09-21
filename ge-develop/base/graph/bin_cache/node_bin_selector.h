/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_H_
#include "graph/node.h"
#include "graph/bin_cache/node_compile_cache_module.h"
#include "ge/ge_api_types.h"
#include "graph/ge_local_context.h"

namespace ge {
class NodeBinSelector {
 public:
  /**
   * select a bin for node execution
   * @param node
   * @return
   */
  virtual NodeCompileCacheItem *SelectBin(const NodePtr &node, const GEThreadLocalContext *ge_context,
                                          std::vector<domi::TaskDef> &task_defs) = 0;
  virtual Status Initialize() = 0;
  NodeBinSelector() = default;
  virtual ~NodeBinSelector() = default;

 protected:
  NodeBinSelector(const NodeBinSelector&) = delete;
  NodeBinSelector &operator=(const NodeBinSelector &) & = delete;
};
}  // namespace ge

#endif // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_NODE_BIN_SELECTOR_H_
