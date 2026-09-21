/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_ONE_NODE_SINGLE_BIN_SELECTOR_H_
#define AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_ONE_NODE_SINGLE_BIN_SELECTOR_H_

#include "graph/bin_cache/node_bin_selector.h"
#include "ge/ge_api_error_codes.h"
#include "graph/bin_cache/node_bin_selector_factory.h"
namespace ge {
class OneNodeSingleBinSelector : public NodeBinSelector {
 public:
  OneNodeSingleBinSelector() = default;
  NodeCompileCacheItem *SelectBin(const NodePtr &node, const GEThreadLocalContext *ge_context,
                                  std::vector<domi::TaskDef> &task_defs) override;
  Status Initialize() override;

 private:
  NodeCompileCacheItem cci_;
};
}  // namespace ge

#endif  // AIR_CXX_EXECUTOR_HYBRID_COMMON_BIN_CACHE_ONE_NODE_SINGLE_BIN_SELECTOR_H_
