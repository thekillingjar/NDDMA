/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_COMMON_NODE_OPTIMIZE_UTILS_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_COMMON_NODE_OPTIMIZE_UTILS_H_

#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"

namespace fe {
class NodeOptimizeUtils {
 public:
  static Status GetPreNode(const ge::NodePtr &node_ptr, const int &index, ge::NodePtr &pre_node_ptr);
};
}
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_NODE_OPTIMIZE_COMMON_NODE_OPTIMIZE_UTILS_H_
