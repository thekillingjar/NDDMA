/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_FE_GRAPH_COMMON_H_
#define FUSION_ENGINE_INC_COMMON_FE_GRAPH_COMMON_H_

#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
namespace fe {
class FeGraphCommon {
 public:
  FeGraphCommon() = default;
  ~FeGraphCommon() = default;
  static bool IsNodeOfUnknownRootGraph(const ge::Node &node);
  static bool IsNodeOfUnknownGraph(const ge::Node &node);
  static bool IsUnknownGraph(const ge::ComputeGraphPtr &graph_ptr);
};
} // namespace fe

#endif  // FUSION_ENGINE_INC_COMMON_FE_GRAPH_COMMON_H_
