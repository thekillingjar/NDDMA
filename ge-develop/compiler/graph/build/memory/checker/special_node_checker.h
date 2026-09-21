/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_BUILD_MEMORY_CHECKER_SPECIAL_NODE_CHECKER_H_
#define GE_GRAPH_BUILD_MEMORY_CHECKER_SPECIAL_NODE_CHECKER_H_

#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "ge/ge_api_types.h"
#include "dependency_analyzer.h"

namespace ge {
class SpecialNodeChecker {
 public:
  // 内存分配完成后调用，用于检查节点offset是否满足一些算子特殊的内存要求
  static Status Check(const ComputeGraphPtr &graph);

  // 内存分配前调用，用于检查一些特殊节点上的属性是否满足约束
  static Status CheckBeforeAssign(const ComputeGraphPtr &graph);
};
} // namespace ge
#endif  // GE_GRAPH_BUILD_MEMORY_CHECKER_SPECIAL_NODE_CHECKER_H_
