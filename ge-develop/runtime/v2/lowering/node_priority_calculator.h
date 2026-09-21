/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_NODE_PRIORITY_CALCULATOR_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_NODE_PRIORITY_CALCULATOR_H_
#include "graph/compute_graph.h"
#include "graph/fast_graph/execute_graph.h"
#include "exe_graph/lowering/graph_frame.h"
namespace gert {
namespace bg {
class NodePriorityCalculator {
 public:
  explicit NodePriorityCalculator(const GraphFrame &frame);
  ge::graphStatus CalcNodeExecutionPriorities(const std::vector<ge::FastNode *> &main_graph_nodes,
                                              const size_t root_all_nodes_cnt);

 private:
  const GraphFrame &frame_;
};
}  // namespace bg
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_LOWERING_NODE_PRIORITY_CALCULATOR_H_
