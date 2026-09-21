/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RTS_ENGINE_GRAPH_OPTIMIZER_RTS_GRAPH_OPTIMIZER_H
#define RTS_ENGINE_GRAPH_OPTIMIZER_RTS_GRAPH_OPTIMIZER_H

#include "common/optimizer/graph_optimizer.h"
#include "common/optimizer/graph_optimizer_types.h"
#include "graph/compute_graph.h"

namespace cce {
namespace runtime {
const string RTS_GRAPH_OPTIMIZER_NAME = "rts_graph_optimizer";
constexpr std::int64_t RTS_FORMAT_PAIRED_INPUT_OUTPUT = 2;

class RtsGraphOptimizer : public ge::GraphOptimizer {
 public:
  RtsGraphOptimizer();

  ~RtsGraphOptimizer() override;

  ge::Status Initialize(const map<std::string, std::string> &options,
                        ge::OptimizeUtility *const optimizeUtility) override;

  ge::Status Finalize() override;

  ge::Status OptimizeGraphPrepare(ge::ComputeGraph &graph) override;

  ge::Status OptimizeWholeGraph(ge::ComputeGraph &graph) override;

  ge::Status GetAttributes(ge::GraphOptimizerAttribute &attrs) const override;

  ge::Status OptimizeOriginalGraph(ge::ComputeGraph &graph) override;

  ge::Status OptimizeFusedGraph(ge::ComputeGraph &graph) override;

  ge::Status OptimizeGraphBeforeBuild(ge::ComputeGraph &graph) override;

 private:
  rtError_t CheckSupportedOP(const std::string &sCollectiveType);

  ge::Status PorcMemtypeRange(ge::ComputeGraph &graph);

  ge::Status ProcConditionNode(ge::ComputeGraph &graph);

  rtError_t CheckConditionOPAndGetInputNodeNum(const std::string &sCollectiveType, uint32_t *inputNum);

  ge::Status SetMemTypeRange(const ge::NodePtr &node);

  ge::Status SetMemTypeRange(ge::ComputeGraph &graph);

  ge::Status InsertMemcpyAsyncNodeFunc(const ge::NodePtr &nexNode, ge::NodePtr &memcpyAsyncNode,
                                       ge::OpDescPtr &memcpyAsyncOpDesc, uint32_t index);

  ge::Status InsertMemcpyAsyncNodeAndSetMemType(const ge::NodePtr &nexNode, ge::ComputeGraph &graph,
                                                uint32_t inputNodeNum);

  ge::OpDescPtr CreateMemcpyAsyncOpByIndex(const ge::NodePtr &nexNode, uint32_t index);

  ge::Status SetMemTypeInputRange(const ge::NodePtr &node, uint32_t index);

  ge::Status SetMemTypeOutputRange(const ge::NodePtr &node, uint32_t index);
};
}  // namespace runtime
}  // namespace cce

#endif  // RTS_ENGINE_GRAPH_OPTIMIZER_RTS_GRAPH_OPTIMIZER_H
