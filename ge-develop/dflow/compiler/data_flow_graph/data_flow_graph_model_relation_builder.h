/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_MODEL_RELATION_BUILDER_H
#define AIR_COMPILER_PNE_DATA_FLOW_GRAPH_DATA_FLOW_GRAPH_MODEL_RELATION_BUILDER_H

#include "dflow/base/model/model_relation.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph.h"

namespace ge {
class DataFlowGraphModelRelationBuilder : public ModelRelationBuilder {
 public:
  Status BuildFromDataFlowGraph(const DataFlowGraph &data_flow_graph, std::unique_ptr<ModelRelation> &model_relation);

 private:
  struct NodeFlowInfo {
    std::string name;
    int32_t index;
    std::string type;
  };
  Status DoBuildForFlowData(const DataFlowGraph &data_flow_graph, const NodePtr &node,
                            std::map<NodePtr, std::map<int, std::string>> &paired_inputs);
  Status DoBuildForNodeOutput(const DataFlowGraph &data_flow_graph,
                              const NodePtr &node,
                              const std::string &endpoint_name,
                              std::map<NodePtr, std::map<int, std::string>> &paired_inputs);
  Status DoBuildForFlowNode(const DataFlowGraph &data_flow_graph, const NodePtr &node,
                            std::map<NodePtr, std::map<int, std::string>> &paired_inputs);
  Status GetOrCreateModelQueueInfoForDataFlowGraph(const DataFlowGraph &data_flow_graph,
                                                   const NodeFlowInfo &node_flow_info,
                                                   ModelRelation::ModelEndpointInfo *&model_endpoint_info,
                                                   uint32_t &queue_index);
};
}  // namespace ge
#endif
