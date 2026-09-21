/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_LOWERING_GRAPH_CONVERTER_H_
#define AIR_CXX_RUNTIME_V2_LOWERING_GRAPH_CONVERTER_H_
#include "exe_graph/lowering/lowering_global_data.h"
#include "register/node_converter_registry.h"
#include "runtime/gert_api.h"
#include "lowering/model_converter.h"
namespace gert {
const LowerResult *ConvertComputeSubgraphToExecuteGraph(const ge::ComputeGraphPtr &graph,
                                                        LoweringGlobalData &global_data, int32_t start_index,
                                                        const std::vector<int32_t> &parent_inputs_placement = {},
                                                        const std::vector<int32_t> &parent_outputs_placement = {});
struct OrderInputs {
  std::vector<bg::ValueHolderPtr> ordered_inputs_list;
  std::set<bg::ValueHolderPtr> ordered_inputs_set;
};
const LowerResult *LoweringNode(const ge::NodePtr &node, LowerInput &inputs,
                                const std::vector<bg::ValueHolderPtr> &ordered_inputs);
HyperStatus ConstructInputs(const ge::NodePtr &node, int32_t inputs_placement, LowerInput &lower_input,
                            OrderInputs &order_inputs);

const LowerResult *LoweringComputeGraph(const ge::ComputeGraphPtr &graph, LoweringGlobalData &global_data);

class GraphConverter {
 public:
  ge::ExecuteGraphPtr ConvertComputeGraphToExecuteGraph(const ge::ComputeGraphPtr &graph,
                                                        const LoweringOption &optimize_option,
                                                        LoweringGlobalData &global_data) const;

  ge::ExecuteGraphPtr ConvertComputeGraphToExecuteGraph(const ge::ComputeGraphPtr &graph,
                                                        LoweringGlobalData &global_data) const {
    return ConvertComputeGraphToExecuteGraph(graph, {}, global_data);
  }
  GraphConverter &SetModelDescHolder(ModelDescHolder *model_desc_holder) {
    model_desc_holder_ = model_desc_holder;
    return *this;
  }

 private:
  ge::graphStatus AppendGraphLevelData(const bg::GraphFrame &frame, const ge::ComputeGraphPtr &compute_graph,
                                       ge::ExecuteGraph *const execute_graph,
                                       const std::vector<ge::FastNode *> &root_graph_nodes) const;

 private:
  ModelDescHolder *model_desc_holder_ = nullptr;
};
}  // namespace gert

#endif  // AIR_CXX_RUNTIME_V2_LOWERING_GRAPH_CONVERTER_H_
