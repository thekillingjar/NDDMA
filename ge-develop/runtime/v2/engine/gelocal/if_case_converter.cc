/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/ge_inner_attrs.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/bg_condition.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/value_holder_generator.h"
#include "lowering/graph_converter.h"
#include "register/node_converter_registry.h"

namespace gert {
namespace {
std::vector<bg::ValueHolderPtr> BuildSubgraph(const ge::NodePtr &node, int32_t subgraph_index,
                                              const LowerInput &parent_lower_input) {
  auto subgraph = ge::NodeUtils::GetSubgraph(*node, subgraph_index);
  if (subgraph == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Failed to get subgraph index %d from %s node %s", subgraph_index,
           node->GetType().c_str(), node->GetName().c_str());
    return {};
  }
  GELOGD("Start lower subgraph %s of %s %s", subgraph->GetName().c_str(), node->GetNamePtr(), node->GetTypePtr());
  auto &global_data = *parent_lower_input.global_data;
  std::vector<int32_t> parent_inputs_placement;
  for (auto &parent_input_addr : parent_lower_input.input_addrs) {
    GELOGD("Node %s subgraph %s input %zu from %d", node->GetNamePtr(), subgraph->GetName().c_str(),
           parent_inputs_placement.size(), parent_input_addr->GetPlacement());
    parent_inputs_placement.emplace_back(parent_input_addr->GetPlacement());
  }
  auto lower_result = ConvertComputeSubgraphToExecuteGraph(subgraph, global_data, 1, parent_inputs_placement);
  if (lower_result == nullptr) {
    GELOGE(ge::FAILED, "Failed to lowering subgraph index %d for %s node %s, return nullptr", subgraph_index,
           node->GetType().c_str(), node->GetName().c_str());
    return {};
  }
  if (!lower_result->result.IsSuccess()) {
    GELOGE(ge::FAILED, "Failed to lowering subgraph index %d for %s node %s, reason %s", subgraph_index,
           node->GetType().c_str(), node->GetName().c_str(), lower_result->result.GetErrorMessage());
    return {};
  }
  std::vector<bg::ValueHolderPtr> result = lower_result->out_shapes;
  result.insert(result.cend(), lower_result->out_addrs.cbegin(), lower_result->out_addrs.cend());
  return result;
}
LowerResult CheckOutputsAndReturn(const ge::NodePtr &node, const vector<bg::DevMemValueHolderPtr> &outputs) {
  if (outputs.empty()) {
    return CreateErrorLowerResult("Failed to lowering %s node %s, empty outputs", node->GetType().c_str(),
                                  node->GetName().c_str());
  }

  auto data_out_count = static_cast<size_t>(node->GetAllOutDataAnchorsSize());
  if (data_out_count == 0UL) {
    if (outputs.size() != 1UL) {
      return CreateErrorLowerResult("Failed to lowering %s node %s, the output count %zu, expect only one output",
                                    node->GetTypePtr(), node->GetNamePtr(), outputs.size());
    }
    std::vector<bg::ValueHolderPtr> order_holders(outputs.cbegin(), outputs.cend());
    return {HyperStatus::Success(), order_holders, {}, {}};
  } else {
    if (outputs.size() != data_out_count * 2UL) {
      return CreateErrorLowerResult(
          "Failed to lowering %s node %s, the output count %zu not match with 2 times of if output count %u",
          node->GetType().c_str(), node->GetName().c_str(), outputs.size(), data_out_count);
    }
    auto addr_start = outputs.cbegin() + static_cast<int64_t>(data_out_count);
    for (auto iter = addr_start; iter != outputs.cend(); iter++) {
      bg::ValueHolder::CreateVoidGuarder("FreeMemory", *iter, {});
    }
    return {HyperStatus::Success(), {outputs[0U]}, {outputs.cbegin(), addr_start}, {addr_start, outputs.cend()}};
  }
}
}  // namespace
/*
 *                 If(ExeGraph)
 *               /    \
 *   GenIndexForIf    <all-input-shapes>, <all-input_addrs>, <other-inputs-using-in-subgraph(like stream...)>
 *       |
 *  input-shapes[0](Tensor)
 */
LowerResult LoweringIf(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGD("Start lowering node %s type %s", node->GetNamePtr(), node->GetTypePtr());
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  // todo create stream ValueHolder on root graph
  auto gen_index_for_if = bg::DevMemValueHolder::CreateSingleDataOutput(
      "GenIndexForIf", {lower_input.input_shapes[0]}, op_desc->GetStreamId());
  // the `LoweringSubgraphInput` use the attr to determine the starting index of InnerData
  LOWER_REQUIRE(ge::AttrUtils::SetInt(op_desc, "OuterStartIndex", 1),
                "Failed to lower node %s, add start index failed", node->GetNamePtr());

  std::vector<bg::ValueHolderPtr> if_inputs;
  if_inputs.emplace_back(gen_index_for_if);
  if_inputs.insert(if_inputs.cend(), lower_input.input_shapes.cbegin(), lower_input.input_shapes.cend());
  if_inputs.insert(if_inputs.cend(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());

  // On the If node, the 0th graph is then graph, and the 1st graph is else graph

  auto outputs =
      bg::Cond<bg::DevMemValueHolder>(if_inputs, "If", {ge::kThenGraph, ge::kElseGraph},
                                      {[&node, &lower_input]() { return BuildSubgraph(node, 0, lower_input); },
                                       [&node, &lower_input]() { return BuildSubgraph(node, 1, lower_input); }},
                                       op_desc->GetStreamId());

  return CheckOutputsAndReturn(node, outputs);
}

/*
 *                           Case
 *                          /    \
 *             GenIndexForCase   <all-input-shapes>, <all-input_addrs>, <other-inputs-using-in-subgraph(like stream...)>
 *              /            \
 *  input-shapes[0](Tensor)  branch-num(Const)
 */
LowerResult LoweringCase(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGD("Start lowering node %s type %s", node->GetNamePtr(), node->GetTypePtr());
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const uint32_t branch_num = static_cast<uint32_t>(op_desc->GetSubgraphInstanceNames().size());
  auto branch_num_holder = bg::ValueHolder::CreateConst(&branch_num, sizeof(branch_num));
  const int64_t stream_id = op_desc->GetStreamId();
  auto gen_index_for_case = bg::DevMemValueHolder::CreateSingleDataOutput(
      "GenIndexForCase", {lower_input.input_shapes[0U], branch_num_holder}, stream_id);
  // the `LoweringSubgraphInput` use the attr to determine the starting index of InnerData
  LOWER_REQUIRE(ge::AttrUtils::SetInt(op_desc, "OuterStartIndex", 1),
                "Failed to lower node %s, add start index failed", node->GetNamePtr());

  std::vector<bg::ValueHolderPtr> inputs;
  inputs.emplace_back(gen_index_for_case);
  inputs.insert(inputs.cend(), lower_input.input_shapes.cbegin(), lower_input.input_shapes.cend());
  inputs.insert(inputs.cend(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());

  std::vector<std::string> subgraph_names;
  std::vector<bg::SubgraphBuilder> builders;
  for (uint32_t i = 0U; i < branch_num; ++i) {
    auto subgraph = ge::NodeUtils::GetSubgraph(*node, i);
    std::stringstream ss;
    ss << ge::kRelativeBranch << '_' << i;
    subgraph_names.emplace_back(ss.str());
    builders.emplace_back(
        [&node, i, &lower_input]() { return BuildSubgraph(node, static_cast<int32_t>(i), lower_input); });
  }

  const auto &outputs = bg::Cond<bg::DevMemValueHolder>(inputs, "Case", subgraph_names, builders, stream_id);
  return CheckOutputsAndReturn(node, outputs);
}

REGISTER_NODE_CONVERTER("StatelessIf", LoweringIf);
REGISTER_NODE_CONVERTER("If", LoweringIf);
REGISTER_NODE_CONVERTER("_If", LoweringIf);
REGISTER_NODE_CONVERTER("Case", LoweringCase);
}  // namespace gert
