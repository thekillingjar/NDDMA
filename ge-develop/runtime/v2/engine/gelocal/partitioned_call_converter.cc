/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/lowering_definitions.h"
#include "framework/common/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/bg_identity.h"
#include "graph_builder/converter_checker.h"
#include "lowering/graph_converter.h"
#include "lowering/placement/placed_lowering_result.h"
#include "lowering/static_compiled_graph_converter.h"
#include "register/node_converter_registry.h"

namespace gert {
namespace {
constexpr const char *kLowerPartitionedCallOutput = "_lower_partitioned_call_output";
constexpr const char *kLowerPartitionedCallInput = "_lower_partitioned_call_input";

const OutputLowerResult *GetLowerResultFromParentNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  uint32_t index = 0U;
  GE_ASSERT_TRUE(ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::ATTR_NAME_PARENT_NODE_INDEX, index),
                 "node[%s] get %s failed", node->GetNamePtr(), ge::ATTR_NAME_PARENT_NODE_INDEX.c_str());
  const auto graph = node->GetOwnerComputeGraphBarePtr();
  GE_ASSERT_NOTNULL(graph);
  const auto parent_partitioned_call = graph->GetParentNodeBarePtr();
  GE_ASSERT_NOTNULL(parent_partitioned_call);

  const auto &in_anchor = parent_partitioned_call->GetInDataAnchor(static_cast<int32_t>(index));
  GE_ASSERT_NOTNULL(in_anchor);
  const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
  GE_ASSERT_NOTNULL(peer_out_anchor);

  const auto &peer_in_node = peer_out_anchor->GetOwnerNode();
  GE_ASSERT_NOTNULL(peer_in_node);
  const auto *const_lower_result = peer_in_node->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
  GE_ASSERT_NOTNULL(const_lower_result, "get lowering result failed. peer in node:%s, current inner data:%s",
                    peer_in_node->GetNamePtr(), node->GetNamePtr());
  auto *lower_result = const_cast<PlacedLoweringResult *>(const_lower_result);
  auto holder = bg::ValueHolder::SetScopedCurrentComputeNode(peer_in_node);
  auto result = lower_result->GetOutputResult(*lower_input.global_data, peer_out_anchor->GetIdx(),
                                              {kOnDeviceHbm, bg::kMainStream}, false);
  GE_ASSERT_NOTNULL(result);
  return result;
}
}  // namespace

LowerResult LoweringPartitionedCallOutput(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGI("Start lowering partitioned call output %s", node->GetNamePtr());
  LowerResult ret;
  ret.out_addrs.insert(ret.out_addrs.cbegin(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  ret.out_shapes.insert(ret.out_shapes.cbegin(), lower_input.input_shapes.cbegin(), lower_input.input_shapes.cend());
  return ret;
}

LowerResult LoweringPartitionedCallInput(const ge::NodePtr &node, const LowerInput &lower_input) {
  GELOGD("Start lowering partitioned call subgraph inner data %s", node->GetNamePtr());
  auto *result = GetLowerResultFromParentNode(node, lower_input);
  LOWER_REQUIRE_NOTNULL(result);
  return {HyperStatus::Success(), {}, {result->shape}, {result->address}};
}

LowerResult LoweringPartitionedCall(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto root_compute_graph = ge::GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  LOWER_REQUIRE_NOTNULL(root_compute_graph);
  const auto &graph = root_compute_graph->GetSubgraph(node->GetOpDescBarePtr()->GetSubgraphInstanceName(0U));
  LOWER_REQUIRE_NOTNULL(graph, "Root graph %s has no subgraph named %s", root_compute_graph->GetName().c_str(),
                        node->GetOpDescBarePtr()->GetSubgraphInstanceName(0U).c_str());
  for (auto direct_node : graph->GetDirectNodePtr()) {
    if (direct_node->GetType() == ge::DATA) {
      (void)ge::AttrUtils::SetStr(direct_node->GetOpDescBarePtr(), "_ge_attr_lowering_func",
                                  kLowerPartitionedCallInput);
      continue;
    }
    if (direct_node->GetType() == ge::NETOUTPUT) {
      (void)ge::AttrUtils::SetStr(direct_node->GetOpDescBarePtr(), "_ge_attr_lowering_func",
                                  kLowerPartitionedCallOutput);
    }
  }
  auto lower_ret = LoweringComputeGraph(graph, *lower_input.global_data);
  LOWER_REQUIRE_NOTNULL(lower_ret);
  return *lower_ret;
}

REGISTER_NODE_CONVERTER_PLACEMENT("PartitionedCall", kOnDeviceHbm, LoweringPartitionedCall);
REGISTER_NODE_CONVERTER(kLowerPartitionedCallOutput, LoweringPartitionedCallOutput);
REGISTER_NODE_CONVERTER(kLowerPartitionedCallInput, LoweringPartitionedCallInput);
}  // namespace gert
