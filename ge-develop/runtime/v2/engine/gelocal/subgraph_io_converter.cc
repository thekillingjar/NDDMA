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
#include "graph_builder/converter_checker.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_builder/bg_identity.h"
#include "register/node_converter_registry.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/value_holder_utils.h"

namespace gert {
LowerResult LoweringSubgraphInput(const ge::NodePtr &node, const LowerInput &lower_input) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  int32_t start_index = 0;
  (void)ge::AttrUtils::GetInt(op_desc, "_inner_data_start_index", start_index);
  LowerResult ret;
  const auto owner_graph = node->GetOwnerComputeGraphBarePtr();
  LOWER_REQUIRE_NOTNULL(owner_graph, "Owner graph of node %s is nullptr", node->GetNamePtr());
  const auto parent_node = owner_graph->GetParentNodeBarePtr();
  LOWER_REQUIRE_NOTNULL(parent_node, "Parent node of subgraph %s is nullptr", owner_graph->GetName().c_str());
  auto num_inputs = parent_node->GetAllInDataAnchorsSize();
  int32_t tensor_index = 0;
  LOWER_REQUIRE(ge::AttrUtils::GetInt(op_desc, ge::ATTR_NAME_PARENT_NODE_INDEX, tensor_index),
                "Failed get attr '%s' from compute data node %s", ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(),
                node->GetNamePtr());
  auto shape_holder = bg::ValueHolder::CreateSingleDataOutput("InnerData", lower_input.input_shapes);
  LOWER_REQUIRE_VALID_HOLDER(shape_holder, "Failed create InnerData for cond shape holder");
  int32_t shape_index = tensor_index + start_index;
  LOWER_REQUIRE(
      ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(shape_holder), ge::ATTR_NAME_INDEX, shape_index),
      "Failed set attr 'index' to lower data node %s", bg::ValueHolderUtils::GetNodeNameBarePtr(shape_holder));
  LOWER_REQUIRE(shape_holder->AddInnerDataToKVMap(shape_index).IsSuccess());
  GELOGI("Get parent node index: %d of node: %s and set index: %d to shape holder: %s",
         tensor_index, node->GetNamePtr(), shape_index, bg::ValueHolderUtils::GetNodeNameBarePtr(shape_holder));
  ret.out_shapes.push_back(shape_holder);

  std::vector<bg::ValueHolderPtr> input_addr(lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  const auto &addr_holder =
      bg::DevMemValueHolder::CreateSingleDataOutput("InnerData", input_addr, op_desc->GetStreamId());
  LOWER_REQUIRE_VALID_HOLDER(addr_holder, "Failed create InnerData for cond value holder");
  int32_t addr_index = tensor_index + num_inputs + start_index;
  LOWER_REQUIRE(
      ge::AttrUtils::SetInt(bg::ValueHolderUtils::GetNodeOpDescBarePtr(addr_holder), ge::ATTR_NAME_INDEX, addr_index),
      "Failed set attr 'index' to lower data node %s", bg::ValueHolderUtils::GetNodeNameBarePtr(addr_holder));
  LOWER_REQUIRE(addr_holder->AddInnerDataToKVMap(addr_index).IsSuccess());
  GELOGI("Get parent node index: %d of node: %s and set index: %d to addr holder: %s", tensor_index, node->GetNamePtr(),
         addr_index, bg::ValueHolderUtils::GetNodeNameBarePtr(addr_holder));

  int32_t attr_placement = -1;
  if (ge::AttrUtils::GetInt(op_desc, "_placement", attr_placement)) {
    LOWER_REQUIRE((attr_placement >= -1) && (attr_placement <= static_cast<int>(TensorPlacement::kTensorPlacementEnd)));
    addr_holder->SetPlacement(attr_placement);
  }
  ret.out_addrs.push_back(addr_holder);
  return ret;
}

LowerResult LoweringSubgraphOutput(const ge::NodePtr &node, const LowerInput &lower_input) {
  const auto &identified_shapes = bg::IdentityShape(lower_input.input_shapes);
  const auto &identified_addrs = bg::IdentityAddr(lower_input.input_addrs, node->GetOpDescBarePtr()->GetStreamId());
  std::vector<bg::ValueHolderPtr> ordered_holders;
  if (!identified_addrs.empty()) {
    ordered_holders = {identified_shapes.front(), identified_addrs.front()};
  }
  return {HyperStatus::Success(), ordered_holders, identified_shapes, identified_addrs};
}

REGISTER_NODE_CONVERTER(ge::kSubgraphInput, LoweringSubgraphInput);
REGISTER_NODE_CONVERTER(ge::kSubgraphOutput, LoweringSubgraphOutput);
}  // namespace gert
