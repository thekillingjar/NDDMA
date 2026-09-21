/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/converter_checker.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph/utils/node_utils.h"
#include "graph_builder/bg_tensor.h"
#include "register/node_converter_registry.h"
#include "lowering/placement/placed_lowering_result.h"


namespace gert {
LowerResult LoweringConditionCalc(const ge::NodePtr &node, const LowerInput &lower_input) {
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  const std::string *cond_fn = ge::AttrUtils::GetStr(op_desc, "cond_func");
  LOWER_REQUIRE(cond_fn != nullptr, "Failed get attr 'cond_func' from node %s", node->GetNamePtr());
  auto cond_fn_name = bg::ValueHolder::CreateConst(cond_fn->c_str(), cond_fn->size() + 1, true);
  LOWER_REQUIRE_VALID_HOLDER(cond_fn_name, "Failed create lower cond fn node.");
  auto cond_func = bg::ValueHolder::CreateSingleDataOutput("FindKernelFunc", {cond_fn_name});
  LOWER_REQUIRE_VALID_HOLDER(cond_func, "Failed create lower find kernel func node.");

  auto merged_holders = lower_input.input_shapes;
  merged_holders.insert(merged_holders.cend(), lower_input.input_addrs.cbegin(), lower_input.input_addrs.cend());
  merged_holders.push_back(cond_func);  // Ends with kernel func

  std::vector<int64_t> value_depend_type;
  LOWER_REQUIRE(ge::AttrUtils::GetListInt(op_desc, "x_dependency", value_depend_type),
                "Failed get attr 'x_dependency' from node %s", node->GetNamePtr());
  LOWER_REQUIRE(value_depend_type.size() == node->GetAllInDataAnchorsSize(),
                "Attr 'x_dependency' size mismatch %zu vs expect %u", value_depend_type.size(),
                node->GetAllInDataAnchorsSize());

  const static int64_t kDependTensor = 1;
  for (size_t i = 0U; i < value_depend_type.size(); i++) {
    if (value_depend_type[i] != kDependTensor) {
      continue;
    }

    // todo metadef 添加两个接口: NodeUtils::GetInDataNodeAndAnchorByIndex, GetExtAttr提供非const输出
    const auto &dst_anchor = node->GetInDataAnchor(static_cast<int32_t>(i));
    LOWER_REQUIRE_NOTNULL(dst_anchor, "Failed to get in data anchor from index %zu for node %s", i, node->GetNamePtr());
    const auto &src_anchor = dst_anchor->GetPeerOutAnchor();
    LOWER_REQUIRE_NOTNULL(src_anchor, "Failed to get peer out data anchor from index %zu for node %s", i,
                          node->GetNamePtr());
    const auto src_node = src_anchor->GetOwnerNodeBarePtr();
    LOWER_REQUIRE_NOTNULL(src_node, "Failed to get in data node from index %zu for node %s", i, node->GetNamePtr());

    const auto *const_src_result = src_node->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    LOWER_REQUIRE_NOTNULL(const_src_result, "Failed to get lowering result from node %s when lowering node %s",
                          src_node->GetNamePtr(), node->GetNamePtr());
    auto src_result = const_cast<PlacedLoweringResult *>(const_src_result);
    auto host_result =
        src_result->GetOutputResult(*lower_input.global_data, src_anchor->GetIdx(), {kOnHost, bg::kMainStream}, true);
    LOWER_REQUIRE_NOTNULL(host_result,
                          "Failed to get output result from src node %s when lowering node %s, src anchor idx %zu",
                          src_node->GetNamePtr(), node->GetNamePtr(), src_anchor->GetIdx());
    LOWER_REQUIRE(i < merged_holders.size(), "index %zu >= merged_holders.size %zu", i, merged_holders.size());
    merged_holders[i] = host_result->shape;
  }

  auto cond_func_ret = bg::ValueHolder::CreateSingleDataOutput("ConditionCalc", merged_holders);
  LOWER_REQUIRE_VALID_HOLDER(cond_func_ret, "Failed to create lower run kernel func node.");
  int64_t logic_stream_id = 0;
  auto stream_id_holder = bg::ValueHolder::CreateConst(&logic_stream_id, sizeof(logic_stream_id));
  LOWER_REQUIRE_VALID_HOLDER(stream_id_holder, "Failed to create lower stream id");
  auto lower_tensor_data = bg::DevMemValueHolder::CreateSingleDataOutput(
      "BuildUnmanagedTensorData", {cond_func_ret, stream_id_holder}, op_desc->GetStreamId());
  LOWER_REQUIRE_VALID_HOLDER(lower_tensor_data, "Failed to create lower build tensor data node");
  lower_tensor_data->SetPlacement(kOnHost);

  const static auto kScalarRtShape = StorageShape();
  auto shape_node = bg::ValueHolder::CreateConst(&kScalarRtShape, sizeof(kScalarRtShape));
  LOWER_REQUIRE_VALID_HOLDER(shape_node, "Failed to create lower shape node");

  return {HyperStatus::Success(),
          {},  // no ordered holders
          {shape_node},
          {lower_tensor_data}};
}

REGISTER_NODE_CONVERTER("ConditionCalc", LoweringConditionCalc);
}  // namespace gert
