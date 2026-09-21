/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/node_converter_registry.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph_builder/bg_tensor.h"
#include "graph_builder/converter_checker.h"
#include "lowering/placement/placed_lowering_result.h"
#include "kernel/common_kernel_impl/memory_copy.h"

namespace gert {
LowerResult LoweringNetOutput(const ge::NodePtr &node, const LowerInput &lower_input) {
  (void)lower_input;
  auto output_datas = bg::ValueHolder::CreateDataOutput("OutputData", {}, node->GetAllInDataAnchorsSize());
  LowerResult ret = {HyperStatus::Success(), {}, output_datas, {}};
  size_t i = 0U;
  for (const auto &in_node_and_anchor : node->GetInDataNodesAndAnchors()) {
    // todo 使用lower_input中的shape和addr就好了，应该不用去前面的节点上取
    auto place_in_ret = in_node_and_anchor.first->GetOpDescBarePtr()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    LOWER_REQUIRE_NOTNULL(place_in_ret,
                          "Failed to lowering Netoutput, can not find the PlacedLowerResult from its input node %s",
                          in_node_and_anchor.first->GetNamePtr());
    auto in_ret = place_in_ret->GetResult();
    LOWER_REQUIRE_NOTNULL(in_ret, "Failed to lowering Netoutput, can not find the LowerResult from its input node %s",
                          in_node_and_anchor.first->GetNamePtr());

    if (!in_ret->result.IsSuccess()) {
      return CreateErrorLowerResult(
          "Failed to lowering Netoutput, can not find the LowerResult of input node %s is error",
          in_node_and_anchor.first->GetName().c_str());
    }
    // todo placement需要根据实际情况填入
    auto build_tensor_attr =
        bg::CreateBuildTensorAttr(in_node_and_anchor.first, in_node_and_anchor.second->GetIdx(), kOnDeviceHbm);

    LOWER_REQUIRE(output_datas.size() > i);
    auto ensure_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>(
        kernel::kEnsureTensorAtOutMemory, {lower_input.input_shapes[i],
                                           lower_input.input_addrs[i], build_tensor_attr,
                                           lower_input.global_data->GetStream(), output_datas[i]});
    ++i;
    if (ensure_holder == nullptr) {
      return CreateErrorLowerResult("Failed to lowering Netoutput, tensor is nullptr. input node: %s, index: %d",
                                    in_node_and_anchor.first->GetName().c_str(), in_node_and_anchor.second->GetIdx());
    }
    ret.order_holders.emplace_back(ensure_holder);
  }
  return ret;
}

REGISTER_NODE_CONVERTER("NetOutput", LoweringNetOutput);
}  // namespace gert
