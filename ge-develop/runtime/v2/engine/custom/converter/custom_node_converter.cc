/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_node_converter.h"
#include "graph_builder/converter_checker.h"
#include "graph_builder/bg_tensor.h"
#include "common/checker.h"
#include "exe_graph/lowering/frame_selector.h"
#include "kernel/common_kernel_impl/build_tensor.h"
#include "graph/custom_op_factory.h"
#include "lowering/placement/placed_lowering_result.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "common/ge_common/ge_types.h"

namespace gert {
namespace {
bg::ValueHolderPtr FindCustomExecutorFunc(const ge::NodePtr &node, const LowerInput &lower_input) {
  auto builder = [&node]() -> std::vector<bg::ValueHolderPtr> {
    return bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto node_type = bg::ValueHolder::CreateConst(node->GetTypePtr(), node->GetType().size() + 1, true);
      return {bg::ValueHolder::CreateSingleDataOutput("FindCustomOp", {node_type})};
    });
  };
  return lower_input.global_data->GetOrCreateUniqueValueHolder(node->GetType() + "_FindCustomOp_", builder)[0];
}

ge::graphStatus BuildInputTensors(const ge::NodePtr &node, const LowerInput &lower_input,
  std::vector<bg::ValueHolderPtr> &input_tensor_holders, std::vector<bg::ValueHolderPtr> &input_addr_holders) {
  const ge::OpDescPtr op_desc = node->GetOpDesc();
  for (const ge::InDataAnchorPtr &in_data_anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(in_data_anchor);
    // optional场景
    ge::OutDataAnchorPtr out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    if (out_data_anchor == nullptr) {
      continue;
    }
    ge::NodePtr peer_node = out_data_anchor->GetOwnerNode();
    GE_ASSERT_NOTNULL(peer_node);
    const auto *const_lower_result = peer_node->GetOpDesc()->GetExtAttr<PlacedLoweringResult>(kLoweringResult);
    GE_ASSERT_NOTNULL(const_lower_result, "Lowering result of node [%s, %s] is not found.", peer_node->GetNamePtr(),
        peer_node->GetTypePtr());
    auto *lower_result = const_cast<PlacedLoweringResult *>(const_lower_result);
    GE_ASSERT_NOTNULL(lower_result);
    // remove last true
    const OutputLowerResult *result = lower_result->GetOutputTensorResult(
        *lower_input.global_data, out_data_anchor->GetIdx(), {kOnDeviceHbm, node->GetOpDesc()->GetStreamId()});
    GE_ASSERT_NOTNULL(result, "Lowering result of node [%s, %s] output[%d] is nullptr.",
        peer_node->GetNamePtr(), peer_node->GetTypePtr(), out_data_anchor->GetIdx());
    GE_ASSERT_NOTNULL(result->shape);
    input_tensor_holders.emplace_back(result->shape);
    input_addr_holders.emplace_back(result->address);
  }
  GE_ASSERT_TRUE(lower_input.input_shapes.size() == input_tensor_holders.size(),
      "Size[%zu] of input shapes and size[%zu] of input tensor is not same.",
      lower_input.input_shapes.size(), input_tensor_holders.size());
  return ge::SUCCESS;
}
}

LowerResult LoweringCustomNode(const ge::NodePtr &node, const LowerInput &lower_input) {
  LOWER_REQUIRE_HYPER_SUCCESS(CheckLowerInput(lower_input));
  std::vector<bg::ValueHolderPtr> input_holders;
  std::vector<bg::ValueHolderPtr> input_addr_holders;
  LOWER_REQUIRE_SUCCESS(BuildInputTensors(node, lower_input, input_holders, input_addr_holders));
  // Allocate
  auto allocator_holder =
      lower_input.global_data->GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  auto custom_executor_func = FindCustomExecutorFunc(node, lower_input);
  // Create op executeFunc
  input_holders.emplace_back(allocator_holder);
  input_holders.emplace_back(lower_input.global_data->GetStream());
  input_holders.emplace_back(custom_executor_func);
  // 最后需要一个workspace地址
  std::vector<bg::ValueHolderPtr> output_tensor_holders =
      bg::ValueHolder::CreateDataOutput("ExecuteCustomOp",
      input_holders, node->GetAllOutDataAnchorsSize() + 1);
  auto op_desc = node->GetOpDesc();
  LOWER_REQUIRE_NOTNULL(op_desc);
  std::vector<bg::ValueHolderPtr> output_shapes;
  std::vector<bg::DevMemValueHolderPtr> output_addrs;
  for (size_t i = 0UL; i < node->GetAllOutDataAnchorsSize(); i++) {
    auto split_outputs = bg::DevMemValueHolder::CreateDataOutput(kernel::kSplitDataTensor,
        {output_tensor_holders[i], allocator_holder}, static_cast<size_t>(kernel::SplitTensorOutputs::kNum),
        op_desc->GetStreamId());
    CONVERTER_CHECK_HOLDERS_ALL_OK(split_outputs, static_cast<size_t>(kernel::SplitTensorOutputs::kNum));
    LOWER_REQUIRE_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory",
        split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)], {}));
    output_shapes.emplace_back(split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kShape)]);
    output_addrs.emplace_back(split_outputs[static_cast<size_t>(kernel::SplitTensorOutputs::kTensorData)]);
  }
  LOWER_REQUIRE_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeCustomOpWorkspaces",
      output_tensor_holders.back(), {allocator_holder}));
  // 输入tensor需要添加对地址guard的依赖边，否则会出现提前释放
  for (auto &addr : input_addr_holders) {
    auto guarder = addr->GetGuarder();
    if (guarder != nullptr) {
      GE_ASSERT_HYPER_SUCCESS(bg::ValueHolder::AddDependency(output_tensor_holders.front(), guarder));
    }
  }
  return {HyperStatus::Success(), {}, output_shapes, output_addrs};
}
REGISTER_NODE_CONVERTER_PLACEMENT(ge::kCustomOpKernelLibName.c_str(), kOnDeviceHbm, LoweringCustomNode);
}