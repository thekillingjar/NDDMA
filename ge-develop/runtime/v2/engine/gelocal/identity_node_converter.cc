/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/checker.h"
#include "exe_graph/lowering/value_holder.h"
#include "graph/node.h"
#include "graph_builder/bg_infer_shape.h"
#include "graph_builder/bg_memory.h"
#include "graph_builder/converter_checker.h"
#include "kernel/common_kernel_impl/build_tensor.h"
#include "register/node_converter_registry.h"
#include "runtime/mem.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
bg::DevMemValueHolderPtr CopyTensor2D(LoweringGlobalData *global_data, const ge::NodePtr &node,
                                      const bg::ValueHolderPtr &src_addr, const bg::ValueHolderPtr &src_shape,
                                      const bg::ValueHolderPtr &data_type, const bg::ValueHolderPtr &tensor_size) {
  auto allocator = global_data->GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  auto dst_address = bg::DevMemValueHolder::CreateSingleDataOutput(
      "MakeSureTensorAtDevice", {global_data->GetStream(), allocator, src_addr, tensor_size, src_shape, data_type},
      node->GetOpDescBarePtr()->GetStreamId());
  GE_ASSERT_NOTNULL(dst_address);
  dst_address->SetPlacement(kOnDeviceHbm);
  GE_ASSERT_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", dst_address, {}));
  return dst_address;
}

bg::DevMemValueHolderPtr CopyTensorD2D(LoweringGlobalData *global_data, const ge::NodePtr &node,
                                       const bg::ValueHolderPtr &src_addr, const bg::ValueHolderPtr &tensor_size,
                                       LowerResult &ret) {
  auto allocator = global_data->GetOrCreateAllocator({kOnDeviceHbm, AllocatorUsage::kAllocNodeOutput});
  auto dst_addr = bg::DevMemValueHolder::CreateSingleDataOutput("AllocMemHbm", {allocator, tensor_size},
                                                                node->GetOpDescBarePtr()->GetStreamId());
  GE_ASSERT_NOTNULL(dst_addr);
  dst_addr->SetPlacement(kOnDeviceHbm);
  GE_ASSERT_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", dst_addr, {}));
  auto d2d_holder = bg::ValueHolder::CreateVoid<bg::ValueHolder>(
      "CopyD2D", {src_addr, dst_addr, tensor_size, global_data->GetStream()});
  ret.order_holders.emplace_back(d2d_holder);
  return dst_addr;
}

bg::DevMemValueHolderPtr CopyTensorH2H(const ge::NodePtr &node, const bg::ValueHolderPtr &src_addr,
                                       const bg::ValueHolderPtr &src_shape, const bg::ValueHolderPtr &data_type,
                                       const bg::ValueHolderPtr &allocator) {
  auto dst_address = bg::DevMemValueHolder::CreateSingleDataOutput(
      "CopyTensorDataH2H", {src_addr, src_shape, data_type, allocator}, node->GetOpDescBarePtr()->GetStreamId());
  if (dst_address == nullptr) {
    return nullptr;
  }
  dst_address->SetPlacement(kOnHost);
  GE_ASSERT_NOTNULL(bg::ValueHolder::CreateVoidGuarder("FreeMemory", dst_address, {}));
  return dst_address;
}

bg::DevMemValueHolderPtr CopyTensor(const ge::NodePtr &node, const LowerInput &input, const size_t index,
                                    LowerResult &ret) {
  GE_ASSERT_TRUE(index < input.input_addrs.size());
  GE_ASSERT_TRUE(index < input.input_shapes.size());
  GE_ASSERT_TRUE(index < ret.out_shapes.size());
  bg::DevMemValueHolderPtr dst_addr;
  auto src_addr = input.input_addrs[index];
  auto src_shape = input.input_shapes[index];
  auto out_shape = ret.out_shapes[index];
  GE_ASSERT_NOTNULL(node);
  const auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  if (TensorPlacementUtils::IsOnDevice(static_cast<TensorPlacement>(src_addr->GetPlacement()))) {
    auto tensor_size = bg::CalcOutTensorSize(node, static_cast<int32_t>(index), out_shape);
    dst_addr = CopyTensorD2D(input.global_data, node, src_addr, tensor_size, ret);
  } else if (TensorPlacementUtils::IsOnHost(static_cast<TensorPlacement>(src_addr->GetPlacement()))) {
    const auto &output_desc = op_desc->GetOutputDescPtr(static_cast<uint32_t>(index));
    GE_ASSERT_NOTNULL(output_desc);
    auto data_type = output_desc->GetDataType();
    auto data_type_holder = bg::ValueHolder::CreateConst(&data_type, sizeof(data_type));

    // todo：此处是临时方案，直接从init graph中获取allocator
    // 正式方案应该在cem pass中判断，如果一个节点除allocator之外全部是ce nodes，考虑移动到init graph
    auto init_allocator = bg::FrameSelector::OnInitRoot([&input]() -> std::vector<bg::ValueHolderPtr> {
      auto allocator_holder = input.global_data->GetOrCreateAllocator({kOnHost, AllocatorUsage::kAllocNodeWorkspace});
      return {allocator_holder};
    });
    dst_addr = CopyTensorH2H(node, src_addr, out_shape, data_type_holder, init_allocator[0]);
    ret.order_holders.emplace_back(dst_addr);
  } else {
    const auto &input_desc = op_desc->GetInputDescPtr(static_cast<uint32_t>(index));
    GE_ASSERT_NOTNULL(input_desc);
    auto src_data_type = input_desc->GetDataType();
    auto data_type_holder = bg::ValueHolder::CreateConst(&src_data_type, sizeof(src_data_type));
    auto tensor_size = bg::CalcOutTensorSize(node, index, out_shape);
    dst_addr = CopyTensor2D(input.global_data, node, src_addr, src_shape, data_type_holder, tensor_size);
    ret.order_holders.emplace_back(dst_addr);
  }
  return dst_addr;
}

LowerResult LoweringIdentityLikeNode(const ge::NodePtr &node, const LowerInput &input) {
  LowerResult ret;
  if (node->GetType() == "MemcpyAddrAsync") {
    ret.out_shapes = input.input_shapes;
  } else {
    auto out_shapes = bg::InferStorageShape(node, input.input_shapes, *(input.global_data));
    if (out_shapes.size() != input.input_shapes.size()) {
      GELOGE(ge::GRAPH_FAILED, "Output shapes size %zu is not equal with input shapes size %zu.", out_shapes.size(),
             input.input_shapes.size());
      return CreateErrorLowerResult("Output shapes size %zu is not equal with input shapes size %zu.",
                                    out_shapes.size(), input.input_shapes.size());
    }
    ret.out_shapes = out_shapes;
  }

  for (size_t i = 0U; i < input.input_addrs.size(); ++i) {
    auto out_tensor_data = CopyTensor(node, input, i, ret);
    LOWER_REQUIRE_NOTNULL(out_tensor_data, "Fail to lower CopyTensor for node %s", node->GetNamePtr());
    ret.out_addrs.emplace_back(out_tensor_data);
  }
  ret.result = HyperStatus::Success();
  return ret;
}
REGISTER_NODE_CONVERTER("Identity", LoweringIdentityLikeNode);
REGISTER_NODE_CONVERTER("IdentityN", LoweringIdentityLikeNode);
REGISTER_NODE_CONVERTER("ReadVariableOp", LoweringIdentityLikeNode);
REGISTER_NODE_CONVERTER("MemcpyAsync", LoweringIdentityLikeNode);
// 在rt1中图模式没有支持MemcpyAddrAsync，单算子支持了MemcpyAddrAsync的下发，但实际执行是走了MemcpyAsync
// 所以这里复用identity的lowering
// 实际上memcpy add async的能力主要由静态子图承载。单算子就走memcpy async。
REGISTER_NODE_CONVERTER("MemcpyAddrAsync", LoweringIdentityLikeNode);
}  // namespace gert
