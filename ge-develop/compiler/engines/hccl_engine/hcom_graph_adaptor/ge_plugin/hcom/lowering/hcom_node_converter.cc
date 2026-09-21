/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcom_node_converter.h"
#include "hcom_build_graph.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/lowering/generate_exe_graph.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph/utils/op_desc_utils.h"
#include "hccl/hcom.h"

namespace hccl {
gert::LowerResult LoweringHcomNode(const ge::NodePtr &node, const gert::LowerInput &lowerInput) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return {gert::HyperStatus::ErrorStatus(static_cast<const char *>("lowering hcom node failed, node is null.")),
            {},
            {},
            {}};
  }
  HCCL_RUN_INFO("lowering for node:%s(%s)", node->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
  // 1. const args
  auto hcomOpArgsHolder = GenerateHcomOpArgs(node);
  if (hcomOpArgsHolder == nullptr) {
    return {
        gert::HyperStatus::ErrorStatus(static_cast<const char *>("generate hcom op args node failed.")), {}, {}, {}};
  }

  // broadcast output使用input的addrs和shape，因此不需要推导，直接使用input
  if (node->GetOpDesc()->GetType() == HCCL_KERNEL_OP_TYPE_BROADCAST ||
      node->GetOpDesc()->GetType() == HCCL_KERNEL_OP_TYPE_SEND) {
    auto launchHolder =
        LaunchHcomOpKernel({hcomOpArgsHolder, lowerInput.global_data->GetStream()}, lowerInput.input_addrs,
                           lowerInput.input_addrs, lowerInput.input_shapes, lowerInput.input_shapes);
    return {gert::HyperStatus::Success(), launchHolder, lowerInput.input_shapes, lowerInput.input_addrs};
  }

  // 2. infershape
  auto outputShapes = gert::bg::GenerateExeGraph::InferShape(node, lowerInput.input_shapes, *(lowerInput.global_data));

  // 3. alloc output memory
  auto outputSizes = gert::bg::GenerateExeGraph::CalcTensorSize(node, outputShapes);

  auto outputAddrs =
      gert::bg::GenerateExeGraph::AllocOutputMemory(gert::kOnDeviceHbm, node, outputSizes, *(lowerInput.global_data));

  // 4. launch kernel
  auto launchHolder = LaunchHcomOpKernel({hcomOpArgsHolder, lowerInput.global_data->GetStream()},
                                         lowerInput.input_addrs, outputAddrs, lowerInput.input_shapes, outputShapes);

  auto hcomInitCommHolder = gert::bg::FrameSelector::OnInitRoot([]() -> std::vector<gert::bg::ValueHolderPtr> {
    const auto holder = gert::bg::ValueHolder::CreateSingleDataOutput("LaunchHcomKernelInitComm", {});
    return {holder};
  });

  return {gert::HyperStatus::Success(), launchHolder, outputShapes, outputAddrs};
}
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_BROADCAST.c_str(), gert::kOnDeviceHbm, LoweringHcomNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_ALLREDUCE.c_str(), gert::kOnDeviceHbm, LoweringHcomNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_ALLGATHER.c_str(), gert::kOnDeviceHbm, LoweringHcomNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_REDUCESCATTER.c_str(), gert::kOnDeviceHbm, LoweringHcomNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_SEND.c_str(), gert::kOnDeviceHbm, LoweringHcomNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_REDUCE.c_str(), gert::kOnDeviceHbm, LoweringHcomNode);

gert::LowerResult LoweringAlltoAllNode(const ge::NodePtr &node, const gert::LowerInput &lowerInput) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return {gert::HyperStatus::ErrorStatus(static_cast<const char *>("lowering hcom node failed, node is null.")),
            {},
            {},
            {}};
  }
  HCCL_RUN_INFO("lowering for node:%s(%s)", node->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
  // 1. const args
  auto hcomOpArgsHolder = GenerateHcomOpArgs(node);
  if (hcomOpArgsHolder == nullptr) {
    return {
        gert::HyperStatus::ErrorStatus(static_cast<const char *>("generate hcom op args node failed.")), {}, {}, {}};
  }

  // 1.1 指定alltoallv中input的内存位置
  std::vector<gert::bg::DevMemValueHolderPtr> inputs_addr;
  inputs_addr = MakeSureCommAlltoAllInput(node, lowerInput);

  // 2. infershape
  auto outputShapes = gert::bg::GenerateExeGraph::InferShape(node, lowerInput.input_shapes, *(lowerInput.global_data));

  // 3. alloc output memory
  auto outputSizes = gert::bg::GenerateExeGraph::CalcTensorSize(node, outputShapes);
  auto outputAddrs =
      gert::bg::GenerateExeGraph::AllocOutputMemory(gert::kOnDeviceHbm, node, outputSizes, *(lowerInput.global_data));

  // 4. launch kernel
  auto launchHolder = LaunchHcomOpKernel({hcomOpArgsHolder, lowerInput.global_data->GetStream()}, inputs_addr,
                                         outputAddrs, lowerInput.input_shapes, outputShapes);
  // 为保证前序算子执行完成后才执行make sure at host，通信算子执行时输入数据正确性，
  // 需要将make sure at host加入返回的order holder中
  for (u32 i = 1; i < inputs_addr.size(); i++) {
    launchHolder.push_back(inputs_addr[i]);
  }

  auto hcomInitCommHolder = gert::bg::FrameSelector::OnInitRoot([]() -> std::vector<gert::bg::ValueHolderPtr> {
    const auto holder = gert::bg::ValueHolder::CreateSingleDataOutput("LaunchHcomKernelInitComm", {});
    return {holder};
  });
  return {gert::HyperStatus::Success(), launchHolder, outputShapes, outputAddrs};
}
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_ALLTOALLV.c_str(), gert::kOnDeviceHbm, LoweringAlltoAllNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_ALLTOALLVC.c_str(), gert::kOnDeviceHbm, LoweringAlltoAllNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_ALLTOALL.c_str(), gert::kOnDeviceHbm, LoweringAlltoAllNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_ALLGATHERV.c_str(), gert::kOnDeviceHbm, LoweringAlltoAllNode);
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_REDUCESCATTERV.c_str(), gert::kOnDeviceHbm, LoweringAlltoAllNode);

gert::LowerResult LoweringRecvNode(const ge::NodePtr &node, const gert::LowerInput &lowerInput) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    return {gert::HyperStatus::ErrorStatus(static_cast<const char *>("lowering hcom node failed, node is null.")),
            {},
            {},
            {}};
  }
  HCCL_RUN_INFO("lowering for node:%s(%s)", node->GetName().c_str(), node->GetOpDesc()->GetType().c_str());
  // 1. const args
  auto hcomOpArgsHolder = GenerateHcomOpArgs(node);
  if (hcomOpArgsHolder == nullptr) {
    return {
        gert::HyperStatus::ErrorStatus(static_cast<const char *>("generate hcom op args node failed.")), {}, {}, {}};
  }

  // launch kernel
  auto outPut = LaunchRecvOpKernel({hcomOpArgsHolder, lowerInput.global_data->GetStream()}, node, lowerInput);
  if (outPut.size() != LAUNCH_RECV_KERNEL_OUTPUT_SIZE) {
    return {gert::HyperStatus::ErrorStatus(static_cast<const char *>("excute LaunchRecvOpKernel failed.")), {}, {}, {}};
  }

  gert::bg::ValueHolder::CreateVoidGuarder("FreeMemory", outPut[1], {});

  return {gert::HyperStatus::Success(), {outPut[1]}, {outPut[0]}, {outPut[1]}};
}
REGISTER_NODE_CONVERTER_PLACEMENT(HCCL_KERNEL_OP_TYPE_RECEIVE.c_str(), gert::kOnDeviceHbm, LoweringRecvNode);

gert::bg::DevMemValueHolderPtr MakeSureInputAtHost(const ge::NodePtr &node, const gert::LowerInput &lowerInput,
                                                   u32 index) {
  gert::bg::ValueHolderPtr stream = lowerInput.global_data->GetStream();
  if (index >= lowerInput.input_shapes.size() || index >= lowerInput.input_addrs.size()) {
    HCCL_ERROR("Node[%s]: MakeSureInputAtHost failed, index:%u, expect < %u.", node->GetName().c_str(), index,
               lowerInput.input_shapes.size());
    return nullptr;
  }
  ge::GeTensorDesc inputTensor = node->GetOpDesc()->GetInputDesc(index);
  ge::DataType geDataType = inputTensor.GetDataType();
  auto dtype = gert::bg::ValueHolder::CreateConst(&geDataType, sizeof(geDataType));
  // NOTE: 以下两个 kernel 构图，需要GE后续封装API
  gert::bg::ValueHolderPtr input_addr = lowerInput.input_addrs[index];
  gert::bg::ValueHolderPtr size =
      gert::bg::ValueHolder::CreateSingleDataOutput("CalcTensorSizeFromShape", {dtype, lowerInput.input_shapes[index]});
  auto allocator =
      lowerInput.global_data->GetOrCreateAllocator({gert::kOnHost, gert::AllocatorUsage::kAllocNodeWorkspace});
  return gert::bg::DevMemValueHolder::CreateSingleDataOutput(
      "MakeSureTensorAtHost", {stream, allocator, input_addr, size}, node->GetOpDesc()->GetStreamId());
}

std::vector<gert::bg::DevMemValueHolderPtr> MakeSureCommAlltoAllInput(const ge::NodePtr &node,
                                                                      const gert::LowerInput &lower_input) {
  // alltoallc/alltoallvc 算子input[0]位于device内存, 其余input位于host内存
  gert::bg::ValueHolderPtr stream = lower_input.global_data->GetStream();
  size_t inputNum = lower_input.input_shapes.size();
  std::vector<gert::bg::DevMemValueHolderPtr> inputs(inputNum);
  inputs[0] = lower_input.input_addrs[0];
  for (u32 i = 1; i < inputs.size(); i++) {
    ge::GeTensorDesc inputTensor = node->GetOpDesc()->GetInputDesc(i);
    ge::DataType geDataType = inputTensor.GetDataType();
    auto data_holder = gert::bg::ValueHolder::CreateConst(&geDataType, sizeof(geDataType));
    // NOTE: 以下两个 kernel 构图，需要GE后续封装API
    gert::bg::ValueHolderPtr tensor_size = gert::bg::ValueHolder::CreateSingleDataOutput(
        "CalcTensorSizeFromShape", {data_holder, lower_input.input_shapes[i]});
    auto allocator =
        lower_input.global_data->GetOrCreateAllocator({gert::kOnHost, gert::AllocatorUsage::kAllocNodeWorkspace});
    inputs[i] = gert::bg::DevMemValueHolder::CreateSingleDataOutput(
        "MakeSureTensorAtHost", {stream, allocator, lower_input.input_addrs[i], tensor_size},
        node->GetOpDesc()->GetStreamId());
  }
  HCCL_INFO("[hcom][lowering]make sure tensor at current placement success.");
  return inputs;
}

}  // namespace hccl