/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_impl/data_flow_op_impl.h"
#include "register/node_converter_registry.h"
#include "register/kernel_registry.h"
#include "exe_graph/lowering/frame_selector.h"

namespace gert {
thread_local std::vector<std::string> stat_exec_nodes_count;
thread_local std::vector<void*> memory_holders;

REGISTER_NODE_CONVERTER("SequenceStub", [](const ge::NodePtr &node, const LowerInput &lower_input) {
  LowerResult ret;
  auto builder = [&]() -> bg::ValueHolderPtr {
    auto resource_holder = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      std::string name = "aicpu_resource";
      auto name_holder = bg::ValueHolder::CreateConst(name.c_str(), name.size(), true);
      auto create_container_holder = bg::ValueHolder::CreateSingleDataOutput("CreateTestStepContainer", {name_holder});
      bg::ValueHolder::CreateVoidGuarder("DestroyTestStepContainer", create_container_holder, {});
      auto clear_builder = [&]() -> bg::ValueHolderPtr {
        return bg::ValueHolder::CreateVoid<bg::ValueHolder>("ClearTestStepContainer", {create_container_holder});
      };
      auto clear_holder = bg::FrameSelector::OnMainRootLast(clear_builder);
      return {create_container_holder};
    });
    return resource_holder[0];
  };
  auto holder_0 = lower_input.global_data->GetOrCreateUniqueValueHolder("aicpu_container_0", builder);

  auto merged_holders = lower_input.input_shapes;
  merged_holders.insert(merged_holders.end(), lower_input.input_addrs.begin(), lower_input.input_addrs.end());
  merged_holders.emplace_back(holder_0);
  auto holders = bg::DevMemValueHolder::CreateDataOutput((node->GetType()).c_str(), merged_holders,
                                                         node->GetAllOutDataAnchorsSize() << 1,
                                                         node->GetOpDesc()->GetStreamId());
  ret.out_shapes.insert(ret.out_shapes.end(), holders.begin(), holders.begin() + node->GetAllOutDataAnchorsSize());
  ret.out_addrs.insert(ret.out_addrs.end(), holders.begin() + node->GetAllOutDataAnchorsSize(), holders.end());

  return ret;
});

ge::graphStatus SequenceStub(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SequenceOutputCreator(const ge::FastNode *node, KernelContext *context) {
  auto output_num = context->GetOutputNum();
  size_t output_index = 0;
  for (; output_index < output_num / 2; ++output_index) {
    context->GetOutput(output_index)->SetWithDefaultDeleter(new StorageShape());
  }
  for (; output_index < output_num; ++output_index) {
    size_t output_size = 128U;
    memory_holders.emplace_back(malloc(output_size));
    memset(memory_holders.back(), 0, output_size);
    context->GetOutput(output_index)->SetWithDefaultDeleter(
        new TensorData(memory_holders.back(), nullptr, 128U, kOnDeviceHbm));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateTestStepContainer(KernelContext *context) {
  (void)context;
  stat_exec_nodes_count.emplace_back("CreateTestStepContainer");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ClearTestStepContainer(KernelContext *context) {
  (void)context;
  stat_exec_nodes_count.emplace_back("ClearTestStepContainer");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus DestroyTestStepContainer(KernelContext *context) {
  (void)context;
  stat_exec_nodes_count.emplace_back("DestroyTestStepContainer");
  return ge::GRAPH_SUCCESS;
}

std::vector<std::string> GetStatExecNodesCount() {
  return stat_exec_nodes_count;
}

void FreeDataFlowMemory() {
  for (auto memory_holder : memory_holders) {
    free(memory_holder);
  }
  stat_exec_nodes_count.clear();
  memory_holders.clear();
}

REGISTER_KERNEL(SequenceStub).RunFunc(SequenceStub).OutputsCreator(SequenceOutputCreator);
REGISTER_KERNEL(CreateTestStepContainer).RunFunc(CreateTestStepContainer);
REGISTER_KERNEL(ClearTestStepContainer).RunFunc(ClearTestStepContainer);
REGISTER_KERNEL(DestroyTestStepContainer).RunFunc(DestroyTestStepContainer);
}  // namespace gert
