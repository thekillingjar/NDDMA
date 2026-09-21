/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/memory/sink_weight_assigner.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/kernel_log.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tensor.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"
#include "register/kernel_registry.h"
#include "runtime/model_v2_executor.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"

namespace gert {
namespace kernel {
ge::graphStatus CreateMemAssigner(KernelContext *context) {
  auto assinger_chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(assinger_chain);
  auto mem_base =
      context->GetInputPointer<gert::GertTensorData>(static_cast<size_t>(CreateMemAssignerInputs::kBaseMem));
  auto assinger = new (std::nothrow) memory::SinkWeightAssigner(mem_base->GetAddr(), mem_base->GetSize());
  GE_ASSERT_NOTNULL(assinger);
  assinger_chain->SetWithDefaultDeleter(assinger);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus AssignWeightMemory(KernelContext *context) {
  auto tensor = context->GetInputPointer<TensorData>(static_cast<size_t>(AssignWeightMemoryInputs::kOffsetTensor));
  auto assigner = context->MutableInputPointer<memory::SinkWeightAssigner>(
      static_cast<size_t>(AssignWeightMemoryInputs::kAssigner));
  auto device_gtd =
      context->GetOutputPointer<GertTensorData>(static_cast<size_t>(AssignWeightMemoryOutputs::kTensorData));
  if (tensor == nullptr || assigner == nullptr || device_gtd == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "input or output is invalid.");
    return ge::GRAPH_FAILED;
  }
  void *device_addr = assigner->Assign(tensor->GetSize(), ge::PtrToValue(tensor->GetAddr()));
  if (device_addr == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "assign device addr failed.");
    return ge::GRAPH_FAILED;
  }
  auto stream_id = context->GetInputPointer<int64_t>(static_cast<size_t>(AssignWeightMemoryInputs::kStreamId));
  GE_ASSERT_NOTNULL(stream_id);
  auto placement = device_gtd->GetPlacement();
  auto size = tensor->GetSize();
  *device_gtd = GertTensorData{device_addr, size, placement, *stream_id};
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus AssignMemoryOutputCreator(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto input_gtd = context->GetInputPointer<GertTensorData>(
      static_cast<size_t>(AssignWeightMemoryInputs::kOffsetTensor));
  GE_ASSERT_NOTNULL(input_gtd);

  auto chain = context->GetOutput(static_cast<size_t>(AssignWeightMemoryOutputs::kTensorData));
  GE_ASSERT_NOTNULL(chain);
  auto out_gtd = new (std::nothrow) GertTensorData();
  GE_ASSERT_NOTNULL(out_gtd);
  *out_gtd = GertTensorData{input_gtd->GetAddr(), input_gtd->GetSize(), input_gtd->GetPlacement(), -1};
  chain->SetWithDefaultDeleter(out_gtd);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(CreateMemAssigner).RunFunc(CreateMemAssigner);

REGISTER_KERNEL(AssignWeightMemory).RunFunc(AssignWeightMemory).OutputsCreator(AssignMemoryOutputCreator);

ge::graphStatus GetOrCreateWeightMem(KernelContext *context) {
  auto requred_size = context->GetInputValue<size_t>(static_cast<size_t>(GetOrCreateWeightInputs::kRequiredSize));
  auto appointed_weight =
      context->GetInputPointer<OuterWeightMem>(static_cast<size_t>(GetOrCreateWeightInputs::kAppointedWeight));
  auto device_mem =
      context->GetOutputPointer<GertTensorData>(static_cast<size_t>(GetOrCreateWeightOutputs::kTensorData));
  auto gert_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(GetOrCreateWeightInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  if ((appointed_weight->weight_ptr == nullptr) || (appointed_weight->weight_size < requred_size)) {
    // if given weight size is null or small than required, then malloc by self
    GELOGI("appointed weight info is invalid, given_size[%zu], requred_size[%zu], need to alloc inner.",
           appointed_weight->weight_size, requred_size);
    auto block = gert_allocator->Malloc(requred_size);
    KERNEL_CHECK_NOTNULL(block);
    KERNEL_CHECK(block->GetAddr() != nullptr, "malloc failed, tensor size[%zu]", requred_size);
    device_mem->ShareFrom({requred_size, device_mem->GetPlacement(), gert_allocator->GetStreamId(), block});
  } else {
    *device_mem = GertTensorData{const_cast<void *>(appointed_weight->weight_ptr), appointed_weight->weight_size,
                                 device_mem->GetPlacement(), gert_allocator->GetStreamId()};
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus GetOrCreateWeightOutputCreator(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto gtd = new (std::nothrow) GertTensorData(0U, kOnDeviceHbm, -1, nullptr);
  GE_ASSERT_NOTNULL(gtd);
  auto chain = context->GetOutput(static_cast<size_t>(GetOrCreateWeightOutputs::kTensorData));
  GE_ASSERT_NOTNULL(chain);
  chain->SetWithDefaultDeleter(gtd);
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetOrCreateWeightMem)
    .RunFunc(GetOrCreateWeightMem)
    .OutputsCreator(GetOrCreateWeightOutputCreator)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
} // namespace kernel
} // namespace gert
