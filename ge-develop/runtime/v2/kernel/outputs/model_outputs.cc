/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "model_outputs.h"
#include "register/kernel_registry.h"
#include "exe_graph/runtime/tensor.h"
#include "common/checker.h"
#include "common/plugin/ge_make_unique_util.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "core/debug/kernel_tracing.h"
#include "framework/runtime/subscriber/global_tracer.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "exe_graph/runtime/tensor_data_utils.h"
#include "core/utils/tensor_utils.h"

namespace gert {
namespace kernel {
namespace {
ge::graphStatus AllocMemoryIfNeed(KernelContext *context, Tensor *tensor, GertTensorData *gtd, bool &alloced) {
  alloced = false;
  auto gert_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(AllocModelOutTensorInputs::kAllocator));
  auto tensor_size = context->GetInputValue<size_t>(static_cast<size_t>(AllocModelOutTensorInputs::kAllocSize));
  auto placement = gert_allocator->GetPlacement();
  if ((tensor->GetAddr() != nullptr) && (placement == tensor->GetPlacement())) {
    return ge::SUCCESS;
  }
  alloced = true;

  /*
   * |kernel|外部没有申请地址|外部申请了地址，但是placement错误|
   * |--|---|---|
   * |AllocModelOutTensor|MallocFromL1,所有权在td,gtd RefIn|Malloc,所有权在gtd|
   * |EnsureTensorAtUserMem|不做事情|将gtd拷贝到td，gtd通过Free kernel释放|
   */

  // 外部没有申请地址
  if (tensor->GetAddr() == nullptr) {
    tensor->MutableTensorData() = gert_allocator->MallocTensorDataFromL1(tensor_size);
    GE_ASSERT_NOTNULL(tensor->GetAddr(), "Failed to alloc model output, alloc size %zu", tensor_size);
    GE_ASSERT_SUCCESS(TensorUtils::ShareTdToGtd(tensor->GetTensorData(), *gert_allocator, *gtd));
    KERNEL_TRACE(TRACE_STR_ALLOC_MEM ", as model out memory", gert_allocator->GetStreamId(), gtd->GetGertMemBlock(),
                 tensor->GetAddr(), tensor_size);
  } else {  // placement != tensor->GetPlacement()
    // todo 外部申请了地址，但是size错误，当前GE的对齐规则是512对齐，外部传入的内存是32对齐+32，因此当前还不能直接判断
    //  下一步修改GE的内存对齐规则与外部一致，然后再添加校验
    //  1205更新，即使GE修改了内存申请规则，这里仍然不能被放开，因为在PT场景，PTA会传入没有经过padding的size
    //  这会导致即使使用32对齐+32的方式，在PT场景仍然会失败，因此这里暂时还是不做校验，待PT校验策略确定后修改
    *gtd = gert_allocator->MallocTensorData(tensor_size);
    GE_ASSERT_NOTNULL(gtd->GetAddr(), "Failed to alloc model output, alloc size %zu", tensor_size);
    KERNEL_TRACE(TRACE_STR_ALLOC_MEM ", placement = %s, out memory placement = %s", gert_allocator->GetStreamId(),
                 gtd->GetGertMemBlock(), gtd->GetAddr(), tensor_size, GetPlacementStr(placement),
                 GetPlacementStr(tensor->GetPlacement()));
  }

  return ge::GRAPH_SUCCESS;
}
}  // namespace
ge::graphStatus CreateOutputForKernelAllocModelOutTensor(const ge::FastNode *node, KernelContext *context) {
  if (node == nullptr) {
    GELOGW("The node is nullptr when create output for kernel AllocModelOutTensor");
  }
  auto tensor_data_out = context->GetOutput(0);
  if (tensor_data_out == nullptr) {
    return ge::GRAPH_SUCCESS;
  }
  auto tensor_data = new (std::nothrow) GertTensorData();
  if (tensor_data == nullptr) {
    return ge::GRAPH_FAILED;
  }
  tensor_data_out->SetWithDefaultDeleter(tensor_data);
  return ge::GRAPH_SUCCESS;
}

/**
 * 申请模型输出tensor
 *
 * kernel输入
 *
 * |序号|数据类型|描述|
 * |--|---|---|
 * |0|GertAllocator|内存申请使用的allocator|
 * |1|size_t|输出Tensor的size，单位是字节|
 * |2|Tensor|模型的输出Tensor，模型执行完成后，外部需要将其取走|
 *
 * kernel输出
 *
 * |序号|数据类型|描述|
 * |--|---|---|
 * |0|GertTensorData|输出Tensor中的数据地址|
 *
 * @param context
 * @return
 */
ge::graphStatus AllocModelOutTensor(KernelContext *context) {
  auto tensor_data =
      context->GetOutputPointer<GertTensorData>(static_cast<size_t>(AllocModelOutTensorOutputs::kTensorData));
  auto tensor_chain = context->MutableInput(static_cast<size_t>(AllocModelOutTensorInputs::kOutputData));
  auto gert_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(AllocModelOutTensorInputs::kAllocator));
  if (tensor_data == nullptr || tensor_chain == nullptr || gert_allocator == nullptr) {
    return ge::PARAM_INVALID;
  }

  // 外部没有创建Tensor
  auto tensor = tensor_chain->GetPointer<Tensor>();
  if (tensor == nullptr) {
    auto tensor_holder = ge::MakeUnique<Tensor>();
    if (tensor_holder == nullptr) {
      GELOGE(ge::FAILED, "Failed to create output tensor");
      return ge::FAILED;
    }
    tensor = tensor_holder.release();
    KERNEL_TRACE("Alloc Model out tensor %p", tensor);
    tensor_chain->SetWithDefaultDeleter(tensor);
  }
  bool alloced = false;
  GE_ASSERT_SUCCESS(AllocMemoryIfNeed(context, tensor, tensor_data, alloced), "alloc memory failed.");
  if (alloced) {
    return ge::SUCCESS;
  }
  // 外部申请了placement、size都合法的地址，使用外部提供的地址，零拷贝生效
  KERNEL_TRACE("Zero copy takes effect, using the output address specified by the user %p", tensor->GetAddr());
  GE_ASSERT_SUCCESS(TensorUtils::ShareTdToGtd(tensor->GetTensorData(), *gert_allocator, *tensor_data));
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(AllocModelOutTensor)
    .RunFunc(AllocModelOutTensor)
    .OutputsCreator(CreateOutputForKernelAllocModelOutTensor)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert
