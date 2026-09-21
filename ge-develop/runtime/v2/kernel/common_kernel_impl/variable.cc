/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "variable.h"
#include <unordered_set>
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "common/checker.h"
#include "graph/manager/graph_var_manager.h"
#include "framework/runtime/rt_session.h"
#include "core/utils/tensor_utils.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"

namespace gert {
namespace kernel {
namespace {
ge::graphStatus SplitVariable(KernelContext *context) {
  // todo: Remove this after rt-session supported
  // 当前GE现存的VariableManager，按照Global -> Session -> Device的层次管理，使得任何时候想要获取变量地址，
  // 都需要获取Device并在全局信息中查找，这对于RT2执行图来说难以接受。RT2执行器就像TF图永远不知道自己在被哪个Session执行一样，
  // 算子只应该按照IR语义工作，像是在算子执行时，获取SessionId/DeviceId信息的行为，是无法接受的。RT1在实现时，在执行器加载时，
  // 先根据加载时的Device信息，查询并缓存当前模型所需的资源。与之类似，RtSession我们认为包含当前模型执行所需的Session资源，
  // 但并不保证包含全部的Session级资源（可以认为算子只应该看到它必须看到的），RtSession我们就认为是RT2模型所应该看到的Session资源最小集。

  // 从GE变量的IR定义上，变量名是变量的逻辑ID，在确定的上下文中，根据这个ID就可以确定唯一的Variable实例，当前RT2已经约束了网络在特定
  // 上下文中执行，上下文由调用者保证，因此给到Kernel的RtSession（输入）信息，需要可以直接根据变量ID确定实例信息，而不应该再
  // 附加SessionId/DeviceId等信息查询。

  // 当前RtSession的机制还没有实现，临时最小化通过一个线程变量实现。并在加载时保证存在，这更像是一个RT2模型加载的
  // 上下文。因此当前需要保证变量的Split动作，在加载时就完成。等待RtSession支持后，RtSession将变成一个模型级的Session资源访问器，
  // 可以访问当前模型执行所需的Session信息。
  auto rt_session = context->GetInputValue<RtSession *>(static_cast<size_t>(SplitVariableInputs::kRtSession));
  GE_ASSERT_NOTNULL(rt_session);
  auto var_id = context->GetInputValue<char *>(static_cast<size_t>(SplitVariableInputs::kVarId));
  GE_ASSERT_NOTNULL(var_id);
  auto gert_allocator = context->GetInputValue<GertAllocator *>(static_cast<size_t>(SplitVariableInputs::kAllocator));
  GE_ASSERT_NOTNULL(gert_allocator);
  auto shape = context->GetOutputPointer<StorageShape>(static_cast<size_t>(SplitVariableOutputs::kShape));
  GE_ASSERT_NOTNULL(shape);
  auto gtd = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(SplitVariableOutputs::kTensorData));
  GE_ASSERT_NOTNULL(gtd);

  auto var_manager = rt_session->GetVarManager();
  GE_ASSERT_NOTNULL(var_manager, "Var manager not found when split variable %s", var_id);
  TensorData tmp_td;
  GE_ASSERT_SUCCESS(var_manager->GetVarShapeAndMemory(var_id, *shape, tmp_td),
                    "Var manager %p get variable '%s' shape and memory failed", var_manager, var_id);
  GE_ASSERT_SUCCESS(TensorUtils::ShareTdToGtd(tmp_td, *gert_allocator, *gtd));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateVariableOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto shape_chain = context->GetOutput(static_cast<size_t>(SplitVariableOutputs::kShape));
  GE_ASSERT_NOTNULL(shape_chain);
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  auto output_desc = extend_context->GetOutputDesc(0UL);
  GE_ASSERT_NOTNULL(output_desc);
  auto shape_tensor = new (std::nothrow) Tensor(StorageShape(),
      output_desc->GetFormat(), output_desc->GetDataType());
  shape_chain->SetWithDefaultDeleter(shape_tensor);

  auto tensor_data_chain = context->GetOutput(static_cast<size_t>(SplitVariableOutputs::kTensorData));
  GE_ASSERT_NOTNULL(tensor_data_chain);
  auto td = new (std::nothrow) GertTensorData();
  tensor_data_chain->SetWithDefaultDeleter(td);
  return ge::GRAPH_SUCCESS;
}
}  // namespace
REGISTER_KERNEL(SplitVariable)
    .RunFunc(SplitVariable)
    .OutputsCreator(CreateVariableOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
}  // namespace kernel
}  // namespace gert
