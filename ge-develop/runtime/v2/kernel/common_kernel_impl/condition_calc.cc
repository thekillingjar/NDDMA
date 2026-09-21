/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "condition_calc.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "register/op_impl_registry.h"
#include "framework/common/debug/ge_log.h"
#include "transfer_shape_according_to_format.h"
#include "graph/node.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "common/checker.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace kernel {
ge::graphStatus FindKernelFunc(KernelContext *context) {
  auto func_name = context->GetInputValue<char *>(0);
  if (func_name == nullptr) {
    GELOGE(ge::FAILED, "Failed to find condition calc kernel as input kernel type is nullptr");
    return ge::GRAPH_FAILED;
  }
  auto kernel_funcs = KernelRegistry::GetInstance().FindKernelFuncs(func_name);
  if (kernel_funcs == nullptr || kernel_funcs->run_func == nullptr) {
    GELOGE(ge::FAILED, "No condition calc kernel func was registered for lower node type %s", func_name);
    return ge::GRAPH_FAILED;
  }
  auto kernel_func = context->GetOutputPointer<KernelRegistry::KernelFunc>(0);
  GE_CHECK_NOTNULL(kernel_func);
  *kernel_func = kernel_funcs->run_func;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ConditionCalc(KernelContext *context) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto input_num = context->GetInputNum();
  if (input_num < 1U) {  // The last input must be kernel func
    GELOGE(ge::FAILED, "Failed get calc kernel func for node %s as input num is 0", extend_context->GetNodeName());
    return ge::GRAPH_FAILED;
  }
  auto kernel_func = context->GetInputValue<KernelRegistry::KernelFunc>(input_num - 1U);
  GE_CHECK_NOTNULL(kernel_func);

  auto ret = kernel_func(context);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ret, "Run condition calc kernel func for node %s failed", extend_context->GetNodeName());
    return ret;
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildUnmanagedTensorData(KernelContext *context) {
  auto gtd =
      context->GetOutputPointer<GertTensorData>(static_cast<size_t>(BuildUnmanagedTensorDataOutputs::kTensorData));
  auto stream_id = context->GetInputPointer<int64_t>(static_cast<size_t>(BuildUnmanagedTensorDataInputs::kStreamId));
  GE_ASSERT_NOTNULL(stream_id);
  GE_CHECK_NOTNULL(gtd);
  auto addr = context->GetInputPointer<uint8_t>(static_cast<size_t>(BuildUnmanagedTensorDataInputs::kAddr));
  auto size = gtd->GetSize();
  auto placement = gtd->GetPlacement();
  *gtd = GertTensorData{const_cast<uint8_t *>(addr), size, placement, *stream_id};
  return ge::GRAPH_SUCCESS;
}

namespace {
ge::graphStatus CreateTensorDataOutputs(const ge::FastNode *node, KernelContext *context) {
  auto chain = context->GetOutput(static_cast<size_t>(BuildUnmanagedTensorDataOutputs::kTensorData));
  GE_CHECK_NOTNULL(chain);
  auto gtd = new (std::nothrow) GertTensorData();
  if (gtd == nullptr) {
    GELOGE(ge::FAILED, "Failed create outputs for node %s", node->GetName().c_str());
    return ge::GRAPH_FAILED;
  }
  chain->SetWithDefaultDeleter(gtd);
  return ge::GRAPH_SUCCESS;
}
}  // namespace

REGISTER_KERNEL(FindKernelFunc).RunFunc(FindKernelFunc);
REGISTER_KERNEL(ConditionCalc).RunFunc(ConditionCalc);
REGISTER_KERNEL(BuildUnmanagedTensorData).RunFunc(BuildUnmanagedTensorData).OutputsCreator(CreateTensorDataOutputs);
}  // namespace kernel
}  // namespace gert
