/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hybrid/node_executor/ge_local/data_flow_kernels.h"
#include "hybrid/executor/hybrid_execution_context.h"

namespace ge {
namespace hybrid {
DataFlowKernelFactory &DataFlowKernelFactory::GetInstance() {
  static DataFlowKernelFactory instance;
  return instance;
}

void DataFlowKernelFactory::RegisterKernelCreator(const std::string &type, const DataFlowKernelCreator &creator) {
  const std::lock_guard<std::mutex> lock(data_flow_kernel_mutex_);
  if (data_flow_kernel_creators_.count(type) != 0UL) {
    GELOGI("The kernel has already been registered, type:%s.", type.c_str());
  }
  (void)data_flow_kernel_creators_.insert({type, creator});
}

DataFlowKernelBasePtr DataFlowKernelFactory::CreateKernel(const std::string &type) {
  const std::lock_guard<std::mutex> lock(data_flow_kernel_mutex_);
  const auto iter = data_flow_kernel_creators_.find(type);
  if (iter != data_flow_kernel_creators_.end()) {
    return iter->second();
  }
  return nullptr;
}

/// ******************************
///   Definition of Stack series
/// ******************************
Status DataFlowStack::Compute(TaskContext &context, const int64_t handle) {
  GE_CHECK_NOTNULL(context.GetExecutionContext());
  const DataFlowResourcePtr res = context.GetExecutionContext()->res_manager.GetDataFlowResource(handle);
  GE_CHECK_NOTNULL(res);
  if (!(res->IsMaxSizeConst())) {
    // Get max size from input 0, usually it is not needed
    const auto size_tensor_value = context.MutableInput(0);
    GE_CHECK_NOTNULL(size_tensor_value);
    if (size_tensor_value->GetSize() < sizeof(int32_t)) {
      GELOGE(PARAM_INVALID, "[Check][Size]Stack %s max_size memory size:%zu, expect at least %zu.",
             context.GetNodeName(), size_tensor_value->GetSize(), sizeof(int32_t));
      return PARAM_INVALID;
    }
    std::vector<uint8_t> buffer(size_tensor_value->GetSize());
    GE_CHK_RT_RET(rtMemcpy(buffer.data(),
                           size_tensor_value->GetSize(),
                           size_tensor_value->MutableData(),
                           size_tensor_value->GetSize(),
                           RT_MEMCPY_DEVICE_TO_HOST));
    const int32_t *const max_size_ptr = PtrToPtr<const uint8_t, const int32_t>(buffer.data());
    int64_t max_size = static_cast<int64_t>(*max_size_ptr);
    GELOGD("Stack[%s] get max size:%ld.", context.GetNodeName(), max_size);
    max_size = (max_size > 0) ? max_size : std::numeric_limits<int64_t>::max();
    res->SetMaxSize(max_size);
  }
  const auto tensor_value = context.MutableOutput(0);
  GE_CHECK_NOTNULL(tensor_value);
  // Has set dependent_for_execution before, the input is ready when get here.
  GE_CHK_RT_RET(rtMemcpyAsync(tensor_value->MutableData(), tensor_value->GetSize(), &handle, sizeof(int64_t),
                              RT_MEMCPY_HOST_TO_DEVICE, context.GetStream()));
  res->SetClosed(false);
  GELOGD("Stack[%s] compute successfully, handle[%ld].", context.GetNodeName(), handle);
  return SUCCESS;
}

Status DataFlowStackPush::Compute(TaskContext &context, const int64_t handle) {
  GE_CHECK_NOTNULL(context.GetExecutionContext());
  const DataFlowResourcePtr res = context.GetExecutionContext()->res_manager.GetDataFlowResource(handle);
  GE_CHECK_NOTNULL(res);
  if (res->IsClosed()) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to push, resource is closed. StackPush name:%s, handle:%" PRId64 ".",
                       context.GetNodeName(), handle);
    GELOGE(INTERNAL_ERROR, "Resource is closed, StackPush name:%s, handle:%ld.", context.GetNodeName(), handle);
    return INTERNAL_ERROR;
  }
  const auto data_tensor_value = context.MutableInput(1);
  GE_CHECK_NOTNULL(data_tensor_value);
  const auto data_desc = context.MutableInputDesc(1);
  GE_CHECK_NOTNULL(data_desc);
  const Status ret = res->EmplaceBack(std::make_pair(*data_tensor_value, *data_desc));
  if (ret != SUCCESS) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to push, StackPush name:%s, handle:%" PRId64 ".",
                       context.GetNodeName(), handle);
    GELOGE(ret, "Failed to push, StackPush name:%s, handle:%ld.", context.GetNodeName(), handle);
    return ret;
  }
  const auto output_desc = context.MutableOutputDesc(0);
  GE_CHECK_NOTNULL(output_desc);
  int64_t copy_size = 0;
  GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorSizeInBytes(*output_desc, copy_size));
  if (copy_size != 0) {
    const auto out_tensor_value = context.MutableOutput(0);
    GE_CHECK_NOTNULL(out_tensor_value);
    GE_CHK_RT_RET(rtMemcpyAsync(out_tensor_value->MutableData(), out_tensor_value->GetSize(),
                                data_tensor_value->GetData(), static_cast<uint64_t>(copy_size),
                                RT_MEMCPY_DEVICE_TO_DEVICE, context.GetStream()));
  }
  GELOGD("StackPush[%s] compute successfully, handle[%ld].", context.GetNodeName(), handle);
  return SUCCESS;
}

Status DataFlowStackPop::Compute(TaskContext &context, const int64_t handle) {
  GE_CHECK_NOTNULL(context.GetExecutionContext());
  const DataFlowResourcePtr res = context.GetExecutionContext()->res_manager.GetDataFlowResource(handle);
  GE_CHECK_NOTNULL(res);
  if (res->IsClosed()) {
    REPORT_INNER_ERR_MSG("E19999", "Failed to push, resource is closed. StackPop name:%s, handle:%" PRId64 ".",
                       context.GetNodeName(), handle);
    GELOGE(INTERNAL_ERROR, "Resource is closed, StackPop name:%s, handle:%ld.", context.GetNodeName(), handle);
    return INTERNAL_ERROR;
  }
  if (res->Empty()) {
    REPORT_INNER_ERR_MSG("E19999", "Resource is empty, StackPop name:%s, handle:%" PRId64 ".",
                       context.GetNodeName(), handle);
    GELOGE(INTERNAL_ERROR, "Resource is empty, StackPop name:%s, handle:%ld.",
           context.GetNodeName(), handle);
    return INTERNAL_ERROR;
  }
  const DataFlowTensor &top = res->Back();
  (void)context.SetOutput(0, top.first);
  const auto tensor_desc = context.MutableOutputDesc(0);
  GE_CHECK_NOTNULL(tensor_desc);
  *tensor_desc = top.second;
  res->PopBack();
  GELOGD("StackPop[%s] compute successfully, handle[%ld].", context.GetNodeName(), handle);
  return SUCCESS;
}

Status DataFlowStackClose::Compute(TaskContext &context, const int64_t handle) {
  GE_CHECK_NOTNULL(context.GetExecutionContext());
  const DataFlowResourcePtr res = context.GetExecutionContext()->res_manager.GetDataFlowResource(handle);
  GE_CHECK_NOTNULL(res);
  res->Clear();
  res->SetClosed(true);
  GELOGD("StackClose[%s] compute successfully, handle[%ld].", context.GetNodeName(), handle);
  return SUCCESS;
}

/// ******************************
///   Register data flow kernels
/// ******************************
REGISTER_DATA_FLOW_KERNEL(STACK, DataFlowStack);
REGISTER_DATA_FLOW_KERNEL(STACKPUSH, DataFlowStackPush);
REGISTER_DATA_FLOW_KERNEL(STACKPOP, DataFlowStackPop);
REGISTER_DATA_FLOW_KERNEL(STACKCLOSE, DataFlowStackClose);
}  // namespace hybrid
}  // namespace ge
