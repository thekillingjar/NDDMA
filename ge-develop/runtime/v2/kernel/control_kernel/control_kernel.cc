/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "control_kernel.h"
#include <cmath>
#include "common/ge_inner_attrs.h"
#include "common/checker.h"
#include "graph/ge_error_codes.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "register/kernel_registry.h"
#include "runtime/mem.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph_metadef/common/ge_common/util.h"

namespace gert {
namespace kernel {
namespace {
ge::graphStatus EmptyFunc(KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

template <typename T>
bool IntegerToBool(const void *data) {
  return (*static_cast<const T *>(data)) != T(0);
}

template <>
bool IntegerToBool<bool>(const void *data) {
  return *static_cast<const bool *>(data);
}

template <typename T>
bool FloatToBool(const void *data) {
  return fabs(*static_cast<const T *>(data)) > std::numeric_limits<T>::epsilon();
}

const std::vector<std::function<bool(const void *)>> kBoolConverter = []() noexcept {
  std::vector<std::function<bool(const void *)>> converts;
  converts.resize(static_cast<size_t>(ge::DT_MAX), nullptr);
  converts[ge::DT_FLOAT] = &FloatToBool<ge::float32_t>;
  converts[ge::DT_DOUBLE] = &FloatToBool<ge::float64_t>;
  converts[ge::DT_INT8] = &IntegerToBool<int8_t>;
  converts[ge::DT_UINT8] = &IntegerToBool<uint8_t>;
  converts[ge::DT_INT16] = &IntegerToBool<int16_t>;
  converts[ge::DT_UINT16] = &IntegerToBool<uint16_t>;
  converts[ge::DT_INT32] = &IntegerToBool<int32_t>;
  converts[ge::DT_UINT32] = &IntegerToBool<uint32_t>;
  converts[ge::DT_INT64] = &IntegerToBool<int64_t>;
  converts[ge::DT_UINT64] = &IntegerToBool<uint64_t>;
  converts[ge::DT_BOOL] = &IntegerToBool<bool>;
  return converts;
}();

ge::graphStatus GetCondState(const void *tensor_addr, const StorageShape &shape, const ge::DataType &data_type,
                             bool &cond_state) {
  if (shape.GetOriginShape().IsScalar()) {
    const auto numeral_dtype = static_cast<size_t>(data_type);
    GE_ASSERT_TRUE(numeral_dtype < kBoolConverter.size(), "Unsupported cond data type %s(%zu)",
                   ge::TypeUtils::DataTypeToSerialString(data_type).c_str(), numeral_dtype);
    GE_ASSERT_NOTNULL(tensor_addr, "Failed to generate index as tensor data is nullptr");
    cond_state = kBoolConverter[numeral_dtype](tensor_addr);
  } else {
    // if it is not scalar, non-empty tensor means true, otherwise false
    cond_state = true;
    for (size_t i = 0U; i < shape.GetOriginShape().GetDimNum(); ++i) {
      if (shape.GetOriginShape().GetDim(i) == 0) {
        cond_state = false;
        break;
      }
    }
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

ge::graphStatus GenIndexForIf(KernelContext *context) {
  auto tensor = context->GetInputPointer<Tensor>(0UL);
  auto index = context->GetOutputPointer<uint32_t>(0UL);
  if (tensor == nullptr || index == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Failed to generate index for if, no tensor or index input");
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  bool cond_state = false;
  GE_ASSERT_SUCCESS(GetCondState(tensor->GetAddr(), tensor->GetShape(), tensor->GetDataType(), cond_state),
                    "Failed to get cond state for If");
  *index = cond_state ? 0U : 1U;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenIndexForCase(KernelContext *context) {
  auto tensor = context->GetInputPointer<Tensor>(0UL);
  auto branch_num = context->GetInputValue<uint32_t>(1UL);
  auto index = context->GetOutputPointer<uint32_t>(0UL);
  if (tensor == nullptr || tensor->GetData<int32_t>() == nullptr || index == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Failed to generate index for case, tensor or index is nullptr");
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  if (branch_num == 0U) {
    GELOGE(ge::PARAM_INVALID, "Failed to generate index for case, at least one default branch");
    return ge::PARAM_INVALID;
  }

  auto branch_index = *tensor->GetData<int32_t>();
  if (branch_index < 0 || static_cast<uint32_t>(branch_index) >= branch_num) {
    *index = branch_num - 1U;
    GELOGD("The branch index %d is less than 0 or reach the max num of branch num(%u), select the default branch %u",
           branch_index, branch_num, *index);
  } else {
    *index = static_cast<uint32_t>(branch_index);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GenCondForWhile(KernelContext *context) {
  auto tensor = context->GetInputPointer<GertTensorData>(0U);
  GE_ASSERT_NOTNULL(tensor);
  auto shape = context->GetInputPointer<StorageShape>(1U);
  GE_ASSERT_NOTNULL(shape);
  auto data_type = context->GetInputValue<ge::DataType>(2U);
  auto out = context->GetOutputPointer<bool>(0U);
  GE_ASSERT_NOTNULL(out);
  GE_ASSERT_SUCCESS(GetCondState(tensor->GetAddr(), *shape, data_type, *out), "Failed to get cond state for While");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SwitchNotifyKernel(KernelContext *context) {
  auto branch_index = context->GetInputValue<uint32_t>(0UL);
  auto signal = context->GetOutputPointer<NotifySignal>(0UL);
  if (signal == nullptr) {
    GELOGE(ge::GE_GRAPH_PARAM_NULLPTR, "Failed to execute SwitchNotifyKernel, signal is nullptr");
    return ge::GE_GRAPH_PARAM_NULLPTR;
  }
  if (branch_index >= signal->receiver_num) {
    GELOGE(ge::PARAM_INVALID, "Invalid branch index %u, receiver num %zu", branch_index, signal->receiver_num);
    return ge::PARAM_INVALID;
  }

  signal->target_receiver = signal->receivers[branch_index];
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateSwitchNotifyOutputs(const ge::FastNode *node, KernelContext *context) {
  Watcher *watcher = nullptr;
  int64_t address_buffer = 0;
  if (!ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), ge::kWatcherAddress, address_buffer)) {
    GELOGE(ge::FAILED, "Failed get attr '%s' from node %s", ge::kWatcherAddress, node->GetName().c_str());
    return ge::GRAPH_FAILED;
  }
  if (memcpy_s(&watcher, sizeof(Watcher *), &address_buffer, sizeof(Watcher *)) != EOK) {
    GELOGE(ge::FAILED, "Failed copy watcher address from src buffer");
    return ge::GRAPH_FAILED;
  }
  GE_ASSERT_EQ(watcher->watch_num, node->GetAllOutEdgesSize());

  // Finetune the watcher order from Target|Branch|Control -> Target|Control|Branch
  // This is needed as resource guarder maybe have control dependency from 'SwitchNotify',
  // Now 'SwitchNotify' active one of data outputs and all control outputs
  auto watchers_num = watcher->watch_num;
  auto outputs_num = node->GetAllOutDataEdgesSize();
  std::vector<NodeIdentity> optimized_watcher;
  optimized_watcher.reserve(watchers_num);
  optimized_watcher.push_back(watcher->node_ids[0]); // Target receiver
  for (size_t i = outputs_num; i < watchers_num; i++) {
    optimized_watcher.push_back(watcher->node_ids[i]);
  }
  watcher->watch_num = optimized_watcher.size();
  auto data_watchers = &watcher->node_ids[optimized_watcher.size()];
  for (size_t i = 1U; i < outputs_num; i++) {
    optimized_watcher.push_back(watcher->node_ids[i]);
  }
  GE_ASSERT_EOK(memcpy_s(watcher->node_ids, watchers_num * sizeof(NodeIdentity), optimized_watcher.data(),
                         optimized_watcher.size() * sizeof(NodeIdentity)));

  auto chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(chain);
  auto signal = new (std::nothrow) NotifySignal(*watcher->node_ids, data_watchers, outputs_num - 1U);
  if (signal == nullptr) {
    GELOGE(ge::FAILED, "Failed create outputs for node %s", node->GetName().c_str());
    return ge::GRAPH_FAILED;
  }
  chain->SetWithDefaultDeleter(signal);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CondSwitchNotifyKernel(KernelContext *context) {
  auto signal = context->GetOutputPointer<NotifySignal>(0);
  GE_ASSERT_NOTNULL(signal);
  size_t branch_index = context->GetInputValue<bool>(0) ? 0U : 1U;
  GE_ASSERT_TRUE(signal->receiver_num > branch_index);
  signal->target_receiver = signal->receivers[branch_index];
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(BranchPivot).RunFunc(EmptyFunc);
REGISTER_KERNEL(BranchDone).RunFunc(EmptyFunc);
REGISTER_KERNEL(WatcherPlaceholder).RunFunc(EmptyFunc);
REGISTER_KERNEL(WaitAnyone).RunFunc(EmptyFunc);
REGISTER_KERNEL(GenIndexForCase).RunFunc(GenIndexForCase);
REGISTER_KERNEL(GenIndexForIf).RunFunc(GenIndexForIf);
REGISTER_KERNEL(GenCondForWhile).RunFunc(GenCondForWhile);
REGISTER_KERNEL(Enter).RunFunc(EmptyFunc);
REGISTER_KERNEL(Exit).RunFunc(EmptyFunc);
REGISTER_KERNEL(NoOp).RunFunc(EmptyFunc);
REGISTER_KERNEL(UnfedData).RunFunc(EmptyFunc);

// 引入OutputsCreater的NotifySignal结构，主要是为了存储原始的watch_num，防止Case的index越界
REGISTER_KERNEL(SwitchNotify).RunFunc(SwitchNotifyKernel).OutputsCreator(CreateSwitchNotifyOutputs);
REGISTER_KERNEL(CondSwitchNotify).RunFunc(CondSwitchNotifyKernel).OutputsCreator(CreateSwitchNotifyOutputs);
}  // namespace kernel
}  // namespace gert
