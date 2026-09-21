/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/exe_graph_resource_guard.h"
#include "exe_graph/runtime/kernel_context.h"

namespace gert {
void *ResourceGuard::ResetExecutionData(std::unique_ptr<uint8_t[]> execution_data) {
  execution_data_holder_ = std::move(execution_data);
  return execution_data_holder_.get();
}
void *ResourceGuard::GetExecutionData() {
  return execution_data_holder_.get();
}
void TopologicalResourceGuard::ResetAnyValue(std::unique_ptr<uint8_t[]> any_values, size_t count) {
  any_values_num_ = count;
  any_values_guard_ = std::move(any_values);
}
void TopologicalResourceGuard::PushNode(void *node) {
  nodes_guarder_.emplace_back(node, free);
}
void TopologicalResourceGuard::PushWatcher(void *watcher) {
  watchers_guarder_.emplace_back(watcher, free);
}
void *TopologicalResourceGuard::ResetNodesArray(std::unique_ptr<uint8_t[]> nodes_array) {
  nodes_array_guarder_ = std::move(nodes_array);
  return nodes_array_guarder_.get();
}
void *TopologicalResourceGuard::ResetStartNodesArray(std::unique_ptr<uint8_t[]> start_nodes_array) {
  start_nodes_array_guarder_ = std::move(start_nodes_array);
  return start_nodes_array_guarder_.get();
}
void *TopologicalResourceGuard::ResetNodesIndgreeArray(std::unique_ptr<uint8_t[]> nodes_indgree_array) {
  nodes_indgree_array_guarder_ = std::move(nodes_indgree_array);
  return nodes_indgree_array_guarder_.get();
}
void *TopologicalResourceGuard::ResetNodesWaitIndgreeArray(std::unique_ptr<uint8_t[]> nodes_indgree_array) {
  nodes_wait_indgree_array_guarder_ = std::move(nodes_indgree_array);
  return nodes_wait_indgree_array_guarder_.get();
}
void *TopologicalResourceGuard::ResetInputsArray(std::unique_ptr<uint8_t[]> inputs_array) {
  inputs_array_guarder_ = std::move(inputs_array);
  return inputs_array_guarder_.get();
}
void *TopologicalResourceGuard::ResetOutputsArray(std::unique_ptr<uint8_t[]> outputs_array) {
  outputs_array_guarder_ = std::move(outputs_array);
  return outputs_array_guarder_.get();
}
void *TopologicalResourceGuard::ResetWatchersArray(std::unique_ptr<uint8_t[]> watchers_array) {
  watchers_array_guarder_ = std::move(watchers_array);
  return watchers_array_guarder_.get();
}
void *TopologicalResourceGuard::ResetReadyQueue(void *ready_queue) {
  ready_queue_guarder_ = std::unique_ptr<void, decltype(&free)>(ready_queue, free);
  return ready_queue;
}
void *TopologicalResourceGuard::ResetBuffer(std::unique_ptr<uint8_t[]> buffer) {
  buffer_guarder_ = std::move(buffer);
  return buffer_guarder_.get();
}
void *TopologicalResourceGuard::ResetComputeNodeInfo(std::unique_ptr<uint8_t[]> compute_node_info) {
  compute_node_info_guarder_ = std::move(compute_node_info);
  return compute_node_info_guarder_.get();
}
void *TopologicalResourceGuard::ResetKernelExtendInfo(std::unique_ptr<uint8_t[]> kernel_extend_info) {
  kernel_extend_info_guarder_ = std::move(kernel_extend_info);
  return kernel_extend_info_guarder_.get();
}
void *TopologicalResourceGuard::ResetModelDesc(std::unique_ptr<uint8_t[]> model_desc) {
  model_desc_guarder_ = std::move(model_desc);
  return model_desc_guarder_.get();
}
TopologicalResourceGuard::~TopologicalResourceGuard() {
  auto any_values = reinterpret_cast<Chain *>(any_values_guard_.get());
  if (any_values != nullptr) {
    for (size_t i = any_values_num_; i > 0U; --i) {
      any_values[i - 1].Set(nullptr, nullptr);
    }
    any_values = nullptr;
  }
}
}
