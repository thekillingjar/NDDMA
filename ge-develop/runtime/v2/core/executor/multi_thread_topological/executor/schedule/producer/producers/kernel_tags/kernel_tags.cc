/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_tags.h"
#include <map>
#include "core/execution_data.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "register/kernel_registry.h"
#include "critical_section_config.h"
#include "core/builder/node_types.h"

namespace gert {
namespace {
const std::map<const std::string, ExecTaskType> critical_to_types = {
    {kKernelUseMemory, ExecTaskType::MEMORY}
};
// 内存线程->普通线程断流水场景：
// 1. AllocHostCpuOutputMemory -> PackHostKernel
// 2. PackHostKernel ->IdentityAddr -> BuildTensor -> InferShape
// 3. SplitDataTensor -> BuildTensor
// 4. MakeSureTensorAtHost - > BuildTensor
// 特别注意：MakeSureTensorAtHost包含内存申请操作，如果要挪到普通线程则要求所有host内存申请和释放的kernel都放到普通线程上，暂不实施
bool IsBrokenNormalToMemoryWaterFlowNode(const char *const node_type) {
  return IsAllocHostCpuOutputMemoryNode(node_type) || IsSplitDataTensorNode(node_type) ||
         IsBuildTensorNode(node_type) || IsIdentityNode(node_type);
}

bool IsMemoryToNormalWaterFlowNode(const char *const node_type) {
  return IsExecuteOpPrepareNode(node_type);
}
}

void KernelTags::Reset(size_t node_num, size_t thread_num) {
  thread_num_ = thread_num;
  node_tags_.resize(node_num + 1, ExecTaskType::MAX);
}

ExecTaskType KernelTags::UpdateTag(const Node *kernel) {
  node_tags_[kernel->node_id] = ExecTaskType::NORMAL;
  const auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(&kernel->context);
  const auto &node_type = extend_context->GetKernelType();
  if (node_type == nullptr) {
    return ExecTaskType::NORMAL;
  }

  const auto kernel_info = KernelRegistry::GetInstance().FindKernelInfo(node_type);
  if (kernel_info == nullptr) {
    return ExecTaskType::NORMAL;
  }
  if (IsLaunchNode(node_type)) {
    node_tags_[kernel->node_id] = ExecTaskType::LAUNCH;
  }
  const auto iter = critical_to_types.find(kernel_info->critical_section);
  if (iter != critical_to_types.cend()) {
    node_tags_[kernel->node_id] = iter->second;
  }
  // 断流水的内存线程上的kernel只能在线程数为3时放到普通线程上
  if (IsBrokenNormalToMemoryWaterFlowNode(node_type) && (thread_num_ == 3U)) {
    node_tags_[kernel->node_id] = ExecTaskType::NORMAL;
  }
  if (IsMemoryToNormalWaterFlowNode(node_type)) {
    node_tags_[kernel->node_id] = ExecTaskType::MEMORY;
  }
  return node_tags_[kernel->node_id];
}
}  // namespace gert
