/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base_executor_profiler.h"
#include "core/builder/node_types.h"

namespace gert {
namespace {
constexpr size_t kInvalidHashId = 0UL;
constexpr const char *kUnknownNodeName = "UnknownNodeName";
const char *KAtomicSuffix[] = {"_MemSet", "_AtomicClean"};
}  // namespace

void BaseExecutorProfiler::InitNameAndTypeWithHash(const ExecutionData &execution_data) {
  if (is_str_to_hash_inited_) {
    return;
  }
  (void)InitNameAndTypeWithHash(execution_data, prof_extend_infos_);
  is_str_to_hash_inited_ = true;
}

ge::graphStatus BaseExecutorProfiler::InitNameAndTypeWithHash(const ExecutionData &execution_data,
                                                              std::vector<ProfExtendInfo> &prof_extend_infos) const {
  prof_extend_infos.resize(execution_data.base_ed.node_num, ProfExtendInfo{});
  for (size_t i = 0UL; i < execution_data.base_ed.node_num; ++i) {
    const auto node = execution_data.base_ed.nodes[i];
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
    if (kernel_extend_info == nullptr) {
      continue;
    }
    GE_ASSERT_TRUE(node->node_id < execution_data.base_ed.node_num);

    auto kernel_type = kernel_extend_info->GetKernelType();
    uint64_t kernel_type_hash = kInvalidHashId;
    if (SubscriberUtils::IsNodeNeedProf(kernel_type)) {
      kernel_type_hash = MsprofGetHashId(kernel_type, strlen(kernel_type));
    }
    prof_extend_infos[node->node_id].kernel_type_idx = kernel_type_hash;

    const char *node_name = kUnknownNodeName;
    const char *node_type = kUnknownNodeName;
    const auto compute_node_info = static_cast<const ComputeNodeInfo *>(node->context.compute_node_info);
    if (compute_node_info != nullptr) {
      node_type = compute_node_info->GetNodeType();
      node_name = compute_node_info->GetNodeName();
      for (const auto suffix : KAtomicSuffix) {
        const char *find_pos = strstr(node_name, suffix);
        if (find_pos != nullptr) {
          std::string tmp = std::string(node_name).substr(0UL, find_pos - node_name);
          prof_extend_infos[node->node_id].origin_name_hash_for_hash = MsprofGetHashId(tmp.c_str(), tmp.length());
          break;
        }
      }
    }

    uint64_t node_type_hash = MsprofGetHashId(node_type, strlen(node_type));
    uint64_t name_hash = MsprofGetHashId(node_name, strlen(node_name));
    prof_extend_infos[node->node_id].node_type_idx = node_type_hash;
    prof_extend_infos[node->node_id].node_name_idx = name_hash;
    prof_extend_infos[node->node_id].kernel_prof_type = SubscriberUtils::GetProfKernelType(kernel_type, true);

    GELOGI(
        "[Cann Profiling] node_id: %zu, kernel_type: %s %llu, node_type: %s %llu, node_name: %s %llu, "
        "kernel_prof_type: %u.",
        node->node_id, kernel_type, kernel_type_hash, node_type, node_type_hash, node_name, name_hash,
        prof_extend_infos[node->node_id].kernel_prof_type);
  }
  return ge::SUCCESS;
}
}  // namespace gert
