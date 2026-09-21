/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ge_host_profiler.h"
#include "runtime/model_v2_executor.h"
namespace gert {
namespace {
constexpr const char *kUnknownNodeName = "UnknownNodeName";
}
GeHostProfiler::GeHostProfiler(const std::shared_ptr<const SubscriberExtendInfo> &extend_info)
    : extend_info_(extend_info) {
  if ((extend_info_ != nullptr) && (extend_info_->executor != nullptr) &&
      (GetGlobalProf()->IsEnabled(ProfilingType::kGeHost))) {
    Init();
  }
}

void GeHostProfiler::Init() {
  if (is_inited_) {
    return;
  }
  GELOGI("[Ge Profiling] Init for ge host profiling.");
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  if (execution_data == nullptr) {
    GELOGW("[Ge Profiling] Execution data is empty, do not init profiler.");
    return;
  }
  prof_extend_infos_.resize(execution_data->base_ed.node_num, ProfExtendInfo{});
  for (size_t i = 0UL; i < execution_data->base_ed.node_num; ++i) {
    const auto node = execution_data->base_ed.nodes[i];
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(node->context.kernel_extend_info);
    if (kernel_extend_info == nullptr) {
      continue;
    }
    const auto kernel_type = kernel_extend_info->GetKernelType();
    prof_extend_infos_[node->node_id].kernel_type_idx = GetGlobalProf()->RegisterString(kernel_type);

    const auto compute_node_info =
        static_cast<const ComputeNodeInfo *>(execution_data->base_ed.nodes[i]->context.compute_node_info);
    uint64_t name_idx = 0UL;
    if (compute_node_info == nullptr) {
      name_idx = GetGlobalProf()->RegisterString(kUnknownNodeName);
    } else {
      name_idx = GetGlobalProf()->RegisterString(compute_node_info->GetNodeName());
    }
    prof_extend_infos_[node->node_id].node_name_idx = name_idx;
  }
  is_inited_ = true;
}

void GeHostProfiler::OnExecuteEvent(int32_t sub_exe_graph_type, GeHostProfiler *profiler, ExecutorEvent event,
                                    const void *node, KernelStatus result) {
  (void)sub_exe_graph_type;
  (void)result;
  if (profiler == nullptr) {
    return;
  }
  if (profiler->IsEnabled(ProfilingType::kGeHost)) {
    if (event == ExecutorEvent::kModelStart) {
      profiler->Init();
    }
    profiler->Record(static_cast<const Node *>(node), event);
  }
}
}  // namespace gert
