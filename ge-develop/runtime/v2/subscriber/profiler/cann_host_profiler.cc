/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cann_host_profiler.h"
#include "subscriber/subscriber_utils.h"
#include "runtime/model_v2_executor.h"

namespace gert {
namespace {
constexpr const char *kUnknownNodeName = "UnknownNodeName";
}
CannHostProfiler::CannHostProfiler(const std::shared_ptr<const SubscriberExtendInfo> &extend_info)
    : extend_info_(extend_info) {
  if ((extend_info_ != nullptr) && (extend_info_->executor != nullptr) && IsEnabled(ProfilingType::kCannHost)) {
    Init();
  }
}

void CannHostProfiler::InitHostSchData(const ExecutionData &execution_data) {
  host_sch_infos_.resize(execution_data.base_ed.node_num, MsprofApi{});
  for (size_t i = 0UL; i < execution_data.base_ed.node_num; ++i) {
    const auto node = execution_data.base_ed.nodes[i];
    auto kernel_context = reinterpret_cast<KernelContext *>(&node->context);
    if (kernel_context == nullptr) {
      GELOGW("Kernel context is nullptr.");
      continue;
    }
    const auto kernel_extend_info = static_cast<const KernelExtendInfo *>(kernel_context->GetKernelExtend());
    if (kernel_extend_info == nullptr) {
      GELOGW("Kernel extend info is nullptr.");
      continue;
    }
    const auto prof_type = SubscriberUtils::GetProfKernelType(kernel_extend_info->GetKernelType(), false);
    if (prof_type != kInvalidProfKernelType) {
      host_sch_infos_[node->node_id].level = MSPROF_REPORT_NODE_LEVEL;
      host_sch_infos_[node->node_id].type = prof_type;
      uint64_t name_hash_id;
      const auto compute_node_info = static_cast<const ComputeNodeInfo *>(kernel_context->GetComputeNodeExtend());
      if (compute_node_info == nullptr) {
        name_hash_id = MsprofGetHashId(kUnknownNodeName, std::string(kUnknownNodeName).length());
      } else {
        const auto &node_name = compute_node_info->GetNodeName();
        name_hash_id = MsprofGetHashId(node_name, std::string(node_name).length());
      }
      host_sch_infos_[node->node_id].itemId = name_hash_id;
    }
  }
}

void CannHostProfiler::Init() {
  if (is_host_prof_inited_) {
    return;
  }
  const auto execution_data = static_cast<const ExecutionData *>(
      extend_info_->executor->GetExeGraphExecutor(kMainExeGraph)->GetExecutionData());
  if (execution_data == nullptr) {
    GELOGW("[Cann Profiling] Execution data is empty, do not init profiler.");
    return;
  }
  GELOGI("[Cann Profiling] Init for cann host profiling.");
  InitHostSchData(*execution_data);
  is_host_prof_inited_ = true;
}

ge::Status CannHostProfiler::DoProf(ExecutorEvent event, const Node *node) {
  if (node->node_id >= host_sch_infos_.size()) {
    return ge::SUCCESS;
  }
  auto &host_sch_data = host_sch_infos_[node->node_id];
  if (host_sch_data.itemId == 0UL) {
    return ge::SUCCESS;
  }
  const auto prof_time = MsprofSysCycleTime();
  thread_local const auto tid = mmGetTid();
  if (event == kExecuteStart) {
    host_sch_data.threadId = tid;
    host_sch_data.beginTime = prof_time;
  } else {
    host_sch_data.endTime = prof_time;
    GE_ASSERT_MSPROF_OK(MsprofReportApi(true, &host_sch_data));
  }
  return ge::SUCCESS;
}

void CannHostProfiler::OnExecuteEvent(int32_t sub_exe_graph_type, CannHostProfiler *profiler, ExecutorEvent event,
                                      const void *node, KernelStatus result) {
  (void)result;
  (void)sub_exe_graph_type;
  if (event == kModelEnd || profiler == nullptr) {
    return;
  }

  if (event == kModelStart) {
    profiler->Init();
  } else {
    (void)profiler->DoProf(event, static_cast<const Node *>(node));
  }
}
}  // namespace gert
