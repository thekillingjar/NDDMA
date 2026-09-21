/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_GE_HOST_PROFILER_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_GE_HOST_PROFILER_H_
#include "base_executor_profiler.h"
#include "core/execution_data.h"

namespace gert {
class GeHostProfiler : public BaseExecutorProfiler {
 public:
  static void OnExecuteEvent(int32_t sub_exe_graph_type, GeHostProfiler *profiler, ExecutorEvent event,
                             const void *node, KernelStatus result);

  explicit GeHostProfiler(const std::shared_ptr<const SubscriberExtendInfo> &extend_info);

  void Init();

  void Record(const Node *node, ExecutorEvent event) const {
    if (event == kModelStart || event == kModelEnd) {
      GetGlobalProf()->Record(profiling::kModel, profiling::kExecute, event, std::chrono::system_clock::now());
      return;
    }
    const auto prof_extend_info = prof_extend_infos_[node->node_id];
    GetGlobalProf()->Record(prof_extend_info.node_name_idx, prof_extend_info.kernel_type_idx, event,
                            std::chrono::system_clock::now());
  }

 private:
  std::shared_ptr<const SubscriberExtendInfo> extend_info_{nullptr};
  bool is_inited_{false};
  std::vector<ProfExtendInfo> prof_extend_infos_{};
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_GE_HOST_PROFILER_H_
