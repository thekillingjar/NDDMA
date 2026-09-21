/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_MEMORY_PROFILER_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_MEMORY_PROFILER_H_
#include <atomic>

#include "base_executor_profiler.h"
#include "exe_graph/lowering/graph_frame.h"
#include "core/execution_data.h"
#include "aprof_pub.h"

namespace gert {
class CannMemoryProfiler : public BaseExecutorProfiler {
 public:
  static void OnExecuteEvent(int32_t sub_exe_graph_type, CannMemoryProfiler *profiler, ExecutorEvent event,
                             const void *node, KernelStatus result);
  explicit CannMemoryProfiler(const std::shared_ptr<const SubscriberExtendInfo> &extend_info);

  void Init();
 private:
  ge::graphStatus DoProf(const Node *node, const int32_t subgraph_type);
  std::shared_ptr<const SubscriberExtendInfo> extend_info_{nullptr};
  bool is_device_prof_inited_{false};
  std::vector<std::vector<ProfExtendInfo>> prof_extend_info_vec_;
  MsprofAdditionalInfo task_memory_info_;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_MEMORY_PROFILER_H_
