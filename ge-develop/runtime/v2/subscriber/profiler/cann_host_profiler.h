/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_HOST_PROFILER_H_
#define AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_HOST_PROFILER_H_

#include "base_executor_profiler.h"
#include "core/execution_data.h"
#include "aprof_pub.h"

namespace gert {
class CannHostProfiler : public BaseExecutorProfiler {
 public:
  explicit CannHostProfiler(const std::shared_ptr<const SubscriberExtendInfo> &extend_info);
  void Init();
  static void OnExecuteEvent(int32_t sub_exe_graph_type, CannHostProfiler *profiler, ExecutorEvent event,
                             const void *node, KernelStatus result);
  void InitHostSchData(const ExecutionData &execution_data);
  ge::Status DoProf(ExecutorEvent event, const Node *node);
  bool IsHostProfInited() const {
    return is_host_prof_inited_;
  }

  const std::vector<MsprofApi> &GetHostSchData() const {
    return host_sch_infos_;
  }

 private:
  std::shared_ptr<const SubscriberExtendInfo> extend_info_{nullptr};
  std::vector<MsprofApi> host_sch_infos_{};
  bool is_host_prof_inited_{false};
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_SUBSCRIBER_PROFILER_CANN_HOST_PROFILER_H_
