/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_CANN_TRACING_PROFILER_H
#define AIR_CXX_CANN_TRACING_PROFILER_H
#include "subscriber/profiler/base_executor_profiler.h"
#include "exe_graph/lowering/graph_frame.h"
#include "core/execution_data.h"
#include "aprof_pub.h"

namespace gert {
class CannTracingProfiler : public BaseExecutorProfiler {
 public:
  static void OnExecuteEvent(int32_t sub_exe_graph_type, CannTracingProfiler *profiler, ExecutorEvent event,
                             const void *node, KernelStatus result);
  void Init();
  explicit CannTracingProfiler(SubscriberExtendInfo extend_info);
  void IncreaseIterationNum() {
    ++iteration_num_;
  }
  ge::Status ReportTraceInfo(uint16_t tag_id, const Node *node);
  ge::Status ReportStartTraceInfo(const Node *node);
  ge::Status ReportEndTraceInfo(const Node *node);
 private:
  SubscriberExtendInfo extend_info_;
  std::map<uint64_t, TraceAttr> node_ids_to_attrs_;
  uint64_t iteration_num_{1UL};
  ContinuousVector *rt_streams_{nullptr};
};
}  // namespace gert
#endif  // AIR_CXX_CANN_TRACING_PROFILER_H
