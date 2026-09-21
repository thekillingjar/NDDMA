/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "runtime/subscriber/executor_subscribers_scheduler.h"
#include "common/global_variables/diagnose_switch.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/profiler/ge_host_profiler.h"
#include "subscriber/profiler/cann_memory_profiler.h"
#include "subscriber/tracer/executor_tracer.h"
#include "subscriber/profiler/cann_host_profiler.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "subscriber/dumper/host_executor_dumper.h"

namespace gert {
namespace {
constexpr size_t kOnlyOneSubscriber = 1UL;
}  // namespace

// caller must ensure sub_exe_graph_type < kSubExeGraphTypeEnd
void ExecutorSubscribersScheduler::OnExecuteEvent(SubExeGraphType sub_exe_graph_type,
                                                  const ExecutorSubscribersScheduler *ins, ExecutorEvent event,
                                                  const void *node, KernelStatus result) {
  for (auto &subscriber_guarder : ins->working_sub_exe_graph_subscribers_) {
    auto &subscriber = subscriber_guarder->GetSubscriber();
    subscriber.callback(sub_exe_graph_type, subscriber.arg, event, node, result);
  }
}

ExecutorSubscriber &ExecutorSubscribersScheduler::GetSubscriber(SubExeGraphType sub_exe_graph_type) {
  // Subscriber which does not work will not be in working_sub_exe_graph_subscribers_
  working_sub_exe_graph_subscribers_.clear();
  for (const auto &subscriber_guarder : sub_exe_graph_subscribers_[sub_exe_graph_type]) {
    if (subscriber_guarder->IsEnabled()) {
      working_sub_exe_graph_subscribers_.emplace_back(subscriber_guarder);
    }
  }
  if (working_sub_exe_graph_subscribers_.size() == kOnlyOneSubscriber) {
    return working_sub_exe_graph_subscribers_[0]->GetSubscriber();
  }
  return subscriber_wrapper_;
}

void ExecutorSubscribersScheduler::Init(const std::shared_ptr<const SubscriberExtendInfo> &extend_info) {
  // profiler will be in a factory
  const auto is_ge_prof_enabled = []() -> bool {
    return GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kGeHost);
  };
  AddBuiltIn<GeHostProfiler>(BuiltInSubscriberType::kGeProfiling, 0UL, extend_info, kMainExeGraph, is_ge_prof_enabled);

  // 如果在打印 kernel trace 前其他 subscriber 下发了 Task (例如 kDumper), 那么 kernel trace 中的 task_id 会不准确
  // 未来新增 subscriber, 请把会额外下发 task的 subscriber 放在 kTracer 之后
  const auto is_tracer_enabled = []() -> bool { return GlobalTracer::GetInstance()->GetEnableFlags() != 0UL; };
  AddBuiltIn<ExecutorTracer>(BuiltInSubscriberType::kTracer, 0UL, extend_info, kSubExeGraphTypeEnd, is_tracer_enabled);

  const auto is_dump_enabled = []() -> bool { return (GlobalDumper::GetInstance()->IsEnableSubscribeDump()); };
  AddBuiltIn<ExecutorDumper>(BuiltInSubscriberType::kDumper, 0UL, extend_info, kMainExeGraph, is_dump_enabled);

  const auto is_cann_profiler_enabled = []() -> bool {
    return GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kTaskTime);
  };
  AddBuiltIn<CannProfilerV2>(BuiltInSubscriberType::kCannProfilerV2, 0UL, extend_info, kSubExeGraphTypeEnd,
                             is_cann_profiler_enabled);

  const auto is_cann_memory_prof_enabled = []() -> bool {
    return GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kMemory);
  };
  AddBuiltIn<CannMemoryProfiler>(BuiltInSubscriberType::kMemoryProfiler, 0UL, extend_info, kSubExeGraphTypeEnd,
                                 is_cann_memory_prof_enabled);

  const auto is_cann_host_prof_v2_enabled = []() -> bool {
    return GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kCannHost);
  };
  AddBuiltIn<CannHostProfiler>(BuiltInSubscriberType::kCannHostProfiler, 0UL, extend_info, kMainExeGraph,
                               is_cann_host_prof_v2_enabled);

  const auto is_host_dump_enabled = []() -> bool { return GlobalDumper::GetInstance()->IsEnable(DumpType::kHostDump); };
  AddBuiltIn<HostExecutorDumper>(BuiltInSubscriberType::kHostDumper, 0UL, extend_info, kMainExeGraph,
                                 is_host_dump_enabled);
}
}  // namespace gert
