/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_HANDLE_MANAGER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_HANDLE_MANAGER_H_

#include <string>
#include <map>
#include <mutex>
#include "atrace_types.h"
#include "trace_handle_manager/trace_msg/trace_msg_base.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class TraceHandleManager {
public:
  TraceHandleManager(const TraceHandleManager &) = delete;
  TraceHandleManager &operator=(const TraceHandleManager &) = delete;
  static TraceHandleManager& Instance();
  Status Initialize();
  void Finalize();
  void AddSubGraphTraceHandle();
  void SubmitGlobalTrace(const std::string &trace_msg) const;
  void SubmitGlobalTrace(const TraceMsgBasePtr &trace_msg) const;
  void SubmitStatisticsTrace(const std::string &trace_msg) const;

private:
  TraceHandleManager();
  ~TraceHandleManager();
  static bool SubmitTrace(const TraHandle &trace_handle, const std::string &trace_msg);
  void SaveAndDestroyTraceHandle();
  bool is_init_;
  TraHandle global_handle_;
  TraHandle statistics_handle_;
  TraEventHandle finalize_event_handle_;
  std::map<uint64_t, TraHandle> subgraph_handle_map_;
  std::map<uint64_t, TraEventHandle> subgraph_event_map_;
  mutable std::mutex subgraph_mutex_;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_HANDLE_MANAGER_H_
