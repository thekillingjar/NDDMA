/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_GRAPH_EXECUTOR_TRACER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_GRAPH_EXECUTOR_TRACER_H_
#include <utility>
#include "exe_graph/lowering/value_holder.h"
class ExecutorTracerOn {
 public:
  ExecutorTracerOn() {
    old_log_level_ = dlog_getlevel(GE_MODULE_NAME, &old_event_level_);
    dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, old_event_level_);
  }
  ~ExecutorTracerOn() {
    dlog_setlevel(GE_MODULE_NAME, old_event_level_, old_event_level_);
  }
 private:
  int old_log_level_;
  int old_event_level_;
};
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_COMMON_GRAPH_EXECUTOR_TRACER_H_
