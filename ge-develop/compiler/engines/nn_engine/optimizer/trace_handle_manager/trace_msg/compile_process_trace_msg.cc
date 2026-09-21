/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trace_handle_manager/trace_msg/compile_process_trace_msg.h"
#include <sstream>
#include "common/fe_utils.h"

namespace fe {
CompileProcessTraceMsg::CompileProcessTraceMsg(const size_t total_task_count, const size_t wait_task_count)
  : is_compile_end_(false), total_task_count_(total_task_count), wait_task_count_(wait_task_count) {}

CompileProcessTraceMsg::CompileProcessTraceMsg(const size_t total_task_count)
  : is_compile_end_(true), total_task_count_(total_task_count), wait_task_count_(0) {}

CompileProcessTraceMsg::~CompileProcessTraceMsg() {}

std::string CompileProcessTraceMsg::GenerateTraceMsg() {
  std::stringstream ss;
  if (is_compile_end_) {
    ss << "Finish subgraph compile. ThreadId:" << GetCurThreadIdStr() << "|";
    ss << "Total task cout:" << std::to_string(total_task_count_);
  } else {
    ss << "Compile process status. ThreadId:" << GetCurThreadIdStr() << "|";
    ss << "Total task cout:" << std::to_string(total_task_count_) << "|";
    ss << "Finished task cout:" << std::to_string(total_task_count_ - wait_task_count_) << "|";
    ss << "Waiting task cout:" << std::to_string(wait_task_count_);
  }

  return ss.str();
}
}
