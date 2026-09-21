/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_COMPILE_PROCESS_TRACE_MSG_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_COMPILE_PROCESS_TRACE_MSG_H_

#include "trace_handle_manager/trace_msg/trace_msg_base.h"

namespace fe {
class CompileProcessTraceMsg : public TraceMsgBase {
public:
  CompileProcessTraceMsg(const size_t total_task_count, const size_t wait_task_count);
  explicit CompileProcessTraceMsg(const size_t total_task_count);
  ~CompileProcessTraceMsg();
  std::string GenerateTraceMsg() override;

private:
  bool is_compile_end_;
  size_t total_task_count_;
  size_t wait_task_count_;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_COMPILE_PROCESS_TRACE_MSG_H_
