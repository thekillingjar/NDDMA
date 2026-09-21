/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_LONG_TIME_TRACE_MSG_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_LONG_TIME_TRACE_MSG_H_

#include "trace_handle_manager/trace_msg/trace_msg_base.h"

namespace fe {
class LongTimeTraceMsg : public TraceMsgBase {
public:
  LongTimeTraceMsg(const bool is_fusion_op, const int64_t op_id, const std::string &op_type,
                   const size_t task_wait_second);
  ~LongTimeTraceMsg();
  std::string GenerateTraceMsg() override;

private:
  bool is_fusion_op_;
  int64_t op_id_;
  std::string op_type_;
  size_t task_wait_second_;
};
}
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_TRACE_HANDLE_MANAGER_TRACE_MSG_LONG_TIME_TRACE_MSG_H_
