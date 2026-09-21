/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trace_handle_manager/trace_msg/long_time_trace_msg.h"
#include <sstream>
#include "common/fe_utils.h"

namespace fe {
LongTimeTraceMsg::LongTimeTraceMsg(const bool is_fusion_op, const int64_t op_id, const std::string &op_type,
                                   const size_t task_wait_second)
  : is_fusion_op_(is_fusion_op), op_id_(op_id), op_type_(op_type), task_wait_second_(task_wait_second) {}

LongTimeTraceMsg::~LongTimeTraceMsg() {}

std::string LongTimeTraceMsg::GenerateTraceMsg() {
  std::stringstream ss;
  ss << "ThreadId:" << std::to_string(GetCurThreadId()) << "|";
  if (is_fusion_op_) {
    ss << "FusionOp:";
  } else {
    ss << "SingleOp:";
  }
  ss << op_type_ << "_" << std::to_string(op_id_) << "|";
  ss << "Task wait second:" << std::to_string(task_wait_second_);
  return ss.str();
}
}
