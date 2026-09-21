/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "atracing_reporter.h"
#include "atrace_pub.h"
#include "graph/def_types.h"
namespace ge {
Status AtracingReporter::Report() const {
  if ((tracing_record_ == nullptr) || (handle_ < 0)) {
    GELOGW("Report failed as handle or record is invalid");
    return FAILED;
  }
  const auto msg = ToAtracingProfilingData();
  size_t buf_size = msg.size() > kMaxAtracingProfilingMsgSize ? kMaxAtracingProfilingMsgSize : msg.size();
  TraStatus trace_status = AtraceSubmit(handle_, PtrToPtr<char_t, void>(msg.c_str()), buf_size);
  GELOGD("Record atracing msg: %s success", msg.c_str());
  return (trace_status == TRACE_SUCCESS) ? SUCCESS : FAILED;
}

std::string AtracingReporter::ToAtracingProfilingData() const {
  std::string merged_msg;
  merged_msg.append("pid=")
      .append(std::to_string(tracing_record_->pid))
      .append(",tid=")
      .append(std::to_string(tracing_record_->thread))
      .append(",start=")
      .append(std::to_string(tracing_record_->start))
      .append(",duration=")
      .append(std::to_string(tracing_record_->duration)).append("ns")
      .append(",msg=");
  int32_t id = 0;
  for (const auto &msg : tracing_record_->tracing_msgs) {
    if (id != 0) {
      merged_msg.append(",");
    }
    merged_msg.append(msg);
    id++;
  }
  return merged_msg;
}
}
