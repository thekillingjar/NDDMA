/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_ATRACING_REPORTER_H
#define AUTOFUSE_ATRACING_REPORTER_H
#include "common/tracing_type.h"
#include "atrace_types.h"
#include "atrace_pub.h"
namespace ge {
class AtracingReporter {
 public:
  AtracingReporter(const TraHandle handle, const TracingRecord *tracing_record)
      : tracing_record_(tracing_record), handle_(handle) {
  }
  Status Report() const;

 private:
  std::string ToAtracingProfilingData() const;
  const TracingRecord *tracing_record_;
  TraHandle handle_;
};
}  // namespace ge

#endif  // AUTOFUSE_ATRACING_REPORTER_H
