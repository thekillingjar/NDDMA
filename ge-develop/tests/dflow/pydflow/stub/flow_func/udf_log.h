/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMMON_UDF_LOG_H
#define COMMON_UDF_LOG_H

#include <cstdint>
#include <unistd.h>
#include <sys/syscall.h>

namespace FlowFunc {
inline int64_t GetTid() {
  thread_local static const int64_t tid = static_cast<int64_t>(syscall(__NR_gettid));
  return tid;
}

#define UDF_LOG_DEBUG(fmt, ...) \
  printf("[DEBUG][%s][tid:%ld]: " fmt "\n", __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_LOG_INFO(fmt, ...) printf("[INFO][%s][tid:%ld]: " fmt "\n", __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_LOG_WARN(fmt, ...) printf("[WARN][%s][tid:%ld]: " fmt "\n", __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_LOG_ERROR(fmt, ...) \
  printf("[ERROR][%s][tid:%ld]: " fmt "\n", __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
#define UDF_LOG_EVENT(fmt, ...) \
  printf("[EVENT][%s][tid:%ld]: " fmt "\n", __FUNCTION__, FlowFunc::GetTid(), ##__VA_ARGS__)
}  // namespace FlowFunc

#endif  // COMMON_UDF_LOG_H