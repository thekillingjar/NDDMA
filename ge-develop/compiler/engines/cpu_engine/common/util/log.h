/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_LOG_H
#define AICPU_LOG_H

#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>
#include "log_weak.h"
#include "base/err_msg.h"
#include "dlog_pub.h"

inline uint64_t GetTid() {
  thread_local static const uint64_t Tid = static_cast<uint64_t>(syscall(__NR_gettid));
  return Tid;
}

namespace aicpu {
#define AICPUE_ERROR_CODE "E39999"

#ifdef RUN_TEST
#define AICPUE_LOGD(fmt, ...)                                             \
  printf("[DEBUG] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)
#define AICPUE_LOGI(fmt, ...)                                            \
  printf("[INFO] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)
#define AICPUE_LOGW(fmt, ...)                                            \
  printf("[WARN] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)
#define AICPUE_LOGE(fmt, ...)                                             \
  printf("[ERROR] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)
#define AICPUE_LOG_RUN_INFO(fmt, ...)                                         \
  printf("[INFO] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)
#define AICPUE_LOG_RUN_WARN(fmt, ...)                                         \
  printf("[WARN] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, \
         ##__VA_ARGS__)
#define AICPU_REPORT_INNER_ERR_MSG(fmt, ...) AICPUE_LOGE(fmt, ##__VA_ARGS__)
#else
#define AICPU_MODULE_NAME static_cast<int32_t>(AICPU)
#define AICPU_RUN_MODULE_NAME static_cast<int32_t>(static_cast<uint32_t>(AICPU) | static_cast<uint32_t>(RUN_LOG_MASK))
#define AICPUE_LOGD(fmt, ...)                                      \
  if ((&CheckLogLevel != nullptr) && (&DlogRecord != nullptr)) {   \
    dlog_debug(AICPU_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
  }
#define AICPUE_LOGI(fmt, ...)                                     \
  if ((&CheckLogLevel != nullptr) && (&DlogRecord != nullptr)) {   \
    dlog_info(AICPU_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
  }
#define AICPUE_LOGW(fmt, ...)                                     \
  if ((&CheckLogLevel != nullptr) && (&DlogRecord != nullptr)) {   \
    dlog_warn(AICPU_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
  }
#define AICPUE_LOGE(fmt, ...)                                      \
  do {                                                             \
    if (&DlogRecord != nullptr) {   \
      dlog_error(AICPU_MODULE_NAME, "[%s][tid:%lu]%s:" fmt, __FUNCTION__, GetTid(), \
                 ##__VA_ARGS__);   \
    } \
  } while(0)
#define AICPUE_LOG_RUN_INFO(fmt, ...)                                  \
  if ((&CheckLogLevel != nullptr) && (&DlogRecord != nullptr)) {   \
    dlog_info(AICPU_RUN_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
  }
#define AICPUE_LOG_RUN_WARN(fmt, ...)                                  \
  if ((&CheckLogLevel != nullptr) && (&DlogRecord != nullptr)) {   \
    dlog_warn(AICPU_RUN_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
  }

#define AICPU_REPORT_INNER_ERR_MSG(fmt, ...) \
  do {                                                                                            \
    dlog_error(AICPU_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
    REPORT_INNER_ERR_MSG(AICPUE_ERROR_CODE, fmt, ##__VA_ARGS__);                                    \
  } while (0)

#endif
}  // namespace aicpu
#endif
