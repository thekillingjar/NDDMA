/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_UTIL_DVPP_LOG_H_
#define DVPP_ENGINE_UTIL_DVPP_LOG_H_

#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#include "dlog_pub.h"
#include "util/dvpp_error_code.h"
#include "base/err_msg.h"
namespace dvpp {
const std::string DVPP_ERROR_CODE = "EN9999";

inline int32_t GetTid() {
  thread_local static const int32_t tid = syscall(__NR_gettid);
  return tid;
}

#ifdef RUN_TEST
#include <cstdio>

#define DVPP_ENGINE_LOG_DEBUG(fmt, ...)                      \
  do {                                                       \
    printf("[DEBUG][DVPP][ENGINE][%s:%d][tid:%d]" fmt "\n",  \
           &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_ENGINE_LOG_INFO(fmt, ...)                       \
  do {                                                       \
    printf("[INFO][DVPP][ENGINE][%s:%d][tid:%d]" fmt "\n",   \
           &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_ENGINE_LOG_WARNING(fmt, ...)                     \
  do {                                                        \
    printf("[WARNING][DVPP][ENGINE][%s:%d][tid:%d]" fmt "\n", \
           &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__);  \
  } while (false)

#define DVPP_ENGINE_LOG_ERROR(fmt, ...)                      \
  do {                                                       \
    printf("[ERROR][DVPP][ENGINE][%s:%d][tid:%d]" fmt "\n",  \
           &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_ENGINE_LOG_EVENT(fmt, ...)                      \
  do {                                                       \
    printf("[EVENT][DVPP][ENGINE][%s:%d][tid:%d]" fmt "\n",  \
           &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_REPORT_INNER_ERR_MSG(fmt, ...)                    \
  do {                                                       \
    printf("[ERROR][DVPP][ENGINE][%s:%d][tid:%d]" fmt "\n",  \
           &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_CHECK_NOTNULL(val)                       \
  do {                                                \
    if ((val) == nullptr) {                           \
      DVPP_REPORT_INNER_ERR_MSG("[%s] is null.", #val); \
      return DvppErrorCode::kInputParamNull;          \
    }                                                 \
  } while (false)

#else // ifdef RUN_TEST

#define DVPP_ENGINE_LOG_DEBUG(fmt, ...)                          \
  do {                                                           \
    dlog_debug(DVPP, "[ENGINE][%s:%d][tid:%d]" fmt,              \
               &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_ENGINE_LOG_INFO(fmt, ...)                          \
  do {                                                          \
    dlog_info(DVPP, "[ENGINE][%s:%d][tid:%d]" fmt,              \
              &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_ENGINE_LOG_WARNING(fmt, ...)                       \
  do {                                                          \
    dlog_warn(DVPP, "[ENGINE][%s:%d][tid:%d]" fmt,              \
              &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

#define DVPP_ENGINE_LOG_ERROR(fmt, ...)                          \
  do {                                                           \
    dlog_error(DVPP, "[ENGINE][%s:%d][tid:%d]" fmt,              \
               &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__); \
  } while (false)

// 不允许打印event日志，此处info+mask可以打印运行日志
#define DVPP_ENGINE_LOG_EVENT(fmt, ...)                              \
  do {                                                               \
    dlog_info((DVPP | RUN_LOG_MASK), "[ENGINE][%s:%d][tid:%d]" fmt,  \
              &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__);      \
  } while (false)

#define DVPP_REPORT_INNER_ERR_MSG(fmt, ...)                            \
  do {                                                               \
    dlog_error(DVPP, "[ENGINE][%s:%d][tid:%d]" fmt,                  \
               &__func__[0], __LINE__, GetTid(), ##__VA_ARGS__);     \
    REPORT_INNER_ERR_MSG(DVPP_ERROR_CODE.c_str(), fmt, ##__VA_ARGS__); \
  } while (false)

#define DVPP_CHECK_NOTNULL(val)                       \
  do {                                                \
    if ((val) == nullptr) {                           \
      DVPP_REPORT_INNER_ERR_MSG("[%s] is null.", #val); \
      return DvppErrorCode::kInputParamNull;          \
    }                                                 \
  } while (false)

#endif // ifdef RUN_TEST
} // namespace dvpp

#endif // DVPP_ENGINE_UTIL_DVPP_LOG_H_
