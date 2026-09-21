/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_
#define INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include "dlog_pub.h"
#include "mmpa_api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GE_GET_ERRORNO_STR "GE_ERRORNO_STR"
#define GE_GET_ERROR_LOG_HEADER "[GE][MODULE]"
#define GE_MODULE_NAME ((int32_t)GE)

#ifndef FAILED
#define FAILED (-1)
#endif
// trace status of log
enum TraceStatus { TRACE_INT = 0, TRACE_RUNNING, TRACE_WAITTING, TRACE_STOP };

static inline uint64_t GetTid(void) {
  const uint64_t tid = (uint64_t)(mmGetTaskId());
  return tid;
}

#ifdef RUN_TEST
#define GELOGD(fmt, ...)                                                                          \
do {                                                                                              \
    printf("[DEBUG][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGI(fmt, ...)                                                                         \
  do {                                                                                           \
    printf("[INFO][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGW(fmt, ...)                                                                            \
  do {                                                                                              \
    printf("[WARNING][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)

#define GELOGE(ERROR_CODE, fmt, ...)                                                                                \
  do {                                                                                                              \
    printf("[ERROR][%s:%d]%ld ErrorNo:%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), (long int)ERROR_CODE, \
           ##__VA_ARGS__);                                                                                          \
  } while (false)

#define GEEVENT(fmt, ...)                                                                         \
  do {                                                                                            \
    printf("[EVENT][%s:%d]%ld " fmt "\n", __FILE__, __LINE__, (long int)GetTid(), ##__VA_ARGS__); \
  } while (false)
#else
#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    DlogRecord(GE_MODULE_NAME, DLOG_DEBUG, "[DEBUG][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,       \
              (long int)GetTid(), ##__VA_ARGS__);                                                  \
  } while (false)

#define GELOGI(fmt, ...)                                                                           \
  do {                                                                                             \
    DlogRecord(GE_MODULE_NAME, DLOG_INFO, "[INFO][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,         \
              (long int)GetTid(), ##__VA_ARGS__);                                                  \
  } while (false)

#define GELOGW(fmt, ...)                                                                           \
  do {                                                                                             \
    DlogRecord(GE_MODULE_NAME, DLOG_WARN, "[WARNING][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,      \
              (long int)GetTid(), ##__VA_ARGS__);                                                  \
  } while (false)

#define GEEVENT(fmt, ...)                                                                          \
  do {                                                                                             \
    DlogRecord(GE_MODULE_NAME, DLOG_EVENT, "[Event][%s:%d]%ld " fmt "\n", __FILE__, __LINE__,       \
              (long int)GetTid(), ##__VA_ARGS__);                                                  \
  } while (false)

#define GELOGE(ERROR_CODE, fmt, ...)                                                               \
  do {                                                                                             \
    DlogRecord(GE_MODULE_NAME, DLOG_ERROR, "[ERROR][%s:%d]%ld:ErrorNo:%u(%s)%s " fmt "\n",          \
              __FILE__, __LINE__, (long int)GetTid(), (uint32_t)ERROR_CODE,                        \
              GE_GET_ERRORNO_STR, GE_GET_ERROR_LOG_HEADER, ##__VA_ARGS__);                         \
  } while (false)
#endif

#ifdef __cplusplus
}
#endif
#endif  // INC_FRAMEWORK_EXECUTOR_C_GE_LOG_H_
