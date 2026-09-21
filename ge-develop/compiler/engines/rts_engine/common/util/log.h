/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef RTS_LOG_H
#define RTS_LOG_H

#include <cstdint>
#include <sys/syscall.h>
#include <unistd.h>
#include <cinttypes>
#include "base/err_msg.h"
#include "dlog_pub.h"

inline uint64_t GetTid() {
  thread_local static const uint64_t Tid = static_cast<uint64_t>(syscall(__NR_gettid));
  return Tid;
}
namespace rts {
#define RTS_ERROR_CODE "EE9999"
#define ERR_MODULE_GE "3U"

#ifdef RUN_TEST
#define RTS_LOGD(fmt, ...) printf("[DEBUG] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define RTS_LOGI(fmt, ...) printf("[INFO] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define RTS_LOGW(fmt, ...) printf("[WARN] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define RTS_LOGE(fmt, ...) printf("[ERROR] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define RTS_LOGEVENT(fmt, ...) printf("[EVENT] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define RTS_LOG_RUN_INFO(fmt, ...) printf("[INFO] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define RTS_LOG_RUN_WARN(fmt, ...) printf("[WARN] %s:%s:%d:" #fmt "\n", __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__)
#define RTS_REPORT_INNER_ERROR(fmt, ...) RTS_LOGE(fmt, ##__VA_ARGS__)
#define RTS_REPORT_CALL_ERROR(fmt, ...) RTS_LOGE(fmt, ##__VA_ARGS__)
#else
#define RTS_MODULE_NAME static_cast<int32_t>(RUNTIME)
#define RTS_RUN_MODULE_NAME static_cast<int32_t>(static_cast<uint32_t>(RUNTIME) | static_cast<uint32_t>(RUN_LOG_MASK))
#define RTS_LOGD(fmt, ...) dlog_debug(RTS_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)
#define RTS_LOGI(fmt, ...) dlog_info(RTS_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)
#define RTS_LOGW(fmt, ...) dlog_warn(RTS_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)
#define RTS_LOGE(fmt, ...)                                                                    \
  do {                                                                                        \
    dlog_error(RTS_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
  } while (0)
#define RTS_LOGEVENT(fmt, ...) \
  dlog_event(RTS_RUN_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)
#define RTS_LOG_RUN_INFO(fmt, ...) \
  dlog_info(RTS_RUN_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)
#define RTS_LOG_RUN_WARN(fmt, ...) \
  dlog_warn(RTS_RUN_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)

#define RTS_REPORT_INNER_ERROR(fmt, ...)                                                      \
  do {                                                                                        \
    dlog_error(RTS_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
    REPORT_INNER_ERR_MSG(RTS_ERROR_CODE, fmt, ##__VA_ARGS__);                                 \
  } while (0)

#define RTS_REPORT_CALL_ERROR(fmt, ...)                                                       \
  do {                                                                                        \
    dlog_error(RTS_MODULE_NAME, "[%s][tid:%lu]:" fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__); \
    REPORT_INNER_ERR_MSG(ERR_MODULE_GE, fmt, ##__VA_ARGS__);                                  \
  } while (0)

#endif
}  // namespace rts
#endif