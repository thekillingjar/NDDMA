/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FFTS_ENGINE_INC_FFTS_LOG_H_
#define FFTS_ENGINE_INC_FFTS_LOG_H_

#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <string>
#include "dlog_pub.h"
#include "inc/ffts_error_codes.h"
#include "base/err_msg.h"
namespace ffts {
/** Assigned FFTS name in log */
const std::string FFTS_MODULE_NAME = "FFTS";
}
inline uint64_t FFTSGetTid() {
  thread_local static const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
  return tid;
}

#define D_FFTS_LOGD(MOD_NAME, fmt, ...) dlog_debug(static_cast<int32_t>(FFTS), "%lu %s:" fmt "\n", FFTSGetTid(), \
                                                   __FUNCTION__, ##__VA_ARGS__)
#define D_FFTS_LOGI(MOD_NAME, fmt, ...) dlog_info(static_cast<int32_t>(FFTS), "%lu %s:" fmt "\n", FFTSGetTid(), \
                                                  __FUNCTION__, ##__VA_ARGS__)
#define D_FFTS_LOGW(MOD_NAME, fmt, ...) dlog_warn(static_cast<int32_t>(FFTS), "%lu %s:" fmt "\n", FFTSGetTid(), \
                                                  __FUNCTION__, ##__VA_ARGS__)
#define D_FFTS_LOGE(MOD_NAME, fmt, ...) dlog_error(static_cast<int32_t>(FFTS), "%lu %s:" fmt "\n", FFTSGetTid(), \
                                                   __FUNCTION__, ##__VA_ARGS__)

#define FFTS_LOGDEBUG(...) D_FFTS_LOGD(FFTS_MODULE_NAME, __VA_ARGS__)
#define FFTS_LOGI(...) D_FFTS_LOGI(FFTS_MODULE_NAME, __VA_ARGS__)
#define FFTS_LOGW(...) D_FFTS_LOGW(FFTS_MODULE_NAME, __VA_ARGS__)
#define FFTS_LOGE(...) D_FFTS_LOGE(FFTS_MODULE_NAME, __VA_ARGS__)

constexpr const int FFTS_MAX_LOG_SIZE = 8192;
constexpr const int FFTS_MSG_LEN = 1024;
constexpr const size_t FFTS_MSG_MAX_LEN = FFTS_MSG_LEN - 200;

#define FFTS_LOGD(...)                                                   \
  do {                                                                        \
    if (CheckLogLevel(static_cast<int32_t>(FFTS), DLOG_DEBUG) != 1) {         \
      break;                                                                  \
    }                                                                         \
    char msgbuf[FFTS_MAX_LOG_SIZE];                                           \
    if (snprintf_s(msgbuf, sizeof(msgbuf), sizeof(msgbuf) - 1, ##__VA_ARGS__) == -1) {   \
      msgbuf[sizeof(msgbuf) - 1] = '\0';                                      \
    }                                                                         \
    size_t msg_len = std::strlen(msgbuf);                                     \
    if (msg_len < FFTS_MSG_MAX_LEN) {                                         \
      FFTS_LOGDEBUG(__VA_ARGS__);                                             \
      break;                                                                  \
    }                                                                         \
    char *msg_chunk_begin = msgbuf;                                           \
    char *msg_chunk_end = nullptr;                                            \
    while (msg_chunk_begin < msgbuf + msg_len) {                              \
      if (msg_chunk_begin[0] == '\n') {                                       \
        FFTS_LOGDEBUG("");                                                    \
        msg_chunk_begin += 1;                                                 \
        continue;                                                             \
      }                                                                       \
      msg_chunk_end = std::strchr(msg_chunk_begin, '\n');                     \
      if (msg_chunk_end == nullptr) {                                         \
        msg_chunk_end = msg_chunk_begin + std::strlen(msg_chunk_begin);       \
      }                                                                       \
      while (msg_chunk_end > msg_chunk_begin) {                               \
        std::string msg_chunk(msg_chunk_begin,                                \
                              std::min(FFTS_MSG_MAX_LEN, static_cast<size_t>(msg_chunk_end - msg_chunk_begin))); \
        FFTS_LOGDEBUG("%s", msg_chunk.c_str());                               \
        msg_chunk_begin += msg_chunk.size();                                  \
      }                                                                       \
      msg_chunk_begin += 1;                                                   \
    }                                                                         \
  } while (0)

#define FFTS_LOGI_IF(cond, ...) \
  if ((cond)) {               \
    FFTS_LOGI(__VA_ARGS__);     \
  }

#define FFTS_LOGE_IF(cond, ...) \
  if ((cond)) {               \
    FFTS_LOGE(__VA_ARGS__);     \
  }

#define FFTS_CHECK(cond, log_func, return_expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      return_expr;                            \
    }                                         \
  } while (0)

// If failed to make_shared, then print log and execute the statement.
#define FFTS_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                         \
    try {                                      \
      exec_expr0;                              \
    } catch (...) {                            \
      FFTS_LOGE("Make shared failed");           \
      exec_expr1;                              \
    }                                          \
  } while (0)

#define FFTS_CHECK_NOTNULL(val)                           \
  do {                                                  \
    if ((val) == nullptr) {                             \
      FFTS_LOGE("Parameter[%s] must not be null.", #val); \
      return ffts::PARAM_INVALID;                         \
    }                                                   \
  } while (0)

#define REPORT_FFTS_ERROR(fmt, ...)  \
  do {                                                                \
    REPORT_INNER_ERR_MSG(EM_INNER_ERROR.c_str(), fmt, ##__VA_ARGS__); \
    FFTS_LOGE(fmt, ##__VA_ARGS__);                        \
  } while (0)
#endif  // FFTS_ENGINE_INC_FFTS_LOG_H_
