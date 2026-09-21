/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TE_FUSION_LOG_H
#define TE_FUSION_LOG_H
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>
#include <cstdlib>
#include <cstring>
#include "dlog_pub.h"
#include "securec.h"
#include "inc/te_fusion_error_code.h"
#include "base/err_msg.h"

namespace te {
namespace fusion {
/**
 * @ingroup tefusion
 * @brief log level fatal
 */
#define TE_FUSION_LOG_FATAL DLOG_ERROR

/**
 * @ingroup tefusion
 * @brief log level error
 */
#define TE_FUSION_LOG_ERROR DLOG_ERROR

/**
 * @ingroup tefusion
 * @brief log level warning
 */
#define TE_FUSION_LOG_WARNING DLOG_WARN

/**
 * @ingroup tefusion
 * @brief log level info
 */
#define TE_FUSION_LOG_INFO DLOG_INFO

/**
 * @ingroup tefusion
 * @brief log level debug
 */
#define TE_FUSION_LOG_DEBUG DLOG_DEBUG

inline uint64_t TeGetTid() {
    thread_local static uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
    return tid;
}

#ifdef TE_FUSION_DEBUG
constexpr const char* MODULE_NAME = "TEFUSION";
constexpr const char *g_levelStr[] = {"ERROR", "WARN", "INFO", "DEBUG"};
#define TE_FUSION_LOG(level, format, ...)                               \
    do {                                                                \
        fprintf(stdout, "[%s] [%s] [%s] [%s] [%s:%d] " format "\n", "", MODULE_NAME, te::fusion::g_levelStr[level], \
                __FUNCTION__, __FILE__, __LINE__, ##__VA_ARGS__);       \
    } while (0);
#else
#define TE_FUSION_LOG(level, format, ...)                               \
    do {                                                                \
        if (0 == CheckLogLevel(TEFUSION, level)) {                      \
            break;                                                      \
        }                                                               \
        if (level == TE_FUSION_LOG_DEBUG) {                             \
            dlog_debug(TEFUSION, "%lu %s " format "\n", TeGetTid(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                               \
        if (level == TE_FUSION_LOG_INFO) {                              \
            dlog_info(TEFUSION, "%lu %s " format "\n", TeGetTid(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                               \
        if (level == TE_FUSION_LOG_WARNING) {                           \
            dlog_warn(TEFUSION, "%lu %s " format "\n", TeGetTid(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                               \
        if ((level == TE_FUSION_LOG_ERROR) || (level == TE_FUSION_LOG_FATAL)) { \
            dlog_error(TEFUSION, "%lu %s " format "\n", TeGetTid(), __FUNCTION__, ##__VA_ARGS__); \
        }                                                               \
                                                                        \
    } while (0);
#endif

#define TE_FUSION_LOG_EXEC(level, format, ...)       \
    {                                                \
        TE_FUSION_LOG(level, format, ##__VA_ARGS__); \
    }

constexpr const int MAX_LOG_SIZE = 8192;
#define TE_FUSION_LOG_FULL(level, format, ...)                          \
    do {                                                                \
        if (0 == CheckLogLevel(TEFUSION, level)) {                      \
            break;                                                      \
        }                                                               \
        char msgbufxyz[MAX_LOG_SIZE];                                   \
        size_t msgmaxlen = (MSG_LENGTH - 200);                          \
        int rettmp = snprintf_s(msgbufxyz, sizeof(msgbufxyz),           \
                                  sizeof(msgbufxyz) - 1, format, ##__VA_ARGS__); \
        if (rettmp == -1) {                                             \
            msgbufxyz[sizeof(msgbufxyz) - 1] = '\0';                    \
        }                                                               \
        size_t msglength = std::strlen(msgbufxyz);                      \
        if (msglength < msgmaxlen) {                                    \
            TE_FUSION_LOG(level, "%s", msgbufxyz);                      \
            break;                                                      \
        }                                                               \
        char *msgchunkbegin = msgbufxyz;                                \
        char *msgchunkend = nullptr;                                    \
        while (msgchunkbegin < msgbufxyz + msglength) {                 \
            if (msgchunkbegin[0] == '\n') {                             \
                TE_FUSION_LOG(level, "");                               \
                msgchunkbegin += 1;                                     \
                continue;                                               \
            }                                                           \
            msgchunkend = std::strchr(msgchunkbegin, '\n');             \
            if (msgchunkend == nullptr) {                               \
                msgchunkend = msgchunkbegin + std::strlen(msgchunkbegin); \
            }                                                           \
            while (msgchunkend > msgchunkbegin) {                       \
                std::string msgchunk(msgchunkbegin,                     \
                                       std::min(msgmaxlen, static_cast<size_t>(msgchunkend - msgchunkbegin))); \
                TE_FUSION_LOG(level, "%s", msgchunk.c_str());           \
                msgchunkbegin += msgchunk.size();                       \
            }                                                           \
            msgchunkbegin += 1;                                         \
        }                                                               \
    } while (0)

#define TE_DBGLOG(format, ...)  TE_FUSION_LOG(TE_FUSION_LOG_DEBUG, format, ##__VA_ARGS__)
#define TE_INFOLOG(format, ...)  TE_FUSION_LOG(TE_FUSION_LOG_INFO, format, ##__VA_ARGS__)
#define TE_ERRLOG(format, ...)  TE_FUSION_LOG(TE_FUSION_LOG_ERROR, format, ##__VA_ARGS__)
#define TE_WARNLOG(format, ...)  TE_FUSION_LOG(TE_FUSION_LOG_WARNING, format, ##__VA_ARGS__)

#define TE_DBGLOGF(format, ...)  TE_FUSION_LOG_FULL(TE_FUSION_LOG_DEBUG, format, ##__VA_ARGS__)
#define TE_INFOLOGF(format, ...)  TE_FUSION_LOG_FULL(TE_FUSION_LOG_INFO, format, ##__VA_ARGS__)
#define TE_ERRLOGF(format, ...)  TE_FUSION_LOG_FULL(TE_FUSION_LOG_ERROR, format, ##__VA_ARGS__)
#define TE_WARNLOGF(format, ...)  TE_FUSION_LOG_FULL(TE_FUSION_LOG_WARNING, format, ##__VA_ARGS__)

#define REPORT_TE_INNER_ERROR(format, ...)                              \
    do {                                                                \
        REPORT_INNER_ERR_MSG(EM_INNER_ERROR, format, ##__VA_ARGS__);      \
        TE_ERRLOGF(format, ##__VA_ARGS__);                              \
    } while (0)

#define REPORT_TE_INNER_WARN(format, ...)                                \
    do {                                                                 \
        REPORT_INNER_ERR_MSG(EM_INNER_WARNING, format, ##__VA_ARGS__);     \
        TE_WARNLOGF(format, ##__VA_ARGS__);                              \
    } while (0)

} // namespace fusion
} // namespace te
#endif /* TE_FUSION_LOG_H */
