/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dlog_pub.h"

#include <ctime>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

static int g_runlog_level = DLOG_INFO;
static int g_debuglog_level = DLOG_WARN;
static int g_enableEvent = 1;

#define __DO_PRINT(log_level)                        \
  do {                                               \
    const int FMT_BUFF_SIZE = 1024;                  \
    char fmt_buff[FMT_BUFF_SIZE] = {0};              \
    va_list valist;                                  \
    va_start(valist, fmt);                           \
    vsnprintf(fmt_buff, FMT_BUFF_SIZE, fmt, valist); \
    va_end(valist);                                  \
    struct timeval ts;                               \
    gettimeofday(&ts, 0);                            \
    time_t t = time(NULL);                           \
    struct tm *lt = localtime(&t);                   \
    printf("[%s]%04d-%02d-%02d-%02d:%02d:%02d.%03d.%03d %s \n", #log_level,  \
            (lt->tm_year + 1900), (lt->tm_mon + 1), lt->tm_mday, lt->tm_hour, \
            lt->tm_min, lt->tm_sec, static_cast<int>(ts.tv_usec / 1000),      \
            static_cast<int>(ts.tv_usec % 1000), fmt_buff); \
  } while (0)

void DlogRecord(int module_id, int level, const char *fmt, ...) {
  auto log_level = dlog_getlevel(module_id, nullptr);
  if (log_level > level) {
    return;
  }
  switch (level) {
    case DLOG_DEBUG:
      __DO_PRINT(DEBUG);
      break;
    case DLOG_INFO:
      __DO_PRINT(INFO);
      break;
    case DLOG_WARN:
      __DO_PRINT(WARN);
      break;
    case DLOG_ERROR:
      __DO_PRINT(ERROR);
      break;
    default:
      __DO_PRINT(UNKNOWN);
      break;
  }
}

void DlogVaList(int module_id, int level, const char *fmt, va_list valist) {
  auto log_level = dlog_getlevel(module_id, nullptr);
  if (log_level > level) {
    return;
  }
  const int FMT_BUFF_SIZE = 1024;
  char fmt_buff[FMT_BUFF_SIZE] = {0};
  vsnprintf(fmt_buff, FMT_BUFF_SIZE, fmt, valist);
  switch (level) {
    case DLOG_DEBUG:
      printf("[DEBUG]%s\n", fmt_buff);
      break;
    case DLOG_INFO:
      printf("[INFO]%s\n", fmt_buff);
      break;
    case DLOG_WARN:
      printf("[WARN]%s\n", fmt_buff);
      break;
    case DLOG_ERROR:
      printf("[ERROR]%s\n", fmt_buff);
      break;
    default:
      printf("[UNKNOWN]%s\n", fmt_buff);
      break;
  }
}

int CheckLogLevel(int module_id, int log_level_check) {
  auto log_level = dlog_getlevel(module_id, nullptr);
  return log_level <= log_level_check;
}

int DlogSetAttr(LogAttr logAttr) {
  return 0;
}

int dlog_getlevel(int module_id, int *enable_event) {
  if (enable_event != nullptr) {
    *enable_event = g_enableEvent;
  }
  return (module_id & RUN_LOG_MASK) != 0 ? g_runlog_level : g_debuglog_level;
}

int dlog_setlevel(int module_id, int level, int enable_event) {
  if ((module_id & RUN_LOG_MASK) != 0) {
    g_runlog_level = level;
  } else {
    g_debuglog_level = level;
  }
  g_enableEvent = enable_event;
  return 0;
}

void DlogFlush() {}

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus
int StackcoreSetSubdirectory(const char *subdir) {
  return 0;
}
#ifdef __cplusplus
}
#endif
