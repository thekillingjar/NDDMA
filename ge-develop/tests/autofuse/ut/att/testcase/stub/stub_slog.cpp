/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <map>

#define DLOG_DEBUG 0x0      // debug level id
#define DLOG_INFO  0x1      // info level id
#define DLOG_WARN  0x2      // warning level id
#define DLOG_ERROR 0x3      // error level id

const char *g_open_log = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
const char *g_log_level = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");

#define __DO_PRINT(LEVEL)                            \
  do {                                               \
    const int FMT_BUFF_SIZE = 1024;                  \
    char fmt_buff[FMT_BUFF_SIZE] = {0};              \
    va_list valist;                                  \
    va_start(valist, fmt);                           \
    vsnprintf(fmt_buff, FMT_BUFF_SIZE, fmt, valist); \
    va_end(valist);                                  \
    printf("%s %s \n",log_level_map[LEVEL].c_str(), fmt_buff);                  \
  } while (false)

void DlogRecord(int moduleId, int level, const char *fmt, ...);
void DlogRecord(int moduleId, int level, const char *fmt, ...) {
  static std::map<int, std::string> log_level_map {{DLOG_ERROR, "[ERROR]"},
                                                   {DLOG_WARN, "[WARN]"},
                                                   {DLOG_INFO, "[INFO]"},
                                                   {DLOG_DEBUG, "[DEBUG]"}};
  if (g_log_level == nullptr) {
    return;
  }
  if (std::atoi(g_log_level) > level) {
    return;
  }
  __DO_PRINT(level);
}