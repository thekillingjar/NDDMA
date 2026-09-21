/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_LOG_H_
#define OP_LOG_H_
#include <iostream>
#include <string>
#include <cstdarg>
#include <stdlib.h>
const char *g_open_log = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
const char *g_log_level = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
const char *g_event_enable = std::getenv("ASCEND_GLOBAL_EVENT_ENABLE");
void OP_LOGD(const std::string &op_name, const std::string &log, ...) {
  const char* SLOG_PRINT = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
  if (SLOG_PRINT != nullptr) {
    if (std::atoi(SLOG_PRINT) == 0) {
      return;
    }
    const char* LOG_LEVEL = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (LOG_LEVEL != nullptr) {
      if (std::atoi(LOG_LEVEL) <= 0) {
        std::string slog = "[" + op_name + "][DEBUG]" + log + "\n";
        va_list args;
        va_start(args, log);
        vprintf(slog.c_str(), args);
        va_end(args);
      }
    }
  }
}


void OP_LOGI(const std::string &op_name, const std::string &log, ...) {
  const char* SLOG_PRINT = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
  if (SLOG_PRINT != nullptr) {
    if (std::atoi(SLOG_PRINT) == 0) {
      return;
    }
    const char* LOG_LEVEL = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (LOG_LEVEL != nullptr) {
      if (std::atoi(LOG_LEVEL) <= 1) {
        std::string slog = "[" + op_name + "][INFO]" + log + "\n";
        va_list args;
        va_start(args, log);
        vprintf(slog.c_str(), args);
        va_end(args);
      }
    }
  }
}


void OP_LOGW(const std::string &op_name, const std::string &log, ...) {
  const char* SLOG_PRINT = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
  if (SLOG_PRINT != nullptr) {
    if (std::atoi(SLOG_PRINT) == 0) {
      return;
    }
    const char* LOG_LEVEL = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (LOG_LEVEL != nullptr) {
      if (std::atoi(LOG_LEVEL) <= 2) {
        std::string slog = "[" + op_name + "][WARNING]" + log + "\n";
        va_list args;
        va_start(args, log);
        vprintf(slog.c_str(), args);
        va_end(args);
      }
    }
  }
}


void OP_LOGE(const std::string &op_name, const std::string &log, ...) {
  const char* SLOG_PRINT = std::getenv("ASCEND_SLOG_PRINT_TO_STDOUT");
  if (SLOG_PRINT != nullptr) {
    if (std::atoi(SLOG_PRINT) == 0) {
      return;
    }
    const char* LOG_LEVEL = std::getenv("ASCEND_GLOBAL_LOG_LEVEL");
    if (LOG_LEVEL != nullptr) {
      if (std::atoi(LOG_LEVEL) <= 3) {
        std::string slog = "[" + op_name + "][ERROR]" + log + "\n";
        va_list args;
        va_start(args, log);
        vprintf(slog.c_str(), args);
        va_end(args);
      }
    }
  }
}

void OP_EVENT(const std::string &op_name, const std::string &log, ...) {
  if (g_open_log == nullptr) {
    return;
  }
  if (g_event_enable != nullptr) {
    if (std::atoi(g_event_enable) == 1) {
      std::string slog = "[" + op_name + "][EVENT]" + log + "\n";
      va_list args;
      va_start(args, log);
      vprintf(slog.c_str(), args);
      va_end(args);
    }
  }
}
#endif