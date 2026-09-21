/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_LOG_H
#define FLOW_FUNC_LOG_H
#include <cstdarg>
#include "flow_func_defines.h"

namespace FlowFunc {
enum class FLOW_FUNC_VISIBILITY FlowFuncLogType { DEBUG_LOG = 0, RUN_LOG = 1 };
enum class FLOW_FUNC_VISIBILITY FlowFuncLogLevel { DEBUG = 0, INFO = 1, WARN = 2, ERROR = 3 };

#define FLOW_FUNC_LOG_DEBUG(fmt, ...)                                           \
  do {                                                                          \
    printf("[DEBUG][%s:%d][%s]%s: " fmt "\n", __FILE__, __LINE__, __FUNCTION__, \
           FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);         \
  } while (0)

#define FLOW_FUNC_LOG_INFO(fmt, ...)                                           \
  do {                                                                         \
    printf("[INFO][%s:%d][%s]%s: " fmt "\n", __FILE__, __LINE__, __FUNCTION__, \
           FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);        \
  } while (0)

#define FLOW_FUNC_LOG_WARN(fmt, ...)                                           \
  do {                                                                         \
    printf("[WARN][%s:%d][%s]%s: " fmt "\n", __FILE__, __LINE__, __FUNCTION__, \
           FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);        \
  } while (0)

#define FLOW_FUNC_LOG_ERROR(fmt, ...)                                           \
  do {                                                                          \
    printf("[ERROR][%s:%d][%s]%s: " fmt "\n", __FILE__, __LINE__, __FUNCTION__, \
           FlowFunc::FlowFuncLogger::GetLogExtHeader(), ##__VA_ARGS__);         \
  } while (0)

class FLOW_FUNC_VISIBILITY FlowFuncLogger {
 public:
  static FlowFuncLogger &GetLogger(FlowFuncLogType type);

  static const char *GetLogExtHeader();

  FlowFuncLogger() = default;

  explicit FlowFuncLogger(FlowFuncLogType type) : type_(type) {}

  ~FlowFuncLogger() = default;

  virtual bool IsLogEnable(FlowFuncLogLevel level) {
    return true;
  }

  virtual void Error(const char *fmt, ...) {
    va_list argList;
    va_start(argList, fmt);
    printf("[ERROR][%s:%d]:", __FILE__, __LINE__);
    vprintf(fmt, argList);
    printf("\n");
    va_end(argList);
  }

  virtual void Warn(const char *fmt, ...) {
    va_list argList;
    va_start(argList, fmt);
    printf("[WARN][%s:%d]:", __FILE__, __LINE__);
    vprintf(fmt, argList);
    printf("\n");
    va_end(argList);
  }

  virtual void Info(const char *fmt, ...) {
    va_list argList;
    va_start(argList, fmt);
    printf("[INFO][%s:%d]:", __FILE__, __LINE__);
    vprintf(fmt, argList);
    printf("\n");
    va_end(argList);
  }

  virtual void Debug(const char *fmt, ...) {
    va_list argList;
    va_start(argList, fmt);
    printf("[DEBUG][%s:%d]:", __FILE__, __LINE__);
    vprintf(fmt, argList);
    printf("\n");
    va_end(argList);
  }

 private:
  FlowFuncLogType type_;
};
}  // namespace FlowFunc
#endif  // FLOW_FUNC_LOG_H
