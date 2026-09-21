/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef METADEF_CXX_INC_GRAPH_ASCENDC_IR_ASCENDC_IR_CHECK_H_
#define METADEF_CXX_INC_GRAPH_ASCENDC_IR_ASCENDC_IR_CHECK_H_
#include "framework/common/debug/ge_log.h"
#include "common/checker.h"

namespace ge {
class AscIRException : public std::exception {
 public:
  struct Info {
    graphStatus error_code;
    std::string error_msg;
  };
  explicit AscIRException(const Info &info);
  const Info &GetInfo() const;
  const char *what() const noexcept override {
    return info_.error_msg.c_str();
  }
 private:
  Info info_;
};
}

inline bool IsVarNameValidAllowEmpty(const std::string &str) {
  if (str.empty()) {
    return true;
  }
  // 首字符必须是字母或下划线
  char first = str[0];
  if (!std::isalpha(static_cast<unsigned char>(first)) && first != '_') {
    return false;
  }

  // 后续字符只能是字母、数字或下划线
  for (size_t i = 1U; i < str.size(); ++i) {
    char c = str[i];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
      return false;
    }
  }

  return true;
}

#define CHECK_NOTNULL_WITH_THROW_EXCEPTION(val, ...)                                        \
 ASCIR_ASSERT_NOTNULL(val, __VA_ARGS__)

#define CHECK_BOOL_WITH_THROW_EXCEPTION(val, ...)                               \
 ASCIR_ASSERT((val), __VA_ARGS__)

#define ASCIR_ASSERT(exp, ...) \
  do {                                                                                                                 \
    if (!(exp)) {                                                                                                      \
      auto msg = CreateErrorMsg(__VA_ARGS__);                                                                          \
      if (msg.empty()) {                                                                                               \
        REPORT_INNER_ERR_MSG("E19999", "Assert %s failed", #exp);                                                        \
        GELOGE(ge::FAILED, "Assert %s failed", #exp);                                                                  \
        throw ge::AscIRException({ge::FAILED, #exp});                                                                  \
      } else {                                                                                                         \
        REPORT_INNER_ERR_MSG("E19999", "%s", msg.data());                                                                \
        GELOGE(ge::FAILED, "%s", msg.data());                                                                          \
        throw ge::AscIRException({ge::FAILED, msg.data()});                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  } while (false)
#define ASCIR_ASSERT_NOTNULL(v, ...) ASCIR_ASSERT(((v) != nullptr), __VA_ARGS__)
#endif // METADEF_CXX_INC_GRAPH_ASCENDC_IR_ASCENDC_IR_CHECK_H_
