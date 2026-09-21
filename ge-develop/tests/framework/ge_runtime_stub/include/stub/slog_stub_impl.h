/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_SLOG_STUB_IMPL_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_SLOG_STUB_IMPL_H_
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <regex>
#include <exception>
#include "depends/slog/src/slog_stub.h"
namespace gert {
namespace {
bool IsGroupMatch(const std::smatch &matches,
                  const std::vector<std::pair<int32_t, std::string>> &expect_groups) {
#if 0
  std::cout << "SN_DEBUG: matches" << std::endl;
  for (const auto &ma : matches) {
    std::cout << "SN_DEBUG: " << ma.str() << std::endl;
  }
#endif
  for (const auto &index_and_str : expect_groups) {
    if (index_and_str.first >= (int32_t)matches.size()) {
      throw std::range_error("Index out of range");
    }
    if (index_and_str.second != matches[index_and_str.first].str()) {
      return false;
    }
  }
  return true;
}
}
struct OneLog {
  int level;
  std::string content;
  bool EndsWith(const char *expect) const {
    return content.find_last_of(expect) == content.size() - 1;
  }
};
class SlogStubImpl : public ge::SlogStub {
 public:
  void Log(int module_id, int level, const char *fmt, va_list args) override {
    if (level < GetLevel()) {
      return;
    }
    const int FMT_BUFF_SIZE = 2048;
    char fmt_buff[FMT_BUFF_SIZE] = {0};
    if (Format(fmt_buff, sizeof(fmt_buff), module_id, level, fmt, args) > 0) {
      if (console_print_) {
        printf("%s \n", fmt_buff);
      }
      std::lock_guard<std::mutex> lock(logs_mutex_);
      logs_.push_back({level, fmt_buff});
    }
  }
  SlogStubImpl &NoConsoleOut() {
    console_print_ = false;
    return *this;
  }
  void Clear() {
    logs_.clear();
  }
  std::vector<OneLog> &GetLogs() {
    return logs_;
  }
  const std::vector<OneLog> &GetLogs() const {
    return logs_;
  }
  const std::vector<OneLog> GetLogs(int level) const {
    std::vector<OneLog> logs;
    for (const auto &log : logs_) {
      if (log.level == level) {
        logs.emplace_back(log);
      }
    }
    return logs;
  }
  int64_t FindLogEndswith(int level, const char *s) const {
    int64_t current_index = 0;
    for (const auto &log : logs_) {
      if (level < 0 || level == log.level) {
        if (log.EndsWith(s)) {
          return current_index;
        }
        ++current_index;
      }
    }
    return -1;
  }
  int64_t FindLog(int level, const char *s) const {
    int64_t current_index = 0;
    for (const auto &log : logs_) {
      if (level < 0 || level == log.level) {
        if (log.content.find(s) != std::string::npos) {
          return current_index;
        }
        ++current_index;
      }
    }
    return -1;
  }

  int64_t CountLog(int level, const char *s) const {
    int64_t log_count = 0;
    for (const auto &log : logs_) {
      if (level < 0 || level == log.level) {
        if (log.content.find(s) != std::string::npos) {
          log_count++;
        }
      }
    }
    return log_count;
  }

  /**
   * 使用正则表达式搜索指定level的日志，如果找到了，继续使用expect_groups中指定的group判断是否正确，
   * 如果expect_groups指定的内容也对应，那么返回本条log对应的index。如果全部日志遍历完也没有匹配的日志，那么返回-1
   * @param level 日志级别，例如DLOG_DEBUG
   * @param s 正则表达式
   * @param expect_groups first为第几个group,second为期望的值
   * @return
   */
  int64_t FindLogRegex(int level, const char *s,
                       const std::vector<std::pair<int32_t, std::string>> &expect_groups) const {
    std::regex re(s);
    int64_t current_index = 0;
    std::smatch matches;
    for (const auto &log : logs_) {
      if (level < 0 || level == log.level) {
        if (std::regex_search(log.content, matches, re)) {
          if (IsGroupMatch(matches, expect_groups)) {
            return current_index;
          }
        }
        ++current_index;
      }
    }
    return -1;
  }
  int64_t FindLogRegex(int level, const char *s) const {
    return FindLogRegex(level, s, {});
  }
  int64_t FindErrorLogEndsWith(const char *s) const {
    return FindLogEndswith(DLOG_ERROR, s);
  }
  int64_t FindWarnLogEndsWith(const char *s) const {
    return FindLogEndswith(DLOG_WARN, s);
  }
  int64_t FindInfoLogEndsWith(const char *s) const {
    return FindLogEndswith(DLOG_INFO, s);
  }
  int64_t FindInfoLogRegex(const char *s) const {
    return FindLogRegex(DLOG_INFO, s);
  }
  int64_t FindInfoLogRegex(const char *s, const std::vector<std::pair<int32_t, std::string>> &groups) const {
    return FindLogRegex(DLOG_INFO, s, groups);
  }

 private:
  std::mutex logs_mutex_;
  std::vector<OneLog> logs_;
  bool console_print_{true};
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_STUB_SLOG_STUB_IMPL_H_
