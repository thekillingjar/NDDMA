/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <sstream>
#include <securec.h>
#include "mmpa/mmpa_api.h"
#include "dlog_pub.h"
#include "graph/def_types.h"
#include "base/err_msg.h"

#define GE_MODULE_NAME static_cast<int32_t>(GE)
namespace {
const std::string kParamCheckErrorSuffix = "8888";
class GeLog {
 public:
  static uint64_t GetTid() {
#ifdef __GNUC__
    thread_local static const uint64_t tid = static_cast<uint64_t>(syscall(__NR_gettid));
#else
    thread_local static const uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
    return tid;
  }
};

inline bool IsLogEnable(const int32_t module_name, const int32_t log_level) {
  const int32_t enable = CheckLogLevel(module_name, log_level);
  // 1:enable, 0:disable
  return (enable == 1);
}

std::string CurrentTimeFormatStr() {
  std::string time_str;
  auto now = std::chrono::system_clock::now();
  auto milli_seconds = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto micro_seconds = std::chrono::time_point_cast<std::chrono::microseconds>(now);
  const auto now_t = std::chrono::system_clock::to_time_t(now);
  const std::tm *tm_now = std::localtime(&now_t);
  if (tm_now == nullptr) {
    return time_str;
  }

  constexpr int32_t year_base = 1900;
  constexpr size_t kMaxTimeLen = 128U;
  constexpr int64_t kOneThousandMs = 1000L;
  error_message::char_t format_time[kMaxTimeLen] = {};
  (void) snprintf_s(format_time, kMaxTimeLen, kMaxTimeLen - 1U, "%04d-%02d-%02d-%02d:%02d:%02d.%03ld.%03ld",
                    tm_now->tm_year + year_base, tm_now->tm_mon + 1, tm_now->tm_mday, tm_now->tm_hour, tm_now->tm_min,
                    tm_now->tm_sec, milli_seconds.time_since_epoch().count() % kOneThousandMs,
                    micro_seconds.time_since_epoch().count() % kOneThousandMs);
  time_str = format_time;
  return time_str;
}
}

#define GELOGE(fmt, ...) \
  do {                                                                                            \
    dlog_error(GE_MODULE_NAME, "%" PRIu64 " %s" fmt, GeLog::GetTid(), &__FUNCTION__[0],       \
               ##__VA_ARGS__);                \
  } while (false)

#define GELOGW(fmt, ...)                                                                          \
  do {                                                                                            \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_WARN)) {                                                 \
      dlog_warn(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0],         \
                ##__VA_ARGS__);                                                                   \
    }                                                                                             \
  } while (false)

#define GELOGI(fmt, ...)                                                                          \
  do {                                                                                            \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {                                                 \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0],         \
                ##__VA_ARGS__);                                                                   \
    }                                                                                             \
  } while (false)

#define GELOGD(fmt, ...)                                                                           \
  do {                                                                                             \
    if (IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {                                                 \
      dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), &__FUNCTION__[0],         \
                 ##__VA_ARGS__);                                                                   \
    }                                                                                              \
  } while (false)

namespace {
int32_t ReportInnerErrorMessage(const char *file_name, const char *func, uint32_t line, const char *error_code,
                                const char *format, va_list arg_list) {
  std::vector<char> buf(LIMIT_PER_MESSAGE, '\0');
  auto ret = vsprintf_s(buf.data(), LIMIT_PER_MESSAGE, format, arg_list);
  if (ret < 0) {
    GELOGE("[Check][Param] FormatErrorMessage failed, ret:%d, file:%s, line:%u", ret, file_name, line);
    return -1;
  }
  ret = sprintf_s(buf.data() + ret, LIMIT_PER_MESSAGE - static_cast<size_t>(ret), "[FUNC:%s][FILE:%s][LINE:%u]",
                  func, error_message::TrimPath(std::string(file_name)).c_str(), line);
  if (ret < 0) {
    GELOGE("[Check][Param] FormatErrorMessage failed, ret:%d, file:%s, line:%u", ret, file_name, line);
    return -1;
  }

  return REPORT_USER_DEFINED_ERR_MSG(error_code, buf.data());
}

std::unique_ptr<error_message::char_t[]> CreateUniquePtrFromString(const std::string &str) {
  const size_t buf_size = str.empty() ? 1 : (str.size() + 1);
  auto uni_ptr = std::unique_ptr<error_message::char_t[]>(new (std::nothrow) error_message::char_t[buf_size]);
  if (uni_ptr == nullptr) {
    return nullptr;
  }

  if (str.empty()) {
    uni_ptr[0U] = '\0';
  } else {
    // 当src size < dst size时，strncpy_s会在末尾str.size()位置添加'\0'
    if (strncpy_s(uni_ptr.get(), str.size() + 1, str.c_str(), str.size()) != EOK) {
      return nullptr;
    }
  }
  return uni_ptr;
}
}  // namespace


namespace error_message {
// first stage
const std::string kInitialize   = "INIT";
const std::string kModelCompile = "COMP";
const std::string kModelLoad    = "LOAD";
const std::string kModelExecute = "EXEC";
const std::string kFinalize     = "FINAL";

// SecondStage
// INITIALIZE
const std::string kParser               = "PARSER";
const std::string kOpsProtoInit         = "OPS_PRO";
const std::string kSystemInit           = "SYS";
const std::string kEngineInit           = "ENGINE";
const std::string kOpsKernelInit        = "OPS_KER";
const std::string kOpsKernelBuilderInit = "OPS_KER_BLD";
// MODEL_COMPILE
const std::string kPrepareOptimize    = "PRE_OPT";
const std::string kOriginOptimize     = "ORI_OPT";
const std::string kSubGraphOptimize   = "SUB_OPT";
const std::string kMergeGraphOptimize = "MERGE_OPT";
const std::string kPreBuild           = "PRE_BLD";
const std::string kStreamAlloc        = "STM_ALLOC";
const std::string kMemoryAlloc        = "MEM_ALLOC";
const std::string kTaskGenerate       = "TASK_GEN";
// COMMON
const std::string kOther = "DEFAULT";

#ifdef __GNUC__
std::string TrimPath(const std::string &str) {
  if (str.find_last_of('/') != std::string::npos) {
    return str.substr(str.find_last_of('/') + 1U);
  }
  return str;
}
#else
std::string TrimPath(const std::string &str) {
  if (str.find_last_of('\\') != std::string::npos) {
    return str.substr(str.find_last_of('\\') + 1U);
  }
  return str;
}
#endif

int32_t FormatErrorMessage(char_t *str_dst, size_t dst_max, const char_t *format, ...) {
  int32_t ret;
  va_list arg_list;

  va_start(arg_list, format);
  ret = vsprintf_s(str_dst, dst_max, format, arg_list);
  (void)arg_list;
  va_end(arg_list);
  if (ret < 0) {
    GELOGE("[Check][Param] FormatErrorMessage failed, ret:%d, pattern:%s", ret, format);
  }
  return ret;
}

void ReportInnerError(const char_t *file_name, const char_t *func, uint32_t line, const std::string error_code,
                      const char_t *format, ...) {
  va_list arg_list;
  va_start(arg_list, format);
  (void)ReportInnerErrorMessage(file_name, func, line, error_code.c_str(), format, arg_list);
  va_end(arg_list);
  return;
}
}

namespace {
#ifdef __GNUC__
constexpr const error_message::char_t *const kErrorCodePath = "../conf/error_manager/error_code.json";
constexpr const error_message::char_t *const kSeparator = "/";
#else
const error_message::char_t *const kErrorCodePath = "..\\conf\\error_manager\\error_code.json";
const error_message::char_t *const kSeparator = "\\";
#endif

constexpr uint64_t kLength = 2UL;

void Ltrim(std::string &s) {
  (void) s.erase(s.begin(),
                 std::find_if(s.begin(),
                              s.end(),
                              [](const error_message::char_t c) -> bool {
                                return static_cast<bool>(std::isspace(static_cast<uint8_t>(c)) == 0);
                              }));
}

void Rtrim(std::string &s) {
  (void) s.erase(std::find_if(s.rbegin(),
                              s.rend(),
                              [](const error_message::char_t c) -> bool {
                                return static_cast<bool>(std::isspace(static_cast<uint8_t>(c)) == 0);
                              }).base(),
                 s.end());
}

/// @ingroup domi_common
/// @brief trim space
void Trim(std::string &s) {
  Rtrim(s);
  Ltrim(s);
}

// split string
std::vector<std::string> SplitByDelim(const std::string &str, const error_message::char_t delim) {
  std::vector<std::string> elems;

  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream ss(str);
  std::string item;

  while (getline(ss, item, delim)) {
    Trim(item);
    elems.push_back(item);
  }
  const auto str_size = str.size();
  if ((str_size > 0U) && (str[str_size - 1U] == delim)) {
    elems.emplace_back("");
  }

  return elems;
}
}  // namespace

struct StubErrorItem {
  std::string error_id;
  std::string error_title;
  std::string error_message;
  std::string possible_cause;
  std::string solution;
  std::map<std::string, std::string> args_map;
  std::string report_time;

  friend bool operator==(const StubErrorItem &lhs, const StubErrorItem &rhs) noexcept {
    return (lhs.error_id == rhs.error_id) && (lhs.error_message == rhs.error_message) &&
        (lhs.possible_cause == rhs.possible_cause) && (lhs.solution == rhs.solution);
  }
};

std::mutex stub_mutex_;
static std::vector<StubErrorItem> stub_error_message_process_;

namespace error_message {
int32_t RegisterFormatErrorMessage(const char_t *error_msg, size_t error_msg_len) {
  (void)error_msg_len;
  (void)error_msg;
  return 0;
}

int32_t ReportInnerErrMsg(const char *file_name, const char *func, uint32_t line, const char *error_code,
                          const char *format, ...) {
  (void)file_name;
  (void)func;
  (void)line;
  std::lock_guard<std::mutex> lock(stub_mutex_);
  StubErrorItem item = {error_code, "", format};
  stub_error_message_process_.emplace_back(item);
  return 0;
}
}  // namespace error_message
