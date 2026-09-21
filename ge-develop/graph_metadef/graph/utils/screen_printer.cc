/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/screen_printer.h"

#include <iostream>
#include "framework/common/debug/ge_log.h"
#include "graph/ge_context.h"
#include "mmpa/mmpa_api.h"

namespace ge {
namespace {
constexpr size_t kMaxLogLen = 1024U;
constexpr size_t kMaxTimeLen = 128U;
constexpr int64_t kOneThousandMs = 1000L;
constexpr const char_t *kModeDisable = "disable";

std::string CurrentTimeFormatStr() {
  mmSystemTime_t system_time;
  if (mmGetSystemTime(&system_time) != EN_OK) {
    return "";
  }
  mmTimeval tv;
  if (mmGetTimeOfDay(&tv, nullptr) != EN_OK) {
    return "";
  }
  char_t format_time[kMaxTimeLen] = {};
  if (snprintf_s(format_time, kMaxTimeLen, kMaxTimeLen - 1U, "[%04d-%02d-%02d-%02d:%02d:%02d.%03ld.%03ld]",
                 system_time.wYear, system_time.wMonth, system_time.wDay, system_time.wHour, system_time.wMinute,
                 system_time.wSecond, (tv.tv_usec / kOneThousandMs), (tv.tv_usec % kOneThousandMs)) == -1) {
    return "";
  }
  return format_time;
}
}

ScreenPrinter &ScreenPrinter::GetInstance() {
  static ScreenPrinter instance;
  return instance;
}

void ScreenPrinter::Log(const char *fmt, ...) {
  if (fmt == nullptr) {
    GELOGE(FAILED, "param is nullptr and will not print message.");
    return;
  }
  if (print_mode_ == PrintMode::DISABLE) {
    return;
  }
  va_list va_list;
  va_start(va_list, fmt);
  char_t str[kMaxLogLen + 1U] = {};
  if (vsnprintf_s(str, kMaxLogLen + 1U, kMaxLogLen, fmt, va_list) == -1) {
    va_end(va_list);
    GELOGE(FAILED, "sprintf log failed and will not print message.");
    return;
  }
  va_end(va_list);

  const auto &format_time = CurrentTimeFormatStr();
  if (format_time.empty()) {
    GELOGE(FAILED, "construct format time failed and will not print message.");
    return;
  }

  const std::lock_guard<std::mutex> lk(mutex_);
  std::cout << format_time << mmGetTid() << " " << str << std::endl;
  return;
}

void ScreenPrinter::Init(const std::string &print_mode) {
  if ((!print_mode.empty()) && (print_mode == kModeDisable)) {
    print_mode_ = PrintMode::DISABLE;
  } else {
    print_mode_ = PrintMode::ENABLE;
  }
  GELOGD("Screen print mode:%u", print_mode_);
}
}  // namespace ge
