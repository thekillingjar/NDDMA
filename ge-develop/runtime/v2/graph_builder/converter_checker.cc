/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "converter_checker.h"
#include "register/node_converter_registry.h"
namespace gert {
namespace {
HyperStatus CheckAllHolders(const std::vector<bg::ValueHolderPtr> &holders) {
  for (const auto &holder : holders) {
    if ((holder == nullptr) || !holder->IsOk()) {
      return HyperStatus::ErrorStatus(static_cast<const char*>("Error in inputs"));
    }
  }
  return HyperStatus::Success();
}
HyperStatus CheckAllHolders(const std::vector<bg::DevMemValueHolderPtr> &holders) {
  for (const auto &holder : holders) {
    if ((holder == nullptr) || !holder->IsOk()) {
      return HyperStatus::ErrorStatus(static_cast<const char*>("Error in inputs"));
    }
  }
  return HyperStatus::Success();
}
} // namespace

LowerResult CreateErrorLowerResult(const char *error_msg, ...) {
  va_list arg;
  va_start(arg, error_msg);
  auto msg = std::unique_ptr<char[]>(CreateMessage(error_msg, arg));
  va_end(arg);
  return {HyperStatus::ErrorStatus(std::move(msg)), {}, {}, {}};
}

bool IsValidHolder(const bg::ValueHolderPtr &holder) {
  return (holder != nullptr) && holder->IsOk();
}

bool IsValidHolder(const std::vector<bg::ValueHolderPtr> &holders) {
  return (!holders.empty()) && std::all_of(holders.begin(), holders.end(),
                                           [](const bg::ValueHolderPtr &holder) { return IsValidHolder(holder); });
}
bool IsValidHolder(const bg::DevMemValueHolderPtr &holder) {
  return (holder != nullptr) && holder->IsOk();
}

bool IsValidHolder(const std::vector<bg::DevMemValueHolderPtr> &holders) {
  return (!holders.empty()) && std::all_of(holders.begin(), holders.end(),
                                           [](const bg::ValueHolderPtr &holder) { return IsValidHolder(holder); });
}

HyperStatus CheckLowerInput(const LowerInput &lower_input) {
  auto ret = CheckAllHolders(lower_input.input_shapes);
  if (!ret.IsSuccess()) {
    return ret;
  }
  ret = CheckAllHolders(lower_input.input_addrs);
  if (!ret.IsSuccess()) {
    return ret;
  }
  return HyperStatus::Success();
}

HyperStatus CheckFFTSLowerInput(const FFTSLowerInput &lower_input) {
  auto ret = CheckAllHolders(lower_input.input_shapes);
  if (!ret.IsSuccess()) {
    return ret;
  }
  ret = CheckAllHolders(lower_input.input_addrs);
  if (!ret.IsSuccess()) {
    return ret;
  }
  if (!IsValidHolder(lower_input.task_info) || !IsValidHolder(lower_input.thread_dim) ||
      !IsValidHolder(lower_input.window_size) || !IsValidHolder(lower_input.ffts_mem_allocator)) {
    return HyperStatus::ErrorStatus(static_cast<const char*>("Error ffts inputs."));
  }
  if (lower_input.ffts_thread_fun == nullptr) {
    return HyperStatus::ErrorStatus(static_cast<const char*>("Error ffts thread func is nullptr."));
  }
  return HyperStatus::Success();
}

HyperStatus CheckFFTSStaLowerInput(const FFTSLowerInput &lower_input) {
  auto ret = CheckAllHolders(lower_input.input_shapes);
  if (!ret.IsSuccess()) {
    return ret;
  }
  ret = CheckAllHolders(lower_input.input_addrs);
  if (!ret.IsSuccess()) {
    return ret;
  }
  if (!IsValidHolder(lower_input.task_info) || !IsValidHolder(lower_input.args_para)) {
    return HyperStatus::ErrorStatus(static_cast<const char*>("Error task or args."));
  }
  return HyperStatus::Success();
}
}  // namespace gert
