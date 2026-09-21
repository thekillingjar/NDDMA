/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_GE_PLUGIN_CALLER_H_
#define GE_COMMON_GE_PLUGIN_CALLER_H_

#include <mutex>
#include "ge/ge_api_types.h"
#include "framework/common/debug/log.h"
#include "common/ge_common/scope_guard.h"
#include "mmpa/mmpa_api.h"
#include "common/checker.h"

namespace ge {
class PluginCaller {
 public:
  explicit PluginCaller(const std::string &lib_name) : handle_(nullptr), lib_name_(lib_name) {}
  ~PluginCaller() = default;
  template <typename RetType, typename FuncType, typename... Args>
  auto CallFunction(const std::string &function_name, Args... args) -> RetType {
    const Status ret = LoadLib();
    GE_CHK_BOOL_RET_SPECIAL_STATUS(ret == NOT_CHANGED, SUCCESS, "Can not open lib, function_name = %s.",
                                   function_name.c_str());
    GE_ASSERT_SUCCESS(ret);
    GE_MAKE_GUARD(unload_libs, [this]() { UnloadLib(); });
    const auto func = reinterpret_cast<FuncType>(mmDlsym(handle_, function_name.c_str()));
    GE_ASSERT_NOTNULL(func, "Failed to find symbol %s.", function_name.c_str());
    return func(args...);
  }

 private:
  ge::Status LoadLib();
  void UnloadLib();
  void ReportInnerError() const;
  void *handle_;
  std::string lib_name_;
  std::mutex mutex_;
};
}  // namespace ge

#endif  // GE_COMMON_GE_PLUGIN_CALLER_H_
