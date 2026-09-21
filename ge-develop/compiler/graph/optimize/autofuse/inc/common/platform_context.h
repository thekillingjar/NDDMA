/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCGEN_DEV_BASE_COMMON_PLATFORM_CONTEXT_H
#define ASCGEN_DEV_BASE_COMMON_PLATFORM_CONTEXT_H

#include <string>
#include <mutex>
#include "ge_common/ge_api_error_codes.h"

namespace ge {
class PlatformContext {
 public:
  static PlatformContext& GetInstance();

  PlatformContext(const PlatformContext &) = delete;
  PlatformContext &operator=(const PlatformContext &) = delete;
  PlatformContext(PlatformContext &&) = delete;
  PlatformContext &operator=(PlatformContext &&) = delete;

  // 外部设置 platform
  void SetPlatform(const std::string &platform_name);

  // 获取当前 platform 字符串
  ge::Status GetCurrentPlatformString(std::string &platform_name);

  void Reset() {
    initialized_ = false;
    current_platform_ = "";
  }

 private:
  ge::Status Initialize();
  PlatformContext() = default;
  std::string current_platform_;
  bool initialized_ = false;
  static std::mutex mutex_;
};
}  // namespace optimize
#endif  // ASCGEN_DEV_BASE_COMMON_PLATFORM_CONTEXT_H
