/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/platform_context.h"

namespace ge {
PlatformContext &PlatformContext::GetInstance() {
  static PlatformContext instance;
  return instance;
}
std::mutex PlatformContext::mutex_;

void PlatformContext::SetPlatform(const std::string &platform_name) {
  std::lock_guard<std::mutex> lg(mutex_);
  current_platform_ = platform_name;
  initialized_ = true;
}

ge::Status PlatformContext::GetCurrentPlatformString(std::string &platform_name) {
  if (!initialized_) {
    std::lock_guard<std::mutex> lg(mutex_);
    // Stub默认平台，用于测试
    current_platform_ = "2201";
    initialized_ = true;
  }
  std::lock_guard<std::mutex> lg(mutex_);
  platform_name = current_platform_;
  return ge::SUCCESS;
}
}  // namespace ge
