/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "platform_factory.h"
#include "common/platform_context.h"

namespace optimize {
PlatformFactory &PlatformFactory::GetInstance() {
  static PlatformFactory instance;
  return instance;
}
void PlatformFactory::RegisterPlatform(const std::string &platform_name, PlatformCreator creator) {
  if (platform_name_to_creators_.find(platform_name) == platform_name_to_creators_.end()) {
    platform_name_to_creators_[platform_name] = creator;
  }
}
BasePlatform *PlatformFactory::GetPlatform() {
  std::string platform_name;
  GE_ASSERT_SUCCESS(ge::PlatformContext::GetInstance().GetCurrentPlatformString(platform_name),
                   "Failed to get platform info.");
  GELOGD("Current platform info is %s", platform_name.c_str());
  auto it = platform_name_to_instances_.find(platform_name);
  if (it != platform_name_to_instances_.end()) {
    return it->second.get();
  }

  auto creator_it = platform_name_to_creators_.find(platform_name);
  if (creator_it != platform_name_to_creators_.end()) {
    platform_name_to_instances_[platform_name] = creator_it->second();
    return platform_name_to_instances_[platform_name].get();
  }

  GELOGE(ge::FAILED, "Can't find platform %s", platform_name.c_str());
  return nullptr;
}

}  // namespace