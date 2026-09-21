/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_OPTIMIZE_COMMON_PARAMS_H_
#define GE_GRAPH_OPTIMIZE_COMMON_PARAMS_H_

#include <string>

#include "common/singleton.h"
#include "framework/common/types.h"

namespace ge {
class Params : public Singleton<Params> {
 public:
  DECLARE_SINGLETON_CLASS(Params);

  void SetTarget(const char* target) {
    std::string tmp_target = (target != nullptr) ? target : "";

#if defined(__ANDROID__) || defined(ANDROID)
    target_ = "LITE";
    target_8bit_ = TARGET_TYPE_LTTE_8BIT;
#else
    target_ = "MINI";
    target_8bit_ = TARGET_TYPE_MINI_8BIT;
#endif
    if (tmp_target == "mini") {
      target_ = "MINI";
      target_8bit_ = TARGET_TYPE_MINI_8BIT;
    } else if (tmp_target == "lite") {
      target_ = "LITE";
      target_8bit_ = TARGET_TYPE_LTTE_8BIT;
    }
  }

  std::string GetTarget() const { return target_; }

  uint8_t GetTarget_8bit() const { return target_8bit_; }
  ~Params() override = default;

 private:
  Params() : target_("MINI") {}

  std::string target_;
  uint8_t target_8bit_ = TARGET_TYPE_MINI_8BIT;
};
}  // namespace ge

#endif  // GE_GRAPH_OPTIMIZE_COMMON_PARAMS_H_
