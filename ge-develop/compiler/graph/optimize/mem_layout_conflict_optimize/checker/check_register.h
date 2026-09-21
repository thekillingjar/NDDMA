/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <functional>
#include <vector>
#include "checker.h"

namespace ge {
class CheckerRegister {
 public:
  CheckerRegister() = default;
  ~CheckerRegister() = default;
  CheckerRegister(const AnchorAttribute &type_a, const AnchorAttribute &type_b,
                  std::function<Status(CheckFuncContext &)> func);
  CheckerRegister(CheckerRegister &&other) = default;
  CheckerRegister(const CheckerRegister &other) = default;

  CheckerRegister &operator=(const CheckerRegister &other) = default;
  CheckerRegister &operator=(CheckerRegister &&other) = default;

  CheckerRegister &CallAsSymbol();
 private:
  std::shared_ptr<CheckerFunc> checker_func_;
};

std::vector<vector<CheckerFunc *>> &GetRegister();

#define REGISTER_FUNC(type_a, type_b, func_name) \
  static auto g_var_##type_a##type_b##func_name = CheckerRegister(type_a, type_b, func_name)
}  // namespace ge
