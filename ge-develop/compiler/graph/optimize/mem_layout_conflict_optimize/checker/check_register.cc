/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "checker.h"
#include "check_register.h"

namespace ge {
std::vector<vector<CheckerFunc *>> &GetRegister() {
  static std::vector<vector<CheckerFunc *>> ins(
      ATTR_BIT_MAX_LEN, vector<CheckerFunc *>(ATTR_BIT_MAX_LEN));
  return ins;
}

CheckerRegister::CheckerRegister(const AnchorAttribute &type_a, const AnchorAttribute &type_b,
                                 std::function<Status(CheckFuncContext &)> func)
    : checker_func_(new (std::nothrow) CheckerFunc(type_a, type_b, func)) {
  if (checker_func_ != nullptr) {
    GetRegister()[static_cast<size_t>(checker_func_->type_a_)][static_cast<size_t>(checker_func_->type_b_)]
        = checker_func_.get();
    GetRegister()[static_cast<size_t>(checker_func_->type_b_)][static_cast<size_t>(checker_func_->type_a_)]
        = checker_func_.get();
  }
}

// 表示每个symbol只调用一次这个函数, 默认情况是symbol下的所有anchor两两组合，每个组合都调用一次checker函数
CheckerRegister &CheckerRegister::CallAsSymbol() {
  checker_func_->SetCallAsSymbol();
  return *this;
}
}  // namespace ge
