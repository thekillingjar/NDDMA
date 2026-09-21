/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_GENERATOR_PREPROCESS_UTILS_H_
#define ATT_GENERATOR_PREPROCESS_UTILS_H_
#include "base/base_types.h"
#include "generator/preprocess/var_info.h"
#include "common/checker.h"

namespace att {
inline bool IsInExprInfo(const ExprInfoMap &expr_infos, const Expr &expr) {
  if (!IsValid(expr)) {
    GELOGW("Input expr is null.");
    return false;
  }
  if (expr_infos.find(expr) == expr_infos.end()) {
    GELOGW("Expr infos has no var: [%s].", expr.Str().get());
    return false;
  }
  return true;
}
}  // namespace att
#endif  // ATT_GENERATOR_PREPROCESS_UTILS_H_