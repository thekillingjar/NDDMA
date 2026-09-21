/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attribute_group/attr_group_shape_env.h"
#include "graph/symbolizer/symbolic.h"
#include "common/checker.h"

namespace ge {
namespace sym {
bool ExpectSymbolEq(const Expression &e0, const Expression &e1,
    const char_t *file, const int64_t line) {
  GE_ASSERT_NOTNULL(file);
  bool res = ExpectSymbolBool(sym::Eq(e0, e1), file, line);
  if (res) {
    GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendReplacement(e0, e1),
        "[%s:%lld] Append symbol equivalence %s to %s failed",
        file, line, e0.Serialize().get(), e1.Serialize().get());
  }
  return res;
}

bool ExpectSymbolNe(const Expression &e0, const Expression &e1,
    const char_t *file, const int64_t line) {
  GE_ASSERT_NOTNULL(file);
  bool res = ExpectSymbolBool(sym::Ne(e0, e1), file, line);
  if (!res) {
    GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendReplacement(e0, e1),
        "[%s:%lld] Append symbol equivalence %s to %s failed",
        file, line, e0.Serialize().get(), e1.Serialize().get());
  }
  return res;
}

bool ExpectSymbolBool(const Expression &expr, const char_t *file, const int64_t line) {
  GE_ASSERT_NOTNULL(file);
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "Only boolean expr can be use to check symbol, expr: %s",
      expr.Serialize().get());
  if (expr.IsConstExpr()) {
    bool const_value = false;
    GE_ASSERT_TRUE(expr.GetConstValue(const_value));
    return const_value;
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot check symbol, expr: %s", expr.Serialize().get());
    return false;
  }
  bool hint_value = false;
  GE_ASSERT_TRUE(expr.GetHint(hint_value));
  if (hint_value) {
    GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendSymbolCheckInfo(expr.Simplify(), file, line));
  } else {
    GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendSymbolCheckInfo(sym::Not(expr.Simplify()), file, line));
  }
  return hint_value;
}
  
bool AssertSymbolEq(const Expression &e0, const Expression &e1,
    const char_t *file, const int64_t line) {
  GE_ASSERT_NOTNULL(file);
  GE_ASSERT_TRUE(AssertSymbolBool(ge::sym::Eq(e0, e1), file, line));
  GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendReplacement(e0, e1),
      "[%s:%lld] Append symbol equivalence %s to %s failed",
      file, line, e0.Serialize().get(), e1.Serialize().get());
  return true;
}
  
bool AssertSymbolBool(const Expression &expr, const char_t *file, const int64_t line) {
  GE_ASSERT_NOTNULL(file);
  GE_ASSERT_TRUE(expr.IsBooleanExpr(), "[%s:%lld] Only boolean expr can be used to assert, expr: %s",
      file, line, expr.Serialize().get());
  if (expr.IsConstExpr()) {
    bool const_value = false;
    GE_ASSERT_TRUE(expr.GetConstValue(const_value));
    GE_ASSERT_TRUE(const_value, "[%s:%lld] Assert %s failed",
        file, line, expr.Serialize().get());
    return const_value;
  }
  if (GetCurShapeEnvContext() == nullptr) {
    GELOGW("Shape env is nullptr, cannot check symbol, expr: %s", expr.Serialize().get());
    return false;
  }
  bool hint_value = false;
  GE_ASSERT_TRUE(expr.GetHint(hint_value));
  GE_ASSERT_TRUE(hint_value, "[%s:%lld] Assert %s failed", file, line, expr.Serialize().get());
  GE_ASSERT_SUCCESS(GetCurShapeEnvContext()->AppendSymbolAssertInfo(expr.Simplify(), file, line));
  return true;
}
}
}