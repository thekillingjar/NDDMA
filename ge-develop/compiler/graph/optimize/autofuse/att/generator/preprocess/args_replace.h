/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_CODE_GEN_PREPROCESS_ARGS_REPLACE_H_
#define ATT_CODE_GEN_PREPROCESS_ARGS_REPLACE_H_
#include <map>
#include <vector>
#include <queue>

#include "base/model_info.h"
#include "generator/preprocess/var_info.h"

namespace att {
class ArgsReplacer {
 public:
  explicit ArgsReplacer(const ExprInfoMap &var_infos) : vars_infos_(var_infos) {}
  ~ArgsReplacer() = default;
  void GetReplaceResult(ExprExprMap &replaced_vars, ExprExprMap &replacements,
                        ExprExprMap &new_expr_replacements);
  bool DoReplace(const std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs);
  void GetNewExprInitValue(ExprExprMap &new_expr_init_values);
 private:
  // 按对齐值来做变量替换，如果对齐值为1，那么不需要做变量替换，否则替换为x_div_align*align
  Expr ReplaceCommonExpr(const Expr &e, const Expr &align, ExprExprMap &new_expr_ori_expr_map,
                         ExprExprMap &new_expr_replacements);

  bool IsAllFactorReplaced(const ExprExprMap &replaced_vars, std::vector<Expr> &factors) const;
  bool InitWithEqCons(const std::map<std::string, std::vector<std::pair<Expr, Expr>>> &eq_exprs);
  bool GetLeafExprs();
  void GetAlignReplacedVars(const Expr &expr);
  void GetFactorReplacedVars(const Expr &expr);
  void GetSelfReplacedVars(const Expr &expr);

  bool ReplaceParentExprs();
  void ReplaceNaiveExpr();
  void Reset();

 private:
  // 原始变量到新变量的表达式：例如 m = 16 * m_div_align
  ExprExprMap ori_expr_new_expr_map_;
  // 新变量到原始变量：例如 m_div_align -- m
  ExprExprMap new_expr_ori_expr_map_;
  // 切分轴之间的关联关系；例如 basem -- stepm
  ExprExprMap factor_expr_map_;
  // 新变量到原始变量的表达式 例如 m_div_align = m / 16
  ExprExprMap new_expr_replacements_;
  ExprExprMap new_expr_init_values_;
  std::queue<Expr> replaced_expr_queue_;
  std::map<Expr, std::vector<Expr>, ExprCmp> expr_factors_map_;
  const ExprInfoMap &vars_infos_;
};
}  // namespace att

#endif  // ATT_CODE_GEN_PREPROCESS_ARGS_REPLACE_H_