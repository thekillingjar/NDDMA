/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_L2_SOLVER_GEN_H_
#define ATT_L2_SOLVER_GEN_H_
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <sstream>
#include "generator/solver_pass_gen/solver_gen.h"
#include "base/base_types.h"
#include "util/base_types_printer.h"
#include "common/checker.h"

namespace att
{
  class L2TileSolverGen : public SolverGen
  {
  public:
    explicit L2TileSolverGen(const std::string &tiling_case_id, const std::string &type_name)
      : SolverGen(tiling_case_id, type_name) {}
    ~L2TileSolverGen() override = default;
    std::string GenSolverClassImpl() override;
    std::string GenSolverFuncImpl() override;
    std::string GenSolverFuncInvoke() override;
    void SetInputArgs(const std::vector<Expr> &input_args) { input_args_ = input_args; }
    void SetConstVars(const ExprUintMap &const_vars) { const_vars_ = const_vars; }
    void SetL0Args(const std::vector<Expr> &l0_args) { l0_args_ = l0_args; }
    void SetL2Args(const std::vector<Expr> &l2_args) { l2_args_ = l2_args; }
    void SetL2Use(const Expr &expr) { l2_use_ = expr; }
    void SetArgtMaxValueMap(const ExprExprMap &arg_max_value_map)
    {
      arg_max_value_map_ = arg_max_value_map;
    }
    void SetArgAlignMap(const ExprExprMap &arg_align_map)
    {
      arg_align_map_ = arg_align_map;
    }

  private:
    std::string GenClassDef();
    std::string GenSolverInvokeDoc() const;
    std::string GetL2RelInputArg();
    Expr GetRelateL0Arg(const Expr &l2_arg);
    bool IsRepeatArgs(const Expr &arg) const;
    bool IsClashPossible() const;
    std::vector<Expr> l0_args_;
    ExprUintMap const_vars_;
    std::vector<Expr> l2_args_;
    std::vector<Expr> input_args_;
    Expr l2_use_;
    ExprExprMap arg_max_value_map_;
    ExprExprMap arg_align_map_;
  };
} // namespace att
#endif