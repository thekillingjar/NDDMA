/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ATT_GENERAL_SOLVER_GEN_H_
#define ATT_GENERAL_SOLVER_GEN_H_
#include <map>
#include <set>
#include <vector>
#include "base/base_types.h"
#include "code_printer.h"
#include "util/base_types_printer.h"
#include "generator/solver_pass_gen/solver_gen.h"
namespace att
{
  enum class FuncType
  {
    OBJ = 0,
    LEQ = 1,
    BUFFER = 2,
  };

  class GeneralSolverGen : public SolverGen
  {
  public:
    explicit GeneralSolverGen(const std::string &tiling_case_id, const std::string &type_name, bool open_dt = false, bool training = false)
      : SolverGen(tiling_case_id, type_name), open_dt_(open_dt), training_(training) {}
    ~GeneralSolverGen() override = default;
    std::string GenSolverClassImpl() override;
    std::string GenSolverFuncImpl() override;
    std::string GenSolverFuncInvoke() override;
    
    std::string GenSolverDTImpl();
    std::string GenSolverDTInvoke();

    void SetSearchArgs(const std::vector<Expr> &search_args);
    void SetSolvedArgs(const std::vector<Expr> &solved_args);
    void SetConstArgs(const ExprUintMap &const_args);
    void SetInputAlign(const ExprExprMap &input_align);

    void SetObj(const std::map<PipeType, Expr> &obj);
    void SetHeadCost(const Expr &head_cost);
    void SetBufferCons(const std::map<HardwareDef, Expr> &buffer_cons);
    void SetCutCons(const std::vector<Expr> &cut_cons);
    void SetExprRelation(const ExprExprMap &expr_relation, const ExprExprMap &vars_relation);
    void SetMaxValue(const ExprExprMap &max_value);
    void SetMinValue(const ExprExprMap &min_value);
    void SetInitValue(const ExprExprMap &init_value);
    void SetInputArgs(const std::vector<Expr> &input_args);
    void SetInnestDim(const std::vector<Expr> &innest_dim);
    void SetReplaceVars(const std::vector<std::pair<Expr, Expr>> &replace_vars);
    void SetExeTimeMap(const std::map<Expr, std::vector<Expr>, ExprCmp> &exe_time_map);

    void FixVar(uint64_t var_idx, uint32_t value);
    void FixRange(uint64_t var_idx, uint16_t lower_bound, uint16_t upper_bound);

    void SetClassName(const std::string &class_name);
    
    std::string invoke_codes_ = "";
    ExprExprMap init_value_;

  private:
    bool CheckUseful(Expr arg);
    bool GetLeqInfo(int i, std::string &local_valid_func, std::string &update_func);
    
    std::string GetDoubleVars(const std::string &indent, const std::vector<Expr> &related_expr);
    std::string GetUIntVars(const std::string &indent);
    std::string GenAnnotation(FuncType func_type) const;
    std::string GenDiffAnnotation(FuncType func_type) const;
    std::string GetWeightedDiff(FuncType func_type);
    std::string InitiateMemoryPool() const;
    std::string GenMemoryPool();
    std::string InitiateValue();
    std::string InitiateDefInputs();
    std::string GenClassAnotataion();
    std::string GenArgDef(std::vector<Expr> args) const;
    std::string InitiateArgs(std::vector<Expr> args) const;
    std::string InitiateDefArgs(std::vector<Expr> args);
    
    std::string GenAlignInput(const Expr arg, const std::string indent);
    std::string GenDTInit();
    bool GenVarDef();
    bool GenFuncDef();

    bool GenConstVars();
    bool GenClassDef();
    bool GenBuffFunc();
    bool GenBuffExpr();
    bool GenInit();
    Expr GenFuncExpr(FuncType func_type);
    bool GenSubFunc(const std::map<std::string, Expr> &funcs);
    
    bool GenGetFunc(FuncType func_type);
    bool GenGetSmoothObj();
    bool GenGetWeightedDiff(FuncType func_type);
    bool GenLeqInfo();
    bool GenMapVarVal();
    bool GenDisplayVarVal();
    bool GenGetResult();

    bool CreateInput();
    bool CreateConfig();
    virtual bool RunSolver(bool is_dt = false);
    bool IsRelated(const Expr &expr);

    std::vector<Expr> search_args_;
    std::vector<Expr> input_args_;
    std::vector<Expr> solved_args_;
    std::vector<Expr> hardware_args_;
    std::vector<bool> innest_dim_;
    std::map<uint64_t, uint32_t> fixed_args_;
    ExprUintMap const_args_;
    ExprExprMap input_align_;
    ExprExprMap expr_relation_;
    ExprExprMap max_value_;
    ExprExprMap min_value_;

    Expr head_cost_;
    std::vector<std::pair<Expr, Expr>> replace_vars_;
    std::map<Expr, std::vector<Expr>, ExprCmp> exe_time_map_;

    std::map<std::string, Expr> obj_;
    std::map<std::string, Expr> buffer_cost_;
    std::map<std::string, Expr> leq_cost_;
    std::map<std::string, uint64_t> leq_map_;
    std::vector<Expr> leqs_;
    bool open_dt_{false};
    bool training_{false};

    std::string impl_codes_ = "";
  };
} // namespace att
#endif
