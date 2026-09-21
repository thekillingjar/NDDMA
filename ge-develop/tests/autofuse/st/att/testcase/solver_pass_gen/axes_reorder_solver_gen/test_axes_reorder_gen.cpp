/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "base/base_types.h"
#define private public
#include "generator/solver_pass_gen/axes_reorder_solver/axes_reorder_solver_gen.h"
#include "test_common_utils.h"
using namespace att;

class TestAxesReorderSolverGen : public ::testing::Test {
 public:
  void TearDown() override {
    // 清理测试生成的临时文件
    autofuse::test::CleanupTestArtifacts();
  }
};

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::map<HardwareDef, Expr> hardware_cons;
  std::vector<Expr> search_args;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;
  search_args = {x0, x1};

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.Arrange();
  
  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x0");
}

TEST_F(TestAxesReorderSolverGen, TEST_ARRANGE_SIZE_VAR_AS_INPUT)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  Expr x2 = CreateExpr("x2");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::map<HardwareDef, Expr> hardware_cons;
  std::vector<Expr> search_args;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::UB] = x0 + x1 + x2;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 1;
  axis_priority[x1] = 2;
  search_args = {x0, x1};

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.Arrange();
  
  auto vars = solver_gen.input_args_;
  EXPECT_NE(vars.size(), 0);
  EXPECT_EQ(Str(vars[0]), "x2");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("x1");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0,x1};
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;
  EXPECT_EQ(Str(vars[0]), "x1");

  std::string impl_code = solver_gen.GenSolverFuncImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, TEST_GEN_SOLVER_case2)
{
  Expr x0 = CreateExpr("x0");
  Expr x1 = CreateExpr("block_dim");
  std::vector<Expr> cut_cons;
  std::vector<Expr> input_args;
  std::vector<Expr> search_args{x0,x1};
  input_args.emplace_back(x1);
  std::map<HardwareDef, Expr> hardware_cons;
  std::map<Expr, Expr, ExprCmp> arg_align_map;
  std::map<Expr, uint32_t, ExprCmp> const_args;
  std::map<Expr, uint32_t, ExprCmp> axis_priority;
  std::map<Expr, std::vector<Expr>, ExprCmp> from_axes_map;
  arg_align_map[x0] = ge::Symbol(16);
  cut_cons.emplace_back(x0 - x1);
  hardware_cons[HardwareDef::L2] = x0 + x1;
  from_axes_map[x0] = {x1};
  axis_priority[x0] = 2;
  axis_priority[x1] = 1;

  AxesReorderSolverGen solver_gen("case_test", "TilingData");
  solver_gen.SetArgAlignMap(arg_align_map);
  solver_gen.SetTotalCutCons(cut_cons);
  solver_gen.SetBufferUseAlg(hardware_cons);
  solver_gen.SetConstArgs(const_args);
  solver_gen.SetInputArgs(input_args);
  solver_gen.SetSearchArgs(search_args);
  solver_gen.SetFromAxesMap(from_axes_map);
  solver_gen.SetVarPriority(axis_priority);
  solver_gen.Arrange();

  auto vars = solver_gen.local_buffer_tiling_vars_;

  std::string impl_code = solver_gen.GenSolverClassImpl();
  std::string invoke_code = solver_gen.GenSolverFuncInvoke();
  EXPECT_NE(impl_code, "");
  EXPECT_NE(invoke_code, "");
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundFunc_ConstExpr) {
    Expr var = CreateExpr("var");
    AxesReorderSolverGen solver_gen("case_test", "TilingData");
    solver_gen.from_axes_map_[var] = {CreateExpr(2), CreateExpr(3)};
    
    std::string result = solver_gen.GenUpperBoundFunc(var);
    
    std::string expected = R"(    GetUpperBoundFuncPtr var_upper_bound = [](Variable **parent_vars) {
      int64_t upper_bound = 1;
      upper_bound *= 2;
      upper_bound *= 3;
      return upper_bound;
    };
    var.upper_bound = var_upper_bound;
)";
    EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, GenUpperBoundInfo_ConstExpr) {
    Expr var = CreateExpr("var");
    AxesReorderSolverGen solver("case_test", "TilingData");
    solver.from_axes_map_[var] = {CreateExpr(5)}; // Constant expression
    EXPECT_EQ(solver.GenUpperBoundInfo(var), "");
}

TEST_F(TestAxesReorderSolverGen, InitiateArgs_MixedValidInvalidArgs) {
    // Setup test data with mixed valid and invalid args
    AxesReorderSolverGen solver_("case_test", "TilingData");
    Expr valid = CreateExpr("valid");
    Expr invalid = CreateExpr("invalid");
    solver_.input_args_ = {valid, invalid};
    solver_.input_align_ = {{valid, ge::Symbol(4)}, {invalid, ge::Symbol(1)}}; // One needs alignment
    solver_.mc_args_ = {};
    solver_.local_buffer_tiling_vars_ = {};

    // Expected output
    std::string expected = 
        "    Variable valid;\n"
        "    valid.value = (tiling_data.get_valid() + std::max(1, 4) - 1) / std::max(1, 4) * std::max(1, 4);\n"
        "    Variable invalid;\n"
        "    invalid.value = tiling_data.get_invalid();\n";

    std::string result = solver_.InitiateArgs();
    EXPECT_EQ(result, expected);
}

TEST_F(TestAxesReorderSolverGen, NeitherInPriorityMap) {
    // Neither expression in priority map
    // Should use string comparison
    Expr expr1 = CreateExpr("var1");
    Expr expr2 = CreateExpr("var2");
    Expr expr3 = CreateExpr("var3");
    Expr expr4 = CreateExpr("var4");
    AxesReorderSolverGen solver("case_test", "TilingData");
    solver.VarCmp(expr1, expr2);
}


TEST_F(TestAxesReorderSolverGen, GenInputInfoTest) {
    std::vector<Expr> all_cons;
    std::vector<Expr> local_buffer_cons;
    std::vector<Expr> mc_mixed_cons;

    AxesReorderSolverGen solver("case_test", "TilingData");
    std::map<HardwareDef, Expr> hardware_use_map_;
    std::vector<Expr> total_cut_cons_;
    std::vector<Expr> mc_args_;
    std::vector<Expr> local_buffer_tiling_vars_;
    ExprExprMap arg_align_map_;
    ExprUintMap arg_prompt_align_map_;
    ExprUintMap is_concat_outer_map_;
    std::vector<Expr> concat_inner_dims_;

    // Initialize test data
    hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
    hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
    hardware_use_map_[HardwareDef::UB] = CreateExpr(512);
    
    total_cut_cons_.push_back(CreateExpr(256));
    total_cut_cons_.push_back(CreateExpr(128));

    
    mc_args_.push_back(CreateExpr("mc_var1"));
    mc_args_.push_back(CreateExpr("mc_var2"));

    Expr var1 = CreateExpr("local_var1");
    Expr var2 = CreateExpr("local_var2");
    Expr var3 = ge::Symbol(255, "local_var3");

    
    local_buffer_tiling_vars_.push_back(var1);
    local_buffer_tiling_vars_.push_back(var2);
    
    arg_align_map_[var1] = ge::Symbol(64);
    arg_align_map_[var2] = ge::Symbol(32);
    
    arg_prompt_align_map_[var1] = 128;
    arg_prompt_align_map_[var3] = 64;
    
    is_concat_outer_map_[var1] = 1;
    is_concat_outer_map_[var2] = 0;
    
    concat_inner_dims_.push_back(var3);
    
    // Set the test data to the solver
    solver.hardware_use_map_ = hardware_use_map_;
    solver.total_cut_cons_ = total_cut_cons_;
    solver.mc_args_ = mc_args_;
    solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
    solver.arg_align_map_ = arg_align_map_;
    solver.arg_prompt_align_map_ = arg_prompt_align_map_;
    solver.is_concat_outer_map_ = is_concat_outer_map_;
    solver.concat_inner_dims_ = concat_inner_dims_;
    
    std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

    // 4. Check local buffer variables processing
    // Verify alignment strings are generated
    EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
    EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);
    
    std::cout<<result<<std::endl;
    // Verify prompt alignment for concat outer case
    EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);
    
    // Verify no prompt alignment for non-concat outer case
    EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}


TEST_F(TestAxesReorderSolverGen, GenInputInfoCase2Test) {
    std::vector<Expr> all_cons;
    std::vector<Expr> local_buffer_cons;
    std::vector<Expr> mc_mixed_cons;

    AxesReorderSolverGen solver("case_test", "TilingData");
    std::map<HardwareDef, Expr> hardware_use_map_;
    std::vector<Expr> total_cut_cons_;
    std::vector<Expr> mc_args_;
    std::vector<Expr> local_buffer_tiling_vars_;
    ExprExprMap arg_align_map_;
    ExprUintMap arg_prompt_align_map_;
    ExprUintMap is_concat_outer_map_;
    std::vector<Expr> concat_inner_dims_;

    // Initialize test data
    hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(1);
    hardware_use_map_[HardwareDef::GM] = CreateExpr(1024);
    hardware_use_map_[HardwareDef::UB] = CreateExpr(512);
    
    total_cut_cons_.push_back(CreateExpr(256));
    total_cut_cons_.push_back(CreateExpr(128));

    
    mc_args_.push_back(CreateExpr("mc_var1"));
    mc_args_.push_back(CreateExpr("mc_var2"));

    Expr var1 = CreateExpr("local_var1");
    Expr var2 = CreateExpr("local_var2");
    Expr var3 = ge::Symbol("local_var3");

    
    local_buffer_tiling_vars_.push_back(var1);
    local_buffer_tiling_vars_.push_back(var2);
    
    arg_align_map_[var1] = ge::Symbol(64);
    arg_align_map_[var2] = ge::Symbol(32);
    
    arg_prompt_align_map_[var1] = 128;
    arg_prompt_align_map_[var3] = 64;
    
    is_concat_outer_map_[var1] = 1;
    is_concat_outer_map_[var2] = 0;
    
    concat_inner_dims_.push_back(var3);
    
    // Set the test data to the solver
    solver.hardware_use_map_ = hardware_use_map_;
    solver.total_cut_cons_ = total_cut_cons_;
    solver.mc_args_ = mc_args_;
    solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
    solver.arg_align_map_ = arg_align_map_;
    solver.arg_prompt_align_map_ = arg_prompt_align_map_;
    solver.is_concat_outer_map_ = is_concat_outer_map_;
    solver.concat_inner_dims_ = concat_inner_dims_;
    
    std::string result = solver.GenInputInfo(all_cons, local_buffer_cons, mc_mixed_cons);

    // 4. Check local buffer variables processing
    // Verify alignment strings are generated
    EXPECT_TRUE(result.find("local_var1.align = std::max(1, 64)") != std::string::npos);
    EXPECT_TRUE(result.find("local_var2.align = std::max(1, 32)") != std::string::npos);
    
    std::cout<<result<<std::endl;
    // Verify prompt alignment for concat outer case
    EXPECT_TRUE(result.find("local_var1.prompt_align = 128") != std::string::npos);
    
    // Verify no prompt alignment for non-concat outer case
    EXPECT_TRUE(result.find("local_var2.prompt_align") == std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenUBThresholdFunc_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.enable_multicore_ub_tradeoff_ = true;
  std::string actual = solver.GenUBThresholdFunc();
  EXPECT_TRUE(actual.find("return (ub_size - 0) > static_cast<uint32_t>(input_.ub_threshold * input_.ub_size);") != std::string::npos);
}

TEST_F(TestAxesReorderSolverGen, GenSolverFuncImpl_WithUBAndVariables) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.enable_multicore_ub_tradeoff_ = true;
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}
TEST_F(TestAxesReorderSolverGen, test_contain_heavy_op) {
  // Case 3: Normal case with UB and variables
  AxesReorderSolverGen solver("case_test", "TilingData");
  std::vector<Expr> mc_args_;
  std::vector<Expr> local_buffer_tiling_vars_;
  std::map<HardwareDef, Expr> hardware_use_map_;
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr var3 = CreateExpr("var3");
  Expr var4 = CreateExpr("var4");

  mc_args_.push_back(var1);
  mc_args_.push_back(var2);

  local_buffer_tiling_vars_.push_back(var3);
  local_buffer_tiling_vars_.push_back(var4);

  hardware_use_map_[HardwareDef::CORENUM] = CreateExpr(100) / var1 * var3;
  hardware_use_map_[HardwareDef::UB] = var3 * var4;

  solver.mc_args_ = mc_args_;
  solver.local_buffer_tiling_vars_ = local_buffer_tiling_vars_;
  solver.hardware_use_map_ = hardware_use_map_;
  solver.enable_multicore_ub_tradeoff_ = true;
  std::string actual = solver.GenSolverFuncImpl();
  EXPECT_TRUE(actual.find("solver.Run(true, ") != std::string::npos);
}