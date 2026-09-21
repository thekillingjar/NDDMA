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
#include <fstream>
#include "base/base_types.h"
#include "base/model_info.h"
#include "test_expr/test_stub.h"
#include "stub/stub_model_info.h"
#include "common/util/mem_utils.h"
#define private public
#define protected public
#include "generator/preprocess/args_manager.h"
#include "generator/tiling_code_gen_impl.h"
#include "generator/axes_reorder_tiling_code_gen_impl.h"

using namespace att;

class TestArgManager : public ::testing::Test {
 public:
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  
  void SetUp() override {
    model_info_ = CreateModelInfo();
  }

  void TearDown() override {
  }
 ModelInfo model_info_;
};

std::set<std::string> GetVarsName(const std::vector<Expr>& args)
{
  std::set<std::string> vars_set;
  for (const auto arg : args) {
    DUMPS(Str(arg));
    vars_set.insert(Str(arg));
  }
  return vars_set;
}

TEST_F(TestArgManager, test_split_args) {
  ModelInfo model_info;
  Expr s0 = CreateExpr("s0");
  Expr s1 = CreateExpr("s1");
  SymVarInfoPtr sym_m = std::make_shared<SymVarInfo>(s0 * s1);
  AttAxisPtr z0 = std::make_shared<AttAxis>();
  z0->axis_pos = AxisPosition::ORIGIN;
  z0->size = sym_m;
  model_info.arg_list.emplace_back(z0);
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  auto input_args = args_manager.GetInputVars();
  std::set<std::string> target_search_args = {"s0", "s1"};
  for (auto input_arg : input_args) {
    EXPECT_TRUE(target_search_args.find(Str(input_arg)) != target_search_args.end());
  }
}

TEST_F(TestArgManager, process_no_replace)
{
  ArgsManager args_manager(model_info_);
  EXPECT_TRUE(args_manager.Process(false));
  auto search_args = args_manager.GetSearchableVars();
  auto search_args_set = GetVarsName(search_args);
  std::set<std::string> target_search_args = {"tilem_size", "stepm_size", "basem_size",
                                              "tilen_size", "stepn_size", "basen_size"};
  EXPECT_TRUE(search_args_set == target_search_args);

  auto input_args = args_manager.GetInputVars();
  auto input_args_set = GetVarsName(input_args);
  std::set<std::string> target_input_args = {"m_size", "n_size"};
  EXPECT_TRUE(input_args_set == target_input_args);

  auto const_args = args_manager.GetConstVars();
  std::map<std::string, uint32_t> const_values;
  for (const auto const_var : const_args) {
    const_values[Str(const_var.first)] = const_var.second;
  }
  std::map<std::string, uint32_t> target_const_args = {{"k_size", 128}};
  EXPECT_TRUE(const_values == target_const_args);

  auto l0a_args = args_manager.GetSearchableVars(HardwareDef::L0A);
  EXPECT_FALSE(l0a_args.empty());
  auto l0a_args_set = GetVarsName(l0a_args);
  std::set<std::string> target_l0a_args = {"basem_size"};
  EXPECT_TRUE(l0a_args_set == target_l0a_args);

  auto l0b_args = args_manager.GetSearchableVars(HardwareDef::L0B);
  auto l0b_args_set = GetVarsName(l0b_args);
  std::set<std::string> target_l0b_args = {"basen_size"};
  EXPECT_TRUE(l0b_args_set == target_l0b_args);

  auto l0c_args = args_manager.GetSearchableVars(HardwareDef::L0C);
  auto l0c_args_set = GetVarsName(l0c_args);
  std::set<std::string> target_l0c_args = {"basem_size", "basen_size"};
  EXPECT_TRUE(l0c_args_set == target_l0c_args);

  auto l1_args = args_manager.GetSearchableVars(HardwareDef::L1);
  auto l1_args_set = GetVarsName(l1_args);
  std::set<std::string> target_l1_args = {"stepm_size", "stepn_size"};
  EXPECT_TRUE(l1_args_set == target_l1_args);

  auto l2_args = args_manager.GetSearchableVars(HardwareDef::L2);
  auto l2_args_set = GetVarsName(l2_args);
  std::set<std::string> target_l2_args = {"tilem_size", "tilen_size"};
  EXPECT_TRUE(l2_args_set == target_l2_args);

  auto core_args = args_manager.GetSearchableVars(HardwareDef::CORENUM);
  auto core_args_set = GetVarsName(core_args);
  std::set<std::string> target_core_args = {"stepm_size", "stepn_size"};
  EXPECT_TRUE(core_args_set == target_core_args);

  auto replaced_vars = args_manager.GetVarsRelations();
  EXPECT_TRUE(replaced_vars.empty());
  auto replaced_expr = args_manager.GetExprRelations();
  EXPECT_TRUE(replaced_expr.empty());
  auto solved_expr = args_manager.GetSolvedVars();
  EXPECT_TRUE(solved_expr.empty());

  auto baseM_scopes = args_manager.GetRelatedHardware(CreateExpr("basem_size"));
  std::vector<HardwareDef> target_baseM_scopes{HardwareDef::L0A, HardwareDef::L0C};
  EXPECT_TRUE(baseM_scopes == target_baseM_scopes);

  auto baseN_scopes = args_manager.GetRelatedHardware(CreateExpr("basen_size"));
  std::vector<HardwareDef> target_baseN_scopes{HardwareDef::L0B, HardwareDef::L0C};
  EXPECT_TRUE(baseN_scopes == target_baseN_scopes);

  auto total_scope_cons = args_manager.GetTotalHardwareCons();
  EXPECT_EQ(total_scope_cons.size(), 7);

  auto l0a_occupy = args_manager.GetUsedHardwareInfo(HardwareDef::L0A);
  DUMPS(Str(l0a_occupy));
  DUMPS(Str(total_scope_cons[HardwareDef::L0A]));
  EXPECT_TRUE(l0a_occupy == (total_scope_cons[HardwareDef::L0A]));
  auto l0b_occupy = args_manager.GetUsedHardwareInfo(HardwareDef::L0B);
  DUMPS(Str(l0b_occupy));
  DUMPS(Str(total_scope_cons[HardwareDef::L0B]));
  EXPECT_TRUE(l0b_occupy == (total_scope_cons[HardwareDef::L0B]));
  EXPECT_TRUE(l0a_occupy.ContainVar(CreateExpr("basem_size")));
  EXPECT_TRUE(l0b_occupy.ContainVar(CreateExpr("basen_size")));

  auto total_cut_cons = args_manager.GetTotalCutCons();
  EXPECT_EQ(total_cut_cons.size(), 4);
  DUMPS(Str(total_cut_cons[0]));

  auto objs = args_manager.GetObjectFunc();
  EXPECT_EQ(objs.size(), 2);

  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("basem_size"))[0]), "m_size");
  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("stepm_size"))[0]), "m_size");
  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("tilem_size"))[0]), "m_size");
  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("k_size"))[0]), "k_size");

  EXPECT_TRUE(args_manager.GetMaxValue(CreateExpr("basem_size")) == (CreateExpr("m_size")));
  EXPECT_TRUE(args_manager.GetMaxValue(CreateExpr("stepm_size")) == (CreateExpr("m_size")));
  EXPECT_TRUE(args_manager.GetMaxValue(CreateExpr("tilem_size")) == (CreateExpr("m_size")));

  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("basem_size")) == (CreateExpr("m_size")));
  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("stepm_size")) == (CreateExpr(16)));   // align
  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("tilem_size")) == (CreateExpr(16)));   // align
  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("k_size")) == (CreateExpr(128)));

  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("basem_size"))[0] == (CreateExpr("stepm_size")));
  DUMPS(Str(args_manager.GetParentVars(CreateExpr("basem_size"))[0]));
  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("stepm_size"))[0] == (CreateExpr("tilem_size")));
  DUMPS(Str(args_manager.GetParentVars(CreateExpr("stepm_size"))[0]));
  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("tilem_size"))[0] == (CreateExpr("m_size")));
  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("m_size")).empty());

  auto innerest_dims = args_manager.GetNodeInnerestDimSizes();
  auto innerest_dims_set = GetVarsName(innerest_dims);
  std::set<std::string> target_innerest_dims = {"tilem_size", "stepm_size", "basem_size",
                                                "tilen_size", "stepn_size", "basen_size"};
  EXPECT_TRUE(innerest_dims_set == target_innerest_dims);
}

TEST_F(TestArgManager, process_do_replace)
{
  ArgsManager args_manager(model_info_);
  EXPECT_TRUE(args_manager.Process(true));
  EXPECT_EQ(args_manager.GetTilingCaseId(), 0);
  auto search_args = args_manager.GetSearchableVars();
  auto search_args_set = GetVarsName(search_args);
  std::set<std::string> target_search_args = {"tilem_size_div_align", "stepm_size_div_align",
                                              "basem_size_base", "tilen_size_div_align",
                                              "stepn_size_div_align", "basen_size_base"};
  EXPECT_TRUE(search_args_set == target_search_args);

  auto input_args = args_manager.GetInputVars();
  auto input_args_set = GetVarsName(input_args);
  std::set<std::string> target_input_args = {"m_size", "n_size"};
  EXPECT_TRUE(input_args_set == target_input_args);

  auto const_args = args_manager.GetConstVars();
  std::map<std::string, uint32_t> const_values;
  for (const auto const_var : const_args) {
    const_values[Str(const_var.first)] = const_var.second;
  }
  std::map<std::string, uint32_t> target_const_args = {{"k_size", 128}};
  EXPECT_TRUE(const_values == target_const_args);

  auto l0a_args = args_manager.GetSearchableVars(HardwareDef::L0A);
  EXPECT_FALSE(l0a_args.empty());
  auto l0a_args_set = GetVarsName(l0a_args);
  std::set<std::string> target_l0a_args = {"basem_size_base"};
  EXPECT_TRUE(l0a_args_set == target_l0a_args);

  auto l0b_args = args_manager.GetSearchableVars(HardwareDef::L0B);
  auto l0b_args_set = GetVarsName(l0b_args);
  std::set<std::string> target_l0b_args = {"basen_size_base"};
  EXPECT_TRUE(l0b_args_set == target_l0b_args);

  auto l0c_args = args_manager.GetSearchableVars(HardwareDef::L0C);
  auto l0c_args_set = GetVarsName(l0c_args);
  std::set<std::string> target_l0c_args = {"basem_size_base", "basen_size_base"};
  EXPECT_TRUE(l0c_args_set == target_l0c_args);

  auto l1_args = args_manager.GetSearchableVars(HardwareDef::L1);
  auto l1_args_set = GetVarsName(l1_args);
  std::set<std::string> target_l1_args = {"stepm_size_div_align", "stepn_size_div_align"};
  EXPECT_TRUE(l1_args_set == target_l1_args);

  auto l2_args = args_manager.GetSearchableVars(HardwareDef::L2);
  auto l2_args_set = GetVarsName(l2_args);
  std::set<std::string> target_l2_args = {"tilem_size_div_align", "tilen_size_div_align"};
  EXPECT_TRUE(l2_args_set == target_l2_args);

  auto core_args = args_manager.GetSearchableVars(HardwareDef::CORENUM);
  auto core_args_set = GetVarsName(core_args);
  std::set<std::string> target_core_args = {"stepm_size_div_align", "stepn_size_div_align"};
  EXPECT_TRUE(core_args_set == target_core_args);

  auto replaced_vars = args_manager.GetVarsRelations();
  EXPECT_FALSE(replaced_vars.empty());
  for (auto replaced_var : replaced_vars) {
    DUMPS("new var: " + Str(replaced_var.first) + " -- old vars: " + Str(replaced_var.second));
  }
  auto replaced_expr = args_manager.GetExprRelations();
  for (auto replaced_var : replaced_expr) {
    DUMPS("olg var: " + Str(replaced_var.first) + " -- new exprs: " + Str(replaced_var.second));
  }
  EXPECT_FALSE(replaced_expr.empty());
  auto solved_expr = args_manager.GetSolvedVars();
  EXPECT_TRUE(solved_expr.empty());

  auto baseM_scopes = args_manager.GetRelatedHardware(CreateExpr("basem_size_base"));
  std::vector<HardwareDef> target_baseM_scopes{HardwareDef::L0A, HardwareDef::L0C};
  EXPECT_TRUE(baseM_scopes == target_baseM_scopes);

  auto baseN_scopes = args_manager.GetRelatedHardware(CreateExpr("basen_size_base"));
  std::vector<HardwareDef> target_baseN_scopes{HardwareDef::L0B, HardwareDef::L0C};
  EXPECT_TRUE(baseN_scopes == target_baseN_scopes);

  auto stepN_scopes = args_manager.GetRelatedHardware(CreateExpr("stepn_size_div_align"));
  std::vector<HardwareDef> target_stepN_scopes{HardwareDef::L1, HardwareDef::CORENUM};
  EXPECT_TRUE(stepN_scopes == target_stepN_scopes);

  auto total_scope_cons = args_manager.GetTotalHardwareCons();
  EXPECT_EQ(total_scope_cons.size(), 7);

  auto l0a_occupy = args_manager.GetUsedHardwareInfo(HardwareDef::L0A);
  EXPECT_TRUE(l0a_occupy == (total_scope_cons[HardwareDef::L0A]));
  auto l0b_occupy = args_manager.GetUsedHardwareInfo(HardwareDef::L0B);
  EXPECT_TRUE(l0b_occupy == (total_scope_cons[HardwareDef::L0B]));
  EXPECT_TRUE(l0a_occupy.ContainVar(CreateExpr("basem_size_base")));
  EXPECT_TRUE(l0b_occupy.ContainVar(CreateExpr("basen_size_base")));

  auto total_cut_cons = args_manager.GetTotalCutCons();
  EXPECT_EQ(total_cut_cons.size(), 4);
  DUMPS(Str(total_cut_cons[0]));

  auto objs = args_manager.GetObjectFunc();
  EXPECT_EQ(objs.size(), 2);

  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("basem_size_base"))[0]), "m_size");
  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("stepm_size_div_align"))[0]), "m_size");
  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("tilem_size_div_align"))[0]), "m_size");
  EXPECT_EQ(Str(args_manager.GetAncestor(CreateExpr("k_size"))[0]), "k_size");

  EXPECT_FALSE(args_manager.GetMaxValue(CreateExpr("basem_size_base")) == (CreateExpr("m_size")));
  EXPECT_FALSE(args_manager.GetMaxValue(CreateExpr("stepm_size_div_align")) == (CreateExpr("m_size")));
  EXPECT_FALSE(args_manager.GetMaxValue(CreateExpr("tilem_size_div_align")) == (CreateExpr("m_size")));

  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("basem_size_base")) == (ge::sym::Log((CreateExpr("m_size") / CreateExpr(16)), CreateExpr(2))));
  DUMPS(Str(args_manager.GetDefaultInitValue(CreateExpr("basem_size_base"))));
  EXPECT_FALSE(args_manager.GetDefaultInitValue(CreateExpr("stepm_size_div_align")) == (CreateExpr(16)));   // align
  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("stepm_size_div_align")) == ge::sym::kSymbolOne);   // align
  DUMPS(Str(args_manager.GetDefaultInitValue(CreateExpr("stepm_size_div_align"))));
  EXPECT_FALSE(args_manager.GetDefaultInitValue(CreateExpr("tilem_size_div_align")) == (CreateExpr(16)));
  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("tilem_size_div_align")) == ge::sym::kSymbolOne);
  DUMPS(Str(args_manager.GetDefaultInitValue(CreateExpr("tilem_size_div_align"))));
  EXPECT_TRUE(args_manager.GetDefaultInitValue(CreateExpr("k_size")) == (CreateExpr(128)));

  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("basem_size"))[0] == (CreateExpr("stepm_size")));
  DUMPS(Str(args_manager.GetParentVars(CreateExpr("basem_size"))[0]));
  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("stepm_size"))[0] == (CreateExpr("tilem_size")));
  DUMPS(Str(args_manager.GetParentVars(CreateExpr("stepm_size"))[0]));
  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("tilem_size"))[0] == (CreateExpr("m_size")));
  EXPECT_TRUE(args_manager.GetParentVars(CreateExpr("m_size")).empty());
  EXPECT_FALSE(args_manager.GetParentVars(CreateExpr("basem_size_base"))[0] == (CreateExpr("stepm_size")));
  DUMPS(Str(args_manager.GetParentVars(CreateExpr("basem_size_base"))[0]));

  auto innerest_dims = args_manager.GetNodeInnerestDimSizes();
  auto innerest_dims_set = GetVarsName(innerest_dims);
  std::set<std::string> target_innerest_dims = {"tilem_size_div_align", "stepm_size_div_align",
                                              "basem_size_base", "tilen_size_div_align",
                                              "stepn_size_div_align", "basen_size_base"};
  EXPECT_TRUE(innerest_dims_set == target_innerest_dims);
}

TEST_F(TestArgManager, process_replace_after_no_replace)
{
  ArgsManager args_manager(model_info_);
  EXPECT_TRUE(args_manager.Process(false));
  auto ori_search_args = args_manager.GetSearchableVars();
  auto ori_search_args_set = GetVarsName(ori_search_args);
  std::set<std::string> ori_target_search_args = {"tilem_size", "stepm_size", "basem_size",
                                              "tilen_size", "stepn_size", "basen_size"};
  EXPECT_TRUE(ori_search_args_set == ori_target_search_args);

  auto input_args = args_manager.GetInputVars();
  auto input_args_set = GetVarsName(input_args);
  std::set<std::string> target_input_args = {"m_size", "n_size"};
  EXPECT_TRUE(input_args_set == target_input_args);

  auto const_args = args_manager.GetConstVars();
  std::map<std::string, uint32_t> const_values;
  for (const auto const_var : const_args) {
    const_values[Str(const_var.first)] = const_var.second;
  }
  std::map<std::string, uint32_t> target_const_args = {{"k_size", 128}};
  EXPECT_TRUE(const_values == target_const_args);

  std::vector<Expr> sloved_vars;
  sloved_vars.push_back(CreateExpr("basem_size"));
  sloved_vars.push_back(CreateExpr("basen_size"));
  args_manager.SetSolvedVars(sloved_vars);
  EXPECT_TRUE(args_manager.DoVarsReplace() == true);
  auto manager_solved_args = args_manager.GetSolvedVars();
  auto manager_solved_args_set = GetVarsName(manager_solved_args);
  std::set<std::string> target_solved_args = {"basem_size", "basen_size"};
  EXPECT_TRUE(manager_solved_args_set == target_solved_args);
  auto search_args = args_manager.GetSearchableVars();
  auto search_args_set = GetVarsName(search_args);
  std::set<std::string> target_search_args = {"tilem_size_div_align", "stepm_size_div_align",
                                              "tilen_size_div_align", "stepn_size_div_align"};
  EXPECT_TRUE(search_args_set == target_search_args);
  auto replaced_vars = args_manager.GetVarsRelations();
  EXPECT_FALSE(replaced_vars.empty());
  for (auto replaced_var : replaced_vars) {
    DUMPS("new var: " + Str(replaced_var.first) + " -- old vars: " + Str(replaced_var.second));
  }
  auto replaced_expr = args_manager.GetExprRelations();
  for (auto replaced_var : replaced_expr) {
    DUMPS("olg var: " + Str(replaced_var.first) + " -- new exprs: " + Str(replaced_var.second));
  }
  EXPECT_FALSE(replaced_expr.empty());
}

TEST_F(TestArgManager, process_replace_dumps) {
  ArgsManager args_manager(model_info_);
  EXPECT_TRUE(args_manager.Process(true));
  auto vars_string = MakeJson(args_manager.vars_infos_);
  std::ofstream out_file("./vars_info_st.json");
  if (out_file.is_open()) {
    out_file << vars_string;
    out_file.close();
  }
}

TEST_F(TestArgManager, vars_not_find) {
  ArgsManager args_manager(model_info_);
  auto test = CreateExpr("test");
  EXPECT_TRUE(args_manager.GetAncestor(test).empty());
  EXPECT_TRUE(args_manager.GetMaxValue(test) == 1);
  EXPECT_TRUE(args_manager.GetMinValue(test) == 1);
  EXPECT_TRUE(args_manager.GetDefaultInitValue(test) == 1);
  EXPECT_TRUE(args_manager.GetVarAlignValue(test) == 0u);
  EXPECT_TRUE(args_manager.GetVarPromptAlignValue(test) == 0u);
}

TEST_F(TestArgManager, IsConcatOuterDim) {
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr undefined_var = CreateExpr("undefined_var");

  ArgsManager manager(model_info_);
  
  VarInfo info1;
  info1.is_concat_outer_dim = true;
  manager.vars_infos_[var1] = info1;
  
  VarInfo info2;
  info2.is_concat_outer_dim = false;
  manager.vars_infos_[var2] = info2;
  EXPECT_TRUE(manager.IsConcatOuterDim(var1));
  EXPECT_FALSE(manager.IsConcatOuterDim(var2));
  EXPECT_FALSE(manager.IsConcatOuterDim(undefined_var));
}

TEST_F(TestArgManager, IsConcatInnerDim) {
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr undefined_var = CreateExpr("undefined_var");

  ArgsManager manager(model_info_);
  
  VarInfo info1;
  info1.is_concat_inner_dim= true;
  manager.vars_infos_[var1] = info1;
  
  VarInfo info2;
  info2.is_concat_inner_dim = false;
  manager.vars_infos_[var2] = info2;
  EXPECT_TRUE(manager.IsConcatInnerDim(var1));
  EXPECT_FALSE(manager.IsConcatInnerDim(var2));
  EXPECT_FALSE(manager.IsConcatInnerDim(undefined_var));
}

TEST_F(TestArgManager, test_small_shape_pattern) {
  ModelInfo model_info;
  Expr s0 = CreateExpr("s0");
  Expr s1 = CreateExpr("s1");
  SymVarInfoPtr sym_m = std::make_shared<SymVarInfo>(s0);
  AttAxisPtr z0 = std::make_shared<AttAxis>();
  z0->axis_pos = AxisPosition::ORIGIN;
  z0->size = sym_m;
  z0->name = "z0";

  SymVarInfoPtr sym_n = std::make_shared<SymVarInfo>(s1);
  AttAxisPtr z1 = std::make_shared<AttAxis>();
  z1->axis_pos = AxisPosition::ORIGIN;
  z1->size = sym_n;
  z1->name = "z1";

  model_info.arg_list.emplace_back(z0);
  model_info.arg_list.emplace_back(z1);
  std::map<HardwareDef, Expr> hardware_cons;
  hardware_cons[HardwareDef::UB] = CreateExpr(1024) * s1;
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024) * s0;
  model_info.hardware_cons = hardware_cons;
  TilingCodeGenConfig config;
  TilingModelInfo model_infos{model_info};
  ScoreFuncs score_funcs;
  TilingCodeGenImplPtr impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(ge::MakeShared<AxesReorderTilingCodeGenImpl>(
    "op", config, model_infos, score_funcs, false));
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  EXPECT_FALSE(impl->HitSmallShapePattern(args_manager));
}

TEST_F(TestArgManager, test_small_shape_pattern_case2) {
  ModelInfo model_info;
  Expr s0 = CreateExpr(128);
  Expr s1 = CreateExpr("s1_size");
  SymConstInfoPtr sym_m = std::make_shared<SymConstInfo>(s0);
  AttAxisPtr z0 = std::make_shared<AttAxis>();
  z0->axis_pos = AxisPosition::ORIGIN;
  z0->size = sym_m; 
  z0->name = "z0";

  SymVarInfoPtr sym_n = std::make_shared<SymVarInfo>(s1);
  AttAxisPtr z0t = std::make_shared<AttAxis>();
  z0t->axis_pos = AxisPosition::INNER;
  z0t->size = sym_n; 
  z0t->name = "z0t";
  z0t->orig_axis.emplace_back(z0.get());
  
  model_info.arg_list.emplace_back(z0);
  model_info.arg_list.emplace_back(z0t);


  std::map<HardwareDef, Expr> hardware_cons;
  hardware_cons[HardwareDef::UB] = CreateExpr(1024) * s1;
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024) * s0;
  model_info.hardware_cons = hardware_cons;
  TilingCodeGenConfig config;
  TilingModelInfo model_infos = {model_info};
  ScoreFuncs score_funcs;
  TilingCodeGenImplPtr impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(ge::MakeShared<AxesReorderTilingCodeGenImpl>(
    "op", config, model_infos, score_funcs, false));
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  EXPECT_TRUE(impl->HitSmallShapePattern(args_manager));
}