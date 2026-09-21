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
#include "gmock/gmock.h"
#include "base/base_types.h"
#include "base/model_info.h"
#include "generator/preprocess/var_info.h"
#include "common/util/mem_utils.h"
#include <iostream>
#include <fstream>
#define private public
#include "generator/preprocess/args_replace.h"
#include "generator/preprocess/args_manager.h"
#define private public
#define protected public
#include "generator/tiling_code_gen_impl.h"
#include "generator/axes_reorder_tiling_code_gen_impl.h"
using namespace att;
using namespace testing;

class ArgsManagerUtest : public ::testing::Test {
 public:
  static void TearDownTestCase()
  {
    std::cout << "Test end." << std::endl;
  }
  static void SetUpTestCase()
  {
    std::cout << "Test begin." << std::endl;
  }
  void SetUp() override {
  }

  void TearDown() override {
  }
  void dumps(const std::string& res)
  {
    std::cout << res << std::endl;
  }
};

TEST_F(ArgsManagerUtest, test_set_size_info)
{
  VarInfo info;
  SymVarInfoPtr sym_var = std::make_shared<SymVarInfo>(CreateExpr("basem_size"));
  SymVarInfoPtr sym_var1 = std::make_shared<SymVarInfo>(CreateExpr("stepm_size"));
  SymVarInfoPtr sym_var2 = std::make_shared<SymVarInfo>(CreateExpr("tilem_size"));
  AttAxisPtr axis = std::make_shared<AttAxis>();
  axis->name = "basem";
  AttAxisPtr ori_axis1 = std::make_shared<AttAxis>();
  ori_axis1->name = "stepm";
  AttAxisPtr ori_axis2 = std::make_shared<AttAxis>();
  ori_axis2->name = "tilem";
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(info.max_value == 1);
  axis->orig_axis.emplace_back(ori_axis1.get());
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(info.max_value == 1);
  axis->orig_axis.clear();
  ori_axis1->size = sym_var1;
  axis->orig_axis.emplace_back(ori_axis1.get());
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(IsValid(info.max_value));
  EXPECT_TRUE(info.max_value == CreateExpr("stepm_size"));
  ori_axis2->size = sym_var2;
  axis->orig_axis.emplace_back(ori_axis2.get());
  info = ArgsManager::SetSizeInfo(info, sym_var, axis.get());
  EXPECT_TRUE(IsValid(info.max_value));
  EXPECT_TRUE(info.max_value == (CreateExpr("stepm_size") * CreateExpr("tilem_size")));
}

TEST_F(ArgsManagerUtest, test_set_init_info)
{
  VarInfo info;
  info.is_const_var = true;
  info.const_value = 1u;
  info = ArgsManager::SetInitSize(info, false);
  EXPECT_TRUE(info.init_value == 1);
  info = ArgsManager::SetInitSize(info, true);
  EXPECT_TRUE(info.init_value == 1);
  info.is_const_var = false;
  info.max_value = CreateExpr("m");
  info.align = ge::Symbol(16u);
  info = ArgsManager::SetInitSize(info, false);
  EXPECT_TRUE(info.init_value == 16);
  info = ArgsManager::SetInitSize(info, true);
  EXPECT_TRUE(info.init_value == CreateExpr("m"));
}

ModelInfo StubModelInfo()
{
  ModelInfo model_info;
  // set m
  Expr expr_m = CreateExpr("m_size");
  Expr expr_tilem = CreateExpr("tilem_size");
  Expr expr_stepm = CreateExpr("stepm_size");
  Expr expr_basem = CreateExpr("basem_size");
  SymVarInfoPtr sym_m = std::make_shared<SymVarInfo>(expr_m);
  SymVarInfoPtr sym_tilem = std::make_shared<SymVarInfo>(expr_tilem);
  sym_tilem->align = ge::Symbol(16);
  sym_tilem->related_scope = {HardwareDef::L2};
  SymVarInfoPtr sym_stepm = std::make_shared<SymVarInfo>(expr_stepm);
  sym_stepm->align = ge::Symbol(16);
  sym_stepm->related_scope = {HardwareDef::L1, HardwareDef::CORENUM};
  SymVarInfoPtr sym_basem = std::make_shared<SymVarInfo>(expr_basem);
  sym_basem->align = ge::Symbol(16);
  sym_basem->related_scope = {HardwareDef::L0A, HardwareDef::L0C};
  AttAxisPtr m = std::make_shared<AttAxis>();
  AttAxisPtr tilem = std::make_shared<AttAxis>();
  AttAxisPtr stepm = std::make_shared<AttAxis>();
  AttAxisPtr basem = std::make_shared<AttAxis>();
  m->name = "m";
  m->axis_pos = AxisPosition::ORIGIN;
  m->bind_multicore = false;
  m->is_last = false;
  m->is_node_innerest_dim = false;
  m->size = sym_m;
  
  tilem->name = "tilem";
  tilem->axis_pos = AxisPosition::INNER;
  tilem->bind_multicore = false;
  tilem->is_last = false;
  tilem->is_node_innerest_dim = true;
  tilem->size = sym_tilem;
  tilem->orig_axis.push_back(m.get());
  tilem->from_axis = {m.get()};

  stepm->name = "stepm";
  stepm->axis_pos = AxisPosition::INNER;
  stepm->bind_multicore = true;
  stepm->is_last = false;
  stepm->is_node_innerest_dim = true;
  stepm->size = sym_stepm;
  stepm->orig_axis.push_back(m.get());
  stepm->from_axis = {tilem.get()};

  basem->name = "basem";
  basem->axis_pos = AxisPosition::INNER;
  basem->bind_multicore = false;
  basem->is_last = true;
  basem->is_node_innerest_dim = true;
  basem->size = sym_basem;
  basem->orig_axis.push_back(m.get());
  basem->from_axis = {stepm.get()};
  model_info.arg_list.emplace_back(m);
  model_info.arg_list.emplace_back(tilem);
  model_info.arg_list.emplace_back(stepm);
  model_info.arg_list.emplace_back(basem);

  Expr l0a_occupy = (expr_basem * CreateExpr(4));
  Expr l0c_occupy = (expr_basem * CreateExpr(4));
  Expr l1_occupy = (expr_stepm * CreateExpr(4));
  Expr l2_occupy = (expr_tilem * CreateExpr(2));
  Expr core_num = (expr_tilem / expr_stepm);
  std::map<HardwareDef, Expr> hardware_cons;
  model_info.hardware_cons[HardwareDef::L0A] = l0a_occupy;
  model_info.hardware_cons[HardwareDef::L0C] = l0c_occupy;
  model_info.hardware_cons[HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[HardwareDef::L2] = l2_occupy;
  model_info.hardware_cons[HardwareDef::CORENUM] = core_num;
  model_info.hardware_cons[HardwareDef::UB] = CreateExpr(0L);
  
  Expr mac = (expr_basem / CreateExpr(256));
  model_info.objects[PipeType::AIC_MAC] = mac;
  model_info.tiling_case_id = 0;
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepm, expr_basem));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilem - expr_stepm));
  return model_info;
}

TEST_F(ArgsManagerUtest, test_set_ori_exprs)
{
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  args_manager.vars_infos_ = ArgsManager::GetOrigVarInfos(info);
  args_manager.SetOrigExprs();
  EXPECT_EQ(args_manager.hardware_cons_.size(), 6u);
  EXPECT_EQ(args_manager.cut_leq_cons_.size(), 1u);
  EXPECT_EQ(args_manager.cut_eq_cons_.size(), 0);
}

TEST_F(ArgsManagerUtest, test_replace_vars)
{
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  args_manager.vars_infos_ = ArgsManager::GetOrigVarInfos(info);
  args_manager.SetOrigExprs();
  ExprExprMap replaced_vars;
  ExprExprMap replacements;
  ExprExprMap new_expr_replacements;
  auto res = args_manager.ReplaceVars(replaced_vars, replacements, new_expr_replacements);
  EXPECT_TRUE(res);
  EXPECT_EQ(replaced_vars.size(), 3u);
  EXPECT_EQ(replacements.size(), 3u);
  EXPECT_TRUE(replacements.find(CreateExpr("basem_size")) != replacements.end());
  EXPECT_TRUE(replacements[CreateExpr("basem_size")] == (CreateExpr(16) * ge::sym::Pow(CreateExpr(2), CreateExpr("basem_size_base"))));
  EXPECT_TRUE(replacements.find(CreateExpr("stepm_size")) != replacements.end());
  EXPECT_TRUE(replacements[CreateExpr("stepm_size")] == ((ge::sym::Max((CreateExpr(16) * ge::sym::Pow(CreateExpr(2), CreateExpr("basem_size_base"))), CreateExpr(16)) * CreateExpr("stepm_size_div_align"))));
  for (auto hard_cons : args_manager.hardware_cons_) {
    dumps("scope: " + std::to_string(static_cast<int32_t>(hard_cons.first)) + " --- " + Str(hard_cons.second));
  }
  for (auto obj : args_manager.objs_) {
    dumps("Pipe: " + std::to_string(static_cast<int32_t>(obj.first)) + " --- " + Str(obj.second));
  }
  for (auto leq_cons : args_manager.cut_leq_cons_) {
    dumps("Leq: " + Str(leq_cons));
  }
  for (auto eq_cons : args_manager.cut_eq_cons_) {
    dumps("Eq: " + Str(eq_cons.first / eq_cons.second));
  }
}

TEST_F(ArgsManagerUtest, test_update_vars)
{
  ModelInfo model_info;
  ArgsManager args_manager(model_info);
  ExprExprMap replaced_vars;
  ExprExprMap replacements;
  ExprExprMap new_expr_replacements;
  replaced_vars.emplace(CreateExpr("basem_base"), CreateExpr("basem"));
  replacements.emplace(CreateExpr("basem"), (CreateExpr(16) * ge::sym::Pow(CreateExpr(2), CreateExpr("basem_base"))));
  replaced_vars.emplace(CreateExpr("stepm_align"), CreateExpr("stepm"));
  replacements.emplace(CreateExpr("stepm"), (ge::sym::Max((CreateExpr(16) * ge::sym::Pow(CreateExpr(2), CreateExpr("basem_base"))), CreateExpr(16)) * CreateExpr("stepm_align")));
  new_expr_replacements.emplace(CreateExpr("basem_base"), ge::sym::Log(CreateExpr(2), (CreateExpr("basem") / CreateExpr(16))));
  new_expr_replacements.emplace(CreateExpr("stepm_align"), (CreateExpr("stepm") / ge::sym::Max((CreateExpr(16) * ge::sym::Pow(CreateExpr(2), CreateExpr("basem_base"))), CreateExpr(16))));
  args_manager.ori_var_init_values_ = {{CreateExpr("basem"), CreateExpr(16)}, {CreateExpr("stepm"), CreateExpr(16)}};
  args_manager.ori_var_max_values_ = {{CreateExpr("basem"), CreateExpr("m")}, {CreateExpr("stepm"), CreateExpr("m")}};
  VarInfo basem_info;
  VarInfo stepm_info;
  VarInfo m_info;
  basem_info.do_search = true;
  basem_info.align = ge::Symbol(16u);
  basem_info.init_value = CreateExpr("m");
  basem_info.max_value = CreateExpr("m");
  stepm_info.do_search = true;
  stepm_info.align = ge::Symbol(16u);
  stepm_info.max_value = CreateExpr("m");
  stepm_info.init_value = CreateExpr("m");
  args_manager.vars_infos_.emplace(CreateExpr("basem"), basem_info);
  args_manager.vars_infos_.emplace(CreateExpr("stepm"), stepm_info);
  args_manager.vars_infos_.emplace(CreateExpr("m"), m_info);
  auto res = args_manager.UpdateVarInfos(replaced_vars, replacements, new_expr_replacements);
  EXPECT_TRUE(res);
  EXPECT_FALSE(args_manager.vars_infos_[CreateExpr("basem")].do_search);
  EXPECT_TRUE(args_manager.vars_infos_[CreateExpr("basem_base")].do_search);
  EXPECT_FALSE(args_manager.vars_infos_[CreateExpr("stepm")].do_search);
  EXPECT_TRUE(args_manager.vars_infos_[CreateExpr("stepm_align")].do_search);
  EXPECT_TRUE(IsValid(args_manager.vars_infos_[CreateExpr("stepm_align")].init_value));
  EXPECT_FALSE(!IsValid(args_manager.vars_infos_[CreateExpr("basem_base")].init_value));
  EXPECT_FALSE(!IsValid(args_manager.vars_infos_[CreateExpr("stepm_align")].max_value));
  EXPECT_TRUE(IsValid(args_manager.vars_infos_[CreateExpr("basem_base")].max_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("stepm_align")].init_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("basem_base")].max_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("basem_base")].init_value));
  dumps(Str(args_manager.vars_infos_[CreateExpr("stepm_align")].max_value));
}

TEST_F(ArgsManagerUtest, test_process_true)
{
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  auto res = args_manager.Process(true);
  EXPECT_TRUE(res);
  EXPECT_TRUE(args_manager.replacement_done_);
  auto var_info_str = MakeJson(args_manager.vars_infos_);\
  std::ofstream out_file("./vars_info_ut.json");
  if (out_file.is_open()) {
    out_file << var_info_str;
    out_file.close();
  }
}

TEST_F(ArgsManagerUtest, test_process_false)
{
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  auto res = args_manager.Process(false);
  EXPECT_TRUE(res);
  EXPECT_FALSE(args_manager.replacement_done_);
}

TEST_F(ArgsManagerUtest, test_do_var_replace)
{
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  auto res = args_manager.DoVarsReplace();
  EXPECT_FALSE(res);
  EXPECT_FALSE(args_manager.replacement_done_);
}

TEST_F(ArgsManagerUtest, test_do_var_replace_success)
{
  ModelInfo info = StubModelInfo();
  ArgsManager args_manager(info);
  args_manager.Process(false);
  auto res = args_manager.DoVarsReplace();
  EXPECT_TRUE(res);
  EXPECT_TRUE(args_manager.replacement_done_);
}

TEST_F(ArgsManagerUtest, test_get_orig_var_info)
{
  ModelInfo info = StubModelInfo();
  auto expr_infos = ArgsManager::GetOrigVarInfos(info);
  EXPECT_EQ(expr_infos.size(), 4);
  auto m_info = expr_infos[CreateExpr("m_size")];
  EXPECT_EQ(m_info.cut_eq_cons.size(), 0);
  EXPECT_EQ(m_info.cut_leq_cons.size(), 0);
  auto basem_info = expr_infos[CreateExpr("basem_size")];
  EXPECT_EQ(basem_info.cut_eq_cons.size(), 1);
  EXPECT_EQ(basem_info.cut_leq_cons.size(), 0);
  auto stepm_info = expr_infos[CreateExpr("stepm_size")];
  EXPECT_EQ(stepm_info.cut_eq_cons.size(), 1);
  EXPECT_EQ(stepm_info.cut_leq_cons.size(), 1);
  auto tilem_info = expr_infos[CreateExpr("tilem_size")];
  EXPECT_EQ(tilem_info.cut_eq_cons.size(), 0);
  EXPECT_EQ(tilem_info.cut_leq_cons.size(), 1);
}

TEST_F(ArgsManagerUtest, test_naive_var_info)
{
  ModelInfo model_info;
  ArgsManager args_manager(model_info);
  AttAxisPtr axis = std::make_shared<AttAxis>();
  auto info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.scopes.empty());
  SymVarInfoPtr var_size = std::make_shared<SymVarInfo>(CreateExpr("m"));
  SymConstInfoPtr const_size = std::make_shared<SymConstInfo>(CreateExpr("m"));
  const_size->const_value = 16u;
  axis->size = var_size;
  // MOCK_METHOD3(ArgsManager::SetSizeInfo).stubs().will(invoke(StubSetSizeInfo));
  // MOCK_METHOD2(ArgsManager::SetInitSize).stubs().will(invoke(StubSetInitSize));
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_FALSE(info.do_search);
  axis->axis_pos = AxisPosition::ORIGIN;
  axis->size = var_size;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_FALSE(info.do_search);
  EXPECT_TRUE(info.is_input_var);
  axis->axis_pos = AxisPosition::INNER;
  axis->size = var_size;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.do_search);
  EXPECT_TRUE(info.from_axis_size.empty());
  axis->size = const_size;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.is_const_var);
  EXPECT_EQ(info.const_value, 16u);
  EXPECT_FALSE(info.do_search);

  auto parent_axis = std::make_unique<AttAxis>();
  parent_axis->name = "ori_m";
  parent_axis->axis_pos = AxisPosition::ORIGIN;
  axis->from_axis.emplace_back(parent_axis.get());
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_TRUE(info.from_axis_size.empty());
  axis->from_axis[0]->size = std::make_shared<SymVarInfo>(CreateExpr("ori_m_size"));
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_EQ(Str(info.from_axis_size[0]), "ori_m_size");
  EXPECT_TRUE(info.orig_axis_size.empty());
  axis->orig_axis = axis->from_axis;
  info = ArgsManager::GetNaiveVarInfo(axis.get());
  EXPECT_EQ(Str(info.orig_axis_size[0]), "ori_m_size");
  info = ArgsManager::GetNaiveVarInfo(axis->from_axis[0]);
  EXPECT_EQ(Str(info.orig_axis_size[0]), "ori_m_size");
}

// TEST_F(ArgsManagerUtest, test_update_expr_value)
// {
//   Expr max_value = Mul(CreateExpr("m"), CreateExpr("k"));
//   Expr replace_value = Mul(Max(CreateExpr(16), Pow(CreateExpr(2), CreateExpr("new_m"))), CreateExpr("new_k"));
//   auto updated_expr = UpdateExprValue(max_value, replace_value);
//   dumps(Str(updated_expr));
//   Expr replace_value1 = Max(CreateExpr(16), Pow(CreateExpr(2), CreateExpr("new_m")));
//   auto updated_expr1 = UpdateExprValue(max_value, replace_value1);
//   EXPECT_EQ(updated_expr1, nullptr);
//   Expr replace_value2 = Mul(CreateExpr("x"), Pow(CreateExpr(2), CreateExpr("new_m")));
//   auto updated_expr2 = UpdateExprValue(max_value, replace_value2);
//   EXPECT_NE(updated_expr2, nullptr);
//   dumps(Str(updated_expr2));

//   Expr replace_value3 = Mul(Max(CreateExpr("m"), CreateExpr(16)), Pow(CreateExpr(2), CreateExpr("new_m")));
//   auto updated_expr3 = UpdateExprValue(max_value, replace_value3);
//   EXPECT_NE(updated_expr3, nullptr);
//   dumps(Str(updated_expr3));
// }

TEST_F(ArgsManagerUtest, test_get_searchable_vars)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  VarInfo var_info2;
  var_info2.do_search = false;
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetSearchableVars();
  EXPECT_EQ(exprs.size(), 1u);
  EXPECT_TRUE(exprs[0] == CreateExpr("m"));
}

TEST_F(ArgsManagerUtest, test_get_searchable_vars2)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  var_info1.scopes.push_back(HardwareDef::L0A);
  VarInfo var_info2;
  var_info2.do_search = false;
  var_info2.scopes.push_back(HardwareDef::L0A);
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetSearchableVars(HardwareDef::L0A);
  EXPECT_EQ(exprs.size(), 1u);
  EXPECT_TRUE(exprs[0] == CreateExpr("m"));
  exprs = args_manager.GetSearchableVars(HardwareDef::L0B);
  EXPECT_EQ(exprs.size(), 0);
}

TEST_F(ArgsManagerUtest, test_get_replaced_vars)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  VarInfo var_info2;
  var_info2.do_search = false;
  var_info2.replacement.orig_expr = CreateExpr("n");
  args_manager.vars_infos_.emplace(CreateExpr("new_m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("new_n"), var_info2);
  auto exprs_map = args_manager.GetVarsRelations();
  EXPECT_EQ(exprs_map.size(), 1u);
  for (auto expr_pair : exprs_map) {
    EXPECT_EQ(Str(expr_pair.first), "new_n");
    EXPECT_EQ(Str(expr_pair.second), "n");
  }
}

TEST_F(ArgsManagerUtest, test_get_replaced_exprs)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.do_search = true;
  VarInfo var_info2;
  var_info2.do_search = false;
  var_info2.replacement.new_replaced_expr = ge::sym::Pow(CreateExpr(2), CreateExpr("new_n"));
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs_map = args_manager.GetExprRelations();
  EXPECT_EQ(exprs_map.size(), 1u);
  for (auto expr_pair : exprs_map) {
    EXPECT_EQ(Str(expr_pair.first), "n");
    EXPECT_TRUE(expr_pair.second == ge::sym::Pow(CreateExpr(2), CreateExpr("new_n")));
  }
}

TEST_F(ArgsManagerUtest, test_get_input_vars)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.is_input_var = true;
  VarInfo var_info2;
  var_info2.is_input_var = false;
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetInputVars();
  EXPECT_EQ(exprs.size(), 1u);
  EXPECT_TRUE(exprs[0] == CreateExpr("m"));
}

TEST_F(ArgsManagerUtest, test_get_const_vars)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.is_const_var = true;
  VarInfo var_info2;
  var_info2.is_const_var = false;
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  args_manager.vars_infos_.emplace(CreateExpr("n"), var_info2);
  auto exprs = args_manager.GetConstVars();
  EXPECT_EQ(exprs.size(), 1u);
  for (auto expr_pair : exprs) {
    EXPECT_EQ(Str(expr_pair.first), "m");
    EXPECT_EQ(expr_pair.second, 0);
  }
}

TEST_F(ArgsManagerUtest, test_get_related_scope)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo var_info1;
  var_info1.scopes = {HardwareDef::L0A};
  args_manager.vars_infos_.emplace(CreateExpr("m"), var_info1);
  auto scopes = args_manager.GetRelatedHardware(CreateExpr("m"));
  EXPECT_EQ(scopes.size(), 1u);
  EXPECT_TRUE(scopes[0] == HardwareDef::L0A);
  scopes = args_manager.GetRelatedHardware(CreateExpr("n"));
  EXPECT_TRUE(scopes.empty());
}

TEST_F(ArgsManagerUtest, test_split_args) {
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

TEST_F(ArgsManagerUtest, test_total_scope)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  auto scopes = args_manager.GetTotalHardwareCons();
  EXPECT_TRUE(scopes.empty());
  args_manager.hardware_cons_[HardwareDef::L0A] = (CreateExpr("m") * CreateExpr(2));
  scopes = args_manager.GetTotalHardwareCons();
  EXPECT_EQ(scopes.size(), 1u);
}

TEST_F(ArgsManagerUtest, test_total_scope_with_replace)
{
  ModelInfo info;
  Expr m = CreateExpr("m");
  info.variable_expr_map[m] = CreateExpr("n") + ge::sym::kSymbolOne;
  ArgsManager args_manager(info);
  auto scopes = args_manager.GetTotalHardwareCons(false);
  EXPECT_TRUE(scopes.empty());
  args_manager.hardware_cons_[HardwareDef::L0A] = (m * CreateExpr(2));
  scopes = args_manager.GetTotalHardwareCons();
  EXPECT_EQ(scopes.size(), 1u);
}

TEST_F(ArgsManagerUtest, test_used_scope)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  auto scopes = args_manager.GetUsedHardwareInfo(HardwareDef::L0A);
  EXPECT_TRUE(!IsValid(scopes));
  args_manager.hardware_cons_[HardwareDef::L0A] = (CreateExpr("m") * CreateExpr(2));
  scopes = args_manager.GetUsedHardwareInfo(HardwareDef::L0A);
  EXPECT_TRUE(IsValid(scopes));
}

TEST_F(ArgsManagerUtest, test_get_obj)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  Expr mac = (CreateExpr("m") / CreateExpr(16));
  Expr mte = (CreateExpr("n") / CreateExpr(256));
  args_manager.objs_.emplace(PipeType::AIC_MAC, mac);
  args_manager.objs_.emplace(PipeType::AIC_MTE2, mte);
  auto obj = args_manager.GetObjectFunc();
  EXPECT_TRUE(obj[PipeType::AIC_MAC] == mac);
  EXPECT_TRUE(obj[PipeType::AIC_MTE2] == mte);
  dumps(Str(obj[PipeType::AIC_MAC]));
  dumps(Str(obj[PipeType::AIC_MTE2]));
}

TEST_F(ArgsManagerUtest, test_get_ancestor)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.orig_axis_size = {CreateExpr("m")};
  args_manager.vars_infos_.emplace(CreateExpr("m"), info1);
  auto m_ori_name = Str(args_manager.GetAncestor(CreateExpr("m"))[0]);
  EXPECT_EQ(m_ori_name, "m");
  auto n_ori_size = args_manager.GetAncestor(CreateExpr("n"));
  EXPECT_TRUE(n_ori_size.empty());
}

TEST_F(ArgsManagerUtest, test_get_max_value)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.max_value = CreateExpr("m");
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  auto m_ori_name = args_manager.GetMaxValue(CreateExpr("tilem"));
  EXPECT_TRUE(m_ori_name == CreateExpr("m"));
  m_ori_name = args_manager.GetMaxValue(CreateExpr("m"));
  EXPECT_TRUE(m_ori_name == 1);
}

TEST_F(ArgsManagerUtest, test_get_min_value)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  auto test = args_manager.GetMaxValue(CreateExpr("test"));
  EXPECT_TRUE(test == 1);
}

TEST_F(ArgsManagerUtest, test_get_init_value)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.init_value = CreateExpr("m");
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  auto m_ori_name = args_manager.GetDefaultInitValue(CreateExpr("tilem"));
  EXPECT_TRUE(m_ori_name == CreateExpr("m"));
  m_ori_name = args_manager.GetDefaultInitValue(CreateExpr("m"));
  EXPECT_TRUE(m_ori_name == ge::sym::kSymbolOne);
}

TEST_F(ArgsManagerUtest, test_get_align_value)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.align = ge::Symbol(16u);
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  auto align_value = args_manager.GetVarAlignValue(CreateExpr("tilem"));
  EXPECT_EQ(align_value, ge::Symbol(16u));
  align_value = args_manager.GetVarAlignValue(CreateExpr("m"));
  EXPECT_EQ(align_value, ge::Symbol(0));
}

TEST_F(ArgsManagerUtest, test_set_solved_vars)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.align = ge::Symbol(16u);
  info1.do_search = true;
  args_manager.vars_infos_.emplace(CreateExpr("tilem"), info1);
  args_manager.vars_infos_.emplace(CreateExpr("basem"), info1);
  std::vector<Expr> solved_vars{CreateExpr("basem")};
  args_manager.SetSolvedVars(solved_vars);
  EXPECT_TRUE(args_manager.solved_vars_.size() == 1u);
  EXPECT_TRUE(args_manager.solved_vars_[0] == CreateExpr("basem"));
  EXPECT_FALSE(args_manager.vars_infos_[CreateExpr("basem")].do_search);
  EXPECT_TRUE(args_manager.vars_infos_[CreateExpr("tilem")].do_search);
  auto vars = args_manager.GetSolvedVars();
  EXPECT_TRUE(vars.size() == 1u);
  EXPECT_TRUE(vars[0] == CreateExpr("basem"));
}

TEST_F(ArgsManagerUtest, test_get_total_cut_cons)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  args_manager.cut_leq_cons_.push_back(CreateExpr("m"));
  auto leq_exprs = args_manager.GetTotalCutCons();
  EXPECT_TRUE(leq_exprs.size() == 1u);
  EXPECT_TRUE(leq_exprs[0] == CreateExpr("m"));
}

TEST_F(ArgsManagerUtest, test_get_parent_var)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  args_manager.vars_infos_.emplace(CreateExpr("m"), info1);
  auto parent_m = args_manager.GetParentVars(CreateExpr("m"));
  EXPECT_TRUE(parent_m.empty());
  auto parent_n = args_manager.GetParentVars(CreateExpr("n"));
  EXPECT_TRUE(parent_n.empty());
  VarInfo info2;
  info2.from_axis_size.emplace_back(CreateExpr("k"));
  args_manager.vars_infos_.emplace(CreateExpr("k"), info2);
  auto parent_k = args_manager.GetParentVars(CreateExpr("k"));
  EXPECT_FALSE(parent_k.empty());
}

TEST_F(ArgsManagerUtest, test_get_innerest_var)
{
  ModelInfo info;
  ArgsManager args_manager(info);
  VarInfo info1;
  info1.is_node_innerest_dim_size = true;
  VarInfo info2;
  info2.is_node_innerest_dim_size = true;
  info2.do_search = true;
  args_manager.vars_infos_.emplace(CreateExpr("m"), info1);
  args_manager.vars_infos_.emplace(CreateExpr("k"), info2);
  auto innerest_vars = args_manager.GetNodeInnerestDimSizes();
  EXPECT_EQ(innerest_vars.size(), 1u);
  EXPECT_TRUE(innerest_vars[0] == CreateExpr("k"));
}

TEST_F(ArgsManagerUtest, test_replace_new_expr)
{
  ExprExprMap new_expr_replacements;
  Expr new_m = CreateExpr("new_m");
  Expr m = CreateExpr("m");
  Expr stepm = CreateExpr("stepm");
  Expr new_stepm = CreateExpr("new_stepm");
  Expr new_m_expr = m / CreateExpr(16);
  Expr new_stepm_expr = ge::sym::Log(CreateExpr(2), (stepm / new_m));
  new_expr_replacements.emplace(new_m, new_m_expr);
  new_expr_replacements.emplace(new_stepm, new_stepm_expr);
  ArgsManager::ReplaceNewExpr(new_expr_replacements);
  EXPECT_FALSE(new_expr_replacements[new_stepm].ContainVar(new_m));
  for (auto expr : new_expr_replacements) {
    dumps(Str(expr.first) + "---------" + Str(expr.second));
  }
}

TEST_F(ArgsManagerUtest, vars_not_find) {
  ModelInfo info;
  ArgsManager args_manager(info);
  auto test = CreateExpr("test");
  EXPECT_TRUE(args_manager.GetAncestor(test).empty());
  EXPECT_TRUE(args_manager.GetMaxValue(test) == 1);
  EXPECT_TRUE(args_manager.GetMinValue(test) == 1);
  EXPECT_TRUE(args_manager.GetDefaultInitValue(test) == 1);
  EXPECT_TRUE(args_manager.GetVarAlignValue(test) == 0u);
  EXPECT_TRUE(args_manager.GetVarPromptAlignValue(test) == 0u);
}

TEST_F(ArgsManagerUtest, IsConcatOuterDim) {
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr undefined_var = CreateExpr("undefined_var");
  ModelInfo info;
  ArgsManager manager(info);
  
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

TEST_F(ArgsManagerUtest, IsConcatInnerDim) {
  Expr var1 = CreateExpr("var1");
  Expr var2 = CreateExpr("var2");
  Expr undefined_var = CreateExpr("undefined_var");

  ModelInfo info;
  ArgsManager manager(info);
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

TEST_F(ArgsManagerUtest, test_small_shape_pattern) {
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

TEST_F(ArgsManagerUtest, test_small_shape_pattern_case2) {
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

TEST_F(ArgsManagerUtest, test_small_shape_pattern_case3) {
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
  hardware_cons[HardwareDef::L1] = CreateExpr(1024) * s1;
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024) * s0;
  model_info.hardware_cons = hardware_cons;
  TilingCodeGenConfig config;
  TilingModelInfo model_infos = {model_info};
  ScoreFuncs score_funcs;
  TilingCodeGenImplPtr impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(ge::MakeShared<AxesReorderTilingCodeGenImpl>(
    "op", config, model_infos, score_funcs, false));
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  EXPECT_FALSE(impl->HitSmallShapePattern(args_manager));
}

TEST_F(ArgsManagerUtest, test_small_shape_pattern_case4) {
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
  hardware_cons[HardwareDef::UB] = CreateExpr(1024);
  hardware_cons[HardwareDef::CORENUM] = CreateExpr(1024);
  model_info.hardware_cons = hardware_cons;
  TilingCodeGenConfig config;
  TilingModelInfo model_infos = {model_info};
  ScoreFuncs score_funcs;
  TilingCodeGenImplPtr impl = std::shared_ptr<AxesReorderTilingCodeGenImpl>(ge::MakeShared<AxesReorderTilingCodeGenImpl>(
    "op", config, model_infos, score_funcs, false));
  ArgsManager args_manager(model_info);
  EXPECT_TRUE(args_manager.Process(true));
  EXPECT_FALSE(impl->HitSmallShapePattern(args_manager));
}