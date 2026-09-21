/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "solver_pass_manager/stub_model_info.h"
namespace att {
ModelInfo CreateModelInfo(const uint32_t m_align, const ge::ExprType expr_type)
{
  ModelInfo model_info;
  Expr default_expr;
  bool is_const = true;
  if (expr_type == ge::ExprType::kExprConstantRation) {
    default_expr = ge::Symbol(8, "tmp") / ge::Symbol(3, "tmp");
  } else if (expr_type == ge::ExprType::kExprConstantInteger) {
    default_expr = ge::Symbol(16, "tmp");
  } else if (expr_type == ge::ExprType::kExprVariable) {
    is_const = false;
  }
  // set m
  Expr expr_m = is_const ? default_expr : CreateExpr("m_size");
  Expr expr_tilem = is_const ? default_expr : CreateExpr("tilem_size");
  Expr expr_stepm = is_const ? default_expr : CreateExpr("stepm_size");
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
  sym_m->value_range.first = 1;
  sym_m->value_range.second = 10000;
  sym_m->align = ge::Symbol(m_align);
  m->size = sym_m;
  std::map<uint32_t, std::vector<int64_t>> map_info;
  map_info[0] = {1, 2};
  map_info[1] = {INT64_MAX};
  map_info[2] = {3};
  map_info[3] = {4};
  
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
  Optional test_optional;
  test_optional.optional_name = "test";
  test_optional.data_type = "int32_t";
  test_optional.min_value = "1";
  test_optional.max_value = "100";
  InputTensor input_tensor;
  input_tensor.data_type = 1;

  // set n
  Expr expr_n = is_const ? default_expr : CreateExpr("n_size");
  Expr expr_tilen = is_const ? default_expr : CreateExpr("tilen_size");
  Expr expr_stepn = is_const ? default_expr : CreateExpr("stepn_size");
  Expr expr_basen = is_const ? default_expr : CreateExpr("basen_size");
  SymVarInfoPtr sym_n = std::make_shared<SymVarInfo>(expr_n);
  SymVarInfoPtr sym_tilen = std::make_shared<SymVarInfo>(expr_tilen);
  sym_tilen->align = ge::Symbol(16);
  sym_tilen->related_scope = {HardwareDef::L2};
  SymVarInfoPtr sym_stepn = std::make_shared<SymVarInfo>(expr_stepn);
  sym_stepn->align = ge::Symbol(128);
  sym_stepn->related_scope = {HardwareDef::L1, HardwareDef::CORENUM};
  SymVarInfoPtr sym_basen = std::make_shared<SymVarInfo>(expr_basen);
  sym_basen->align = ge::Symbol(16);
  sym_basen->related_scope = {HardwareDef::L0B, HardwareDef::L0C};
  AttAxisPtr n = std::make_shared<AttAxis>();
  AttAxisPtr tilen = std::make_shared<AttAxis>();
  AttAxisPtr stepn = std::make_shared<AttAxis>();
  AttAxisPtr basen = std::make_shared<AttAxis>();
  n->name = "n";
  n->axis_pos = AxisPosition::ORIGIN;
  n->bind_multicore = false;
  n->is_last = false;
  n->is_node_innerest_dim = false;
  n->size = sym_n;
  
  tilen->name = "tilen";
  tilen->axis_pos = AxisPosition::INNER;
  tilen->bind_multicore = false;
  tilen->is_last = false;
  tilen->is_node_innerest_dim = true;
  tilen->size = sym_tilen;
  tilen->orig_axis.push_back(n.get());
  tilen->from_axis = {n.get()};

  stepn->name = "stepn";
  stepn->axis_pos = AxisPosition::INNER;
  stepn->bind_multicore = true;
  stepn->is_last = false;
  stepn->is_node_innerest_dim = true;
  stepn->size = sym_stepn;
  stepn->orig_axis.push_back(n.get());
  stepn->from_axis = {tilen.get()};

  basen->name = "basen";
  basen->axis_pos = AxisPosition::INNER;
  basen->bind_multicore = false;
  basen->is_last = true;
  basen->is_node_innerest_dim = true;
  basen->size = sym_basen;
  basen->orig_axis.push_back(n.get());
  basen->from_axis = {stepn.get()};


  model_info.arg_list.emplace_back(n);
  model_info.arg_list.emplace_back(tilen);
  model_info.arg_list.emplace_back(stepn);
  model_info.arg_list.emplace_back(basen);

  // setk
  Expr expr_k = CreateExpr("k_size");
  SymConstInfoPtr sym_k = std::make_shared<SymConstInfo>(expr_k);
  sym_k->const_value = 128u;
  AttAxisPtr k = std::make_shared<AttAxis>();
  k->name = "k";
  k->axis_pos = AxisPosition::ORIGIN;
  k->bind_multicore = false;
  k->is_last = false;
  k->is_node_innerest_dim = false;
  k->size = sym_k;
  model_info.arg_list.emplace_back(k);


  Expr l0a_occupy = expr_basem * expr_k * CreateExpr(4);
  Expr l0b_occupy = expr_k * expr_basen * CreateExpr(4);
  Expr l0c_occupy = expr_basem * expr_basen * CreateExpr(4);
  Expr l1_occupy = (expr_k * expr_stepm * CreateExpr(4)) + (expr_k * expr_stepn * CreateExpr(4));
  Expr l2_occupy = (expr_tilen * expr_tilem * CreateExpr(2)) + ((expr_tilen + expr_tilem) * expr_k * CreateExpr(2));
  Expr core_num = ((expr_tilem / expr_stepm) * (expr_tilen / expr_stepn));
  std::map<HardwareDef, Expr> hardware_cons;
  model_info.hardware_cons[HardwareDef::L0A] = l0a_occupy;
  model_info.hardware_cons[HardwareDef::L0B] = l0b_occupy;
  model_info.hardware_cons[HardwareDef::L0C] = l0c_occupy;
  model_info.hardware_cons[HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[HardwareDef::L2] = l2_occupy;
  model_info.hardware_cons[HardwareDef::UB] = expr_m * CreateExpr(10);
  model_info.hardware_cons[HardwareDef::CORENUM] = core_num;
  
  Expr mac = ((expr_basem * expr_basen * expr_k) / (CreateExpr(16) * CreateExpr(256)));
  Expr mte = (((expr_stepm * expr_k) / CreateExpr(32)) + ((expr_stepn * expr_k) / CreateExpr(32)));
  model_info.objects[PipeType::AIC_MAC] = mac;
  model_info.objects[PipeType::AIC_MTE2] = mte;
  model_info.tiling_case_id = 0;
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepm, expr_basem));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepn, expr_basen));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilem - expr_stepm));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilen - expr_stepn));
  model_info.container_exprs["Q1"] = (expr_m + expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (expr_m + expr_n);
  model_info.output_size = 1;
  return model_info;
}
}  // namespace att
