/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include "ffn/stub_modelinfo.h"
namespace att {
ModelInfo GenFFNModelInfo() {
  ModelInfo model_info;

  Expr expr_maxTokens = CreateExpr("maxTokens");
  Expr expr_basem1 = CreateExpr("base_m1");
  Expr expr_basem2 = CreateExpr("base_m2");
  Expr expr_ubm = CreateExpr("ub_m");
  SymVarInfoPtr sym_maxTokens = std::make_shared<SymVarInfo>(expr_maxTokens);
  SymVarInfoPtr sym_basem1 = std::make_shared<SymVarInfo>(expr_basem1);
  sym_basem1->align = ge::Symbol(8);
  sym_basem1->related_scope = {HardwareDef::L0C};
  SymVarInfoPtr sym_ubm = std::make_shared<SymVarInfo>(expr_ubm);
  sym_ubm->align = ge::Symbol(8);
  sym_ubm->related_scope = {HardwareDef::UB};
  SymVarInfoPtr sym_basem2 = std::make_shared<SymVarInfo>(expr_basem2);
  sym_basem2->align = ge::Symbol(8);
  sym_basem2->related_scope = {HardwareDef::L0C};
  AttAxisPtr maxTokens = std::make_shared<AttAxis>();
  AttAxisPtr basem1 = std::make_shared<AttAxis>();
  AttAxisPtr ubm = std::make_shared<AttAxis>();
  AttAxisPtr basem2 = std::make_shared<AttAxis>();

  maxTokens->name = "maxTokens";
  maxTokens->axis_pos = AxisPosition::ORIGIN;
  maxTokens->bind_multicore = false;
  maxTokens->is_last = false;
  maxTokens->is_node_innerest_dim = false;
  maxTokens->size = sym_maxTokens;
  
  basem1->name = "base_m1";
  basem1->axis_pos = AxisPosition::INNER;
  basem1->bind_multicore = false;
  basem1->is_last = false;
  basem1->is_node_innerest_dim = false;
  basem1->size = sym_basem1;
  basem1->orig_axis.push_back(maxTokens.get());
  basem1->from_axis = {maxTokens.get()};
  
  ubm->name = "ub_m";
  ubm->axis_pos = AxisPosition::INNER;
  ubm->bind_multicore = false;
  ubm->is_last = true;
  ubm->is_node_innerest_dim = false;
  ubm->size = sym_ubm;
  ubm->orig_axis.push_back(maxTokens.get());
  ubm->from_axis = {basem1.get()};
  
  basem2->name = "base_m2";
  basem2->axis_pos = AxisPosition::INNER;
  basem2->bind_multicore = false;
  basem2->is_last = true;
  basem2->is_node_innerest_dim = false;
  basem2->size = sym_basem2;
  basem2->orig_axis.push_back(maxTokens.get());
  basem2->from_axis = {maxTokens.get()};

  Expr expr_n1 = CreateExpr("N1");
  Expr expr_basen1 = CreateExpr("base_n1");
  SymVarInfoPtr sym_n1 = std::make_shared<SymVarInfo>(expr_n1);
  SymVarInfoPtr sym_basen1 = std::make_shared<SymVarInfo>(expr_basen1);
  sym_basen1->align = ge::Symbol(8);
  sym_basen1->related_scope = {HardwareDef::L0C, HardwareDef::UB, HardwareDef::BTBUF};
  AttAxisPtr n1 = std::make_shared<AttAxis>();
  AttAxisPtr basen1 = std::make_shared<AttAxis>();

  n1->name = "N1";
  n1->axis_pos = AxisPosition::ORIGIN;
  n1->bind_multicore = false;
  n1->is_last = false;
  n1->is_node_innerest_dim = false;
  n1->size = sym_n1;
  
  basen1->name = "base_n1";
  basen1->axis_pos = AxisPosition::INNER;
  basen1->bind_multicore = false;
  basen1->is_last = true;
  basen1->is_node_innerest_dim = true;
  basen1->size = sym_basen1;
  basen1->orig_axis.push_back(n1.get());
  basen1->from_axis = {n1.get()};

  Expr expr_k1 = CreateExpr("K1");
  SymVarInfoPtr sym_k1 = std::make_shared<SymVarInfo>(expr_k1);
  AttAxisPtr k1 = std::make_shared<AttAxis>();
  k1->name = "K1";
  k1->axis_pos = AxisPosition::ORIGIN;
  k1->bind_multicore = false;
  k1->is_last = false;
  k1->is_node_innerest_dim = false;
  k1->size = sym_k1;


  Expr expr_n2 = CreateExpr("N2");
  Expr expr_basen2 = CreateExpr("base_n2");
  SymVarInfoPtr sym_n2 = std::make_shared<SymVarInfo>(expr_n2);
  SymVarInfoPtr sym_basen2 = std::make_shared<SymVarInfo>(expr_basen2);
  sym_basen2->align = ge::Symbol(8);
  sym_basen2->related_scope = {HardwareDef::L0C, HardwareDef::BTBUF};
  AttAxisPtr n2 = std::make_shared<AttAxis>();
  AttAxisPtr basen2 = std::make_shared<AttAxis>();
  n2->name = "N2";
  n2->axis_pos = AxisPosition::ORIGIN;
  n2->bind_multicore = false;
  n2->is_last = false;
  n2->is_node_innerest_dim = false;
  n2->size = sym_n2;

  basen2->name = "base_n2";
  basen2->axis_pos = AxisPosition::INNER;
  basen2->bind_multicore = false;
  basen2->is_last = true;
  basen2->is_node_innerest_dim = true;
  basen2->size = sym_basen2;
  basen2->orig_axis.push_back(n2.get());
  basen2->from_axis = {n2.get()};

  Expr btbuf_occupy = ge::sym::Max((CreateExpr(4) * expr_basen1), (CreateExpr(4) * expr_basen2));
  Expr l0c_occupy = ge::sym::Max((CreateExpr(4) * expr_basen1 * expr_basem1), (CreateExpr(4) * expr_basen2 * expr_basem2));
  Expr ub_occupy = (CreateExpr(4) * expr_basen1 * expr_ubm);
  std::map<HardwareDef, Expr> hardware_cons;
  model_info.hardware_cons[HardwareDef::BTBUF] = btbuf_occupy;
  model_info.hardware_cons[HardwareDef::L0C] = l0c_occupy;
  model_info.hardware_cons[HardwareDef::UB] = ub_occupy;
  
  Expr m1_cnt = ge::sym::Ceiling(expr_maxTokens / expr_basem1);
  Expr m2_cnt = ge::sym::Ceiling(expr_maxTokens / expr_basem2);
  Expr n1_cnt = ge::sym::Ceiling(expr_n1 / expr_basen1);
  Expr n2_cnt = ge::sym::Ceiling(expr_n2 / expr_basen2);
  Expr ubm_cnt = ge::sym::Ceiling(expr_basem1 / expr_ubm);

  Expr vec_ub = ((CreateExpr(4) * expr_basen1 * expr_ubm) / (CreateExpr(-1) + expr_basen1) + CreateExpr(4));
  Expr vec_m1n1 = ((CreateExpr(8) * expr_basem1 * expr_basen1) / (CreateExpr(-1) + expr_basen1) + CreateExpr(4));
  Expr vec_m2n2 = ((CreateExpr(8) * expr_basem2 * expr_basen2) / (CreateExpr(-1) + expr_basen2) + CreateExpr(4));
  Expr vec = (vec_ub * (m1_cnt * n1_cnt * ubm_cnt)) + (vec_m1n1 * (m1_cnt * n1_cnt)) + (vec_m2n2 * (m2_cnt * n2_cnt));

  Expr mte3_ub = ((CreateExpr(0.01741f) * expr_basen1 * expr_ubm) + CreateExpr(0.22f));
  Expr v_mte3 = mte3_ub * (m1_cnt * n1_cnt * ubm_cnt);

  Expr mte2_n1 = ((CreateExpr(5.01f) / (CreateExpr(27240.69f) + expr_basen1)) + CreateExpr(1051.66f)) * (expr_basen1 / CreateExpr(30421.24f));
  Expr mte2_n2 = ((CreateExpr(5.01f) / (CreateExpr(27240.69f) + expr_basen2)) + CreateExpr(1051.66f)) * (expr_basen2 / CreateExpr(30421.24f));
  Expr mte2_ub = (CreateExpr(0.007f) * expr_basen1 * expr_ubm) + CreateExpr(7.97f);
  Expr v_mte2 = mte2_n1 * n1_cnt + mte2_n2 * n2_cnt + mte2_ub * (m1_cnt * n1_cnt * ubm_cnt);

  Expr expr_m1n1 = ((((CreateExpr(0.05624f) * expr_basem1) + CreateExpr(0.3984f)) * CreateExpr(6.2712e-05f) * expr_k1 * expr_basen1) + (CreateExpr(0.0008295f) * expr_k1 * expr_basen1));
  Expr weight_m1n1 = ((CreateExpr(0.05761f) * expr_basen1) + CreateExpr(0.0f));
  Expr mte2_m1n1 = expr_m1n1  * weight_m1n1;
  Expr expr_n1m1 = ((((CreateExpr(0.05940f) * expr_basen1) + CreateExpr(20.0944f)) * CreateExpr(6.2712e-05f) * expr_k1 * expr_basem1) + (CreateExpr(0.0008295f) * expr_k1 * expr_basem1));
  Expr weight_n1m1 = ((CreateExpr(0.07543f) * expr_k1) + CreateExpr(0.0f));
  Expr mte2_n1m1 = expr_n1m1  * weight_n1m1;
  Expr weight1_cube1 = (CreateExpr(0.000216f) * expr_basen1) + (CreateExpr(0.0003614f) * expr_basem1) + (CreateExpr(0.0005757f) * expr_k1);
  Expr weight2_cube1 = (CreateExpr(0.0f) * expr_k1 * expr_basem1) + (CreateExpr(0.0f) * expr_basem1 * expr_basen1) + (CreateExpr(0.0f) * expr_k1 * expr_basen1);
  Expr mte2_cube1 = (mte2_m1n1 + mte2_n1m1) * (n1_cnt * m1_cnt) / (weight1_cube1 + weight2_cube1);

  Expr expr_m2n2 = ((((CreateExpr(0.05624f) * expr_basem2) + CreateExpr(0.3984f)) * CreateExpr(6.2712e-05f) * expr_n1 * expr_basen2) + (CreateExpr(0.0008295f) * expr_n1 * expr_basen2));
  Expr weight_m2n2 = ((CreateExpr(0.05761f) * expr_basen2) + CreateExpr(0.0f));
  Expr mte2_m2n2 = expr_m2n2  * weight_m2n2;
  Expr expr_n2m2 = ((((CreateExpr(0.05940f) * expr_basen2) + CreateExpr(20.0944f)) * CreateExpr(6.2712e-05f) * expr_n1 * expr_basem2) + (CreateExpr(0.0008295f) * expr_n1 * expr_basem2));
  Expr weight_n2m2 = ((CreateExpr(0.07543f) * expr_n1) + CreateExpr(0.0f));
  Expr mte2_n2m2 = expr_n2m2  * weight_n2m2;
  Expr weight1_cube2 = (CreateExpr(0.000216f) * expr_basen2) + (CreateExpr(0.0003614f) * expr_basem2) + (CreateExpr(0.0005757f) * expr_n1);
  Expr weight2_cube2 = (CreateExpr(0.0f) * expr_n1 * expr_basem2) + (CreateExpr(0.0f) * expr_basem2 * expr_basen2) + (CreateExpr(0.0f) * expr_n1 * expr_basen2);
  Expr mte2_cube2 = (mte2_m2n2 + mte2_n2m2) * (n2_cnt * m2_cnt) / (weight1_cube2 + weight2_cube2);
  Expr mte2 = mte2_cube1 + mte2_cube2;

  model_info.objects[PipeType::AIV_MTE2] = v_mte2;
  model_info.objects[PipeType::AIV_MTE3] = v_mte3;
  model_info.objects[PipeType::AIC_MTE2] = mte2;
  model_info.objects[PipeType::AIV_VEC] = vec;
  model_info.tiling_case_id = 0;
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_basem1, expr_ubm));
  model_info.output_size = 1;
  
  model_info.arg_list.emplace_back(maxTokens);
  model_info.arg_list.emplace_back(basen1);
  model_info.arg_list.emplace_back(basen2);
  model_info.arg_list.emplace_back(n1);
  model_info.arg_list.emplace_back(basem1);
  model_info.arg_list.emplace_back(k1);
  model_info.arg_list.emplace_back(n2);
  model_info.arg_list.emplace_back(basem2);
  model_info.arg_list.emplace_back(ubm);
  return model_info;
}
}  // namespace att
