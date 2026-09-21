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
#include "stub/stub_model_info.h"
namespace att {
ModelInfo CreateModelInfo(const ge::ExprType expr_type)
{
  ModelInfo model_info;
  Expr default_expr;
  bool is_const = true;
  if (expr_type == ge::ExprType::kExprConstantRation) {
    default_expr = ge::Symbol(8, "tmp") / ge::Symbol(3, "tmp");
  } else if (expr_type == ge::ExprType::kExprConstantInteger) {
    default_expr = ge::Symbol(2, "tmp");
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
  Optional test_optional;
  test_optional.optional_name = "test";
  test_optional.data_type = "int32_t";
  test_optional.min_value = "1";
  test_optional.max_value = "100";
  model_info.graph_input_infos.optional_atts[1U] = test_optional;
  
  // set n
  Expr expr_n = CreateExpr("n_size");
  Expr expr_tilen = CreateExpr("tilen_size");
  Expr expr_stepn = CreateExpr("stepn_size");
  Expr expr_basen = CreateExpr("basen_size");
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
  Expr core_num = (expr_tilem / expr_stepm) * (expr_tilen / expr_stepn);
  std::map<HardwareDef, Expr> hardware_cons;
  model_info.hardware_cons[HardwareDef::L0A] = l0a_occupy;
  model_info.hardware_cons[HardwareDef::L0B] = l0b_occupy;
  model_info.hardware_cons[HardwareDef::L0C] = l0c_occupy;
  model_info.hardware_cons[HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[HardwareDef::L2] = l2_occupy;
  model_info.hardware_cons[HardwareDef::CORENUM] = core_num;
  model_info.hardware_cons[HardwareDef::UB] = CreateExpr(0L);
  
  Expr mac = (expr_basem * expr_basen * expr_k) / (CreateExpr(16) * CreateExpr(256));
  Expr mte = (((expr_stepm * expr_k) / CreateExpr(32)) + ((expr_stepn * expr_k) / CreateExpr(32)));
  model_info.objects[PipeType::AIC_MAC] = mac;
  model_info.objects[PipeType::AIC_MTE2] = mte;
  model_info.tiling_case_id = 0;
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepm, expr_basem));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepn, expr_basen));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilem - expr_m));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_stepm - expr_tilem));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilen - expr_n));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_stepn - expr_tilen));
  model_info.output_size = 1;

  return model_info;
}

ModelInfo GetMatmulL2TileInfo() {
  ModelInfo model_info;
  Expr expr_corenum = CreateExpr("block_dim");
  SymVarInfoPtr sym_corenum = std::make_shared<SymVarInfo>(expr_corenum);
  AttAxisPtr core = std::make_shared<AttAxis>();
  core->name = "corenum";
  core->axis_pos = AxisPosition::ORIGIN;
  core->bind_multicore = false;
  core->is_last = false;
  core->is_node_innerest_dim = false;
  core->size = sym_corenum;
  model_info.arg_list.emplace_back(core);
  // set m
  Expr expr_m = CreateExpr("m_size");
  Expr expr_tilem = CreateExpr("tilem_size");
  Expr expr_basem = CreateExpr("basem_size");
  SymVarInfoPtr sym_m = std::make_shared<SymVarInfo>(expr_m);
  SymVarInfoPtr sym_tilem = std::make_shared<SymVarInfo>(expr_tilem);
  sym_tilem->align = ge::Symbol(16);
  sym_tilem->related_scope = {HardwareDef::L2};
  SymVarInfoPtr sym_basem = std::make_shared<SymVarInfo>(expr_basem);
  sym_basem->align = ge::Symbol(16);
  sym_basem->related_scope = {HardwareDef::L0A, HardwareDef::L0C, HardwareDef::L1};
  AttAxisPtr m = std::make_shared<AttAxis>();
  AttAxisPtr tilem = std::make_shared<AttAxis>();
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

  basem->name = "basem";
  basem->axis_pos = AxisPosition::INNER;
  basem->bind_multicore = false;
  basem->is_last = true;
  basem->is_node_innerest_dim = false;
  basem->size = sym_basem;
  basem->orig_axis.push_back(m.get());
  basem->from_axis = {tilem.get()};
  model_info.arg_list.emplace_back(m);
  model_info.arg_list.emplace_back(tilem);
  model_info.arg_list.emplace_back(basem);
  
  // set n
  Expr expr_n = CreateExpr("n_size");
  Expr expr_tilen = CreateExpr("tilen_size");
  Expr expr_basen = CreateExpr("basen_size");
  SymVarInfoPtr sym_n = std::make_shared<SymVarInfo>(expr_n);
  SymVarInfoPtr sym_tilen = std::make_shared<SymVarInfo>(expr_tilen);
  sym_tilen->align = ge::Symbol(16);
  sym_tilen->related_scope = {HardwareDef::L2};
  SymVarInfoPtr sym_basen = std::make_shared<SymVarInfo>(expr_basen);
  sym_basen->align = ge::Symbol(16);
  sym_basen->related_scope = {HardwareDef::L0B, HardwareDef::L0C, HardwareDef::L1};
  AttAxisPtr n = std::make_shared<AttAxis>();
  AttAxisPtr tilen = std::make_shared<AttAxis>();
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

  basen->name = "basen";
  basen->axis_pos = AxisPosition::INNER;
  basen->bind_multicore = false;
  basen->is_last = true;
  basen->is_node_innerest_dim = true;
  basen->size = sym_basen;
  basen->orig_axis.push_back(n.get());
  basen->from_axis = {tilen.get()};


  model_info.arg_list.emplace_back(n);
  model_info.arg_list.emplace_back(tilen);
  model_info.arg_list.emplace_back(basen);

  // setk
  Expr expr_k = CreateExpr("k_size");
  Expr expr_stepka = CreateExpr("stepka_size");
  Expr expr_stepkb = CreateExpr("stepkb_size");
  Expr expr_basek = CreateExpr("basek_size");
  SymVarInfoPtr sym_k = std::make_shared<SymVarInfo>(expr_k);
  SymVarInfoPtr sym_stepka = std::make_shared<SymVarInfo>(expr_stepka);
  sym_stepka->align = ge::Symbol(256);
  sym_stepka->related_scope = {HardwareDef::L1};
  SymVarInfoPtr sym_stepkb = std::make_shared<SymVarInfo>(expr_stepkb);
  sym_stepkb->align = ge::Symbol(16);
  sym_stepkb->related_scope = {HardwareDef::L1};
  SymVarInfoPtr sym_basek = std::make_shared<SymVarInfo>(expr_basek);
  sym_basek->align = ge::Symbol(16);
  sym_basek->related_scope = {HardwareDef::L0A, HardwareDef::L0B};
  AttAxisPtr k = std::make_shared<AttAxis>();
  k->name = "k";
  k->axis_pos = AxisPosition::ORIGIN;
  k->bind_multicore = false;
  k->is_last = false;
  k->is_node_innerest_dim = false;
  k->size = sym_k;

  AttAxisPtr stepka = std::make_shared<AttAxis>();
  stepka->name = "stepka";
  stepka->axis_pos = AxisPosition::INNER;
  stepka->bind_multicore = false;
  stepka->is_last = false;
  stepka->is_node_innerest_dim = true;
  stepka->size = sym_stepka;
  stepka->orig_axis.push_back(k.get());
  stepka->from_axis = {k.get()};

  AttAxisPtr stepkb = std::make_shared<AttAxis>();
  stepkb->name = "stepkb";
  stepkb->axis_pos = AxisPosition::INNER;
  stepkb->bind_multicore = false;
  stepkb->is_last = false;
  stepkb->is_node_innerest_dim = true;
  stepkb->size = sym_stepkb;
  stepkb->orig_axis.push_back(k.get());
  stepkb->from_axis = {stepka.get()};

  AttAxisPtr basek = std::make_shared<AttAxis>();
  basek->name = "basek";
  basek->axis_pos = AxisPosition::INNER;
  basek->bind_multicore = false;
  basek->is_last = true;
  basek->is_node_innerest_dim = false;
  basek->size = sym_basek;
  basek->orig_axis.push_back(k.get());
  basek->from_axis = {stepkb.get()};


  model_info.arg_list.emplace_back(k);
  model_info.arg_list.emplace_back(stepka);
  model_info.arg_list.emplace_back(stepkb);
  model_info.arg_list.emplace_back(basek);


  Expr l0a_occupy = expr_basem * expr_basek * CreateExpr(4);
  Expr l0b_occupy = expr_basek * expr_basen * CreateExpr(4);
  Expr l0c_occupy = expr_basem * expr_basen * CreateExpr(4);
  Expr l1_occupy = (expr_stepka * expr_basem * CreateExpr(4)) + (expr_stepkb * expr_basen * CreateExpr(4));
  Expr l2_occupy = (expr_tilen * expr_tilem * CreateExpr(2)) + ((expr_tilen + expr_tilem) * expr_k * CreateExpr(2));
  std::map<HardwareDef, Expr> hardware_cons;
  model_info.hardware_cons[HardwareDef::L0A] = l0a_occupy;
  model_info.hardware_cons[HardwareDef::L0B] = l0b_occupy;
  model_info.hardware_cons[HardwareDef::L0C] = l0c_occupy;
  model_info.hardware_cons[HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[HardwareDef::L2] = l2_occupy;
  model_info.hardware_cons[HardwareDef::UB] = CreateExpr(0L);
  
  Expr tile_cnt = ((expr_n / expr_tilen) * (expr_m / expr_tilem));
  Expr base_cnt = ge::sym::Max(ge::sym::kSymbolOne, (((expr_tilem * expr_tilen) / (expr_basem * expr_basen)) / CreateExpr("block_dim")));
  Expr al1_cnt = (expr_k / expr_stepka);
  Expr bl1_cnt = (expr_stepka / expr_stepkb);
  Expr l0_cnt = (expr_stepkb / expr_basek);
  Expr l1_cnt = (al1_cnt * bl1_cnt);
  Expr base_fixpipe_cost = ((expr_basem * expr_basen * CreateExpr(4)) / CreateExpr(32));
  Expr al1_mte2 = (((expr_basem * expr_stepka * CreateExpr(2)) / (CreateExpr(32) / ge::sym::Max(ge::sym::kSymbolOne, (CreateExpr(256) / expr_stepka)))) + CreateExpr(210));
  std::cout << "AL1 mte2: " << al1_mte2 << std::endl;
  Expr bl1_mte2 = (((expr_basen * expr_stepkb * CreateExpr(2)) / (CreateExpr(32) / ge::sym::Max(ge::sym::kSymbolOne, (CreateExpr(256) / expr_basen)))) + CreateExpr(210));
  std::cout << "BL1 mte2: " << bl1_mte2 << std::endl;
  Expr mac = (((tile_cnt * base_cnt * l1_cnt * l0_cnt)) * (expr_basem * expr_basen * expr_k) / (CreateExpr(16) * CreateExpr(256)));
  Expr mte2 = (tile_cnt * base_cnt * al1_cnt * (al1_mte2 + (bl1_cnt * bl1_mte2)));
  std::cout << "mte2: " << mte2 << std::endl;
  Expr fixpipe = (tile_cnt * base_cnt * base_fixpipe_cost);
  model_info.objects[PipeType::AIC_MAC] = mac;
  model_info.objects[PipeType::AIC_MTE2] = mte2;
  model_info.objects[PipeType::AIC_FIXPIPE] = fixpipe;
  model_info.tiling_case_id = 1;
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepka, expr_stepkb));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_tilen, expr_basen));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_tilem, expr_basem));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepkb, expr_basek));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilem - expr_m));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilen - expr_n));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_stepka - expr_k));
  model_info.container_exprs["Q1"] = (expr_m + expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (expr_m + expr_n);
  model_info.output_size = 1;

  return model_info;
}

ModelInfo CreateCeilingModel()
{
  ModelInfo model_info;
  Expr expr_s1 = CreateExpr("s1_size");
  Expr expr_s2 = CreateExpr("s2_size");
  Expr expr_s2t = CreateExpr("s2t_size");
  Expr expr_s2T = ge::sym::Ceiling(expr_s2 / expr_s2t);
  Expr expr_s1s2T = (expr_s1 * expr_s2T);
  Expr expr_s1s2Tb = CreateExpr("s1s2Tb_size");
  SymVarInfoPtr sym_s1 = std::make_shared<SymVarInfo>(expr_s1);
  SymVarInfoPtr sym_s2 = std::make_shared<SymVarInfo>(expr_s2);
  SymVarInfoPtr sym_s2t = std::make_shared<SymVarInfo>(expr_s2t);
  SymVarInfoPtr sym_s1s2Tb = std::make_shared<SymVarInfo>(expr_s1s2Tb);
  
  AttAxisPtr s1 = std::make_shared<AttAxis>();
  s1->name = "s1";
  s1->axis_pos = AxisPosition::ORIGIN;
  s1->bind_multicore = false;
  s1->is_last = false;
  s1->is_node_innerest_dim = false;
  s1->size = sym_s1;
  
  AttAxisPtr s2 = std::make_shared<AttAxis>();
  s2->name = "s2";
  s2->axis_pos = AxisPosition::ORIGIN;
  s2->bind_multicore = false;
  s2->is_last = false;
  s2->is_node_innerest_dim = false;
  s2->size = sym_s2;

  AttAxisPtr s2t = std::make_shared<AttAxis>();
  s2t->name = "s2t";
  s2t->axis_pos = AxisPosition::INNER;
  s2t->bind_multicore = false;
  s2t->is_last = false;
  s2t->is_node_innerest_dim = false;
  s2t->size = sym_s2t;
  s2t->orig_axis.push_back(s2.get());
  s2t->from_axis = {s2.get()};

  AttAxisPtr s1s2Tb = std::make_shared<AttAxis>();
  s1s2Tb->name = "s1s2Tb";
  s1s2Tb->axis_pos = AxisPosition::INNER;
  s1s2Tb->bind_multicore = false;
  s1s2Tb->is_last = false;
  s1s2Tb->is_node_innerest_dim = true;
  s1s2Tb->size = sym_s1s2Tb;
  s1s2Tb->orig_axis.push_back(s1.get());
  s1s2Tb->orig_axis.push_back(s2.get());
  s1s2Tb->from_axis = {s1.get()};

  model_info.arg_list.emplace_back(s1);
  model_info.arg_list.emplace_back(s2);
  model_info.arg_list.emplace_back(s2t);
  model_info.arg_list.emplace_back(s1s2Tb);

  Expr core_num = ge::sym::Ceiling((expr_s1 * ge::sym::Ceiling(expr_s2 / expr_s2t)) / expr_s1s2Tb);
  std::map<HardwareDef, Expr> hardware_cons;
  model_info.hardware_cons[HardwareDef::UB] = expr_s1 * CreateExpr(10);
  model_info.hardware_cons[HardwareDef::CORENUM] = core_num;

  model_info.output_size = 1;
  model_info.tiling_case_id = 0;
  return model_info;
}
}  // namespace att
