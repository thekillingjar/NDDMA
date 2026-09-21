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
#include "matmul/stub_modelinfo.h"
namespace att {
ModelInfo GenMatmulModelInfo() {
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

  Expr l0a_occupy =  expr_basem * expr_basek * CreateExpr(4);
  Expr l0b_occupy =  expr_basek * expr_basen * CreateExpr(4);
  Expr l0c_occupy =  expr_basem * expr_basen * CreateExpr(4);
  Expr l1_occupy =  (expr_stepka * expr_basem * CreateExpr(4)) + (expr_stepkb * expr_basen * CreateExpr(4));
  Expr l2_occupy =  (expr_tilen * expr_tilem * CreateExpr(2)) + ((expr_tilen + expr_tilem) * expr_k * CreateExpr(2));
  std::map<HardwareDef, Expr> hardware_cons;
  model_info.hardware_cons[HardwareDef::L0A] = l0a_occupy;
  model_info.hardware_cons[HardwareDef::L0B] = l0b_occupy;
  model_info.hardware_cons[HardwareDef::L0C] = l0c_occupy;
  model_info.hardware_cons[HardwareDef::L1] = l1_occupy;
  model_info.hardware_cons[HardwareDef::L2] = l2_occupy;
  model_info.hardware_cons[HardwareDef::UB] = CreateExpr(0L);
  
  Expr tile_cnt = ((expr_n / expr_tilen) * (expr_m / expr_tilem));
  Expr base_cnt = ge::sym::Max(ge::sym::kSymbolOne, (((expr_tilem / expr_basem) * (expr_tilen / expr_basen)) / expr_corenum));
  Expr al1_cnt = expr_k / expr_stepka;
  Expr bl1_cnt = expr_stepka / expr_stepkb;
  Expr l1_cnt = al1_cnt * bl1_cnt;
  Expr l0_cnt = expr_stepkb / expr_basek;

  Expr al0_mte1 = (((expr_basem * expr_basek) * CreateExpr(2)) / CreateExpr(512)) + CreateExpr(26);
  Expr bl0_mte1 = (((expr_basek * expr_basen) * CreateExpr(2)) / CreateExpr(256)) + CreateExpr(26);
  Expr l0_mte1 = al0_mte1 + bl0_mte1;
  Expr mte1 = (tile_cnt * base_cnt * l1_cnt * l0_cnt) * l0_mte1;
  std::cout << "mte1: " << mte1 << std::endl;
  
  Expr l0_mac = ge::sym::Ceiling(expr_basem / CreateExpr(16)) * ge::sym::Ceiling(expr_basek / CreateExpr(16)) * ge::sym::Ceiling(expr_basen / CreateExpr(16));
  Expr mac = (tile_cnt * base_cnt * l1_cnt * l0_cnt) * l0_mac;
  std::cout << "mac: " << mac << std::endl;
  
  Expr al1_mte2 = (((expr_basem * expr_stepka) * CreateExpr(2)) / (CreateExpr(32) / ge::sym::Max(ge::sym::kSymbolOne, (CreateExpr(256) / expr_stepka)))) + CreateExpr(210);
  Expr bl1_mte2 = (((expr_stepkb * expr_basen) * CreateExpr(2)) / (CreateExpr(32) / ge::sym::Max(ge::sym::kSymbolOne, (CreateExpr(256) / expr_basen)))) + CreateExpr(210);
  Expr mte2 = tile_cnt * base_cnt * al1_cnt * (al1_mte2 + (bl1_cnt * bl1_mte2));
  std::cout << "mte2: " << mte2 << std::endl;
  
  Expr base_fixpipe = ((expr_basem * expr_basen) * CreateExpr(4)) / CreateExpr(32);
  Expr fixpipe = (tile_cnt * base_cnt) * base_fixpipe;

  model_info.objects[PipeType::AIC_MAC] = mac;
  model_info.objects[PipeType::AIC_MTE1] = mte1;
  model_info.objects[PipeType::AIC_MTE2] = mte2;
  model_info.objects[PipeType::AIC_FIXPIPE] = fixpipe;
  model_info.tiling_case_id = 1;
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepka, expr_stepkb));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_stepkb, expr_basek));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_tilen, expr_basen));
  model_info.eq_exprs[kFatherToChildNoTail].push_back(std::pair(expr_tilem, expr_basem));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilem - expr_m));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_tilen - expr_n));
  model_info.leq_exprs[kFatherToChildLarger].push_back((expr_stepka - expr_k));
  model_info.container_exprs["Q1"] = (expr_m + expr_n);
  model_info.tensor_exprs["MATMUL_OUTPUT1"] = (expr_m + expr_n);
  model_info.output_size = 1;

  return model_info;
}
}  // namespace att
