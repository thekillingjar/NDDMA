/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "custom_ascend_graph.h"
#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "ascir_ops.h"
using namespace ge::ascir_op;
/********************************************************************
 * AscendIR原图连接关系表达
 * for bngs1TB in range(ceiling((ceiling((B * G * N * ceiling((S1 / (s1t_size))))) / (bngs1Tb_size))))
 * for bngs1Tb in range(bngs1Tb_size)
 * for s2T in range(ceiling((S2 / (s2t_size))))
 * query_output_0 [s1t_size,D]  = query()
 * key_output_0 [D,s2t_size]  = key()
 * bmm1_output_0 [s1t_size,s2t_size] : Q0_hbm_size = bmm1(query_output_0, key_output_0)
 * for s1tT in range(ceiling((s1t_size / (s1tt_size))))
 * load1_output_0 [s1tt_size,s2t_size] : Q1_ub_size = load1(bmm1_output_0)
 * loadPse_output_0 [s1tt_size,s2t_size] : BUF0_ub_size = loadPse(pse)
 * castPse_output_0 [s1tt_size,s2t_size] : BUF1_ub_size = castPse(loadPse_output_0)
 * add1_output_0 [s1tt_size,s2t_size] : Q1_ub_size = add1(load1_output_0, castPse_output_0)
 * mul1_output_0 [s1tt_size,s2t_size] : Q1_ub_size = mul1(add1_output_0, scaleValue)
 * loadAttenMask_output_0 [s1tt_size,s2t_size] : BUF2_ub_size = loadAttenMask(attenMask)
 * select_output_0 [s1tt_size,s2t_size] : Q1_ub_size = select(mul1_output_0, loadAttenMask_output_0)
 * softmaxExp_output_0 [s1tt_size,BL] : Q3_ub_size = softmaxExp()
 * softmaxApiTmpBuf_output_0 [s1tt_size,s2t_size] : BUF1_ub_size = softmaxApiTmpBuf()
 * flashSoftmax_output_0 [s1tt_size,s2t_size] : Q1_ub_size, flashSoftmax_output_1 [s1tt_size,BL] : BUF4_ub_size, flashSoftmax_output_2 [s1tt_size,BL] : Q2_ub_size = flashSoftmax(select_output_0, softmaxExp_output_0, softmaxApiTmpBuf_output_0)
 * storeSoftmaxMax_output_0 [s1tt_size,BL]  = storeSoftmaxMax(flashSoftmax_output_2)
 * loadDropMask_output_0 [s1tt_size,s2t_size] : BUF3_ub_size = loadDropMask(dropMask)
 * dropout_output_0 [s1tt_size,s2t_size] : Q1_ub_size = dropout(flashSoftmax_output_0, loadDropMask_output_0)
 * castVec1Res_output_0 [s1tt_size,s2t_size] : BUF0_ub_size = castVec1Res(dropout_output_0)
 * storeVec1Res_output_0 [s1tt_size,s2t_size] : Q4_hbm_size = storeVec1Res(castVec1Res_output_0)
 * value_output_0 [s2t_size,D]  = value()
 * bmm2_output_0 [s1t_size,D] : Q5_hbm_size = bmm2(storeVec1Res_output_0, value_output_0)
 * for s1tT2 in range(ceiling((s1t_size / (s1tt2_size))))
 * load2_output_0 [s1tt2_size,D] : BUF1_ub_size = load2(bmm2_output_0)
 * addResOut_output_0 [s1tt2_size,D] : Q6_hbm_size = addResOut()
 * loadAddResOut_output_0 [s1tt2_size,D] : BUF5_ub_size = loadAddResOut(addResOut_output_0)
 * mulRes_output_0 [s1tt2_size,D] : BUF5_ub_size = mulRes(loadAddResOut_output_0, softmaxExp_output_0)
 * addRes_output_0 [s1tt2_size,D] : BUF5_ub_size = addRes(load2_output_0, mulRes_output_0)
 * div_output_0 [s1tt2_size,D] : BUF5_ub_size = div(addRes_output_0, flashSoftmax_output_2)
 * castBmm2Res_output_0 [s1tt2_size,D] : BUF5_ub_size = castBmm2Res(div_output_0)
 * store_output_0 [s1tt2_size,D]  = store(castBmm2Res_output_0)
 *************************************************/
// TODO to be modify to new ascir ir
void BuildOriginGraph(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto S1 = ge::Symbol("S1");
  auto S2 = ge::Symbol("S2");
  auto D = ge::Symbol("D");
  auto BL = ge::Symbol(8, "BL");

  auto b = graph.CreateAxis("b", B, 1, 100000);
  auto n = graph.CreateAxis("n", N, 1, 100000);
  auto g = graph.CreateAxis("g", G, 1, 100000);
  auto s1 = graph.CreateAxis("s1", S1, 1, 100000);
  auto s2 = graph.CreateAxis("s2", S2, 1, 100000);
  auto d = graph.CreateAxis("d", D, 1, 100000);
  auto bl = graph.CreateAxis("l", BL, 1, 100000);
  graph.CreateOptionalAtt("head_num", 1, 1, 10);

  auto bmm1ResAxis = {b.id, n.id, g.id, s1.id, s2.id, d.id, bl.id};
  std::initializer_list<ge::Expression> bmm1ResRepeat = {B, N, G, S1, S2, ONE, ONE};
  std::initializer_list<ge::Expression> bmm1ResStride = {N*G*S1*S2, G*S1*S2, S1*S2, S2, ONE, ZERO, ZERO};

  std::initializer_list<ge::Expression> vec1ResRepeat = {B, N, G, S1, S2, ONE, ONE};
  std::initializer_list<ge::Expression> vec1ResStride = {N*G*S1*S2, G*S1*S2, S1*S2, S2, ONE, ZERO, ZERO};

  auto bmm2ResAxis = {b.id, n.id, g.id, s1.id, s2.id, d.id, bl.id};
  std::initializer_list<ge::Expression> bmm2ResRepeat = {B, N, G, S1, ONE, D, ONE};
  std::initializer_list<ge::Expression> bmm2ResStride = {N*G*S1*D, G*S1*D, S1*D, D, ZERO, ONE, ZERO};

  std::initializer_list<ge::Expression> vec2ResRepeat = {B, N, G, S1, ONE, D, ONE};
  std::initializer_list<ge::Expression> vec2ResStride = {N*G*S1*D, G*S1*D, S1*D, D, ZERO, ONE, ZERO};

  std::initializer_list<ge::Expression> reduceResRepeat = {ONE, ONE, ONE, S1, ONE, ONE, BL};
  std::initializer_list<ge::Expression> reduceResStride = {ZERO, ZERO, ZERO, BL, ZERO, ZERO, ONE};

  int32_t exec_order = 0;
  Data query("query", graph);
  query.format = ge::FORMAT_ND;
  query.attr.sched.exec_order = exec_order++;
  query.attr.sched.axis = bmm1ResAxis;
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = bmm1ResAxis;
  *query.y.repeats = {B, N, G, S1, ONE, D, ONE};
  *query.y.strides = {N*G*S1*D, G*S1*D, S1*D, D, ZERO, ONE, ZERO};
  query.index = 0;
  query.axis_continuous_map = {{0, 1}, {2}, {3}, {4}, {INT64_MAX}, {5}, {INT64_MAX}};

  Data key("key", graph);
  key.format = ge::FORMAT_ND;
  key.attr.sched.exec_order = exec_order++;
  key.attr.sched.axis = bmm1ResAxis;
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = bmm1ResAxis;
  *key.y.repeats = {B, N, G, ONE, S2, D, ONE};
  *key.y.strides = {N*S1*D, S2*D, S2*D, ZERO, D, ONE, ZERO};
  key.index = 1;
  key.axis_continuous_map = {{0, 1}, {2}, {3}, {INT64_MAX}, {4}, {5}, {INT64_MAX}};

  MatMul bmm1("bmm1");
  bmm1.x1 = query.y;
  bmm1.x2 = key.y;
  bmm1.attr.sched.exec_order = exec_order++;
  bmm1.attr.sched.axis = bmm1ResAxis;
  bmm1.y.dtype = ge::DT_FLOAT;
  *bmm1.y.axis = bmm1ResAxis;
  *bmm1.y.repeats = bmm1ResRepeat;
  *bmm1.y.strides = bmm1ResStride;

  Load load1("load1");
  load1.x = bmm1.y;
  load1.attr.sched.exec_order = exec_order++;
  load1.attr.sched.axis = bmm1ResAxis;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = bmm1ResAxis;
  *load1.y.repeats = vec1ResRepeat;
  *load1.y.strides = vec1ResStride;

  Data pse("pse", graph);
  pse.format = ge::FORMAT_ND;
  pse.attr.sched.exec_order = exec_order++;
  pse.y.dtype = ge::DT_FLOAT16;
  *pse.y.axis = bmm1ResAxis;
  *pse.y.repeats = vec1ResRepeat;
  *pse.y.strides = vec1ResStride;

  Load loadPse("loadPse");
  loadPse.x = pse.y;
  loadPse.attr.sched.exec_order = exec_order++;
  loadPse.attr.sched.axis = bmm1ResAxis;
  loadPse.y.dtype = ge::DT_FLOAT16;
  *loadPse.y.axis = bmm1ResAxis;
  *loadPse.y.repeats = vec1ResRepeat;
  *loadPse.y.strides = vec1ResStride;

  Cast castPse("castPse");
  castPse.x = loadPse.y;
  castPse.attr.sched.exec_order = exec_order++;
  castPse.attr.sched.axis = bmm1ResAxis;
  castPse.y.dtype = ge::DT_FLOAT;
  *castPse.y.axis = bmm1ResAxis;
  *castPse.y.repeats = vec1ResRepeat;
  *castPse.y.strides = vec1ResStride;

  ge::ascir_op::Add add1("add1");
  add1.x1 = load1.y;
  add1.x2 = castPse.y;
  add1.attr.sched.exec_order = exec_order++;
  add1.attr.sched.axis = bmm1ResAxis;
  add1.y.dtype = ge::DT_FLOAT;
  *add1.y.axis = bmm1ResAxis;
  *add1.y.repeats = vec1ResRepeat;
  *add1.y.strides = vec1ResStride;

  Data scaleValue("scaleValue", graph);
  scaleValue.format = ge::FORMAT_ND;
  scaleValue.attr.sched.exec_order = exec_order++;
  scaleValue.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Muls mul1("mul1");
  mul1.x1 = add1.y;
  mul1.x2 = scaleValue.y;
  mul1.attr.sched.exec_order = exec_order++;
  mul1.attr.sched.axis = bmm1ResAxis;
  mul1.y.dtype = ge::DT_FLOAT;
  *mul1.y.axis = bmm1ResAxis;
  *mul1.y.repeats = vec1ResRepeat;
  *mul1.y.strides = vec1ResStride;

  Data attenMask("attenMask", graph);
  attenMask.format = ge::FORMAT_ND;
  attenMask.attr.sched.exec_order = exec_order++;
  attenMask.y.dtype = ge::DT_UINT8;
  *attenMask.y.axis = bmm1ResAxis;
  *attenMask.y.repeats = {B, ONE, ONE, S1, S2, ONE, ONE};
  *attenMask.y.strides = {S1*S2, S1*S2, S1*S2, S2, ONE, ZERO, ZERO};

  Load loadAttenMask("loadAttenMask");
  loadAttenMask.x = attenMask.y;
  loadAttenMask.attr.sched.exec_order = exec_order++;
  loadAttenMask.attr.sched.axis = bmm1ResAxis;
  loadAttenMask.y.dtype = ge::DT_UINT8;
  *loadAttenMask.y.axis = bmm1ResAxis;
  *loadAttenMask.y.repeats = {B, ONE, ONE, S1, S2, ONE, ONE};
  *loadAttenMask.y.strides = {S1*S2, S1*S2, S1*S2, S2, ONE, ZERO, ZERO};

  Select select("select");
  select.x1 = mul1.y;
  select.x2 = loadAttenMask.y;
  select.attr.sched.exec_order = exec_order++;
  select.attr.sched.axis = bmm1ResAxis;
  select.y.dtype = ge::DT_FLOAT;
  *select.y.axis = bmm1ResAxis;
  *select.y.repeats = vec1ResRepeat;
  *select.y.strides = vec1ResStride;

  TbufData softmaxExp("softmaxExp", graph);
  softmaxExp.attr.sched.exec_order = exec_order++;
  softmaxExp.attr.sched.axis = bmm1ResAxis;
  softmaxExp.y.dtype = ge::DT_FLOAT;
  *softmaxExp.y.axis = bmm1ResAxis;
  *softmaxExp.y.repeats = reduceResRepeat;
  *softmaxExp.y.strides = reduceResStride;

  TbufData softmaxApiTmpBuf("softmaxApiTmpBuf", graph);
  softmaxApiTmpBuf.attr.sched.exec_order = exec_order++;
  softmaxApiTmpBuf.attr.sched.axis = bmm1ResAxis;
  softmaxApiTmpBuf.y.dtype = ge::DT_FLOAT;
  *softmaxApiTmpBuf.y.axis = bmm1ResAxis;
  *softmaxApiTmpBuf.y.repeats = {ONE, ONE, ONE, S1, S2, ONE, ONE};
  *softmaxApiTmpBuf.y.strides = {ZERO, ZERO, ZERO, S2, ONE, ZERO, ZERO};

  FlashSoftmax flashSoftmax("flashSoftmax");
  flashSoftmax.x1 = select.y;
  flashSoftmax.x2 = softmaxExp.y;
  flashSoftmax.x3 = softmaxApiTmpBuf.y;
  flashSoftmax.attr.sched.exec_order = exec_order++;
  flashSoftmax.attr.sched.axis = bmm1ResAxis;
  flashSoftmax.y1.dtype = ge::DT_FLOAT;
  *flashSoftmax.y1.axis = bmm1ResAxis;
  *flashSoftmax.y1.repeats = vec1ResRepeat;
  *flashSoftmax.y1.strides = vec1ResStride;

  flashSoftmax.y2.dtype = ge::DT_FLOAT;
  *flashSoftmax.y2.axis = bmm1ResAxis;
  *flashSoftmax.y2.repeats = {B, N, G, S1, S2, ONE, BL};
  *flashSoftmax.y2.strides = {N*G*S1*BL, G*S1*BL, S1*BL, BL, ZERO, ZERO, ONE};

  flashSoftmax.y3.dtype = ge::DT_FLOAT;
  *flashSoftmax.y3.axis = bmm1ResAxis;
  *flashSoftmax.y3.repeats = {B, N, G, S1, S2, ONE, BL};
  *flashSoftmax.y3.strides = {N*G*S1*BL, G*S1*BL, S1*BL, BL, ZERO, ZERO, ONE};

  Store storeSoftmaxMax("storeSoftmaxMax");
  storeSoftmaxMax.x = flashSoftmax.y3;
  storeSoftmaxMax.attr.sched.exec_order = exec_order++;
  storeSoftmaxMax.attr.sched.axis = bmm1ResAxis;
  storeSoftmaxMax.y.dtype = ge::DT_FLOAT;
  *storeSoftmaxMax.y.axis = bmm1ResAxis;
  *storeSoftmaxMax.y.repeats = {B, N, G, S1, S2, ONE, BL};
  *storeSoftmaxMax.y.strides = {N*G*S1*BL, G*S1*BL, S1*BL, BL, ZERO, ZERO, ONE};

  Output softmaxMax("softmaxMax");
  softmaxMax.x = storeSoftmaxMax.y;
  softmaxMax.attr.sched.exec_order = exec_order++;

  Data dropMask("dropMask", graph);
  dropMask.format = ge::FORMAT_ND;
  dropMask.attr.sched.exec_order = exec_order++;
  dropMask.y.dtype = ge::DT_UINT8;
  *dropMask.y.axis = bmm1ResAxis;
  *dropMask.y.repeats = vec1ResRepeat;
  *dropMask.y.strides = vec1ResStride;

  Load loadDropMask("loadDropMask");
  loadDropMask.x = dropMask.y;
  loadDropMask.attr.sched.exec_order = exec_order++;
  loadDropMask.attr.sched.axis = bmm1ResAxis;
  loadDropMask.y.dtype = ge::DT_UINT8;
  *loadDropMask.y.axis = bmm1ResAxis;
  *loadDropMask.y.repeats = vec1ResRepeat;
  *loadDropMask.y.strides = vec1ResStride;

  Dropout dropout("dropout");
  dropout.x1 = flashSoftmax.y1;
  dropout.x2 = loadDropMask.y;
  dropout.attr.sched.exec_order = exec_order++;
  dropout.attr.sched.axis = bmm1ResAxis;
  dropout.y.dtype = ge::DT_FLOAT;
  *dropout.y.axis = bmm1ResAxis;
  *dropout.y.repeats = vec1ResRepeat;
  *dropout.y.strides = vec1ResStride;

  Cast castVec1Res("castVec1Res");
  castVec1Res.x = dropout.y;
  castVec1Res.attr.sched.exec_order = exec_order++;
  castVec1Res.attr.sched.axis = bmm1ResAxis;
  castVec1Res.y.dtype = ge::DT_FLOAT16;
  *castVec1Res.y.axis = bmm1ResAxis;
  *castVec1Res.y.repeats = vec1ResRepeat;
  *castVec1Res.y.strides = vec1ResStride;

  Store storeVec1Res("storeVec1Res");
  storeVec1Res.x = castVec1Res.y;
  storeVec1Res.attr.sched.exec_order = exec_order++;
  storeVec1Res.attr.sched.axis = bmm1ResAxis;
  storeVec1Res.y.dtype = ge::DT_FLOAT16;
  *storeVec1Res.y.axis = bmm1ResAxis;
  *storeVec1Res.y.repeats = vec1ResRepeat;
  *storeVec1Res.y.strides = vec1ResStride;

  Data value("value", graph);
  value.format = ge::FORMAT_ND;
  value.attr.sched.exec_order = exec_order++;
  value.attr.sched.axis = bmm2ResAxis;
  value.y.dtype = ge::DT_FLOAT16;
  *value.y.axis = bmm2ResAxis;
  *value.y.repeats = {B, N, G, ONE, S2, D, ONE};
  *value.y.strides = {N*S2*D, S2*D, S2*D, ZERO, D, ONE, ZERO};

  MatMul bmm2("bmm2");
  bmm2.x1 = storeVec1Res.y;
  bmm2.x2 = value.y;
  bmm2.attr.sched.exec_order = exec_order++;
  bmm2.attr.sched.axis = bmm2ResAxis;
  bmm2.y.dtype = ge::DT_FLOAT;
  *bmm2.y.axis = bmm2ResAxis;
  *bmm2.y.repeats = {B, N, G, S1, ONE, D, ONE};
  *bmm2.y.strides = {N*G*S1*D, G*S1*D, S1*D, D, ZERO, ONE, ZERO};

  Load load2("load2");
  load2.x = bmm2.y;
  load2.attr.sched.exec_order = exec_order++;
  load2.attr.sched.axis = bmm2ResAxis;
  load2.y.dtype = ge::DT_FLOAT;
  *load2.y.axis = bmm2ResAxis;
  *load2.y.repeats = vec2ResRepeat;
  *load2.y.strides = vec2ResStride;

  Workspace addResOut("addResOut", graph);
  addResOut.attr.sched.exec_order = exec_order++;
  addResOut.attr.sched.axis = bmm2ResAxis;
  addResOut.y.dtype = ge::DT_FLOAT;
  *addResOut.y.axis = bmm2ResAxis;
  *addResOut.y.repeats = vec2ResRepeat;
  *addResOut.y.strides = vec2ResStride;

  Load loadAddResOut("loadAddResOut");
  loadAddResOut.x = addResOut.y;
  loadAddResOut.attr.sched.exec_order = exec_order++;
  loadAddResOut.attr.sched.axis = bmm2ResAxis;
  loadAddResOut.y.dtype = ge::DT_FLOAT;
  *loadAddResOut.y.axis = bmm2ResAxis;
  *loadAddResOut.y.repeats = vec2ResRepeat;
  *loadAddResOut.y.strides = vec2ResStride;

  ge::ascir_op::Mul mulRes("mulRes");
  mulRes.x1 = loadAddResOut.y;
  mulRes.x2 = softmaxExp.y;
  mulRes.attr.sched.exec_order = exec_order++;
  mulRes.attr.sched.axis = bmm2ResAxis;
  mulRes.y.dtype = ge::DT_FLOAT;
  *mulRes.y.axis = bmm2ResAxis;
  *mulRes.y.repeats = vec2ResRepeat;
  *mulRes.y.strides = vec2ResStride;

  ge::ascir_op::Add addRes("addRes");
  addRes.x1 = load2.y;
  addRes.x2 = mulRes.y;
  addRes.attr.sched.exec_order = exec_order++;
  addRes.attr.sched.axis = bmm2ResAxis;
  addRes.y.dtype = ge::DT_FLOAT;
  *addRes.y.axis = bmm2ResAxis;
  *addRes.y.repeats = vec2ResRepeat;
  *addRes.y.strides = vec2ResStride;

  ge::ascir_op::Div div("div");
  div.x1 = addRes.y;
  div.x2 = flashSoftmax.y3;
  div.attr.sched.exec_order = exec_order++;
  div.attr.sched.axis = bmm2ResAxis;
  div.y.dtype = ge::DT_FLOAT;
  *div.y.axis = bmm2ResAxis;
  *div.y.repeats = vec2ResRepeat;
  *div.y.strides = vec2ResStride;

  Cast castBmm2Res("castBmm2Res");
  castBmm2Res.x = div.y;
  castBmm2Res.attr.sched.exec_order = exec_order++;
  castBmm2Res.attr.sched.axis = bmm2ResAxis;
  castBmm2Res.y.dtype = ge::DT_FLOAT16;
  *castBmm2Res.y.axis = bmm2ResAxis;
  *castBmm2Res.y.repeats = vec2ResRepeat;
  *castBmm2Res.y.strides = vec2ResStride;

  Store store("store");
  store.x = castBmm2Res.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = bmm2ResAxis;
  store.y.dtype = ge::DT_FLOAT16;
  *store.y.axis = bmm2ResAxis;
  *store.y.repeats = vec2ResRepeat;
  *store.y.strides = vec2ResStride;

  Output buf("buf");
  buf.x = store.y;
  buf.attr.sched.exec_order = exec_order++;
  buf.y.dtype = ge::DT_FLOAT16;
  *buf.y.axis = bmm2ResAxis;
  *buf.y.repeats = vec2ResRepeat;
  *buf.y.strides = vec2ResStride;
}
 /************************************************
 * 轴调度信息表达
 * original axes : [b,n,g,s1,s2,d,l,]
 * bngs1T : merged by [b,n,g,s1T,]
 * s1 : split into [s1T,s1t,]
 * s2 : split into [s2T,s2t,]
 * s1t : split into [s1tT,s1tt,s1tT2,s1tt2,]
 * bngs1T : split into [bngs1TB,bngs1Tb,]
 *************************************************/
void AddScheInfoToGraph(ge::AscGraph &graph) {
  auto b = graph.GetAllAxis()[0]->id;
  auto n = graph.GetAllAxis()[1]->id;
  auto g = graph.GetAllAxis()[2]->id;
  auto s1 = graph.GetAllAxis()[3]->id;
  auto s2 = graph.GetAllAxis()[4]->id;
  auto d = graph.GetAllAxis()[5]->id;
  auto bl = graph.GetAllAxis()[6]->id;

  std::tuple<ge::AxisPtr, ge::AxisPtr> split = graph.TileSplit(s1);
  auto s1T = *(std::get<0>(split));
  auto s1t = *(std::get<1>(split));
  graph.FindAxis(s1t.id)->align = ge::Symbol(128);
  auto mcAxis = *graph.MergeAxis({b, n, g, s1T.id});
  split = graph.BlockSplit(mcAxis.id);
  auto mcAxisB = *(std::get<0>(split));
  auto mcAxisb = *(std::get<1>(split));

  split = graph.TileSplit(s2);
  auto s2T = *(std::get<0>(split));
  auto s2t = *(std::get<1>(split));
  graph.FindAxis(s2t.id)->align = ge::Symbol(256);

  split = graph.TileSplit(s1t.id);
  auto s1tT = *(std::get<0>(split));
  auto s1tt = *(std::get<1>(split));
  graph.FindAxis(s1tt.id)->align = ge::Symbol(8);
  graph.FindAxis(s2t.id)->allow_unaligned_tail = false;
  vector<int64_t> bmm1VectorizedAxis{s1t.id, s2t.id, d};
  vector<int64_t> vec1VectorizedAxis{s1tt.id, s2t.id, d};
  vector<int64_t> bmm2VectorizedAxis{s1t.id, d, s2t.id};

  auto query = graph.FindNode("query");
  graph.ApplySplit(query, s1T.id, s1t.id);
  graph.ApplyMerge(query, mcAxis.id);
  graph.ApplySplit(query, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(query, s2T.id, s2t.id);
  graph.ApplyReorder(query, {mcAxisB.id, mcAxisb.id, s2T.id, s1t.id, s2t.id, d, bl});
  query->attr.sched.loop_axis = s2T.id;
  query->outputs[0].vectorized_axis = {s1t.id, s2t.id, d};

  auto key = graph.FindNode("key");
  graph.ApplySplit(key, s1T.id, s1t.id);
  graph.ApplyMerge(key, mcAxis.id);
  graph.ApplySplit(key, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(key, s2T.id, s2t.id);
  graph.ApplyReorder(key, {mcAxisB.id, mcAxisb.id, s2T.id, s1t.id, d, s2t.id, bl});
  key->attr.sched.loop_axis = s2T.id;
  key->outputs[0].vectorized_axis = {s1t.id, d, s2t.id};

  auto bmmReorderedAxis = {mcAxisB.id, mcAxisb.id, s2T.id, s1t.id, s2t.id, d, bl};
  auto vecReorderedAxis = {mcAxisB.id, mcAxisb.id, s2T.id, s1tT.id, s1tt.id, s2t.id, d, bl};

  auto bmm1 = graph.FindNode("bmm1");
  graph.ApplySplit(bmm1, s1T.id, s1t.id);
  graph.ApplyMerge(bmm1, mcAxis.id);
  graph.ApplySplit(bmm1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(bmm1, s2T.id, s2t.id);
  graph.ApplyReorder(bmm1, bmmReorderedAxis);
  bmm1->attr.sched.loop_axis = s2T.id;
  bmm1->outputs[0].vectorized_axis = bmm1VectorizedAxis;

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, s1T.id, s1t.id);
  graph.ApplyMerge(load1, mcAxis.id);
  graph.ApplySplit(load1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(load1, s2T.id, s2t.id);
  graph.ApplySplit(load1, s1tT.id, s1tt.id);
  graph.ApplyReorder(load1, vecReorderedAxis);
  load1->attr.sched.loop_axis = s1tT.id;
  load1->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto loadPse = graph.FindNode("loadPse");
  graph.ApplySplit(loadPse, s1T.id, s1t.id);
  graph.ApplyMerge(loadPse, mcAxis.id);
  graph.ApplySplit(loadPse, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadPse, s2T.id, s2t.id);
  graph.ApplySplit(loadPse, s1tT.id, s1tt.id);
  graph.ApplyReorder(loadPse, vecReorderedAxis);
  loadPse->attr.sched.loop_axis = s1tT.id;
  loadPse->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto castPse = graph.FindNode("castPse");
  graph.ApplySplit(castPse, s1T.id, s1t.id);
  graph.ApplyMerge(castPse, mcAxis.id);
  graph.ApplySplit(castPse, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(castPse, s2T.id, s2t.id);
  graph.ApplySplit(castPse, s1tT.id, s1tt.id);
  graph.ApplyReorder(castPse, vecReorderedAxis);
  castPse->attr.sched.loop_axis = s1tT.id;
  castPse->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto add1 = graph.FindNode("add1");
  graph.ApplySplit(add1, s1T.id, s1t.id);
  graph.ApplyMerge(add1, mcAxis.id);
  graph.ApplySplit(add1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(add1, s2T.id, s2t.id);
  graph.ApplySplit(add1, s1tT.id, s1tt.id);
  graph.ApplyReorder(add1, vecReorderedAxis);
  add1->attr.sched.loop_axis = s1tT.id;
  add1->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto mul1 = graph.FindNode("mul1");
  graph.ApplySplit(mul1, s1T.id, s1t.id);
  graph.ApplyMerge(mul1, mcAxis.id);
  graph.ApplySplit(mul1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(mul1, s2T.id, s2t.id);
  graph.ApplySplit(mul1, s1tT.id, s1tt.id);
  graph.ApplyReorder(mul1, vecReorderedAxis);
  mul1->attr.sched.loop_axis = s1tT.id;
  mul1->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto loadAttenMask = graph.FindNode("loadAttenMask");
  graph.ApplySplit(loadAttenMask, s1T.id, s1t.id);
  graph.ApplyMerge(loadAttenMask, mcAxis.id);
  graph.ApplySplit(loadAttenMask, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadAttenMask, s2T.id, s2t.id);
  graph.ApplySplit(loadAttenMask, s1tT.id, s1tt.id);
  graph.ApplyReorder(loadAttenMask, vecReorderedAxis);
  loadAttenMask->attr.sched.loop_axis = s1tT.id;
  loadAttenMask->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto select = graph.FindNode("select");
  graph.ApplySplit(select, s1T.id, s1t.id);
  graph.ApplyMerge(select, mcAxis.id);
  graph.ApplySplit(select, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(select, s2T.id, s2t.id);
  graph.ApplySplit(select, s1tT.id, s1tt.id);
  graph.ApplyReorder(select, vecReorderedAxis);
  select->attr.sched.loop_axis = s1tT.id;
  select->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto loadDropMask = graph.FindNode("loadDropMask");
  graph.ApplySplit(loadDropMask, s1T.id, s1t.id);
  graph.ApplyMerge(loadDropMask, mcAxis.id);
  graph.ApplySplit(loadDropMask, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadDropMask, s2T.id, s2t.id);
  graph.ApplySplit(loadDropMask, s1tT.id, s1tt.id);
  graph.ApplyReorder(loadDropMask, vecReorderedAxis);
  loadDropMask->attr.sched.loop_axis = s1tT.id;
  loadDropMask->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto dropout = graph.FindNode("dropout");
  graph.ApplySplit(dropout, s1T.id, s1t.id);
  graph.ApplyMerge(dropout, mcAxis.id);
  graph.ApplySplit(dropout, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(dropout, s2T.id, s2t.id);
  graph.ApplySplit(dropout, s1tT.id, s1tt.id);
  graph.ApplyReorder(dropout, vecReorderedAxis);
  dropout->attr.sched.loop_axis = s1tT.id;
  dropout->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto castVec1Res = graph.FindNode("castVec1Res");
  graph.ApplySplit(castVec1Res, s1T.id, s1t.id);
  graph.ApplyMerge(castVec1Res, mcAxis.id);
  graph.ApplySplit(castVec1Res, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(castVec1Res, s2T.id, s2t.id);
  graph.ApplySplit(castVec1Res, s1tT.id, s1tt.id);
  graph.ApplyReorder(castVec1Res, vecReorderedAxis);
  castVec1Res->attr.sched.loop_axis = s1tT.id;
  castVec1Res->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto storeVec1Res = graph.FindNode("storeVec1Res");
  graph.ApplySplit(storeVec1Res, s1T.id, s1t.id);
  graph.ApplyMerge(storeVec1Res, mcAxis.id);
  graph.ApplySplit(storeVec1Res, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(storeVec1Res, s2T.id, s2t.id);
  graph.ApplySplit(storeVec1Res, s1tT.id, s1tt.id);
  graph.ApplyReorder(storeVec1Res, vecReorderedAxis);
  storeVec1Res->attr.sched.loop_axis = s1tT.id;
  storeVec1Res->outputs[0].vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d};

  auto softmaxExp = graph.FindNode("softmaxExp");
  graph.ApplySplit(softmaxExp, s1T.id, s1t.id);
  graph.ApplyMerge(softmaxExp, mcAxis.id);
  graph.ApplySplit(softmaxExp, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(softmaxExp, s2T.id, s2t.id);
  graph.ApplySplit(softmaxExp, s1tT.id, s1tt.id);
  graph.ApplyReorder(softmaxExp, vecReorderedAxis);
  softmaxExp->attr.sched.loop_axis = s1tT.id;
  softmaxExp->outputs[0].vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d, bl};

  auto softmaxApiTmpBuf = graph.FindNode("softmaxApiTmpBuf");
  graph.ApplySplit(softmaxApiTmpBuf, s1T.id, s1t.id);
  graph.ApplyMerge(softmaxApiTmpBuf, mcAxis.id);
  graph.ApplySplit(softmaxApiTmpBuf, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(softmaxApiTmpBuf, s2T.id, s2t.id);
  graph.ApplySplit(softmaxApiTmpBuf, s1tT.id, s1tt.id);
  graph.ApplyReorder(softmaxApiTmpBuf, vecReorderedAxis);
  softmaxApiTmpBuf->attr.sched.loop_axis = s1tT.id;
  softmaxApiTmpBuf->outputs[0].vectorized_axis = vec1VectorizedAxis;

  auto flashSoftmax = graph.FindNode("flashSoftmax");
  graph.ApplySplit(flashSoftmax, s1T.id, s1t.id);
  graph.ApplyMerge(flashSoftmax, mcAxis.id);
  graph.ApplySplit(flashSoftmax, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(flashSoftmax, s2T.id, s2t.id);
  graph.ApplySplit(flashSoftmax, s1tT.id, s1tt.id);
  graph.ApplyReorder(flashSoftmax, vecReorderedAxis);
  flashSoftmax->attr.sched.loop_axis = s1tT.id;
  flashSoftmax->outputs[0].vectorized_axis = vec1VectorizedAxis;
  flashSoftmax->outputs[1].vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d, bl};
  flashSoftmax->outputs[2].vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d, bl};

  auto storeSoftmaxMax = graph.FindNode("storeSoftmaxMax");
  graph.ApplySplit(storeSoftmaxMax, s1T.id, s1t.id);
  graph.ApplyMerge(storeSoftmaxMax, mcAxis.id);
  graph.ApplySplit(storeSoftmaxMax, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(storeSoftmaxMax, s2T.id, s2t.id);
  graph.ApplySplit(storeSoftmaxMax, s1tT.id, s1tt.id);
  graph.ApplyReorder(storeSoftmaxMax, vecReorderedAxis);
  storeSoftmaxMax->attr.sched.loop_axis = s1tT.id;
  storeSoftmaxMax->outputs[0].vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d, bl};

  auto value = graph.FindNode("value");
  graph.ApplySplit(value, s1T.id, s1t.id);
  graph.ApplyMerge(value, mcAxis.id);
  graph.ApplySplit(value, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(value, s2T.id, s2t.id);
  graph.ApplyReorder(value, bmmReorderedAxis);
  value->attr.sched.loop_axis = s2T.id;
  value->outputs[0].vectorized_axis = {s1t.id, s2t.id, d};

  auto bmm2 = graph.FindNode("bmm2");
  graph.ApplySplit(bmm2, s1T.id, s1t.id);
  graph.ApplyMerge(bmm2, mcAxis.id);
  graph.ApplySplit(bmm2, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(bmm2, s2T.id, s2t.id);
  graph.ApplyReorder(bmm2, bmmReorderedAxis);
  bmm2->attr.sched.loop_axis = s2T.id;
  bmm2->outputs[0].vectorized_axis = {s1t.id, s2t.id, d};

  split = graph.TileSplit(s1t.id, "s1tT2", "s1tt2");
  auto s1Vec2tT = *(std::get<0>(split));
  auto s1Vec2tt = *(std::get<1>(split));
  vector<int64_t> vec2VectorizedAxis{s1Vec2tt.id, d, s2t.id};
  auto vec2ReorderedAxis = {mcAxisB.id, mcAxisb.id, s2T.id, s1Vec2tT.id, s1Vec2tt.id, s2t.id, d, bl};

  auto load2 = graph.FindNode("load2");
  graph.ApplySplit(load2, s1T.id, s1t.id);
  graph.ApplyMerge(load2, mcAxis.id);
  graph.ApplySplit(load2, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(load2, s2T.id, s2t.id);
  graph.ApplySplit(load2, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(load2, vec2ReorderedAxis);
  load2->attr.sched.loop_axis = s1Vec2tT.id;
  load2->outputs[0].vectorized_axis = vec2VectorizedAxis;

  auto addResOut = graph.FindNode("addResOut");
  graph.ApplySplit(addResOut, s1T.id, s1t.id);
  graph.ApplyMerge(addResOut, mcAxis.id);
  graph.ApplySplit(addResOut, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(addResOut, s2T.id, s2t.id);
  graph.ApplySplit(addResOut, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(addResOut, vec2ReorderedAxis);
  addResOut->attr.sched.loop_axis = s1Vec2tT.id;
  addResOut->outputs[0].vectorized_axis = vec2VectorizedAxis;

  auto loadAddResOut = graph.FindNode("loadAddResOut");
  graph.ApplySplit(loadAddResOut, s1T.id, s1t.id);
  graph.ApplyMerge(loadAddResOut, mcAxis.id);
  graph.ApplySplit(loadAddResOut, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadAddResOut, s2T.id, s2t.id);
  graph.ApplySplit(loadAddResOut, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(loadAddResOut, vec2ReorderedAxis);
  loadAddResOut->attr.sched.loop_axis = s1Vec2tT.id;
  loadAddResOut->outputs[0].vectorized_axis = vec2VectorizedAxis;

  auto mulRes = graph.FindNode("mulRes");
  graph.ApplySplit(mulRes, s1T.id, s1t.id);
  graph.ApplyMerge(mulRes, mcAxis.id);
  graph.ApplySplit(mulRes, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(mulRes, s2T.id, s2t.id);
  graph.ApplySplit(mulRes, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(mulRes, vec2ReorderedAxis);
  mulRes->attr.sched.loop_axis = s1Vec2tT.id;
  mulRes->outputs[0].vectorized_axis = vec2VectorizedAxis;

  auto addRes = graph.FindNode("addRes");
  graph.ApplySplit(addRes, s1T.id, s1t.id);
  graph.ApplyMerge(addRes, mcAxis.id);
  graph.ApplySplit(addRes, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(addRes, s2T.id, s2t.id);
  graph.ApplySplit(addRes, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(addRes, vec2ReorderedAxis);
  addRes->attr.sched.loop_axis = s1Vec2tT.id;
  addRes->outputs[0].vectorized_axis = vec2VectorizedAxis;

  auto div = graph.FindNode("div");
  graph.ApplySplit(div, s1T.id, s1t.id);
  graph.ApplyMerge(div, mcAxis.id);
  graph.ApplySplit(div, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(div, s2T.id, s2t.id);
  graph.ApplySplit(div, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(div, vec2ReorderedAxis);
  div->attr.sched.loop_axis = s1Vec2tT.id;
  div->outputs[0].vectorized_axis = vec2VectorizedAxis;

  auto castBmm2Res = graph.FindNode("castBmm2Res");
  graph.ApplySplit(castBmm2Res, s1T.id, s1t.id);
  graph.ApplyMerge(castBmm2Res, mcAxis.id);
  graph.ApplySplit(castBmm2Res, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(castBmm2Res, s2T.id, s2t.id);
  graph.ApplySplit(castBmm2Res, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(castBmm2Res, vec2ReorderedAxis);
  castBmm2Res->attr.sched.loop_axis = s1Vec2tT.id;
  castBmm2Res->outputs[0].vectorized_axis = vec2VectorizedAxis;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, s1T.id, s1t.id);
  graph.ApplyMerge(store, mcAxis.id);
  graph.ApplySplit(store, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(store, s2T.id, s2t.id);
  graph.ApplySplit(store, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(store, vec2ReorderedAxis);
  store->attr.sched.loop_axis = s1Vec2tT.id;
  store->outputs[0].vectorized_axis = vec2VectorizedAxis;
}

/********************************************************************
 * 内存信息表达
 * original axes : [b,n,g,s1,s2,d,l,]
 * bngs1T : merged by [b,n,g,s1T,]
 * s1 : split into [s1T,s1t,]
 * s2 : split into [s2T,s2t,]
 * s1t : split into [s1tT,s1tt,s1tT2,s1tt2,]
 * bngs1T : split into [bngs1TB,bngs1Tb,]
 * 
 * Q0 : [bmm1_output_0]
 * Q1 : [load1_output_0, add1_output_0, mul1_output_0, select_output_0, flashSoftmax_output_0, dropout_output_0]
 * Q2 : [flashSoftmax_output_2]
 * Q3 : [softmaxExp_output_0]
 * Q4 : [storeVec1Res_output_0]
 * Q5 : [bmm2_output_0]
 * Q6 : [addResOut_output_0]
 * BUF0 : [loadPse_output_0, castVec1Res_output_0]
 * BUF1 : [castPse_output_0, softmaxApiTmpBuf_output_0, load2_output_0]
 * BUF2 : [loadAttenMask_output_0]
 * BUF3 : [loadDropMask_output_0]
 * BUF4 : [flashSoftmax_output_1]
 * BUF5 : [{castBmm2Res_output_0, castBmm2Res_output_0}, loadAddResOut_output_0, mulRes_output_0, addRes_output_0, div_output_0, castBmm2Res_output_0]
 * ********************************************************************/
void AddBuffInfoToGraph(ge::AscGraph &graph) {
  int32_t tensorID = 0;
  int32_t queID = 0;
  int32_t bufID = 0;
  int32_t mmRes1Que = queID++;
  int32_t stage1Que = queID++;
  int32_t pseTBuf = bufID++;
  int32_t commonTBuf = bufID++;
  int32_t maskTbufPing = bufID++;
  int32_t maskTbufPong = bufID++;
  int32_t softmaxMaxBuf = bufID++;
  int32_t softmaxSumQueue = queID++;
  int32_t softmaxExpQueue = queID++;
  int32_t stage2Buf = bufID++;
  int32_t stage1ResQueue = queID++;
  int32_t mm2ResQueue = queID++;
  int32_t vec2ResQueue = queID++;

  auto query = graph.FindNode("query");
  query->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  query->outputs[0].attr.mem.position = ge::Position::POSITION_GM;

  auto key = graph.FindNode("key");
  key->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  key->outputs[0].attr.mem.position = ge::Position::POSITION_GM;

  auto value = graph.FindNode("value");
  value->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  value->outputs[0].attr.mem.position = ge::Position::POSITION_GM;

  auto bmm1 = graph.FindNode("bmm1");
  bmm1->outputs[0].attr.mem.tensor_id = tensorID++;
  bmm1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  bmm1->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  bmm1->outputs[0].attr.mem.position = ge::Position::POSITION_GM;
  bmm1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  bmm1->outputs[0].attr.buf.id = ge::kIdNone;
  bmm1->outputs[0].attr.que.id = mmRes1Que;
  bmm1->outputs[0].attr.que.depth = 2;
  bmm1->outputs[0].attr.que.buf_num = 2;
  bmm1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = tensorID++;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  load1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  load1->outputs[0].attr.buf.id = ge::kIdNone;
  load1->outputs[0].attr.que.id = stage1Que;
  load1->outputs[0].attr.que.depth = 2;
  load1->outputs[0].attr.que.buf_num = 2;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto loadPse = graph.FindNode("loadPse");
  loadPse->outputs[0].attr.mem.tensor_id = tensorID++;
  loadPse->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  loadPse->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  loadPse->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  loadPse->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  loadPse->outputs[0].attr.buf.id = pseTBuf;
  loadPse->outputs[0].attr.que.id = ge::kIdNone;
  loadPse->outputs[0].attr.que.depth = ge::kIdNone;
  loadPse->outputs[0].attr.que.buf_num = ge::kIdNone;
  loadPse->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto castPse = graph.FindNode("castPse");
  castPse->outputs[0].attr.mem.tensor_id = tensorID++;
  castPse->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  castPse->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  castPse->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  castPse->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  castPse->outputs[0].attr.buf.id = commonTBuf;
  castPse->outputs[0].attr.que.id = ge::kIdNone;
  castPse->outputs[0].attr.que.depth = ge::kIdNone;
  castPse->outputs[0].attr.que.buf_num = ge::kIdNone;
  castPse->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto add1 = graph.FindNode("add1");
  add1->outputs[0].attr.mem.tensor_id = tensorID++;
  add1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  add1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  add1->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  add1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  add1->outputs[0].attr.buf.id = ge::kIdNone;
  add1->outputs[0].attr.que.id = stage1Que;
  add1->outputs[0].attr.que.depth = 2;
  add1->outputs[0].attr.que.buf_num = 2;
  add1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto mul1 = graph.FindNode("mul1");
  mul1->outputs[0].attr.mem.tensor_id = tensorID++;
  mul1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  mul1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mul1->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mul1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  mul1->outputs[0].attr.buf.id = ge::kIdNone;
  mul1->outputs[0].attr.que.id = stage1Que;
  mul1->outputs[0].attr.que.depth = 2;
  mul1->outputs[0].attr.que.buf_num = 2;
  mul1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto select = graph.FindNode("select");
  select->outputs[0].attr.mem.tensor_id = tensorID++;
  select->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  select->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  select->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  select->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  select->outputs[0].attr.buf.id = ge::kIdNone;
  select->outputs[0].attr.que.id = stage1Que;
  select->outputs[0].attr.que.depth = 2;
  select->outputs[0].attr.que.buf_num = 2;
  select->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto softmaxExp = graph.FindNode("softmaxExp");
  softmaxExp->outputs[0].attr.mem.tensor_id = tensorID++;
  softmaxExp->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  softmaxExp->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  softmaxExp->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  softmaxExp->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  softmaxExp->outputs[0].attr.buf.id = ge::kIdNone;
  softmaxExp->outputs[0].attr.que.id = softmaxExpQueue;
  softmaxExp->outputs[0].attr.que.depth = 2;
  softmaxExp->outputs[0].attr.que.buf_num = 2;
  softmaxExp->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto softmaxApiTmpBuf = graph.FindNode("softmaxApiTmpBuf");
  softmaxApiTmpBuf->outputs[0].attr.mem.tensor_id = tensorID++;
  softmaxApiTmpBuf->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  softmaxApiTmpBuf->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  softmaxApiTmpBuf->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  softmaxApiTmpBuf->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  softmaxApiTmpBuf->outputs[0].attr.buf.id = commonTBuf;
  softmaxApiTmpBuf->outputs[0].attr.que.id = ge::kIdNone;
  softmaxApiTmpBuf->outputs[0].attr.que.depth = ge::kIdNone;
  softmaxApiTmpBuf->outputs[0].attr.que.buf_num = ge::kIdNone;
  softmaxApiTmpBuf->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto flashSoftmax = graph.FindNode("flashSoftmax");
  flashSoftmax->outputs[0].attr.mem.tensor_id = tensorID++;
  flashSoftmax->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  flashSoftmax->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  flashSoftmax->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  flashSoftmax->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  flashSoftmax->outputs[0].attr.buf.id = ge::kIdNone;
  flashSoftmax->outputs[0].attr.que.id = stage1Que;
  flashSoftmax->outputs[0].attr.que.depth = 2;
  flashSoftmax->outputs[0].attr.que.buf_num = 2;
  flashSoftmax->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  flashSoftmax->outputs[1].attr.mem.tensor_id = tensorID++;
  flashSoftmax->outputs[1].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  flashSoftmax->outputs[1].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  flashSoftmax->outputs[1].attr.mem.position = ge::Position::kPositionVecOut;
  flashSoftmax->outputs[1].attr.mem.reuse_id = ge::kIdNone;
  flashSoftmax->outputs[1].attr.buf.id = softmaxMaxBuf;
  flashSoftmax->outputs[1].attr.que.id = ge::kIdNone;
  flashSoftmax->outputs[1].attr.que.depth = ge::kIdNone;
  flashSoftmax->outputs[1].attr.que.buf_num = ge::kIdNone;
  flashSoftmax->outputs[1].attr.opt.ref_tensor = ge::kIdNone;

  flashSoftmax->outputs[2].attr.mem.tensor_id = tensorID++;
  flashSoftmax->outputs[2].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  flashSoftmax->outputs[2].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  flashSoftmax->outputs[2].attr.mem.position = ge::Position::kPositionVecOut;
  flashSoftmax->outputs[2].attr.mem.reuse_id = ge::kIdNone;
  flashSoftmax->outputs[2].attr.buf.id = ge::kIdNone;
  flashSoftmax->outputs[2].attr.que.id = softmaxSumQueue;
  flashSoftmax->outputs[2].attr.que.depth = 2;
  flashSoftmax->outputs[2].attr.que.buf_num = 2;
  flashSoftmax->outputs[2].attr.opt.ref_tensor = ge::kIdNone;

  auto loadAttenMask = graph.FindNode("loadAttenMask");
  loadAttenMask->outputs[0].attr.mem.tensor_id = tensorID++;
  loadAttenMask->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  loadAttenMask->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  loadAttenMask->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  loadAttenMask->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  loadAttenMask->outputs[0].attr.buf.id = maskTbufPing;
  loadAttenMask->outputs[0].attr.que.id = ge::kIdNone;
  loadAttenMask->outputs[0].attr.que.depth = ge::kIdNone;
  loadAttenMask->outputs[0].attr.que.buf_num = ge::kIdNone;
  loadAttenMask->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto loadDropMask = graph.FindNode("loadDropMask");
  loadDropMask->outputs[0].attr.mem.tensor_id = tensorID++;
  loadDropMask->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  loadDropMask->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  loadDropMask->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  loadDropMask->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  loadDropMask->outputs[0].attr.buf.id = maskTbufPong;
  loadDropMask->outputs[0].attr.que.id = ge::kIdNone;
  loadDropMask->outputs[0].attr.que.depth = ge::kIdNone;
  loadDropMask->outputs[0].attr.que.buf_num = ge::kIdNone;
  loadDropMask->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto dropout = graph.FindNode("dropout");
  dropout->outputs[0].attr.mem.tensor_id = tensorID++;
  dropout->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  dropout->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  dropout->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  dropout->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  dropout->outputs[0].attr.buf.id = ge::kIdNone;
  dropout->outputs[0].attr.que.id = stage1Que;
  dropout->outputs[0].attr.que.depth = 2;
  dropout->outputs[0].attr.que.buf_num = 2;
  dropout->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto castVec1Res = graph.FindNode("castVec1Res");
  castVec1Res->outputs[0].attr.mem.tensor_id = tensorID++;
  castVec1Res->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  castVec1Res->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  castVec1Res->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  castVec1Res->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  castVec1Res->outputs[0].attr.buf.id = pseTBuf;
  castVec1Res->outputs[0].attr.que.id = ge::kIdNone;
  castVec1Res->outputs[0].attr.que.depth = ge::kIdNone;
  castVec1Res->outputs[0].attr.que.buf_num = ge::kIdNone;
  castVec1Res->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto storeVec1Res = graph.FindNode("storeVec1Res");
  storeVec1Res->outputs[0].attr.mem.tensor_id = tensorID++;
  storeVec1Res->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  storeVec1Res->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  storeVec1Res->outputs[0].attr.mem.position = ge::Position::POSITION_GM;
  storeVec1Res->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  storeVec1Res->outputs[0].attr.buf.id = ge::kIdNone;
  storeVec1Res->outputs[0].attr.que.id = stage1ResQueue;
  storeVec1Res->outputs[0].attr.que.depth = 2;
  storeVec1Res->outputs[0].attr.que.buf_num = 2;
  storeVec1Res->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto bmm2 = graph.FindNode("bmm2");
  bmm2->outputs[0].attr.mem.tensor_id = tensorID++;
  bmm2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  bmm2->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  bmm2->outputs[0].attr.mem.position = ge::Position::POSITION_GM;
  bmm2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  bmm2->outputs[0].attr.buf.id = ge::kIdNone;
  bmm2->outputs[0].attr.que.id = mm2ResQueue;
  bmm2->outputs[0].attr.que.depth = 2;
  bmm2->outputs[0].attr.que.buf_num = 2;
  bmm2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto load2 = graph.FindNode("load2");
  load2->outputs[0].attr.mem.tensor_id = tensorID++;
  load2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  load2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load2->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  load2->outputs[0].attr.buf.id = commonTBuf;
  load2->outputs[0].attr.que.id = ge::kIdNone;
  load2->outputs[0].attr.que.depth = ge::kIdNone;
  load2->outputs[0].attr.que.buf_num = ge::kIdNone;
  load2->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto addResOut = graph.FindNode("addResOut");
  addResOut->outputs[0].attr.mem.tensor_id = tensorID++;
  addResOut->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  addResOut->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  addResOut->outputs[0].attr.mem.position = ge::Position::POSITION_GM;
  addResOut->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  addResOut->outputs[0].attr.buf.id = ge::kIdNone;
  addResOut->outputs[0].attr.que.id = vec2ResQueue;
  addResOut->outputs[0].attr.que.depth = 2;
  addResOut->outputs[0].attr.que.buf_num = 2;
  addResOut->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto loadAddResOut = graph.FindNode("loadAddResOut");
  loadAddResOut->outputs[0].attr.mem.tensor_id = tensorID++;
  loadAddResOut->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  loadAddResOut->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  loadAddResOut->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  loadAddResOut->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  loadAddResOut->outputs[0].attr.buf.id = stage2Buf;
  loadAddResOut->outputs[0].attr.que.id = ge::kIdNone;
  loadAddResOut->outputs[0].attr.que.depth = ge::kIdNone;
  loadAddResOut->outputs[0].attr.que.buf_num = ge::kIdNone;
  loadAddResOut->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto mulRes = graph.FindNode("mulRes");
  mulRes->outputs[0].attr.mem.tensor_id = tensorID++;
  mulRes->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  mulRes->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mulRes->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  mulRes->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  mulRes->outputs[0].attr.buf.id = stage2Buf;
  mulRes->outputs[0].attr.que.id = ge::kIdNone;
  mulRes->outputs[0].attr.que.depth = ge::kIdNone;
  mulRes->outputs[0].attr.que.buf_num = ge::kIdNone;
  mulRes->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto addRes = graph.FindNode("addRes");
  addRes->outputs[0].attr.mem.tensor_id = tensorID++;
  addRes->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  addRes->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  addRes->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  addRes->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  addRes->outputs[0].attr.buf.id = stage2Buf;
  addRes->outputs[0].attr.que.id = ge::kIdNone;
  addRes->outputs[0].attr.que.depth = ge::kIdNone;
  addRes->outputs[0].attr.que.buf_num = ge::kIdNone;
  addRes->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto div = graph.FindNode("div");
  div->outputs[0].attr.mem.tensor_id = tensorID++;
  div->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  div->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  div->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  div->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  div->outputs[0].attr.buf.id = stage2Buf;
  div->outputs[0].attr.que.id = ge::kIdNone;
  div->outputs[0].attr.que.depth = ge::kIdNone;
  div->outputs[0].attr.que.buf_num = ge::kIdNone;
  div->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto castBmm2Res = graph.FindNode("castBmm2Res");
  castBmm2Res->outputs[0].attr.mem.tensor_id = tensorID++;
  castBmm2Res->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  castBmm2Res->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  castBmm2Res->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  castBmm2Res->outputs[0].attr.mem.reuse_id = castBmm2Res->outputs[0].attr.mem.tensor_id;
  castBmm2Res->outputs[0].attr.buf.id = stage2Buf;
  castBmm2Res->outputs[0].attr.que.id = ge::kIdNone;
  castBmm2Res->outputs[0].attr.que.depth = ge::kIdNone;
  castBmm2Res->outputs[0].attr.que.buf_num = ge::kIdNone;
  castBmm2Res->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto store = graph.FindNode("store");
  store->outputs[0].attr.mem.tensor_id = tensorID++;
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::ALLOC_TYPE_GLOBAL;
  store->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  store->outputs[0].attr.mem.position = ge::Position::POSITION_GM;
  store->outputs[0].attr.mem.reuse_id = 0;
  store->outputs[0].attr.opt.ref_tensor = 0;

  auto storeSoftmaxMax = graph.FindNode("storeSoftmaxMax");
  storeSoftmaxMax->outputs[0].attr.mem.tensor_id = tensorID++;
  storeSoftmaxMax->outputs[0].attr.mem.alloc_type = ge::AllocType::ALLOC_TYPE_GLOBAL;
  storeSoftmaxMax->outputs[0].attr.mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  storeSoftmaxMax->outputs[0].attr.mem.position = ge::Position::POSITION_GM;
  storeSoftmaxMax->outputs[0].attr.mem.reuse_id = 0;
  storeSoftmaxMax->outputs[0].attr.opt.ref_tensor = 0;
}

ge::Status GenerateAscGraphs(std::vector<ge::AscGraph> &graphs) {
  ge::AscGraph graph("graph");
  BuildOriginGraph(graph);
  AddScheInfoToGraph(graph);
  AddBuffInfoToGraph(graph);
  graph.SetTilingKey(0);
  graphs.emplace_back(graph);
  return ge::SUCCESS;
}

void GeneratorAttOptions(std::map<std::string, std::string> &options) {
  (void)options;
}
