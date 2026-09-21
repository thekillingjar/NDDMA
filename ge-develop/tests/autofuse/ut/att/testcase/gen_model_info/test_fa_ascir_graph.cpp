/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "test_fa_ascir_graph.h"
#include "base/base_types.h"

namespace att {
using namespace att;
using namespace ge::ascir_op;
void FaBeforeAutoFuse(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;

  auto B = ge::Symbol("B");
  auto N = ge::Symbol("N");
  auto G = ge::Symbol("G");
  auto S1 = ge::Symbol("S1");
  auto S2 = ge::Symbol("S2");
  auto D = ge::Symbol("D");
  auto BL = ge::Symbol(8, "BL");

  auto b = graph.CreateAxis("b", B);
  auto n = graph.CreateAxis("n", N);
  auto g = graph.CreateAxis("g", G);
  auto s1 = graph.CreateAxis("s1", S1);
  auto s2 = graph.CreateAxis("s2", S2);
  auto d = graph.CreateAxis("d", D);
  auto bl = graph.CreateAxis("l", BL);

  auto bmm1ResAxis = {b.id, n.id, g.id, s1.id, s2.id, d.id, bl.id};
  std::initializer_list<Expr> bmm1ResRepeat = {B, N, G, S1, S2, ONE, ONE};
  std::initializer_list<Expr> bmm1ResStride = {N*G*S1*S2, G*S1*S2, S1*S2, S2, ONE, ZERO, ZERO};

  std::initializer_list<Expr> vec1ResRepeat = {B, N, G, S1, S2, ONE, ONE};
  std::initializer_list<Expr> vec1ResStride = {N*G*S1*S2, G*S1*S2, S1*S2, S2, ONE, ZERO, ZERO};

  auto bmm2ResAxis = {b.id, n.id, g.id, s1.id, s2.id, d.id, bl.id};
  std::initializer_list<Expr> bmm2ResRepeat = {B, N, G, S1, ONE, D, ONE};
  std::initializer_list<Expr> bmm2ResStride = {N*G*S1*D, G*S1*D, S1*D, D, ZERO, ONE, ZERO};

  std::initializer_list<Expr> vec2ResRepeat = {B, N, G, S1, ONE, D, ONE};
  std::initializer_list<Expr> vec2ResStride = {N*G*S1*D, G*S1*D, S1*D, D, ZERO, ONE, ZERO};

  std::initializer_list<Expr> reduceResRepeat = {ONE, ONE, ONE, S1, ONE, ONE, BL};
  std::initializer_list<Expr> reduceResStride = {ZERO, ZERO, ZERO, BL, ZERO, ZERO, ONE};

  int32_t exec_order = 0;
  Data query("query", graph);
  query.attr.sched.exec_order = exec_order++;
  query.attr.sched.axis = bmm1ResAxis;
  query.y.dtype = ge::DT_FLOAT16;
  *query.y.axis = bmm1ResAxis;
  *query.y.repeats = {B, N, G, S1, ONE, D, ONE};
  *query.y.strides = {N*G*S1*D, G*S1*D, S1*D, D, ZERO, ONE, ZERO};

  Data key("key", graph);
  key.attr.sched.exec_order = exec_order++;
  key.attr.sched.axis = bmm1ResAxis;
  key.y.dtype = ge::DT_FLOAT16;
  *key.y.axis = bmm1ResAxis;
  *key.y.repeats = {B, N, G, ONE, S2, D, ONE};
  *key.y.strides = {N*S1*D, S2*D, S2*D, ZERO, D, ONE, ZERO};

  Add bmm1("bmm1");
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
  scaleValue.attr.sched.exec_order = exec_order++;
  scaleValue.y.dtype = ge::DT_FLOAT;

  ge::ascir_op::Add mul1("mul1");
  mul1.x1 = add1.y;
  mul1.x2 = scaleValue.y;
  mul1.attr.sched.exec_order = exec_order++;
  mul1.attr.sched.axis = bmm1ResAxis;
  mul1.y.dtype = ge::DT_FLOAT;
  *mul1.y.axis = bmm1ResAxis;
  *mul1.y.repeats = vec1ResRepeat;
  *mul1.y.strides = vec1ResStride;

  Data attenMask("attenMask", graph);
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

  Data softmaxExp("softmaxExp", graph);
  softmaxExp.attr.sched.exec_order = exec_order++;
  softmaxExp.attr.sched.axis = bmm1ResAxis;
  softmaxExp.y.dtype = ge::DT_FLOAT;
  *softmaxExp.y.axis = bmm1ResAxis;
  *softmaxExp.y.repeats = reduceResRepeat;
  *softmaxExp.y.strides = reduceResStride;

  Data softmaxApiTmpBuf("softmaxApiTmpBuf", graph);
  softmaxApiTmpBuf.attr.sched.exec_order = exec_order++;
  softmaxApiTmpBuf.attr.sched.axis = bmm1ResAxis;
  softmaxApiTmpBuf.y.dtype = ge::DT_FLOAT;
  *softmaxApiTmpBuf.y.axis = bmm1ResAxis;
  *softmaxApiTmpBuf.y.repeats = {ONE, ONE, ONE, S1, S2, ONE, ONE};
  *softmaxApiTmpBuf.y.strides = {ZERO, ZERO, ZERO, S2, ONE, ZERO, ZERO};

  //这里用Concat替代FlashSoftmax
  Concat flashSoftmax("flashSoftmax");
  flashSoftmax.x = {select.y, softmaxExp.y, softmaxApiTmpBuf.y};
  flashSoftmax.attr.sched.exec_order = exec_order++;
  flashSoftmax.attr.sched.axis = bmm1ResAxis;
  flashSoftmax.y.dtype = ge::DT_FLOAT;
  *flashSoftmax.y.axis = bmm1ResAxis;
  *flashSoftmax.y.repeats = vec1ResRepeat;
  *flashSoftmax.y.strides = vec1ResStride;

  Store storeSoftmaxMax("storeSoftmaxMax");
  storeSoftmaxMax.x = flashSoftmax.y;
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

  Add dropout("dropout");
  dropout.x1 = flashSoftmax.y;
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
  *storeVec1Res.y.strides = vec1ResRepeat;

  Data value("value", graph);
  value.attr.sched.exec_order = exec_order++;
  value.attr.sched.axis = bmm2ResAxis;
  value.y.dtype = ge::DT_FLOAT16;
  *value.y.axis = bmm2ResAxis;
  *value.y.repeats = {B, N, G, ONE, S2, D, ONE};
  *value.y.strides = {N*S2*D, S2*D, S2*D, ZERO, D, ONE, ZERO};

  Add bmm2("bmm2");
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

  Workspace addResOut("addResOut");
  addResOut.attr.sched.exec_order = exec_order++;
  addResOut.attr.sched.axis = bmm2ResAxis;
  addResOut.x = load2.y;
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
  div.x2 = flashSoftmax.y;
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

void FaAfterApiInfo(ge::AscGraph &graph) {
  auto query = graph.FindNode("query");
  query->attr.api.type = ge::ApiType::kAPITypeBuffer;
  query->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto key = graph.FindNode("key");
  key->attr.api.type = ge::ApiType::kAPITypeBuffer;
  key->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto bmm1 = graph.FindNode("bmm1");
  bmm1->attr.api.type = ge::ApiType::kAPITypeCompute;
  bmm1->attr.api.unit = ge::ComputeUnit::kUnitCube;

  auto load1 = graph.FindNode("load1");
  load1->attr.api.type = ge::ApiType::kAPITypeCompute;
  load1->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto pse = graph.FindNode("pse");
  pse->attr.api.type = ge::ApiType::kAPITypeBuffer;
  pse->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto loadPse = graph.FindNode("loadPse");
  loadPse->attr.api.type = ge::ApiType::kAPITypeBuffer;
  loadPse->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto castPse = graph.FindNode("castPse");
  castPse->attr.api.type = ge::ApiType::kAPITypeCompute;
  castPse->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto add1 = graph.FindNode("add1");
  add1->attr.api.type = ge::ApiType::kAPITypeCompute;
  add1->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto scaleValue = graph.FindNode("scaleValue");
  scaleValue->attr.api.type = ge::ApiType::kAPITypeBuffer;
  scaleValue->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto mul1 = graph.FindNode("mul1");
  mul1->attr.api.type = ge::ApiType::kAPITypeCompute;
  mul1->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto attenMask = graph.FindNode("attenMask");
  attenMask->attr.api.type = ge::ApiType::kAPITypeBuffer;
  attenMask->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto loadAttenMask = graph.FindNode("loadAttenMask");
  loadAttenMask->attr.api.type = ge::ApiType::kAPITypeCompute;
  loadAttenMask->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto select = graph.FindNode("select");
  select->attr.api.type = ge::ApiType::kAPITypeCompute;
  select->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto softmaxExp = graph.FindNode("softmaxExp");
  softmaxExp->attr.api.type = ge::ApiType::kAPITypeBuffer;
  softmaxExp->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto flashSoftmax = graph.FindNode("flashSoftmax");
  flashSoftmax->attr.api.type = ge::ApiType::kAPITypeCompute;
  flashSoftmax->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto softmaxSum = graph.FindNode("softmaxSum");
  softmaxSum->attr.api.type = ge::ApiType::kAPITypeBuffer;
  softmaxSum->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto storeSoftmaxMax = graph.FindNode("storeSoftmaxMax");
  storeSoftmaxMax->attr.api.type = ge::ApiType::kAPITypeCompute;
  storeSoftmaxMax->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto softmaxMax = graph.FindNode("softmaxMax");
  softmaxMax->attr.api.type = ge::ApiType::kAPITypeBuffer;
  softmaxMax->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto dropMask = graph.FindNode("dropMask");
  dropMask->attr.api.type = ge::ApiType::kAPITypeBuffer;
  dropMask->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto loadDropMask = graph.FindNode("loadDropMask");
  loadDropMask->attr.api.type = ge::ApiType::kAPITypeCompute;
  loadDropMask->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto dropout = graph.FindNode("dropout");
  dropout->attr.api.type = ge::ApiType::kAPITypeBuffer;
  dropout->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto castVec1Res = graph.FindNode("castVec1Res");
  castVec1Res->attr.api.type = ge::ApiType::kAPITypeCompute;
  castVec1Res->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto storeVec1Res = graph.FindNode("storeVec1Res");
  storeVec1Res->attr.api.type = ge::ApiType::kAPITypeCompute;
  storeVec1Res->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto value = graph.FindNode("value");
  value->attr.api.type = ge::ApiType::kAPITypeBuffer;
  value->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto bmm2 = graph.FindNode("bmm2");
  bmm2->attr.api.type = ge::ApiType::kAPITypeCompute;
  bmm2->attr.api.unit = ge::ComputeUnit::kUnitCube;

  auto load2 = graph.FindNode("load2");
  load2->attr.api.type = ge::ApiType::kAPITypeCompute;
  load2->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto addResOut = graph.FindNode("addResOut");
  addResOut->attr.api.type = ge::ApiType::kAPITypeBuffer;
  addResOut->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto loadAddResOut = graph.FindNode("loadAddResOut");
  loadAddResOut->attr.api.type = ge::ApiType::kAPITypeCompute;
  loadAddResOut->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto mulRes = graph.FindNode("mulRes");
  mulRes->attr.api.type = ge::ApiType::kAPITypeCompute;
  mulRes->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto addRes = graph.FindNode("addRes");
  addRes->attr.api.type = ge::ApiType::kAPITypeCompute;
  addRes->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto div = graph.FindNode("div");
  div->attr.api.type = ge::ApiType::kAPITypeCompute;
  div->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto castBmm2Res = graph.FindNode("castBmm2Res");
  castBmm2Res->attr.api.type = ge::ApiType::kAPITypeCompute;
  castBmm2Res->attr.api.unit = ge::ComputeUnit::kUnitVector;

  auto store = graph.FindNode("store");
  store->attr.api.type = ge::ApiType::kAPITypeCompute;
  store->attr.api.unit = ge::ComputeUnit::kUnitNone;

  auto buf = graph.FindNode("buf");
  buf->attr.api.type = ge::ApiType::kAPITypeBuffer;
  buf->attr.api.unit = ge::ComputeUnit::kUnitNone;
}

void FaAfterScheduler(ge::AscGraph &graph) {
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
  graph.FindAxis(s1tt.id)->allow_unaligned_tail = false;
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
  query->outputs[0].attr.vectorized_axis = {s1t.id, s2t.id, d};

  auto key = graph.FindNode("key");
  graph.ApplySplit(key, s1T.id, s1t.id);
  graph.ApplyMerge(key, mcAxis.id);
  graph.ApplySplit(key, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(key, s2T.id, s2t.id);
  graph.ApplyReorder(key, {mcAxisB.id, mcAxisb.id, s2T.id, s1t.id, d, s2t.id, bl});
  key->attr.sched.loop_axis = s2T.id;
  key->outputs[0].attr.vectorized_axis = {s1t.id, d, s2t.id};

  auto bmmReorderedAxis = {mcAxisB.id, mcAxisb.id, s2T.id, s1t.id, s2t.id, d, bl};
  auto vecReorderedAxis = {mcAxisB.id, mcAxisb.id, s2T.id, s1tT.id, s1tt.id, s2t.id, d, bl};

  auto bmm1 = graph.FindNode("bmm1");
  graph.ApplySplit(bmm1, s1T.id, s1t.id);
  graph.ApplyMerge(bmm1, mcAxis.id);
  graph.ApplySplit(bmm1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(bmm1, s2T.id, s2t.id);
  graph.ApplyReorder(bmm1, bmmReorderedAxis);
  bmm1->attr.sched.loop_axis = s2T.id;
  bmm1->outputs[0].attr.vectorized_axis = bmm1VectorizedAxis;

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, s1T.id, s1t.id);
  graph.ApplyMerge(load1, mcAxis.id);
  graph.ApplySplit(load1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(load1, s2T.id, s2t.id);
  graph.ApplySplit(load1, s1tT.id, s1tt.id);
  graph.ApplyReorder(load1, vecReorderedAxis);
  load1->attr.sched.loop_axis = s1tT.id;
  load1->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto loadPse = graph.FindNode("loadPse");
  graph.ApplySplit(loadPse, s1T.id, s1t.id);
  graph.ApplyMerge(loadPse, mcAxis.id);
  graph.ApplySplit(loadPse, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadPse, s2T.id, s2t.id);
  graph.ApplySplit(loadPse, s1tT.id, s1tt.id);
  graph.ApplyReorder(loadPse, vecReorderedAxis);
  loadPse->attr.sched.loop_axis = s1tT.id;
  loadPse->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto castPse = graph.FindNode("castPse");
  graph.ApplySplit(castPse, s1T.id, s1t.id);
  graph.ApplyMerge(castPse, mcAxis.id);
  graph.ApplySplit(castPse, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(castPse, s2T.id, s2t.id);
  graph.ApplySplit(castPse, s1tT.id, s1tt.id);
  graph.ApplyReorder(castPse, vecReorderedAxis);
  castPse->attr.sched.loop_axis = s1tT.id;
  castPse->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto add1 = graph.FindNode("add1");
  graph.ApplySplit(add1, s1T.id, s1t.id);
  graph.ApplyMerge(add1, mcAxis.id);
  graph.ApplySplit(add1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(add1, s2T.id, s2t.id);
  graph.ApplySplit(add1, s1tT.id, s1tt.id);
  graph.ApplyReorder(add1, vecReorderedAxis);
  add1->attr.sched.loop_axis = s1tT.id;
  add1->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto mul1 = graph.FindNode("mul1");
  graph.ApplySplit(mul1, s1T.id, s1t.id);
  graph.ApplyMerge(mul1, mcAxis.id);
  graph.ApplySplit(mul1, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(mul1, s2T.id, s2t.id);
  graph.ApplySplit(mul1, s1tT.id, s1tt.id);
  graph.ApplyReorder(mul1, vecReorderedAxis);
  mul1->attr.sched.loop_axis = s1tT.id;
  mul1->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto loadAttenMask = graph.FindNode("loadAttenMask");
  graph.ApplySplit(loadAttenMask, s1T.id, s1t.id);
  graph.ApplyMerge(loadAttenMask, mcAxis.id);
  graph.ApplySplit(loadAttenMask, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadAttenMask, s2T.id, s2t.id);
  graph.ApplySplit(loadAttenMask, s1tT.id, s1tt.id);
  graph.ApplyReorder(loadAttenMask, vecReorderedAxis);
  loadAttenMask->attr.sched.loop_axis = s1tT.id;
  loadAttenMask->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto select = graph.FindNode("select");
  graph.ApplySplit(select, s1T.id, s1t.id);
  graph.ApplyMerge(select, mcAxis.id);
  graph.ApplySplit(select, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(select, s2T.id, s2t.id);
  graph.ApplySplit(select, s1tT.id, s1tt.id);
  graph.ApplyReorder(select, vecReorderedAxis);
  select->attr.sched.loop_axis = s1tT.id;
  select->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto loadDropMask = graph.FindNode("loadDropMask");
  graph.ApplySplit(loadDropMask, s1T.id, s1t.id);
  graph.ApplyMerge(loadDropMask, mcAxis.id);
  graph.ApplySplit(loadDropMask, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadDropMask, s2T.id, s2t.id);
  graph.ApplySplit(loadDropMask, s1tT.id, s1tt.id);
  graph.ApplyReorder(loadDropMask, vecReorderedAxis);
  loadDropMask->attr.sched.loop_axis = s1tT.id;
  loadDropMask->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto dropout = graph.FindNode("dropout");
  graph.ApplySplit(dropout, s1T.id, s1t.id);
  graph.ApplyMerge(dropout, mcAxis.id);
  graph.ApplySplit(dropout, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(dropout, s2T.id, s2t.id);
  graph.ApplySplit(dropout, s1tT.id, s1tt.id);
  graph.ApplyReorder(dropout, vecReorderedAxis);
  dropout->attr.sched.loop_axis = s1tT.id;
  dropout->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto castVec1Res = graph.FindNode("castVec1Res");
  graph.ApplySplit(castVec1Res, s1T.id, s1t.id);
  graph.ApplyMerge(castVec1Res, mcAxis.id);
  graph.ApplySplit(castVec1Res, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(castVec1Res, s2T.id, s2t.id);
  graph.ApplySplit(castVec1Res, s1tT.id, s1tt.id);
  graph.ApplyReorder(castVec1Res, vecReorderedAxis);
  castVec1Res->attr.sched.loop_axis = s1tT.id;
  castVec1Res->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto storeVec1Res = graph.FindNode("storeVec1Res");
  graph.ApplySplit(storeVec1Res, s1T.id, s1t.id);
  graph.ApplyMerge(storeVec1Res, mcAxis.id);
  graph.ApplySplit(storeVec1Res, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(storeVec1Res, s2T.id, s2t.id);
  graph.ApplySplit(storeVec1Res, s1tT.id, s1tt.id);
  graph.ApplyReorder(storeVec1Res, vecReorderedAxis);
  storeVec1Res->attr.sched.loop_axis = s1tT.id;
  storeVec1Res->outputs[0].attr.vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d};

  auto softmaxExp = graph.FindNode("softmaxExp");
  graph.ApplySplit(softmaxExp, s1T.id, s1t.id);
  graph.ApplyMerge(softmaxExp, mcAxis.id);
  graph.ApplySplit(softmaxExp, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(softmaxExp, s2T.id, s2t.id);
  graph.ApplySplit(softmaxExp, s1tT.id, s1tt.id);
  graph.ApplyReorder(softmaxExp, vecReorderedAxis);
  softmaxExp->attr.sched.loop_axis = s1tT.id;
  softmaxExp->outputs[0].attr.vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d, bl};

  auto softmaxApiTmpBuf = graph.FindNode("softmaxApiTmpBuf");
  graph.ApplySplit(softmaxApiTmpBuf, s1T.id, s1t.id);
  graph.ApplyMerge(softmaxApiTmpBuf, mcAxis.id);
  graph.ApplySplit(softmaxApiTmpBuf, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(softmaxApiTmpBuf, s2T.id, s2t.id);
  graph.ApplySplit(softmaxApiTmpBuf, s1tT.id, s1tt.id);
  graph.ApplyReorder(softmaxApiTmpBuf, vecReorderedAxis);
  softmaxApiTmpBuf->attr.sched.loop_axis = s1tT.id;
  softmaxApiTmpBuf->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto flashSoftmax = graph.FindNode("flashSoftmax");
  graph.ApplySplit(flashSoftmax, s1T.id, s1t.id);
  graph.ApplyMerge(flashSoftmax, mcAxis.id);
  graph.ApplySplit(flashSoftmax, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(flashSoftmax, s2T.id, s2t.id);
  graph.ApplySplit(flashSoftmax, s1tT.id, s1tt.id);
  graph.ApplyReorder(flashSoftmax, vecReorderedAxis);
  flashSoftmax->attr.sched.loop_axis = s1tT.id;
  flashSoftmax->outputs[0].attr.vectorized_axis = vec1VectorizedAxis;

  auto storeSoftmaxMax = graph.FindNode("storeSoftmaxMax");
  graph.ApplySplit(storeSoftmaxMax, s1T.id, s1t.id);
  graph.ApplyMerge(storeSoftmaxMax, mcAxis.id);
  graph.ApplySplit(storeSoftmaxMax, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(storeSoftmaxMax, s2T.id, s2t.id);
  graph.ApplySplit(storeSoftmaxMax, s1tT.id, s1tt.id);
  graph.ApplyReorder(storeSoftmaxMax, vecReorderedAxis);
  storeSoftmaxMax->attr.sched.loop_axis = s1tT.id;
  storeSoftmaxMax->outputs[0].attr.vectorized_axis = {s1tT.id, s1tt.id, s2t.id, d, bl};

  auto value = graph.FindNode("value");
  graph.ApplySplit(value, s1T.id, s1t.id);
  graph.ApplyMerge(value, mcAxis.id);
  graph.ApplySplit(value, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(value, s2T.id, s2t.id);
  graph.ApplyReorder(value, bmmReorderedAxis);
  value->attr.sched.loop_axis = s2T.id;
  value->outputs[0].attr.vectorized_axis = {s1t.id, s2t.id, d};

  auto bmm2 = graph.FindNode("bmm2");
  graph.ApplySplit(bmm2, s1T.id, s1t.id);
  graph.ApplyMerge(bmm2, mcAxis.id);
  graph.ApplySplit(bmm2, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(bmm2, s2T.id, s2t.id);
  graph.ApplyReorder(bmm2, bmmReorderedAxis);
  bmm2->attr.sched.loop_axis = s2T.id;
  bmm2->outputs[0].attr.vectorized_axis = {s1t.id, s2t.id, d};

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
  load2->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;

  auto addResOut = graph.FindNode("addResOut");
  graph.ApplySplit(addResOut, s1T.id, s1t.id);
  graph.ApplyMerge(addResOut, mcAxis.id);
  graph.ApplySplit(addResOut, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(addResOut, s2T.id, s2t.id);
  graph.ApplySplit(addResOut, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(addResOut, vec2ReorderedAxis);
  addResOut->attr.sched.loop_axis = s1Vec2tT.id;
  addResOut->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;

  auto loadAddResOut = graph.FindNode("loadAddResOut");
  graph.ApplySplit(loadAddResOut, s1T.id, s1t.id);
  graph.ApplyMerge(loadAddResOut, mcAxis.id);
  graph.ApplySplit(loadAddResOut, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(loadAddResOut, s2T.id, s2t.id);
  graph.ApplySplit(loadAddResOut, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(loadAddResOut, vec2ReorderedAxis);
  loadAddResOut->attr.sched.loop_axis = s1Vec2tT.id;
  loadAddResOut->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;

  auto mulRes = graph.FindNode("mulRes");
  graph.ApplySplit(mulRes, s1T.id, s1t.id);
  graph.ApplyMerge(mulRes, mcAxis.id);
  graph.ApplySplit(mulRes, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(mulRes, s2T.id, s2t.id);
  graph.ApplySplit(mulRes, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(mulRes, vec2ReorderedAxis);
  mulRes->attr.sched.loop_axis = s1Vec2tT.id;
  mulRes->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;

  auto addRes = graph.FindNode("addRes");
  graph.ApplySplit(addRes, s1T.id, s1t.id);
  graph.ApplyMerge(addRes, mcAxis.id);
  graph.ApplySplit(addRes, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(addRes, s2T.id, s2t.id);
  graph.ApplySplit(addRes, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(addRes, vec2ReorderedAxis);
  addRes->attr.sched.loop_axis = s1Vec2tT.id;
  addRes->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;

  auto div = graph.FindNode("div");
  graph.ApplySplit(div, s1T.id, s1t.id);
  graph.ApplyMerge(div, mcAxis.id);
  graph.ApplySplit(div, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(div, s2T.id, s2t.id);
  graph.ApplySplit(div, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(div, vec2ReorderedAxis);
  div->attr.sched.loop_axis = s1Vec2tT.id;
  div->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;

  auto castBmm2Res = graph.FindNode("castBmm2Res");
  graph.ApplySplit(castBmm2Res, s1T.id, s1t.id);
  graph.ApplyMerge(castBmm2Res, mcAxis.id);
  graph.ApplySplit(castBmm2Res, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(castBmm2Res, s2T.id, s2t.id);
  graph.ApplySplit(castBmm2Res, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(castBmm2Res, vec2ReorderedAxis);
  castBmm2Res->attr.sched.loop_axis = s1Vec2tT.id;
  castBmm2Res->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;

  auto store = graph.FindNode("store");
  graph.ApplySplit(store, s1T.id, s1t.id);
  graph.ApplyMerge(store, mcAxis.id);
  graph.ApplySplit(store, mcAxisB.id, mcAxisb.id);
  graph.ApplySplit(store, s2T.id, s2t.id);
  graph.ApplySplit(store, s1Vec2tT.id, s1Vec2tt.id);
  graph.ApplyReorder(store, vec2ReorderedAxis);
  store->attr.sched.loop_axis = s1Vec2tT.id;
  store->outputs[0].attr.vectorized_axis = vec2VectorizedAxis;
}

void FaAfterQueBufAlloc(ge::AscGraph &graph) {
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
  query->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  query->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto key = graph.FindNode("key");
  key->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  key->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto value = graph.FindNode("value");
  value->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  value->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  auto bmm1 = graph.FindNode("bmm1");
  bmm1->outputs[0].attr.mem.tensor_id = tensorID++;
  bmm1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  bmm1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bmm1->outputs[0].attr.mem.position = ge::Position::kPositionGM;
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
  storeVec1Res->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  storeVec1Res->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  storeVec1Res->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  storeVec1Res->outputs[0].attr.buf.id = ge::kIdNone;
  storeVec1Res->outputs[0].attr.que.id = stage1ResQueue;
  storeVec1Res->outputs[0].attr.que.depth = 2;
  storeVec1Res->outputs[0].attr.que.buf_num = 2;
  storeVec1Res->outputs[0].attr.opt.ref_tensor = ge::kIdNone;

  auto bmm2 = graph.FindNode("bmm2");
  bmm2->outputs[0].attr.mem.tensor_id = tensorID++;
  bmm2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeQueue;
  bmm2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  bmm2->outputs[0].attr.mem.position = ge::Position::kPositionGM;
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
  addResOut->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  addResOut->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  addResOut->outputs[0].attr.mem.position = ge::Position::kPositionGM;
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
  store->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store->outputs[0].attr.mem.reuse_id = 0;
  store->outputs[0].attr.opt.ref_tensor = 0;

  auto storeSoftmaxMax = graph.FindNode("storeSoftmaxMax");
  storeSoftmaxMax->outputs[0].attr.mem.tensor_id = tensorID++;
  storeSoftmaxMax->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  storeSoftmaxMax->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  storeSoftmaxMax->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  storeSoftmaxMax->outputs[0].attr.mem.reuse_id = 0;
  storeSoftmaxMax->outputs[0].attr.opt.ref_tensor = 0;
}

void UnknownGraph(ge::AscGraph &graph) {
  uint16_t tensorID = 0;
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;

  auto X = ge::Symbol("X");
  auto Y = ge::Symbol("Y");

  auto x = graph.CreateAxis("x", X);
  auto y = graph.CreateAxis("y", Y);

  auto unknownResAxis = {x.id, y.id};
  std::initializer_list<Expr> unknownResRepeat = {X, Y};
  std::initializer_list<Expr> unknownResStride = {Y, ONE};

  int32_t exec_order = 0;
  Data input("input", graph);
  input.attr.sched.exec_order = exec_order++;
  input.attr.sched.axis = unknownResAxis;
  input.y.dtype = ge::DT_UINT32;
  *input.y.axis = unknownResAxis;
  *input.y.repeats = unknownResRepeat;
  *input.y.strides = unknownResStride;

  auto input_node = graph.FindNode("input");
  input_node->attr.api.type = ge::ApiType::kAPITypeBuffer;
  input_node->attr.api.unit = ge::ComputeUnit::kUnitNone;
  input_node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  input_node->outputs[0].attr.mem.position = ge::Position::kPositionGM;

  Load load("load");
  load.x = input.y;
  load.attr.sched.exec_order = exec_order++;
  load.attr.sched.axis = unknownResAxis;
  load.y.dtype = ge::DT_UINT32;
  *load.y.axis = unknownResAxis;
  *load.y.repeats = unknownResRepeat;
  *load.y.strides = unknownResStride;

  auto load_node = graph.FindNode("load");
  load_node->attr.api.type = ge::ApiType::kAPITypeCompute;
  load_node->attr.api.unit = ge::ComputeUnit::kUnitNone;
  load_node->attr.sched.loop_axis = x.id;
  load_node->outputs[0].attr.mem.tensor_id = tensorID++;
  load_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  load_node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load_node->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load_node->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  load_node->outputs[0].attr.buf.id = 0;
  load_node->outputs[0].attr.que.id = ge::kIdNone;
  load_node->outputs[0].attr.que.depth = ge::kIdNone;
  load_node->outputs[0].attr.que.buf_num = ge::kIdNone;
  load_node->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  load_node->outputs[0].attr.vectorized_axis = {y.id};

  Nop unknown("unknown");
  unknown.x = load.y;
  unknown.attr.sched.exec_order = exec_order++;
  unknown.attr.sched.axis = unknownResAxis;
  unknown.y.dtype = ge::DT_UINT32;
  *unknown.y.axis = unknownResAxis;
  *unknown.y.repeats = unknownResRepeat;
  *unknown.y.strides = unknownResStride;

  auto unknown_node = graph.FindNode("unknown");
  unknown_node->attr.api.type = ge::ApiType::kAPITypeBuffer;
  unknown_node->attr.api.unit = ge::ComputeUnit::kUnitVector;
  unknown_node->attr.sched.loop_axis = x.id;
  unknown_node->outputs[0].attr.mem.tensor_id = tensorID++;
  unknown_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  unknown_node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  unknown_node->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  unknown_node->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  unknown_node->outputs[0].attr.buf.id = 0;
  unknown_node->outputs[0].attr.que.id = ge::kIdNone;
  unknown_node->outputs[0].attr.que.depth = ge::kIdNone;
  unknown_node->outputs[0].attr.que.buf_num = ge::kIdNone;
  unknown_node->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  unknown_node->outputs[0].attr.vectorized_axis = {y.id};

  Store store("store");
  store.x = unknown.y;
  store.attr.sched.exec_order = exec_order++;
  store.attr.sched.axis = unknownResAxis;
  store.y.dtype = ge::DT_UINT32;
  *store.y.axis = unknownResAxis;
  *store.y.repeats = unknownResRepeat;
  *store.y.strides = unknownResStride;

  auto store_node = graph.FindNode("store");
  store_node->attr.api.type = ge::ApiType::kAPITypeCompute;
  store_node->attr.api.unit = ge::ComputeUnit::kUnitNone;
  store_node->attr.sched.loop_axis = x.id;
  store_node->outputs[0].attr.mem.tensor_id = tensorID++;
  store_node->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeGlobal;
  store_node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store_node->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store_node->outputs[0].attr.mem.reuse_id = 0;
  store_node->outputs[0].attr.opt.ref_tensor = 0;
  store_node->outputs[0].attr.vectorized_axis = {y.id};

  Output output("output");
  output.x = store.y;
  output.attr.sched.exec_order = exec_order++;

  auto output_node = graph.FindNode("output");
  output_node->attr.api.type = ge::ApiType::kAPITypeBuffer;
  output_node->attr.api.unit = ge::ComputeUnit::kUnitNone;
  output_node->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  output_node->outputs[0].attr.mem.position = ge::Position::kPositionGM;
}
}
