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
 * for ndB in range(ceiling((ND / (ndb_size))))
 * for ndbT in range(ceiling((ndb_size / (ndbt_size))))
 * input_output_0 [ndbt_size]  = input()
 * load1_output_0 [ndbt_size] : BUF0_ub_size = load1(input_output_0)
 * mul_tmp_value_output_0 [ndbt_size] : BUF1_ub_size = mul_tmp_value()
 * div_tmp_value_output_0 [ndbt_size] : BUF2_ub_size = div_tmp_value()
 * exp_tmp_value_output_0 [ndbt_size] : BUF3_ub_size = exp_tmp_value()
 * vf_call1_output_0 [ndbt_size] : BUF1_ub_size, vf_call1_output_1 [ndbt_size] : BUF2_ub_size = add1(input_output_0, mul_tmp_value_output_0, div_tmp_value_output_0, exp_tmp_value_output_0)
 * vf_call2_output_0 [ndbt_size] : BUF0_ub_size = add2(vf_call1_output_0, vf_call1_output_1, input_output_0)
 * store_vf_call_output_0 [ndbt_size] : BUF4_hbm_size = store_vf_call(vf_call2_output_0)
 ********************************************************************/
void BuildOriginGraph(ge::AscGraph &graph) {
  auto ONE = ge::sym::kSymbolOne;
  auto ZERO = ge::sym::kSymbolZero;
  // 定义轴大小
  auto ND = ge::Symbol("ND");
  // 定义轴
  auto nd = graph.CreateAxis("nd", ND);
  std::initializer_list<ge::AxisId> elementwise_axis = {nd.id};
  std::initializer_list<ge::Expression> elementwise_repeat = {ND};
  std::initializer_list<ge::Expression> elementwise_stride = {ONE};
  // 定义节点
  int32_t exec_order = 0;
  Data input("input", graph);
  input.attr.sched.exec_order = exec_order++;
  input.attr.sched.axis = elementwise_axis;
  input.y.dtype = ge::DT_FLOAT;
  *input.y.axis = elementwise_axis;
  *input.y.repeats = elementwise_repeat;
  *input.y.strides = elementwise_stride;
  input.ir_attr.SetIndex(0);

  Load load1("load1");
  load1.attr.sched.exec_order = exec_order++;
  load1.attr.sched.axis = elementwise_axis;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = elementwise_axis;
  *load1.y.repeats = elementwise_repeat;
  *load1.y.strides = elementwise_stride;
  load1.x = input.y;

  Data mul_tmp_value("mul_tmp_value", graph);
  mul_tmp_value.attr.sched.exec_order = exec_order++;
  mul_tmp_value.attr.sched.axis = elementwise_axis;
  mul_tmp_value.y.dtype = ge::DT_FLOAT;
  *mul_tmp_value.y.axis = elementwise_axis;
  *mul_tmp_value.y.repeats = elementwise_repeat;
  *mul_tmp_value.y.strides = elementwise_stride;

  Data div_tmp_value("div_tmp_value", graph);
  div_tmp_value.attr.sched.exec_order = exec_order++;
  div_tmp_value.attr.sched.axis = elementwise_axis;
  div_tmp_value.y.dtype = ge::DT_FLOAT;
  *div_tmp_value.y.axis = elementwise_axis;
  *div_tmp_value.y.repeats = elementwise_repeat;
  *div_tmp_value.y.strides = elementwise_stride;

  Data exp_tmp_value("exp_tmp_value", graph);
  exp_tmp_value.attr.sched.exec_order = exec_order++;
  exp_tmp_value.attr.sched.axis = elementwise_axis;
  exp_tmp_value.y.dtype = ge::DT_FLOAT;
  *exp_tmp_value.y.axis = elementwise_axis;
  *exp_tmp_value.y.repeats = elementwise_repeat;
  *exp_tmp_value.y.strides = elementwise_stride;

  Add add1("add1");
  add1.attr.sched.exec_order = exec_order++;
  add1.attr.sched.axis = elementwise_axis;
  add1.y.dtype = ge::DT_FLOAT;
  *add1.y.axis = elementwise_axis;
  *add1.y.repeats = elementwise_repeat;
  *add1.y.strides = elementwise_stride;

  add1.x1 = input.y;
  add1.x2 = mul_tmp_value.y;

  Add add2("add2");
  add2.attr.sched.exec_order = exec_order++;
  add2.attr.sched.axis = elementwise_axis;
  add2.y.dtype = ge::DT_FLOAT;
  *add2.y.axis = elementwise_axis;
  *add2.y.repeats = elementwise_repeat;
  *add2.y.strides = elementwise_stride;

  add2.x1 = add1.y;
  add2.x2 = add1.y;
  *add2.y.axis = elementwise_axis;
  *add2.y.repeats = elementwise_repeat;
  *add2.y.strides = elementwise_stride;

  Store store_vf_call("store_vf_call");
  store_vf_call.attr.sched.exec_order = exec_order++;
  store_vf_call.attr.sched.axis = elementwise_axis;
  store_vf_call.y.dtype = ge::DT_FLOAT;
  *store_vf_call.y.axis = elementwise_axis;
  *store_vf_call.y.repeats = elementwise_repeat;
  *store_vf_call.y.strides = elementwise_stride;
  store_vf_call.x = add2.y;

  Output vf_call_output("vf_call_output");
  vf_call_output.x = store_vf_call.y;
  vf_call_output.attr.sched.exec_order = exec_order++;
  vf_call_output.y.dtype = ge::DT_FLOAT;
  *vf_call_output.y.axis = elementwise_axis;
  *vf_call_output.y.repeats = elementwise_repeat;
  *vf_call_output.y.strides = elementwise_stride;
}
/************************************************
 * 轴调度信息表达
 * original axes : [nd,]
 * nd : split into [ndB,ndb,]
 * ndb : split into [ndbT,ndbt,]
*************************************************/
void AddScheInfoToGraph(ge::AscGraph &graph) {
  auto nd_axis = graph.GetAllAxis()[0]->id;
  // split by vector core
  auto [core_num, core_element_size] = graph.BlockSplit(nd_axis);
  // split by UB size
  auto [UB_loop_num, UB_element_size] = graph.TileSplit(core_element_size->id);
  // to aligned
  vector<ge::AxisId> elementwise_vectorized_axis{UB_element_size->id};
  auto input = graph.FindNode("input");
  graph.ApplySplit(input, core_num->id, core_element_size->id);
  graph.ApplySplit(input, UB_loop_num->id, UB_element_size->id);
  // 每个Core内会调用多少次API
  input->attr.sched.loop_axis = UB_loop_num->id;
  // 每个API会调用多少个元素，也会拿来算API输出输入占用多少空间
  input->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, core_num->id, core_element_size->id);
  graph.ApplySplit(load1, UB_loop_num->id, UB_element_size->id);
  load1->attr.sched.loop_axis = UB_loop_num->id;
  load1->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;

  auto mul_tmp_value = graph.FindNode("mul_tmp_value");
  graph.ApplySplit(mul_tmp_value, core_num->id, core_element_size->id);
  graph.ApplySplit(mul_tmp_value, UB_loop_num->id, UB_element_size->id);
  mul_tmp_value->attr.sched.loop_axis = UB_loop_num->id;
  mul_tmp_value->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;

  auto div_tmp_value = graph.FindNode("div_tmp_value");
  graph.ApplySplit(div_tmp_value, core_num->id, core_element_size->id);
  graph.ApplySplit(div_tmp_value, UB_loop_num->id, UB_element_size->id);
  div_tmp_value->attr.sched.loop_axis = UB_loop_num->id;
  div_tmp_value->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;

  auto exp_tmp_value = graph.FindNode("exp_tmp_value");
  graph.ApplySplit(exp_tmp_value, core_num->id, core_element_size->id);
  graph.ApplySplit(exp_tmp_value, UB_loop_num->id, UB_element_size->id);
  exp_tmp_value->attr.sched.loop_axis = UB_loop_num->id;
  exp_tmp_value->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;

  auto add1 = graph.FindNode("add1");
  graph.ApplySplit(add1, core_num->id, core_element_size->id);
  graph.ApplySplit(add1, UB_loop_num->id, UB_element_size->id);
  add1->attr.sched.loop_axis = UB_loop_num->id;
  add1->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;

  auto add2 = graph.FindNode("add2");
  graph.ApplySplit(add2, core_num->id, core_element_size->id);
  graph.ApplySplit(add2, UB_loop_num->id, UB_element_size->id);
  add2->attr.sched.loop_axis = UB_loop_num->id;
  add2->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;

  auto store_vf_call = graph.FindNode("store_vf_call");
  graph.ApplySplit(store_vf_call, core_num->id, core_element_size->id);
  graph.ApplySplit(store_vf_call, UB_loop_num->id, UB_element_size->id);
  store_vf_call->attr.sched.loop_axis = UB_loop_num->id;
  store_vf_call->outputs[0].attr.vectorized_axis = elementwise_vectorized_axis;
}

/********************************************************************
 * 内存信息表达
 * BUF0 : [load1_output_0, vf_call2_output_0]
 * BUF1 : [mul_tmp_value_output_0, vf_call1_output_0]
 * BUF2 : [div_tmp_value_output_0, vf_call1_output_1]
 * BUF3 : [exp_tmp_value_output_0]
 * BUF4 : [store_vf_call_output_0]
 * ********************************************************************/
void AddBuffInfoToGraph(ge::AscGraph &graph) {
  int32_t tensor_id = 0;
  int32_t buf_id = 0;
  int32_t load1_buffer_id = buf_id++;
  int32_t mul_tmp_buffer_id = buf_id++;
  int32_t div_tmp_buffer_id = buf_id++;
  int32_t exp_tmp_buffer_id = buf_id++;
  int32_t store_buffer_id = buf_id++;
  // node1
  auto input = graph.FindNode("input");
  input->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  input->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  // node2
  auto load1 = graph.FindNode("load1");
  load1->outputs[0].attr.mem.tensor_id = tensor_id++;
  auto load1_tensor_id = load1->outputs[0].attr.mem.tensor_id;
  load1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  load1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  load1->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  load1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  load1->outputs[0].attr.buf.id = load1_buffer_id;
  load1->outputs[0].attr.que.id = ge::kIdNone;
  load1->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  // node3
  auto mul_tmp_value = graph.FindNode("mul_tmp_value");
  mul_tmp_value->outputs[0].attr.mem.tensor_id = tensor_id++;
  auto mul_tmp_out_tensor_id = mul_tmp_value->outputs[0].attr.mem.tensor_id;
  mul_tmp_value->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  mul_tmp_value->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  mul_tmp_value->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  mul_tmp_value->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  mul_tmp_value->outputs[0].attr.buf.id = mul_tmp_buffer_id;
  mul_tmp_value->outputs[0].attr.que.id = ge::kIdNone;
  mul_tmp_value->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  // node4
  auto div_tmp_value = graph.FindNode("div_tmp_value");
  div_tmp_value->outputs[0].attr.mem.tensor_id = tensor_id++;
  auto div_tmp_out_tensor_id = div_tmp_value->outputs[0].attr.mem.tensor_id;
  div_tmp_value->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  div_tmp_value->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  div_tmp_value->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  div_tmp_value->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  div_tmp_value->outputs[0].attr.buf.id = div_tmp_buffer_id;
  div_tmp_value->outputs[0].attr.que.id = ge::kIdNone;
  div_tmp_value->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  // node5
  auto exp_tmp_value = graph.FindNode("exp_tmp_value");
  exp_tmp_value->outputs[0].attr.mem.tensor_id = tensor_id++;
  auto exp_tmp_out_tensor_id = exp_tmp_value->outputs[0].attr.mem.tensor_id;
  exp_tmp_value->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  exp_tmp_value->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  exp_tmp_value->outputs[0].attr.mem.position = ge::Position::kPositionVecIn;
  exp_tmp_value->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  exp_tmp_value->outputs[0].attr.buf.id = exp_tmp_buffer_id;
  exp_tmp_value->outputs[0].attr.que.id = ge::kIdNone;
  exp_tmp_value->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
  // node6 add1
  auto add1 = graph.FindNode("add1");
  add1->outputs[0].attr.mem.tensor_id = tensor_id++;
  add1->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  add1->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  add1->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  add1->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  add1->outputs[0].attr.buf.id = mul_tmp_buffer_id;
  add1->outputs[0].attr.que.id = ge::kIdNone;
  add1->outputs[0].attr.opt.ref_tensor = mul_tmp_buffer_id;
  // node7, add2
  auto add2 = graph.FindNode("add2");
  add2->outputs[0].attr.mem.tensor_id = tensor_id++;
  add2->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  add2->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareUB;
  add2->outputs[0].attr.mem.position = ge::Position::kPositionVecOut;
  add2->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  add2->outputs[0].attr.buf.id = load1_buffer_id;
  add2->outputs[0].attr.que.id = ge::kIdNone;
  add2->outputs[0].attr.opt.ref_tensor = load1_buffer_id;
  // node8
  auto store_vf_call_res = graph.FindNode("store_vf_call");
  store_vf_call_res->outputs[0].attr.mem.tensor_id = tensor_id++;
  store_vf_call_res->outputs[0].attr.mem.alloc_type = ge::AllocType::kAllocTypeBuffer;
  store_vf_call_res->outputs[0].attr.mem.hardware = ge::MemHardware::kMemHardwareGM;
  store_vf_call_res->outputs[0].attr.mem.position = ge::Position::kPositionGM;
  store_vf_call_res->outputs[0].attr.mem.reuse_id = ge::kIdNone;
  store_vf_call_res->outputs[0].attr.buf.id = store_buffer_id;
  store_vf_call_res->outputs[0].attr.que.id = ge::kIdNone;
  store_vf_call_res->outputs[0].attr.opt.ref_tensor = ge::kIdNone;
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
