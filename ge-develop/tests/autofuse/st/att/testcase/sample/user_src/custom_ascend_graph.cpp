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
// 负责数据流图的构建
void BuildOriginGraph(ge::AscGraph &graph) {
  /* 以下仅作为示例参考代码 */
  auto ONE = ge::expression::Expression::CreateExpression(1);
  // Expression::CreateExpression:创建一个符号，用于表达原始轴的大小，注意原始轴的大小是tiling函数运行时TilingContext的输入
  auto ND = ge::expression::Expression::CreateExpression("ND"); // 创建一个变量符号ND
  // AscGraph::CreateAxis:创建一根轴
  auto nd = graph.CreateAxis("nd", ND);
  std::initializer_list<ge::AxisId> elementwise_axis = {nd.id};
  std::initializer_list<ge::expression::ExpressionPtr> elementwise_repeat = {ND};
  std::initializer_list<ge::expression::ExpressionPtr> elementwise_stride = {ONE};
  // 定义输入节点
  int32_t exec_order = 0;
  Data input("input", graph); // 定义Data输入参数
  input.attr.sched.exec_order = exec_order++; // 设置调度顺序
  input.attr.sched.axis = elementwise_axis; // 设置调度轴
  input.y.dtype = ge::DT_FLOAT; // 设置类型
  *input.y.axis = elementwise_axis; // 设置节点轴信息
  *input.y.repeats = elementwise_repeat;
  *input.y.strides = elementwise_stride;
  // 定义加载节点
  Load load1("load1");
  load1.attr.sched.exec_order = exec_order++;
  load1.attr.sched.axis = elementwise_axis;
  load1.y.dtype = ge::DT_FLOAT;
  *load1.y.axis = elementwise_axis;
  *load1.y.repeats = elementwise_repeat;
  *load1.y.strides = elementwise_stride;
  load1.x = input.y;
  /* 以上仅作为示例参考代码 */
}

// Tiling策略表达的构建
void AddScheInfoToGraph(ge::AscGraph &graph) {
  /* 以下仅作为示例参考代码 */
  auto nd_axis = graph.GetAllAxis()[0]->id;
  // 按照Block进行切分
  auto [core_num, core_element_size] = graph.BlockSplit(nd_axis);
  // 按照Tile进行切分
  auto [UB_loop_num, UB_element_size] = graph.TileSplit(core_element_size->id);
  vector<ge::AxisId> elementwise_vectorized_axis{UB_element_size->id};
  auto input = graph.FindNode("input");
  // 对节点应用切分
  graph.ApplySplit(input, core_num->id, core_element_size->id);
  graph.ApplySplit(input, UB_loop_num->id, UB_element_size->id);
  // 每个Core内会调用多少次API
  input->attr.sched.loop_axis = UB_loop_num->id;
  // 每个API会调用多少个元素，也会拿来算API输出输入占用多少空间
  input->outputs[0].vectorized_axis = elementwise_vectorized_axis;

  auto load1 = graph.FindNode("load1");
  graph.ApplySplit(load1, core_num->id, core_element_size->id);
  graph.ApplySplit(load1, UB_loop_num->id, UB_element_size->id);
  load1->attr.sched.loop_axis = UB_loop_num->id;
  load1->outputs[0].vectorized_axis = elementwise_vectorized_axis;
  /* 以上仅作为示例参考代码 */
}

// 内存分配的表达
void AddBuffInfoToGraph(ge::AscGraph &graph) {
  /* 以下仅作为示例参考代码 */
  int32_t tensor_id = 0;
  int32_t buf_id = 0;
  int32_t load1_buffer_id = buf_id++;
  // node1
  auto input = graph.FindNode("input");
  input->outputs[0].mem.hardware = ge::MemHardware::MEM_HARDWARE_GM;
  input->outputs[0].mem.position = ge::Position::POSITION_GM;
  // node2
  auto load1 = graph.FindNode("load1");
  load1->outputs[0].mem.tensor_id = tensor_id++;
  auto load1_tensor_id = load1->outputs[0].mem.tensor_id;
  load1->outputs[0].mem.alloc_type = ge::AllocType::ALLOC_TYPE_BUFFER;
  load1->outputs[0].mem.hardware = ge::MemHardware::MEM_HARDWARE_UB;
  load1->outputs[0].mem.position = ge::Position::POSITION_VECIN;
  load1->outputs[0].mem.reuse_id = ge::kIdNone;
  load1->outputs[0].buf.id = load1_buffer_id;
  load1->outputs[0].que.id = ge::kIdNone;
  load1->outputs[0].opt.ref_tensor = ge::kIdNone;
  /* 以上仅作为示例参考代码 */
}

// 调用Ascir构图/构造调度信息/添加内存信息，并添加AscGraph返回
ge::Status GenerateAscGraphs(std::vector<ge::AscGraph> &graphs) {
  /* 以下仅作为示例参考代码 */
  std::map<std::string, std::string> options;
  ge::AscGraph graph("graph");
  BuildOriginGraph(graph);
  AddScheInfoToGraph(graph);
  AddBuffInfoToGraph(graph);
  graph.SetTilingKey(0);
  // to set graphs
  (void)graphs;
  /* 以上仅作为示例参考代码 */
  return ge::SUCCESS;
}

// 配置Option
// 当前支持的Option可以参考头文件att/gen_tiling_impl.h
void GeneratorAttOptions(std::map<std::string, std::string> &options) {
  (void)options;
}