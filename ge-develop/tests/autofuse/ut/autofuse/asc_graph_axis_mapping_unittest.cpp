/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "can_fuse/backend/asc_graph_axis_mapping.h"
#include "graph/compute_graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/operator_factory.h"
#include "ascir_ops.h"
#include "utils/autofuse_attrs.h"
#include "utils/autofuse_utils.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "ascgen_log.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "post_process/post_process_util.h"

namespace ge {
class AscGraphAxisMappingTest : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

namespace {

std::shared_ptr<AscGraph> CreatBroadcastAddAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");
  const Expression F = graph.CreateSizeVar("F");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  auto f = graph.CreateAxis("F", F);

  ge::ascir_op::Data x1("x1_broadcast_add", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1.y.repeats = {A, B, C, D, E, ONE};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

  ge::ascir_op::Load x1Local("x1Local_broadcast_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, B, C, D, E, F};
  *x1Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Data x2("x2_broadcast_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2.y.repeats = {A, B, C, D, E, F};
  *x2.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Load x2Local("x2Local_broadcast_add");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.repeats = {A, B, C, D, E, F};
  *x2Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Add add("add_broadcast_add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.repeats = {A, B, C, D, E, F};
  *add.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Store x_out("x_out_broadcast_add");
  x_out.x = add.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_output1("x_output1");
  x_output1.x = x_out.y;
  x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_output1.attr.sched.loop_axis = c.id;
  x_output1.y.dtype = DT_FLOAT;
  *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_output1.y.repeats = {A, B, C, D, E};
  *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  auto x_out_node = graph.FindNode("x_output1");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

// 新反推方式，load和graph比较
std::shared_ptr<AscGraph> CreatBroadcastAddAscGraph2(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");
  const Expression F = graph.CreateSizeVar("F");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  auto f = graph.CreateAxis("F", F);

  ge::ascir_op::Data x1("x1_broadcast_add", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1.y.repeats = {A, B, C, D, E, ONE};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

  ge::ascir_op::Load x1Local("x1Local_broadcast_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, B, C, D, E, ONE};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

  ge::ascir_op::Data x2("x2_broadcast_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2.y.repeats = {A, B, C, D, E, F};
  *x2.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Load x2Local("x2Local_broadcast_add");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.repeats = {A, B, C, D, E, F};
  *x2Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Add add("add_broadcast_add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.repeats = {A, B, C, D, E, F};
  *add.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Store x_out("x_out_broadcast_add");
  x_out.x = add.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_output1("x_output1");
  x_output1.x = x_out.y;
  x_output1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_output1.attr.sched.loop_axis = c.id;
  x_output1.y.dtype = DT_FLOAT;
  *x_output1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_output1.y.repeats = {A, B, C, D, E};
  *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  auto x_out_node = graph.FindNode("x_output1");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static std::shared_ptr<ge::AscGraph> CreatAbsAscGraph(ge::AscGraph &graph, size_t out_num = 1, size_t in_num = 1) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  std::string data_name = "data" + graph.GetName();
  ge::ascir_op::Data x1(data_name.c_str(), graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("load");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  if (in_num == 2) {
    ge::ascir_op::Data x2(data_name.c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("load");
    x2Local.x = x1.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  }

  ge::ascir_op::Abs abs(graph.GetName().c_str());
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.repeats = {A, B, C, D, E};
  *abs.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("store");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("out");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  auto x_out_node = graph.FindNode("out");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  if (out_num == 2) {
    ge::ascir_op::Store x_store1("store1");
    x_store1.x = abs.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1("out1");
    x_out1.x = x_store1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node1 = graph.FindNode("out1");
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 1}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
  } else {
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
  }
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static std::shared_ptr<ge::AscGraph> CreatAddAscGraph2Input(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("Data1", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("load1");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("Data2", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("load2");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add("add_add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add.y.repeats = {A, B, C, D, E};
  *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("store");
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("out");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode("out");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static std::shared_ptr<ge::AscGraph> CreatAddAscGraph(ge::AscGraph &graph, size_t out_num = 1, size_t in_num = 1) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  std::string data_name = "data" + graph.GetName();
  ge::ascir_op::Data x1(data_name.c_str(), graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("load");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  if (in_num == 2) {
    ge::ascir_op::Data x2(data_name.c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("load");
    x2Local.x = x1.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  }

  ge::ascir_op::Add add(graph.GetName().c_str());
  add.x1 = x1Local.y;
  add.x2 = x1Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add.y.repeats = {A, B, C, D, E};
  *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("store");
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("out");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  auto x_out_node = graph.FindNode("out");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  if (out_num == 2) {
    ge::ascir_op::Store x_store1("store1");
    x_store1.x = add.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1("out1");
    x_out1.x = x_store1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node1 = graph.FindNode("out1");
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 1}};
    compute_graph->SetOutputSize(2U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
  } else {
    std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
    compute_graph->SetOutputSize(1U);
    compute_graph->SetGraphOutNodesInfo(output_nodes);
  }
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static std::shared_ptr<ge::AscGraph> CreatConcatAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("data1", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("load1");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("data2", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("load2");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x3("data3", graph);
  x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x3.attr.sched.loop_axis = c.id;
  *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3.y.repeats = {A, B, C, D, E};
  *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("load3");
  x3Local.x = x3.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Concat concat(graph.GetName().c_str());
  concat.x = {x1Local.y, x2Local.y, x3Local.y};
  concat.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *concat.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *concat.y.repeats = {A, B, C, D, E};
  *concat.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("store");
  x_store.x = concat.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("out");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode("out");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();

  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static std::shared_ptr<ge::AscGraph> CreatSplitAscGraphSame(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto TWO = Symbol(2);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1((graph.GetName() + "_data1").c_str(), graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, TWO * C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local((graph.GetName()+ "_load1").c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, TWO * C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store((graph.GetName() + "_store").c_str());
  x_store.x = x1Local.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out((graph.GetName() + "_out").c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode((graph.GetName() + "_out").c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();

  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatAddAscGraphFor3Repeats(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);

  ge::ascir_op::Data x1("x1_1", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id};
  *x1.y.repeats = {A, B, C};
  *x1.y.strides = {B * C, C, ONE};

  ge::ascir_op::Load x1Local("x1Local_2");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id};
  *x1Local.y.axis = {a.id, b.id, c.id};
  *x1Local.y.repeats = {A, B, C};
  *x1Local.y.strides = {B * C, C, ONE};

  ge::ascir_op::Data x2("x2_3", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id};
  *x2.y.repeats = {A, B, C};
  *x2.y.strides = {B * C, C, ONE};

  ge::ascir_op::Load x2Local("x2Local_4");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id};
  *x2Local.y.axis = {a.id, b.id, c.id};
  *x2Local.y.repeats = {A, B, C};
  *x2Local.y.strides = {B * C, C, ONE};

  ge::ascir_op::Add add("add_4");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id};
  *add.y.axis = {a.id, b.id, c.id};
  *add.y.repeats = {A, B, C};
  *add.y.strides = {B * C, C, ONE};

  ge::ascir_op::Store x_out("x_out_5");
  x_out.x = add.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id};
  *x_out.y.repeats = {A, B, C};
  *x_out.y.strides = {B * C, C, ONE};

  ge::ascir_op::Output x_output1("x_output1");
  x_output1.x = x_out.y;
  x_output1.attr.sched.axis = {a.id, b.id, c.id};
  x_output1.attr.sched.loop_axis = c.id;
  *x_output1.y.axis = {a.id, b.id, c.id};
  *x_output1.y.repeats = {A, B, C};
  *x_output1.y.strides = {B * C, C, ONE};
  auto x_out_node = graph.FindNode("x_output1");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

// 测试只有data load场景，不能循环融合
std::shared_ptr<AscGraph> CreatAddAscGraphOnlyDataLoad(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto d = graph.CreateAxis("D", D);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);

  ge::ascir_op::Data x1("x1_1", graph);
  x1.attr.sched.axis = {b.id, c.id, d.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {d.id, b.id, c.id};
  *x1.y.repeats = {D, B, C};
  *x1.y.strides = {B * C * D, C * D, D, ONE};
  //
  ge::ascir_op::Load xLocal1("xLocal1");
  xLocal1.x = x1.y;
  xLocal1.attr.sched.axis = {b.id, c.id, d.id};
  xLocal1.attr.sched.loop_axis = c.id;
  *xLocal1.y.axis = {d.id, b.id, c.id};
  *xLocal1.y.repeats = {D, B, C};
  *xLocal1.y.strides = {B * C * D, C * D, D, ONE};

  ge::ascir_op::Output x_output1("x_output1");
  x_output1.x = xLocal1.y;
  x_output1.attr.sched.axis = {b.id, c.id, d.id};
  x_output1.attr.sched.loop_axis = c.id;
  *x_output1.y.axis = {d.id, b.id, c.id};
  *x_output1.y.repeats = {D, B, C};
  *x_output1.y.strides = {B * C * D, C * D, D, ONE};
  auto x_out_node = graph.FindNode("x_output1");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatAddAscGraphTmp(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  const Expression F = graph.CreateSizeVar("F");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto f = graph.CreateAxis("F", F);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("x1_1", graph);
  x1.attr.sched.axis = {f.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {f.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {F, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("x1Local_2");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {f.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {f.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {F, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_3", graph);
  x2.attr.sched.axis = {f.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {f.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {F, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("x2Local_4");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {f.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {f.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {F, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add("add_4");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {f.id, b.id, c.id, d.id, e.id};
  *add.y.axis = {f.id, b.id, c.id, d.id, e.id};
  *add.y.repeats = {F, B, C, D, E};
  *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_out("x_out_5");
  x_out.x = add.y;
  x_out.attr.sched.axis = {f.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {f.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {F, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_output1("x_output1");
  x_output1.x = x_out.y;
  x_output1.attr.sched.axis = {f.id, b.id, c.id, d.id, e.id};
  x_output1.attr.sched.loop_axis = c.id;
  *x_output1.y.axis = {f.id, b.id, c.id, d.id, e.id};
  *x_output1.y.repeats = {F, B, C, D, E};
  *x_output1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  auto x_out_node = graph.FindNode("x_output1");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

// dim 2做了reduce
std::shared_ptr<AscGraph> CreatReduceAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("x1_reduce", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("x1Local_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Max reduce("reduce_reduce");
  reduce.x = x1Local.y;
  reduce.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *reduce.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *reduce.y.repeats = {A, ONE, C, D, E};
  *reduce.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_reduce");
  x_store.x = reduce.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, ONE, C, D, E};
  *x_store.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, ONE, C, D, E};
  *x_out.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  auto x_out_node = graph.FindNode("x_out_reduce");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatAbsBroadcastAfterReduceAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar(1);
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");
  const Expression F = graph.CreateSizeVar("F");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  auto f = graph.CreateAxis("F", F);

  ge::ascir_op::Data x1("x1_abs_after_reduce", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1.y.repeats = {A, ONE, C, D, E, ONE};
  *x1.y.strides = {B * C * D * E, ZERO, D * E, E, ONE, ZERO};

  ge::ascir_op::Load x1Local("x1Local_abs_after_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, ONE, C, D, E, F};
  *x1Local.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};

  ge::ascir_op::Abs abs("abs_abs_after_reduce");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.repeats = {A, ONE, C, D, E, F};
  *abs.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};

  ge::ascir_op::Store x_store("x_store_abs_after_reduce");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, ONE, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_out("x_out_abs_after_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, ONE, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, ZERO, D * E * F, E * F, F, ONE};
  auto x_out_node = graph.FindNode("x_out_abs_after_reduce");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatAbsBroadcastAfterReduceAscGraph2(ge::AscGraph &graph) {
  //  pre store and graph sched axis can not axis map
  //  pre store and cur data axis can axis map
  //  pre node is reduction, graph sched axis can't map, can't fuse.
  auto ONE = Symbol(1);
  auto TWO = Symbol(2);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("G");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);

  ge::ascir_op::Data x1("x1_abs_after_reduce", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, ONE, C, D, E};
  *x1.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Load x1Local("x1Local_abs_after_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {
      a.id, b.id, c.id, d.id, e.id,
  };
  *x1Local.y.repeats = {A, TWO, C, D, E};
  *x1Local.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Abs abs("abs_abs_after_reduce");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.repeats = {A, TWO, C, D, E};
  *abs.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_abs_after_reduce");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, TWO, C, D, E};
  *x_store.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_abs_after_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, TWO, C, D, E};
  *x_out.y.strides = {TWO * C * D * E, C * D * E, D * E, E, ONE};
  auto x_out_node = graph.FindNode("x_out_abs_after_reduce");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static Status SetAttrsGroup(const NodePtr &node) {
  auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  GE_ASSERT_NOTNULL(attr);

  ge::AscGraph add_graph(node->GetName().c_str());
  if (node->GetName() == "D") {
    attr->SetAscGraph(CreatAddAscGraph(add_graph, 2U));
  } else if ((node->GetName() == "AddN1") || (node->GetName() == "Assign1") || (node->GetName() == "AddN0")) {
    attr->SetAscGraph(CreatAddAscGraph(add_graph, 1U, 2U));
  } else if (node->GetName().find("Concat") != std::string::npos) {
    attr->SetAscGraph(CreatConcatAscGraph(add_graph), loop::FuseType::kConcat);
  } else if (node->GetName().find("SliceNodeSame") != std::string::npos) {
    attr->SetAscGraph(CreatSplitAscGraphSame(add_graph), loop::FuseType::kSliceSplit);
  } else if (node->GetName().find("Data") != std::string::npos) {
  } else {
    attr->SetAscGraph(CreatAddAscGraph(add_graph));
  }

  for (const auto out_anchor : node->GetAllOutDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(out_anchor);
    const auto node_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(node_desc);
    auto output_tensor_desc = node_desc->MutableOutputDesc(out_anchor->GetIdx());
    gert::SymbolShape symbol_shape1({Symbol(1), Symbol(2), Symbol(3), Symbol(4)});
    gert::SymbolShape symbol_shape2({Symbol(2), Symbol(2), Symbol(3), Symbol(4)});
    gert::SymbolShape symbol_shape;
    if (node_desc->GetName() == "A") {
      symbol_shape = symbol_shape2;
    } else {
      symbol_shape = symbol_shape1;
    }
    output_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
        symbol_shape;
  }
  for (const auto in_anchor : node->GetAllInDataAnchorsPtr()) {
    GE_ASSERT_NOTNULL(in_anchor);
    const auto node_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(node_desc);
    auto input_tensor_desc = node_desc->MutableInputDesc(in_anchor->GetIdx());

    const auto &peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      GELOGW("Node:%s in_anchor:%u peer_out_anchor is nullptr.", node->GetNamePtr(), in_anchor->GetIdx());
      continue;
    }
    const auto peer_node = peer_out_anchor->GetOwnerNodeBarePtr();
    GE_ASSERT_NOTNULL(peer_node);
    const auto peer_node_desc = peer_node->GetOpDesc();
    GE_ASSERT_NOTNULL(peer_node_desc);
    const auto output_tensor_desc = peer_node_desc->MutableOutputDesc(peer_out_anchor->GetIdx());
    GE_ASSERT_NOTNULL(output_tensor_desc);
    const auto attr_group = output_tensor_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    GE_ASSERT_NOTNULL(attr_group);

    input_tensor_desc->GetOrCreateAttrsGroup<SymbolicDescAttr>()->symbolic_tensor.MutableOriginSymbolShape() =
        attr_group->symbolic_tensor.MutableOriginSymbolShape();
  }
  // 给节点添加origin_input_names和origin_output_names
  if (node->GetType() == kAscBackendType) {
    std::vector<std::pair<std::string, int32_t>> origin_input_names;
    for (auto i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
      origin_input_names.emplace_back("origin_input" + std::to_string(i), i);
    }
    std::vector<std::pair<std::string, int32_t>> origin_output_names;
    for (auto i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
      origin_output_names.emplace_back("origin_output" + std::to_string(i), i);
    }
    GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_input_names_ = origin_input_names;
    GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_output_names_ = origin_output_names;
  }
  return SUCCESS;
}

/**
 *      netoutput1
 *         |
 *       shapeNo1
 *        |
 *      addnYes1
 *     /    \.
 *   /       \.
 * const1   const2
 */
ComputeGraphPtr BuildGraph1(const std::string node_type = "") {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1)
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1)
                   .OutNames({"y"}).Build("Data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(2).OutCnt(1).OutNames({"y"})
                   .Build("AddN1");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Shape");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  return graph;
}

/**
 *      netoutput1
 *         |
 *       shapeNo1  netoutput2
 *        |       /
 *      addnYes1
 *     /    \.
 *   /       \.
 * const1   const2
 */
ComputeGraphPtr BuildGraph1_1(const std::string node_type = "") {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1)
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1)
                   .OutNames({"y"}).Build("Data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(2).OutCnt(1).OutNames({"y"})
                   .Build("AddN1");
  auto shape1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Shape1");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape1)->NODE("NetOutput1", kNetOutputType));
    CHAIN(NODE(addn1)->EDGE(0, 1)->NODE("NetOutput2", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 6);
  return graph;
}

/**
 *      netoutput1
 *         |
 *       shapeNo1
 *        |
 *      addnYes1  abs
 *     /    \     /
 *   /       \   /
 * const2   const1
 */
ComputeGraphPtr BuildGraph1_2(const std::string node_type = "") {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1)
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1)
                   .OutNames({"y"}).Build("Data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(2).OutCnt(1).OutNames({"y"})
                   .Build("AddN1");
  auto shape1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Shape1");
  auto abs = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AbsN1");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(abs));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape1)->NODE("NetOutput1", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 6);
  return graph;
}

/**
 *
 *          netoutput1
 *        /   \      \.
 *     add1  assign1   \.
 *    /   \  /     \     \.
 * var1  var2    const1  var3
 */
ComputeGraphPtr BuildGraph2(const std::string node_type = "") {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data4");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(2).OutCnt(1).OutNames({"y"})
                   .Build("AddN1");
  auto assign1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                     .OutNames({"y"}).Build("Assign1");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(assign1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(data4)->EDGE(0, 2)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(assign1)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  return graph;
}

/**
 *    netoutput1
 *        |
 *     addnYes1
 *        |
 *      const1
 */
ComputeGraphPtr BuildGraph3(const std::string node_type = "") {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1)
                   .OutNames({"y"}).Build("Data1");
  auto add1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
                  .Build("Add1");

  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(add1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 3);
  return graph;
}

// 这里将decider中的FuseNode函数及其相关函数移植过来，用于生成一个类型是FusedAscBackend的节点
class MockAscBackendFusionDecider {
  public:
  Status UpdateNewNodeAttrInTest(const OpDescPtr op, const NodePtr &node1, const NodePtr &node2,
                                                  const NodeFuseInfo &node_fuse_info) {
  auto attr = GetOrCreateAutoFuseAttrs(op);
  GE_ASSERT_NOTNULL(attr);
  auto autofuse_attr1 = BackendUtils::GetNodeAutoFuseAttr(node1);
  GE_ASSERT_NOTNULL(autofuse_attr1);
  auto autofuse_attr2 = BackendUtils::GetNodeAutoFuseAttr(node2);
  GE_ASSERT_NOTNULL(autofuse_attr2);
  GetInterAttrs(attr).origin_nodes = GetInterAttrs(autofuse_attr1).origin_nodes;
  GetInterAttrs(attr).origin_nodes.insert(GetInterAttrs(attr).origin_nodes.end(),
                                          GetInterAttrs(autofuse_attr2).origin_nodes.begin(),
                                          GetInterAttrs(autofuse_attr2).origin_nodes.end());
  GetInterAttrs(autofuse_attr1).origin_nodes.clear();
  GetInterAttrs(autofuse_attr2).origin_nodes.clear();

  auto merge_output_buffer = [](const std::vector<ge::OutDataAnchor *> &vec, const std::vector<int32_t> &indices,
                                std::vector<ge::OutDataAnchor *> &result) -> Status {
    if (vec.empty()) {
      return SUCCESS;
    }
    for (const auto &index : indices) {
      if (index == -1) {
        continue;
      }
      GE_ASSERT_TRUE(static_cast<size_t>(index) < vec.size());
      result.push_back(vec[index]);
    }
    return SUCCESS;
  };
  GE_ASSERT_SUCCESS(merge_output_buffer(GetInterAttrs(autofuse_attr1).output_buffers,
                                        node_fuse_info.GetNode1OutputMap(), GetInterAttrs(attr).output_buffers));
  GE_ASSERT_SUCCESS(merge_output_buffer(GetInterAttrs(autofuse_attr2).output_buffers,
                                        node_fuse_info.GetNode2OutputMap(), GetInterAttrs(attr).output_buffers));

  auto &subgraph1_output_nodes = GetInterAttrs(autofuse_attr1).fused_subgraph_outputs;
  auto &subgraph2_output_nodes = GetInterAttrs(autofuse_attr2).fused_subgraph_outputs;
  GetInterAttrs(attr).fused_subgraph_outputs = subgraph1_output_nodes;
  GetInterAttrs(attr).fused_subgraph_outputs.insert(GetInterAttrs(attr).fused_subgraph_outputs.end(),
                                                    subgraph2_output_nodes.begin(), subgraph2_output_nodes.end());

  auto fuse_type =
      MergeFuseType(GetInterAttrs(autofuse_attr1).fuse_type, GetInterAttrs(autofuse_attr1).fuse_type);
  attr->SetAscGraph(BackendUtils::GetNodeFusedAscGraph(node1));
  GetInterAttrs(attr).fuse_type = fuse_type;

  return SUCCESS;
}

NodePtr FuseNodeTest(NodePtr node1, NodePtr node2, const ComputeGraphPtr merged_graph, const NodeFuseInfo &node_fuse_info) {
  auto graph = node1->GetOwnerComputeGraph();
  GE_ASSERT_EQ(graph, node2->GetOwnerComputeGraph());

  // 创建一个融合node
  auto node_name = "fused_graph_" + std::to_string(AutofuseUtils::GenUniqueNumber());
  auto op = ComGraphMakeShared<OpDesc>(node_name, ((merged_graph == nullptr) ? kFusedAscBackendType : kAscBackendType));
  GE_ASSERT_NOTNULL(op);
  GE_ASSERT_SUCCESS(UpdateNewNodeAttrInTest(op, node1, node2, node_fuse_info));
  auto new_node = graph->AddNode(op);
  GE_ASSERT_NOTNULL(new_node);
  GE_ASSERT_SUCCESS(new_node->SetOwnerComputeGraph(graph));

  GE_ASSERT_GRAPH_SUCCESS(ge::NodeUtils::AppendInputAnchor(new_node, node_fuse_info.GetNode2InputMap().size()));
  GE_ASSERT_GRAPH_SUCCESS(ge::NodeUtils::AppendOutputAnchor(new_node, node_fuse_info.GetNode2OutputMap().size()));

  GELOGI("replace data anchord from node %s(%s) to node %s(%s), input map %s, output map %s.", node1->GetNamePtr(),
         node1->GetType().c_str(), new_node->GetNamePtr(), new_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode1InputMap()).c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode1OutputMap()).c_str());
  GELOGI("replace data anchord from node %s(%s) to node %s(%s), input map %s, output map %s.", node2->GetNamePtr(),
         node2->GetType().c_str(), new_node->GetNamePtr(), new_node->GetType().c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode2InputMap()).c_str(),
         AutofuseUtils::VectorToStr(node_fuse_info.GetNode2OutputMap()).c_str());
  // 为新节点的每个输出描述创建属性组
  GE_ASSERT_SUCCESS(BackendUtils::CreateNewNodeOutputDescAttr(
      new_node, node1, node2, node_fuse_info.GetNode1OutputMap(), node_fuse_info.GetNode2OutputMap()));

  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(new_node, node1, node_fuse_info.GetNode1InputMap(),
                                                             node_fuse_info.GetNode1OutputMap()));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::ReplaceNodeDataAnchors(new_node, node2, node_fuse_info.GetNode2InputMap(),
                                                             node_fuse_info.GetNode2OutputMap()));
  GE_ASSERT_SUCCESS(BackendUtils::TryRemoveNodesCtrEdges(node1, node2));
  GE_ASSERT_SUCCESS(BackendUtils::TryRemoveNodesCtrEdges(node2, node1));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveInCtrlEdges(node1, new_node));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveInCtrlEdges(node2, new_node));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveOutCtrlEdges(node1, new_node));
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveOutCtrlEdges(node2, new_node));
  GELOGI("node %s(%s) and node %s(%s) fuse success.", node1->GetNamePtr(), node1->GetType().c_str(),
         node2->GetNamePtr(), node2->GetType().c_str());
  // 原图上清理掉融合后的节点
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::RemoveJustNodes(graph, {node1, node2}));
  NodeUtils::UnlinkAll(*node1);
  NodeUtils::UnlinkAll(*node2);
  return new_node;
}

ComputeGraphPtr CreateAscBackendNodeSubGraphTest(const NodePtr &node, uint32_t in_nums, uint32_t out_nums,
                                                 const std::vector<uint32_t> &node_output_index,
                                                 const std::vector<std::pair<ge::NodePtr, int32_t>> &pre_nodes) {
  const auto &sub_graph =
      ComGraphMakeShared<ComputeGraph>("FusedAscBackendNode_graph_" + std::to_string(AutofuseUtils::GenUniqueNumber()));
  GE_ASSERT_NOTNULL(sub_graph);
  GE_ASSERT_GRAPH_SUCCESS(GraphUtils::MoveNodeToGraph(node, *sub_graph));
  GE_ASSERT_SUCCESS(BackendUtils::CreateSubGraphInput(sub_graph, node, in_nums, pre_nodes));
  GE_ASSERT_SUCCESS(BackendUtils::CreateSubGraphOutput(sub_graph, node, out_nums, node_output_index));
  return sub_graph;
}
};

TEST_F(AscGraphAxisMappingTest, NodeFuseInfo_RollbackNode2InputMap_ok) {
  NodeFuseInfo fuse_info;
  fuse_info.RollbackNode2InputMap(0);
  vector<int32_t> expected_node2_input_map = {0};
  ASSERT_EQ(fuse_info.GetNode2InputMap(), expected_node2_input_map);
}

// 测试垂直融合场景获取节点融合信息
TEST_F(AscGraphAxisMappingTest, NodeFuseInfo_UpdateNodeFuseInfo_For_Vertical_Merge_Ok) {
  vector<int32_t> node1_input_map = {0, 1};
  vector<int32_t> node2_input_map = {-1, -1};
  vector<int32_t> node1_output_map = {};
  vector<int32_t> node2_output_map = {0};
  vector<std::pair<int32_t, int32_t>> same_input_map = {};
  ComputeGraphPtr compute_graph = BuildGraph1();
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);

  NodeFuseInfo fuse_info;
  ASSERT_EQ(fuse_info.UpdateNodeFuseInfo(addn1, shape1), SUCCESS);
  ASSERT_EQ(fuse_info.GetNode1InputMap(), node1_input_map);
  ASSERT_EQ(fuse_info.GetNode2InputMap(), node2_input_map);
  ASSERT_EQ(fuse_info.GetNode1OutputMap(), node1_output_map);
  ASSERT_EQ(fuse_info.GetNode2OutputMap(), node2_output_map);
  ASSERT_EQ(fuse_info.GetSameInputMap(), same_input_map);
}

// 测试水平融合场景获取节点融合信息
TEST_F(AscGraphAxisMappingTest, NodeFuseInfo_UpdateNodeFuseInfo_For_Horizontal_Merge_Ok) {
  vector<int32_t> node1_input_map = {0, 1};
  vector<int32_t> node2_input_map = {-1, -1, 1};
  vector<int32_t> node1_output_map = {0};
  vector<int32_t> node2_output_map = {-1, 0};
  vector<std::pair<int32_t, int32_t>> same_input_map = {{1, 0}};
  ComputeGraphPtr compute_graph = BuildGraph2("AscBc");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  auto add1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(add1, nullptr);
  auto assign1 = compute_graph->FindNode("Assign1");
  ASSERT_NE(assign1, nullptr);

  NodeFuseInfo fuse_info;
  ASSERT_EQ(fuse_info.UpdateNodeFuseInfo(add1, assign1), SUCCESS);
  ASSERT_EQ(fuse_info.GetNode1InputMap(), node1_input_map);
  ASSERT_EQ(fuse_info.GetNode2InputMap(), node2_input_map);
  ASSERT_EQ(fuse_info.GetNode1OutputMap(), node1_output_map);
  ASSERT_EQ(fuse_info.GetNode2OutputMap(), node2_output_map);
  ASSERT_EQ(fuse_info.GetSameInputMap(), same_input_map);
}

// 测试const node没有输入的场景下获取节点融合信息
TEST_F(AscGraphAxisMappingTest, NodeFuseInfo_UpdateNodeFuseInfo_For_Const_Node_Vertical_Merge_Ok) {
  vector<int32_t> node1_input_map = {};
  vector<int32_t> node2_input_map = {};
  vector<int32_t> node1_output_map = {};
  vector<int32_t> node2_output_map = {0};
  vector<std::pair<int32_t, int32_t>> same_input_map = {};
  ComputeGraphPtr compute_graph = BuildGraph3();
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 3);
  auto const1 = compute_graph->FindNode("Data1");
  ASSERT_NE(const1, nullptr);
  auto addn1 = compute_graph->FindNode("Add1");
  ASSERT_NE(addn1, nullptr);

  NodeFuseInfo fuse_info;
  ASSERT_EQ(fuse_info.UpdateNodeFuseInfo(const1, addn1), SUCCESS);
  ASSERT_EQ(fuse_info.GetNode1InputMap(), node1_input_map);
  ASSERT_EQ(fuse_info.GetNode2InputMap(), node2_input_map);
  ASSERT_EQ(fuse_info.GetNode1OutputMap(), node1_output_map);
  ASSERT_EQ(fuse_info.GetNode2OutputMap(), node2_output_map);
  ASSERT_EQ(fuse_info.GetSameInputMap(), same_input_map);
}

TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_CreateSubGraphAxisMapInfo_For_Horizontal_Merge_ok) {
  ComputeGraphPtr compute_graph = BuildGraph2("AscBc");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  auto add1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(add1, nullptr);
  auto assign1 = compute_graph->FindNode("Assign1");
  ASSERT_NE(assign1, nullptr);

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(add1, assign1), SUCCESS);
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(add1, assign1, node_fuse_info), SUCCESS);
  EXPECT_EQ(asc_graph_axis_map.FlushSubGraphAxisInfo(add1, asc_graph_axis_map.GetNode1AxisMap(), true),
    SUCCESS);
  EXPECT_EQ(asc_graph_axis_map.FlushSubGraphAxisInfo(assign1, asc_graph_axis_map.GetNode2AxisMap(), true),
    SUCCESS);
}

TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_CreateSubGraphAxisMapInfo_For_Vertical_Merge_ok1) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ge::AscGraph add_graph1("add1");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  ge::AscGraph add_graph2("add2");
  attr2->SetAscGraph(CreatAddAscGraph(add_graph2));

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(addn1, shape1), SUCCESS);
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(addn1, shape1, node_fuse_info), SUCCESS);
  EXPECT_EQ(asc_graph_axis_map.FlushSubGraphAxisInfo(addn1, asc_graph_axis_map.GetNode1AxisMap(), true),
    SUCCESS);
  EXPECT_EQ(asc_graph_axis_map.FlushSubGraphAxisInfo(shape1, asc_graph_axis_map.GetNode2AxisMap(), true),
    SUCCESS);
}

// 测试node1 repeats < node2 repeats场景
TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_CreateSubGraphAxisMapInfo_For_Vertical_Merge_ok2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraphFor3Repeats(add_graph1));
  ge::AscGraph concat_graph2("concat");
  attr2->SetAscGraph(CreatBroadcastAddAscGraph(concat_graph2));

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(addn1, shape1), SUCCESS);
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(addn1, shape1, node_fuse_info), SUCCESS);
}

// 测试包含FusedAscBackend node的场景,前置node是FusedAscBackend node，可以循环融合
TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_CreateSubGraphAxisMapInfo_failed) {
  MockAscBackendFusionDecider fusion_decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto const1 = compute_graph->FindNode("Data1");
  ASSERT_NE(const1, nullptr);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);

  auto op_desc1 = const1->GetOpDescBarePtr();
  auto op_desc2 = addn1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ge::AscGraph const_graph1("const");
  attr1->SetAscGraph(CreatAddAscGraphOnlyDataLoad(const_graph1));
  ge::AscGraph add_graph1("add");
  attr2->SetAscGraph(CreatAddAscGraph(add_graph1));

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(const1, addn1), SUCCESS);
  auto fused_node = fusion_decider.FuseNodeTest(const1, addn1, nullptr, node_fuse_info);
  ASSERT_EQ(fused_node->GetType(), kFusedAscBackendType);
  auto attr3 = BackendUtils::GetNodeAutoFuseAttr(fused_node);
  attr3->SetFuseComputeGraph(fusion_decider.CreateAscBackendNodeSubGraphTest(addn1, node_fuse_info.GetNode1InDataSize(),
    node_fuse_info.GetNode1OutNodeSize(), node_fuse_info.GetNode1OutputIndex(), node_fuse_info.GetNode1PreNodes()));

  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);
  auto op_desc4 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc4, nullptr);
  auto attr4 = GetOrCreateAutoFuseAttrs(op_desc4);
  ASSERT_NE(attr4, nullptr);
  ge::AscGraph shape_graph2("shape");
  attr4->SetAscGraph(CreatBroadcastAddAscGraph(shape_graph2));

  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(fused_node, shape1), SUCCESS);
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(fused_node, shape1, node_fuse_info), SUCCESS);
}

// 测试包含FusedAscBackend node的场景,前置node是const, merge const1 -> fused_graph_0, graph_attr2 is nullptr
TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_CreateSubGraphAxisMapInfo_failed2) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  ge::AscGraph concat_graph2("concat");
  attr2->SetAscGraph(CreatBroadcastAddAscGraph(concat_graph2), loop::FuseType::kConcat);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(addn1, shape1, nullptr);
  ASSERT_EQ(fused_node->GetType(), kFusedAscBackendType);

  AscGraphAxisMapping asc_graph_axis_map;
  auto const1 = compute_graph->FindNode("Data1");
  ASSERT_NE(const1, nullptr);
  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(const1, fused_node), SUCCESS);
  auto op_desc3 = const1->GetOpDescBarePtr();
  ASSERT_NE(op_desc3, nullptr);
  auto attr4 = GetOrCreateAutoFuseAttrs(op_desc3);
  ASSERT_NE(attr4, nullptr);
  ge::AscGraph add_graph2("add");
  attr4->SetAscGraph(CreatAddAscGraph(add_graph2));
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(const1, fused_node, node_fuse_info), FAILED);
  EXPECT_EQ(asc_graph_axis_map.FlushSubGraphAxisInfo(fused_node, asc_graph_axis_map.GetNode2AxisMap(), false),
    SUCCESS);
}

// 测试前置node是输出多引用, 后续node是带有broadcast，不能循环合并
TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_CreateSubGraphAxisMapInfo_failed3) {
  ComputeGraphPtr compute_graph = BuildGraph1_1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 6);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  ge::AscGraph broadcast_graph2("broadcast");
  attr2->SetAscGraph(CreatBroadcastAddAscGraph2(broadcast_graph2));

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(addn1, shape1, nullptr);
  ASSERT_EQ(fused_node, nullptr);
}

// 测试垂直融合node1两个输出data anchor，其中第0个data anchor没有连接下一个node,循环合并node1 to node2 output map映射为(1，0)
TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_Node1ToNode2LinkMap) {
  ComputeGraphPtr compute_graph = BuildGraph1_2("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 6);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape1");
  ASSERT_NE(shape1, nullptr);
  auto abs = compute_graph->FindNode("AbsN1");
  ASSERT_NE(abs, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  auto op_desc3 = abs->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  ASSERT_NE(op_desc3, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  auto attr3 = GetOrCreateAutoFuseAttrs(op_desc3);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ASSERT_NE(attr3, nullptr);
  ge::AscGraph add_graph1("add");
  attr1->SetAscGraph(CreatAddAscGraph2Input(add_graph1));
  ge::AscGraph broadcast_graph2("broadcast");
  attr2->SetAscGraph(CreatBroadcastAddAscGraph2(broadcast_graph2));
  ge::AscGraph broadcast_graph3("abs_no_edge");
  attr3->SetAscGraph(CreatAbsAscGraph(broadcast_graph3));

  AscBackendFusionDecider decider;
  auto fused_node0 = decider.Fuse(abs, addn1, nullptr);
  ASSERT_NE(fused_node0, nullptr);
  auto fused_node1 = decider.Fuse(fused_node0, shape1, nullptr);
  ASSERT_NE(fused_node1, nullptr);

  auto fused_desc1 = fused_node1->GetOpDescBarePtr();
  ASSERT_NE(fused_desc1, nullptr);
  auto fused_attr1 = fused_desc1->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr1, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr1->GetAscGraph())), nullptr);
  for (auto node : AscGraphUtils::GetComputeGraph(*(fused_attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == "Abs") {
      // 校验abs节点后面是store
      std::vector<NodePtr> next_node;
      asc_adapt::GetPeerInNodes(node, next_node, 0);
      ASSERT_EQ(next_node[0]->GetName(), "store");
    } else if (node->GetName() == "add_add") {
      // 校验add节点后面是add
      std::vector<NodePtr> next_node;
      asc_adapt::GetPeerInNodes(node, next_node, 0);
      ASSERT_EQ(next_node[0]->GetName(), "add_broadcast_add");
    }
  }
}

TEST_F(AscGraphAxisMappingTest, AscGraphAxisMapping_CreateSubGraphAxisMapInfo_For_Vertical_Merge_fail1) {
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ge::AscGraph add_graph1("add1");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  ge::AscGraph add_graph2("add2");
  attr2->SetAscGraph(CreatAddAscGraphTmp(add_graph2));

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(addn1, shape1), SUCCESS);
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(addn1, shape1, node_fuse_info), FAILED);
}

TEST_F(AscGraphAxisMappingTest, AscBackendFusionDecider_CreateSubGraphAxisMapInfo_For_Reduce_Vertical_Merge_Ok) {
  AscBackendFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  ASSERT_NE(compute_graph, nullptr);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph reduce_graph1("reduce");
  ge::AscGraph abs_graph2("abs");
  attr1->SetAscGraph(CreatReduceAscGraph(reduce_graph1), loop::FuseType::kReduction);
  attr2->SetAscGraph(CreatAbsBroadcastAfterReduceAscGraph(abs_graph2));

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(addn1, shape1), SUCCESS);
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(addn1, shape1, node_fuse_info), SUCCESS);
}

TEST_F(AscGraphAxisMappingTest, AscBackendFusionDecider_CreateSubGraphAxisMapInfo_For_Reduce_Vertical_Merge_Fail) {
  AscBackendFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  ASSERT_NE(compute_graph, nullptr);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("AddN1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("Shape");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph reduce_graph1("reduce");
  ge::AscGraph abs_graph2("abs");
  attr1->SetAscGraph(CreatReduceAscGraph(reduce_graph1), loop::FuseType::kReduction);
  attr2->SetAscGraph(CreatAbsBroadcastAfterReduceAscGraph2(abs_graph2));

  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(addn1, shape1), SUCCESS);
  AscGraphAxisMapping asc_graph_axis_map;
  EXPECT_EQ(asc_graph_axis_map.CreateSubGraphAxisMapInfo(addn1, shape1, node_fuse_info), FAILED);
}
TEST_F(AscGraphAxisMappingTest, AscBackendFusionDecider_Slice_Horizontal_Has_Same_Load_Fuse) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto slice1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNodeSame1");
  auto slice2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNodeSame2");
  auto slice3 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("SliceNodeSame3"); 
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("ConcatNode");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice1));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice2));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(slice3));
    CHAIN(NODE(slice1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(slice2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(slice3)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(concat)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  auto slice1_node = graph->FindNode("SliceNodeSame1");
  ASSERT_NE(slice1_node, nullptr);
  auto slice2_node = graph->FindNode("SliceNodeSame2");
  ASSERT_NE(slice2_node, nullptr);
  NodeFuseInfo node_fuse_info;
  ASSERT_EQ(node_fuse_info.UpdateNodeFuseInfo(slice1_node, slice2_node), SUCCESS);
}
}  // namespace
}  // namespace ge
