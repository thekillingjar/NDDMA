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
#include "graph/graph.h"
#include "graph/node.h"
#include "ascir_ops.h"
#include "utils/autofuse_attrs.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "utils/auto_fuse_config.h"
#include "post_process/asc_backend_post_processor.h"
#include "ascgen_log.h"
#include "utils/autofuse_utils.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "attribute_group/attr_group_shape_env.h"

namespace ge {
using namespace autofuse;
class AscBackendFusionDeciderTest : public testing::Test {
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

  ge::ascir_op::Store x_store("x_store_broadcast_add");
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, B, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_out("x_out_broadcast_add");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};
  auto x_out_node = graph.FindNode("x_out_broadcast_add");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatBroadcastAddAscGraph1(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);

  ge::ascir_op::Data x1("x1_broadcast_add", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id};
  *x1.y.repeats = {A, B, C};
  *x1.y.strides = {B * C, C, ONE};

  ge::ascir_op::Load x1Local("x1Local_broadcast_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id};
  *x1Local.y.axis = {a.id, b.id, c.id};
  *x1Local.y.repeats = {A, B, C};
  *x1Local.y.strides = {B * C, C, ONE};

  ge::ascir_op::Data x2("x2_broadcast_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id};
  *x2.y.repeats = {A, B, C, D};
  *x2.y.strides = {B * C * D , C * D, D, ONE};

  ge::ascir_op::Load x2Local("x2Local_broadcast_add");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id};
  *x2Local.y.repeats = {A, B, C, D};
  *x2Local.y.strides = {B * C * D , C * D, D, ONE};

  ge::ascir_op::Add add("add_broadcast_add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id};
  *add.y.axis = {a.id, b.id, c.id, d.id};
  *add.y.repeats = {A, B, C, D};
  *add.y.strides = {B * C * D , C * D, D, ONE};

  ge::ascir_op::Store x_store("x_store_broadcast_add");
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id};
  *x_store.y.repeats = {A, B, C, D};
  *x_store.y.strides = {B * C * D , C * D, D, ONE};

  ge::ascir_op::Output x_out("x_out_broadcast_add");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id};
  *x_out.y.repeats = {A, B, C, D};
  *x_out.y.strides = {B * C * D , C * D, D, ONE};
  auto x_out_node = graph.FindNode("x_out_broadcast_add");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatMoreAxisAddAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("x1_more_add", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1.y.repeats = {A, B, C, D, E, F};
  *x1.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Load x1Local("x1Local_more_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, B, C, D, E, F};
  *x1Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Data x2("x2_more_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2.y.repeats = {A, B, C, D, E, F};
  *x2.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Load x2Local("x2Local_more_add");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.repeats = {A, B, C, D, E, F};
  *x2Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Add add("add_more_add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.repeats = {A, B, C, D, E, F};
  *add.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Store x_store("x_store_more_add");
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, B, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_out("x_out_more_add");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};
  auto x_out_node = graph.FindNode("x_out_more_add");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatBroadcastAbsAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("x1_broadcast_abs", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1.y.repeats = {A, B, C, D, E, ONE};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

  ge::ascir_op::Load x1Local("x1Local_broadcast_abs");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, B, C, D, E, F};
  *x1Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Abs abs("add_broadcast_abs");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.repeats = {A, B, C, D, E, F};
  *abs.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Store x_store("x_store_broadcast_abs");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, B, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_out("x_out_broadcast_abs");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};
  auto x_out_node = graph.FindNode("x_out_broadcast_abs");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}
std::shared_ptr<AscGraph> CreatBroadcastAbsAddGraph(ge::AscGraph &graph) {
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
  *x2.y.repeats = {A, B, C, D, E, ONE};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

  ge::ascir_op::Load x2Local("x1Local_broadcast_add");
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

  ge::ascir_op::Store x_store("x_store_broadcast_abs");
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, B, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_out("x_out_broadcast_abs");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};
  auto x_out_node = graph.FindNode("x_out_broadcast_abs");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatAdd6Axis1OneRepeatAddGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");
  const Expression F = graph.CreateSizeVar(1);

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
  *x1Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ZERO};

  ge::ascir_op::Data x2("x2_broadcast_add", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2.y.repeats = {A, B, C, D, E, ONE};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO};

  ge::ascir_op::Load x2Local("x1Local_broadcast_add");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x2Local.y.repeats = {A, B, C, D, E, F};
  *x2Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ZERO};

  ge::ascir_op::Add add("add_broadcast_add");
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *add.y.repeats = {A, B, C, D, E, F};
  *add.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ZERO};

  ge::ascir_op::Store x_store("x_store_broadcast_abs");
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, B, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ZERO};

  ge::ascir_op::Output x_out("x_out_broadcast_abs");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ZERO};
  auto x_out_node = graph.FindNode("x_out_broadcast_abs");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatBroadcastAbsAfterBroadcastAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");
  const Expression F = graph.CreateSizeVar("F");
  const Expression G = graph.CreateSizeVar("G");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  auto f = graph.CreateAxis("F", F);
  auto g = graph.CreateAxis("G", G);

  ge::ascir_op::Data x1("x1_broadcast2_abs", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  *x1.y.repeats = {A, B, C, D, E, ONE, ONE};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE, ZERO, ZERO};

  ge::ascir_op::Load x1Local("x1Local_broadcast2_abs");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  *x1Local.y.repeats = {A, B, C, D, E, F, G};
  *x1Local.y.strides = {B * C * D * E * F * G, C * D * E * F * G, D * E * F * G, E * F * G, F * G, G, ONE};

  ge::ascir_op::Abs abs("add_broadcast2_abs");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  *abs.y.repeats = {A, B, C, D, E, F, G};
  *abs.y.strides = {B * C * D * E * F * G, C * D * E * F * G, D * E * F * G, E * F * G, F * G, G, ONE};

  ge::ascir_op::Store x_store("x_store_broadcast2_abs");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  *x_store.y.repeats = {A, B, C, D, E, F, G};
  *x_store.y.strides = {B * C * D * E * F * G, C * D * E * F * G, D * E * F * G, E * F * G, F * G, G, ONE};

  ge::ascir_op::Output x_out("x_out_broadcast2_abs");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id, g.id};
  *x_out.y.repeats = {A, B, C, D, E, F, G};
  *x_out.y.strides = {B * C * D * E * F * G, C * D * E * F * G, D * E * F * G, E * F * G, F * G, G, ONE};

  auto x_out_node = graph.FindNode("x_out_broadcast2_abs");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatCalcRstdAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("x1_calc_rstd", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("x1Local_calc_rstd");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_calc_rstd", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A, B, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("x2Local_calc_rstd");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x3("x3_calc_rstd", graph);
  x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x3.attr.sched.loop_axis = c.id;
  *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3.y.repeats = {A, B, C, D, E};
  *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("x3Local_calc_rstd");
  x3Local.x = x3.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A, B, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::FlashSoftmax calcRstd("calcRstd_calc_rstd");
  calcRstd.x1 = x1Local.y;
  calcRstd.x2 = x2Local.y;
  calcRstd.x3 = x3Local.y;
  calcRstd.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y1.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y1.repeats = {A, B, C, D, E};
  *calcRstd.y1.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  *calcRstd.y2.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y2.repeats = {A, B, C, D, E};
  *calcRstd.y2.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  *calcRstd.y3.axis = {a.id, b.id, c.id, d.id, e.id};
  *calcRstd.y3.repeats = {A, B, C, D, E};
  *calcRstd.y3.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_calc_rstd");
  x_store.x = calcRstd.y1;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Store x_store1("x_store1_calc_rstd");
  x_store1.x = calcRstd.y2;
  x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store1.attr.sched.loop_axis = c.id;
  *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store1.y.repeats = {A, B, C, D, E};
  *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_calc_rstd");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  ge::ascir_op::Output x_out1("x_out1_calc_rstd");
  x_out1.x = x_store1.y;
  x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out1.attr.sched.loop_axis = c.id;
  *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out1.y.repeats = {A, B, C, D, E};
  *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode("x_out_calc_rstd");
  auto x_out_node1 = graph.FindNode("x_out1_calc_rstd");
  AscGraphUtils::GetComputeGraph(graph)->AddOutputNodeByIndex(x_out_node, 0);
  AscGraphUtils::GetComputeGraph(graph)->AddOutputNodeByIndex(x_out_node1, 1);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

// dim 2做了reduce
static std::shared_ptr<AscGraph> CreatReduceAscGraph(ge::AscGraph &graph, size_t out_num = 1, size_t in_num = 1) {
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

  std::string data_name = "data" + graph.GetName();
  ge::ascir_op::Data x1((data_name + "1").c_str(), graph);
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

  if (in_num == 2) {
    ge::ascir_op::Data x2((data_name + "2").c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local("x2Local_reduce");
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  }

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

static std::shared_ptr<ge::AscGraph> CreatGatherAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar(1);
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
  x1.attr.sched.loop_axis = b.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id};
  *x1.y.repeats = {A, B, D, E};
  *x1.y.strides = {B * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("data2", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = e.id;
  x2.y.dtype = DT_INT32;
  *x2.y.axis = {a.id, b.id};
  *x2.y.repeats = {ONE, C};
  *x2.y.strides = {ZERO, ONE};

  ge::ascir_op::Gather gather(graph.GetName().c_str());
  gather.x1 = {x1.y};
  gather.x2 = {x2.y};
  gather.ir_attr.SetAxis(1);
  gather.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *gather.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *gather.y.repeats = {A, ONE, C, D, E};
  *gather.y.strides = {C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Store x_store("store");
  x_store.x = gather.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, ONE, C, D, E};
  *x_store.y.strides = {C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Output x_out("out");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, ONE, C, D, E};
  *x_out.y.strides = {C * D * E, ZERO, D * E, E, ONE};

  auto x_out_node = graph.FindNode("out");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();

  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatAbsAfterReduceAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
  auto ZERO = Symbol(0);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar(1);
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
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, ONE, C, D, E};
  *x1Local.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Abs abs("abs_abs_after_reduce");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *abs.y.repeats = {A, ONE, C, D, E};
  *abs.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_abs_after_reduce");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, ONE, C, D, E};
  *x_store.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_abs_after_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, ONE, C, D, E};
  *x_out.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  auto x_out_node = graph.FindNode("x_out_abs_after_reduce");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatReduceAfterReduceAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("x1_reduce_after_reduce", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, ONE, C, D, E};
  *x1.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Load x1Local("x1Local_reduce_after_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, ONE, C, D, E};
  *x1Local.y.strides = {B * C * D * E, ZERO, D * E, E, ONE};

  ge::ascir_op::Max reduce("reduce_after_reduce");
  reduce.x = x1Local.y;
  reduce.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *reduce.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *reduce.y.repeats = {ONE, ONE, C, D, E};
  *reduce.y.strides = {ZERO, ZERO, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_reduce_after_reduce");
  x_store.x = reduce.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {ONE, ONE, C, D, E};
  *x_store.y.strides = {ZERO, ZERO, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_reduce_after_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {ONE, ONE, C, D, E};
  *x_out.y.strides = {ZERO, ZERO, D * E, E, ONE};

  auto x_out_node = graph.FindNode("x_out_reduce_after_reduce");
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
  ge::ascir_op::Data x1((data_name + "1").c_str(), graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A, B, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local((graph.GetName() + "_load1").c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add((graph.GetName() + "_add").c_str());
  if (in_num == 2) {
    ge::ascir_op::Data x2((data_name + "2").c_str(), graph);
    x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x2.attr.sched.loop_axis = c.id;
    *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2.y.repeats = {A, B, C, D, E};
    *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Load x2Local((graph.GetName() + "_load2").c_str());
    x2Local.x = x2.y;
    x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x2Local.y.repeats = {A, B, C, D, E};
    *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    add.x1 = x1Local.y;
    add.x2 = x2Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  } else {
    add.x1 = x1Local.y;
    add.x2 = x1Local.y;
    add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *add.y.repeats = {A, B, C, D, E};
    *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  }
  ge::ascir_op::Store x_store((graph.GetName() + "_store").c_str());
  x_store.x = add.y;
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
  if (out_num == 2) {
    ge::ascir_op::Store x_store1((graph.GetName() + "_store1").c_str());
    x_store1.x = add.y;
    x_store1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_store1.attr.sched.loop_axis = c.id;
    *x_store1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_store1.y.repeats = {A, B, C, D, E};
    *x_store1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

    ge::ascir_op::Output x_out1((graph.GetName() + "_out1").c_str());
    x_out1.x = x_store1.y;
    x_out1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
    x_out1.attr.sched.loop_axis = c.id;
    *x_out1.y.axis = {a.id, b.id, c.id, d.id, e.id};
    *x_out1.y.repeats = {A, B, C, D, E};
    *x_out1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
    auto x_out_node1 = graph.FindNode((graph.GetName() + "_out1").c_str());
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

static std::shared_ptr<AscGraph> CreatStridedSliceAscGraph(ge::AscGraph &graph) {
  auto ONE = Symbol(1);
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

  ge::ascir_op::Data x("data_stridedSlice", graph);
  x.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x.attr.sched.loop_axis = c.id;
  *x.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x.y.repeats = {A + ONE, B + ONE, C, D, E, F};
  *x.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};
  x.ir_attr.SetIndex(0);

  ge::ascir_op::Load xLocal("xLocal_stridedSlice");
  xLocal.x = x.y;
  xLocal.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *xLocal.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *xLocal.y.repeats = {A + ONE, B + ONE, C, D, E, F};
  *xLocal.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Store x_store("x_store_stridedSlice");
  x_store.x = xLocal.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, B, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_out("x_out_stridedSlice");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  auto x_out_node = graph.FindNode("x_out_stridedSlice");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

static std::shared_ptr<ge::AscGraph> CreatSplitAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("data_1", graph);
  x1.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x1.attr.sched.loop_axis = c.id;
  *x1.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1.y.repeats = {A + ONE, B + ONE, C, D, E};
  *x1.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x1Local("load1");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A + ONE, B + ONE, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("data_2", graph);
  x2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x2.attr.sched.loop_axis = c.id;
  *x2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2.y.repeats = {A + ONE, B + ONE, C, D, E};
  *x2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x2Local("load2");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A + ONE, B + ONE, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x3("data_3", graph);
  x3.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x3.attr.sched.loop_axis = c.id;
  *x3.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3.y.repeats = {A + ONE, B + ONE, C, D, E};
  *x3.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Load x3Local("load3");
  x3Local.x = x3.y;
  x3Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x3Local.y.repeats = {A + ONE, B + ONE, C, D, E};
  *x3Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("store");
  x_store.x = x1Local.y;
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

  ge::ascir_op::Store x_store1("store1");
  x_store1.x = x2Local.y;
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

  ge::ascir_op::Store x_store2("store2");
  x_store2.x = x3Local.y;
  x_store2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store2.attr.sched.loop_axis = c.id;
  *x_store2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store2.y.repeats = {A, B, C, D, E};
  *x_store2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out2("out2");
  x_out2.x = x_store2.y;
  x_out2.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out2.attr.sched.loop_axis = c.id;
  *x_out2.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out2.y.repeats = {A, B, C, D, E};
  *x_out2.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};
  auto x_out_node2 = graph.FindNode("out2");

  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}, {x_out_node1, 1}, {x_out_node2, 2}};
  compute_graph->SetOutputSize(3U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  
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

static Status SetAttrsGroup(const NodePtr &node) {
  auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto attr = GetOrCreateAutoFuseAttrs(op_desc);
  GE_ASSERT_NOTNULL(attr);

  ge::AscGraph add_graph(node->GetName().c_str());
  if (node->GetName() == "D") {
    attr->SetAscGraph(CreatAddAscGraph(add_graph, 2U));
  } else if ((node->GetName() == "AddN1") || (node->GetName() == "AddN0")) {
    attr->SetAscGraph(CreatAddAscGraph(add_graph, 1U, 2U), loop::FuseType::kPointwise);
  } else if (node->GetName().find("Concat") != std::string::npos) {
    attr->SetAscGraph(CreatConcatAscGraph(add_graph), loop::FuseType::kConcat);
  } else if (node->GetName() == "Reduce") {
    attr->SetAscGraph(CreatReduceAscGraph(add_graph, 1U, 1U), loop::FuseType::kReduction);
  } else if (node->GetName() == "A1") {
    attr->SetAscGraph(CreatAddAscGraph(add_graph, 2, 2));
  } else if (node->GetName() == "HuberLoss1") {
    attr->SetAscGraph(CreatCalcRstdAscGraph(add_graph));
  } else if (node->GetName() == "AddN2" || node->GetName() == "Assign1") {
    attr->SetAscGraph(CreatBroadcastAddAscGraph(add_graph));
  } else if (node->GetName() == "AddN3") {
    attr->SetAscGraph(CreatBroadcastAddAscGraph1(add_graph));
  } else if (node->GetName().find("BroadCastAbs1") != std::string::npos) {
    attr->SetAscGraph(CreatBroadcastAbsAscGraph(add_graph));
  } else if (node->GetName().find("Add6Axis1OneRepeat") != std::string::npos) {
    attr->SetAscGraph(CreatAdd6Axis1OneRepeatAddGraph(add_graph));
  } else if (node->GetName().find("CreatBroadcastAbsAddGraph") != std::string::npos) {
    attr->SetAscGraph(CreatBroadcastAbsAddGraph(add_graph));
  } else if (node->GetName() == "AbsAfterReduce") {
    attr->SetAscGraph(CreatAbsAfterReduceAscGraph(add_graph), loop::FuseType::kPointwise);
  } else if (node->GetName() == "AbsBroadCastAfterReduce") {
    attr->SetAscGraph(CreatAbsBroadcastAfterReduceAscGraph(add_graph), loop::FuseType::kPointwise);
  } else if (node->GetName() == "ReduceAfterReduce") {
    attr->SetAscGraph(CreatReduceAfterReduceAscGraph(add_graph), loop::FuseType::kReduction);
  } else if (node->GetName() == "BroadcastAbsAfterBroadcast" || node->GetName() == "BroadCastAbs2") {
    attr->SetAscGraph(CreatBroadcastAbsAfterBroadcastAscGraph(add_graph));
  } else if (node->GetName().find("Slice") != std::string::npos) {
    attr->SetAscGraph(CreatStridedSliceAscGraph(add_graph));
  } else if (node->GetName() == "HuberLoss2") {
    attr->SetAscGraph(CreatCalcRstdAscGraph(add_graph), loop::FuseType::kConcat);
  } else if (node->GetName().find("Gather") != std::string::npos) {
    attr->SetAscGraph(CreatGatherAscGraph(add_graph), loop::FuseType::kGather);
  } else if (node->GetName() == "AddN1_1") {
    attr->SetAscGraph(CreatMoreAxisAddAscGraph(add_graph));
  } else if (node->GetName().find("Split") != std::string::npos) {
    attr->SetAscGraph(CreatSplitAscGraph(add_graph), loop::FuseType::kSplit);
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
}  // namespace

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_StridedSliceAndStridedSlice_Ok) {
  AscBackendFusionDecider decider;
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto stridedslice1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedSlice1");
  auto stridedslice2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedSlice2");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(stridedslice1)->EDGE(0, 0)->NODE(stridedslice2));
    CHAIN(NODE(stridedslice2)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto compute_graph = ToComputeGraph(g1);
  for (const auto &node : compute_graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto stridedSlice1_node = compute_graph->FindNode("StridedSlice1");
  ASSERT_NE(stridedSlice1_node, nullptr);
  auto stridedSlice2_node = compute_graph->FindNode("StridedSlice2");
  ASSERT_NE(stridedSlice2_node, nullptr);
  auto op_desc1 = stridedSlice1_node->GetOpDescBarePtr();
  auto op_desc2 = stridedSlice2_node->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }

  ASSERT_EQ(decider.CanFuseVertical(stridedSlice1_node, stridedSlice2_node), true);
  auto fused_node = decider.Fuse(stridedSlice1_node, stridedSlice2_node, nullptr);
  ASSERT_NE(fused_node, nullptr);
}

// slice算子的AscBackend节点在和其他AscBackend水平融合时不能去修改load的offset属性
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_ConcatAndSlice_Ok) {
  AscBackendFusionDecider decider;
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("data");
  auto relu1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("Relu1");
  auto stridedslice1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("StridedSlice1");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
      .OutNames({"y"}).Build("D");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(stridedslice1));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(relu1));
    CHAIN(NODE(relu1)->EDGE(0, 0)->NODE(d));
    CHAIN(NODE(stridedslice1)->EDGE(0, 1)->NODE(d));
    CHAIN(NODE(d)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto compute_graph = ToComputeGraph(g1);
  for (const auto &node : compute_graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto stridedSlice1_node = compute_graph->FindNode("StridedSlice1");
  ASSERT_NE(stridedSlice1_node, nullptr);
  auto relu1_node = compute_graph->FindNode("Relu1");
  ASSERT_NE(relu1_node, nullptr);
  auto op_desc1 = stridedSlice1_node->GetOpDescBarePtr();
  auto op_desc2 = relu1_node->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("64")), SUCCESS);
    }
  }

  auto fused_node = decider.Fuse(relu1_node, stridedSlice1_node, nullptr);
  ASSERT_NE(fused_node, nullptr);
  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  auto output = AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->FindNode("xLocal_stridedSlice");
  ASSERT_NE(output->GetOpDesc(), nullptr);
  const auto &attr = output->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  ASSERT_NE(attr, nullptr);
  auto load_node_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
  ASSERT_NE(load_node_attr, nullptr);
  Expression load_offset;
  load_node_attr->GetOffset(load_offset);
  ASSERT_EQ(load_offset, Symbol("64"));
}

// split和concat可以融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_SplitAndConcat_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto split1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("Split1");
  auto concat1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(3).OutCnt(1).OutNames({"y"})
                   .Build("Concat1");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Shape");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(split1));
    CHAIN(NODE(split1)->EDGE(0, 0)->NODE(concat1));
    CHAIN(NODE(split1)->EDGE(1, 1)->NODE(concat1));
    CHAIN(NODE(split1)->EDGE(2, 2)->NODE(concat1));
    CHAIN(NODE(concat1)->EDGE(0, 0)->NODE(shape)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Split1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Concat1");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// split和reduce不可以融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_SplitAndReduce_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto split1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(3).InNames({"x"})
                   .OutNames({"y"}).Build("Split1");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Shape");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(split1));
    CHAIN(NODE(split1)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(shape)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Split1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Reduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), false);

  ASSERT_EQ(decider.CanFuseVertical(node2, node1), false);    

  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// elementwise和elementwise可以融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_CanFuse1_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
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
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Shape");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// elementwise和reduce可以融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_CanFuse2_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(2).OutCnt(1).OutNames({"y"})
                   .Build("AddN1");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(reduce)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);

  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Reduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// reduce+element可以融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_CanFuse3_Ok) {
  AscBackendFusionDecider decider;
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto reduce = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Reduce");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
                   .Build("AddN1");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(reduce));
    CHAIN(NODE(reduce)->EDGE(0, 0)->NODE(addn1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 4);

  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AddN1");
  ASSERT_NE(node2, nullptr);

  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// reduce+reduce不可以垂直融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_CanFuse_Nok) {
  AscBackendFusionDecider decider;
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto reduce1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Reduce");
  auto reduce2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("ReduceAfterReduce");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(reduce1));
    CHAIN(NODE(reduce1)->EDGE(0, 0)->NODE(reduce2)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 4);

  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("ReduceAfterReduce");
  ASSERT_NE(node2, nullptr);
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), false);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 *
 *       netoutput
 *        /     \
 *     addn1  concat
 *    /   \  /   \   \
 * data1  data2 data3 data4
 *******************
 *        netoutput
 *          /   \
 *       FusedAscBc_1
 *     /   \    \    \
 * data1  data2  data3 data4
 */
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseNoLoopMerge1_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto huber_loss2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(huber_loss2));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(huber_loss2));
    CHAIN(NODE(data4)->EDGE(0, 2)->NODE(huber_loss2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(huber_loss2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("HuberLoss2");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 *          netoutput
 *          /       \
 *    shapeNo1     concat
 *         \      /   | \
 *           addN1    |  \
 *           |    \   |   \
 *          |      \  |    \
 *        data1    data2  data3
 *
 *******************
 *         netoutput1
 *          /    |
 *    shapeNo1   |
 *         \     |
 *       FusedAscBc_1
 *       /   |  \
 *     /     |   \
 * data1   data2 data3
 *
 */
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseNoLoopMerge2_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(2).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("ShapeNo1");
  auto huber_loss2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(1, 1)->NODE(huber_loss2));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(huber_loss2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(huber_loss2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(huber_loss2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("HuberLoss2");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 *
 *          netoutput1
 *        /   \      \.
 *     add1  assign1   \.
 *    /   \  /     \     \.
 * data1  data2   data3 data4
 *******************
 *          netoutput1
 *        /   \      \.
 *       AscBc_1      \.
 *     /   \    \      \.
 * data1  data2 data3 data4
 */
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge1_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto assign1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                     .OutNames({"y"}).Build("Assign1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(assign1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(data4)->EDGE(0, 2)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(assign1)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Assign1");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 15);
  EXPECT_EQ(graph->GetAllNodesSize(), 6);

  ASSERT_EQ(fused_node->GetInDataNodes().size(), 3);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 2);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);

  int64_t index = -1;
  int64_t res_index = -1;
  auto output_nodes = GetInterAttrs(fused_attr).fused_subgraph_outputs;
  EXPECT_EQ(output_nodes.size(), 2);
  for (const auto &output_node : output_nodes) {
    const auto attr = output_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
    ASSERT_NE(attr, nullptr);
    attr->ir_attr->GetAttrValue("index", res_index);
    ASSERT_EQ(res_index, ++index);
  }
}

/**
 *         netoutput1
 *          /       \.
 *    shapeNo1     add1
 *         \      /   |
 *      huberLoss1   |
 *      |   \   \   |
 *     |     \   \ |
 * data1   data2 data3
 *
 *******************
 *         netoutput1
 *          /    |
 *    shapeNo1   |
 *         \     |
 *        AscBc_1
 *       /    |  \
 *     /     |    \
 * data1  data2  data3
 *
 */
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge2_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto huber_loss1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(2).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss1");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto shape_no1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                       .OutNames({"y"}).Build("ShapeNo1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(huber_loss1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(shape_no1));
    CHAIN(NODE(huber_loss1)->EDGE(1, 0)->NODE(addn1));
    CHAIN(NODE(shape_no1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn1)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("HuberLoss1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AddN1");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 13);
  EXPECT_EQ(graph->GetAllNodesSize(), 6);

  ASSERT_EQ(fused_node->GetInDataNodes().size(), 3);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 2);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 *    data1  data2
 *       \    /
 *        add0
 * data3   |    data4
 *   \    /   \   /
 *    add1    add2
 *     |       |
 *     netoutput
 *
 *******************
 *    data1  data2
 *       \    /
 *        add0
 *   data3  |   data4
 *      \   |   /
 *        AscBc_1
 *          |
 *      netoutput
 *
 */
// elementwise + broadcast水平融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge3_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN0");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto addn2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn0));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn2));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(addn2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 8);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AddN2");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 16);
  auto output = AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->FindNode("add_broadcast_add");
  ASSERT_NE(output, nullptr);

  EXPECT_EQ(graph->GetAllNodesSize(), 7);

  ASSERT_EQ(fused_node->GetInDataNodes().size(), 3);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 2);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 *    data1  data2
 *       \    /
 *        add0
 * data3   |    data4
 *   \    /   \   /
 *    add1    add2
 *     |       |
 *     netoutput
 *
 *******************
 *    data1  data2
 *       \    /
 *        add0
 *   data3  |  data4
 *      \   |   /
 *        AscBc_1
 *          |
 *      netoutput
 *
 */
// broadcast + elementwise 水平融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge3_2_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN0");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto addn2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn0));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn2));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(addn2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 8);
  auto node1 = graph->FindNode("AddN2");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AddN1");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 16);
  auto output = AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->FindNode("add_broadcast_add");
  ASSERT_NE(output, nullptr);

  EXPECT_EQ(graph->GetAllNodesSize(), 7);

  ASSERT_EQ(fused_node->GetInDataNodes().size(), 3);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 2);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 * data1     data2
 *    \       /
 *     \     /
 *      addn1
 *        |
 *      shape
 *        |
 *    netoutput
 */
// element + broadcast可以垂直融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge4_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("BroadCastAbs");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("BroadCastAbs");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 * data1     data2
 *    \       /|
 *     \     / |
 *      addn1  |
 *        \    |
 *         \   |
 *          add
 *            |
 *        netoutput
 */
// element + broadcast可以垂直融合也可以水平融合，水平融合轴映射失败，垂直融合轴映射成功
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge4_1_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("CreatAdd6Axis1OneRepeatAddGraph");
  auto add = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("CreatBroadcastAbsAddGraph");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("CreatAdd6Axis1OneRepeatAddGraph");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("CreatBroadcastAbsAddGraph");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 12);
  auto output = AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->FindNode("add_broadcast_add");
  ASSERT_NE(output, nullptr);

  EXPECT_EQ(graph->GetAllNodesSize(), 4);

  EXPECT_EQ(fused_node->GetInDataNodes().size(), 2);
  EXPECT_EQ(fused_node->GetOutDataNodes().size(), 1);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 * data1     data2
 *  | \       /|
 *  |  \     / |
 *  |   addn1  |
 *  |   / \    |
 *  |  /   \   |
 *  add1    add2
 *    \       /
 *     \     /
 *      add3
 *       |
 *    netoutput
 */
// element + broadcast可以垂直融合也可以水平融合，水平融合轴映射失败，垂直融合轴映射成功, 但是node1有输出多引用，不能融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge4_2_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("CreatAdd6Axis1OneRepeatAddGraph");
  auto add1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("add1");
  auto add2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("CreatBroadcastAbsAddGraph");
  auto add3 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("add3");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(add1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(add2));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(add2));
    CHAIN(NODE(add1)->EDGE(0, 0)->NODE(add3));
    CHAIN(NODE(add1)->EDGE(0, 1)->NODE(add3));
    CHAIN(NODE(add3)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("CreatAdd6Axis1OneRepeatAddGraph");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("CreatBroadcastAbsAddGraph");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_EQ(fused_node, nullptr);
}

// reduce+element可以垂直融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge5_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Reduce");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AbsAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsAfterReduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// reduce+broadcast可以垂直融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge6_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Reduce");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AbsBroadCastAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsBroadCastAfterReduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// broadcast + broadcast可以垂直融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge7_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN2");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("BroadcastAbsAfterBroadcast");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn2));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn2));
    CHAIN(NODE(addn2)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("AddN2");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("BroadcastAbsAfterBroadcast");
  ASSERT_NE(node2, nullptr);

  { // 测试format传递
    auto node1_op_desc = node1->GetOpDescBarePtr();
    const auto node1_input_desc0 = node1_op_desc->MutableInputDesc(0);
    auto node1_attr0 = node1_input_desc0->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    const auto node1_input_desc1 = node1_op_desc->MutableInputDesc(1);
    auto node1_attr1 = node1_input_desc1->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    node1_input_desc0->SetFormat(FORMAT_NC1HWC0);
    node1_input_desc0->SetOriginFormat(FORMAT_NHWC);
    node1_input_desc1->SetFormat(FORMAT_NHWC);
    node1_input_desc1->SetOriginFormat(FORMAT_NC1HWC0);
    auto node2_op_desc = node2->GetOpDescBarePtr();
    auto node2_output_desc = node2_op_desc->MutableOutputDesc(0);
    node2_output_desc->SetFormat(FORMAT_NHWC);
    node2_output_desc->SetOriginFormat(FORMAT_NHWC);
  }

  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);

  { // 测试format传递
    auto fused_node_op_desc = fused_node->GetOpDescBarePtr();
    const auto fused_node_input_desc0 = fused_node_op_desc->MutableInputDesc(0);
    auto fused_node_attr0 = fused_node_input_desc0->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    const auto fused_node_input_desc1 = fused_node_op_desc->MutableInputDesc(1);
    auto fused_node_attr1 = fused_node_input_desc1->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();
    ASSERT_EQ(fused_node_input_desc0->GetFormat(), FORMAT_NC1HWC0);
    ASSERT_EQ(fused_node_input_desc0->GetOriginFormat(), FORMAT_NHWC);
    ASSERT_EQ(fused_node_input_desc1->GetFormat(), FORMAT_NHWC);
    ASSERT_EQ(fused_node_input_desc1->GetOriginFormat(), FORMAT_NC1HWC0);
    auto fused_node_output_desc = fused_node_op_desc->MutableOutputDesc(0);
    ASSERT_EQ(fused_node_output_desc->GetFormat(), FORMAT_NHWC);
    ASSERT_EQ(fused_node_output_desc->GetOriginFormat(), FORMAT_NHWC);
  }

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// gather+element可以垂直融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge8_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Gather");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AbsAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Gather");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsAfterReduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

/**
 *         netoutput1
 *          /       \.
 *    shapeNo1     concat
 *         \      /   | \
 *      huberLoss1    | add
 *    /      |    \   |   \
 *  /       |      \  |    \
 * data1  data2    data3   data4
 *
 *******************
 *         netoutput1
 *          /    |
 *    shapeNo1   |
 *         \     |
 *        FusedAscBc_1
 *       /    |  \   \
 *     /      |   \   \
 * data1   data2 data3 data4
 *
 */
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseNoLoopMergeAndFuseAgain3_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto huber_loss1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(2).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss1");
  auto add = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                 .OutNames({"y"}).Build("Add");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("ShapeNo1");
  auto huber_loss2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(huber_loss1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(huber_loss2));
    CHAIN(NODE(data4)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 2)->NODE(huber_loss2));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(huber_loss1)->EDGE(1, 0)->NODE(huber_loss2));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(huber_loss2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  AscBackendFusionDecider decider;
  EXPECT_EQ(graph->GetAllNodesSize(), 9);
  auto node1 = graph->FindNode("Add");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("HuberLoss2");
  ASSERT_NE(node2, nullptr);
  auto node3 = graph->FindNode("HuberLoss1");
  ASSERT_NE(node3, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  auto op_desc3 = node3->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  ASSERT_NE(op_desc3, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  auto attr3 = GetOrCreateAutoFuseAttrs(op_desc3);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  ASSERT_NE(attr3, nullptr);

  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  fused_node = decider.Fuse(node3, fused_node, nullptr);
  ASSERT_NE(fused_node, nullptr);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_TryRemoveNodesCtrEdges_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(2).OutCnt(1).OutNames({"y"})
                   .Build("AddN1");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Shape");
  DEF_GRAPH(g) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->CTRL_EDGE()->NODE(shape));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g);
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  const auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("Shape");
  ASSERT_NE(node2, nullptr);
  ASSERT_EQ(BackendUtils::TryRemoveNodesCtrEdges(node1, node2), SUCCESS);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_GetAscGraphAxisGroup_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto huber_loss1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(2).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss1");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto shape_no1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                       .OutNames({"y"}).Build("ShapeNo1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(huber_loss1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(shape_no1));
    CHAIN(NODE(huber_loss1)->EDGE(1, 0)->NODE(addn1));
    CHAIN(NODE(shape_no1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn1)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  optimize::autoschedule::AxisGroup axes_group;
  GetInterAttrs(attr1).axis_group.y_group = {0};
  GetInterAttrs(attr1).axis_group.x_group = {1};
  AxisPairSet axis_map = {{0, 1}};
  ASSERT_NE(BackendUtils::GetAscGraphAxisGroup(node1, axes_group, axis_map), SUCCESS);

  GetInterAttrs(attr1).axis_group.x_group = {1};
  ASSERT_NE(BackendUtils::GetAscGraphAxisGroup(node1, axes_group, axis_map), SUCCESS);

  GetInterAttrs(attr1).axis_group.x_group = {0};
  GetInterAttrs(attr1).axis_group.y_group = {1};
  ASSERT_NE(BackendUtils::GetAscGraphAxisGroup(node1, axes_group, axis_map), SUCCESS);

  GetInterAttrs(attr1).axis_group.y_group = {0};
  GetInterAttrs(attr1).axis_group.r_group = {1};
  ASSERT_NE(BackendUtils::GetAscGraphAxisGroup(node1, axes_group, axis_map), SUCCESS);

  GetInterAttrs(attr1).axis_group.r_group = {0};
  GetInterAttrs(attr1).axis_group.n_group = {1};
  ASSERT_NE(BackendUtils::GetAscGraphAxisGroup(node1, axes_group, axis_map), SUCCESS);

  graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);

  op_desc1 = node1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(BackendUtils::GetAscGraphAxisGroup(node1, axes_group, axis_map), SUCCESS);
  int64_t axis_id = 0;
  ASSERT_EQ(BackendUtils::ConvertAxis({}, axis_id, true), SUCCESS);
  std::vector<int64_t> base_line_axis = {2};
  ASSERT_NE(BackendUtils::ConvertAxis({{1, 0}}, base_line_axis, true), SUCCESS);

  optimize::autoschedule::AxisGroup group1;
  group1.y_group = {1, 2, 3};
  optimize::autoschedule::AxisGroup group2;
  group2.y_group = {1, 2, 3, 4};
  optimize::autoschedule::AxisGroup merged_axes_group;
  EXPECT_TRUE(BackendUtils::IsCanMergeAxisGroup(group1, group2, merged_axes_group));
  group1.y_group = {1, 2, 3, 4};
  group2.y_group = {1, 2, 3};
  EXPECT_TRUE(BackendUtils::IsCanMergeAxisGroup(group1, group2, merged_axes_group));
  group1.y_group = {2, 1, 3, 4};
  group2.y_group = {1, 2, 3, 5};
  EXPECT_TRUE(BackendUtils::IsCanMergeAxisGroup(group1, group2, merged_axes_group));

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_GetAscGraphAxisGroupOrder_Ok) {
  optimize::autoschedule::AxisGroup group1;
  optimize::autoschedule::AxisGroup group2;
  group1.axes_order = {1, 2, 3};
  group2.axes_order = {1, 2, 3, 4};
  BackendUtils::MergeAxesOrder(group1, group2);
  EXPECT_TRUE(group1.axes_order.size() == 4);
  EXPECT_TRUE(group1.axes_order.at(3) == 4);

  group1.axes_order = {1, 2, 3, 4};
  group2.axes_order = {1, 2, 3};
  BackendUtils::MergeAxesOrder(group1, group2);
  EXPECT_TRUE(group2.axes_order.size() == 4);
  EXPECT_TRUE(group2.axes_order.at(3) == 4);

  group1.axes_order = {1, 2, 3, 4};
  group2.axes_order = {2, 3, 4, 5};
  BackendUtils::MergeAxesOrder(group1, group2);
  EXPECT_TRUE(group1.axes_order.size() == 4);
  EXPECT_TRUE(group1.axes_order.at(3) == 4);
  EXPECT_TRUE(group2.axes_order.at(3) == 5);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_CheckAxisSubsetRelation_Ok) {
  std::vector<int64_t> axis1 = {2, 4};
  std::vector<int64_t> axis2 = {1, 2, 3, 4, 5};
  bool result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_TRUE(result);  // axis1 是 axis2 的循序子集

  axis1 = {1, 2, 3, 4, 5};
  axis2 = {2, 4};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_TRUE(result);  // axis2 是 axis1 的循序子集

  axis1 = {1, 2, 3};
  axis2 = {1, 2, 3};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_TRUE(result);  // axis1 和 axis2 完全相等

  axis1 = {1, 3, 5};
  axis2 = {2, 4, 6};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_FALSE(result);  // axis1 和 axis2 没有子集关系

  axis1 = {1, 3, 5};
  axis2 = {1, 2, 3, 4, 5};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_TRUE(result);  // axis1 是 axis2 的循序子集

  axis1 = {1, 3, 5};
  axis2 = {2, 3, 4, 5, 6};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_FALSE(result);  // axis1 和 axis2 部分重叠，但不是子集

  axis1 = {1, 2, 3};
  axis2 = {1, 2, 3, 4};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_TRUE(result);  // axis1 是 axis2 的循序子集

  axis1 = {};
  axis2 = {1, 2, 3};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_TRUE(result);  // 空集是任何集合的子集

  axis1 = {1, 2, 3};
  axis2 = {};
  result = BackendUtils::CheckAxisSubsetRelation(axis1, axis2);
  EXPECT_TRUE(result);
}

// 水平融合场景下两节点融合后输入个数超过阈值不融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FusedNodeInputNumsExceedThresholdCanFuse_fail) {
  AscBackendFusionDecider decider;
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto assign1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                     .OutNames({"y"}).Build("Assign1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 0)->NODE(assign1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(assign1));
    CHAIN(NODE(data4)->EDGE(0, 2)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(assign1)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Assign1");
  ASSERT_NE(node2, nullptr);

  // 配置融合后节点的输入个数阈值是2，add1和assign1融合后输入个数是3
  const auto default_value = AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_input_nums_after_fuse;
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_input_nums_after_fuse = 2;
  EXPECT_EQ(decider.CanFuseHorizontal(node1, node2), false);
  AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_input_nums_after_fuse = default_value;
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_UpdateDataAndOutputIndex_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Reduce");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AbsBroadCastAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsBroadCastAfterReduce");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);

  // 校验index节点加上
  int64_t index = -1;
  int64_t res_index = -1;
  for (const auto &input_node : attr1->GetAscGraph()->GetInputNodes()) {
    const auto attr = input_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
    ASSERT_NE(attr, nullptr);
    attr->ir_attr->GetAttrValue("index", res_index);
    ASSERT_EQ(res_index, ++index);
  }
  index = -1;
  res_index = -1;
  auto output_nodes = GetInterAttrs(attr1).fused_subgraph_outputs;
  for (const auto &output_node : output_nodes) {
    const auto attr = output_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
    ASSERT_NE(attr, nullptr);
    attr->ir_attr->GetAttrValue("index", res_index);
    ASSERT_EQ(res_index, ++index);
  }
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_UpdateDataAndOutputIndexMulRefs_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN0");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto addn2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn0));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn2));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(addn2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 8);
  auto node0 = graph->FindNode("AddN0");
  ASSERT_NE(node0, nullptr);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AddN2");
  ASSERT_NE(node2, nullptr);
  auto op_desc0 = node0->GetOpDescBarePtr();
  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc0, nullptr);
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr0 = GetOrCreateAutoFuseAttrs(op_desc0);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr0, nullptr);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node0, node1), true);
  ASSERT_EQ(decider.CanFuseVertical(node0, node2), true);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);

  // 校验index节点加上
  int64_t index = -1;
  int64_t res_index = -1;
  for (const auto &input_node : attr0->GetAscGraph()->GetInputNodes()) {
    const auto attr = input_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
    ASSERT_NE(attr, nullptr);
    attr->ir_attr->GetAttrValue("index", res_index);
    ASSERT_EQ(res_index, ++index);
  }
  // 去重后指向1个Output节点
  index = 0;
  res_index = -1;
  auto output_nodes = GetInterAttrs(attr0).fused_subgraph_outputs;
  ASSERT_EQ(output_nodes.size(), 2);
  for (const auto &output_node : output_nodes) {
    const auto attr = output_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
    ASSERT_NE(attr, nullptr);
    attr->ir_attr->GetAttrValue("index", res_index);
    ASSERT_EQ(res_index, index);
  }
}

/**
 *           netoutput1
 *          /    |     \
 * shapeNo1  shapeNo2  concat
 *        \/        /  | \
 *       huberLoss1    |  \
 *      /       |      |   \
 * data1  data2    data3   data4
 *
 ********************************
 *           netoutput1
 *          /    |   |
 * shapeNo1 shapeNo2 |
 *        \/         |
 *        FusedAscBc_1
 *       /    |  \   \
 *     /      |   \   \
 * data1   data2 data3 data4
 */
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_UpdateNetoutputMulRefsFusedComputeGraph_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto huber_loss1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(2).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss1");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("ShapeNo1");
  auto shape2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("ShapeNo2");
  auto huber_loss2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(huber_loss1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(huber_loss2));
    CHAIN(NODE(data4)->EDGE(0, 2)->NODE(huber_loss2));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(shape2));
    CHAIN(NODE(huber_loss1)->EDGE(1, 0)->NODE(huber_loss2));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(shape2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(huber_loss2)->EDGE(0, 2)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 9);
  auto node1 = graph->FindNode("HuberLoss1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("HuberLoss2");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  BackendUtils::UpdateFusedAscBackendNetOutput(fused_node);
  auto &fused_compute_graph = fused_attr->GetFuseComputeGraph();
  ASSERT_NE(fused_compute_graph, nullptr);
  auto output_size = fused_node->GetAllOutDataAnchorsSize();
  const auto netoutput = fused_compute_graph->GetOrUpdateNetOutputNode();
  ASSERT_NE(netoutput, nullptr);
  auto netoutput_in_anchor_size = netoutput->GetAllInDataAnchorsSize();
  EXPECT_EQ(output_size, netoutput_in_anchor_size);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_Two_Nodes_No_Relation) {
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                  .OutNames({"y"}).Build("Data");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("B");
  auto c = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("C");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("D");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(c)->EDGE(0, 0)->NODE("NetOutput1", kNetOutputType));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE("NetOutput2", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto node1 = graph->FindNode("C");
  ASSERT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("D");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseHorizontal(node1, node2), false);

  AscBackendSubGraphFusionDecider sub_decider;
  ASSERT_EQ(sub_decider.CanFuseHorizontal(node1, node2), false);
}

/**
 *    data1  data2
 *       \    /
 *        add0
 * data3   |    data4
 *   \    /   \   /
 *    add1    add2
 *     |       |
 *     netoutput
 *
 *******************
 *    data1  data2
 *       \    /
 *        add0
 *   data3  |   data4
 *      \   |   /
 *        AscBc_1
 *          |
 *      netoutput
 *
 */
// elementwise + elementwise水平融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge3_1_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN0");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto addn2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1_1");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn0));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn2));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(addn2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 8);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AddN1_1");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 15);
  auto output = AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->FindNode("add_more_add");
  ASSERT_NE(output, nullptr);

  EXPECT_EQ(graph->GetAllNodesSize(), 7);

  ASSERT_EQ(fused_node->GetInDataNodes().size(), 3);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 2);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

class TestCounter : public Counter {
 public:
  TestCounter() = default;
  virtual ~TestCounter() = default;
  virtual int64_t NextId() {
    return id_++;
  };

 private:
  int64_t id_ = 0;
};

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FusedAscBackendNodeName_Ok) {
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto data3 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data3");
  auto data4 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data4");
  auto huber_loss1 = OP_CFG(kAscBackendType)
                         .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                         .InCnt(3)
                         .OutCnt(1)
                         .InNames({"x"})
                         .OutNames({"y"})
                         .Build("HuberLoss1");
  auto add = OP_CFG(kAscBackendType)
                 .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                 .InCnt(1)
                 .OutCnt(1)
                 .InNames({"x"})
                 .OutNames({"y"})
                 .Build("Add");
  auto shape = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(1)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("ShapeNo1");
  auto huber_loss2 = OP_CFG(kAscBackendType)
                         .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                         .InCnt(3)
                         .OutCnt(1)
                         .InNames({"x"})
                         .OutNames({"y"})
                         .Build("HuberLoss2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(huber_loss1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(huber_loss2));
    CHAIN(NODE(data4)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 2)->NODE(huber_loss2));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(huber_loss2));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(huber_loss2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 9);
  auto node1 = graph->FindNode("HuberLoss1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("HuberLoss2");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  op_desc2->SetName("autofuse_pointwise_0_Add_Abs");
  op_desc1->SetName("autofuse_reduce_1_Exp_ReduceSumD");

  CounterPtr counter = new TestCounter();
  auto fused_node = decider.Fuse(node1, node2, counter);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetName(), "autofuse_fused_0_Exp_ReduceSumD_Add_Abs");
  delete counter;
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_AscBackendNodeName_Ok) {
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto addn1 = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(2)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("Reduce");
  auto shape = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(1)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("AbsAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsAfterReduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  op_desc1->SetName("autofuse_reduce_0_Exp_ReduceSumD");
  op_desc2->SetName("autofuse_pointwise_1_Add_Abs");

  CounterPtr counter = new TestCounter();
  auto fused_node = decider.Fuse(node1, node2, counter);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetName(), "autofuse_0_Add_Abs_Exp_ReduceSumD");
  delete counter;
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_AscBackendNodeName_Ok2) {
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto addn1 = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(2)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("Reduce");
  auto shape = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(1)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("AbsAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsAfterReduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  op_desc1->SetName("autofuse_0_Exp_ReduceSumD");
  op_desc2->SetName("autofuse_1_Add_Abs");

  CounterPtr counter = new TestCounter();
  auto fused_node = decider.Fuse(node1, node2, counter);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetName(), "autofuse_0_Add_Abs_Exp_ReduceSumD");
  delete counter;
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_AscBackendNodeName_Ok3) {
  auto data1 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data1");
  auto data2 = OP_CFG("Data")
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(0)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("data2");
  auto addn1 = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(2)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("Reduce");
  auto shape = OP_CFG(kAscBackendType)
                   .TensorDesc(FORMAT_ND, DT_FLOAT, {1, 2, 3, 4})
                   .InCnt(1)
                   .OutCnt(1)
                   .InNames({"x"})
                   .OutNames({"y"})
                   .Build("AbsAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsAfterReduce");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  op_desc1->SetName("autofuse_0_");
  op_desc2->SetName("aaa_bbb_ccc");

  CounterPtr counter = new TestCounter();
  auto fused_node = decider.Fuse(node1, node2, counter);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetName(), "autofuse_0_");
  delete counter;
}

/**
 *           netoutput1
 *          /       /  \
 *         /      abs   exp
 *        /          \ /
 *    shapeNo1     concat
 *         \      /   | \
 *      huberLoss1    |  add
 *    /      |    \   |   \
 *  /       |      \  |    \
 * data1  data2    data3   data4
 *
 ************************************
 *         netoutput1
 *          /    |
 *    shapeNo1   AscBc1
 *         \     |
 *        FusedAscBc_1
 *       /    |  \   \
 *     /      |   \   \
 * data1   data2 data3 data4
 *
 */
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_FuseNoLoopMergeAndFuseAgain_FusedAscBackendOutputs_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto huber_loss1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(2).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss1");
  auto add = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                 .OutNames({"y"}).Build("Add");
  auto abs = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                 .OutNames({"y"}).Build("Abs");
  auto exp = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                 .OutNames({"y"}).Build("Exp");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("ShapeNo1");
  auto huber_loss2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                         .OutNames({"y"}).Build("HuberLoss2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(huber_loss1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(huber_loss1));
    CHAIN(NODE(data3)->EDGE(0, 1)->NODE(huber_loss2));
    CHAIN(NODE(data4)->EDGE(0, 0)->NODE(add));
    CHAIN(NODE(add)->EDGE(0, 2)->NODE(huber_loss2));
    CHAIN(NODE(huber_loss2)->EDGE(0, 0)->NODE(abs));
    CHAIN(NODE(huber_loss2)->EDGE(0, 0)->NODE(exp));
    CHAIN(NODE(huber_loss1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(huber_loss1)->EDGE(1, 0)->NODE(huber_loss2));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(abs)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(exp)->EDGE(0, 2)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  AscBackendFusionDecider decider;
  EXPECT_EQ(graph->GetAllNodesSize(), 11);
  auto node1 = graph->FindNode("Add");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("HuberLoss2");
  ASSERT_NE(node2, nullptr);
  auto node3 = graph->FindNode("HuberLoss1");
  ASSERT_NE(node3, nullptr);
  auto node4 = graph->FindNode("Abs");
  ASSERT_NE(node4, nullptr);
  auto node5 = graph->FindNode("Exp");
  ASSERT_NE(node5, nullptr);

  // 先生成FusedAscBackend节点
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), true);
  auto fused_node = decider.Fuse(node1, node2, nullptr);

  // FusedAscBackend节点的输出多引用节点之间水平融合，导致实际FusedAscBackend节点变少
  ASSERT_EQ(decider.CanFuseHorizontal(node4, node5), true);
  auto fused_node2 = decider.Fuse(node4, node5, nullptr);

  // FusedAscBackend节点后续继续融合
  fused_node = decider.Fuse(node3, fused_node, nullptr);
  ASSERT_NE(fused_node, nullptr);

  ComputeGraphPtr subgraph;
  BackendUtils::GetNodeFusedGraph(fused_node, subgraph);
  auto netoutput = subgraph->GetOrUpdateNetOutputNode();
  ASSERT_NE(netoutput, nullptr);
  ASSERT_EQ(netoutput->GetInDataNodesSize(), fused_node->GetOutNodesSize());

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

// 校验AscBackendNoKernel不参与canfuse融合
TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_AscBackendNoKernel_NotFuse_Ok) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Reduce");
  auto shape = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AbsBroadCastAfterReduce");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 5);
  auto node1 = graph->FindNode("Reduce");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AbsBroadCastAfterReduce");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);

  // 设置为AscBackendNoKernelOp类型
  op_desc1->SetType(kAscBackendNoKernelType);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  // 空Tensor lowering后的AscBackendNoKernelOp不融合
  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), false);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_HorizontalFuse_Same_BroadCastInfo_CanFuse) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN0");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("BroadCastAbs1");
  auto addn2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("BroadCastAbs2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn0));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 6);
  auto node1 = graph->FindNode("BroadCastAbs1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("BroadCastAbs2");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 13);
  auto output = AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->FindNode("add_broadcast_abs");
  ASSERT_NE(output, nullptr);

  EXPECT_EQ(graph->GetAllNodesSize(), 5);

  ASSERT_EQ(fused_node->GetInDataNodes().size(), 1);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 2);
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr1->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  for (auto node : AscGraphUtils::GetComputeGraph(*(attr2->GetAscGraph()))->GetDirectNode()) {
    if (node->GetType() == kLoadType) {
      ASSERT_NE(node->GetOpDesc(), nullptr);
      const auto &attr = node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
      ASSERT_NE(attr, nullptr);
      auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
      ASSERT_NE(load_attr, nullptr);
      ASSERT_EQ(load_attr->SetOffset(Symbol("load_offset")), SUCCESS);
    }
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

TEST_F(AscBackendFusionDeciderTest, BackendUtils_IsSameBroadCastInfo) {
  // attr_infos1中broadcast_info不为空，但值不同，返回false
  std::vector<int64_t> broadcast_info1 = {1, 2, 3, 4};
  std::vector<int64_t> broadcast_info2 = {1, 2, 3};
  std::vector<ViewOpAttrInfo> attr_infos1;
  ViewOpAttrInfo attr_info1;
  attr_info1.broadcast_info = broadcast_info1;
  ViewOpAttrInfo attr_info2;
  attr_info2.broadcast_info = broadcast_info2;
  attr_infos1.push_back(attr_info1);
  attr_infos1.push_back(attr_info2);
  std::vector<ViewOpAttrInfo> attr_infos2;
  attr_infos2.push_back(attr_info2);
  ASSERT_EQ(BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2), false);
  // attr_infos1中broadcast_info相同，attr_infos2不为空，但值不同，返回false
  attr_infos1.clear();
  attr_infos2.clear();
  attr_info2.broadcast_info = broadcast_info1;
  attr_infos1.push_back(attr_info1);
  attr_infos1.push_back(attr_info2);
  ViewOpAttrInfo attr_info3;
  attr_info3.broadcast_info = broadcast_info1;
  ViewOpAttrInfo attr_info4;
  attr_info4.broadcast_info = broadcast_info2;
  attr_infos2.push_back(attr_info3);
  attr_infos2.push_back(attr_info4);
  ASSERT_EQ(BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2), false);
  // attr_infos1、attr_infos2中broadcast_info相同，返回true
  attr_infos1.clear();
  attr_infos2.clear();
  attr_info1.broadcast_info = broadcast_info1;
  attr_infos1.push_back(attr_info1);
  attr_infos2.push_back(attr_info1);
  ASSERT_EQ(BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2), true);
  // attr_infos1、attr_infos2中broadcast_info不同，返回false
  broadcast_info1 = {1, 2, 3, 4};
  broadcast_info2 = {1, 2, 3, 5};
  attr_infos1.clear();
  attr_infos2.clear();
  attr_info1.broadcast_info = broadcast_info1;
  attr_info2.broadcast_info = broadcast_info2;
  attr_infos1.push_back(attr_info1);
  attr_infos2.push_back(attr_info2);
  ASSERT_EQ(BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2), false);
  // attr_infos1、attr_infos2中broadcast_info为空，返回true
  std::vector<int64_t> empty_broadcast_info = {};
  attr_infos1.clear();
  attr_infos2.clear();
  attr_info1.broadcast_info = empty_broadcast_info;
  attr_info2.broadcast_info = empty_broadcast_info;
  attr_infos1.push_back(attr_info1);
  attr_infos2.push_back(attr_info2);
  ASSERT_EQ(BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2), true);
  // attr_infos1、attr_infos2都为空，返回true
  attr_infos1.clear();
  attr_infos2.clear();
  ASSERT_EQ(BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2), true);
  // attr_infos1、attr_infos2其中一个为空，另一个不为空，返回false
  attr_infos1.push_back(attr_info1);
  ASSERT_EQ(BackendUtils::IsSameBroadCastInfo(attr_infos1, attr_infos2), false);
}

TEST_F(AscBackendFusionDeciderTest, TestIsAllInputSimplestLoad_FusedAscBackend) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto addn0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN0");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("BroadCastAbs1");
  auto addn2 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(1).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("BroadCastAbs2");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn0));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn2));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn2)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  ASSERT_NE(graph, nullptr);

  int loop_cnt = 0;
  static std::atomic<int64_t> i{0};
  ge::AscGraph graph_fuse1("add");
  auto asc_graph_fuse1 = CreatAddAscGraph(graph_fuse1);
  ASSERT_NE(asc_graph_fuse1, nullptr);
  NodePtr node_fused = nullptr;
  for (const auto &node : graph->GetDirectNode()) {
    const auto &op_desc = node->GetOpDesc();
    if (loop_cnt == 0) {
      node_fused = node;
      std::string type = "FusedAscBackend";
      op_desc->SetType(type);
      auto attr = GetOrCreateAutoFuseAttrs(op_desc);
      ASSERT_NE(attr, nullptr);
      attr->SetFuseComputeGraph(AscGraphUtils::GetComputeGraph(*asc_graph_fuse1));
    }
    loop_cnt++;
  }
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  // ComputeGraphPtr fused_graph;
  // GE_ASSERT_SUCCESS(BackendUtils::GetNodeFusedGraph(node_fused, fused_graph));
  // auto subgraph_input_nodes = fused_graph->GetInputNodes();
  ASSERT_EQ(BackendUtils::IsNodeAllInputsAreSimplestLoad(node_fused), SUCCESS);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_Node_Get_Subgraph_Failed) {
  auto data = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                  .OutNames({"y"}).Build("Data");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("B");
  auto c = OP_CFG("test_op").TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("C");
  auto d = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("D");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(c)->EDGE(0, 0)->NODE("NetOutput1", kNetOutputType));
    CHAIN(NODE(data)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE(d)->EDGE(0, 0)->NODE("NetOutput2", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto node1 = graph->FindNode("C");
  ASSERT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("D");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseHorizontal(node1, node2), false);
  ASSERT_EQ(decider.CanFuseHorizontal(node2, node1), false);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_Node_Get_Ascgraph_Axis_Group_Failed1) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data2");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("B");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(3).OutCnt(1).OutNames({"y"})
                    .Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(concat)->EDGE(0, 0)->NODE("NetOutput1", kNetOutputType));
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(b));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(b)->EDGE(0, 2)->NODE(concat));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto node1 = graph->FindNode("A");
  ASSERT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("Concat");
  ASSERT_NE(node2, nullptr);
  const auto node3 = graph->FindNode("B");
  ASSERT_NE(node3, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  auto attr = BackendUtils::GetNodeAutoFuseAttr(fused_node);
  ASSERT_NE(attr, nullptr);
  GetInterAttrs(attr).axis_group = optimize::autoschedule::AxisGroup();
  ASSERT_EQ(decider.CanFuseHorizontal(node3, fused_node), false);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_Update_Subgraph_Output_Attr_Failed1) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data2");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("B");
  auto test = OP_CFG(kFusedAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
                  .Build("test_op");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(test)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE("NetOutput1", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto node1 = graph->FindNode("test_op");
  ASSERT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("A");
  ASSERT_NE(node2, nullptr);

  auto attr = BackendUtils::GetNodeAutoFuseAttr(node1);
  ASSERT_NE(attr, nullptr);
  GetInterAttrs(attr).axis_group = optimize::autoschedule::AxisGroup();
  ge::AscGraph add_graph(node1->GetName().c_str());
  auto asc_graph = CreatAddAscGraph(add_graph);
  auto compute_graph = AscGraphUtils::GetComputeGraph(*asc_graph);
  attr->SetFuseComputeGraph(compute_graph);
  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseHorizontal(node1, node2), false);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_Update_Subgraph_Output_Attr_Failed2) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data2");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("B");
  auto test = OP_CFG(kFusedAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
                  .Build("test_op");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(test)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE("NetOutput1", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto node1 = graph->FindNode("test_op");
  ASSERT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("A");
  ASSERT_NE(node2, nullptr);

  auto attr = BackendUtils::GetNodeAutoFuseAttr(node1);
  ASSERT_NE(attr, nullptr);
  GetInterAttrs(attr).axis_group = optimize::autoschedule::AxisGroup();
  ge::AscGraph add_graph(node1->GetName().c_str());
  auto asc_graph = CreatAddAscGraph(add_graph);
  auto compute_graph = AscGraphUtils::GetComputeGraph(*asc_graph);
  attr->SetFuseComputeGraph(compute_graph);
  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseHorizontal(node2, node1), false);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_Node_Get_Ascgraph_Axis_Group_Failed2) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data3");
  auto a = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("A");
  auto b = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(1).OutCnt(1).OutNames({"y"})
               .Build("B");
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {2,2,3,4}).InCnt(3).OutCnt(1).OutNames({"y"})
                    .Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(a)->EDGE(0, 0)->NODE(concat)->EDGE(0, 0)->NODE(b)->EDGE(0, 0)->NODE("NetOutput1", kNetOutputType));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(concat));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(concat));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  const auto node1 = graph->FindNode("A");
  ASSERT_NE(node1, nullptr);
  const auto node2 = graph->FindNode("Concat");
  ASSERT_NE(node2, nullptr);
  const auto node3 = graph->FindNode("B");
  ASSERT_NE(node3, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  auto attr = BackendUtils::GetNodeAutoFuseAttr(fused_node);
  ASSERT_NE(attr, nullptr);
  GetInterAttrs(attr).axis_group = optimize::autoschedule::AxisGroup();
  // 这里将fusetype设置为Pointwise是为了不走自定义融合策略
  attr->SetFuseType(loop::FuseType::kPointwise);
  ASSERT_EQ(decider.CanFuseHorizontal(fused_node, node3), false);
}

TEST_F(AscBackendFusionDeciderTest, Vector_Core_Num_Not_Equal) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("Data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
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
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Shape");
  ASSERT_NE(node2, nullptr);

  const auto op_desc1 = node1->GetOpDesc();
  ASSERT_NE(op_desc1, nullptr);
  const auto attr1 = op_desc1->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr1, nullptr);
  GetInterAttrs(attr1).vector_core_num = 16;

  const auto op_desc2 = node2->GetOpDesc();
  ASSERT_NE(op_desc2, nullptr);
  const auto attr2 = op_desc2->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(attr2, nullptr);
  GetInterAttrs(attr2).vector_core_num = 20;

  AscBackendFusionDecider decider;
  ASSERT_EQ(decider.CanFuseVertical(node1, node2), false);
}

TEST_F(AscBackendFusionDeciderTest, AscBackendFusionDecider_GraphAndLoad_AxisNum_NotEqual) {
  auto data1 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data1");
  auto data2 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data2");
  auto data3 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data3");
  auto data4 = OP_CFG("Data").TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(0).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("data4");
  auto addn0 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN0");
  auto addn1 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN1");
  auto addn3 = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(2).OutCnt(1).InNames({"x"})
                   .OutNames({"y"}).Build("AddN3");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn0));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn0));
    CHAIN(NODE(data3)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(addn0)->EDGE(0, 0)->NODE(addn3));
    CHAIN(NODE(data4)->EDGE(0, 1)->NODE(addn3));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(addn3)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }

  EXPECT_EQ(graph->GetAllNodesSize(), 8);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("AddN3");
  ASSERT_NE(node2, nullptr);

  auto op_desc1 = node1->GetOpDescBarePtr();
  auto op_desc2 = node2->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);
  AscBackendFusionDecider decider;
  ASSERT_EQ(asc_adapt::GeFallback(graph), SUCCESS);
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_NE(fused_node, nullptr);
  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
}

}  // namespace ge
