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
#include <gmock/gmock.h>
#include "can_fuse/backend/asc_backend_torch_fusion_decider.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "ascir_ops.h"
#include "utils/autofuse_attrs.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "utils/auto_fuse_config.h"
#include "post_process/asc_backend_post_processor.h"
#include "ascgen_log.h"
#include "utils/autofuse_utils.h"

namespace ge {
using namespace autofuse;
class AscBackendTorchFusionDeciderTest : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

namespace {
class GraphBuilder {
 public:
  GraphBuilder(const std::string &name) {
    graph_ = std::make_shared<ComputeGraph>(name);
  }

  GraphBuilder(const std::string &name, const std::string &node_type) {
    graph_ = std::make_shared<ComputeGraph>(name);
    node_type_ = node_type;
  }

  NodePtr AddNode(const std::string &name, const std::string &type, int32_t in_cnt, int32_t out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 1, 1}) {
    auto tensor_desc = std::make_shared<GeTensorDesc>();
    tensor_desc->SetShape(GeShape(std::move(shape)));
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    tensor_desc->GetOrCreateAttrsGroup<ge::SymbolicDescAttr>();

    auto op_desc = std::make_shared<OpDesc>(name, (node_type_ == "") ? type : "AscBackend");
    for (int32_t i = 0; i < in_cnt; ++i) {
      op_desc->AddInputDesc(tensor_desc->Clone());
    }
    for (int32_t i = 0; i < out_cnt; ++i) {
      op_desc->AddOutputDesc(tensor_desc->Clone());
    }
    op_desc->AddInferFunc([](Operator &op) { return GRAPH_SUCCESS; });
    return graph_->AddNode(op_desc);
  }

  void AddDataEdge(const NodePtr &src_node, int32_t src_idx, const NodePtr &dst_node, int32_t dst_idx) {
    GraphUtils::AddEdge(src_node->GetOutDataAnchor(src_idx), dst_node->GetInDataAnchor(dst_idx));
  }

  NodePtr AddNodeByIr(const std::string &op_name, const std::string &op_type) {
    auto op = ge::OperatorFactory::CreateOperator(op_name.c_str(), op_type.c_str());
    if (op.IsEmpty()) {
      return nullptr;
    }
    OpDescPtr op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
    return graph_->AddNode(op_desc);
  }

  void AddControlEdge(const NodePtr &src_node, const NodePtr &dst_node) {
    GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor());
  }

  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
  std::string node_type_;
};

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
  auto builder = GraphBuilder("test", node_type);
  auto const1 = builder.AddNode("const1", "CONSTANT", 0, 1);
  auto const2 = builder.AddNode("const2", "CONSTANT", 0, 1);
  auto addn1 = builder.AddNode("addn1", "AddNYes", 2, 1);
  auto shape1 = builder.AddNode("shape1", "ShapeNo", 1, 1);
  auto netoutput1 = builder.AddNode("netoutput", "NETOUTPUT", 1, 0);

  builder.AddDataEdge(const1, 0, addn1, 0);
  builder.AddDataEdge(const2, 0, addn1, 1);
  builder.AddDataEdge(addn1, 0, shape1, 0);
  builder.AddDataEdge(shape1, 0, netoutput1, 0);

  return builder.GetGraph();
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
  auto builder = GraphBuilder("test", node_type);
  auto var1 = builder.AddNode("var1", "Variable", 0, 1);
  auto var2 = builder.AddNode("var2", "VariableV2", 0, 1);
  auto var3 = builder.AddNode("var3", "VarHandleOp", 0, 1);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto assign1 = builder.AddNode("assign1", "Assign", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput1", "Netoutput", 3, 0);

  builder.AddDataEdge(var1, 0, add1, 0);
  builder.AddDataEdge(var2, 0, add1, 1);
  builder.AddDataEdge(var2, 0, assign1, 1);
  builder.AddDataEdge(var3, 0, netoutput1, 2);
  builder.AddDataEdge(const1, 0, assign1, 0);
  builder.AddDataEdge(add1, 0, netoutput1, 0);
  builder.AddDataEdge(assign1, 0, netoutput1, 1);

  return builder.GetGraph();
}

/**
 *         netoutput1
 *          /       \.
 *    shapeNo1     add1
 *         \      /   |
 *      huberLoss1    |
 *    /      |    \   |
 *  /       |      \  |
 * const1  const2  const3
 */
ComputeGraphPtr BuildGraph3(const std::string node_type = "") {
  auto builder = GraphBuilder("test", node_type);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto const2 = builder.AddNode("const2", "Const", 0, 1);
  auto const3 = builder.AddNode("const3", "Const", 0, 1);
  auto huberLoss1 = builder.AddNode("huberLoss1", "HuberLossYes", 3, 2);
  auto shape1 = builder.AddNode("shape1", "ShapeNo", 1, 1);
  auto add1 = builder.AddNode("add1", "Add", 2, 1);
  auto netoutput1 = builder.AddNode("netoutput", "NetOutput", 2, 0);

  builder.AddDataEdge(const1, 0, huberLoss1, 0);
  builder.AddDataEdge(const2, 0, huberLoss1, 1);
  builder.AddDataEdge(const3, 0, huberLoss1, 2);
  builder.AddDataEdge(huberLoss1, 0, shape1, 0);
  builder.AddDataEdge(huberLoss1, 1, add1, 0);
  builder.AddDataEdge(shape1, 0, netoutput1, 0);
  builder.AddDataEdge(add1, 0, netoutput1, 1);
  builder.AddDataEdge(const3, 0, add1, 1);
  return builder.GetGraph();
}

std::shared_ptr<AscGraph> CreatAddAscGraph(ge::AscGraph &graph, bool is_incremental = true) {
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
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *add.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *add.y.repeats = {A, B, C, D, E};
  *add.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

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

  ge::ascir_op::Load x1Local("x1Local_broadcast_add");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, B, C, D, E, F};
  *x1Local.y.strides = {B * C * D * E * F, C * D * E * F, D * E * F, E * F, F, ONE};

  ge::ascir_op::Data x2("x2_broadcast_add", graph);

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

  ge::ascir_op::Load x1Local("x1Local_calc_rstd");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_calc_rstd", graph);

  ge::ascir_op::Load x2Local("x2Local_calc_rstd");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x3("x3_calc_rstd", graph);

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

std::shared_ptr<AscGraph> CreatSubAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("x1_sub", graph);

  ge::ascir_op::Load x1Local("x1Local_sub");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2("x2_sub", graph);

  ge::ascir_op::Load x2Local("x2Local_sub");
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Sub sub("sub_sub");
  sub.x1 = x1Local.y;
  sub.x2 = x2Local.y;
  sub.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *sub.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *sub.y.repeats = {A, B, C, D, E};
  *sub.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_sub");
  x_store.x = sub.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_sub");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode("x_out_sub");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatConcatAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("x1_sub", graph);

  ge::ascir_op::Load x1Local("x1Local_concat");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Concat concat("concat_concat");
  concat.x = {x1Local.y};
  concat.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *concat.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *concat.y.repeats = {A, B, C, D, E};
  *concat.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store("x_store_sub");
  x_store.x = concat.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A, B, C, D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out("x_out_sub");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A, B, C, D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode("x_out_sub");
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

std::shared_ptr<AscGraph> CreatAbsBroadcastAscGraph(ge::AscGraph &graph) {
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

  ge::ascir_op::Data x1("x1_abs_after_reduce", graph);

  ge::ascir_op::Load x1Local("x1Local_abs_after_reduce");
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x1Local.y.repeats = {A, B, C, D, E, F};
  *x1Local.y.strides = {B * C * D * E * F, C * D * E, D * E * F, E * F, F, ONE};

  ge::ascir_op::Abs abs("abs_abs_after_reduce");
  abs.x = x1Local.y;
  abs.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *abs.y.repeats = {A, B, C, D, E, F};
  *abs.y.strides = {B * C * D * E * F, C * D * E, D * E * F, E * F, F, ONE};

  ge::ascir_op::Store x_store("x_store_abs_after_reduce");
  x_store.x = abs.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_store.y.repeats = {A, B, C, D, E, F};
  *x_store.y.strides = {B * C * D * E * F, C * D * E, D * E * F, E * F, F, ONE};

  ge::ascir_op::Output x_out("x_out_abs_after_reduce");
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, d.id, e.id, f.id};
  *x_out.y.repeats = {A, B, C, D, E, F};
  *x_out.y.strides = {B * C * D * E * F, C * D * E, D * E * F, E * F, F, ONE};

  auto x_out_node = graph.FindNode("x_out_abs_after_reduce");
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatTransposeAscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto c = graph.CreateAxis("C", C);
  auto b = graph.CreateAxis("B", B);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x1_transpose(("x1_transpose" + extern_name).c_str());
  x1_transpose.x = x1Local.y;
  x1_transpose.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1_transpose.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1_transpose.y.repeats = {A, C, B, D, E};
  *x1_transpose.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x2_transpose(("x2_transpose" + extern_name).c_str());
  x2_transpose.x = x2Local.y;
  x2_transpose.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2_transpose.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2_transpose.y.repeats = {A, C, B, D, E};
  *x2_transpose.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1_transpose.y;
  add.x2 = x2_transpose.y;
  add.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *add.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *add.y.repeats = {A, C, B, D, E};
  *add.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x_store.y.repeats = {A, C, B, D, E};
  *x_store.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x_out.y.repeats = {A, C, B, D, E};
  *x_out.y.strides = {C * B * D * E, B * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatInvalidTransposeAscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto c = graph.CreateAxis("C", C);
  auto b = graph.CreateAxis("B", B);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x1_transpose(("x1_transpose" + extern_name).c_str());
  x1_transpose.x = x1Local.y;
  x1_transpose.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1_transpose.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1_transpose.y.repeats = {A, C, B, D, E};
  *x1_transpose.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x2_transpose(("x2_transpose" + extern_name).c_str());
  x2_transpose.x = x2Local.y;
  x2_transpose.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2_transpose.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2_transpose.y.repeats = {A, C, B, D, E};
  *x2_transpose.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1_transpose.y;
  add.x2 = x2_transpose.y;
  add.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *add.y.axis = {a.id, c.id, 10, d.id, e.id};
  *add.y.repeats = {A, C, B, D, E};
  *add.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x_store.y.repeats = {A, C, B, D, E};
  *x_store.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x_out.y.repeats = {A, C, B, D, E};
  *x_out.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatInvalidTranspose1AscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto c = graph.CreateAxis("C", C);
  auto b = graph.CreateAxis("B", B);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x1_transpose(("x1_transpose" + extern_name).c_str());
  x1_transpose.x = x1Local.y;
  x1_transpose.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1_transpose.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1_transpose.y.repeats = {A, C, B, D, E};
  *x1_transpose.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x2_transpose(("x2_transpose" + extern_name).c_str());
  x2_transpose.x = x2Local.y;
  x2_transpose.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2_transpose.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2_transpose.y.repeats = {A, C, B, D, E};
  *x2_transpose.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1_transpose.y;
  add.x2 = x2_transpose.y;
  add.attr.sched.axis = {a.id, c.id, 100, d.id, e.id};
  *add.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *add.y.repeats = {A, C, B, D, E};
  *add.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x_store.y.repeats = {A, C, B, D, E};
  *x_store.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x_out.y.repeats = {A, C, B, D, E};
  *x_out.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatTranspose1AscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id};
  *x1Local.y.axis = {a.id, b.id, c.id};
  *x1Local.y.repeats = {A, B, C};
  *x1Local.y.strides = {B * C, C, ONE};

  ge::ascir_op::Transpose x1_transpose(("x1_transpose" + extern_name).c_str());
  x1_transpose.x = x1Local.y;
  x1_transpose.attr.sched.axis = {a.id, c.id, b.id};
  *x1_transpose.y.axis = {a.id, c.id, b.id};
  *x1_transpose.y.repeats = {A, C, B};
  *x1_transpose.y.strides = {B * C, B, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id};
  *x2Local.y.axis = {a.id, b.id, c.id};
  *x2Local.y.repeats = {A, B, C};
  *x2Local.y.strides = {B * C, C, ONE};

  ge::ascir_op::Transpose x2_transpose(("x2_transpose" + extern_name).c_str());
  x2_transpose.x = x2Local.y;
  x2_transpose.attr.sched.axis = {a.id, c.id, b.id};
  *x2_transpose.y.axis = {a.id, c.id, b.id};
  *x2_transpose.y.repeats = {A, C, B};
  *x2_transpose.y.strides = {B * C, B, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1_transpose.y;
  add.x2 = x2_transpose.y;
  add.attr.sched.axis = {a.id, c.id, b.id};
  *add.y.axis = {a.id, c.id, b.id};
  *add.y.repeats = {A, C, B};
  *add.y.strides = {B * C, B, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, c.id, b.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, c.id, b.id};
  *x_store.y.repeats = {A, C, B};
  *x_store.y.strides = {B * C, B, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, c.id, b.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, c.id, b.id};
  *x_out.y.repeats = {A, C, B};
  *x_out.y.strides = {B * C, B, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

// transpose垂直融合
std::shared_ptr<AscGraph> CreatTransposeAfterTransposeAscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto c = graph.CreateAxis("C", C);
  auto b = graph.CreateAxis("B", B);
  auto e = graph.CreateAxis("E", E);
  auto d = graph.CreateAxis("D", D);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x1Local.y.repeats = {A, C, B, D, E};
  *x1Local.y.strides = {B * C * D * E, B * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x1_transpose(("x1_transpose" + extern_name).c_str());
  x1_transpose.x = x1Local.y;
  x1_transpose.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  *x1_transpose.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x1_transpose.y.repeats = {A, C, B, E, D};
  *x1_transpose.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2Local.y.axis = {a.id, c.id, b.id, d.id, e.id};
  *x2Local.y.repeats = {A, C, B, D, E};
  *x2Local.y.strides = {B * C * D * E, B * D * E, D * E, E, ONE};

  ge::ascir_op::Transpose x2_transpose(("x2_transpose" + extern_name).c_str());
  x2_transpose.x = x2Local.y;
  x2_transpose.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  *x2_transpose.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x2_transpose.y.repeats = {A, C, B, E, D};
  *x2_transpose.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1_transpose.y;
  add.x2 = x2_transpose.y;
  add.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  *add.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *add.y.repeats = {A, C, B, E, D};
  *add.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x_store.y.repeats = {A, C, B, E, D};
  *x_store.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x_out.y.repeats = {A, C, B, E, D};
  *x_out.y.strides = {C * B * E * D, B * E * D, E * D, D, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

// transpose水平融合
std::shared_ptr<AscGraph> CreatTransposeAfterTranspose1AscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C);
  auto e = graph.CreateAxis("E", E);
  auto d = graph.CreateAxis("D", D);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A, B, C, D, E};
  *x1Local.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Transpose x1_transpose(("x1_transpose" + extern_name).c_str());
  x1_transpose.x = x1Local.y;
  x1_transpose.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  *x1_transpose.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x1_transpose.y.repeats = {A, B, C, E, D};
  *x1_transpose.y.strides = {B * C * D * E, D * E, C * D * E, ONE, E};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A, B, C, D, E};
  *x2Local.y.strides = {B * C * D * E, D * E, C * D * E, E, ONE};

  ge::ascir_op::Transpose x2_transpose(("x2_transpose" + extern_name).c_str());
  x2_transpose.x = x2Local.y;
  x2_transpose.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  *x2_transpose.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x2_transpose.y.repeats = {A, B, C, E, D};
  *x2_transpose.y.strides = {B * C * D * E, D * E, C * D * E, ONE, E};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1_transpose.y;
  add.x2 = x2_transpose.y;
  add.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  *add.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *add.y.repeats = {A, B, C, E, D};
  *add.y.strides = {B * C * D * E, D * E, C * D * E, ONE, E};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x_store.y.repeats = {A, B, C, E, D};
  *x_store.y.strides = {B * C * D * E, D * E, C * D * E, ONE, E};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, c.id, b.id, e.id, d.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, c.id, b.id, e.id, d.id};
  *x_out.y.repeats = {A, B, C, E, D};
  *x_out.y.strides = {B * C * D * E, D * E, C * D * E, ONE, E};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatMergeAdd1AscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A * B);
  auto c = graph.CreateAxis("C", C);
  auto d = graph.CreateAxis("D", D);
  auto e = graph.CreateAxis("E", E);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, c.id, d.id, e.id};
  *x1Local.y.axis = {a.id, c.id, d.id, e.id};
  *x1Local.y.repeats = {A * B, C, D, E};
  *x1Local.y.strides = {C * D * E, D * E, E, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, c.id, d.id, e.id};
  *x2Local.y.axis = {a.id, c.id, d.id, e.id};
  *x2Local.y.repeats = {A * B, C, D, E};
  *x2Local.y.strides = {C * D * E, D * E, E, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, c.id, d.id, e.id};
  *add.y.axis = {a.id, c.id, d.id, e.id};
  *add.y.repeats = {A * B, C, D, E};
  *add.y.strides = {C * D * E, D * E, E, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, c.id, d.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, c.id, d.id, e.id};
  *x_store.y.repeats = {A * B, C, D, E};
  *x_store.y.strides = {C * D * E, D * E, E, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, c.id, d.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, c.id, d.id, e.id};
  *x_out.y.repeats = {A * B, C, D, E};
  *x_out.y.strides = {C * D * E, D * E, E, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatMergeAdd2AscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression E = graph.CreateSizeVar("E");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C * D);
  auto e = graph.CreateAxis("E", E);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, e.id};
  *x1Local.y.repeats = {A, B, C * D, E};
  *x1Local.y.strides = {B * C * D * E, C * D * E, E, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, e.id};
  *x2Local.y.repeats = {A, B, C * D, E};
  *x2Local.y.strides = {B * C * D * E, C * D * E, E, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, e.id};
  *add.y.axis = {a.id, b.id, c.id, e.id};
  *add.y.repeats = {A, B, C * D, E};
  *add.y.strides = {B * C * D * E, C * D * E, E, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, e.id};
  *x_store.y.repeats = {A, B, C * D, E};
  *x_store.y.strides = {B * C * D * E, C * D * E, E, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, e.id};
  *x_out.y.repeats = {A, B, C * D, E};
  *x_out.y.strides = {B * C * D * E, C * D * E, E, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

std::shared_ptr<AscGraph> CreatMergeAddErrAscGraph(ge::AscGraph &graph, bool is_incremental = true) {
  auto ONE = Symbol(1);
  const Expression A = graph.CreateSizeVar("A");
  const Expression B = graph.CreateSizeVar("B");
  const Expression C = graph.CreateSizeVar("C");
  const Expression D = graph.CreateSizeVar("D");
  const Expression F = graph.CreateSizeVar("F");

  auto a = graph.CreateAxis("A", A);
  auto b = graph.CreateAxis("B", B);
  auto c = graph.CreateAxis("C", C * D);
  auto e = graph.CreateAxis("F", F);
  std::string extern_name;
  if (is_incremental) {
    extern_name = std::to_string(AutofuseUtils::GenUniqueNumber());
  }
  ge::ascir_op::Data x1(("x1_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x1Local(("x1Local_add" + extern_name).c_str());
  x1Local.x = x1.y;
  x1Local.attr.sched.axis = {a.id, b.id, c.id, e.id};
  *x1Local.y.axis = {a.id, b.id, c.id, e.id};
  *x1Local.y.repeats = {A, B, C * D, F};
  *x1Local.y.strides = {B * C * D * F, C * D * F, F, ONE};

  ge::ascir_op::Data x2(("x2_add" + extern_name).c_str(), graph);

  ge::ascir_op::Load x2Local(("x2Local_add" + extern_name).c_str());
  x2Local.x = x2.y;
  x2Local.attr.sched.axis = {a.id, b.id, c.id, e.id};
  *x2Local.y.axis = {a.id, b.id, c.id, e.id};
  *x2Local.y.repeats = {A, B, C * D, F};
  *x2Local.y.strides = {B * C * D * F, C * D * F, F, ONE};

  ge::ascir_op::Add add(("add_add" + extern_name).c_str());
  add.x1 = x1Local.y;
  add.x2 = x2Local.y;
  add.attr.sched.axis = {a.id, b.id, c.id, e.id};
  *add.y.axis = {a.id, b.id, c.id, e.id};
  *add.y.repeats = {A, B, C * D, F};
  *add.y.strides = {B * C * D * F, C * D * F, F, ONE};

  ge::ascir_op::Store x_store(("x_store_add" + extern_name).c_str());
  x_store.x = add.y;
  x_store.attr.sched.axis = {a.id, b.id, c.id, e.id};
  x_store.attr.sched.loop_axis = c.id;
  *x_store.y.axis = {a.id, b.id, c.id, e.id};
  *x_store.y.repeats = {A, B, C * D, F};
  *x_store.y.strides = {B * C * D * F, C * D * F, F, ONE};

  ge::ascir_op::Output x_out(("x_out_add" + extern_name).c_str());
  x_out.x = x_store.y;
  x_out.attr.sched.axis = {a.id, b.id, c.id, e.id};
  x_out.attr.sched.loop_axis = c.id;
  *x_out.y.axis = {a.id, b.id, c.id, e.id};
  *x_out.y.repeats = {A, B, C * D, F};
  *x_out.y.strides = {B * C * D * F, C * D * F, F, ONE};

  auto x_out_node = graph.FindNode(("x_out_add" + extern_name).c_str());
  auto compute_graph = x_out_node->GetOwnerComputeGraph();
  std::vector<std::pair<NodePtr, int32_t>> output_nodes{{x_out_node, 0}};
  compute_graph->SetOutputSize(1U);
  compute_graph->SetGraphOutNodesInfo(output_nodes);
  return std::shared_ptr<ge::AscGraph>(new ge::AscGraph(graph));
}

Status FindReduceType(const NodePtr &node, const NodePtr &asc_node, const ComputeGraphPtr &graph,
                      const AscTensorAttr *output_desc_tensor_attr, AutoFuseAttrs *attr) {
  for (size_t index = 0U; index < output_desc_tensor_attr->axis.size(); index++) {
    if ((output_desc_tensor_attr->repeats[index] == kSymbolOne) &&
        (output_desc_tensor_attr->strides[index] == kSymbolZero)) {
      auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
      GE_ASSERT_NOTNULL(graph_attr);
      auto axis_id = output_desc_tensor_attr->axis[index];
      for (const auto &graph_axis : graph_attr->axis) {
        if ((graph_axis->id == axis_id) && (graph_axis->size != kSymbolOne)) {
          GELOGI("node: %s(%s) have reduction node: %s(%s).", node->GetNamePtr(), node->GetType().c_str(),
                 asc_node->GetNamePtr(), asc_node->GetType().c_str());
          GetInterAttrs(attr).fuse_type |= (1UL << static_cast<uint64_t>(loop::FuseType::kReduction));
          break;
        }
      }
    }
  }
  return SUCCESS;
}

Status CompleteAutoFuseAttr(const NodePtr &node) {
  auto op_desc = node->GetOpDescBarePtr();
  GE_ASSERT_NOTNULL(op_desc);
  auto attr = op_desc->GetOrCreateAttrsGroup<AutoFuseAttrs>();
  GE_ASSERT_NOTNULL(attr);
  auto asc_graph = attr->GetAscGraph();
  GE_ASSERT_NOTNULL(asc_graph);
  const auto graph = AscGraphUtils::GetComputeGraph(*asc_graph);
  GetInterAttrs(attr).fuse_type = (1UL << static_cast<uint64_t>(loop::FuseType::kPointwise));

  for (auto &asc_node : graph->GetAllNodes()) {
    if (asc_node->GetType() == kTransposeType) {
      GetInterAttrs(attr).fuse_type |= (1UL << static_cast<uint64_t>(loop::FuseType::kTranspose));
      GELOGI("node: %s(%s) have transpose node: %s(%s).", node->GetNamePtr(), node->GetType().c_str(),
             asc_node->GetNamePtr(), asc_node->GetType().c_str());
      continue;
    }
    if (asc_node->GetType() == kConcatType) {
      GetInterAttrs(attr).fuse_type |= (1UL << static_cast<uint64_t>(loop::FuseType::kConcat));
      GELOGI("node: %s(%s) have concat node: %s(%s).", node->GetNamePtr(), node->GetType().c_str(),
             asc_node->GetNamePtr(), asc_node->GetType().c_str());
      continue;
    }
    GE_ASSERT_NOTNULL(asc_node);
    auto asc_node_op_desc = asc_node->GetOpDesc();
    GE_ASSERT_NOTNULL(asc_node_op_desc);

    for (auto &output_desc : asc_node_op_desc->GetAllOutputsDescPtr()) {
      GE_ASSERT_NOTNULL(output_desc);
      auto output_desc_tensor_attr = output_desc->GetAttrsGroup<AscTensorAttr>();
      GE_ASSERT_NOTNULL(output_desc_tensor_attr);
      GE_ASSERT_TRUE(output_desc_tensor_attr->axis.size() == output_desc_tensor_attr->repeats.size());
      GE_ASSERT_TRUE(output_desc_tensor_attr->axis.size() == output_desc_tensor_attr->strides.size());
      // 如果tensor的repeat是1，stride是0，且graph上对应的轴size不为1，说明是reduce node
      GE_ASSERT_SUCCESS(FindReduceType(node, asc_node, graph, output_desc_tensor_attr, attr));
    }
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

// elementwise和elementwise可以融合
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_CanFuse1_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  // elementwise和elementwise可以融合
  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph));
  attr2->SetAscGraph(CreatSubAscGraph(sub_graph));
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
}

// elementwise和reduce可以融合
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_CanFuse2_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  attr1->SetAscGraph(CreatAddAscGraph(add_graph));
  ge::AscGraph reduce_graph("reduce");
  attr2->SetAscGraph(CreatReduceAscGraph(reduce_graph));
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
}

// reduce+element可以融合
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_CanFuse3_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  ASSERT_NE(compute_graph, nullptr);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
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
  attr2->SetAscGraph(CreatAbsAfterReduceAscGraph(abs_graph2), loop::FuseType::kPointwise);
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
}

/**
 *
 *          netoutput1
 *        /   \      \.
 *     add1  assign1   \.
 *    /   \  /     \     \.
 * var1  var2    const1  var3
 *******************
 *          netoutput1
 *        /   \      \.
 *       AscBc_1      \.
 *     /   \    \      \.
 * var1  var2  const1   var3
 */
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge1_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph2("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  auto add1 = compute_graph->FindNode("add1");
  ASSERT_NE(add1, nullptr);
  auto assign1 = compute_graph->FindNode("assign1");
  ASSERT_NE(assign1, nullptr);

  auto op_desc1 = add1->GetOpDescBarePtr();
  auto op_desc2 = assign1->GetOpDescBarePtr();
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
  CompleteAutoFuseAttr(add1);
  CompleteAutoFuseAttr(assign1);
  ASSERT_EQ(decider.CanFuseVertical(add1, assign1), true);
  auto fused_node = decider.Fuse(add1, assign1, nullptr);
  ASSERT_NE(fused_node, nullptr);

  auto fused_desc = fused_node->GetOpDescBarePtr();
  ASSERT_NE(fused_desc, nullptr);
  auto fused_attr = fused_desc->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attr, nullptr);
  ASSERT_NE(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph())), nullptr);

  EXPECT_EQ(AscGraphUtils::GetComputeGraph(*(fused_attr->GetAscGraph()))->GetAllNodesSize(), 13);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 6);
  fused_node = nullptr;
  for (auto &node : compute_graph->GetAllNodes()) {
    auto op_desc = node->GetOpDescBarePtr();
    ASSERT_NE(op_desc, nullptr);
    auto attr = op_desc->GetAttrsGroup<AutoFuseAttrs>();
    if (attr != nullptr) {
      fused_node = node;
      break;
    }
  }
  ASSERT_NE(fused_node, nullptr);

  ASSERT_EQ(fused_node->GetInDataNodes().size(), 3);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 2);
}

// reduce+element可以垂直融合
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge2_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  ASSERT_NE(compute_graph, nullptr);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
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
  attr2->SetAscGraph(CreatAbsAfterReduceAscGraph(abs_graph2), loop::FuseType::kPointwise);
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
  auto fused_node = decider.Fuse(addn1, shape1, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);
}

// reduce+broadcast可以垂直融合
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge3_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  ASSERT_NE(compute_graph, nullptr);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
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
  attr2->SetAscGraph(CreatAbsBroadcastAfterReduceAscGraph(abs_graph2), loop::FuseType::kPointwise);
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
  auto fused_node = decider.Fuse(addn1, shape1, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);
}

// broadcast + broadcast可以垂直融合
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge4_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  ASSERT_NE(compute_graph, nullptr);
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
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
  attr1->SetAscGraph(CreatBroadcastAddAscGraph(add_graph1));
  ge::AscGraph abs_graph2("abs");
  attr2->SetAscGraph(CreatBroadcastAbsAfterBroadcastAscGraph(abs_graph2));
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
  auto fused_node = decider.Fuse(addn1, shape1, nullptr);
  ASSERT_NE(fused_node, nullptr);
  ASSERT_EQ(fused_node->GetInDataNodes().size(), 2);
  ASSERT_EQ(fused_node->GetOutDataNodes().size(), 1);
}

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_FuseLoopMerge5_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph3("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  auto add1 = compute_graph->FindNode("add1");
  ASSERT_NE(add1, nullptr);
  auto huber_loss1 = compute_graph->FindNode("huberLoss1");
  ASSERT_NE(huber_loss1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = add1->GetOpDescBarePtr();
  auto op_desc2 = huber_loss1->GetOpDescBarePtr();
  auto op_desc3 = shape1->GetOpDescBarePtr();
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
  attr1->SetAscGraph(CreatAddAscGraph(add_graph1));
  ge::AscGraph calcRstd_graph2("calcRstd");
  attr2->SetAscGraph(CreatCalcRstdAscGraph(calcRstd_graph2));
  ge::AscGraph abs_graph2("abs");
  attr3->SetAscGraph(CreatReduceAscGraph(abs_graph2));
  CompleteAutoFuseAttr(huber_loss1);
  CompleteAutoFuseAttr(add1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(huber_loss1, add1), true);
  auto fused_node = decider.Fuse(huber_loss1, add1, nullptr);
  fused_node = decider.Fuse(fused_node, shape1, nullptr);
  ASSERT_NE(fused_node, nullptr);
}
#if 0
TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_Transpose_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatTransposeAscGraph(add_graph));
  attr2->SetAscGraph(CreatTransposeAfterTransposeAscGraph(sub_graph));
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
  auto fused_node = decider.Fuse(addn1, shape1, nullptr);
  ASSERT_NE(fused_node, nullptr);
}

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_Transpose1_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph2("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  auto add1 = compute_graph->FindNode("add1");
  ASSERT_NE(add1, nullptr);
  auto assign1 = compute_graph->FindNode("assign1");
  ASSERT_NE(assign1, nullptr);

  auto op_desc1 = add1->GetOpDescBarePtr();
  auto op_desc2 = assign1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatTransposeAscGraph(add_graph));
  attr2->SetAscGraph(CreatTransposeAfterTranspose1AscGraph(sub_graph));
  CompleteAutoFuseAttr(add1);
  CompleteAutoFuseAttr(assign1);
  ASSERT_EQ(decider.CanFuseVertical(add1, assign1), true);
  auto fused_node = decider.Fuse(add1, assign1, nullptr);
  ASSERT_NE(fused_node, nullptr);
}
#endif

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_AxisMerge_Ok) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  // elementwise和elementwise可以融合
  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatMergeAdd1AscGraph(add_graph));
  attr2->SetAscGraph(CreatMergeAdd2AscGraph(sub_graph));
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), true);
  auto fused_node = decider.Fuse(addn1, shape1, nullptr);
  ASSERT_NE(fused_node, nullptr);
}

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_CanFuse_failed) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);
  auto netoutput = compute_graph->FindNode("netoutput");
  ASSERT_NE(netoutput, nullptr);

  ASSERT_EQ(decider.CanFuseVertical(netoutput, shape1), false);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), false);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatMergeAdd1AscGraph(add_graph), loop::FuseType::kConcat);
  attr2->SetAscGraph(CreatConcatAscGraph(sub_graph), loop::FuseType::kConcat);
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), false);

  attr1->SetAscGraph(CreatMergeAdd1AscGraph(add_graph));
  attr2->SetAscGraph(CreatMergeAddErrAscGraph(sub_graph));
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), false);

  attr1->SetAscGraph(CreatMergeAdd1AscGraph(add_graph));
  attr2->SetAscGraph(CreatMergeAdd2AscGraph(sub_graph));
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), false);
}

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_Transpose_failed) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatTranspose1AscGraph(add_graph));
  attr2->SetAscGraph(CreatTransposeAfterTransposeAscGraph(sub_graph));
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), false);
}

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_Transpose1_failed) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph2("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  auto add1 = compute_graph->FindNode("add1");
  ASSERT_NE(add1, nullptr);
  auto assign1 = compute_graph->FindNode("assign1");
  ASSERT_NE(assign1, nullptr);

  auto op_desc1 = add1->GetOpDescBarePtr();
  auto op_desc2 = assign1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatInvalidTransposeAscGraph(add_graph));
  attr2->SetAscGraph(CreatTransposeAfterTranspose1AscGraph(sub_graph));
  CompleteAutoFuseAttr(add1);
  CompleteAutoFuseAttr(assign1);
  ASSERT_EQ(decider.CanFuseVertical(add1, assign1), false);
}

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_Transpose2_failed) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph2("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 7);
  auto add1 = compute_graph->FindNode("add1");
  ASSERT_NE(add1, nullptr);
  auto assign1 = compute_graph->FindNode("assign1");
  ASSERT_NE(assign1, nullptr);

  auto op_desc1 = add1->GetOpDescBarePtr();
  auto op_desc2 = assign1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatInvalidTranspose1AscGraph(add_graph));
  attr2->SetAscGraph(CreatTransposeAfterTranspose1AscGraph(sub_graph));
  CompleteAutoFuseAttr(add1);
  CompleteAutoFuseAttr(assign1);
  ASSERT_EQ(decider.CanFuseVertical(add1, assign1), false);
}

TEST_F(AscBackendTorchFusionDeciderTest, AscBackendFusionDecider_Transpose3_failed) {
  AscBackendTorchFusionDecider decider;
  ComputeGraphPtr compute_graph = BuildGraph1("AscBackend");
  EXPECT_EQ(compute_graph->GetAllNodesSize(), 5);
  auto addn1 = compute_graph->FindNode("addn1");
  ASSERT_NE(addn1, nullptr);
  auto shape1 = compute_graph->FindNode("shape1");
  ASSERT_NE(shape1, nullptr);

  auto op_desc1 = addn1->GetOpDescBarePtr();
  auto op_desc2 = shape1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  ASSERT_NE(op_desc2, nullptr);
  auto attr1 = GetOrCreateAutoFuseAttrs(op_desc1);
  auto attr2 = GetOrCreateAutoFuseAttrs(op_desc2);
  ASSERT_NE(attr1, nullptr);
  ASSERT_NE(attr2, nullptr);

  ge::AscGraph add_graph("add");
  ge::AscGraph sub_graph("sub");
  attr1->SetAscGraph(CreatTransposeAscGraph(add_graph));
  attr2->SetAscGraph(CreatTransposeAfterTransposeAscGraph(sub_graph));
  const auto graph = AscGraphUtils::GetComputeGraph(add_graph);
  auto graph_attr = graph->GetAttrsGroup<AscGraphAttr>();
  ASSERT_NE(graph_attr, nullptr);
  for (auto &axis_info : graph_attr->axis) {
    axis_info->id = 100;
    break;
  }
  CompleteAutoFuseAttr(addn1);
  CompleteAutoFuseAttr(shape1);
  ASSERT_EQ(decider.CanFuseVertical(addn1, shape1), false);
}
}  // namespace ge
