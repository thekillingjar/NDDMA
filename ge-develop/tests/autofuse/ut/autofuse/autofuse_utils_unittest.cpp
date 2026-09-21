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
#include <queue>
#include "utils/autofuse_utils.h"
#include "utils/autofuse_attrs.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "attribute_group/attr_group_symbolic_desc.h"
#include "common/ge_common/ge_types.h"
#include "utils/node_utils.h"
#include "can_fuse/backend/backend_utils.h"
#include "graph/detail/model_serialize_imp.h"
#include "ascir_ops.h"
#include "attribute_group/attr_group_shape_env.h"
#include "ascgen_log.h"
#include "ge_graph_dsl/graph_dsl.h"

namespace ge {
class AutofuseUtilsTest : public testing::Test {
 protected:
  void SetUp() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
  void TearDown() override {
    dlog_setlevel(ASCGEN_MODULE_NAME, DLOG_ERROR, 0);
  }
};

namespace {
class GraphInputShapeSourceStub : public Source {
public:
  GraphInputShapeSourceStub(int32_t input_data_idx, size_t dim_idx)
      : input_data_idx_(input_data_idx), dim_idx_(dim_idx) {}

  std::string GetSourceStr() const override {
    return R"([&]() -> int64_t {
      const auto *tensor = context->GetGraphInputTensor()" + std::to_string(input_data_idx_) + R"();
      if (tensor == nullptr) {
        return -1;
      }
      return tensor->GetOriginShape().GetDim()" + std::to_string(dim_idx_) + R"();
    }())";
  }

private:
  int32_t input_data_idx_;  // Data的index，描述symbol来自于graph输入中第几个输入data
  size_t dim_idx_;          // 描述symbol来自于data中对应shape的第几个dim
};

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
    for (uint32_t i = 0; i < node->GetAllInDataAnchorsSize(); ++i) {
      origin_input_names.emplace_back("origin_input" + std::to_string(i), i);
    }
    std::vector<std::pair<std::string, int32_t>> origin_output_names;
    for (uint32_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
      origin_output_names.emplace_back("origin_output" + std::to_string(i), i);
    }
    GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_input_names_ = origin_input_names;
    GetInterAttrs(GetOrCreateAutoFuseAttrs(node->GetOpDesc())).origin_output_names_ = origin_output_names;
  }
  return SUCCESS;
}
};

// node是AscBackendNode的场景
TEST_F(AutofuseUtilsTest, ConvertAscBackendNodeToAscGraphNode_ok1) {
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

  auto op_desc1 = node1->GetOpDescBarePtr();
  ASSERT_NE(op_desc1, nullptr);
  // 测试符号序列化
  op_desc1->MutableOutputDesc(0)->GetAttrsGroup<ge::SymbolicDescAttr>()->
      symbolic_tensor.MutableOriginSymbolShape().MutableDims().push_back(Symbol("s1"));

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  std::string output;
  EXPECT_EQ(AutofuseUtils::SerilizeAscBackend(node1.get(), output), SUCCESS);
}

// node是FusedAscBackendNode的场景
TEST_F(AutofuseUtilsTest, ConvertAscBackendNodeToAscGraphNode_ok2) {
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
  auto concat = OP_CFG(kAscBackendType).TensorDesc(FORMAT_ND, DT_FLOAT, {1,2,3,4}).InCnt(3).OutCnt(1).InNames({"x"})
                    .OutNames({"y"}).Build("Concat");
  DEF_GRAPH(g1) {
    CHAIN(NODE(data1)->EDGE(0, 0)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(0, 1)->NODE(addn1));
    CHAIN(NODE(data2)->EDGE(1, 1)->NODE(concat));
    CHAIN(NODE(data3)->EDGE(0, 2)->NODE(concat));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(concat));
    CHAIN(NODE(addn1)->EDGE(0, 0)->NODE(shape));
    CHAIN(NODE(shape)->EDGE(0, 0)->NODE("NetOutput", kNetOutputType));
    CHAIN(NODE(concat)->EDGE(0, 1)->NODE("NetOutput", kNetOutputType));
  };
  auto graph = ToComputeGraph(g1);
  for (const auto &node : graph->GetAllNodes()) {
    SetAttrsGroup(node);
  }
  EXPECT_EQ(graph->GetAllNodesSize(), 7);
  auto node1 = graph->FindNode("AddN1");
  ASSERT_NE(node1, nullptr);
  auto node2 = graph->FindNode("Concat");
  ASSERT_NE(node2, nullptr);

  AscBackendFusionDecider decider;
  auto fused_node = decider.Fuse(node1, node2, nullptr);
  ASSERT_EQ(fused_node->GetType(), kFusedAscBackendType);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  Symbol s0 = shape_env_attr->CreateSymbol(2, std::make_shared<GraphInputShapeSourceStub>(0, 0));
  std::string output;
  EXPECT_EQ(AutofuseUtils::SerilizeAscBackend(fused_node.get(), output), SUCCESS);
}

// node是非AscBackendNode与FusedAscBackendNode的场景
TEST_F(AutofuseUtilsTest, ConvertAscBackendNodeToAscGraphNode_ok3) {
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
  auto node1 = graph->FindNode("Data1");
  ASSERT_NE(node1, nullptr);

  auto shape_env_attr = graph->GetOrCreateAttrsGroup<ShapeEnvAttr>();
  ASSERT_NE(shape_env_attr, nullptr);
  std::string output;
  EXPECT_EQ(AutofuseUtils::SerilizeAscBackend(node1.get(), output), SUCCESS);
}

//测试dump图接口
TEST_F(AutofuseUtilsTest, DumpGEGraph_ok1) {
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

  const char_t *const kDumpGEGraph = "DUMP_GE_GRAPH";

  // DUMP_GE_GRAPH = 0或者大于等于4 -> 不需要dump
  setenv(kDumpGEGraph, "0", 1);
  AutofuseUtils::DumpGEGraph(graph, "", "");
  AutofuseUtils::DumpGEGraphLevel1(graph, "", "");
  setenv(kDumpGEGraph, "4", 1);
  AutofuseUtils::DumpGEGraph(graph, "", "");
  AutofuseUtils::DumpGEGraphLevel1(graph, "", "");
  // DUMP_GE_GRAPH = 1 -> 可以dump
  setenv(kDumpGEGraph, "1", 1);
  const std::string prefix = "tmp1";
  AutofuseUtils::DumpGEGraph(graph, prefix, "test_1");
  AutofuseUtils::DumpGEGraphLevel1(graph, "", "");
  system(("rm -rf " + prefix).c_str());
  unsetenv(kDumpGEGraph);
}

TEST_F(AutofuseUtilsTest, IsNoNeedDump_ok) {
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

  // 1.不设置DUMP_GE_GRAPH就不dump图，直接返回
  AutofuseUtils::DumpGraphToOnnx(*graph, "test", "");
  AutofuseUtils::DumpGraphToOnnxLevel1(*graph, "test", "");
  // 2.设置DUMP_GE_GRAPH=0
  const char_t *const kDumpGEGraph = "DUMP_GE_GRAPH";
  setenv(kDumpGEGraph, "0", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "test", "");
  AutofuseUtils::DumpGraphToOnnxLevel1(*graph, "test", "");
  unsetenv(kDumpGEGraph);
  // 3.设置DUMP_GE_GRAPH=4
  setenv(kDumpGEGraph, "4", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "test", "");
  AutofuseUtils::DumpGraphToOnnxLevel1(*graph, "test", "");
  unsetenv(kDumpGEGraph);
  // 4.设置DUMP_GE_GRAPH=1
  setenv(kDumpGEGraph, "1", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "test", "");
  AutofuseUtils::DumpGraphToOnnxLevel1(*graph, "test", "");
  system(("rm -rf " + std::string("./test")).c_str());
  unsetenv(kDumpGEGraph);
}

TEST_F(AutofuseUtilsTest, NoNeedDumpGraphBySuffix_ok) {
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

  const char_t *const kDumpGEGraph = "DUMP_GE_GRAPH";
  setenv(kDumpGEGraph, "2", 1);
  // 1.设置DUMP_GRAPH_LEVEL为字符串
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  setenv(kDumpGraphLevel, "a|b", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "test", "");
  system(("rm -rf " + std::string("./test")).c_str());
  unsetenv(kDumpGraphLevel);
  // 2.设置DUMP_GRAPH_LEVEL为数字1
  setenv(kDumpGraphLevel, "1", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "test", "");
  AutofuseUtils::DumpGraphToOnnxLevel1(*graph, "test", "");
  system(("rm -rf " + std::string("./test")).c_str());
  unsetenv(kDumpGraphLevel);
  // 3.设置DUMP_GRAPH_LEVEL为数字2
  setenv(kDumpGraphLevel, "2", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "test", "");
  system(("rm -rf " + std::string("./test")).c_str());
  unsetenv(kDumpGraphLevel);
  unsetenv(kDumpGEGraph);
}

TEST_F(AutofuseUtilsTest, NeedDumpOriginGraphOrPostProcessGraph_ok) {
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

  const char_t *const kDumpGEGraph = "DUMP_GE_GRAPH";
  setenv(kDumpGEGraph, "2", 1);
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  setenv(kDumpGraphLevel, "1", 1);
  // 1.不设置kExperimentalAutofusionDumpOriginGraph，dump_name为canfuse
  const char_t *const kExperimentalAutofusionDumpOriginGraph = "experimental_autofusion_dump_origin_graph";
  AutofuseUtils::DumpGraphToOnnx(*graph, "canfuse", "test_suffix");
  system(("rm -rf " + std::string("./canfuse")).c_str());
  // 2.不设置kExperimentalAutofusionDumpOriginGraph，dump_name为postprocess
  AutofuseUtils::DumpGraphToOnnx(*graph, "postprocess", "test_suffix");
  system(("rm -rf " + std::string("./postprocess")).c_str());
  // 3.设置kExperimentalAutofusionDumpOriginGraph=1
  setenv(kExperimentalAutofusionDumpOriginGraph, "1", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "canfuse", "test_suffix");
  system(("rm -rf " + std::string("./canfuse")).c_str());
  unsetenv(kExperimentalAutofusionDumpOriginGraph);
  // 4.设置kExperimentalAutofusionDumpOriginGraph=2，dump_name为canfuse
  setenv(kExperimentalAutofusionDumpOriginGraph, "2", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "canfuse", "test_suffix");
  system(("rm -rf " + std::string("./canfuse")).c_str());
  unsetenv(kExperimentalAutofusionDumpOriginGraph);
  // 5.设置kExperimentalAutofusionDumpOriginGraph=2，dump_name为postprocess
  setenv(kExperimentalAutofusionDumpOriginGraph, "2", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "postprocess", "test_suffix");
  system(("rm -rf " + std::string("./postprocess")).c_str());
  unsetenv(kExperimentalAutofusionDumpOriginGraph);
  unsetenv(kDumpGraphLevel);
  unsetenv(kDumpGEGraph);
}

TEST_F(AutofuseUtilsTest, GetDumpGraphPrefixAndCreateDir_ok) {
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

  const char_t *const kDumpGEGraph = "DUMP_GE_GRAPH";
  setenv(kDumpGEGraph, "2", 1);
  const char_t *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
  setenv(kDumpGraphLevel, "1", 1);
  const char_t *const kExperimentalAutofusionDumpOriginGraph = "experimental_autofusion_dump_origin_graph";
  setenv(kExperimentalAutofusionDumpOriginGraph, "1", 1);
  // 1.设置NPU_COLLECT_PATH
  const char_t *const kNpuCollectPath = "NPU_COLLECT_PATH";
  setenv(kNpuCollectPath, "test_npu_collection_path", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "canfuse", "test_suffix");
  system(("rm -rf " + std::string("./test_npu_collection_path")).c_str());
  unsetenv(kNpuCollectPath);
  // 2.不设置NPU_COLLECT_PATH，设置DUMP_GRAPH_PATH
  unsetenv(kNpuCollectPath);
  const char_t *const kDumpGraphPath = "DUMP_GRAPH_PATH";
  setenv(kDumpGraphPath, "test_dir", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "canfuse", "test_suffix");
  system(("rm -rf " + std::string("./test_dir")).c_str());
  unsetenv(kDumpGraphPath);
  // 3.不设置NPU_COLLECT_PATH，不设置DUMP_GRAPH_PATH，设置ASCEND_WORK_PATH
  const char_t *kAscendWorkPathEnvName = "ASCEND_WORK_PATH";
  setenv(kAscendWorkPathEnvName, "./test_ascend_work_path", 1);
  AutofuseUtils::DumpGraphToOnnx(*graph, "canfuse", "test_suffix");
  system(("rm -rf " + std::string("./test_ascend_work_path")).c_str());
  unsetenv(kAscendWorkPathEnvName);
  unsetenv(kExperimentalAutofusionDumpOriginGraph);
  unsetenv(kDumpGraphLevel);
  unsetenv(kDumpGEGraph);
}

}  // namespace ge