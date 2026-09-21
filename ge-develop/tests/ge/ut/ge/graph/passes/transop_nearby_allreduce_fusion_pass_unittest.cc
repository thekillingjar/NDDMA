/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <gtest/gtest.h>
#include "macro_utils/dt_public_scope.h"
#include "graph/passes/format_optimize/transop_nearby_allreduce_fusion_pass.h"
#include "common/ge_inner_error_codes.h"
#include "graph/passes/standard_optimize/addn_pass.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/utils/graph_utils_ex.h"
#include "macro_utils/dt_public_unscope.h"
namespace ge {

namespace {

class NodeBuilder {
 public:
  NodeBuilder(const string &name, const string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }
  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, Format format, DataType data_type, size_t count = 1) {
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(GeShape(vector<int64_t>(shape)));
    tensor_desc.SetFormat(format);
    tensor_desc.SetDataType(data_type);
    for (size_t i = 0U; i < count; i++) {
      op_desc_->AddInputDesc(tensor_desc);
    }
    return *this;
  }
  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, Format format, DataType data_type,
                             size_t count = 1) {
    GeTensorDesc tensor_desc;
    tensor_desc.SetShape(GeShape(vector<int64_t>(shape)));
    tensor_desc.SetFormat(format);
    tensor_desc.SetDataType(data_type);
    for (size_t i = 0U; i < count; i++) {
      op_desc_->AddOutputDesc(tensor_desc);
    }
    return *this;
  }

  NodePtr Build(const ComputeGraphPtr &graph) {
    NodePtr node = graph->AddNode(op_desc_);
    return node;
  }

 private:
  OpDescPtr op_desc_;
};

ComputeGraphPtr GetGraph1() { return nullptr; }

ComputeGraphPtr GetGraph2() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = nullptr;
  graph->AddNode(node);
  return graph;
}

ComputeGraphPtr GetGraph3() {
  // HcomAllReduce
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodeBuilder("HcomAllreduce3", HCOMALLREDUCE).Build(graph);
  return graph;
}

ComputeGraphPtr GetGraph4() {
  ///     TransData
  ///         |
  ///  HcomAllReduce
  ///         |
  ///    TransData
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                           .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .Build(graph);
  NodePtr allreduce = NodeBuilder("allreduce45", HCOMALLREDUCE)
                          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(0));
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  return graph;
}

ComputeGraphPtr GetGraph5() {
  ///       relu
  ///         |
  ///    TransData
  ///         |
  ///   HcomAllReduce
  ///        |
  ///     TransData
  ///         |
  ///       relu
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1 = NodeBuilder("Relu1", RELU)
                      .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .Build(graph);
  NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                           .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .Build(graph);
  NodePtr allreduce = NodeBuilder("allreduce45", HCOMALLREDUCE)
                          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  NodePtr relu2 = NodeBuilder("Relu2", RELU)
                      .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .Build(graph);
  GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), transdata1->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(0));
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata2->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));
  return graph;
}

ComputeGraphPtr GetGraph6() {
  ///      relu
  ///        |
  ///    TransData
  ///        |
  ///  HcomAllReduce
  ///        |
  ///    TransData
  ///        |
  ///      relu
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr relu1 = NodeBuilder("Relu1", RELU)
                      .AddInputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                      .AddOutputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                      .Build(graph);
  NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                           .AddInputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                           .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .Build(graph);
  NodePtr allreduce = NodeBuilder("allreduce45", HCOMALLREDUCE)
                          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  NodePtr relu2 = NodeBuilder("Relu2", RELU)
                      .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                      .Build(graph);
  GraphUtils::AddEdge(relu1->GetOutDataAnchor(0), transdata1->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(0));
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  GraphUtils::AddEdge(transdata2->GetOutDataAnchor(0), relu2->GetInDataAnchor(0));
  return graph;
}

ComputeGraphPtr GetGraph7(size_t symmetric_transdata_num, size_t asymmetric_transdata_num, size_t paired_others_num) {
  ///     TransData   TransData     ...       MatMul      ...
  ///          \         |           /       /           /
  ///                HcomAllReduce
  ///          /         |           \       \           \.
 ///     TransData   TransData     ...       RealDiv     ...
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr allreduce =
      NodeBuilder("allreduce6", HCOMALLREDUCE)
          .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT, symmetric_transdata_num + asymmetric_transdata_num)
          .AddInputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT, paired_others_num)
          .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT, symmetric_transdata_num + asymmetric_transdata_num)
          .AddOutputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT, paired_others_num)
          .Build(graph);

  for (size_t i = 0; i < symmetric_transdata_num; i++) {
    NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                             .AddInputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                             .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .Build(graph);
    NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                             .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                             .Build(graph);
    GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(i));
    GraphUtils::AddEdge(allreduce->GetOutDataAnchor(i), transdata2->GetInDataAnchor(0));
  }

  for (size_t i = 0; i < asymmetric_transdata_num; i++) {
    NodePtr transdata1 = NodeBuilder("TransData1", TRANSDATA)
                             .AddInputDesc({1, 1, 1, 64}, FORMAT_NHWC, DT_FLOAT)
                             .AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .Build(graph);
    NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                             .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                             .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                             .Build(graph);
    GraphUtils::AddEdge(transdata1->GetOutDataAnchor(0), allreduce->GetInDataAnchor(i + symmetric_transdata_num));
    GraphUtils::AddEdge(allreduce->GetOutDataAnchor(i + symmetric_transdata_num), transdata2->GetInDataAnchor(0));
  }

  for (size_t i = 0; i < paired_others_num; i++) {
    NodePtr matmul = NodeBuilder("matmul", MATMUL)
                         .AddInputDesc({32, 5}, FORMAT_NCHW, DT_FLOAT)
                         .AddInputDesc({32, 64}, FORMAT_NCHW, DT_FLOAT)
                         .AddOutputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT)
                         .Build(graph);
    NodePtr realDiv = NodeBuilder("realDiv", REALDIV)
                          .AddInputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT)
                          .AddOutputDesc({5, 64}, FORMAT_NCHW, DT_FLOAT)
                          .Build(graph);
    GraphUtils::AddEdge(matmul->GetOutDataAnchor(0),
                        allreduce->GetInDataAnchor(i + symmetric_transdata_num + asymmetric_transdata_num));
    GraphUtils::AddEdge(allreduce->GetOutDataAnchor(i + symmetric_transdata_num + asymmetric_transdata_num),
                        realDiv->GetInDataAnchor(0));
  }
  return graph;
}

ComputeGraphPtr GetGraph8() {
  ///     TransData
  ///         |
  ///   HcomAllReduce
  ///         |
  ///     TransData
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr allreduce =
      NodeBuilder("allreduce45", HCOMALLREDUCE).AddOutputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  NodePtr transdata2 = NodeBuilder("TransData2", TRANSDATA)
                           .AddInputDesc({1, 64, 1, 1}, FORMAT_NCHW, DT_FLOAT)
                           .AddOutputDesc({1, 4, 1, 1, 16}, FORMAT_NC1HWC0, DT_FLOAT)
                           .Build(graph);
  GraphUtils::AddEdge(allreduce->GetOutDataAnchor(0), transdata2->GetInDataAnchor(0));
  return graph;
}

TEST(UtestTransopNearbyAllreduceFusionPass, test1_null_graph) {
  ComputeGraphPtr graph = GetGraph1();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), INTERNAL_ERROR);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test2_null_node) {
  ComputeGraphPtr graph = GetGraph2();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test3_OnlyAllreduce) {
  ComputeGraphPtr graph = GetGraph3();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 1);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test4_all_reduce_with_trans_data) {
  ///    TransData
  ///        |
  ///  HcomAllReduce
  ///         |
  ///    TransData
  ComputeGraphPtr graph = GetGraph4();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 1);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test5_all_reduce_with_asymmetric_trans_data_and_relu) {
  ///       relu
  ///         |
  ///     TransData
  ///         |
  ///    HcomAllReduce
  ///         |
  ///     TransData
  ///         |
  ///       relu
  ComputeGraphPtr graph = GetGraph5();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 3);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test6_all_reduce_with_asymmetric_trans_data_and_relu) {
  ///       relu
  ///         |
  ///     TransData
  ///         |
  ///   HcomAllReduce
  ///         |
  ///     TransData
  ///         |
  ///       relu
  ComputeGraphPtr graph = GetGraph6();
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test7_all_reduce_with_multiple_trans_datas_and_other_ops) {
  ///    TransData   TransData     ...       MatMul      ...
  ///        \         |           /       /           /
  ///              HcomAllReduce
  ///         /         |           \       \           \.
  ///    TransData   TransData     ...       RealDiv     ...
  size_t symmetric_transdata_num = 20;
  size_t asymmetric_transdata_num = 20;
  size_t paired_others_num = 20;
  ComputeGraphPtr graph = GetGraph7(symmetric_transdata_num, asymmetric_transdata_num, paired_others_num);
  GEPass ge_pass(graph);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  EXPECT_EQ(ge_pass.Run(names_to_pass), SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), (asymmetric_transdata_num + paired_others_num) * 2 + 1);
}

TEST(UtestTransopNearbyAllreduceFusionPass, test8_in_and_out_data_anchor_are_not_equal) {
  /// HcomAllReduce
  ///       |
  ///    TransData
  ComputeGraphPtr graph = GetGraph8();
  GEPass ge_pass(graph);
  graph->GetAllNodes().at(0)->SetOwnerComputeGraph(nullptr);
  TransOpNearbyAllreduceFusionPass transop_nearby_allreduce_fusion_pass;
  NamesToPass names_to_pass;
  names_to_pass.emplace_back("TransOpNearbyAllreduceFusionPass", &transop_nearby_allreduce_fusion_pass);
  Status ret = ge_pass.Run(names_to_pass);
  EXPECT_EQ(ret, FAILED);
}

TEST(UtestTransopNearbyAllreduceFusionPass, RunFailed) {
  TransOpNearbyAllreduceFusionPass pass;
  NodePtr node = nullptr;
  Status ret = pass.Run(node);
  EXPECT_EQ(ret, SUCCESS);
}

static ComputeGraphPtr BuildGraph() {
  DEF_GRAPH(transopGraph) {
                            CHAIN(NODE("trans0", TRANSDATA)->EDGE(0U, 0U)->NODE("trans1", TRANSDATA));
                            CHAIN(NODE("trans1", TRANSDATA)->EDGE(0U, 1U)->NODE("trans2", TRANSDATA));
                            CHAIN(NODE("trans2", TRANSDATA)->EDGE(0U, 1U)->NODE("trans3", TRANSDATA));

                            CHAIN(NODE("cast0", CAST)->EDGE(0U, 0U)->NODE("cast1", SWITCH));
                            CHAIN(NODE("cast1", CAST)->EDGE(0U, 0U)->NODE("cast2", CAST));
                            CHAIN(NODE("cast1", CAST)->EDGE(0U, 0U)->NODE("cast3", CAST));
                          };

  const auto graph = ToGeGraph(transopGraph);
  const auto compute_graph = GraphUtilsEx::GetComputeGraph(graph);
  compute_graph->TopologicalSorting();

  return compute_graph;
}

TEST(UtestTransopNearbyAllreduceFusionPass, IsSymmetricTransOpsFailed) {
  TransOpNearbyAllreduceFusionPass pass;
  auto graph = BuildGraph();
  auto trans1 = graph->FindNode("trans1");
  auto trans2 = graph->FindNode("trans2");
  auto ret = pass.IsSymmetricTransOps(nullptr, nullptr);
  EXPECT_EQ(ret, false);

  trans1->GetOpDesc()->UpdateInputDesc(0U, GeTensorDesc(GeShape({0, 1}), FORMAT_NCHW, DT_FLOAT));
  ret = pass.IsSymmetricTransOps(trans1, trans2);
  EXPECT_EQ(ret, false);

  trans1->GetOpDesc()->UpdateInputDesc(0U, GeTensorDesc(GeShape({0, 1}), FORMAT_NCHW, DT_INT8));
  ret = pass.IsSymmetricTransOps(trans1, trans2);
  EXPECT_EQ(ret, false);

  trans1->GetOpDesc()->UpdateInputDesc(0U, GeTensorDesc(GeShape({0, 1}), FORMAT_ND, DT_INT8));
  ret = pass.IsSymmetricTransOps(trans1, trans2);
  EXPECT_EQ(ret, false);
}

TEST(UtestTransopNearbyAllreduceFusionPass, RemoveNearbyPairedTransOpsFailed) {
  TransOpNearbyAllreduceFusionPass pass;
  auto ret = pass.RemoveNearbyPairedTransOps(nullptr);
  EXPECT_EQ(ret, FAILED);

  auto graph = BuildGraph();
  auto cast1 = graph->FindNode("cast1");
  ret = pass.RemoveNearbyPairedTransOps(cast1);
  EXPECT_EQ(ret, SUCCESS);

  auto anchor = cast1->impl_->in_data_anchors_[0U];
  cast1->impl_->in_data_anchors_[0U] = nullptr;
  ret = pass.RemoveNearbyPairedTransOps(cast1);
  EXPECT_EQ(ret, SUCCESS);
  cast1->impl_->in_data_anchors_[0U] = anchor;
}


}  // namespace
}  // namespace ge
