/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/pass_utils.h"

#include <gtest/gtest.h>
#include <vector>

#include "common/types.h"
#include "graph/types.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "host_kernels/kernel.h"
#include "host_kernels/kernel_factory.h"

using namespace ge;

class UtestGraphPassesPassUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                            ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                             ge::DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  ge::NodePtr Build(const ge::ComputeGraphPtr &graph) { return graph->AddNode(op_desc_); }

 private:
  ge::GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, ge::Format format = FORMAT_NCHW,
                                       ge::DataType data_type = DT_FLOAT) {
    GeShape ge_shape{std::vector<int64_t>(shape)};
    ge::GeTensorDescPtr tensor_desc = std::make_shared<ge::GeTensorDesc>();
    tensor_desc->SetShape(ge_shape);
    tensor_desc->SetFormat(format);
    tensor_desc->SetDataType(data_type);
    return tensor_desc;
  }

  ge::OpDescPtr op_desc_;
};

TEST_F(UtestGraphPassesPassUtils, set_out_node_weight) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // data
  ge::NodePtr node_data = NodeBuilder("data", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  // const
  ge::NodePtr node_const =
      NodeBuilder("const", CONSTANT).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);
  // relu
  ge::NodePtr node_relu = NodeBuilder("node_relu1", RELU)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);
  // sinh
  ge::NodePtr node_sinh = NodeBuilder("node_sinh", SINH)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);
  // relu
  ge::NodePtr node_relu2 = NodeBuilder("node_relu2", RELU)
                               .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                               .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                               .Build(graph);
  // sinh
  ge::NodePtr node_sinh2 = NodeBuilder("node_sinh2", SINH)
                               .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                               .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                               .Build(graph);

  // add edge
  ge::GraphUtils::AddEdge(node_data->GetOutControlAnchor(), node_const->GetInControlAnchor());
  ge::GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_sinh->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_relu2->GetInControlAnchor());
  ge::GraphUtils::AddEdge(node_relu2->GetOutDataAnchor(0), node_sinh2->GetInDataAnchor(0));

  for (auto node : graph->GetDirectNode()) {
    if (node->GetType() == CONSTANT) {
      int32_t weight[] = {1};
      GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
      GeTensorPtr tensor = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
      vector<GeTensorPtr> tensor_vec = {tensor};
      OpDescUtils::SetWeights(node, tensor_vec);
    }
    if (!node->GetOutDataNodes().empty()) {
      auto out_data_anchor = node->GetOutDataNodes().at(0)->GetOutDataAnchor(0);
      Status status = PassUtils::SetOutNodeWeight(out_data_anchor, node);
      EXPECT_EQ(SUCCESS, status);
    }
  }
}

// only some failure castes for coverage check
TEST_F(UtestGraphPassesPassUtils, is_constant_null) {
  ge::NodePtr node = nullptr;
  bool ret = PassUtils::IsConstant(node);
  EXPECT_EQ(false, ret);
}

TEST_F(UtestGraphPassesPassUtils, get_in_data_node_fail) {
  ge::NodePtr node = nullptr;
  NodePtr in_data_node = PassUtils::GetInDataNode(node, 0);
  EXPECT_EQ(nullptr, in_data_node);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // relu
  ge::NodePtr node_relu = NodeBuilder("relu", RELU)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);
  NodePtr data_node = PassUtils::GetInDataNode(node_relu, 1);
  EXPECT_EQ(nullptr, data_node);
}

TEST_F(UtestGraphPassesPassUtils, get_unique_in_data_anchor_index_failed) {
  int invalid_index = -1;
  ge::NodePtr node = nullptr;
  int status = PassUtils::GetUniqueInDataAnchorIndex(node);
  EXPECT_EQ(invalid_index, status);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // relu
  ge::NodePtr node_relu = NodeBuilder("relu", RELU)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);
  int ret = PassUtils::GetUniqueInDataAnchorIndex(node_relu);
  EXPECT_EQ(invalid_index, ret);
}

TEST_F(UtestGraphPassesPassUtils, unlink_node_with_ctrl_copy_fail) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  // relu
  ge::NodePtr node_relu = NodeBuilder("relu", RELU)
                              .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                              .Build(graph);
  Status status = PassUtils::UnlinkNodeWithControlCopy(node_relu, 1);
  EXPECT_EQ(ge::SUCCESS, status);
  Status ret = PassUtils::UnlinkNodeWithControlCopy(node_relu, 0);
  EXPECT_EQ(ge::FAILED, ret);
}

TEST_F(UtestGraphPassesPassUtils, null_input) {
  std::vector<NodePtr> deleted_nodes;
  std::vector<NodePtr> end_nodes;
  EXPECT_NE(PassUtils::RemoveInactiveBranchToMerge(nullptr, deleted_nodes, end_nodes), 0);
}

///   identity    switch
///          \    /
///           merge
///
TEST_F(UtestGraphPassesPassUtils, test_remove_switch_direct_link_merge) {
  auto builder = ut::GraphBuilder("test");
  auto identity = builder.AddNode("identity", IDENTITY, 0, 1);
  auto switch_node = builder.AddNode("switch", SWITCH, 0, 1);
  auto merge_node = builder.AddNode("merge", MERGE, 2, 1);
  builder.AddDataEdge(identity, 0, merge_node, 0);
  builder.AddDataEdge(switch_node, 0, merge_node, 1);
  auto graph = builder.GetGraph();

  std::vector<NodePtr> deleted_nodes;
  std::vector<NodePtr> end_nodes;
  Status ret = PassUtils::RemoveInactiveBranchToMerge(switch_node->GetOutDataAnchor(0), deleted_nodes, end_nodes);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(end_nodes.size(), 1);
  EXPECT_EQ(end_nodes[0]->GetType(), MERGE);
}
