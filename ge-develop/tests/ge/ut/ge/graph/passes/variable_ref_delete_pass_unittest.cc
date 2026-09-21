/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/variable_optimize/variable_ref_delete_op_pass.h"

#include <gtest/gtest.h>
#include <string>
#include "ge_graph_dsl/graph_dsl.h"
#include "ge/ut/ge/test_tools_task_info.h"

using namespace ge;

class UtestGraphPassesVariableRefDeletePass : public testing::Test {
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

/**
 * variable -- const
 *   \        /
 *   \       /
 *     assign
 *       |
 *       |
 *   variable_ref
 */
TEST_F(UtestGraphPassesVariableRefDeletePass, variable_ref_delete_pass_succ1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr variable_node = NodeBuilder("variable", VARIABLE)
                                  .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr const_node = NodeBuilder("const", CONSTANT)
                               .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .Build(graph);

  ge::NodePtr apply_assign_node = NodeBuilder("assign", ASSIGN)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  ge::NodePtr variable_ref_node = NodeBuilder("variable_ref", VARIABLE)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  std::string ref_var_src_var_name = "variable";
  ge::AttrUtils::SetStr(variable_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(apply_assign_node->GetOutDataAnchor(0), variable_ref_node->GetInDataAnchor(0));

  ge::VariableRefDeleteOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(apply_assign_node->GetOutDataNodes().size(), 0);
  EXPECT_EQ(SUCCESS, status);
}

TEST_F(UtestGraphPassesVariableRefDeletePass, variable_ref_delete_pass_succ2) {
  DEF_GRAPH(g1) {
    CHAIN(NODE("variable", VARIABLE)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("variable_ref", VARIABLE)->
          NODE("Output", NETOUTPUT));
    CHAIN(NODE("data", DATA)->NODE("PartitionedCall_0"));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  ge::AttrUtils::SetInt(graph->FindNode("data")->GetOpDesc(), "index", 0);

  DEF_GRAPH(g2) {
    CHAIN(NODE("sub/variable", VARIABLE)->NODE("sub/assign", ASSIGN)->NODE("sub/Output", NETOUTPUT));
    CHAIN(NODE("sub/data", DATA)->NODE("sub/assign"));
  };
  ComputeGraphPtr subGraph = ToComputeGraph(g2);
  subGraph->SetName("f");
  ge::AttrUtils::SetInt(subGraph->FindNode("sub/data")->GetOpDesc(), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);

  const auto parent_node = graph->FindNode("PartitionedCall_0");
  EXPECT_NE(parent_node, nullptr);
  parent_node->GetOpDesc()->AddSubgraphName("f");
  parent_node->GetOpDesc()->SetSubgraphInstanceName(0, subGraph->GetName());
  EXPECT_NE(parent_node->GetOpDesc()->MutableInputDesc(0), nullptr);
  const auto output_node = subGraph->FindNode("sub/Output");
  EXPECT_NE(output_node, nullptr);
  ge::AttrUtils::SetInt(output_node->GetOpDesc()->MutableInputDesc(0), ge::ATTR_NAME_PARENT_NODE_INDEX, 0);
  const auto variable_ref_node = graph->FindNode("variable_ref");
  EXPECT_NE(variable_ref_node, nullptr);
  ge::AttrUtils::SetStr(variable_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, "variable");

  subGraph->SetParentGraph(graph);
  subGraph->SetParentNode(parent_node);
  (void)graph->AddSubGraph(subGraph);

  ge::VariableRefDeleteOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(SUCCESS, status);
}

/**
 * variable -- const
 *   \        /
 *   \       /
 *     assign
 *       |
 *       |
 *   variable_ref
 */
TEST_F(UtestGraphPassesVariableRefDeletePass, variable_ref_delete_pass_fail1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr variable_node = NodeBuilder("variable", VARIABLE)
                                  .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr const_node = NodeBuilder("const", CONSTANT)
                               .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .Build(graph);

  ge::NodePtr apply_assign_node = NodeBuilder("assign", ASSIGN)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  ge::NodePtr variable_ref_node = NodeBuilder("variable_ref", VARIABLE)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  std::string ref_var_src_var_name = "wrong_variable";
  ge::AttrUtils::SetStr(variable_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::GraphUtils::AddEdge(variable_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(apply_assign_node->GetOutDataAnchor(0), variable_ref_node->GetInDataAnchor(0));

  ge::VariableRefDeleteOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(FAILED, status);
}

/**
 *     assign
 *       |
 *       |
 *   variable_ref
 */
TEST_F(UtestGraphPassesVariableRefDeletePass, variable_ref_delete_pass_fail2) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  ge::NodePtr apply_assign_node = NodeBuilder("assign", ASSIGN)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  ge::NodePtr variable_ref_node = NodeBuilder("variable_ref", VARIABLE)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  std::string ref_var_src_var_name = "variable";
  ge::AttrUtils::SetStr(variable_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::GraphUtils::AddEdge(apply_assign_node->GetOutDataAnchor(0), variable_ref_node->GetInDataAnchor(0));

  ge::VariableRefDeleteOpPass pass_;
  ge::Status status = pass_.Run(graph);
  EXPECT_EQ(FAILED, status);
}

/**
 * refdata -- const
 *   \        /
 *   \       /
 *     assign
 *       |
 *       |
 *   refdata_ref
 */
TEST_F(UtestGraphPassesVariableRefDeletePass, refdata_ref_delete_pass_succ1) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::NodePtr refdata_node = NodeBuilder("refdata", "RefData")
                                  .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                  .Build(graph);

  ge::NodePtr const_node = NodeBuilder("const", CONSTANT)
                               .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                               .Build(graph);

  ge::NodePtr apply_assign_node = NodeBuilder("assign", ASSIGN)
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  ge::NodePtr refdata_ref_node = NodeBuilder("refdata_ref", "RefData")
                                      .AddInputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .AddOutputDesc({2, 16, 2, 2}, FORMAT_NHWC, DT_FLOAT)
                                      .Build(graph);

  std::string ref_var_src_var_name = "refdata";
  ge::AttrUtils::SetStr(refdata_ref_node->GetOpDesc(), REF_VAR_SRC_VAR_NAME, ref_var_src_var_name);

  ge::GraphUtils::AddEdge(refdata_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), apply_assign_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(apply_assign_node->GetOutDataAnchor(0), refdata_ref_node->GetInDataAnchor(0));

  ge::VariableRefDeleteOpPass pass;
  ge::Status status = pass.Run(graph);
  EXPECT_EQ(apply_assign_node->GetOutDataNodes().size(), 0);
  EXPECT_EQ(SUCCESS, status);
}
