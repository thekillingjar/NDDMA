/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/passes/shape_optimize/no_use_reshape_remove_pass.h"

#include <gtest/gtest.h>

#include "common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/manager/graph_manager.h"
#include "graph/manager/graph_manager_utils.h"
#include "graph/op_desc.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "graph/passes/pass_manager.h"
#include "engines/manager/opskernel_manager/ops_kernel_manager.h"

using namespace std;
using namespace testing;
using namespace ge;

class UtestGraphNoUseReshapeRemovePass : public testing::Test {
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
 * data->expanddim->reshape1->squeeze->reshape4->sinh
 *                                      /
 *                                    const
 */
void make_reshape_graph(ComputeGraphPtr &graph) {
  ge::NodePtr node_data = NodeBuilder("Data4D", DATA).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr node_expanddim_1 = NodeBuilder("ExpandDim", EXPANDDIMS)
                                     .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                     .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                     .Build(graph);

  ge::NodePtr node_reshape_1 = NodeBuilder("Reshape_1", RESHAPE)
                                   .AddInputDesc({2, 1, 2, 2}, FORMAT_ND, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 2, 2}, FORMAT_ND, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_squeeze_1 = NodeBuilder("Squeeze", SQUEEZE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);
  ge::NodePtr node_const =
      NodeBuilder("const", CONSTANT).AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT).Build(graph);

  ge::NodePtr node_reshape_4 = NodeBuilder("Reshape_4", RESHAPE)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                   .AddOutputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                   .Build(graph);

  ge::NodePtr node_sinh_1 = NodeBuilder("sinh", SINH)
                                .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
                                .AddOutputDesc({2, 1, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
                                .Build(graph);

  GraphUtils::AddEdge(node_data->GetOutDataAnchor(0), node_expanddim_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_expanddim_1->GetOutDataAnchor(0), node_reshape_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_reshape_1->GetOutDataAnchor(0), node_squeeze_1->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_squeeze_1->GetOutDataAnchor(0), node_reshape_4->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node_reshape_4->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_reshape_4->GetOutDataAnchor(0), node_sinh_1->GetInDataAnchor(0));
}

TEST_F(UtestGraphNoUseReshapeRemovePass, node_to_be_delete_success) {
  ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
  make_reshape_graph(compute_graph);

  // normal case
  NoUseReshapeRemovePass no_use_reshape_pass;
  ge::NodePtr reshape_node = compute_graph->FindNode("Reshape_4");
  Status status = no_use_reshape_pass.Run(reshape_node);
  EXPECT_EQ(status, ge::SUCCESS);

  // not reshape node case
  ge::NodePtr squeeze_node = compute_graph->FindNode("Squeeze");
  status = no_use_reshape_pass.Run(squeeze_node);
  EXPECT_EQ(status, ge::SUCCESS);

  // ND
  ge::NodePtr reshape_node2 = compute_graph->FindNode("Reshape_1");
  status = no_use_reshape_pass.Run(reshape_node2);
  EXPECT_EQ(status, ge::SUCCESS);
}

TEST_F(UtestGraphNoUseReshapeRemovePass, test_exception) {
  // 1. Reshape in or out is unknown shape
  {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
    ge::NodePtr reshape_node = NodeBuilder("reshape", RESHAPE)
        .AddInputDesc({-1, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
        .AddOutputDesc({-1, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
        .Build(compute_graph);
    NoUseReshapeRemovePass no_use_reshape_pass;
    Status status = no_use_reshape_pass.Run(reshape_node);
    EXPECT_EQ(status, ge::SUCCESS);
    EXPECT_EQ(compute_graph->GetDirectNodesSize(), 1);
  }

  // 2. Shape size of reshape in or out is different
  {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
    ge::NodePtr reshape_node = NodeBuilder("reshape", RESHAPE)
        .AddInputDesc({2, 2, 2, 2}, FORMAT_NCHW, DT_FLOAT)
        .AddOutputDesc({2, 2, 2}, FORMAT_NCHW, DT_FLOAT16)
        .Build(compute_graph);
    NoUseReshapeRemovePass no_use_reshape_pass;
    Status status = no_use_reshape_pass.Run(reshape_node);
    EXPECT_EQ(status, ge::SUCCESS);
    EXPECT_EQ(compute_graph->GetDirectNodesSize(), 1);
  }

  // 3. Has not input desc or output desc
  {
    ge::ComputeGraphPtr compute_graph = std::make_shared<ComputeGraph>("test");
    ge::NodePtr reshape_node = NodeBuilder("reshape", RESHAPE).Build(compute_graph);
    NoUseReshapeRemovePass no_use_reshape_pass;
    Status status = no_use_reshape_pass.Run(reshape_node);
    EXPECT_EQ(status, ge::INTERNAL_ERROR);
    EXPECT_EQ(compute_graph->GetDirectNodesSize(), 1);
  }
}
