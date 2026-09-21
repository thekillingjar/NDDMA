/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <map>
#include <set>
#include <vector>

#include "gtest/gtest.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/multi_batch/subgraph_multi_dims_clone_pass.h"
#include "graph/passes/multi_batch/subgraph_multi_dims_pass.h"
#include "macro_utils/dt_public_unscope.h"

#include "framework/common/types.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder_utils.h"
#include "graph/passes/pass_manager.h"

namespace ge {
class SubgraphMultiDimsPassTest : public testing::Test {
 public:
  NodePtr MakeNode(const ComputeGraphPtr &graph, int in_num, int out_num, string name, string type) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, type);
    for (auto i = 0; i < in_num; ++i) {
      op_desc->AddInputDesc(test_desc);
    }
    for (auto i = 0; i < out_num; ++i) {
      op_desc->AddOutputDesc(test_desc);
    }
    return graph->AddNode(op_desc);
  }

  NodePtr MakeConstNode(const ComputeGraphPtr &graph, string name) {
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>(name, "Const");
    GeTensor tensor(test_desc);
    AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, tensor);
    op_desc->AddOutputDesc(test_desc);
    return graph->AddNode(op_desc);
  }

  void CreateRootGraph(ComputeGraphPtr &graph) {
    auto data0 = MakeNode(graph, 0, 1, "data_0", DATA);
    GeTensorDesc tensor_desc0(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
    data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);

    auto data1 = MakeNode(graph, 0, 1, "data_1", DATA);
    GeTensorDesc tensor_desc1(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
    data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc1);

    auto data2 = MakeNode(graph, 0, 1, "data_2", DATA);
    GeTensorDesc tensor_desc2(GeShape({1,-1,-1,224}), FORMAT_NCHW, DT_FLOAT);
    data2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc2);

    auto partitioned_call = MakeNode(graph, 3, 1, "partitioned_call", PARTITIONEDCALL);
    GeTensorDesc tensor_desc(GeShape({-1,-1,-1,224}), FORMAT_NCHW, DT_FLOAT);
    partitioned_call->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    partitioned_call->GetOpDesc()->UpdateInputDesc(1, tensor_desc1);
    partitioned_call->GetOpDesc()->UpdateInputDesc(2, tensor_desc2);
    partitioned_call->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);

    auto netoutput = MakeNode(graph, 1, 0, "netoutput", NETOUTPUT);
    netoutput->GetOpDesc()->UpdateInputDesc(0, tensor_desc);

    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), partitioned_call->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), partitioned_call->GetInDataAnchor(1));
    GraphUtils::AddEdge(data2->GetOutDataAnchor(0), partitioned_call->GetInDataAnchor(2));
    GraphUtils::AddEdge(partitioned_call->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
  }

  void CreateSubGraph(ComputeGraphPtr &graph) {
    ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>("nn_model");
    subgraph->SetParentNode(graph->FindNode("partitioned_call"));
    subgraph->SetParentGraph(graph);
    graph->AddSubgraph(subgraph->GetName(), subgraph);
    AttrUtils::SetBool(subgraph, ATTR_NAME_SUBGRAPH_IS_MULTI_DIMS, true);

    auto data0 = MakeNode(subgraph, 1, 1, "nn_data0", DATA);
    GeTensorDesc tensor_desc0(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
    data0->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);
    (void)AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);

    auto data1 = MakeNode(subgraph, 1, 1, "nn_data1", DATA);
    GeTensorDesc tensor_desc1(GeShape({1,-1,-1,224}), FORMAT_NCHW, DT_FLOAT);
    data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc1);
    data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc1);
    (void)AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 2);
    (void)AttrUtils::SetListInt(data1->GetOpDesc(), ATTR_NAME_OP_MULTI_DIMS_INPUT_DIMS,
                                {1,1,3,224, 1,10,6,224, 1,100,9,224});

    auto data2 = MakeNode(subgraph, 1, 1, "nn_data2", DATA);
    GeTensorDesc tensor_desc2(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
    data2->GetOpDesc()->UpdateInputDesc(0, tensor_desc2);
    data2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc2);
    (void)AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
    (void)AttrUtils::SetListInt(data2->GetOpDesc(), ATTR_NAME_OP_MULTI_DIMS_INPUT_DIMS,
                                {20,3,224,224, 40,3,224,224, 80,3,224,224});

    auto const_node = MakeConstNode(subgraph, "nn_const");

    auto add0 = MakeNode(subgraph, 2, 1, "nn_add0", ADD);
    add0->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
    add0->GetOpDesc()->UpdateInputDesc(1, GeTensorDesc());
    add0->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT));

    auto add1 = MakeNode(subgraph, 2, 1, "nn_add1", ADD);
    add1->GetOpDesc()->UpdateInputDesc(0, tensor_desc1);
    add1->GetOpDesc()->UpdateInputDesc(1, tensor_desc2);
    add1->GetOpDesc()->UpdateOutputDesc(0, GeTensorDesc(GeShape({-1,-1,-1,224}), FORMAT_NCHW, DT_FLOAT));

    auto netoutput = MakeNode(subgraph, 2, 0, "nn_netoutput", NETOUTPUT);
    netoutput->GetOpDesc()->UpdateInputDesc(0, GeTensorDesc(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT));
    netoutput->GetOpDesc()->UpdateInputDesc(1, GeTensorDesc(GeShape({-1,-1,-1,224}), FORMAT_NCHW, DT_FLOAT));

    GraphUtils::AddEdge(data0->GetOutDataAnchor(0), add0->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), add0->GetInDataAnchor(1));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), add1->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2->GetOutDataAnchor(0), add1->GetInDataAnchor(1));
    GraphUtils::AddEdge(add0->GetOutDataAnchor(0), netoutput->GetInDataAnchor(0));
    GraphUtils::AddEdge(add1->GetOutDataAnchor(0), netoutput->GetInDataAnchor(1));
  }

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(SubgraphMultiDimsPassTest, base_test_success) {
  PassManager pass_manager;
  pass_manager.AddPass("SubgraphMultiDimsClonePass", new (std::nothrow) SubgraphMultiDimsClonePass);
  pass_manager.AddPass("SubgraphMultiDimsPass", new (std::nothrow) SubgraphMultiDimsPass);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");

  CreateRootGraph(graph);
  CreateSubGraph(graph);

  auto node = graph->FindNode("partitioned_call");
  auto op_desc = node->GetOpDesc();
  AttrUtils::SetInt(op_desc, ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX, 0);

  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
}
}  // namespace ge
