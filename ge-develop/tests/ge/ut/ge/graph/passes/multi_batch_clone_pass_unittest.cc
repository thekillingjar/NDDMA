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
#include <set>
#include <string>

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/pass_manager.h"
#include "graph/utils/tensor_utils.h"
#include "common/context/local_context.h"
#include "graph/passes/multi_batch/multi_batch_pass.h"
#include "graph/preprocess/insert_op/insert_aipp_op_util.h"
#include "framework/omg/omg_inner_types.h"
#include "register/op_registry.h"
#include "graph_builder_utils.h"
#include "graph/preprocess/multi_batch_copy_graph.h"
#include "graph/preprocess/multi_batch_options.h"
#include "graph/passes/multi_batch/multi_batch_clone_pass.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/optimize/symbolic/infer_symbolic_shape/symbolic_shape_inference.h"
#include "graph/utils/tensor_adapter.h"

namespace ge{
class UtestMultiBatchClonePass : public testing::Test {
protected:
  void SetUp() {
    SetLocalOmgContext(domi::GetContext());
    GetLocalOmgContext().dynamic_image_size.clear();
    GetLocalOmgContext().dynamic_batch_size.clear();
    GetLocalOmgContext().dynamic_dims.clear();
  }
  void TearDown() {
    GetLocalOmgContext().dynamic_image_size.clear();
    GetLocalOmgContext().dynamic_batch_size.clear();
    GetLocalOmgContext().dynamic_node_type.clear();
    GetLocalOmgContext().dynamic_dims.clear();
  }

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

  NodePtr MakeConstNode(const ComputeGraphPtr &graph) {
    static uint32_t index = 0;
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>("dynamic_const_" + std::to_string(index++), "Const");
    op_desc->AddOutputDesc(test_desc);
    return graph->AddNode(op_desc);
  }

  NodePtr MakeVariableNode(const ComputeGraphPtr &graph) {
    static uint32_t index = 0;
    GeTensorDesc test_desc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    auto op_desc = std::make_shared<OpDesc>("variable" + std::to_string(index++), "Variable");
    op_desc->AddOutputDesc(test_desc);
    return graph->AddNode(op_desc);
  }

  void make_original_graph(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 1, "data", "Data");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto bn_conv1 = MakeNode(graph, 4, 1, "bn_conv1", "BNInference");
    {
      GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(2));
      auto const3= MakeConstNode(graph);
      GraphUtils::AddEdge(const3->GetOutDataAnchor(0), bn_conv1->GetInDataAnchor(3));
    }

    auto scale_conv1 = MakeNode(graph, 4, 1, "scale1", "Scale");
    {
      GraphUtils::AddEdge(bn_conv1->GetOutDataAnchor(0), scale_conv1->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), scale_conv1->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), scale_conv1->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(scale_conv1->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void GraphWithJustData(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 1, "data", "Data");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void GraphWithDataAndGetNext(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    auto add_node1 = MakeNode(graph, 2, 1, "add1", "Add");
    auto add_node2 = MakeNode(graph, 2, 1, "add2", "Add");
    auto add_node3 = MakeNode(graph, 2, 1, "add3", "Add");
    {
      auto data1 = MakeNode(graph, 1, 1, "cat", "Data");
      GeTensorDesc tensor_desc1(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      tensor_desc1.SetOriginShape(GeShape({1,3,224,224}));
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc1);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc1);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 1);

      auto data0 = MakeNode(graph, 1, 1, "bed", "Data");
      GeTensorDesc tensor_desc0(GeShape({1,1,224,224}), FORMAT_NCHW, DT_FLOAT);
      tensor_desc0.SetOriginShape(GeShape({1,1,224,224}));
      data0->GetOpDesc()->UpdateInputDesc(0, tensor_desc0);
      data0->GetOpDesc()->UpdateOutputDesc(0, tensor_desc0);
      AttrUtils::SetInt(data0->GetOpDesc(), ATTR_NAME_INDEX, 0);

      auto data2 = MakeNode(graph, 1, 1, "IteratorGetNext_data", "Data");
      GeTensorDesc tensor_desc2(GeShape({1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      tensor_desc2.SetOriginShape(GeShape({1,3,224,224}));
      data2->GetOpDesc()->UpdateInputDesc(0, tensor_desc2);
      data2->GetOpDesc()->UpdateOutputDesc(0, tensor_desc2);
      AttrUtils::SetInt(data2->GetOpDesc(), ATTR_NAME_INDEX, 2);

      auto data3 = MakeNode(graph, 1, 1, "IteratorGetNext_data1", "Data");
      GeTensorDesc tensor_desc3(GeShape({1,1,224,224}), FORMAT_NCHW, DT_FLOAT);
      tensor_desc3.SetOriginShape(GeShape({1,1,224,224}));
      data3->GetOpDesc()->UpdateInputDesc(0, tensor_desc3);
      data3->GetOpDesc()->UpdateOutputDesc(0, tensor_desc3);
      AttrUtils::SetInt(data3->GetOpDesc(), ATTR_NAME_INDEX, 3);

      auto data4 = MakeNode(graph, 1, 1, "apple", "Data");
      GeTensorDesc tensor_desc4(GeShape({3}), FORMAT_NCHW, DT_FLOAT);
      tensor_desc4.SetOriginShape(GeShape({3}));
      data4->GetOpDesc()->UpdateInputDesc(0, tensor_desc4);
      data4->GetOpDesc()->UpdateOutputDesc(0, tensor_desc4);
      AttrUtils::SetInt(data4->GetOpDesc(), ATTR_NAME_INDEX, 4);

      auto data5 = MakeNode(graph, 1, 1, "ear", "RefData");
      GeTensorDesc tensor_desc5(GeShape({3}), FORMAT_NCHW, DT_FLOAT);
      tensor_desc5.SetOriginShape(GeShape({3}));
      data5->GetOpDesc()->UpdateInputDesc(0, tensor_desc5);
      data5->GetOpDesc()->UpdateOutputDesc(0, tensor_desc5);
      AttrUtils::SetInt(data5->GetOpDesc(), ATTR_NAME_INDEX, 5);

      GetLocalOmgContext().user_input_dims = {std::make_pair(data4->GetOpDesc()->GetName(), vector<int64_t>{-1}),
                                              std::make_pair(data0->GetOpDesc()->GetName(), vector<int64_t>{-1,-1,224,224}),
                                              std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224}),
                                              std::make_pair(data5->GetOpDesc()->GetName(), vector<int64_t>{3})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      GraphUtils::AddEdge(data0->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      GraphUtils::AddEdge(data3->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
      GraphUtils::AddEdge(data2->GetOutDataAnchor(0), add_node1->GetInDataAnchor(0));
      GraphUtils::AddEdge(data4->GetOutDataAnchor(0), add_node1->GetInDataAnchor(1));
      GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), add_node2->GetInDataAnchor(0));
      GraphUtils::AddEdge(add_node1->GetOutDataAnchor(0), add_node2->GetInDataAnchor(1));
      GraphUtils::AddEdge(add_node2->GetOutDataAnchor(0), add_node3->GetInDataAnchor(0));
      GraphUtils::AddEdge(data5->GetOutDataAnchor(0), add_node3->GetInDataAnchor(1));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(add_node3->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void GraphWithGetNextNosink(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 1, "IteratorGetNext_data", "Data");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }
 
  // getnext has one data and has one out of shape
  void GraphWithGetNextSink(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 2, "data", "IteratorV2");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(1, shape_desc);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto identity = MakeNode(graph, 1, 0, "identity", "Identity");
      GraphUtils::AddEdge(data1->GetOutDataAnchor(1), identity->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  // getnext has one data and has one out of shape
  void GraphWithGetNextSinkWithSubgraph(const ComputeGraphPtr &graph) {
    auto data1 = MakeNode(graph, 1, 2, "data", "IteratorV2");
    GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
    data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
    data1->GetOpDesc()->UpdateOutputDesc(1, shape_desc);
    AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
    GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

    auto make_subgraph = [this](const std::string &name,
        const ComputeGraphPtr &parent_graph, const NodePtr &parent_node) {
      ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>(name);
      auto sub_data0 = MakeNode(sub_graph, 1, 1, "sub_data0", "Data");
      auto sub_data1 = MakeNode(sub_graph, 1, 1, "sub_data1", "Data");
      auto sub_add0 = MakeNode(sub_graph, 1, 1, "sub_add0", "Add");
      auto sub_output_node = MakeNode(sub_graph, 1, 0, "sub_output", "NetOutput");
      GraphUtils::AddEdge(sub_data0->GetOutDataAnchor(0), sub_add0->GetInDataAnchor(0));
      GraphUtils::AddEdge(sub_data1->GetOutDataAnchor(0), sub_add0->GetInDataAnchor(1));
      GraphUtils::AddEdge(sub_add0->GetOutDataAnchor(0), sub_output_node->GetInDataAnchor(0));
      (void)AttrUtils::SetInt(sub_data0->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 0);
      (void)AttrUtils::SetInt(sub_data1->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, 1);
      (void)AttrUtils::SetInt(sub_output_node->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_PARENT_NODE_INDEX, 0);
      sub_graph->SetParentGraph(parent_graph);
      sub_graph->SetParentNode(parent_node);
      return sub_graph;
    };

    auto if_node = MakeNode(graph, 2, 1, "if_node", "If");
    auto then_graph = make_subgraph("then_graph", graph, if_node);
    auto else_graph = make_subgraph("else_graph", graph, if_node);

    if_node->GetOpDesc()->AddSubgraphName("then_branch");
    if_node->GetOpDesc()->SetSubgraphInstanceName(0, "then_graph");
    if_node->GetOpDesc()->AddSubgraphName("else_branch");
    if_node->GetOpDesc()->SetSubgraphInstanceName(1, "else_graph");
    graph->AddSubgraph(then_graph);
    graph->AddSubgraph(else_graph);

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(data1->GetOutDataAnchor(0), if_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data1->GetOutDataAnchor(1), if_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(if_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void GraphWithVariableGetNextSink(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 1, "conv1", "Conv2D");
    {
      auto data1 = MakeNode(graph, 1, 2, "data", "IteratorV2");
      GeTensorDesc tensor_desc(GeShape({-1,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      GeTensorDesc shape_desc(GeShape({4,3,224,224}), FORMAT_NCHW, DT_FLOAT);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(1, shape_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);
      GetLocalOmgContext().user_input_dims = {std::make_pair(data1->GetOpDesc()->GetName(), vector<int64_t>{-1,3,224,224})};

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto identity = MakeNode(graph, 1, 0, "identity", "Identity");
      GraphUtils::AddEdge(data1->GetOutDataAnchor(1), identity->GetInDataAnchor(0));
      auto var1 = MakeVariableNode(graph);
      GraphUtils::AddEdge(var1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto var2 = MakeVariableNode(graph);
      GraphUtils::AddEdge(var2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto output_node = MakeNode(graph, 1, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  }

  void make_original_yolo2(const ComputeGraphPtr &graph) {
    auto conv2d_node = MakeNode(graph, 3, 2, "conv1", "Conv2D");
    {
      auto data1  = MakeNode(graph, 1, 1, "data", "Data");
      GeTensorDesc tensor_desc(GeShape({-1, 3, 224, 224}), FORMAT_NCHW, DT_FLOAT);
      TensorUtils::SetSize(tensor_desc, 224);
      data1->GetOpDesc()->UpdateInputDesc(0, tensor_desc);
      data1->GetOpDesc()->UpdateOutputDesc(0, tensor_desc);
      AttrUtils::SetInt(data1->GetOpDesc(), ATTR_NAME_INDEX, 0);

      GraphUtils::AddEdge(data1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(0));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(1));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0), conv2d_node->GetInDataAnchor(2));
    }

    auto  image_info = MakeNode(graph, 1, 1, "image_info", "Data");
    auto  detection1 = MakeNode(graph, 4, 1, "detection1", "YoloDetection");
    {
      GraphUtils::AddEdge(image_info->GetOutDataAnchor(0), detection1->GetInDataAnchor(0));
      GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(0), detection1->GetInDataAnchor(1));
      auto const1 = MakeConstNode(graph);
      GraphUtils::AddEdge(const1->GetOutDataAnchor(0),  detection1->GetInDataAnchor(2));
      auto const2 = MakeConstNode(graph);
      GraphUtils::AddEdge(const2->GetOutDataAnchor(0),  detection1->GetInDataAnchor(3));
    }

    auto output_node = MakeNode(graph, 3, 0, "output1", "NetOutput");
    GraphUtils::AddEdge(conv2d_node->GetOutDataAnchor(1), output_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(image_info->GetOutDataAnchor(0), output_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(detection1->GetOutDataAnchor(0), output_node->GetInDataAnchor(2));
  }
};

// graph is nullptr
TEST_F(UtestMultiBatchClonePass, graph_nullptr) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass(0));
  ComputeGraphPtr graph;
  EXPECT_EQ(pass_manager.Run(graph), PARAM_INVALID);
}

// graph with subgraph
TEST_F(UtestMultiBatchClonePass, graph_with_subgraph) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass(0));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_graph(graph);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);

  ComputeGraphPtr owner = std::make_shared<ComputeGraph>("test_owner");
  auto func_node = MakeNode(owner, 3, 1, "test_if", "If");
  graph->SetParentNode(func_node);
  graph->SetParentGraph(owner);
  owner->AddSubgraph(graph->GetName(), graph);
  size_t sub_graph_num = owner->GetAllSubgraphs().size();
  EXPECT_EQ(sub_graph_num, 1);
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
}

//graph is uncompute graph, not need to do multi batch
TEST_F(UtestMultiBatchClonePass, uncompute_graph) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_graph(graph);
  GetLocalOmgContext().need_multi_batch = false;
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
}


//compute_graph with data from DATA
TEST_F(UtestMultiBatchClonePass, compute_graph_with_data) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithDataAndGetNext(graph);
  GetLocalOmgContext().need_multi_batch = true;
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  GetLocalOmgContext().dynamic_node_type = DATA;
  GetLocalOmgContext().dynamic_dims = "1,3,3,1;2,1,3,3;4,2,6,6;8,1,1,2";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().data_nodes.size(), 4);

  EXPECT_EQ(GetLocalOmgContext().user_input_dims[0].first, "bed");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[1].first, "cat");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[2].first, "apple");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[3].first, "ear");
  std::vector<std::vector<int64_t>> batch_shape_expect = {{3,3,1,1},{1,3,3,2},{2,6,6,4},{1,1,2,8}};
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[0], batch_shape_expect[0]);
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[1], batch_shape_expect[1]);
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[2], batch_shape_expect[2]);
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[3], batch_shape_expect[3]);
}

TEST_F(UtestMultiBatchClonePass, compute_graph_with_data_empty_tensor) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithDataAndGetNext(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = DATA;
  GetLocalOmgContext().dynamic_dims = "0,3,3,1;0,1,3,3;0,2,6,6;0,1,1,2";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().data_nodes.size(), 4);
}

TEST_F(UtestMultiBatchClonePass, compute_graph_with_data_error) {
  std::vector<std::vector<int64_t>> shapes = {{-1,3,3,1},{-1,1,3,3}};
  EXPECT_EQ(ge::multibatch::CheckDynamicParams(shapes), PARAM_INVALID);
}

// compute_graph with data from DATA with symbolic
TEST_F(UtestMultiBatchClonePass, compute_graph_with_data_symbolic) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithDataAndGetNext(graph);
  GetLocalOmgContext().need_multi_batch = true;
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  GetLocalOmgContext().dynamic_node_type = DATA;
  GetLocalOmgContext().dynamic_dims = "1,3,3,1;2,1,3,3;4,2,6,6;8,1,1,2";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().data_nodes.size(), 4);

  EXPECT_EQ(GetLocalOmgContext().user_input_dims[0].first, "bed");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[1].first, "cat");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[2].first, "apple");
  EXPECT_EQ(GetLocalOmgContext().user_input_dims[3].first, "ear");
  std::vector<std::vector<int64_t>> batch_shape_expect = {{3,3,1,1},{1,3,3,2},{2,6,6,4},{1,1,2,8}};
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[0], batch_shape_expect[0]);
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[1], batch_shape_expect[1]);
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[2], batch_shape_expect[2]);
  EXPECT_EQ(GetLocalOmgContext().batch_shapes[3], batch_shape_expect[3]);
  std::vector<ge::GeTensor> input_vec;
  std::vector<int64_t> dims_vec0 = {1, 1, 224, 224};
  ge::Shape shape0({dims_vec0});
  ge::TensorDesc td0{shape0, ge::FORMAT_ND, DT_INT32};
  td0.SetOriginShape(shape0);
  ge::Tensor tensor0{td0};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor0));

  std::vector<int64_t> dims_vec1 = {1, 3, 224, 224};
  ge::Shape shape1({dims_vec1});
  ge::TensorDesc td1{shape1, ge::FORMAT_ND, DT_INT32};
  td1.SetOriginShape(shape1);
  ge::Tensor tensor1{td1};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor1));

  std::vector<int64_t> dims_vec2 = {1, 3, 224, 224};
  ge::Shape shape2({dims_vec2});
  ge::TensorDesc td2{shape2, ge::FORMAT_ND, DT_INT32};
  td2.SetOriginShape(shape2);
  ge::Tensor tensor2{td2};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor2));

  std::vector<int64_t> dims_vec3 = {1, 1, 224, 224};
  ge::Shape shape3({dims_vec3});
  ge::TensorDesc td3{shape3, ge::FORMAT_ND, DT_INT32};
  td3.SetOriginShape(shape3);
  ge::Tensor tensor3{td3};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor3));

  std::vector<int64_t> dims_vec4 = {3};
  ge::Shape shape4({dims_vec4});
  ge::TensorDesc td4{shape4, ge::FORMAT_ND, DT_INT32};
  td4.SetOriginShape(shape4);
  ge::Tensor tensor4{td4};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor4));

  std::vector<int64_t> dims_vec5 = {3};
  ge::Shape shape5({dims_vec5});
  ge::TensorDesc td5{shape5, ge::FORMAT_ND, DT_INT32};
  td5.SetOriginShape(shape5);
  ge::Tensor tensor5{td5};
  input_vec.emplace_back(ge::TensorAdapter::AsGeTensor(tensor5));
  ASSERT_EQ(SymbolicShapeSymbolizer::Symbolize(graph, input_vec), ge::SUCCESS);
}

//compute_graph with data from DATA
TEST_F(UtestMultiBatchClonePass, compute_graph_with_data_error_dynamic_dims) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithDataAndGetNext(graph);
  GetLocalOmgContext().need_multi_batch = true;
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  GetLocalOmgContext().dynamic_node_type = DATA;
  GetLocalOmgContext().dynamic_dims = "1,3,3,1:3:2;2,1,3,3;4,2,6,6;8,1,1,2";
  EXPECT_EQ(multi_batch_clone.Run(graph), PARAM_INVALID);
}

//compute_graph with data from GetNext_nosink
TEST_F(UtestMultiBatchClonePass, compute_graph_with_getnext_nosink) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithGetNextNosink(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().getnext_nosink_nodes.size(), 1);
}

//compute_graph with data from GetNext_nosink
TEST_F(UtestMultiBatchClonePass, compute_graph_with_getnext_sink) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithGetNextSink(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().getnext_nosink_nodes.size(), 0);
}

TEST_F(UtestMultiBatchClonePass, compute_graph_with_getnext_sink_with_subgraph) {
  MultiBatchClonePass multi_batch_clone(1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  graph->SetGraphID(1);
  GraphWithGetNextSinkWithSubgraph(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(graph->GetAllSubgraphs().size(), 2);
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(graph->GetGraphID(), 1);
  EXPECT_EQ(graph->GetSessionID(), 1);
  // EXPECT_EQ(graph->GetAllSubgraphs().size(), 12); // with metadef code
}

//no need multi batch in training
TEST_F(UtestMultiBatchClonePass, compute_graph_no_need_multi) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithJustData(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().getnext_nosink_nodes.size(), 0);
}

//compute_graph with GetNext and variable
TEST_F(UtestMultiBatchClonePass, compute_graph_with_getnext_sink_with_variable) {
  MultiBatchClonePass multi_batch_clone(0);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithVariableGetNextSink(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(multi_batch_clone.Run(graph), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().getnext_nosink_nodes.size(), 0);
}

TEST_F(UtestMultiBatchClonePass, compute_graph_switchn_no_need_multi) {
  setenv("MULTI_BATCH_WITH_SWITCHN", "1", 1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  GraphWithJustData(graph);
  GetLocalOmgContext().need_multi_batch = true;
  GetLocalOmgContext().dynamic_node_type = GETNEXT;
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  EXPECT_EQ(ge::multibatch::ProcessMultiBatch(graph, 0), SUCCESS);
  EXPECT_EQ(GetLocalOmgContext().getnext_nosink_nodes.size(), 0);
  unsetenv("MULTI_BATCH_WITH_SWITCHN");
}

TEST_F(UtestMultiBatchClonePass, DataOutputDescSizeCheck) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass(0));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_yolo2(graph);
  MultiBatchClonePass multi_batch_clone(0);
  GetLocalOmgContext().dynamic_batch_size = "1,2,4,8";
  EXPECT_EQ(pass_manager.Run(graph), SUCCESS);
  for (const auto &sub_graph : graph->GetAllSubgraphs()) {
    for (const auto &node : sub_graph->GetAllNodes()) {
      if (node->GetType() == "Data") {
        int64_t size = 0;
        auto tensor_desc = node->GetOpDesc()->GetOutputDesc(0);
        TensorUtils::GetSize(tensor_desc, size);
        EXPECT_TRUE(size == 0);
      }
    }
  }
}

TEST_F(UtestMultiBatchClonePass, DirectToNetOutput) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass(0));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_yolo2(graph);
  MultiBatchClonePass multi_batch_clone(0);
  GetLocalOmgContext().dynamic_dims = "1;2;4;8";
  GetLocalOmgContext().dynamic_batch_size = "1,2,4,8";
  EXPECT_EQ(pass_manager.Run(graph), PARAM_INVALID);
}

TEST_F(UtestMultiBatchClonePass, MaxDimsSizeCheck) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass(0));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_yolo2(graph);
  MultiBatchClonePass multi_batch_clone(0);
  GetLocalOmgContext().dynamic_batch_size = "1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,\
  1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,\
  1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,\
  1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,\
  1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8,1,2,4,8";
  EXPECT_EQ(pass_manager.Run(graph), PARAM_INVALID);
}

TEST_F(UtestMultiBatchClonePass, MinDimsSizeCheck) {
  PassManager pass_manager;
  pass_manager.AddPass("MultiBatchClonePass", new (std::nothrow) MultiBatchClonePass(0));
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph");
  make_original_yolo2(graph);
  MultiBatchClonePass multi_batch_clone(0);
  GetLocalOmgContext().dynamic_batch_size = "1";
  EXPECT_EQ(pass_manager.Run(graph), PARAM_INVALID);
}

}
