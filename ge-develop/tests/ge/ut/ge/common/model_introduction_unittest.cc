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

#include "compute_graph.h"
#include "graph/normal_graph/compute_graph_impl.h"
#include "framework/common/types.h"
#include "utils/graph_utils.h"
#include "graph/passes/graph_builder_utils.h"

#include "macro_utils/dt_public_scope.h"
#include "common/model/model_introduction.h"
#include "macro_utils/dt_public_unscope.h"

#include "graph/debug/ge_attr_define.h"
#include "common/share_graph.h"

namespace ge {
class UTestModelIntroduction : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}

  static ComputeGraphPtr BuildSubComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
    auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
    auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    return graph;
  }

  static ComputeGraphPtr BuildComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto data = builder.AddNode("Data", "Data", 1, 1);
    auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
    transdata->GetOpDesc()->AddSubgraphName("subgraph");
    transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, transdata, 0);
    builder.AddDataEdge(transdata, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    // add subgraph
    transdata->SetOwnerComputeGraph(graph);
    ComputeGraphPtr subgraph = BuildSubComputeGraph();
    subgraph->SetParentGraph(graph);
    subgraph->SetParentNode(transdata);
    graph->AddSubgraph("subgraph", subgraph);
    auto op_desc = netoutput->GetOpDesc();
    std::vector<std::string> src_name{"out"};
    op_desc->SetSrcName(src_name);
    std::vector<int64_t> src_index{0};
    op_desc->SetSrcIndex(src_index);
    return graph;
  }
};

TEST_F(UTestModelIntroduction, InitWithoutOpDescOfCaseType) {
  ModelIntroduction model_introduction;
  auto graph = BuildComputeGraph();
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  std::shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);
  EXPECT_EQ(model_introduction.Init(ge_model, true), SUCCESS);
}

TEST_F(UTestModelIntroduction, InitWithOpDescOfCaseType) {
  ModelIntroduction model_introduction;
  ComputeGraphPtr graph = gert::ShareGraph::CaseGraph();
  NodePtr case_node = graph->FindNode("case");
  OpDescPtr op_desc = case_node->GetOpDesc();
  int32_t dynamic_type[2] = {ge::DYNAMIC_BATCH, ge::DYNAMIC_DIMS};

  for (int32_t i = 0; i < 2; i++) {
    AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type[i]);
    uint32_t batch_num = 2U;
    AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num);
    for (uint32_t i = 0U; i < batch_num; ++i) {
      const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
      std::vector<int64_t> batch_shape = {1, 2};
      AttrUtils::SetListInt(op_desc, attr_name, batch_shape);
    }
    std::vector<std::string> user_designate_shape_order = {"data1", "data2"};
    AttrUtils::SetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order);

    GeModelPtr ge_model = std::make_shared<GeModel>();
    ge_model->SetGraph(graph);
    AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
    AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

    shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
    ge_model->SetModelTaskDef(model_task_def);
    EXPECT_EQ(model_introduction.Init(ge_model), SUCCESS);
  }
}

TEST_F(UTestModelIntroduction, CreateOutput)
{
  ModelIntroduction model_introduction;
  uint32_t index = 0;
  auto op_desc = std::make_shared<ge::OpDesc>("node", "node");
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_FRACTAL_Z);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(ge::GeShape({3, 12, 5, 6}));

  std::vector<std::pair<int64_t, int64_t>> range{{1, 5}, {2, 5}, {3, 6}, {4, 7}};
  tensor_desc.SetShapeRange(range);

  op_desc->AddInputDesc("x", tensor_desc);

  ge::ModelTensorDesc output;
  EXPECT_EQ(model_introduction.CreateOutput(index, op_desc, output), SUCCESS);
}

TEST_F(UTestModelIntroduction, ConstructOutputInfo) {
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("test", "test");
  std::vector<std::string> output_names;
  ge::GeTensorDesc tensor_desc;
  tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
  tensor_desc.SetFormat(ge::FORMAT_FRACTAL_Z);
  tensor_desc.SetDataType(ge::DT_FLOAT16);
  tensor_desc.SetShape(ge::GeShape({3, 12, 5, 6}));
  op_desc->SetSrcName({"output"});
  op_desc->SetSrcIndex({0});

  op_desc->AddOutputDesc("output", tensor_desc);
  op_desc->AddInputDesc("output", tensor_desc);
  std::string output_name = "test:test";

  ModelIntroduction model_introduction;
  model_introduction.output_op_list_.push_back(op_desc);
  model_introduction.out_node_name_.push_back(output_name);

  EXPECT_EQ(model_introduction.ConstructOutputInfo(), SUCCESS);
}

}  // namespace ge
