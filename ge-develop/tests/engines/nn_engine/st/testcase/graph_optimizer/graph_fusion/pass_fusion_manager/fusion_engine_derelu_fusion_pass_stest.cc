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
#include <nlohmann/json.hpp>

#include <memory>
#include <string>
#include <vector>

#define protected public
#define private public
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/attr_utils.h"
#include "pass_manager.h"
#include "common/util/op_info_util.h"
//#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/derelu_fusion_pass.h"
#undef protected
#undef private

using namespace fe;

static const char *RELUGRAD = "ReluGrad";

class STEST_optimizer_fusion_derelu_fusion_pass : public testing::Test {


 protected:
  void SetUp() {
  }

  void TearDown() {

  }

 protected:
  ge::NodePtr AddNode(ge::ComputeGraphPtr graph, const string &name, const string &type,
                  int32_t out_anchors_num = 1, int32_t in_anchors_num = 1) {
    ge::GeTensorDesc tensor_desc;
    ge::OpDescPtr opdesc = std::make_shared<ge::OpDesc>(name, type);
    for (int32_t i = 0; i < out_anchors_num; i++) {
      opdesc->AddOutputDesc(tensor_desc);
    }
    for (int32_t i = 0; i < in_anchors_num; i++) {
      opdesc->AddInputDesc(tensor_desc);
    }
    ge::NodePtr node = graph->AddNode(opdesc);
    return node;
  }

  void SetWeightsFloat(ge::NodePtr node, float w) {
    float data[] = {w};
    ge::GeTensorDesc tensor_desc(ge::GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
    ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data, sizeof(data));
    vector<ge::GeTensorPtr> weights = {tensor};
    ge::OpDescUtils::SetWeights(node, weights);
  }


  ge::ComputeGraphPtr CreateSucessGraph() {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    ge::NodePtr const_node = AddNode(graph, fe::CONSTANTOP, fe::CONSTANTOP);
    ge::NodePtr const_node2 = AddNode(graph, fe::CONSTANTOP, fe::CONSTANTOP);
    ge::NodePtr relu2_node = AddNode(graph, RELU, RELU);
    ge::NodePtr relugrad_node = AddNode(graph, RELUGRAD, RELUGRAD, 1, 2);
    std::vector<int64_t> shape_vec{4, 5, 6, 16};
    ge::GeShape shape_desc = ge::GeShape(shape_vec);
    relu2_node->GetOpDesc()->MutableInputDesc(0)->SetShape(shape_desc);
    relu2_node->GetOpDesc()->MutableInputDesc(0)->SetOriginFormat(ge::FORMAT_NCHW);
    relugrad_node->GetOpDesc()->MutableInputDesc(0)->SetDataType(ge::DT_FLOAT16);
    SetWeightsFloat(const_node, 0.2);
    SetWeightsFloat(const_node2, 0.3);

    ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), relugrad_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0), relugrad_node->GetInDataAnchor(1));
    return graph;
  }

  ge::ComputeGraphPtr CreateFailedGraph() {
    ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test_graph");

    ge::NodePtr const_node = AddNode(graph, fe::CONSTANTOP, fe::CONSTANTOP);
    ge::NodePtr relu2_node = AddNode(graph, RELU, RELU);
    ge::NodePtr relugrad_node = AddNode(graph, RELUGRAD, RELUGRAD);

    SetWeightsFloat(const_node, 0.2);

    ge::GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(relu2_node->GetOutDataAnchor(0), relugrad_node->GetInDataAnchor(0));
    return graph;
  }
};

//TEST_F(STEST_optimizer_fusion_derelu_fusion_pass, derelu_fusion_success) {
//  ge::ComputeGraphPtr graph = CreateSucessGraph();
//
//  fe::DreluFusionPass pass;
//  vector<fe::GraphPass *> passes = {&pass};
//  fe::Status status = fe::PassManager::Run(*graph, passes);
//  EXPECT_EQ(fe::SUCCESS, status);
//}
//
//TEST_F(STEST_optimizer_fusion_derelu_fusion_pass, derelu_fusion_success2) {
//  ge::ComputeGraphPtr graph = CreateFailedGraph();
//
//  fe::DreluFusionPass pass;
//  vector<fe::GraphPass *> passes = {&pass};
//  fe::Status status = fe::PassManager::Run(*graph, passes);
//  EXPECT_EQ(fe::NOT_CHANGED, status);
//}
