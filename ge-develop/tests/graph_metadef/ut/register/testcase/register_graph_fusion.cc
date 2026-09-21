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

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"

#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "register/graph_optimizer/graph_fusion/graph_fusion_pass_base.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

#include "framework/common/debug/ge_log.h"

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe{


class UTESTGraphFusionPass : public testing::Test {
 public:

 protected:
  void SetUp() {

  }

  void TearDown() {
  }

  static ComputeGraphPtr CreateCastReluCastGraph1() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT16);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_d);

    ge::AttrUtils::SetStr(op_desc_relu, "_op_compile_strategy", "{}");
    ge::AttrUtils::SetInt(op_desc_relu, "_keep_dtype", 1);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    return graph;
  }
  static ComputeGraphPtr CreateCastReluCastGraph2() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_d);

    ge::AttrUtils::SetStr(op_desc_relu, "_op_compile_strategy", "{}");
    ge::AttrUtils::SetInt(op_desc_relu, "_keep_dtype", 1);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    return graph;
  }
  static ComputeGraphPtr CreateCastReluCastGraph3() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT16);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT16);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_output->AddInputDesc(tensor_desc_c);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));

    return graph;
  }
  static ComputeGraphPtr CreateCastReluCastGraph4() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT16);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_cast3->AddInputDesc(tensor_desc_b);
    op_desc_cast3->AddOutputDesc(tensor_desc_d);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_d);
    op_desc_output->AddInputDesc(tensor_desc_d);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));

    return graph;
  }
  static ComputeGraphPtr CreateCastReluCastGraph5() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Cast");
    OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT16);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_cast3->AddInputDesc(tensor_desc_c);
    op_desc_cast3->AddOutputDesc(tensor_desc_d);

    op_desc_cast4->AddInputDesc(tensor_desc_c);
    op_desc_cast4->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_d);
    op_desc_output->AddInputDesc(tensor_desc_d);
    op_desc_output->AddInputDesc(tensor_desc_d);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
    NodePtr node_cast4 = graph->AddNode(op_desc_cast4);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(2));

    return graph;
  }
  static ComputeGraphPtr CreateCastReluCastGraph6() {
    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test1");
    OpDescPtr op_desc_cast1 = std::make_shared<OpDesc>("cast1", "Cast");
    OpDescPtr op_desc_cast3 = std::make_shared<OpDesc>("cast3", "Cast");
    OpDescPtr op_desc_cast4 = std::make_shared<OpDesc>("cast4", "Cast");
    OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr op_desc_cast2 = std::make_shared<OpDesc>("cast2", "Cast");
    OpDescPtr op_desc_output = std::make_shared<OpDesc>("output", "NetOutput");
    OpDescPtr op_desc_input = std::make_shared<OpDesc>("other", "Other");

    //add descriptor
    vector<int64_t> dim_a = {8, 4, 16, 16};
    GeShape shape_a(dim_a);
    GeTensorDesc tensor_desc_a(shape_a);
    tensor_desc_a.SetFormat(FORMAT_NCHW);
    tensor_desc_a.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_a.SetDataType(DT_FLOAT16);
    tensor_desc_a.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_b = {1, 4, 64, 64};
    GeShape shape_b(dim_b);
    GeTensorDesc tensor_desc_b(shape_b);
    tensor_desc_b.SetFormat(FORMAT_NCHW);
    tensor_desc_b.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_b.SetDataType(DT_FLOAT);
    tensor_desc_b.SetOriginDataType(DT_FLOAT);

    vector<int64_t> dim_c = {1, 4, 64, 64};
    GeShape shape_c(dim_c);
    GeTensorDesc tensor_desc_c(shape_c);
    tensor_desc_c.SetFormat(FORMAT_NCHW);
    tensor_desc_c.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_c.SetDataType(DT_FLOAT);
    tensor_desc_c.SetOriginDataType(DT_FLOAT);

    //vector<int64_t> dim_d;
    GeShape shape_d(dim_a);
    GeTensorDesc tensor_desc_d(shape_d);
    tensor_desc_d.SetFormat(FORMAT_NCHW);
    tensor_desc_d.SetOriginFormat(FORMAT_NCHW);
    tensor_desc_d.SetDataType(DT_FLOAT16);
    tensor_desc_d.SetOriginDataType(DT_FLOAT);

    op_desc_input->AddOutputDesc(tensor_desc_a);

    op_desc_cast1->AddInputDesc(tensor_desc_a);
    op_desc_cast1->AddOutputDesc(tensor_desc_b);

    op_desc_cast3->AddInputDesc(tensor_desc_c);
    op_desc_cast3->AddOutputDesc(tensor_desc_d);

    op_desc_cast4->AddInputDesc(tensor_desc_c);
    op_desc_cast4->AddOutputDesc(tensor_desc_c);

    op_desc_relu->AddInputDesc(tensor_desc_b);
    op_desc_relu->AddOutputDesc(tensor_desc_c);

    op_desc_cast2->AddInputDesc(tensor_desc_c);
    op_desc_cast2->AddOutputDesc(tensor_desc_d);

    op_desc_output->AddInputDesc(tensor_desc_d);
    op_desc_output->AddInputDesc(tensor_desc_d);
    op_desc_output->AddInputDesc(tensor_desc_c);

    NodePtr node_cast1 = graph->AddNode(op_desc_cast1);
    NodePtr node_cast3 = graph->AddNode(op_desc_cast3);
    NodePtr node_cast4 = graph->AddNode(op_desc_cast4);
    NodePtr node_relu = graph->AddNode(op_desc_relu);
    NodePtr node_cast2 = graph->AddNode(op_desc_cast2);
    NodePtr node_netoutput = graph->AddNode(op_desc_output);
    NodePtr node_other = graph->AddNode(op_desc_input);

    GraphUtils::AddEdge(node_other->GetOutDataAnchor(0), node_cast1->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast1->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast2->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast3->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_relu->GetOutDataAnchor(0), node_cast4->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast2->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(0));
    GraphUtils::AddEdge(node_cast3->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(1));
    GraphUtils::AddEdge(node_cast4->GetOutDataAnchor(0), node_netoutput->GetInDataAnchor(2));

    return graph;
  }
  static void DumpGraph(const ge::ComputeGraphPtr graph, string graph_name) {
    printf("start to dump graph %s...\n", graph_name.c_str());
    for (ge::NodePtr node : graph->GetAllNodes()) {
      printf("node name = %s.\n", node->GetName().c_str());
      for (ge::OutDataAnchorPtr anchor : node->GetAllOutDataAnchors()) {
        for (ge::InDataAnchorPtr peer_in_anchor : anchor->GetPeerInDataAnchors()) {
          printf("    node name = %s[%d], out data node name = %s[%d].\n",
                 node->GetName().c_str(),
                 anchor->GetIdx(),
                 peer_in_anchor->GetOwnerNode()->GetName().c_str(),
                 peer_in_anchor->GetIdx());
        }
      }
      if (node->GetOutControlAnchor() != nullptr) {
        for (ge::InControlAnchorPtr peer_in_anchor : node->GetOutControlAnchor()->GetPeerInControlAnchors()) {
          printf("    node name = %s, out control node name = %s.\n", node->GetName().c_str(),
                 peer_in_anchor->GetOwnerNode()->GetName().c_str());
        }
      }
    }

    return;
  }

};


const char *kOpTypeCast = "Cast";
const char *kOpTypeRelu = "Relu";

const char *kPatternCast0 = "Cast0";
const char *kPatternCast1 = "Cast1";
const char *kPatternRelu = "Relu";
#define UT_CHECK(cond, log_func, return_expr) \
  do {                                        \
    if (cond) {                               \
      log_func;                               \
      return_expr;                            \
    }                                         \
  } while (0)

#define UT_CHECK_NOTNULL(val)                           \
  do {                                                  \
    if ((val) == nullptr) {                             \
      GELOGD("Parameter[%s] must not be null.", #val); \
      return fe::PARAM_INVALID;                         \
    }                                                   \
  } while (0)

string pass_name_test = "CastCastFusionPass";
class TestPass : public fe::PatternFusionBasePass {
  using Mapping = std::map<const std::shared_ptr<OpDesc>, std::vector<ge::NodePtr>, fe::CmpKey>;
 protected:
  vector<FusionPattern *> DefinePatterns() override {
    vector<FusionPattern *> patterns;

    FusionPattern *pattern = new(std::nothrow) FusionPattern("CastCastFusionPass");
    UT_CHECK(pattern == nullptr, GELOGD(" Fail to create a new pattern object."),
             return patterns);
    pattern->AddOpDesc(kPatternCast0, {kOpTypeCast})
        .AddOpDesc(kPatternRelu, {kOpTypeRelu})
        .AddOpDesc(kPatternCast1, {kOpTypeCast})
        .SetInputs(kPatternRelu, {kPatternCast0})
        .SetInputs(kPatternCast1, {kPatternRelu})
        .SetOutput(kPatternCast1);

    patterns.push_back(pattern);

    return patterns;
  }

  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector <ge::NodePtr> &new_nodes) override {
    FusionPattern pattern("CastCastFusionPass");
    DumpMapping(pattern, mapping);

    CheckGraphCycle(graph);
    ge::NodePtr cast_Node0 = GetNodeFromMapping(kPatternCast0, mapping);
    CheckOpSupported(cast_Node0);
    CheckOpSupported(cast_Node0->GetOpDesc());
    CheckAccuracySupported(cast_Node0);

    UT_CHECK(cast_Node0 == nullptr, GELOGD("cast_Node0 is null,fusion failed."),
             return NOT_CHANGED);
    ge::OpDescPtr cast_desc0 = cast_Node0->GetOpDesc();
    UT_CHECK(cast_desc0 == nullptr, GELOGD("cast_Node0's Desc is null,fusion failed."),
             return NOT_CHANGED);

    ge::NodePtr relu_Node = GetNodeFromMapping(kPatternRelu, mapping);
    UT_CHECK(relu_Node == nullptr, GELOGD("relu_Node is null,fusion failed."),
             return NOT_CHANGED);
    ge::OpDescPtr relu_desc = relu_Node->GetOpDesc();
    UT_CHECK(cast_desc0 == nullptr, GELOGD("relu_Node's Desc is null,fusion failed."),
             return NOT_CHANGED);

    auto relu_input = relu_desc->MutableInputDesc(0);
    UT_CHECK_NOTNULL(relu_input);
    auto relu_input_desc_dtype = relu_input->GetDataType();

    auto relu_output = relu_desc->MutableOutputDesc(0);
    UT_CHECK_NOTNULL(relu_output);
    auto relu_output_desc_dtype = relu_output->GetDataType();
    if (relu_input_desc_dtype != DT_FLOAT || relu_output_desc_dtype != DT_FLOAT) {
      GELOGD("Relu node [%s]'s input dtype or output dtype is unsuitable", relu_desc->GetName().c_str());
      return NOT_CHANGED;
    }

    ge::NodePtr cast_Node1 = GetNodeFromMapping(kPatternCast1, mapping);
    UT_CHECK(cast_Node1 == nullptr, GELOGD("cast_Node1 is null,fusion failed."),
             return NOT_CHANGED);
    ge::OpDescPtr cast_desc1 = cast_Node1->GetOpDesc();
    UT_CHECK(cast_desc0 == nullptr, GELOGD("cast_Node1's Desc is null,fusion failed."),
             return NOT_CHANGED);

    auto cast0_input = cast_desc0->MutableInputDesc(0);
    UT_CHECK_NOTNULL(cast0_input);
    DataType cast0_in_d_type = cast0_input->GetDataType();
    auto cast1_output = cast_desc1->MutableOutputDesc(0);
    UT_CHECK_NOTNULL(cast1_output);
    DataType cast1_out_d_type = cast1_output->GetDataType();
    if (cast0_in_d_type != cast1_out_d_type) {
      GELOGD("Cast Node0 [%s] input data type is not equal to Cast Node1 [%s] output data type ",
              cast_Node0->GetName().c_str(), cast_Node1->GetName().c_str());
      return NOT_CHANGED;
    }

    auto cast0_out_data_anchor = cast_Node0->GetOutDataAnchor(0);
    UT_CHECK_NOTNULL(cast0_out_data_anchor);
    if (cast0_out_data_anchor->GetPeerInDataAnchors().size() > 1) {
      GELOGD("The first output anchor of Cast node[%s] has more than one peer in anchor.",
              cast_Node0->GetName().c_str());
      return NOT_CHANGED;
    }

    auto relu_out_data_anchor = relu_Node->GetOutDataAnchor(0);
    UT_CHECK_NOTNULL(relu_out_data_anchor);
    if (relu_out_data_anchor->GetPeerInDataAnchors().size() > 1) {
      for (auto node : relu_Node->GetOutAllNodes()) {
        if (node->GetType() != "Cast") {
          GELOGD("The  output anchor of Relu node has not Cast node,name is [%s] Type is [%s].",
                  node->GetName().c_str(), node->GetType().c_str());
          return NOT_CHANGED;
        }
        auto node_desc = node->GetOpDesc();
        UT_CHECK_NOTNULL(node_desc);
        auto in_dtype = node_desc->MutableInputDesc(0)->GetDataType();
        auto out_dtype = node_desc->MutableOutputDesc(0)->GetDataType();
        if (in_dtype != DT_FLOAT || out_dtype != DT_FLOAT16) {
          GELOGD("The Cast node [%s]'s indatatype is not equal to DT_FLOAT or outdatatype is not equal to DT_FLOAT16.",
                  node->GetName().c_str());
          return NOT_CHANGED;
        }
      }
    }

    ge::ComputeGraphPtr graphPtr = relu_Node->GetOwnerComputeGraph();
    UT_CHECK_NOTNULL(graphPtr);
    if (GraphUtils::IsolateNode(cast_Node0, {0}) != GRAPH_SUCCESS) {
      GELOGD("Isolate op:%s(%s) failed", cast_Node0->GetName().c_str(), cast_Node0->GetType().c_str());
      return FAILED;
    }
    if (GraphUtils::RemoveNodeWithoutRelink(graphPtr, cast_Node0) != GRAPH_SUCCESS) {
      GELOGD("[Remove][Node] %s, type:%s without relink in graph:%s failed",
                      cast_Node0->GetName().c_str(), cast_Node0->GetType().c_str(), graph.GetName().c_str());
      return FAILED;
    }
    for (auto inAnchor : relu_out_data_anchor->GetPeerInDataAnchors()) {
      auto node = inAnchor->GetOwnerNode();
      UT_CHECK_NOTNULL(node);
      if (GraphUtils::IsolateNode(node, {0}) != GRAPH_SUCCESS) {
        GELOGD("Isolate op:%s(%s) failed", node->GetName().c_str(), node->GetType().c_str());
        return FAILED;
      }
      if (GraphUtils::RemoveNodeWithoutRelink(graphPtr, node) != GRAPH_SUCCESS) {
        GELOGD("[Remove][Node] %s, type:%s without relink in graph:%s failed",
                        node->GetName().c_str(), node->GetType().c_str(), graph.GetName().c_str());
        return FAILED;
      }
    }
    relu_desc->MutableInputDesc(0)->SetDataType(cast0_in_d_type);
    relu_desc->MutableOutputDesc(0)->SetDataType(cast1_out_d_type);
    new_nodes.push_back(relu_Node);
    return SUCCESS;
  }
};

TEST_F(UTESTGraphFusionPass, cast_relu_cast_01)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph1();
  TestPass pass;
  DumpGraph(graph, "test1");
  fe::Status status = pass.Run(*graph, nullptr);
  EXPECT_EQ(fe::SUCCESS, status);
  DumpGraph(graph, "test1");

  vector<int64_t> dim_a = {8, 4, 16, 16};
  vector<int64_t> dim_b = {1, 4, 64, 64};

  for(auto node : graph->GetDirectNode()) {
    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc->GetType() == "Relu") {
      EXPECT_EQ(op_desc->MutableInputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableInputDesc(0)->MutableShape().GetDims(), dim_b);

      EXPECT_EQ(op_desc->MutableOutputDesc(0)->GetDataType(), DT_FLOAT16);
      EXPECT_EQ(op_desc->MutableOutputDesc(0)->MutableShape().GetDims(), dim_b);
      NodePtr node_out=node->GetOutDataNodes().at(0);
      EXPECT_EQ(node_out->GetOpDesc()->GetType(),"NetOutput");
      EXPECT_EQ(node_out->GetOpDesc()->MutableInputDesc(0)->GetDataType(),DT_FLOAT16);
      EXPECT_EQ(node_out->GetOpDesc()->MutableInputDesc(0)->MutableShape().GetDims(),dim_a);
      NodePtr  node0=node->GetInDataNodes().at(0);
      EXPECT_EQ(node0->GetOpDesc()->GetType(),"Other");
      EXPECT_EQ(node0->GetOpDesc()->MutableOutputDesc(0)->GetDataType(),DT_FLOAT16);
      EXPECT_EQ(node0->GetOpDesc()->MutableOutputDesc(0)->MutableShape().GetDims(), dim_a);

    }
  }
}

class UtOpsKernel : public OpsKernelInfoStore {
  // initialize opsKernelInfoStore
  Status Initialize(const std::map<std::string, std::string> &options) override {
    return SUCCESS;
  }

  // close opsKernelInfoStore
  Status Finalize() override {
    return SUCCESS;
  }

  // get all opsKernelInfo
  void GetAllOpsKernelInfo(std::map<std::string, OpInfo> &infos) const override {}

  // whether the opsKernelInfoStore is supported based on the operator attribute
  bool CheckSupported(const OpDescPtr &opDescPtr, std::string &un_supported_reason) const override {
    return true;
  }
};

TEST_F(UTESTGraphFusionPass, cast_relu_cast_02)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph2();
  TestPass pass;
  std::shared_ptr<UtOpsKernel> ops_kernel = std::make_shared<UtOpsKernel>();

  std::shared_ptr<OpsKernelInfoStore> base = ops_kernel;
  fe::Status status = pass.Run(*graph, base);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}
TEST_F(UTESTGraphFusionPass, cast_relu_cast_03)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph3();

  TestPass pass;
  std::shared_ptr<UtOpsKernel> ops_kernel = std::make_shared<UtOpsKernel>();

  std::shared_ptr<OpsKernelInfoStore> base = ops_kernel;
  fe::Status status = pass.Run(*graph, base);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}
TEST_F(UTESTGraphFusionPass, cast_relu_cast_04)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph4();
  TestPass pass;
  std::shared_ptr<UtOpsKernel> ops_kernel = std::make_shared<UtOpsKernel>();

  std::shared_ptr<OpsKernelInfoStore> base = ops_kernel;
  fe::Status status = pass.Run(*graph, base);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}
TEST_F(UTESTGraphFusionPass, cast_relu_cast_05)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph5();
  TestPass pass;
  DumpGraph(graph, "test1");
  std::shared_ptr<UtOpsKernel> ops_kernel = std::make_shared<UtOpsKernel>();

  std::shared_ptr<OpsKernelInfoStore> base = ops_kernel;
  fe::Status status = pass.Run(*graph, base);
  EXPECT_EQ(fe::SUCCESS, status);
  DumpGraph(graph, "test1");

}
TEST_F(UTESTGraphFusionPass, cast_relu_cast_06)
{

  ComputeGraphPtr graph = CreateCastReluCastGraph6();
  TestPass pass;
  std::shared_ptr<UtOpsKernel> ops_kernel = std::make_shared<UtOpsKernel>();

  std::shared_ptr<OpsKernelInfoStore> base = ops_kernel;
  fe::Status status = pass.Run(*graph, base);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}

TEST_F(UTESTGraphFusionPass, coverage_01) {
  REGISTER_PASS(pass_name_test, GRAPH_FUSION_PASS_TYPE_RESERVED, TestPass);
  REGISTER_PASS("", BUILT_IN_GRAPH_PASS, TestPass);
  REGISTER_PASS(pass_name_test, BUILT_IN_GRAPH_PASS, TestPass);
  REGISTER_PASS(pass_name_test, BUILT_IN_GRAPH_PASS, TestPass);
  std::map<string, FusionPassRegistry::CreateFn> create_fns =
    FusionPassRegistry::GetInstance().GetCreateFnByType(SECOND_ROUND_BUILT_IN_GRAPH_PASS);
  EXPECT_NO_THROW(
    create_fns =
      FusionPassRegistry::GetInstance().GetCreateFnByType(BUILT_IN_GRAPH_PASS);
  );
}
}
