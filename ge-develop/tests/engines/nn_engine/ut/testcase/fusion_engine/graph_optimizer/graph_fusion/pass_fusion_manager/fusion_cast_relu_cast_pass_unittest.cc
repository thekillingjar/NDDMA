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

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "pass_manager.h"
#include "fe_llt_utils.h"
#include "register/graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/builtin_pass/quant_pass/cast_relu_cast_fusion_pass.h"

#define protected public
#define private public
#include "common/graph/fe_graph_utils.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "common/configuration.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#undef protected
#undef private

using namespace testing;
using namespace ge;
using namespace fe;

namespace fe{

using FEOpsKernelInfoStorePtr = std::shared_ptr<fe::FEOpsKernelInfoStore>;


class fusion_pass_cast_relu_cast_ut : public testing::Test {
public:
  FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr;
protected:
  void SetUp() {
    fe_ops_kernel_info_store_ptr = make_shared<fe::FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
    FEOpsStoreInfo tbe_builtin {
        0,
        "tbe-builtin",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/st/testcase/ops_kernel_store/fe_config/tbe_opinfo",
        "",
        false,
        false,
        false};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_builtin);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr->Initialize(options);
  }

  void TearDown() {
    fe_ops_kernel_info_store_ptr->Finalize();
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

TEST_F(fusion_pass_cast_relu_cast_ut, cast_relu_cast_01)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph1();
  CastReluCastFusionPass pass;
DumpGraph(graph, "test1");
  fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
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

TEST_F(fusion_pass_cast_relu_cast_ut, cast_relu_cast_02)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph2();
  CastReluCastFusionPass pass;
  fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}
TEST_F(fusion_pass_cast_relu_cast_ut, cast_relu_cast_03)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph3();
  CastReluCastFusionPass pass;
  fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}
TEST_F(fusion_pass_cast_relu_cast_ut, cast_relu_cast_04)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph4();
  CastReluCastFusionPass pass;
  fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}
TEST_F(fusion_pass_cast_relu_cast_ut, cast_relu_cast_05)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph5();
  CastReluCastFusionPass pass;
DumpGraph(graph, "test1");
  fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::SUCCESS, status);
DumpGraph(graph, "test1");

}
TEST_F(fusion_pass_cast_relu_cast_ut, cast_relu_cast_06)
{
  ComputeGraphPtr graph = CreateCastReluCastGraph6();
  CastReluCastFusionPass pass;
  fe::Status status = pass.Run(*graph, fe_ops_kernel_info_store_ptr);
  EXPECT_EQ(fe::NOT_CHANGED, status);

}
}
