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
#include <cstdint>
#include <memory>
#include <string>

#include "macro_utils/dt_public_scope.h"
#include "graph/passes/memory_conflict/memcpy_addr_async_pass.h"
#include "common/ge_inner_error_codes.h"
#include "graph/passes/pass_manager.h"
#include "graph/normal_graph/ge_tensor_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "depends/runtime/src/runtime_stub.h"
#include "runtime/rt.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_local_context.h"
#include "macro_utils/dt_public_unscope.h"

namespace ge {
class UtestMemcpyAddrAsyncPass : public testing::Test {
 protected:
  void SetUp() {
    RTS_STUB_SETUP();
  }
  void TearDown() {
    RTS_STUB_TEARDOWN();
  }

public:
/* 设置一个Graph，使其拥有如下网络结构
*
*              Data
*               |
*           StreamSwitchN
*               |
*               A
*               |
*           NetOutput
*/
  void MakeGraphDataInParent1(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);

    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    op_desc_cond->AddOutputDesc(desc);
    ge::AttrUtils::SetBool(op_desc_cond, ATTR_NAME_NODE_CONNECT_INPUT, true);

    OpDescPtr op_desc_a = MakeShared<OpDesc>("A", RESOURCEAPPLYMOMENTUM);
    op_desc_a->AddInputDesc(desc);
    op_desc_a->AddOutputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Node_output", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);

    NodePtr data_node = graph->AddNode(op_desc_data);
    NodePtr cond_node = graph->AddNode(op_desc_cond);
    NodePtr a_node = graph->AddNode(op_desc_a);
    NodePtr out_node = graph->AddNode(op_desc_out);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cond_node->GetOutDataAnchor(0), a_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  void MakeGraphDataInParent2(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);

    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    op_desc_cond->AddOutputDesc(desc);
    ge::AttrUtils::SetBool(op_desc_cond, ATTR_NAME_RTS_LABEL_NODE, true);

    OpDescPtr op_desc_a = MakeShared<OpDesc>("A", RESOURCEAPPLYMOMENTUM);
    op_desc_a->AddInputDesc(desc);
    op_desc_a->AddOutputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Node_output", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);

    NodePtr data_node = graph->AddNode(op_desc_data);
    NodePtr cond_node = graph->AddNode(op_desc_cond);
    NodePtr a_node = graph->AddNode(op_desc_a);
    NodePtr out_node = graph->AddNode(op_desc_out);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(cond_node->GetOutDataAnchor(0), a_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(a_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

/* 设置一个Graph，使其拥有如下网络结构
*
*              Data
*               |
*              If
*               '
*                '
*                 '
*                 Data
*                   |
*             StreamSwitchN
*/
  void MakeGraphDataInSub1(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);
    NodePtr data_node = graph->AddNode(op_desc_data);

    OpDescPtr op_desc_if = MakeShared<OpDesc>("If", IF);
    op_desc_if->AddInputDesc(desc);
    NodePtr if_node = graph->AddNode(op_desc_if);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0));

    string cond_name = "cond";
    ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_name);
    cond_graph->SetParentNode(if_node);
    cond_graph->SetParentGraph(graph);
    if_node->GetOpDesc()->AddSubgraphName(cond_name);
    if_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_name);
    graph->AddSubGraph(cond_graph);


    OpDescPtr op_desc_data_sub = MakeShared<OpDesc>("Data", DATA);
    op_desc_data_sub->AddOutputDesc(desc);
    NodePtr data_node_sub = cond_graph->AddNode(op_desc_data_sub);
    AttrUtils::SetInt(op_desc_data_sub, ATTR_NAME_PARENT_NODE_INDEX, 0);

    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    NodePtr cond_node = cond_graph->AddNode(op_desc_cond);

    GraphUtils::AddEdge(data_node_sub->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
  }

  void MakeGraphDataInSubFailed(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);
    NodePtr data_node = graph->AddNode(op_desc_data);

    OpDescPtr op_desc_if = MakeShared<OpDesc>("If", IF);
    op_desc_if->AddInputDesc(desc);
    NodePtr if_node = graph->AddNode(op_desc_if);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0));

    string cond_name = "cond";
    ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_name);
    cond_graph->SetParentNode(if_node);
    cond_graph->SetParentGraph(graph);
    if_node->GetOpDesc()->AddSubgraphName(cond_name);
    if_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_name);
    graph->AddSubGraph(cond_graph);


    OpDescPtr op_desc_data_sub = MakeShared<OpDesc>("Data", DATA);
    op_desc_data_sub->AddOutputDesc(desc);
    NodePtr data_node_sub = cond_graph->AddNode(op_desc_data_sub);
    AttrUtils::SetInt(op_desc_data_sub, ATTR_NAME_NEED_INFER_AGAIN, 0);

    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    NodePtr cond_node = cond_graph->AddNode(op_desc_cond);

    GraphUtils::AddEdge(data_node_sub->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
  }

/* 设置一个Graph，使其拥有如下网络结构
*
*              Relu
*               |
*              If
*               '
*                '
*                 '
*                 Data
*                   |
*             StreamSwitchN
*/
  void MakeGraphDataInSub2(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Relu", RELU);
    op_desc_data->AddOutputDesc(desc);
    NodePtr data_node = graph->AddNode(op_desc_data);

    OpDescPtr op_desc_if = MakeShared<OpDesc>("If", IF);
    op_desc_if->AddInputDesc(desc);
    NodePtr if_node = graph->AddNode(op_desc_if);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node->GetInDataAnchor(0));

    string cond_name = "cond";
    ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_name);
    cond_graph->SetParentNode(if_node);
    cond_graph->SetParentGraph(graph);
    if_node->GetOpDesc()->AddSubgraphName(cond_name);
    if_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_name);
    graph->AddSubGraph(cond_graph);

    OpDescPtr op_desc_data_sub = MakeShared<OpDesc>("Data", DATA);
    op_desc_data_sub->AddOutputDesc(desc);
    NodePtr data_node_sub = cond_graph->AddNode(op_desc_data_sub);

    AttrUtils::SetInt(op_desc_data_sub, ATTR_NAME_PARENT_NODE_INDEX, 0);
    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    NodePtr cond_node = cond_graph->AddNode(op_desc_cond);

    GraphUtils::AddEdge(data_node_sub->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
  }

/* 设置一个Graph，使其拥有如下网络结构
*
*           Data1
*             |
*            If1
*             '
*              '
*               '
*              Data2
*               |
*              If2
*               '
*                '
*                 '
*                 Data
*                   |
*             StreamSwitchN
*/
  void MakeGraphDataInSub3(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data1 = MakeShared<OpDesc>("Data1", DATA);
    op_desc_data1->AddOutputDesc(desc);
    NodePtr data_node1 = graph->AddNode(op_desc_data1);
    OpDescPtr op_desc_if1 = MakeShared<OpDesc>("If1", IF);
    op_desc_if1->AddInputDesc(desc);
    NodePtr if_node1 = graph->AddNode(op_desc_if1);
    GraphUtils::AddEdge(data_node1->GetOutDataAnchor(0), if_node1->GetInDataAnchor(0));

    string cond_name1 = "cond1";
    ComputeGraphPtr cond_graph1 = std::make_shared<ComputeGraph>(cond_name1);
    cond_graph1->SetParentNode(if_node1);
    cond_graph1->SetParentGraph(graph);
    if_node1->GetOpDesc()->AddSubgraphName(cond_name1);
    if_node1->GetOpDesc()->SetSubgraphInstanceName(0, cond_name1);

    OpDescPtr op_desc_data2 = MakeShared<OpDesc>("Data2", DATA);
    op_desc_data2->AddOutputDesc(desc);
    NodePtr data_node2 = cond_graph1->AddNode(op_desc_data2);
    OpDescPtr op_desc_if2 = MakeShared<OpDesc>("If2", IF);
    op_desc_if2->AddInputDesc(desc);
    NodePtr if_node2 = cond_graph1->AddNode(op_desc_if2);
    GraphUtils::AddEdge(data_node2->GetOutDataAnchor(0), if_node2->GetInDataAnchor(0));
    AttrUtils::SetInt(op_desc_data2, ATTR_NAME_PARENT_NODE_INDEX, 0);

    string cond_name2 = "cond2";
    ComputeGraphPtr cond_graph2 = std::make_shared<ComputeGraph>(cond_name2);
    cond_graph2->SetParentNode(if_node2);
    cond_graph2->SetParentGraph(cond_graph1);
    if_node2->GetOpDesc()->AddSubgraphName(cond_name2);
    if_node2->GetOpDesc()->SetSubgraphInstanceName(0, cond_name2);

    OpDescPtr op_desc_data3 = MakeShared<OpDesc>("Data", DATA);
    op_desc_data3->AddOutputDesc(desc);
    NodePtr data_node3 = cond_graph2->AddNode(op_desc_data3);
    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    NodePtr cond_node = cond_graph2->AddNode(op_desc_cond);
    GraphUtils::AddEdge(data_node3->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
    AttrUtils::SetInt(op_desc_data3, ATTR_NAME_PARENT_NODE_INDEX, 0);

    graph->AddSubGraph(cond_graph2);
    graph->AddSubGraph(cond_graph1);
  }

/* 设置一个Graph，使其拥有如下网络结构
*              Data
*               |
*              If1
*               |
*              If2
*               '
*                '
*                 '
*                 Data
*                   |
*             StreamSwitchN
*/
  void MakeGraphDataInSub4(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);
    NodePtr data_node = graph->AddNode(op_desc_data);

    OpDescPtr op_desc_if1 = MakeShared<OpDesc>("If1", IF);
    op_desc_if1->AddInputDesc(desc);
    op_desc_if1->AddOutputDesc(desc);
    NodePtr if_node1 = graph->AddNode(op_desc_if1);

    OpDescPtr op_desc_if2 = MakeShared<OpDesc>("If2", IF);
    op_desc_if2->AddInputDesc(desc);
    NodePtr if_node2 = graph->AddNode(op_desc_if2);
    AttrUtils::SetInt(op_desc_if2, ATTR_NAME_PARENT_NODE_INDEX, 0);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node1->GetInDataAnchor(0));
    GraphUtils::AddEdge(if_node1->GetOutDataAnchor(0), if_node2->GetInDataAnchor(0));

    string cond_name = "cond";
    ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_name);
    cond_graph->SetParentNode(if_node2);
    cond_graph->SetParentGraph(graph);
    if_node2->GetOpDesc()->AddSubgraphName(cond_name);
    if_node2->GetOpDesc()->SetSubgraphInstanceName(0, cond_name);
    graph->AddSubGraph(cond_graph);

    OpDescPtr op_desc_data_sub = MakeShared<OpDesc>("Data", DATA);
    op_desc_data_sub->AddOutputDesc(desc);
    NodePtr data_node_sub = cond_graph->AddNode(op_desc_data_sub);
    AttrUtils::SetInt(op_desc_data_sub, ATTR_NAME_PARENT_NODE_INDEX, 0);

    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    NodePtr cond_node = cond_graph->AddNode(op_desc_cond);

    GraphUtils::AddEdge(data_node_sub->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
  }

  void MakeGraphDataInSub6(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);
    NodePtr data_node = graph->AddNode(op_desc_data);

    OpDescPtr op_desc_if1 = MakeShared<OpDesc>("Partition_call", PARTITIONEDCALL);
    op_desc_if1->AddInputDesc(desc);
    op_desc_if1->AddOutputDesc(desc);
    NodePtr if_node1 = graph->AddNode(op_desc_if1);

    OpDescPtr op_desc_known = MakeShared<OpDesc>("Partition_call", PARTITIONEDCALL);
    op_desc_known->AddInputDesc(desc);
    NodePtr known_node = graph->AddNode(op_desc_known);
    bool is_unknown_shape = false;
    (void)AttrUtils::SetBool(known_node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), if_node1->GetInDataAnchor(0));
    GraphUtils::AddEdge(if_node1->GetOutDataAnchor(0), known_node->GetInDataAnchor(0));

    string cond_name = "cond";
    ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_name);
    cond_graph->SetParentNode(known_node);
    cond_graph->SetParentGraph(graph);
    known_node->GetOpDesc()->AddSubgraphName(cond_name);
    known_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_name);
    graph->AddSubGraph(cond_graph);

    OpDescPtr op_desc_data_sub = MakeShared<OpDesc>("Data", DATA);
    op_desc_data_sub->AddOutputDesc(desc);
    NodePtr data_node_sub = cond_graph->AddNode(op_desc_data_sub);
    AttrUtils::SetInt(op_desc_data_sub, ATTR_NAME_PARENT_NODE_INDEX, 0);

    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    NodePtr cond_node = cond_graph->AddNode(op_desc_cond);

    GraphUtils::AddEdge(data_node_sub->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
  }

/* 设置一个Graph，使其拥有如下网络结构
*
*              Data
*               |
*          Partition_call
*               '
*                '
*                 '
*                 Data
*                   |
*             StreamSwitchN
*/
  void MakeGraphDataInSub5(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);
    NodePtr data_node = graph->AddNode(op_desc_data);

    OpDescPtr op_desc_known = MakeShared<OpDesc>("Partition_call", PARTITIONEDCALL);
    op_desc_known->AddInputDesc(desc);
    NodePtr known_node = graph->AddNode(op_desc_known);
    bool is_unknown_shape = false;
    (void)AttrUtils::SetBool(known_node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), known_node->GetInDataAnchor(0));

    string cond_name = "cond";
    ComputeGraphPtr cond_graph = std::make_shared<ComputeGraph>(cond_name);
    cond_graph->SetParentNode(known_node);
    cond_graph->SetParentGraph(graph);
    known_node->GetOpDesc()->AddSubgraphName(cond_name);
    known_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_name);
    graph->AddSubGraph(cond_graph);

    OpDescPtr op_desc_data_sub = MakeShared<OpDesc>("Data", DATA);
    op_desc_data_sub->AddOutputDesc(desc);
    NodePtr data_node_sub = cond_graph->AddNode(op_desc_data_sub);

    AttrUtils::SetInt(op_desc_data_sub, ATTR_NAME_PARENT_NODE_INDEX, 0);
    OpDescPtr op_desc_cond = MakeShared<OpDesc>("merge", STREAMMERGE);
    op_desc_cond->AddInputDesc(desc);
    NodePtr cond_node = cond_graph->AddNode(op_desc_cond);

    GraphUtils::AddEdge(data_node_sub->GetOutDataAnchor(0), cond_node->GetInDataAnchor(0));
  }

  void MakeDsaSubGraph(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);
    NodePtr data_node = graph->AddNode(op_desc_data);

    OpDescPtr op_desc_known = MakeShared<OpDesc>("Partition_call", PARTITIONEDCALL);
    op_desc_known->AddInputDesc(desc);
    NodePtr known_node = graph->AddNode(op_desc_known);
    bool is_unknown_shape = false;
   (void)AttrUtils::SetBool(known_node->GetOpDesc(), ATTR_NAME_IS_UNKNOWN_SHAPE, is_unknown_shape);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), known_node->GetInDataAnchor(0));

    string subgraph_name = "dsa";
    ComputeGraphPtr subgraph = std::make_shared<ComputeGraph>(subgraph_name);
    subgraph->SetParentNode(known_node);
    subgraph->SetParentGraph(graph);
    known_node->GetOpDesc()->AddSubgraphName(subgraph_name);
    known_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name);
    graph->AddSubGraph(subgraph);

    OpDescPtr op_desc_data_sub = MakeShared<OpDesc>("Data", DATA);
    op_desc_data_sub->AddOutputDesc(desc);
    NodePtr data_node_sub = subgraph->AddNode(op_desc_data_sub);

    AttrUtils::SetInt(op_desc_data_sub, ATTR_NAME_PARENT_NODE_INDEX, 0);
    OpDescPtr op_desc_dsa = MakeShared<OpDesc>("dsa", "DSARandomUniform");
    op_desc_dsa->AddInputDesc(desc);
    op_desc_dsa->AddOutputDesc(desc);
    op_desc_dsa->SetOpKernelLibName("DSAEngine");
    NodePtr dsa_node = subgraph->AddNode(op_desc_dsa);

    GraphUtils::AddEdge(data_node_sub->GetOutDataAnchor(0), dsa_node->GetInDataAnchor(0));

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Node_output", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);
    NodePtr out_node = subgraph->AddNode(op_desc_out);

    GraphUtils::AddEdge(dsa_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  /* 设置一个Graph，使其拥有如下网络结构
*
*              Data
*               |
*           NetOutput
*/
  void MakeGraphData2NetOutputOnRootGraph(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", DATA);
    op_desc_data->AddOutputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Node_output", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);

    NodePtr data_node = graph->AddNode(op_desc_data);
    NodePtr out_node = graph->AddNode(op_desc_out);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }
  /* 设置一个Graph，使其拥有如下网络结构
*
*              Data
*               |
*           NetOutput
*/
  void MakeGrapConst2NetOutputOnRootGraph(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_const = MakeShared<OpDesc>("Const", CONSTANT);
    op_desc_const->AddOutputDesc(desc);
    OpDescPtr op_desc_out = MakeShared<OpDesc>("Node_output", NETOUTPUT);
    op_desc_out->AddInputDesc(desc);
    NodePtr const_node = graph->AddNode(op_desc_const);
    NodePtr out_node = graph->AddNode(op_desc_out);
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  /* 设置一个Graph，使其拥有如下网络结构
*
*              Refdata
*               |
*            hcomallreduce
*/
  void MakeGraphRefData2HCCL(ComputeGraphPtr &graph) {
    auto desc_ptr = MakeShared<ge::GeTensorDesc>();
    auto desc = *desc_ptr;

    OpDescPtr op_desc_data = MakeShared<OpDesc>("Data", REFDATA);
    op_desc_data->AddOutputDesc(desc);

    OpDescPtr op_desc_out = MakeShared<OpDesc>("Node_output", HCOMALLREDUCE);
    op_desc_out->AddInputDesc(desc);
    op_desc_out->SetOpKernelLibName("ops_kernel_info_hccl");

    NodePtr data_node = graph->AddNode(op_desc_data);
    NodePtr out_node = graph->AddNode(op_desc_out);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), out_node->GetInDataAnchor(0));
  }

  void InitOpDescTensor(NodePtr node)
  {
    float weights_0[] = {1.0};
    GeTensorDesc tensor_desc_0(GeShape({1, 1, 1, 1}), FORMAT_NHWC, DT_FLOAT);
    GeTensorPtr tensor_conv_filter = MakeShared<GeTensor>(tensor_desc_0, (uint8_t *)weights_0, sizeof(weights_0));
    vector<GeTensorPtr> tensor_vec = {tensor_conv_filter};
    OpDescUtils::SetWeights(node, tensor_vec);
  }

  NodePtr AddNode(ComputeGraphPtr graph, const string& name, const string& type,
                  int32_t in_anchors_num = 3, int32_t out_anchors_num = 3)
  {
    GeTensorDesc tensor_desc(GeShape({1, 2, 3, 4}), FORMAT_NHWC, DT_INT32);
    OpDescPtr opdesc = MakeShared<OpDesc>(name, type);
    for (int32_t i = 0; i < in_anchors_num; i++) {
      opdesc->AddInputDesc(tensor_desc);
    }
    for (int32_t i = 0; i <out_anchors_num; i++) {
      opdesc->AddOutputDesc(tensor_desc);
    }

    NodePtr node = graph->AddNode(opdesc);

    if (type == CONVOLUTION) {
      InitOpDescTensor(node);
    }
    return node;
  }

  void InitFullGraph(ComputeGraphPtr graph)
  {
    OpDescPtr data_op_1 = MakeShared<OpDesc>("data_op_1", DATA);
    AttrUtils::SetInt(data_op_1, "index", 0);
    //构造一个input descriptor
    vector<int64_t> dims = {1, 3, 10, 10};
    GeShape shape(dims);
    GeTensorDesc tensor_desc1(shape);
    TensorUtils::SetSize(tensor_desc1, 1200);

    //添加一个描述Tensor
    data_op_1->AddInputDesc(tensor_desc1);
    data_op_1->AddOutputDesc(tensor_desc1);
    data_op_1->AddOutputDesc(tensor_desc1);
    data_op_1->AddOutputDesc(tensor_desc1);
    data_op_1->SetInputOffset({0});
    NodePtr data_node_1 = graph->AddNode(data_op_1);

    //data_op_2
    OpDescPtr data_op_2 = std::make_shared<OpDesc>("data_op_2", DATA);
    AttrUtils::SetInt(data_op_2, "index", 1);
    data_op_2->AddInputDesc(tensor_desc1);
    data_op_2->AddOutputDesc(tensor_desc1);
    data_op_2->AddOutputDesc(tensor_desc1);
    data_op_2->AddOutputDesc(tensor_desc1);
    data_op_2->SetInputOffset({0});
    NodePtr data_node_2 = graph->AddNode(data_op_2);

    GeTensorDesc tensor_desc(GeShape({1}));
    //conv
    NodePtr conv_node = AddNode(graph, "conv_op", CONVOLUTION);
    NodePtr reshape_node_1 = AddNode(graph, "reshape_op_1", RESHAPE);
    NodePtr reshape_node_2 = AddNode(graph, "reshape_op_2", RESHAPE);
    NodePtr concat_node = AddNode(graph, "concat_op", CONCAT);

    ge::GraphUtils::AddEdge(data_node_2->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_node_2->GetOutDataAnchor(0), reshape_node_1->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_node_2->GetOutDataAnchor(0), reshape_node_2->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(data_node_1->GetOutDataAnchor(0), concat_node->GetInDataAnchor(0));
    ge::GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), concat_node->GetInDataAnchor(1));
    ge::GraphUtils::AddEdge(reshape_node_1->GetOutDataAnchor(0), concat_node->GetInDataAnchor(2));
    ge::GraphUtils::AddEdge(reshape_node_2->GetOutDataAnchor(0), concat_node->GetInDataAnchor(3));
  }

};

TEST_F(UtestMemcpyAddrAsyncPass, run) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op, STREAMSWITCH);
  op->SetName("stream_switch");
  op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node = graph->AddNode(op);
  graph->SetGraphUnknownFlag(true);
  MemcpyAddrAsyncPass pass;
  Status ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  graph->SetGraphUnknownFlag(false);
  ret = pass.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncPass, AddMemcpyAsyncNode)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op, STREAMSWITCH);
  op->SetName("stream_switch");
  op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node = graph->AddNode(op);
  graph->SetGraphUnknownFlag(true);
  MemcpyAddrAsyncPass pass;
  Status ret = pass.AddMemcpyAsyncNode(node);
  EXPECT_EQ(ret, SUCCESS);

  op->AddInputDesc("x1", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  op->AddInputDesc("x2", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  op->AddOutputDesc("y", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  node = graph->AddNode(op);
  ret = pass.AddMemcpyAsyncNode(node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncPass, AddMemcpyAddrAsyncNode)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>();
  ge::OpDescUtilsEx::SetType(op, STREAMSWITCH);
  op->SetName("stream_switch");
  op->AddOutputDesc(ge::GeTensorDesc());
  ge::NodePtr node = graph->AddNode(op);
  graph->SetGraphUnknownFlag(true);
  MemcpyAddrAsyncPass pass;
  Status ret = pass.AddMemcpyAddrAsyncNode(graph, node);
  EXPECT_EQ(ret, SUCCESS);

  op->AddInputDesc("x1", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  op->AddInputDesc("x2", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  op->AddOutputDesc("y", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  node = graph->AddNode(op);
  ret = pass.AddMemcpyAddrAsyncNode(graph, node);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncPass, InsertMemAddrAsyncNodeBeforeNetoutput)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  InitFullGraph(graph);

  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("Const", CONSTANT);
  op->AddInputDesc("x1", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  op->AddInputDesc("x2", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  op->AddOutputDesc("y", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  ge::OpDescUtilsEx::SetType(op, CONSTANTOP);
  op->SetName("Const");
  ge::NodePtr node = graph->AddNode(op);

  MemcpyAddrAsyncPass pass;
  Status ret = pass.InsertMemAddrAsyncNodeBeforeNetoutput(graph, node);
  EXPECT_EQ(ret, PARAM_INVALID);
}

TEST_F(UtestMemcpyAddrAsyncPass, FindUserData)
{
  MemcpyAddrAsyncPass pass;
  uint32_t index = 0;

  ComputeGraphPtr call_graph = std::make_shared<ComputeGraph>("PartitionCall_Graph_Success_Test");
  OpDescPtr call_desc = std::make_shared<OpDesc>("node_name", PARTITIONEDCALL);
  call_desc->AddInputDesc("x1", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  call_desc->AddOutputDesc("y", ge::GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW));
  NodePtr call_node = call_graph->AddNode(call_desc);
  EXPECT_NO_THROW(pass.FindUserData(call_node, index));
}

TEST_F(UtestMemcpyAddrAsyncPass, CreateMemcpyAddrAsyncNode)
{
  ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr op = std::make_shared<OpDesc>("A", "test_cce_a");
  GeTensorDesc op_desc(GeShape({1, 1, 1, 1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  op->AddOutputDesc(op_desc);
  auto node = graph->AddNode(op);
  const OutDataAnchorPtr out_data_anchor = node->GetOutDataAnchor(0);
  const NodePtr out_of_user_data = graph->AddNode(op);
  MemcpyAddrAsyncPass pass;
  AttrUtils::SetStr(out_of_user_data->GetOpDesc(), ATTR_NAME_STREAM_LABEL, "default");
  NodePtr nodePtr = pass.CreateMemcpyAddrAsyncNode(graph, out_data_anchor, out_of_user_data);
  EXPECT_NE(nodePtr, nullptr);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_parent)
{
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInParent1(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 4);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(),5);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}


TEST_F(UtestMemcpyAddrAsyncPass, Data2Netoutput_success_parent2)
{
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphData2NetOutputOnRootGraph(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(),2);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, Const2Netoutput_success_parent1) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGrapConst2NetOutputOnRootGraph(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(),3);
  auto const_node = graph->FindNode("Const");
  auto output_node = graph->FindNode("Node_output");
  EXPECT_EQ(const_node->GetOutDataNodes().at(0)->GetType(), "MemcpyAddrAsync");
  EXPECT_EQ(output_node->GetInDataNodes().at(0)->GetType(), "MemcpyAddrAsync");
  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_sub1) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInSub1(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 4);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_sub2) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInSub2(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 4);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 4);
  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_sub3) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInSub3(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 6);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 7);
  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_sub4) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInSub4(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 5);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 6);
  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_sub5) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  graph->SetGraphUnknownFlag(true);
  MakeGraphDataInSub5(graph);
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 4);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(), 4);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_sub6) {
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInSub6(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, GetRtCapability_failed) {
  MemcpyAddrAsyncPass memcpy_addr;
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");

  graph->SetGraphUnknownFlag(true);
  EXPECT_EQ(memcpy_addr.Run(graph), SUCCESS);

  graph->SetGraphUnknownFlag(false);
  EXPECT_EQ(memcpy_addr.Run(graph), SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncPass, IsEmptyTenor_test)
{
  MemcpyAddrAsyncPass memcpy_addr;

  vector<int64_t> dims = {1, 3, 10, 10};
  GeShape shape(dims);

  bool ret = memcpy_addr.IsEmptyTenor(shape);
  EXPECT_EQ(ret, false);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_Run_Test)
{
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInSubFailed(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, INTERNAL_ERROR);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_Run_Test2)
{
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphDataInParent2(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_Run_Test3)
{
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  MakeGraphDataInParent2(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncPass, MemcpyAddrAsyncPass_success_dsa)
{
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsync");
  graph->SetGraphUnknownFlag(true);
  MakeDsaSubGraph(graph);
  MemcpyAddrAsyncPass memcpy_addr;
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
}

TEST_F(UtestMemcpyAddrAsyncPass, RefData2Hccl_success)
{
  ComputeGraphPtr graph = MakeShared<ComputeGraph>("MemcpyAddrAsyncPassSuccess");
  const char_t * const kEnvValue = "SET_CAPA_VALUE";
  // 设置环境变量
  char_t npu_collect_path[MMPA_MAX_PATH] = {};
  mmRealPath(".", &npu_collect_path[0U], MMPA_MAX_PATH);
  const std::string fail_collect_path = (std::string(&npu_collect_path[0U]) + "/mock_fail");
  mmSetEnv(kEnvValue, fail_collect_path.c_str(), 1);

  MakeGraphRefData2HCCL(graph);
  graph->TopologicalSorting();
  MemcpyAddrAsyncPass memcpy_addr;
  EXPECT_EQ(graph->GetAllNodes().size(), 2);
  Status ret = memcpy_addr.Run(graph);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(graph->GetAllNodes().size(),3);
  auto data_node = graph->FindNode("Data");
  auto output_node = graph->FindNode("Node_output");
  EXPECT_EQ(data_node->GetOutDataNodes().at(0)->GetType(),"MemcpyAddrAsync");
  EXPECT_EQ(output_node->GetInDataNodes().at(0)->GetType(),"MemcpyAddrAsync");
  // 清理环境变量
  mmSetEnv(kEnvValue, "", 1);
}
}  // namespace ge
