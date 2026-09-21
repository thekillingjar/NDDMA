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
#include <iostream>
#include <string>
#include <vector>
#include "fe_llt_utils.h"
#define protected public
#define private public
#include "common/configuration.h"
#include "adapter/common/op_store_adapter_manager.h"
#include "adapter/tbe_adapter/tbe_op_store_adapter.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reformat_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include "ops_store/ops_kernel_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

using TransNodeReformatGeneratorPtr = shared_ptr<TransNodeReformatGenerator>;
using TransNodeInsertionPtr = shared_ptr<TransNodeInsertion>;

class TRANS_NODE_REFORMAT_GENERATOR_UTEST: public testing::Test {
 protected:
  void SetUp() {
        std::map<std::string, std::string> options;
    fe_ops_kernel_info_store_ptr_ = make_shared<fe::FEOpsKernelInfoStore>();
    FEOpsStoreInfo tbe_custom {
        6,
        "tbe-custom",
        EN_IMPL_HW_TBE,
        GetCodeDir() + "/tests/engines/nn_engine/ut/testcase/fusion_engine/ops_kernel_store/fe_config/tbe_custom_opinfo",
        ""};
    vector<FEOpsStoreInfo> store_info;
    store_info.emplace_back(tbe_custom);
    Configuration::Instance(fe::AI_CORE_NAME).ops_store_info_vector_ = (store_info);
    OpsKernelManager::Instance(AI_CORE_NAME).Finalize();

    fe_ops_kernel_info_store_ptr_->Initialize(options);
  }

  void TearDown() {
    fe_ops_kernel_info_store_ptr_->Finalize();
  }
  shared_ptr<fe::FEOpsKernelInfoStore> fe_ops_kernel_info_store_ptr_;

  static void CreateGraph(ComputeGraphPtr graph) {
    OpDescPtr const_op1 = std::make_shared<OpDesc>("const1", "Const");
    GeTensorDesc const_tensor_desc1(GeShape({1000}), ge::FORMAT_NHWC, ge::DT_INT32);
    const_tensor_desc1.SetOriginShape(GeShape({1000}));
    const_tensor_desc1.SetOriginFormat(ge::FORMAT_NHWC);
    const_op1->AddOutputDesc(const_tensor_desc1);
    const_op1->AddInputDesc(const_tensor_desc1);
    const_tensor_desc1.SetOriginDataType(ge::DT_INT32);
    auto const_node1 = graph->AddNode(const_op1);

    OpDescPtr const_op2 = std::make_shared<OpDesc>("const2", "Const");
    GeTensorDesc const_tensor_desc2(GeShape({1, 1, 1024, 1000}), ge::FORMAT_HWCN, ge::DT_INT8);
    const_tensor_desc2.SetOriginShape(GeShape({3, 4, 5, 6}));
    const_tensor_desc2.SetOriginFormat(ge::FORMAT_HWCN);
    const_op2->AddOutputDesc(const_tensor_desc2);
    const_op2->AddInputDesc(const_tensor_desc2);
    const_tensor_desc2.SetOriginDataType(ge::DT_INT8);
    auto const_node2 = graph->AddNode(const_op2);

    OpDescPtr data_op = std::make_shared<OpDesc>("data", "Data");
    GeTensorDesc data_tensor_desc(GeShape({1}), ge::FORMAT_NCHW, ge::DT_UINT64);
    data_tensor_desc.SetOriginShape(GeShape({1}));
    data_tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    data_tensor_desc.SetOriginDataType(ge::DT_UINT64);
    data_op->AddOutputDesc(data_tensor_desc);
    data_op->AddInputDesc(data_tensor_desc);
    auto data_node = graph->AddNode(data_op);

    OpDescPtr conv_o_p = std::make_shared<OpDesc>("FullyConnection", "FullyConnection");
    GeTensorDesc conv_tensor_desc1(GeShape({8, 32, 1, 1, 32}), ge::FORMAT_NC1HWC0,ge::DT_INT32);
    conv_tensor_desc1.SetOriginShape(GeShape({8, 1, 1, 1024}));
    conv_tensor_desc1.SetOriginFormat(ge::FORMAT_NCHW);
    conv_tensor_desc1.SetOriginDataType(ge::DT_INT32);
    GeTensorDesc conv_tensor_desc_2(GeShape({32, 63, 16, 32}), ge::FORMAT_FRACTAL_Z, ge::DT_INT8);
    conv_tensor_desc_2.SetOriginShape(GeShape({1, 1, 1024, 1000}));
    conv_tensor_desc_2.SetOriginFormat(ge::FORMAT_NCHW);
    conv_tensor_desc_2.SetOriginDataType(ge::DT_INT8);
    GeTensorDesc conv_tensor_desc_3(GeShape({1, 63, 1, 1, 16}), ge::FORMAT_NC1HWC0, ge::DT_UINT64);
    conv_tensor_desc_3.SetOriginShape(GeShape({1000}));
    conv_tensor_desc_3.SetOriginFormat(ge::FORMAT_NCHW);
    conv_tensor_desc_3.SetOriginDataType(ge::DT_INT64);
    GeTensorDesc conv_tensor_desc(GeShape({8, 1, 1, 1000}), ge::FORMAT_NC1HWC0, ge::DT_INT32);
    conv_tensor_desc.SetOriginShape(GeShape({8, 63, 1, 1, 16}));
    conv_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
    conv_tensor_desc.SetOriginDataType(ge::DT_INT32);

    conv_o_p->AddInputDesc(conv_tensor_desc1);
    conv_o_p->AddInputDesc(conv_tensor_desc_2);
    conv_o_p->AddInputDesc(conv_tensor_desc_3);
    conv_o_p->AddOutputDesc(conv_tensor_desc);
    auto conv_node = graph->AddNode(conv_o_p);
    ge::AttrUtils::SetInt(conv_o_p, FE_IMPLY_TYPE, 6);

    GraphUtils::AddEdge(const_node1->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node2->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(2));
  }
};

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_UTEST, AddTransNode_suc) {
  FEOpsKernelInfoStorePtr fe_ops_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  TransInfoPtr trans_info_ptr = make_shared<TransInfo>();
  TransNodeReformatGeneratorPtr trans_node_reformat_generator = make_shared<TransNodeReformatGenerator>(fe_ops_store_ptr, trans_info_ptr);

  ComputeGraphPtr fused_graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_0 = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("relu", "relu");
  vector<int64_t> dim(4, 4);
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_0->AddInputDesc(out_desc);
  op_desc_0->AddOutputDesc(out_desc);
  op_desc_1->AddInputDesc(out_desc);
  op_desc_1->AddOutputDesc(out_desc);
  NodePtr node_add = fused_graph->AddNode(op_desc_0);
  NodePtr node_relu = fused_graph->AddNode(op_desc_1);
  GraphUtils::AddEdge(node_add->GetOutDataAnchor(0), node_relu->GetInDataAnchor(0));

  trans_info_ptr->src_op_desc = op_desc_0;
  trans_info_ptr->dst_op_desc = op_desc_1;
  trans_info_ptr->src_anchor = node_add->GetOutDataAnchor(0);
  trans_info_ptr->dst_anchor = node_relu->GetInDataAnchor(0);
  trans_info_ptr->src_node_ptr = node_add;
  trans_info_ptr->dst_node_ptr = node_relu;
  Status ret = trans_node_reformat_generator->AddTransNode(*fused_graph, trans_info_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_UTEST, AddTransNode_suc1) {
  FEOpsKernelInfoStorePtr fe_ops_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  TransInfoPtr trans_info_ptr = make_shared<TransInfo>();
  TransNodeReformatGeneratorPtr trans_node_reformat_generator = make_shared<TransNodeReformatGenerator>(fe_ops_store_ptr, trans_info_ptr);

  ComputeGraphPtr fused_graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_ref0 = std::make_shared<OpDesc>("ref0", "RefData");
  OpDescPtr op_desc_ref1 = std::make_shared<OpDesc>("ref1", "RefData");
  OpDescPtr op_desc_sqr = std::make_shared<OpDesc>("square", "Square");
  OpDescPtr op_desc_relu = std::make_shared<OpDesc>("relu", "Relu");
  OpDescPtr op_desc_assign = std::make_shared<OpDesc>("assign", "Assign");
  vector<int64_t> dim = {4, 50, 50, 3};
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_ref0->AddInputDesc(out_desc);
  op_desc_ref0->AddOutputDesc(out_desc);
  op_desc_ref1->AddInputDesc(out_desc);
  op_desc_ref1->AddOutputDesc(out_desc);
  op_desc_sqr->AddInputDesc(out_desc);
  op_desc_sqr->AddOutputDesc(out_desc);
  op_desc_relu->AddInputDesc(out_desc);
  op_desc_relu->AddOutputDesc(out_desc);
  op_desc_assign->AddInputDesc(out_desc);
  op_desc_assign->AddOutputDesc(out_desc);
  AttrUtils::SetStr(op_desc_ref1, ge::REF_VAR_SRC_VAR_NAME, "ref0");
  NodePtr node_ref0 = fused_graph->AddNode(op_desc_ref0);
  NodePtr node_ref1 = fused_graph->AddNode(op_desc_ref1);
  NodePtr node_relu = fused_graph->AddNode(op_desc_relu);
  NodePtr node_sqr = fused_graph->AddNode(op_desc_sqr);
  NodePtr node_assign = fused_graph->AddNode(op_desc_assign);
  GraphUtils::AddEdge(node_ref0->GetOutDataAnchor(0), node_sqr->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_assign->GetOutControlAnchor(), node_sqr->GetInControlAnchor());
  GraphUtils::AddEdge(node_ref1->GetOutControlAnchor(), node_sqr->GetInControlAnchor());
  GraphUtils::AddEdge(node_assign->GetOutControlAnchor(), node_ref1->GetInControlAnchor());
  GraphUtils::AddEdge(node_ref0->GetOutDataAnchor(0), node_assign->GetInDataAnchor(0));
  trans_info_ptr->src_op_desc = op_desc_ref0;
  trans_info_ptr->dst_op_desc = op_desc_sqr;
  trans_info_ptr->src_anchor = node_ref0->GetOutDataAnchor(0);
  trans_info_ptr->dst_anchor = node_sqr->GetInDataAnchor(0);
  trans_info_ptr->src_node_ptr = node_ref0;
  trans_info_ptr->dst_node_ptr = node_sqr;
  Status ret = trans_node_reformat_generator->AddTransNode(*fused_graph, trans_info_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_UTEST, generate_strategy_by_orignal_format_test) {
  std::vector<std::vector<uint32_t>> strategy_vector_combination = {{1}, {2}, {2, 1}};
  std::vector<TransInfoPtr> trans_info_front_and_end_;
  FEOpsKernelInfoStorePtr fe_ops_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  TransNodeInsertionPtr trans_node_instertion_ptr = make_shared<TransNodeInsertion>(fe_ops_store_ptr);
  Status ret = trans_node_instertion_ptr->GenerateStrategyByOrignalFormat(strategy_vector_combination);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_UTEST, trans_node_insert_check) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  CreateGraph(graph);
  TransNodeManager trans_op_insert(fe_ops_kernel_info_store_ptr_);
  trans_op_insert.Initialize();
  Status status = trans_op_insert.InsertAndMergeTransNodes(*(graph.get()));
  EXPECT_EQ(status, fe::SUCCESS);
}
