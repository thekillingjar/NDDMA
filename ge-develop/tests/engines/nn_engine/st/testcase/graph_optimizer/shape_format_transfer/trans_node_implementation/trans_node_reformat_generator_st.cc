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

#define protected public
#define private public
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_reformat_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_transdata_generator.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"
#include "common/range_format_transfer/transfer_range_according_to_format.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

using TransNodeReformatGeneratorPtr = shared_ptr<TransNodeReformatGenerator>;
using TransNodeInsertionPtr = shared_ptr<TransNodeInsertion>;
using TransNodeTransDataGeneratorPtr = shared_ptr<TransNodeTransdataGenerator>;
class TRANS_NODE_REFORMAT_GENERATOR_STEST: public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_STEST, AddTransNode_suc) {
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

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_STEST, AddTransNode_suc1) {
  FEOpsKernelInfoStorePtr fe_ops_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  TransInfoPtr trans_info_ptr = make_shared<TransInfo>();
  TransNodeTransDataGeneratorPtr trans_node_transdata_generator = make_shared<TransNodeTransdataGenerator>(fe_ops_store_ptr, trans_info_ptr);

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
  Status ret = trans_node_transdata_generator->AddTransNode(*fused_graph, trans_info_ptr);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_STEST, AddTransNode_suc2) {
  FEOpsKernelInfoStorePtr fe_ops_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  TransInfoPtr trans_info_ptr = make_shared<TransInfo>();
  TransNodeTransDataGeneratorPtr trans_node_transdata_generator = make_shared<TransNodeTransdataGenerator>(fe_ops_store_ptr, trans_info_ptr);

  ComputeGraphPtr fused_graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc_var0 = std::make_shared<OpDesc>("var0", "Variable");
  OpDescPtr op_desc_var1 = std::make_shared<OpDesc>("va1r", "Variable");
  OpDescPtr op_desc_reshape = std::make_shared<OpDesc>("reshape", "Reshape");
  OpDescPtr op_desc_trans = std::make_shared<OpDesc>("trans", "TransData");
  OpDescPtr op_desc_assign = std::make_shared<OpDesc>("assign", "Assign");
  vector<int64_t> dim = {4, 50, 50, 3};
  GeShape shape(dim);
  GeTensorDesc out_desc(shape);
  op_desc_var0->AddInputDesc(out_desc);
  op_desc_var0->AddOutputDesc(out_desc);
  op_desc_var1->AddInputDesc(out_desc);
  op_desc_var1->AddOutputDesc(out_desc);
  op_desc_trans->AddInputDesc(out_desc);
  op_desc_trans->AddOutputDesc(out_desc);
  op_desc_reshape->AddInputDesc(out_desc);
  op_desc_reshape->AddOutputDesc(out_desc);
  op_desc_assign->AddInputDesc(out_desc);
  op_desc_assign->AddOutputDesc(out_desc);
  AttrUtils::SetStr(op_desc_var1, ge::REF_VAR_SRC_VAR_NAME, "var0");
  NodePtr node_var0 = fused_graph->AddNode(op_desc_var0);
  NodePtr node_var1 = fused_graph->AddNode(op_desc_var1);
  NodePtr node_reshape = fused_graph->AddNode(op_desc_reshape);
  NodePtr node_trans = fused_graph->AddNode(op_desc_trans);
  NodePtr node_assign = fused_graph->AddNode(op_desc_assign);
  GraphUtils::AddEdge(node_var0->GetOutDataAnchor(0), node_trans->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_trans->GetOutDataAnchor(0), node_assign->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_assign->GetOutDataAnchor(0), node_reshape->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_var1->GetOutControlAnchor(), node_reshape->GetInControlAnchor());
  trans_info_ptr->src_op_desc = op_desc_assign;
  trans_info_ptr->dst_op_desc = op_desc_reshape;
  trans_info_ptr->src_anchor = node_assign->GetOutDataAnchor(0);
  trans_info_ptr->dst_anchor = node_reshape->GetInDataAnchor(0);
  trans_info_ptr->src_node_ptr = node_assign;
  trans_info_ptr->dst_node_ptr = node_reshape;
  Status ret = trans_node_transdata_generator->AddTransNode(*fused_graph, trans_info_ptr);

  int count_node = 0;
  bool has_control_edge = false;
  for (auto node : fused_graph->GetDirectNode()) {
    ASSERT_NE(node, nullptr);
    if (node->GetType() != "TransData" || node->GetName() == "trans") {
      continue;
    }
    count_node++;
    has_control_edge = node->GetInControlNodes().size() != 0;
  }
  EXPECT_EQ(count_node, 1);
  EXPECT_EQ(has_control_edge, true);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_STEST, generate_strategy_by_orignal_format_test) {
  std::vector<std::vector<uint32_t>> strategy_vector_combination = {{1}, {2}, {2, 1}};
  std::vector<TransInfoPtr> trans_info_front_and_end_;
  FEOpsKernelInfoStorePtr fe_ops_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  TransNodeInsertionPtr trans_node_instertion_ptr = make_shared<TransNodeInsertion>(fe_ops_store_ptr);
  Status ret = trans_node_instertion_ptr->GenerateStrategyByOrignalFormat(strategy_vector_combination);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_STEST, InsertTransDataNode)
{
  // src:cce op, dst:cce op
  std::vector<std::pair<int64_t, int64_t>> new_range = {{0,1}};
  const int64_t impl_type = 1;
  std::vector<std::pair<int64_t, int64_t>> range_value = {{1, 1}, {2, 2}, {3, 5}, {6, 8}, {1, 7}, {32, 32}};
  std::vector<std::pair<int64_t, int64_t>> nd_value = {{1, 1}, {2, 2}, {3, 5}, {6, 8}, {1, 7}, {32, 32}};
  EXPECT_EQ(RangeTransferAccordingToFormat ::GetFzWinoRangeByAxisValue(new_range, impl_type, range_value, nd_value), fe::SUCCESS);
}

TEST_F(TRANS_NODE_REFORMAT_GENERATOR_STEST, trans_node_instertion_test) {
  TransInfoPtr trans_info_ptr;
  uint32_t front_strategy_vector_index;
  uint32_t end_strategy_vector_index;
  std::vector<std::vector<uint32_t>> strategy_vector_combination;
  FEOpsKernelInfoStorePtr fe_ops_store_ptr = make_shared<FEOpsKernelInfoStore>(fe::AI_CORE_NAME);
  TransNodeInsertionPtr trans_node_instertion_ptr = make_shared<TransNodeInsertion>(fe_ops_store_ptr);
  Status ret = trans_node_instertion_ptr->InsertCastGeneralCase(trans_info_ptr, front_strategy_vector_index,
                                                                end_strategy_vector_index, strategy_vector_combination);
  EXPECT_EQ(ret, fe::FAILED);
  uint64_t global_strategy_id;
  ret = trans_node_instertion_ptr->CombineAllStrategy(trans_info_ptr, global_strategy_id, strategy_vector_combination);
  EXPECT_EQ(ret, fe::FAILED);
}
