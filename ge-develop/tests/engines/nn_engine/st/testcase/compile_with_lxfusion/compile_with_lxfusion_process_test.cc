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
#include "fe_llt_utils.h"
#include "common/fe_op_info_common.h"
#include "common/scope_allocator.h"
#include "compute_graph.h"
#include "graph/utils/attr_utils.h"
#include "itf_handler/itf_handler.h"

#define protected public
#define private public
#include "fusion_manager/fusion_manager.h"
#undef private
#undef protected

using namespace ge;
namespace fe {
class CompileWithLxFusionProcessTest : public testing::Test {
protected:
  static void SetUpTestCase() {
    cout << "CompileWithLxFusionProcessTest TearDown" << endl;
    InitWithSocVersion("Ascend910B1", "allow_fp32_to_fp16");
    FEGraphOptimizerPtr graph_optimizer_ptr = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
    map<string, string> options;
    CreateAndCopyJsonFile();
    EXPECT_EQ(graph_optimizer_ptr->Initialize(options, nullptr), SUCCESS);
  }

  static void TearDownTestCase() {
    cout << "CompileWithLxFusionProcessTest TearDown" << endl;
    Finalize();
  }

  /**
   *    Const \
   * Data -> Conv2D -> Relu -> Add -> SoftmaxV2
   *          Data -> Sigmoid -/
   *
   */
  ge::ComputeGraphPtr CreateGraphWithType(const int32_t &type) {
    vector<int64_t> dims = {3, 4, 5, 6};
    ge::GeShape shape(dims);
    ge::GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
    tensor_desc.SetOriginShape(shape);
    tensor_desc.SetOriginDataType(ge::DT_FLOAT);
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);

    OpDescPtr data1_op = std::make_shared<OpDesc>("data1", "PlaceHolder");
    OpDescPtr data2_op = std::make_shared<OpDesc>("data2", "PlaceHolder");
    OpDescPtr conv_op = std::make_shared<OpDesc>("conv", "Conv2D");
    OpDescPtr relu_op = std::make_shared<OpDesc>("relu", "Relu");
    OpDescPtr const_op = std::make_shared<OpDesc>("const", "Const");
    OpDescPtr softmax_op = std::make_shared<OpDesc>("softmax", "SoftmaxV2");
    OpDescPtr add_op = std::make_shared<OpDesc>("add", "Add");
    OpDescPtr sigmoid_op = std::make_shared<OpDesc>("sigmoid", "Sigmoid");

    data1_op->AddOutputDesc(tensor_desc);
    data2_op->AddOutputDesc(tensor_desc);
    const_op->AddOutputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddInputDesc(tensor_desc);
    conv_op->AddOutputDesc(tensor_desc);
    relu_op->AddInputDesc(tensor_desc);
    relu_op->AddOutputDesc(tensor_desc);
    sigmoid_op->AddInputDesc(tensor_desc);
    sigmoid_op->AddOutputDesc(tensor_desc);
    add_op->AddInputDesc(tensor_desc);
    add_op->AddInputDesc(tensor_desc);
    add_op->AddOutputDesc(tensor_desc);
    softmax_op->AddInputDesc(tensor_desc);
    softmax_op->AddOutputDesc(tensor_desc);

    AttrUtils::SetInt(conv_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(relu_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(sigmoid_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(add_op, "_fe_imply_type", 6);
    AttrUtils::SetInt(softmax_op, "_fe_imply_type", 6);

    AttrUtils::SetListInt(conv_op, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv_op, "dilations", {1, 1, 1, 1});

    AttrUtils::SetBool(conv_op, ATTR_NAME_IS_FIRST_NODE, true);
    AttrUtils::SetBool(softmax_op, ATTR_NAME_IS_LAST_NODE, true);

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data1_node = graph->AddNode(data1_op);
    NodePtr const_node = graph->AddNode(const_op);
    NodePtr conv_node = graph->AddNode(conv_op);
    NodePtr relu_node = graph->AddNode(relu_op);
    NodePtr data2_node = graph->AddNode(data2_op);
    NodePtr sigmoid_node = graph->AddNode(sigmoid_op);
    NodePtr add_node = graph->AddNode(add_op);
    NodePtr softmax_node = graph->AddNode(softmax_op);
    GraphUtils::AddEdge(data1_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(const_node->GetOutDataAnchor(0), conv_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv_node->GetOutDataAnchor(0), relu_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(data2_node->GetOutDataAnchor(0), sigmoid_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sigmoid_node->GetOutDataAnchor(0), add_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(add_node->GetOutDataAnchor(0), softmax_node->GetInDataAnchor(0));

    int64_t scope_id = ScopeAllocator::Instance().AllocateScopeId();
    switch (type) {
      case 1:
        ScopeAllocator::SetScopeAttr(conv_op, scope_id);
        ScopeAllocator::SetScopeAttr(relu_op, scope_id);
        ge::AttrUtils::SetInt(add_op, "_llt_fusion_scope", scope_id);
        break;
      case 2:
        ScopeAllocator::SetScopeAttr(conv_op, scope_id);
        ScopeAllocator::SetScopeAttr(relu_op, scope_id);
        ge::AttrUtils::SetInt(add_op, "_llt_fusion_scope", scope_id);
        ge::AttrUtils::SetBool(add_op, "_ub_compile_fail", true);
        break;
      case 3:
        ge::AttrUtils::SetBool(graph, "_is_pass_changed", true);
        break;
    }

    graph->TopologicalSorting();
    AttrUtils::SetStr(graph, ge::ATTR_NAME_SESSION_GRAPH_ID, "11_22");
    return graph;
  }

  ComputeGraphPtr CreateTwoCubeGraph(const int32_t &strategy) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::OP_TYPE_PLACE_HOLDER);
    OpDescPtr const1 = std::make_shared<OpDesc>("const1", fe::CONSTANT);
    OpDescPtr const2 = std::make_shared<OpDesc>("const2", fe::CONSTANT);
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", "Conv2D");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "Relu");
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "Relu");

    AttrUtils::SetInt(conv1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(conv2, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(relu1, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);
    AttrUtils::SetInt(relu2, FE_IMPLY_TYPE, fe::EN_IMPL_HW_TBE);

    // add descriptor
    vector<int64_t> dim = {4, 4, 1, 4};
    GeShape shape(dim);
    GeTensorDesc tenosr_desc(shape);

    data->AddOutputDesc(tenosr_desc);
    const1->AddOutputDesc(tenosr_desc);
    const2->AddOutputDesc(tenosr_desc);

    conv1->AddInputDesc(tenosr_desc);
    conv1->AddInputDesc(tenosr_desc);
    conv1->AddOutputDesc(tenosr_desc);

    conv2->AddInputDesc(tenosr_desc);
    conv2->AddInputDesc(tenosr_desc);
    conv2->AddOutputDesc(tenosr_desc);

    relu1->AddInputDesc(tenosr_desc);
    relu1->AddOutputDesc(tenosr_desc);
    relu2->AddInputDesc(tenosr_desc);
    relu2->AddOutputDesc(tenosr_desc);

    AttrUtils::SetListInt(conv1, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv1, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv1, "dilations", {1, 1, 1, 1});

    AttrUtils::SetListInt(conv2, "strides", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv2, "pads", {1, 1, 1, 1});
    AttrUtils::SetListInt(conv2, "dilations", {1, 1, 1, 1});

    ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
    NodePtr data_node = graph->AddNode(data);
    NodePtr const1_node = graph->AddNode(const1);
    NodePtr const2_node = graph->AddNode(const2);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);

    int64_t scope_id_1 = ScopeAllocator::Instance().AllocateScopeId();
    int64_t scope_id_2 = ScopeAllocator::Instance().AllocateScopeId();
    ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
    ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
    ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
    ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

    int64_t l1_scope_id = ScopeAllocator::Instance().AllocateScopeId();
    ge::AttrUtils::SetInt(conv1, "_llt_l1_scope", l1_scope_id);
    ge::AttrUtils::SetInt(conv2, "_llt_l1_scope", l1_scope_id);
    ge::AttrUtils::SetInt(relu1, "_llt_l1_scope", l1_scope_id);
    ge::AttrUtils::SetInt(relu2, "_llt_l1_scope", l1_scope_id);
    switch (strategy) {
      case 1:
        break;
      case 2:
        ge::AttrUtils::SetBool(conv1, "_l1_compile_fail", true);
        break;
      case 3:
        ge::AttrUtils::SetBool(conv1, "_l1_compile_fail", true);
        ge::AttrUtils::SetBool(conv1, "_ub_compile_fail", true);
        ge::AttrUtils::SetBool(conv2, "_ub_compile_fail", true);
        break;
    }

    ge::AttrUtils::SetStr(data_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(const1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(const2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(conv1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(conv2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu1_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");
    ge::AttrUtils::SetStr(relu2_node->GetOpDesc(), OPS_PATH_NAME_PREFIX, "");

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(1));
    return graph;
  }
};

TEST_F(CompileWithLxFusionProcessTest, l2_fusion_process_case1) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(1);
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  // EXPECT_EQ(graph_ptr->GetDirectNodesSize(), 9);
}

TEST_F(CompileWithLxFusionProcessTest, l2_fusion_process_case2) {
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(2);
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  // EXPECT_EQ(graph_ptr->GetDirectNodesSize(), 10);
}

TEST_F(CompileWithLxFusionProcessTest, l2_fusion_process_case3) {
  ge::ComputeGraphPtr graph_ptr = CreateGraphWithType(3);
  FusionInfo info1{graph_ptr->GetGraphID(), std::to_string(graph_ptr->GetSessionID()), "TbeConvStridedwriteFusionPass", 8};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info1);
  FusionInfo info2{graph_ptr->GetGraphID(), std::to_string(graph_ptr->GetSessionID()), "MatmulConfusiontransposeUbFusion", 2};
  FusionStatisticRecorder::Instance().UpdateBufferFusionMatchTimes(info2);
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph_ptr);
  // EXPECT_EQ(ret, SUCCESS);
  // EXPECT_EQ(graph_ptr->GetDirectNodesSize(), 11);
  std::string session_and_graph_id = std::to_string(graph_ptr->GetSessionID()) + "_" + std::to_string(graph_ptr->GetGraphID());
  auto iter = FusionStatisticRecorder::Instance().buffer_fusion_info_map_.find(session_and_graph_id);
  if (iter != FusionStatisticRecorder::Instance().buffer_fusion_info_map_.end()) {
    for (auto iter1 = iter->second.begin(); iter1 != iter->second.end(); iter1++) {
      if (iter1->second.GetPassName() == "TbeConvStridedwriteFusionPass") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), 13);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 10);
      }
      if (iter1->second.GetPassName() == "MatmulConfusiontransposeUbFusion") {
        EXPECT_EQ(iter1->second.GetMatchTimes(), -5);
        EXPECT_EQ(iter1->second.GetRepoHitTimes(), 3);
      }
    }
  }
}

TEST_F(CompileWithLxFusionProcessTest, l1_fusion_process_case1) {
  // L1 optimize effect and L1 Fusion will be success
  BufferOptimize buffer_optimize = Configuration::Instance(fe::AI_CORE_NAME).GetBufferOptimize();
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ComputeGraphPtr graph = CreateTwoCubeGraph(1);
  // EXPECT_EQ(graph_optimizer->OptimizeFusedGraph(*graph), fe::SUCCESS);

  // EXPECT_EQ(graph->GetDirectNodesSize(), 10);
  uint32_t conv_count = 0;
  uint32_t relu_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (node->GetType() == "Conv2D") {
      // EXPECT_EQ(op_desc->HasAttr(L1_SCOPE_ID_ATTR), true);
      EXPECT_EQ(op_desc->HasAttr(FAILED_L1_SCOPE_ID_ATTR), false);
      conv_count++;
    }
    if (node->GetType() == "Relu") {
      relu_count++;
    }
  }
  // EXPECT_EQ(conv_count, 1);
  // EXPECT_EQ(relu_count, 0);
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(buffer_optimize);
}

TEST_F(CompileWithLxFusionProcessTest, l1_fusion_process_case2) {
  BufferOptimize buffer_optimize = Configuration::Instance(fe::AI_CORE_NAME).GetBufferOptimize();
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ComputeGraphPtr graph = CreateTwoCubeGraph(2);
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph);
  // EXPECT_EQ(ret, fe::SUCCESS);

  // EXPECT_EQ(graph->GetDirectNodesSize(), 11);
  uint32_t conv_count = 0;
  uint32_t relu_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (node->GetType() == "Conv2D") {
      EXPECT_EQ(op_desc->HasAttr(L1_SCOPE_ID_ATTR), false);
      // EXPECT_EQ(op_desc->HasAttr(FAILED_L1_SCOPE_ID_ATTR), true);
      conv_count++;
    }
    if (node->GetType() == "Relu") {
      relu_count++;
    }
  }
  EXPECT_EQ(conv_count, 2);
  // EXPECT_EQ(relu_count, 0);
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(buffer_optimize);
}

TEST_F(CompileWithLxFusionProcessTest, l1_fusion_process_case3) {
  BufferOptimize buffer_optimize = Configuration::Instance(fe::AI_CORE_NAME).GetBufferOptimize();
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  FEGraphOptimizerPtr graph_optimizer = FusionManager::Instance(AI_CORE_NAME).graph_opt_;
  EXPECT_NE(graph_optimizer, nullptr);
  ComputeGraphPtr graph = CreateTwoCubeGraph(3);
  Status ret = graph_optimizer->OptimizeFusedGraph(*graph);
  // EXPECT_EQ(ret, fe::SUCCESS);

  // EXPECT_EQ(graph->GetDirectNodesSize(), 13);
  uint32_t conv_count = 0;
  uint32_t relu_count = 0;
  for (auto &node : graph->GetDirectNode()) {
    ge::OpDescPtr op_desc = node->GetOpDesc();
    if (node->GetType() == "Conv2D") {
      EXPECT_EQ(op_desc->HasAttr(L1_SCOPE_ID_ATTR), false);
      // EXPECT_EQ(op_desc->HasAttr(FAILED_L1_SCOPE_ID_ATTR), true);
      conv_count++;
    }
    if (node->GetType() == "Relu") {
      relu_count++;
    }
  }
  EXPECT_EQ(conv_count, 2);
  EXPECT_EQ(relu_count, 2);
  Configuration::Instance(fe::AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(buffer_optimize);
}
}
