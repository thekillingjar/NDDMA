/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <map>
#include <memory>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "proto/om.pb.h"

#define protected public
#define private public
#include "common/graph_comm.h"
#include "pass_manager.h"
#include "common/configuration.h"
#include "common/fe_op_info_common.h"
#include "common/scope_allocator.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "graph/op_kernel_bin.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/ub_fusion/buffer_fusion.h"
#include "graph_optimizer/fusion_common/fusion_pass_name.h"
#undef protected
#undef private

using namespace std;
using namespace domi;
using namespace fe;
using namespace ge;

class TbeL1FusionSTest : public testing::Test {
 public:

 protected:
  static void SetUpTestCase() { std::cout << "UB fusion SetUp" << std::endl; }
  static void TearDownTestCase() { std::cout << "UB fusion TearDown" << std::endl; }
  std::shared_ptr<BufferFusion> ub_fusion_ptr_;
  std::shared_ptr<FusionPriorityManager> fusion_priority_mgr_ptr_;
  virtual void SetUp() {
    std::shared_ptr<GraphComm> graph_comm_ptr = std::make_shared<GraphComm>("engineName");
    fusion_priority_mgr_ptr_ = std::make_shared<FusionPriorityManager>("engineName", nullptr);
    ub_fusion_ptr_ = std::make_shared<BufferFusion>(graph_comm_ptr, fusion_priority_mgr_ptr_, nullptr);
    ub_fusion_ptr_->engine_name_ = fe::AI_CORE_NAME;
  }
  virtual void TearDown() {

  }

  void SetPattern(ge::OpDescPtr opdef, const string &optype) {
    auto key_pattern = "_pattern";
    ge::AttrUtils::SetStr(opdef, key_pattern, optype);
  }
  void SetTvmType(ge::OpDescPtr opdef) {
    ge::AttrUtils::SetInt(opdef, ge::ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  }

  // conv - > relu -> conv -> relu
  void BuildGraph(ComputeGraphPtr graph, const int32_t &strategy) {
    OpDescPtr data = std::make_shared<OpDesc>("DATA0", fe::DATA);
    OpDescPtr const1 = std::make_shared<OpDesc>("const1", fe::CONSTANT);
    OpDescPtr const2 = std::make_shared<OpDesc>("const2", fe::CONSTANT);
    OpDescPtr conv1 = std::make_shared<OpDesc>("conv1", "Conv2D");
    OpDescPtr conv2 = std::make_shared<OpDesc>("conv2", "Conv2D");
    OpDescPtr relu1 = std::make_shared<OpDesc>("relu1", "RelU");
    OpDescPtr relu2 = std::make_shared<OpDesc>("relu2", "RelU");

    int64_t scope_id_1 = ScopeAllocator::Instance().AllocateScopeId();
    int64_t scope_id_2 = ScopeAllocator::Instance().AllocateScopeId();
    int64_t scope_id_3 = ScopeAllocator::Instance().AllocateScopeId();
    switch (strategy) {
      case 1:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

        ScopeAllocator::SetL1ScopeAttr(conv1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(conv2, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu2, scope_id_3);
        break;
      case 2:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);
        break;
      case 3:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);
        ScopeAllocator::SetL1ScopeAttr(conv1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu1, scope_id_3);
        break;
      case 4:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

        ScopeAllocator::SetL1ScopeAttr(conv2, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu2, scope_id_3);
        break;
      default:
        ScopeAllocator::SetScopeAttr(conv1, scope_id_1);
        ScopeAllocator::SetScopeAttr(relu1, scope_id_1);
        ScopeAllocator::SetScopeAttr(conv2, scope_id_2);
        ScopeAllocator::SetScopeAttr(relu2, scope_id_2);

        ScopeAllocator::SetL1ScopeAttr(conv1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu1, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(conv2, scope_id_3);
        ScopeAllocator::SetL1ScopeAttr(relu2, scope_id_3);
    }

    SetPattern(conv1, "Convolution");
    SetPattern(conv2, "Convolution");
    SetPattern(relu1, "ElemWise");
    SetPattern(relu2, "ElemWise");

    SetTvmType(conv1);
    SetTvmType(conv2);
    SetTvmType(relu1);
    SetTvmType(relu2);

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

    NodePtr data_node = graph->AddNode(data);
    NodePtr const1_node = graph->AddNode(const1);
    NodePtr const2_node = graph->AddNode(const2);
    NodePtr conv1_node = graph->AddNode(conv1);
    NodePtr conv2_node = graph->AddNode(conv2);
    NodePtr relu1_node = graph->AddNode(relu1);
    NodePtr relu2_node = graph->AddNode(relu2);

    const char tbe_bin[] = "tbe_bin";
    vector<char> buffer(tbe_bin, tbe_bin+strlen(tbe_bin));
    ge::OpKernelBinPtr tbe_kernel_ptr = std::make_shared<ge::OpKernelBin>(conv1_node->GetName(), std::move(buffer));
    conv1_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);

    const char tbe_bin2[] = "tbe_bin";
    vector<char> buffer2(tbe_bin2, tbe_bin2+strlen(tbe_bin2));
    ge::OpKernelBinPtr tbe_kernel_ptr2 = std::make_shared<ge::OpKernelBin>(conv2_node->GetName(), std::move(buffer2));
    conv2_node->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr2);

    GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(conv1_node->GetOutDataAnchor(0), relu1_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(relu1_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(conv2_node->GetOutDataAnchor(0), relu2_node->GetInDataAnchor(0));

    GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0), conv1_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0), conv2_node->GetInDataAnchor(1));
  }
};

TEST_F(TbeL1FusionSTest, test_l1_fusion_1) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out, 1);
  graph_out->TopologicalSorting();
  graph_out->Dump();

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  ub_fusion_ptr_->BuildFusionGraph(*graph_out);
  cerr << endl;
  graph_out->Dump();
  cerr << endl;
  cerr << "TbeL1FusionSTest UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    int64_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType();
    if (GetFusionScopeAttr(node->GetOpDesc(), scope_id)) {
      cerr << ", scope id : " << scope_id;
    }
    cerr << endl;
  }
  EXPECT_EQ(graph_out->GetDirectNodesSize(), 4);
}

TEST_F(TbeL1FusionSTest, test_l1_fusion_2) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out, 2);
  graph_out->TopologicalSorting();
  graph_out->Dump();

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  ub_fusion_ptr_->BuildFusionGraph(*graph_out);
  cerr << endl;
  graph_out->Dump();
  cerr << endl;
  cerr << "TbeL1FusionSTest UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    int64_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType();
    if (GetFusionScopeAttr(node->GetOpDesc(), scope_id)) {
      cerr << ", scope id : " << scope_id;
    }
    cerr << endl;
  }
  EXPECT_EQ(graph_out->GetDirectNodesSize(), 5);
}

TEST_F(TbeL1FusionSTest, test_l1_fusion_3) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out, 3);
  graph_out->TopologicalSorting();
  graph_out->Dump();

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  ub_fusion_ptr_->BuildFusionGraph(*graph_out);
  cerr << endl;
  graph_out->Dump();
  cerr << endl;
  cerr << "TbeL1FusionSTest UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    int64_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType();
    if (GetFusionScopeAttr(node->GetOpDesc(), scope_id)) {
      cerr << ", scope id : " << scope_id;
    }
    cerr << endl;
  }
  EXPECT_EQ(graph_out->GetDirectNodesSize(), 5);
}

TEST_F(TbeL1FusionSTest, test_l1_fusion_4) {
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out, 4);
  graph_out->TopologicalSorting();
  graph_out->Dump();

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  ub_fusion_ptr_->BuildFusionGraph(*graph_out);
  cerr << endl;
  graph_out->Dump();
  cerr << endl;
  cerr << "TbeL1FusionSTest UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    int64_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType();
    if (GetFusionScopeAttr(node->GetOpDesc(), scope_id)) {
      cerr << ", scope id : " << scope_id;
    }
    cerr << endl;
  }
  EXPECT_EQ(graph_out->GetDirectNodesSize(), 5);
}


TEST_F(TbeL1FusionSTest, coverage_01) {
  std::string scope_attr = "test";
  GraphCommPtr graph_comm_ptr = nullptr;
  FusionGraphMerge merge(scope_attr, graph_comm_ptr);
  int64_t origin_index = 1;
  int64_t fusion_index = 2;
  L2FusionInfoPtr origin_l2_info = std::make_shared<TaskL2FusionInfo_t>();
  L2FusionInfoPtr fusion_l2_info = std::make_shared<TaskL2FusionInfo_t>();
  L2FusionData_t data;
  origin_l2_info->output.emplace(std::make_pair(origin_index, data));

  merge.UpdateL2Info(origin_index, fusion_index,
                     origin_l2_info, fusion_l2_info);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  for (uint32_t i = 0; i < L2_MAXDATANUM; i++) {
    origin_l2_info->l2_info.node_name[i] = "test";
  }

  std::map<std::int64_t, ge::NodePtr> inner;
  merge.fusion_op_name_map_all_.emplace(std::make_pair("test", inner));
  EXPECT_EQ(merge.SetL2NameAndIndex(origin_l2_info, fusion_l2_info), fe::SUCCESS);
}

TEST_F(TbeL1FusionSTest, coverage_02) {
  std::string scope_attr = "test";
  GraphCommPtr graph_comm_ptr = nullptr;
  FusionGraphMerge merge(scope_attr, graph_comm_ptr);
  int64_t origin_index = 1;
  int64_t fusion_index = 2;
  L2FusionInfoPtr origin_l2_info = std::make_shared<TaskL2FusionInfo_t>();
  L2FusionInfoPtr fusion_l2_info = std::make_shared<TaskL2FusionInfo_t>();
  L2FusionData_t data;
  origin_l2_info->output.emplace(std::make_pair(origin_index, data));

  merge.UpdateL2Info(origin_index, fusion_index,
      origin_l2_info, fusion_l2_info);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("test", "Test");
  vector<int64_t> dims = {1, 2, 3};
  ge::GeShape shape = ge::GeShape(dims);
  ge::GeTensorDesc tensor(shape);
  op->AddInputDesc(tensor);
  op->AddOutputDesc(tensor);

  auto node = graph->AddNode(op);

  for (uint32_t i = 0; i < L2_MAXDATANUM; i++) {
    origin_l2_info->l2_info.node_name[i] = "test";
  }

  std::map<std::int64_t, ge::NodePtr> inner;
  inner.emplace(std::make_pair(0, node));

  merge.fusion_op_name_map_all_.emplace(std::make_pair("test", inner));

  EXPECT_EQ(merge.SetL2NameAndIndex(origin_l2_info, fusion_l2_info), fe::SUCCESS);
}

TEST_F(TbeL1FusionSTest, coverage_03) {
  std::string scope_attr = "test";
  GraphCommPtr graph_comm_ptr = nullptr;
  FusionGraphMerge merge(scope_attr, graph_comm_ptr);
  int64_t origin_index = 1;
  int64_t fusion_index = 2;
  L2FusionInfoPtr origin_l2_info = std::make_shared<TaskL2FusionInfo_t>();
  L2FusionInfoPtr fusion_l2_info = std::make_shared<TaskL2FusionInfo_t>();
  L2FusionData_t data;
  origin_l2_info->output.emplace(std::make_pair(origin_index, data));

  merge.UpdateL2Info(origin_index, fusion_index,
                     origin_l2_info, fusion_l2_info);

  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  ge::OpDescPtr op = std::make_shared<ge::OpDesc>("test", "Test");
  vector<int64_t> dims = {1, 2, 3};
  ge::GeShape shape = ge::GeShape(dims);
  ge::GeTensorDesc tensor(shape);
  op->AddInputDesc(tensor);

  ge::AttrUtils::SetInt(tensor, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX,0);
  op->AddOutputDesc(tensor);
  auto node = graph->AddNode(op);

  for (uint32_t i = 0; i < L2_MAXDATANUM; i++) {
    origin_l2_info->l2_info.node_name[i] = "test";
  }

  std::map<std::int64_t, ge::NodePtr> inner;
  inner.emplace(std::make_pair(0, node));

  merge.fusion_op_name_map_all_.emplace(std::make_pair("test", inner));

  EXPECT_EQ(merge.SetL2NameAndIndex(origin_l2_info, fusion_l2_info), fe::SUCCESS);
}

TEST_F(TbeL1FusionSTest, coverage_04) {
  setenv("DUMP_GE_GRAPH", "2", 1);
  setenv("DUMP_GRAPH_LEVEL", "3", 1);
  ComputeGraphPtr graph_out = std::make_shared<ComputeGraph>("test");
  BuildGraph(graph_out, 5);
  graph_out->TopologicalSorting();
  graph_out->Dump();

  Configuration::Instance(AI_CORE_NAME).config_param_vec_[static_cast<size_t>(CONFIG_PARAM::BufferOptimize)] =
          static_cast<int64_t>(EN_L1_OPTIMIZE);
  // create fused Graph, and merge matched sub-graphs into fusion ops
  ub_fusion_ptr_->BuildFusionGraph(*graph_out);
  cerr << endl;
  graph_out->Dump();
  cerr << endl;
  cerr << "TbeL1FusionUnitTest UB fusion result" << endl;
  for (auto &node : graph_out->GetDirectNode()) {
    int64_t scope_id = 0;
    cerr << "name: " << node->GetName() << ", type:" << node->GetOpDesc()->GetType();
    if (GetFusionScopeAttr(node->GetOpDesc(), scope_id)) {
      cerr << ", scope id : " << scope_id;
    }
    cerr << endl;
  }
  EXPECT_EQ(graph_out->GetDirectNodesSize(), 4);
  unsetenv("DUMP_GE_GRAPH");
  unsetenv("DUMP_GRAPH_LEVEL");
}
