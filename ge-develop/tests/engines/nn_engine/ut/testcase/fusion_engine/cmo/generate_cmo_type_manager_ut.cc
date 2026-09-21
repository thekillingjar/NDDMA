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
#include <list>

#define private public
#define protected public
#include "cmo/generate_cmo_type_manager.h"
#include "common/configuration.h"
#include "common/fe_op_info_common.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

class GenerateCMOTypeManagerTest : public testing::Test{
protected:
  static void SetUpTestCase() {
    cout << "GenerateCmoTypeManagerTest SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "GenerateCmoTypeManagerTest TearDwon" << endl;
  }
};

TEST_F(GenerateCMOTypeManagerTest, Initialize) {
  auto cmo_manager_ptr = std::make_shared<GenerateCMOTypeManager>();
  Status res = cmo_manager_ptr->Initialize();
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_EQ(cmo_manager_ptr->cmo_generate_map_.size(), 3);

  res = cmo_manager_ptr->Finalize();
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_EQ(cmo_manager_ptr->cmo_generate_map_.size(), 0);

  res = cmo_manager_ptr->Finalize();
  EXPECT_EQ(cmo_manager_ptr->cmo_generate_map_.size(), 0);
}

TEST_F(GenerateCMOTypeManagerTest, Register) {
  auto cmo_manager_ptr = std::make_shared<GenerateCMOTypeManager>();
  Status res = cmo_manager_ptr->Register(CmoType::CMO_TYPE_PREFETCH);
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_EQ(cmo_manager_ptr->cmo_generate_map_.size(), 1);

  res = cmo_manager_ptr->Register(CmoType::CMO_TYPE_BARRIER);
  EXPECT_EQ(res, fe::SUCCESS);
  EXPECT_EQ(cmo_manager_ptr->cmo_generate_map_.size(), 1);

  res = cmo_manager_ptr->Register(CmoType::CMO_TYPE_INVALID);
  EXPECT_EQ(cmo_manager_ptr->cmo_generate_map_.size(), 2);

  res = cmo_manager_ptr->Register(CmoType::CMO_TYPE_WRITEBACK);
  EXPECT_EQ(cmo_manager_ptr->cmo_generate_map_.size(), 3);
}

TEST_F(GenerateCMOTypeManagerTest, GenerateType) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  OpDescPtr op_desc_ptr4 = make_shared<OpDesc>("name4", "type4");
  vector<int64_t> data_dims = {4, 4, 16, 4};
  vector<int> dims_value_vec = {2, 3};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(data_tensor_desc2, (uint8_t *)dims_value_vec.data(), dims_value_vec.size() * sizeof(int));
  vector<ge::GeTensorPtr> weights{dim_tensor};
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc2);
  op_desc_ptr3->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr4->AddInputDesc("input1", data_tensor_desc2);
  
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);
  NodePtr node4 = graph->AddNode(op_desc_ptr4);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  ge::OpDescUtils::SetWeights(node3, weights);
  ge::TensorUtils::SetSize(*node3->GetOpDesc()->MutableInputDesc(1), 4096);
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {2, 0});
  AttrUtils::SetListInt(node4->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {5, 0});
  graph->TopologicalSorting();

  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node2->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node3->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node4->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node1->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 0);
  AttrUtils::SetInt(node2->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 1);
  AttrUtils::SetInt(node3->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 2);
  AttrUtils::SetInt(node4->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 3);
 
  std::vector<int64_t> workspace{222};
  node1->GetOpDesc()->SetWorkspace(workspace);

  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info{};
  ge::MemReuseInfo reuse_info1{node4, MemType::WORKSPACE_MEM, 0};
  ge::MemReuseInfo reuse_info2{node4, MemType::OUTPUT_MEM, 0};
  std::vector<ge::MemReuseInfo> reuse_info_vec1{reuse_info1};
  std::vector<ge::MemReuseInfo> reuse_info_vec2{reuse_info2};
  mem_reuse_info.emplace("workspace0", reuse_info_vec1);
  mem_reuse_info.emplace("output0", reuse_info_vec2);
  node1->GetOpDesc()->SetExtAttr(ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);

  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  auto cmo_manager_ptr = std::make_shared<GenerateCMOTypeManager>();
  cmo_manager_ptr->Initialize();
  for (auto& node_ptr : graph->GetDirectNode()) {
    cmo_manager_ptr->GenerateType(node_ptr, stream_ctls, prefetch_cache_map, stream_node_map);
  }
  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node1->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 2);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);
  EXPECT_EQ(cmo[kCmoPrefetch].size(), 1);

  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);

  cmo = node3->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoWriteback].size(), 1);

  cmo = node4->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoBarrier].size(), 2);
}
