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
#include "cmo/generate_cmo_type_prefetch.h"
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

using GenerateCMOTypeBasePtr = std::shared_ptr<GenerateCMOTypeBase>;
class GenerateCmoTypePrefetchTest : public testing::Test{
protected:
  static void SetUpTestCase() {
    cout << "GenerateCmoTypePrefetchTest SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "GenerateCmoTypePrefetchTest TearDwon" << endl;
  }
  
  virtual void SetUp() {
    cmo_type_base_ = std::make_shared<GenerateCMOTypePrefetch>();
  }

  virtual void TearDown() {
  }
  
public:
  GenerateCMOTypeBasePtr cmo_type_base_;
};

TEST_F(GenerateCmoTypePrefetchTest, GenerateTypeNoWeights) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("name", "type");
  vector<int64_t> data_dims={2};
  GeTensorDesc data_tensor_desc(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc("input1", data_tensor_desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc_ptr);
  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  cmo_type_base_->GenerateType(node, stream_ctls, prefetch_cache_map, stream_node_map);

  map <std::string, std::vector<CmoAttr>> cmo;
  cmo = node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypePrefetchTest, GenerateTypeNoParent) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("name", "type");
  vector<int64_t> data_dims={2};
  vector<int> dims_value_vec = {2, 3};
  GeTensorDesc data_tensor_desc(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)dims_value_vec.data(), 8);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc_ptr);
  vector<ge::GeTensorPtr> weights{dim_tensor};
  ge::OpDescUtils::SetWeights(node, weights);

  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  cmo_type_base_->GenerateType(node, stream_ctls, prefetch_cache_map, stream_node_map);
  map <std::string, std::vector<CmoAttr>> cmo;
  cmo = node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypePrefetchTest, GenerateTypePrefetch) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  vector<int64_t> data_dims={4, 4, 16, 4};
  vector<int64_t> large_data_dims={4, 4, 256, 1024};
  vector<int> dims_value_vec = {2, 3};
  GeTensorDesc data_tensor_desc(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_large_tensor_desc(GeShape(large_data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(data_tensor_desc, (uint8_t *)dims_value_vec.data(), dims_value_vec.size() * sizeof(int));
  GeTensorPtr dim_large_tensor = std::make_shared<GeTensor>(data_large_tensor_desc, (uint8_t *)dims_value_vec.data(), dims_value_vec.size() * sizeof(int));
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc);
  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);
  vector<ge::GeTensorPtr> weights{dim_tensor, dim_large_tensor};
  ge::OpDescUtils::SetWeights(node3, weights);
  ge::TensorUtils::SetSize(*node3->GetOpDesc()->MutableInputDesc(1), 4096);
  ge::TensorUtils::SetSize(*node3->GetOpDesc()->MutableInputDesc(2), 1048576);
  ge::AttrUtils::SetInt(node1->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, 0);
  ge::AttrUtils::SetInt(node2->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, 1);
  ge::AttrUtils::SetInt(node3->GetOpDesc(), ge::ATTR_NAME_OP_READ_WRITE_INDEX, 2);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node2->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));

  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  cmo_type_base_->GenerateType(node1, stream_ctls, prefetch_cache_map, stream_node_map);
  cmo_type_base_->GenerateType(node2, stream_ctls, prefetch_cache_map, stream_node_map);
  cmo_type_base_->GenerateType(node3, stream_ctls, prefetch_cache_map, stream_node_map);
  map <std::string, std::vector<CmoAttr>> cmo;
  cmo = node1->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoPrefetch].size(), 1);
}

TEST_F(GenerateCmoTypePrefetchTest, GenerateTypeSamePrefetch) {
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  OpDescPtr op_desc_ptr4 = make_shared<OpDesc>("name4", "type4");
  OpDescPtr op_const = make_shared<OpDesc>("const", "Const");

  GeTensorDesc const_tensor_desc(GeShape({4, 4, 16, 4}), FORMAT_NCHW, DT_FLOAT);
  op_const->AddInputDesc(const_tensor_desc);
  op_const->AddOutputDesc(const_tensor_desc);
  NodePtr node_const = graph->AddNode(op_const);

  GeTensorDesc data_tensor_desc(GeShape({4, 4, 16, 4}), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc);
  NodePtr node1 = graph->AddNode(op_desc_ptr1);

  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);

  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc);
  op_desc_ptr3->AddInputDesc("input2", data_tensor_desc);
  op_desc_ptr3->AddOutputDesc("output1", data_tensor_desc);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);

  op_desc_ptr4->AddInputDesc("input1", data_tensor_desc);
  op_desc_ptr4->AddInputDesc("input2", data_tensor_desc);
  op_desc_ptr4->AddOutputDesc("output1", data_tensor_desc);
  NodePtr node4 = graph->AddNode(op_desc_ptr4);

  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node3->GetInDataAnchor(1));
  GraphUtils::AddEdge(node_const->GetOutDataAnchor(0), node4->GetInDataAnchor(1));
  vector<int> dims_value_vec = {2, 3};
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(const_tensor_desc, (uint8_t *)dims_value_vec.data(), dims_value_vec.size() * sizeof(int));
  vector<ge::GeTensorPtr> weights;
  weights.emplace_back(dim_tensor);
  ge::OpDescUtils::SetWeights(node3, weights);
  ge::OpDescUtils::SetWeights(node4, weights);
  ge::TensorUtils::SetSize(*node3->GetOpDesc()->MutableInputDesc(1), 4096);
  ge::TensorUtils::SetSize(*node4->GetOpDesc()->MutableInputDesc(1), 4096);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node2->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node1->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 0);
  AttrUtils::SetInt(node2->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 1);
  AttrUtils::SetInt(node3->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 2);
  AttrUtils::SetInt(node4->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 3);
  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  cmo_type_base_->GenerateType(node1, stream_ctls, prefetch_cache_map, stream_node_map);
  cmo_type_base_->GenerateType(node2, stream_ctls, prefetch_cache_map, stream_node_map);
  map <std::string, std::vector<CmoAttr>> cmo;
  cmo_type_base_->GenerateType(node3, stream_ctls, prefetch_cache_map, stream_node_map);
  cmo.clear();
  cmo = node1->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoPrefetch].size(), 1);
  cmo_type_base_->GenerateType(node4, stream_ctls, prefetch_cache_map, stream_node_map);
  cmo.clear();
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}
