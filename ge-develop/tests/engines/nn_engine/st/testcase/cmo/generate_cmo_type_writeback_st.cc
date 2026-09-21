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
#include "cmo/generate_cmo_type_writeback.h"
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
class GenerateCmoTypeWritebackTest : public testing::Test{
protected:
  static void SetUpTestCase() {
    cout << "GenerateCmoTypeWritebackTest SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "GenerateCmoTypeWritebackTest TearDwon" << endl;
  }
  
  virtual void SetUp() {
    cmo_type_base_ = std::make_shared<GenerateCMOTypeWriteback>();
  }

  virtual void TearDown() {
  }
  
public:
  GenerateCMOTypeBasePtr cmo_type_base_;
};

TEST_F(GenerateCmoTypeWritebackTest, GenerateTypeNoParent) {
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

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeWritebackTest, GenerateTypeNoLifeCycleEnd) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  cmo_type_base_->GenerateType(node2, stream_ctls, prefetch_cache_map, stream_node_map);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeWritebackTest, GenerateTypeLackReadDistance) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {0, 0});
  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  cmo_type_base_->GenerateType(node2, stream_ctls, prefetch_cache_map, stream_node_map);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeWritebackTest, GenerateTypeOK) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {6, 0});
  StreamCtrlMap stream_ctls;
  std::unordered_map<ge::NodePtr, ge::NodePtr> prefetch_cache_map;
  std::map<uint32_t, std::map<int64_t, ge::NodePtr>> stream_node_map;
  cmo_type_base_->GenerateType(node2, stream_ctls, prefetch_cache_map, stream_node_map);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node1->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoWriteback].size(), 1);
}
