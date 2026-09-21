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
#include "cmo/generate_cmo_type_invalid.h"
#include "common/configuration.h"
#include "common/fe_op_info_common.h"
#include "common/aicore_util_types.h"
#include "common/aicore_util_attr_define.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/debug/ge_attr_define.h"
#undef private
#undef protected

using namespace std;
using namespace fe;
using namespace ge;

using GenerateCMOTypeInvalidPtr = std::shared_ptr<GenerateCMOTypeInvalid>;
class GenerateCmoTypeInvalidTest : public testing::Test{
protected:
  static void SetUpTestCase() {
    cout << "GenerateCmoTypeInvalidTest SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "GenerateCmoTypeInvalidTest TearDwon" << endl;
  }
  
  virtual void SetUp() {
    cmo_type_inv_ = std::make_shared<GenerateCMOTypeInvalid>();
  }

  virtual void TearDown() {
  }
  
public:
  GenerateCMOTypeInvalidPtr cmo_type_inv_;
};

TEST_F(GenerateCmoTypeInvalidTest, GenerateInputNoParent) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("name", "type");
  vector<int64_t> data_dims={2};
  GeTensorDesc data_tensor_desc(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc("input1", data_tensor_desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc_ptr);
  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateInput(node, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateInputNoLifeCycleEnd) {
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
  cmo_type_inv_->GenerateInput(node2, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateInputOverReadDistance) {
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
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {5, 0});
  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateInput(node2, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateInputNoReuse) {
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
  AttrUtils::SetInt(node2->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {2, 0});
  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateInput(node2, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, cmo);
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateInputLackReuseDistance) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  OpDescPtr op_desc_ptr4 = make_shared<OpDesc>("name4", "type4");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc1);
  op_desc_ptr3->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr4->AddInputDesc("input1", data_tensor_desc1);
  
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);
  NodePtr node4 = graph->AddNode(op_desc_ptr4);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node2->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node3->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  graph->TopologicalSorting();
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {2, 0});
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info{};
  ge::MemReuseInfo reuse_info{node3, MemType::OUTPUT_MEM, 0};
  std::vector<ge::MemReuseInfo> reuse_info_vec{reuse_info};
  mem_reuse_info.emplace("output0", reuse_info_vec);
  node1->GetOpDesc()->SetExtAttr(ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);
  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateInput(node2, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateInputReuseMemNoTask) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  OpDescPtr op_desc_ptr4 = make_shared<OpDesc>("name4", "type4");
  OpDescPtr op_desc_ptr5 = make_shared<OpDesc>("name5", "type5");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc1);
  op_desc_ptr3->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr4->AddInputDesc("input1", data_tensor_desc1);
  op_desc_ptr4->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr5->AddInputDesc("input1", data_tensor_desc1);
  
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);
  NodePtr node4 = graph->AddNode(op_desc_ptr4);
  NodePtr node5 = graph->AddNode(op_desc_ptr5);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node4->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  GraphUtils::AddEdge(node4->GetOutDataAnchor(0), node5->GetInDataAnchor(0));
  graph->TopologicalSorting();
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {2, 0});
  AttrUtils::SetInt(node1->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 0);
  AttrUtils::SetInt(node2->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 1);
  AttrUtils::SetInt(node3->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 2);
  AttrUtils::SetInt(node4->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 3);
  AttrUtils::SetInt(node5->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 4);
  AttrUtils::SetBool(node5->GetOpDesc(), ge::ATTR_NAME_NOTASK, true);
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info{};
  ge::MemReuseInfo reuse_info{node4, MemType::OUTPUT_MEM, 0};
  std::vector<ge::MemReuseInfo> reuse_info_vec{reuse_info};
  mem_reuse_info.emplace("output0", reuse_info_vec);
  node1->GetOpDesc()->SetExtAttr(ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);
  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateInput(node2, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateInputReuseOK) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  OpDescPtr op_desc_ptr4 = make_shared<OpDesc>("name4", "type4");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc1);
  op_desc_ptr3->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr4->AddInputDesc("input1", data_tensor_desc1);
  
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);
  NodePtr node4 = graph->AddNode(op_desc_ptr4);
  AttrUtils::SetInt(node1->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node2->GetOpDesc(), FE_IMPLY_TYPE, 6);
  AttrUtils::SetInt(node4->GetOpDesc(), FE_IMPLY_TYPE, 6);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  AttrUtils::SetInt(node1->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 0);
  AttrUtils::SetInt(node2->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 1);
  AttrUtils::SetInt(node3->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 2);
  AttrUtils::SetInt(node4->GetOpDesc(), ATTR_NAME_OP_READ_WRITE_INDEX, 3);
  graph->TopologicalSorting();
  AttrUtils::SetListInt(node2->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_DATA_VISIT_DISTANCE, {2, 0});
  std::map<std::string, std::vector<ge::MemReuseInfo>> mem_reuse_info{};
  ge::MemReuseInfo reuse_info{node4, MemType::OUTPUT_MEM, 0};
  std::vector<ge::MemReuseInfo> reuse_info_vec{reuse_info};
  mem_reuse_info.emplace("output0", reuse_info_vec);
  node1->GetOpDesc()->SetExtAttr(ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);
  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateInput(node2, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node2->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);

  cmo = node4->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoBarrier].size(), 1);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateWorkSpaceNotAiCore) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  OpDescPtr op_desc_ptr4 = make_shared<OpDesc>("name4", "type4");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc1);
  op_desc_ptr3->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr4->AddInputDesc("input1", data_tensor_desc1);
  
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);
  NodePtr node4 = graph->AddNode(op_desc_ptr4);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
  graph->TopologicalSorting();

  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateWorkSpace(node1, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node1->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 0);
}

TEST_F(GenerateCmoTypeInvalidTest, GenerateWorkSpaceOK) {
  OpDescPtr op_desc_ptr1 = make_shared<OpDesc>("name1", "type1");
  OpDescPtr op_desc_ptr2 = make_shared<OpDesc>("name2", "type2");
  OpDescPtr op_desc_ptr3 = make_shared<OpDesc>("name3", "type3");
  OpDescPtr op_desc_ptr4 = make_shared<OpDesc>("name4", "type4");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc2, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr1->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr2->AddInputDesc("input1", data_tensor_desc2);
  op_desc_ptr2->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr3->AddInputDesc("input1", data_tensor_desc1);
  op_desc_ptr3->AddOutputDesc("output1", data_tensor_desc1);
  op_desc_ptr4->AddInputDesc("input1", data_tensor_desc1);
  
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node1 = graph->AddNode(op_desc_ptr1);
  NodePtr node2 = graph->AddNode(op_desc_ptr2);
  NodePtr node3 = graph->AddNode(op_desc_ptr3);
  NodePtr node4 = graph->AddNode(op_desc_ptr4);
  GraphUtils::AddEdge(node1->GetOutDataAnchor(0), node2->GetInDataAnchor(0));
  GraphUtils::AddEdge(node2->GetOutDataAnchor(0), node3->GetInDataAnchor(0));
  GraphUtils::AddEdge(node3->GetOutDataAnchor(0), node4->GetInDataAnchor(0));
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
  ge::MemReuseInfo reuse_info{node4, MemType::WORKSPACE_MEM, 0};
  std::vector<ge::MemReuseInfo> reuse_info_vec{reuse_info};
  mem_reuse_info.emplace("workspace0", reuse_info_vec);
  node1->GetOpDesc()->SetExtAttr(ATTR_NAME_MEMORY_REUSE_INFO, mem_reuse_info);
  StreamCtrlMap stream_ctls;
  cmo_type_inv_->GenerateWorkSpace(node1, stream_ctls);

  map<std::string, std::vector<CmoAttr>> cmo;
  cmo = node1->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoInvalid].size(), 1);

  cmo = node4->GetOpDesc()->TryGetExtAttr(kOpExtattrNameCmo, map<std::string, std::vector<CmoAttr>>{});
  EXPECT_EQ(cmo.size(), 1);
  EXPECT_EQ(cmo[kCmoBarrier].size(), 1);
}
