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
#include "cmo/generate_cmo_type_base.h"
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
class GenerateCmoTypeBaseTest : public testing::Test{
protected:
  static void SetUpTestCase() {
    cout << "GenerateCmoTypeBaseTest SetUp" << endl;
  }

  static void TearDownTestCase() {
    cout << "GenerateCmoTypeBaseTest TearDwon" << endl;
  }
  
  virtual void SetUp() {
    cmo_type_base_ = std::make_shared<GenerateCMOTypeBase>();
  }

  virtual void TearDown() {
  }
  
public:
  GenerateCMOTypeBasePtr cmo_type_base_;
};

TEST_F(GenerateCmoTypeBaseTest, GetInputTensorSize) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Add", "Add");
  vector<int64_t> data_dims={1, 1, 1};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::TensorUtils::SetSize(data_tensor_desc1, 32);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::TensorUtils::SetSize(data_tensor_desc2, 64);
  op_desc_ptr->AddInputDesc("input1", data_tensor_desc1);
  op_desc_ptr->AddInputDesc("input2", data_tensor_desc2);

  int64_t input_size = cmo_type_base_->GetInputTensorSize(op_desc_ptr);
  EXPECT_EQ(input_size, 96);
}

TEST_F(GenerateCmoTypeBaseTest, GetOutputTensorSize) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Add", "Add");
  vector<int64_t> data_dims={1, 1, 1};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::TensorUtils::SetSize(data_tensor_desc1, 32);
  GeTensorDesc data_tensor_desc2(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::TensorUtils::SetSize(data_tensor_desc2, 32);
  op_desc_ptr->AddOutputDesc(data_tensor_desc1);
  op_desc_ptr->AddOutputDesc(data_tensor_desc2);

  int64_t output_size = cmo_type_base_->GetOutputTensorSize(op_desc_ptr);
  EXPECT_EQ(output_size, 64);
}

TEST_F(GenerateCmoTypeBaseTest, GetWorkSpaceSize) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("Add", "Add");
  vector<int64_t> workspaces = {32, 32, 32};
  op_desc_ptr->SetWorkspaceBytes(workspaces);

  int64_t space_size = cmo_type_base_->GetWorkSpaceSize(op_desc_ptr);
  EXPECT_EQ(space_size, 96);
}

TEST_F(GenerateCmoTypeBaseTest, GetWeightSize) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("name", "type");
  vector<int64_t> data_dims = {3};
  vector<uint8_t> dims_value_vec = {2, 4, 4};
  GeTensorDesc data_tensor_desc1(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  GeTensorPtr dim_tensor = std::make_shared<GeTensor>(data_tensor_desc1, dims_value_vec);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc_ptr);
  vector<ge::GeTensorPtr> weights{dim_tensor};
  ge::OpDescUtils::SetWeights(node, weights);
  ge::TensorUtils::SetSize(*node->GetOpDesc()->MutableInputDesc(0), 4096);
  int64_t weight_size = cmo_type_base_->GetWeightSize(node);
  EXPECT_EQ(weight_size, 4096);
}

TEST_F(GenerateCmoTypeBaseTest, ReadIsLifeCycleEndFalse) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("name", "type");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  op_desc_ptr->AddInputDesc("input1", data_tensor_desc);
  
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc_ptr);
  bool res = cmo_type_base_->ReadIsLifeCycleEnd(node, node->GetInDataAnchor(0));
  EXPECT_FALSE(res);
}

TEST_F(GenerateCmoTypeBaseTest, ReadIsLifeCycleEndTrue) {
  OpDescPtr op_desc_ptr = make_shared<OpDesc>("name", "type");
  vector<int64_t> data_dims = {2};
  GeTensorDesc data_tensor_desc(GeShape(data_dims), FORMAT_NCHW, DT_FLOAT);
  ge::AttrUtils::SetBool(&data_tensor_desc, ge::ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE, true);
  op_desc_ptr->AddInputDesc("input1", data_tensor_desc);

  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  NodePtr node = graph->AddNode(op_desc_ptr);
  bool res = cmo_type_base_->ReadIsLifeCycleEnd(node, node->GetInDataAnchor(0));
  EXPECT_TRUE(res);
}

TEST_F(GenerateCmoTypeBaseTest, CheckParentOpIsAiCore) {
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

  bool res = cmo_type_base_->CheckParentOpIsAiCore(node2->GetInDataAnchor(0));
  EXPECT_TRUE(res);
}
