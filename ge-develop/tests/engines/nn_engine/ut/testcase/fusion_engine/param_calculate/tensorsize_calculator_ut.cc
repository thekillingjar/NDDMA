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
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#define protected public
#define private public
#include "param_calculate/tensorsize_calculator.h"
#undef protected
#undef private

using namespace std;
using namespace fe;
using namespace ge;

class TENSORSIZE_CALCULATOR_UTEST: public testing::Test {
 protected:
  void SetUp() {
  }

  void TearDown() {
  }
};

TEST_F(TENSORSIZE_CALCULATOR_UTEST, CalcSingleTensorSize_suc) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {-1, 1, 1, 1};
  GeShape shape(dim);
  GeTensorDesc in_desc(shape);
  in_desc.SetDataType(DT_FLOAT16);
  op_desc->AddInputDesc(in_desc);
  NodePtr node_0 = graph->AddNode(op_desc);
  ge::GeTensorDescPtr tensor_desc_ptr = op_desc->MutableInputDesc(0);
  string direction;
  size_t i = 0;
  bool output_real_calc_flag = false;
  int64_t tensor_size;

  TensorSizeCalculator tensor_size_calculator;
  Status ret = tensor_size_calculator.CalcSingleTensorSize(*op_desc, tensor_desc_ptr, direction, i,
                                                           output_real_calc_flag, tensor_size);
  EXPECT_EQ(ret, fe::SUCCESS);
}

TEST_F(TENSORSIZE_CALCULATOR_UTEST, CalcSingleTensorSize_failed) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {1, 1, 1, 1};
  GeShape shape(dim);
  GeTensorDesc in_desc(shape);
  in_desc.SetDataType(DT_MAX);
  op_desc->AddInputDesc(in_desc);
  NodePtr node_0 = graph->AddNode(op_desc);
  ge::GeTensorDescPtr tensor_desc_ptr = op_desc->MutableInputDesc(0);
  string direction;
  size_t i = 0;
  bool output_real_calc_flag = false;
  int64_t tensor_size;

  TensorSizeCalculator tensor_size_calculator;
  Status ret = tensor_size_calculator.CalcSingleTensorSize(*op_desc, tensor_desc_ptr, direction, i,
                                                           output_real_calc_flag, tensor_size);
  EXPECT_EQ(ret, fe::FAILED);
}

TEST_F(TENSORSIZE_CALCULATOR_UTEST, SpecialOutput) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_end = std::make_shared<OpDesc>("end", "End");
  vector<int64_t> dim = {100, 200, 300, 1};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetDataType(DT_FLOAT16);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc_end->AddInputDesc(tensor_desc);
  NodePtr node_0 = graph->AddNode(op_desc);
  NodePtr end_node_0 = graph->AddNode(op_desc_end);
  ge::GraphUtils::AddEdge(node_0->GetOutAnchor(0), end_node_0->GetInAnchor(0));
  ge::GeTensorDescPtr in_tensor = op_desc->MutableInputDesc(0);
  ge::GeTensorDescPtr out_tensor = op_desc->MutableOutputDesc(0);

  ge::AttrUtils::SetInt(in_tensor, ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, 5);
  ge::AttrUtils::SetInt(out_tensor, ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, 5);
  auto output_data_anchors = node_0->GetAllOutDataAnchors();
  int output_real_calc_flag = 0;
  Status ret = TensorSizeCalculator::CalcInputOpTensorSize(node_0, output_real_calc_flag);
  ret = TensorSizeCalculator::CalcOutputOpTensorSize(*op_desc, output_real_calc_flag, output_data_anchors);
  EXPECT_EQ(ret, fe::SUCCESS);

  int64_t in_tensor_size = 0;
  int64_t out_tensor_size = 0;

  ge::TensorUtils::GetSize(*in_tensor, in_tensor_size);
  EXPECT_EQ(in_tensor_size, 12000032);


  ge::TensorUtils::GetSize(*out_tensor, out_tensor_size);
  EXPECT_EQ(out_tensor_size, 64);
}

TEST_F(TENSORSIZE_CALCULATOR_UTEST, SpecialOutput_02) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("add", "Add");
  OpDescPtr op_desc_end = std::make_shared<OpDesc>("end", "End");
  vector<int64_t> dim = {100, 200, 300, 1};
  GeShape shape(dim);
  GeTensorDesc tensor_desc(shape);
  tensor_desc.SetDataType(DT_FLOAT16);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc_end->AddInputDesc(tensor_desc);
  NodePtr node_0 = graph->AddNode(op_desc);
  NodePtr end_node_0 = graph->AddNode(op_desc_end);
  ge::GraphUtils::AddEdge(node_0->GetOutAnchor(0), end_node_0->GetInAnchor(0));
  ge::GeTensorDescPtr in_tensor = op_desc->MutableInputDesc(0);
  ge::GeTensorDescPtr out_tensor = op_desc->MutableOutputDesc(0);

  ge::AttrUtils::SetInt(in_tensor, ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, 5);
  ge::AttrUtils::SetInt(out_tensor, ge::ATTR_NAME_SPECIAL_OUTPUT_SIZE, 5);
  auto output_data_anchors = node_0->GetAllOutDataAnchors();
  int output_real_calc_flag = 1;
  Status ret = TensorSizeCalculator::CalcInputOpTensorSize(node_0, output_real_calc_flag);
  ret = TensorSizeCalculator::CalcOutputOpTensorSize(*op_desc, output_real_calc_flag, output_data_anchors);
  EXPECT_EQ(ret, fe::SUCCESS);

  int64_t in_tensor_size = 0;
  int64_t out_tensor_size = 0;

  ge::TensorUtils::GetSize(*in_tensor, in_tensor_size);
  EXPECT_EQ(in_tensor_size, 12000000);


  ge::TensorUtils::GetSize(*out_tensor, out_tensor_size);
  EXPECT_EQ(out_tensor_size, 5);
}


TEST_F(TENSORSIZE_CALCULATOR_UTEST, test_complex_size) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {100, 200, 300, 1};
  GeShape shape(dim);
  GeTensorDesc tensor_desc_0(shape);
  GeTensorDesc tensor_desc_1(shape);
  tensor_desc_0.SetDataType(DT_COMPLEX32);
  tensor_desc_1.SetDataType(DT_COMPLEX64);
  op_desc->AddInputDesc(tensor_desc_0);
  op_desc->AddInputDesc(tensor_desc_1);
  op_desc->AddOutputDesc(tensor_desc_0);
  NodePtr node_0 = graph->AddNode(op_desc);
  ge::GeTensorDescPtr in_tensor_0 = op_desc->MutableInputDesc(0);
  ge::GeTensorDescPtr in_tensor_1 = op_desc->MutableInputDesc(1);
  ge::GeTensorDescPtr out_tensor = op_desc->MutableOutputDesc(0);
  int output_real_calc_flag = 1;
  auto output_data_anchors = node_0->GetAllOutDataAnchors();
  Status ret = TensorSizeCalculator::CalcInputOpTensorSize(node_0, output_real_calc_flag);
  ret = TensorSizeCalculator::CalcOutputOpTensorSize(*op_desc, output_real_calc_flag, output_data_anchors);
  EXPECT_EQ(ret, fe::SUCCESS);

  int64_t in_tensor0_size = 0;
  int64_t in_tensor1_size = 0;
  int64_t out_tensor_size = 0;

  ge::TensorUtils::GetSize(*in_tensor_0, in_tensor0_size);
  EXPECT_EQ(in_tensor0_size, 24000000);

  ge::TensorUtils::GetSize(*in_tensor_1, in_tensor1_size);
  EXPECT_EQ(in_tensor1_size, 48000000);

  ge::TensorUtils::GetSize(*out_tensor, out_tensor_size);
  EXPECT_EQ(out_tensor_size, 24000000);
}

TEST_F(TENSORSIZE_CALCULATOR_UTEST, test_fifo_size) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("add", "Add");
  vector<int64_t> dim = {100, 200, 300, 1};
  GeShape shape(dim);
  GeTensorDesc tensor_desc_0(shape);
  GeTensorDesc tensor_desc_1(shape);
  tensor_desc_0.SetDataType(DT_COMPLEX32);
  tensor_desc_1.SetDataType(DT_COMPLEX64);
  op_desc->AddInputDesc(tensor_desc_0);
  op_desc->AddInputDesc(tensor_desc_1);
  op_desc->AddOutputDesc(tensor_desc_0);
  NodePtr node_0 = graph->AddNode(op_desc);

  OpDescPtr op_desc_1 = std::make_shared<OpDesc>("fifo", "FIFO");
  op_desc_1->AddInputDesc(tensor_desc_0);
  op_desc_1->AddInputDesc(tensor_desc_1);
  op_desc_1->AddOutputDesc(tensor_desc_0);
  NodePtr node_1 = graph->AddNode(op_desc_1);
  ge::GeTensorDescPtr in_tensor_0 = op_desc_1->MutableInputDesc(0);
  ge::GeTensorDescPtr out_tensor = op_desc->MutableOutputDesc(0);

  int64_t tensor_size = 48000032;
  int64_t ddr_base = 2;
  ge::AttrUtils::SetInt(in_tensor_0, ge::ATTR_NAME_SPECIAL_INPUT_SIZE, tensor_size);
  ge::AttrUtils::SetInt(in_tensor_0, ge::ATTR_NAME_TENSOR_MEMORY_SCOPE, ddr_base);

  ge::GraphUtils::AddEdge(node_0->GetOutDataAnchor(0),
                          node_1->GetInDataAnchor(0));

  int output_real_calc_flag = 1;
  Status ret = TensorSizeCalculator::CalcInputOpTensorSize(node_1, output_real_calc_flag);
  EXPECT_EQ(ret, fe::SUCCESS);

  int64_t in_tensor0_size = 0;
  int64_t in_tensor1_size = 0;
  int64_t out_tensor_size = 0;

  ge::TensorUtils::GetSize(*in_tensor_0, in_tensor0_size);
  EXPECT_EQ(in_tensor0_size, 48000064);

  ge::TensorUtils::GetSize(*out_tensor, out_tensor_size);
  EXPECT_EQ(out_tensor_size, 48000064);
}
