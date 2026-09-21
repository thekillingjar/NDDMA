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

#include "common/op/transop_util.h"

#include "common/debug/log.h"
#include "common/types.h"
#include "common/util.h"
#include "compute_graph.h"
#include "graph/common/trans_op_creator.h"

using namespace ge;

class UtestTransopUtil : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestTransopUtil, test_is_transop_true) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("Cast", CAST);
  NodePtr node = graph->AddNode(op_desc);

  bool ret = TransOpUtil::IsTransOp(node);
  EXPECT_TRUE(ret);
}

TEST_F(UtestTransopUtil, test_is_transop_fail) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = std::make_shared<OpDesc>("relu", RELU);
  NodePtr node = graph->AddNode(op_desc);

  bool ret = TransOpUtil::IsTransOp(node);
  EXPECT_FALSE(ret);
}

TEST_F(UtestTransopUtil, test_get_transop_get_index) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr transdata_op_desc = std::make_shared<OpDesc>("Transdata", TRANSDATA);
  OpDescPtr transpose_op_desc = std::make_shared<OpDesc>("Transpose", TRANSPOSE);
  OpDescPtr reshape_op_desc = std::make_shared<OpDesc>("Reshape", RESHAPE);
  OpDescPtr cast_op_desc = std::make_shared<OpDesc>("Cast", CAST);

  NodePtr transdata_node = graph->AddNode(transdata_op_desc);
  NodePtr transpose_node = graph->AddNode(transpose_op_desc);
  NodePtr reshape_node = graph->AddNode(reshape_op_desc);
  NodePtr cast_node = graph->AddNode(cast_op_desc);

  int index1 = TransOpUtil::GetTransOpDataIndex(transdata_node);
  int index2 = TransOpUtil::GetTransOpDataIndex(transpose_node);
  int index3 = TransOpUtil::GetTransOpDataIndex(reshape_node);
  int index4 = TransOpUtil::GetTransOpDataIndex(cast_node);

  EXPECT_EQ(index1, 0);
  EXPECT_EQ(index2, 0);
  EXPECT_EQ(index3, 0);
  EXPECT_EQ(index4, 0);
}
TEST_F(UtestTransopUtil, test_precision_loss_dst_bool) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  GeTensorDesc input_desc({}, FORMAT_ND, DT_FLOAT);
  GeTensorDesc output_desc({}, FORMAT_ND, DT_BOOL);
  auto op_dsec = TransOpCreator::CreateCastOp("cast", input_desc, output_desc,false);
  NodePtr cast_node = graph->AddNode(op_dsec);
  EXPECT_TRUE(TransOpUtil::IsPrecisionLoss(cast_node));
}
TEST_F(UtestTransopUtil, test_precision_loss_int32_2_float) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");

  GeTensorDesc input_desc({}, FORMAT_ND, DT_INT32);
  GeTensorDesc output_desc({}, FORMAT_ND, DT_FLOAT);
  auto op_dsec = TransOpCreator::CreateCastOp("cast", input_desc, output_desc, false);
  NodePtr cast_node = graph->AddNode(op_dsec);
  EXPECT_FALSE(TransOpUtil::IsPrecisionLoss(cast_node));
}
TEST_F(UtestTransopUtil, test_precision_loss_floating_2_int) {
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
   for (auto f_type : {DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_BF16}) {
    for (auto i_type : {DT_INT32, DT_INT64}) {
       GeTensorDesc input_desc({}, FORMAT_ND, f_type);
       GeTensorDesc output_desc({}, FORMAT_ND, i_type);
       auto op_dsec = TransOpCreator::CreateCastOp("cast", input_desc, output_desc, false);
       NodePtr cast_node = graph->AddNode(op_dsec);
       EXPECT_TRUE(TransOpUtil::IsPrecisionLoss(cast_node));
    }
  }
}
