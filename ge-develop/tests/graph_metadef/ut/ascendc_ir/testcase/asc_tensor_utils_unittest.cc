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
#include "graph/ascendc_ir/utils/asc_tensor_utils.h"
#include "testcase/ascendc_ir_dump_test/stub_graph.h"
#include <iostream>
using namespace ge;
class UtestAscirTensorUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};
REG_OP(Constant)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Constant);

REG_OP(Abs)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Abs);

TEST_F(UtestAscirTensorUtils, IsConstTensorTrue) {
  AscGraph g("graph");
  auto op = ascir_op::Constant("abc");
  auto node = g.AddNode(op);
  node->inputs();
  node->outputs();
  EXPECT_EQ(ge::ascir::AscTensorUtils::IsConstTensor(node->outputs[0]), true);
}

TEST_F(UtestAscirTensorUtils, IsConstTensorFalse) {
  AscGraph g("graph");
  auto op = ascir_op::Abs("abs");
  auto node = g.AddNode(op);
  node->inputs();
  node->outputs();
  EXPECT_EQ(ge::ascir::AscTensorUtils::IsConstTensor(node->outputs[0]), false);
}
