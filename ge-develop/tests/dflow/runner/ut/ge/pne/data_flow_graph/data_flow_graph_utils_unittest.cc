/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "dflow/compiler/data_flow_graph/data_flow_graph_utils.h"

using namespace testing;
namespace ge {
class DataFlowGraphUtilsTest : public Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(DataFlowGraphUtilsTest, CreateFlowNodeOpDesc_SUCCESS) {
  OpDescPtr op_desc = nullptr;
  std::string op_name = "TimeBatch";
  uint32_t input_num = 2;
  uint32_t output_num = 2;
  EXPECT_EQ(DataFlowGraphUtils::CreateFlowNodeOpDesc(op_name, input_num, output_num, op_desc), SUCCESS);
}
}
