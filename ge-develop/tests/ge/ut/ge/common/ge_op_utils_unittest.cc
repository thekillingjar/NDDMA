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
#include <vector>
#include "graph/utils/graph_utils.h"
#include "graph/ge_tensor.h"
#include "graph/op_desc.h"
#include "mmpa/mmpa_api.h"
#include "framework/common/helper/model_helper.h"
#include "common/op/ge_op_utils.h"
#include "ge_graph_dsl/graph_dsl.h"

using namespace testing;

namespace ge {
class UtestGeOpUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGeOpUtils, GetConstantStrMemSize_Success) {
  OpDescPtr const_op_desc = std::make_shared<OpDesc>("const", CONSTANT);
  GeShape shape;
  GeTensorDesc tensor_desc(shape, FORMAT_ND, DT_STRING);
  const_op_desc->AddOutputDesc("x", tensor_desc);
  std::vector<uint8_t> data(4, 1);
  ConstGeTensorPtr value = MakeShared<const GeTensor>(tensor_desc, data);
  AttrUtils::SetTensor(const_op_desc, ATTR_NAME_WEIGHTS, value);
  int64_t mem_size = 0;
  ASSERT_EQ(OpUtils::GetConstantStrMemSize(const_op_desc, mem_size), SUCCESS);
  EXPECT_EQ(mem_size, 4);
}
}
