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
#include <iostream>

#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

using namespace std;
using namespace ge;

class UtestGeTestDefType : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestGeTestDefType, quant) {
  OpDescPtr desc_ptr1 = std::make_shared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr1->AddInputDesc("x", GeTensorDesc(GeShape({1, 16, 16, 16}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddInputDesc("w", GeTensorDesc(GeShape({1, 1, 1, 1}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(desc_ptr1->AddOutputDesc("y", GeTensorDesc(GeShape({1, 32, 8, 8}), FORMAT_NCHW)), GRAPH_SUCCESS);

  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(desc_ptr1), false);
  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(*desc_ptr1), false);
}
