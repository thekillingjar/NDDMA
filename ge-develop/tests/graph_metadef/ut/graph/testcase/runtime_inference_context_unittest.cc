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

#include "graph/runtime_inference_context.h"

namespace ge {
class RuntimeInferenceContextTest : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


TEST_F(RuntimeInferenceContextTest, TestSetGetTensor) {
  RuntimeInferenceContext ctx;
  GeTensorDesc desc;
  GeTensorPtr ge_tensor = std::make_shared<GeTensor>(desc);
  ASSERT_EQ(ctx.SetTensor(1, 3, ge_tensor), GRAPH_SUCCESS);
  GeTensorPtr new_tensor;
  ASSERT_EQ(ctx.GetTensor(1, 3, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(2, 0, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(2, -1, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(1, 4, new_tensor), GRAPH_SUCCESS);
  ASSERT_NE(ctx.GetTensor(1, 0, new_tensor), GRAPH_SUCCESS);
  ctx.Release();
  ASSERT_NE(ctx.GetTensor(1, 3, new_tensor), GRAPH_SUCCESS);
}
} // namespace ge
