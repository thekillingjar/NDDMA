/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/runtime/continuous_vector.h"
#include <gtest/gtest.h>
namespace gert {
class ContinuousVectorUT : public testing::Test {};
TEST_F(ContinuousVectorUT, CreateOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(16);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 16);
}
TEST_F(ContinuousVectorUT, CreateNone) {
  auto vec_holder = ContinuousVector::Create<size_t>(0);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 0);
}
TEST_F(ContinuousVectorUT, WriteOk) {
  auto vec_holder = ContinuousVector::Create<size_t>(2);
  auto vec = reinterpret_cast<ContinuousVector *>(vec_holder.get());
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->GetSize(), 0);
  EXPECT_EQ(vec->GetCapacity(), 2);

  EXPECT_EQ(vec->SetSize(2), ge::GRAPH_SUCCESS);
  auto data = reinterpret_cast<size_t *>(vec->MutableData());
  data[0] = 1024;
  data[1] = 2048;
  EXPECT_EQ(vec->GetSize(), 2);
  EXPECT_EQ(reinterpret_cast<const size_t *>(vec->GetData())[0], 1024);
  EXPECT_EQ(reinterpret_cast<const size_t *>(vec->GetData())[1], 2048);
}
}  // namespace gert