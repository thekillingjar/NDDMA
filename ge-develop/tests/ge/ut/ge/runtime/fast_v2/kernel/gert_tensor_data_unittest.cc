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
#include "exe_graph/runtime/gert_tensor_data.h"
#include "faker/multi_stream_allocator_faker.h"

namespace gert {
class GertTensorDataUT : public testing::Test {};
TEST_F(GertTensorDataUT, WanderFrom_Success) {
  auto holder = memory::MultiStreamAllocatorFaker().StreamNum(2).Build();
  auto gtd1 = holder.at(0)->MallocTensorData(1024U);
  ASSERT_NE(gtd1.GetAddr(), nullptr);
  GertTensorData gtd2;
  ASSERT_EQ(gtd2.WanderFrom(gtd1, 1), ge::GRAPH_SUCCESS);
  ASSERT_EQ(gtd2.GetAddr(), gtd1.GetAddr());
  ASSERT_EQ(gtd2.GetStreamId(), 1);
}

TEST_F(GertTensorDataUT, WanderFrom_Invalid_Stream_Id_Failed) {
  auto holder = memory::MultiStreamAllocatorFaker().StreamNum(2).Build();
  auto gtd1 = holder.at(0)->MallocTensorData(1024U);
  ASSERT_NE(gtd1.GetAddr(), nullptr);
  GertTensorData gtd2;
  int64_t invalid_stream_id = 2;
  ASSERT_NE(gtd2.WanderFrom(gtd1, invalid_stream_id), ge::GRAPH_SUCCESS);
}
}  // namespace gert