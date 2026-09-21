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

namespace gert {
class GertTensorDataUT : public testing::Test {};

TEST_F(GertTensorDataUT, Init_success) {
  GertTensorData gert_tensor_data = GertTensorData();
  ASSERT_EQ(gert_tensor_data.GetSize(), 0U);
  ASSERT_EQ(gert_tensor_data.GetPlacement(), kTensorPlacementEnd);
  ASSERT_EQ(gert_tensor_data.GetStreamId(), -1);
}

TEST_F(GertTensorDataUT, SetSize_SetPlacement_SetAddr_success) {
  GertTensorData gert_tensor_data = GertTensorData();
  ASSERT_EQ(gert_tensor_data.GetSize(), 0U);
  ASSERT_EQ(gert_tensor_data.GetPlacement(), kTensorPlacementEnd);
  ASSERT_EQ(gert_tensor_data.GetStreamId(), -1);

  gert_tensor_data.SetSize(100U);
  ASSERT_EQ(gert_tensor_data.GetSize(), 100U);

  gert_tensor_data.SetPlacement(kOnDeviceHbm);
  ASSERT_EQ(gert_tensor_data.GetPlacement(), kOnDeviceHbm);

  gert_tensor_data.MutableTensorData().SetAddr((void *)1, nullptr);
  ASSERT_EQ(gert_tensor_data.GetAddr(), (void *)1);
}

TEST_F(GertTensorDataUT, TensorDataWithoutManager_After_Free_GetAddr_success) {
  GertTensorData gert_tensor_data = GertTensorData();
  void *addr = (void*)1;
  const_cast<TensorData *>(&gert_tensor_data.GetTensorData())->SetAddr(addr, nullptr);
  ASSERT_EQ(gert_tensor_data.FreeHoldAddr(), ge::GRAPH_SUCCESS);
  ASSERT_EQ(gert_tensor_data.GetAddr(), addr);
}
}  // namespace gert
