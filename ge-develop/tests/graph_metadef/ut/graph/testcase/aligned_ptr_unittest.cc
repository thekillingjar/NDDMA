/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/util/mem_utils.h"
#include <gtest/gtest.h>
#include <malloc.h>
#include <memory>
#include <stdio.h>

#include "graph_metadef/graph/aligned_ptr.h"
#include "graph_metadef/graph/aligned_ptr.h"

namespace ge {
class UtestAlignedPtr : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestAlignedPtr, Reset_success) {
  auto aligned_ptr = MakeShared<AlignedPtr>(6);
  uint64_t aligned_data = reinterpret_cast<uintptr_t>(aligned_ptr->Get());
  auto output = aligned_ptr->Reset();
  uint64_t output_data = reinterpret_cast<uintptr_t>(output.get());
  uint64_t output_base = reinterpret_cast<uintptr_t>(aligned_ptr->base_.get());
  ASSERT_EQ(aligned_data, output_data);
  ASSERT_EQ(output_base, 0);

  auto reset_null = aligned_ptr->Reset(nullptr, nullptr);
  ASSERT_EQ(reset_null, nullptr);
  ASSERT_EQ(aligned_data, output_data);

  uint8_t *data_ptr = new uint8_t(10);
  auto deleter = [](uint8_t *ptr) {
    delete ptr;
    ptr = nullptr;
  };
  aligned_ptr->Reset(data_ptr, deleter);
  ASSERT_EQ(aligned_ptr->Get(), data_ptr);
}

TEST_F(UtestAlignedPtr, BuildFromData_success) {
  auto deleter = [](uint8_t *ptr) {
    delete ptr;
    ptr = nullptr;
  };
  uint8_t *data_ptr = new uint8_t(10);
  auto aligned_ptr = AlignedPtr::BuildFromData(data_ptr, deleter);
  uint8_t result = *(aligned_ptr->Get());
  ASSERT_EQ(result, 10);
}
}  // namespace ge
