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
#include "common/memory/external_weight_desc_impl.h"

namespace ge {
class UtestGraphCompileSummaryImpl: public testing::Test {
protected:
  void SetUp() {}
  void TearDown() {}
};

TEST(UtestGraphCompileSummaryImpl, BasicGetters) {
  AscendString location("weight_location");
  AscendString id("weight_id");
  size_t size = 1;
  size_t offset = 1;

  ExternalWeightDescPtr desc = ExternalWeightDesc::Builder::Build(location, size, offset, id);

  ASSERT_EQ(desc->GetLocation(), location);
  ASSERT_EQ(desc->GetSize(), size);
  ASSERT_EQ(desc->GetOffset(), offset);
  ASSERT_EQ(desc->GetId(), id);
}

TEST(UtestGraphCompileSummaryImpl, EmptyValues) {
  AscendString empty_location("");
  AscendString empty_id("");

  ExternalWeightDescPtr desc = ExternalWeightDesc::Builder::Build(
      empty_location, 0, 0, empty_id);

  ASSERT_EQ(desc->GetSize(), 0);
  ASSERT_EQ(desc->GetOffset(), 0);
}

TEST(UtestGraphCompileSummaryImpl, BuilderNullCheck) {
  AscendString location("test");
  ExternalWeightDescPtr desc = ExternalWeightDesc::Builder::Build(
      location, 100, 0, "test_id");

  ASSERT_TRUE(desc != nullptr);
  ASSERT_TRUE(desc->data_ != nullptr);
}

TEST(UtestGraphCompileSummaryImpl, DataIntegrity) {
  ExternalWeightDesc::ExternalWeightDescData data;
  AscendString location("/location");
  AscendString id("weight_id");

  data.SetLocationSizeOffsetId(location, 1, 0, id);

  ASSERT_EQ(data.GetLocation(), location);
  ASSERT_EQ(data.GetSize(), 1);
  ASSERT_EQ(data.GetOffset(), 0);
  ASSERT_EQ(data.GetId(), id);
}
} // namespace ge