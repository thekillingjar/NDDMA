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
#include "nlohmann/json.hpp"
#include "register/tuning_tiling_registry.h"

namespace tuningtiling {
BEGIN_TUNING_TILING_DEF(TestMatmul)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, batchdim);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(TestMatmul, FIELD(TestMatmul, batchdim));

BEGIN_TUNING_TILING_DEF(TestDynamic)
TUNING_TILING_DATA_FIELD_DEF(uint32_t, scheduleId);
TUNING_TILING_DATA_FIELD_DEF(TestMatmul, mmtiling);
END_TUNING_TILING_DEF

DECLARE_SCHEMA(TestDynamic, FIELD(TestDynamic, scheduleId), FIELD(TestDynamic, mmtiling));

REGISTER_TUNING_TILING_CLASS(DynamicRnn, TestDynamic);

class RegisterTuningTilingUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(RegisterTuningTilingUT, from_json_ut) {
  TestDynamic testdyn;
  TestMatmul mm;
  mm.batchdim = 1;
  testdyn.scheduleId = 10;
  testdyn.mmtiling = mm;
  nlohmann::json jsonval;
  testdyn.ToJson(jsonval);
  std::cout << "ori json:" << jsonval.dump() << std::endl;
  TuningTilingDefPtr tuingdef = TuningTilingClassFactory::CreateTilingDataInstance(ge::AscendString("unknow"));
  EXPECT_EQ(tuingdef == nullptr, true);
  tuingdef = TuningTilingClassFactory::CreateTilingDataInstance(ge::AscendString("DynamicRnn"));
  EXPECT_EQ(tuingdef != nullptr, true);
  auto struct_name = tuingdef->GetClassName();
  EXPECT_EQ(strcmp(struct_name.GetString(), "TestDynamic"), 0);
  auto fields = tuingdef->GetItemInfo();
  EXPECT_EQ(fields.size(), 2);
  std::cout << struct_name.GetString() << std::endl;
  tuingdef->FromJson(jsonval);
  nlohmann::json res;
  tuingdef->ToJson(res);
  EXPECT_EQ(res, jsonval);
  std::cout << "expected json:" << res.dump() << std::endl;
}
}  // namespace tuningtiling
