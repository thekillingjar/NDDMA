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
#include <string>
#include "register/op_check_register.h"

bool testFunc(const ge::Operator &op, ge::AscendString &result) {
  return true;
}

namespace {

class OpCheckAPIUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpCheckAPIUT, APITest) {
  ge::AscendString example("test");
  optiling::GEN_SIMPLIFIEDKEY_FUNC pFunc;
  pFunc = testFunc;
  optiling::OpCheckFuncRegistry::RegisterGenSimplifiedKeyFunc(example, pFunc);

  ge::AscendString errName("notExisted");
  auto nullFunc = optiling::OpCheckFuncRegistry::GetGenSimplifiedKeyFun(errName);
  EXPECT_EQ(nullFunc, nullptr);

  auto func = optiling::OpCheckFuncRegistry::GetGenSimplifiedKeyFun(example);
  EXPECT_NE(func, nullptr);
}

}  // namespace
