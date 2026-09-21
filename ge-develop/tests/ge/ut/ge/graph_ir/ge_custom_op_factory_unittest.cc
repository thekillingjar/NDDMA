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

#include "../graph/custom_ops_stub.h"
#include "graph/custom_op_factory.h"
#include "ge/ge_api_error_codes.h"
#include "macro_utils/dt_public_scope.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
class UtestCustomOpFactory : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST(UtestCustomOpFactory, create_custom_op) {
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyCustomOp"));
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyCustomOp2"));
  const auto op = CustomOpFactory::CreateCustomOp("MyCustomOp");
  const auto op2 = CustomOpFactory::CreateCustomOp("NonExists");
  EXPECT_EQ(true, op != nullptr);
  EXPECT_EQ(true, op2 == nullptr);
}

TEST(UtestCustomOpFactory, get_all_ops) {
  std::vector<AscendString> all_registered_ops;
  const auto ret = CustomOpFactory::GetAllRegisteredOps(all_registered_ops);
  EXPECT_EQ(ge::SUCCESS, ret);
  EXPECT_EQ(2, all_registered_ops.size());
}

TEST(UtestCustomOpFactory, register_custom_op_creator) {
  CustomOpFactory::RegisterCustomOpCreator(
      "MyCustomOp3", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MyCustomOp>(); });
  CustomOpFactory::RegisterCustomOpCreator("MyCustomOp4", nullptr);
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyCustomOp3"));
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyCustomOp4"));
}

TEST(UtestCustomOpFactory, creator_register) {
  CustomOpCreatorRegister(
      "MyCustomOp5", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MyCustomOp>(); });
  CustomOpCreatorRegister(
    "MyCustomOp5", []() -> std::unique_ptr<BaseCustomOp> { return std::make_unique<MyCustomOp>(); });
  EXPECT_EQ(true, CustomOpFactory::IsExistOp("MyCustomOp5"));
}