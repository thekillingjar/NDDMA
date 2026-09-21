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
#include <gmock/gmock.h>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "engines/local_engine/ops_kernel_store/op/op_factory.h"
#include "engines/local_engine/ops_kernel_store/op/no_op.h"
#include "engines/local_engine/ops_kernel_store/ge_local_ops_kernel_builder.h"
#include "macro_utils/dt_public_unscope.h"

using namespace std;
using namespace testing;

namespace ge {
namespace ge_local{

class UtestOpFactory : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

std::shared_ptr<Op> testFunc(const Node &node, RunContext &runcontext)
{
  auto op = std::make_shared<NoOp>(node, runcontext);
  return op;
}

TEST_F(UtestOpFactory, Abnormal) {
  auto p = std::make_shared<OpFactory>();
  const OP_CREATOR_FUNC func1 = nullptr;
  const std::string type1 = "test";
  EXPECT_NO_THROW(p->RegisterCreator(type1, func1));
  OP_CREATOR_FUNC func2 = testFunc;
  const std::string type2 = "Reshape";
  EXPECT_NO_THROW(p->RegisterCreator(type2, func2));
  EXPECT_NO_THROW(p->RegisterCreator(type2, func2));
}

}
} // namespace ge
