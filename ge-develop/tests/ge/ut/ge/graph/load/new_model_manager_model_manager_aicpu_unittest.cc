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
#include "common/types.h"

#include "macro_utils/dt_public_scope.h"
#include "common/op/ge_op_utils.h"
#include "graph/load/model_manager/model_manager.h"

using namespace std;
using namespace testing;

namespace ge {

const static std::string ENC_KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

class UtestModelManagerModelManagerAicpu : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestModelManagerModelManagerAicpu, checkAicpuOptype) {
  ModelManager model_manager;
  std::vector<std::string> aicpu_op_list;
  std::vector<std::string> aicpu_tf_list;
  aicpu_tf_list.emplace_back("FrameworkOp");
  aicpu_tf_list.emplace_back("Unique");

  EXPECT_EQ(model_manager.LaunchKernelCheckAicpuOp(aicpu_op_list, aicpu_tf_list), SUCCESS);
  // Load allow listener is null
  // EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, nullptr, nullptr));
}

TEST_F(UtestModelManagerModelManagerAicpu, DestroyAicpuKernel) {
  ModelManager model_manager;
  std::vector<std::string> aicpu_op_list;
  std::vector<std::string> aicpu_tf_list;
  aicpu_tf_list.emplace_back("FrameworkOp");
  aicpu_tf_list.emplace_back("Unique");

  EXPECT_EQ(model_manager.DestroyAicpuKernel(0,0,0), SUCCESS);
  // Load allow listener is null
  // EXPECT_EQ(ge::FAILED, mm.LoadModelOffline(model_id, data, nullptr, nullptr));
}
}  // namespace ge
