/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_ext_calc_param_registry.h"
#include "proto/task.pb.h"
#include <gtest/gtest.h>

class OpExtCalcParamRegistryUnittest : public testing::Test {};

namespace TestOpExtCalcParamRegistry {
ge::Status OpExtCalcParam(const ge::Node &node) {
  return 0;
}

TEST_F(OpExtCalcParamRegistryUnittest, OpExtCalcParamRegisterSuccess_Test) {
  EXPECT_EQ(fe::OpExtCalcParamRegistry::GetInstance().FindRegisterFunc("test"), nullptr);
  REGISTER_NODE_EXT_CALC_PARAM("test", OpExtCalcParam);
  EXPECT_EQ(fe::OpExtCalcParamRegistry::GetInstance().FindRegisterFunc("test"), OpExtCalcParam);
}
}
