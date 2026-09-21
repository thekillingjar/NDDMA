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
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>
#include <vector>

#define protected public
#define private public
#include "common/math_util.h"
#include "common/fe_utils.h"
#include "fusion_rule_manager/fusion_rule_data/fusion_rule_pattern.h"
#undef protected
#undef private

using namespace std;
using namespace fe;

class STEST_math_util_stest : public testing::Test
{
 protected:
  void SetUp()
  {
  }

  void TearDown()
  {
  }
};

TEST_F(STEST_math_util_stest, Uint16ToFloat)
{
    uint16_t number = 1;
    float res = Uint16ToFloat(number);
    EXPECT_NE(res, 4);
}

TEST_F(STEST_math_util_stest, fe_util_get_relu_type)
{
    RuleType ruleType = fe::RuleType::BUILT_IN_GRAPH_RULE;
    std::string res = GetRuleTypeString(ruleType);
    EXPECT_EQ(res, "built-in-graph-rule");
}

TEST_F(STEST_math_util_stest, fe_util_get_relu_type_unknown)
{
    RuleType ruleType = fe::RuleType::RULE_TYPE_RESERVED;
    std::string res = GetRuleTypeString(ruleType);
    EXPECT_EQ(res, "unknown-rule-type 2");
}
