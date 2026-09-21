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

#include "register/scope/scope_fusion_pass_register.h"

using namespace std;
using namespace testing;

namespace ge {

class ScopeUtilUt : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(ScopeUtilUt, StringReplaceAll) {
  AscendString str = ScopeUtil::StringReplaceAll("123456", "123", "abc");
  EXPECT_EQ(str.GetString(), string("abc456"));

  string str2 = ScopeUtil::StringReplaceAll(string("abc456"), string("456"), string("def"));
  EXPECT_EQ(str2, string("abcdef"));
}

TEST_F(ScopeUtilUt, FreeScopePatterns) {
  std::vector<std::vector<ScopePattern *>> scoPatternSub;
  std::vector<ScopePattern *> scoPatternSubSub1;
  std::vector<ScopePattern *> scoPatternSubSub2;
  ScopePattern *scoPattern1;
  ScopePattern *scoPattern2;

  scoPattern1 = new ScopePattern();
  scoPattern2 = new ScopePattern();
  scoPatternSubSub1.push_back(scoPattern1);
  scoPatternSubSub2.push_back(scoPattern2);

  scoPatternSub.push_back(scoPatternSubSub1);
  scoPatternSub.push_back(scoPatternSubSub2);

  ScopeUtil::FreeScopePatterns(scoPatternSub);
  EXPECT_EQ(scoPatternSub.size(), 0);
}

}  // namespace ge
