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
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include "base/att_const_values.h"
#include "util/thread_local_context.h"
namespace att
{
    class ThreadLocalContextUnitTest: public testing::Test {
    public:
        // 前处理：创建一个测试用的空文件
        void SetUp() override {
        }
        // 后处理：删除测试文件
        void TearDown() override {
          GetThreadLocalContext().SetOption({});
        }
    };
TEST_F(ThreadLocalContextUnitTest, SetAndGetOptionSuccess) {
  constexpr char current_dir[] = "./";
  std::string got_option;
  EXPECT_TRUE(GetThreadLocalContext().GetOption(kDumpDebugInfo, got_option) == ge::FAILED);
  ASSERT_TRUE(got_option.empty());
  GetThreadLocalContext().SetOption({{kDumpDebugInfo, current_dir}});
  EXPECT_TRUE(GetThreadLocalContext().GetOption(kDumpDebugInfo, got_option) == ge::SUCCESS);
  EXPECT_TRUE(got_option == current_dir);
}

} //namespace