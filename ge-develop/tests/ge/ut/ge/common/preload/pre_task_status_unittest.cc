/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/pre_task_status.h"
#include <gtest/gtest.h>

namespace ge {
class PreTaskStatusUnittest : public testing::Test {};

TEST_F(PreTaskStatusUnittest, CreateMessageNullPtr) {
  va_list arg;
  EXPECT_EQ(CreateMessage(nullptr, arg), nullptr);
}

TEST_F(PreTaskStatusUnittest, CreateErrorStatusOk) {
  auto status = PreTaskStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  ASSERT_FALSE(status.IsSuccess());
  EXPECT_EQ(strcmp(status.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}

TEST_F(PreTaskStatusUnittest, CreateSuccessStatusOk) {
  auto status = PreTaskStatus::Success();
  EXPECT_TRUE(status.IsSuccess());
}

TEST_F(PreTaskStatusUnittest, CopyAssignOk1) {
  auto error = PreTaskStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = PreTaskStatus::Success();
  success = error;
  EXPECT_FALSE(success.IsSuccess());
  EXPECT_FALSE(error.IsSuccess());
  EXPECT_EQ(strcmp(success.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
  EXPECT_NE(success.GetErrorMessage(), error.GetErrorMessage());
}

TEST_F(PreTaskStatusUnittest, CopyAssginOk2) {
  auto error = PreTaskStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = PreTaskStatus::Success();
  error = success;
  EXPECT_TRUE(success.IsSuccess());
  EXPECT_TRUE(error.IsSuccess());
}

TEST_F(PreTaskStatusUnittest, CopyConstructOk) {
  auto error = PreTaskStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = PreTaskStatus::Success();
  PreTaskStatus e1(error);
  PreTaskStatus s1(success);
  EXPECT_FALSE(e1.IsSuccess());
  EXPECT_TRUE(s1.IsSuccess());
  EXPECT_EQ(strcmp(e1.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}

TEST_F(PreTaskStatusUnittest, MoveConstructOk) {
  auto error = PreTaskStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = PreTaskStatus::Success();
  PreTaskStatus e1(std::move(error));
  PreTaskStatus s1(std::move(success));
  EXPECT_FALSE(e1.IsSuccess());
  EXPECT_TRUE(s1.IsSuccess());
  EXPECT_EQ(strcmp(e1.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}

TEST_F(PreTaskStatusUnittest, MoveAssginOk1) {
  auto error = PreTaskStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = PreTaskStatus::Success();
  error = std::move(success);
  EXPECT_TRUE(error.IsSuccess());
}

TEST_F(PreTaskStatusUnittest, MoveAssginOk2) {
  auto error = PreTaskStatus::ErrorStatus("This is a error message %s, int %d", "Hello", 10);
  auto success = PreTaskStatus::Success();
  success = std::move(error);
  ASSERT_FALSE(success.IsSuccess());
  EXPECT_EQ(strcmp(success.GetErrorMessage(), "This is a error message Hello, int 10"), 0);
}
}
