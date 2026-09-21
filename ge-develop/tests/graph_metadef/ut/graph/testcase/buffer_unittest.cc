/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include <gtest/gtest.h>
#include "graph/buffer.h"

namespace ge {
class BufferUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(BufferUT, ShareFrom1) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateShareFrom(buf); // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, ShareFrom2) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  BufferUtils::ShareFrom(buf, buf1); // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, OperatorAssign) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  buf1 = buf; // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateShareFrom) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 1024;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 1024;
  }
  second_buf[50] = 10;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateShareFrom(buf);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 10;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_NE(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateCopyFrom1) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 2;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 2;
  }
  second_buf[50] = 250;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1;
  BufferUtils::CopyFrom(buf, buf1);
  buf1.GetData()[50] = 250;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}

TEST_F(BufferUT, CreateCopyFrom2) {
  uint8_t first_buf[100];
  for (int i = 0; i < 100; ++i) {
    first_buf[i] = i * 2;
  }
  uint8_t second_buf[100];
  for (int i = 0; i < 100; ++i) {
    second_buf[i] = i * 2;
  }
  second_buf[50] = 250;

  Buffer buf(100);
  memcpy_s(buf.GetData(), buf.GetSize(), first_buf, sizeof(first_buf));
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);

  Buffer buf1 = BufferUtils::CreateCopyFrom(buf);  // The buf1 and buf are ref from the same memory now
  buf1.GetData()[50] = 250;
  EXPECT_EQ(memcmp(buf1.GetData(), second_buf, sizeof(second_buf)), 0);
  EXPECT_EQ(memcmp(buf.GetData(), first_buf, sizeof(first_buf)), 0);
}
}  // namespace ge
