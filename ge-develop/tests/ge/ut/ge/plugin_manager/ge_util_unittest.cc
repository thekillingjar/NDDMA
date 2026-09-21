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
#include "common/base64.h"
#include "graph/def_types.h"
#include "graph/utils/math_util.h"
#include "common/math/math_util.h"
#include "common/plugin/ge_make_unique_util.h"
#include "common/plugin/datatype_util.h"

using namespace ge;
using namespace std;

class UtestGeUtil : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class Foo {
 public:
  int i = 0;

  Foo(int x) { i = x; }

  GE_DELETE_ASSIGN_AND_COPY(Foo);
};

TEST_F(UtestGeUtil, delete_assign_and_copy) {
  Foo f(1);
  ASSERT_EQ(f.i, 1);
}

TEST_F(UtestGeUtil, make_shared) {
  auto f = MakeShared<Foo>(1);
  ASSERT_EQ(f->i, 1);
}

TEST_F(UtestGeUtil, make_unique) {
  auto f = MakeUnique<Foo>(1);
  ASSERT_EQ(f->i, 1);
  auto x = MakeUnique<int[]>(10);
  ASSERT_NE(x.get(), nullptr);
}

TEST_F(UtestGeUtil, base64_test) {
  ge::Status retStatus;
  std::string raw_data = "aa123";
  std::string ret = ge::base64::EncodeToBase64(raw_data);
  ASSERT_EQ(ret.empty(), false);

  std::string decode_data;
  retStatus = ge::base64::DecodeFromBase64(ret, decode_data);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeUtil, NnSet_test) {
  ge::Status retStatus;
  float output1;
  float alpha = 0;
  retStatus = ge::NnSet(1, alpha, &output1);
  EXPECT_EQ(retStatus, SUCCESS);

  float *output2 = (float *)malloc(sizeof(float)*SECUREC_MEM_MAX_LEN);
  ASSERT_NE(output2, nullptr);
  retStatus = ge::NnSet(SECUREC_MEM_MAX_LEN, alpha, output2);
  EXPECT_EQ(retStatus, SUCCESS);
  free(output2);
}

TEST_F(UtestGeUtil, math_utils_test)
{
  Status retStatus;

  retStatus = ge::CheckInt8AddOverflow(INT8_MAX, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt16AddOverflow(INT16_MAX, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt32AddOverflow(INT32_MAX, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt64AddOverflow(std::numeric_limits<int64_t>::max(), 64);
  EXPECT_EQ(retStatus, FAILED);

  retStatus = ge::CheckUint8AddOverflow(UINT8_MAX, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint16AddOverflow(UINT16_MAX, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint64AddOverflow(UINT64_MAX, 64);
  EXPECT_EQ(retStatus, FAILED);

  fp16_t a, b;
  a.val = kFp16ExpMask;
  b.val = 0;
  retStatus = ge::CheckFp16AddOverflow(a, b);
  EXPECT_EQ(retStatus, FAILED);

  retStatus = ge::CheckIntSubOverflow(INT_MIN, 1);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt8SubOverflow(INT8_MIN, 1);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt16SubOverflow(INT16_MIN, 1);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt32SubOverflow(INT32_MIN, 1);
  EXPECT_EQ(retStatus, FAILED);
  //retStatus = ge::CheckInt64SubOverflow((std::numeric_limits<int64_t>::max()+6), -1);
  //EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckUint8SubOverflow(0, 1);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint16SubOverflow(0, 1);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint32SubOverflow(0, 1);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint64SubOverflow(0, 1);
  EXPECT_EQ(retStatus, FAILED);

  retStatus = ge::CheckFp16SubOverflow(a, b);
  EXPECT_EQ(retStatus, FAILED);
  a.val -= 2;
  retStatus = ge::CheckFp16SubOverflow(a, b);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckIntMulOverflow(64, INT_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckIntMulOverflow(64, 64);
  EXPECT_EQ(retStatus, SUCCESS);
  retStatus = ge::CheckIntMulOverflow(INT_MIN, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckIntMulOverflow(-64, 64);
  EXPECT_EQ(retStatus, SUCCESS);
  retStatus = ge::CheckIntMulOverflow(-64, INT_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckIntMulOverflow(0, -64);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckInt8MulOverflow(64, INT8_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt8MulOverflow(2, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt8MulOverflow(INT8_MIN, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt8MulOverflow(-1, 8);
  EXPECT_EQ(retStatus, SUCCESS);
  retStatus = ge::CheckInt8MulOverflow(-64, INT8_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt8MulOverflow(0, -64);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckInt16MulOverflow(64, INT16_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt16MulOverflow(2, INT16_MAX);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt16MulOverflow(INT16_MIN, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt16MulOverflow(-64, 64);
  EXPECT_EQ(retStatus, SUCCESS);
  retStatus = ge::CheckInt16MulOverflow(-64, INT16_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt16MulOverflow(0, -64);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckInt32MulOverflow(64, INT32_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt32MulOverflow(2, INT32_MAX);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt32MulOverflow(INT32_MIN, 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt32MulOverflow(-64, 64);
  EXPECT_EQ(retStatus, SUCCESS);
  retStatus = ge::CheckInt32MulOverflow(-64, INT32_MIN);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt32MulOverflow(0, -64);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckInt64Int32MulOverflow(std::numeric_limits<int64_t>::min(), 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt64Int32MulOverflow(-64, 64);
  EXPECT_EQ(retStatus, SUCCESS);
  //retStatus = ge::CheckInt64Int32MulOverflow(-64, std::numeric_limits<int64_t>::min()+1);
  //EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt64Int32MulOverflow(0, -64);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckInt64MulOverflow(2, std::numeric_limits<int64_t>::max());
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt64MulOverflow(std::numeric_limits<int64_t>::min(), 64);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt64MulOverflow(-64, 64);
  EXPECT_EQ(retStatus, SUCCESS);
  retStatus = ge::CheckInt64MulOverflow(-64, std::numeric_limits<int64_t>::min());
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckInt64MulOverflow(0, -64);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckInt64Uint32MulOverflow(std::numeric_limits<int64_t>::min()+1, 4);
  EXPECT_EQ(retStatus, FAILED);

  retStatus = ge::CheckUint8MulOverflow(UINT8_MAX, 2);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint8MulOverflow(0, 0);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckUint16MulOverflow(UINT16_MAX, 2);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint16MulOverflow(0, 0);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckUint32MulOverflow(UINT32_MAX, 2);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint32MulOverflow(0, 0);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge::CheckUint64MulOverflow(UINT64_MAX, 2);
  EXPECT_EQ(retStatus, FAILED);
  retStatus = ge::CheckUint64MulOverflow(0, 0);
  EXPECT_EQ(retStatus, SUCCESS);

  fp16_t c, d;
  c.val = kFp16ExpMask/2;
  d.val = 2;
  //retStatus = ge::CheckFp16MulOverflow(c, d);
  //EXPECT_EQ(retStatus, FAILED);
  c.val = 6;
  retStatus = ge::CheckFp16MulOverflow(c, d);
  EXPECT_EQ(retStatus, SUCCESS);

}
