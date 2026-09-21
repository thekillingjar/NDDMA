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
#include "register/op_tiling_attr_utils.h"

using namespace std;
using namespace ge;

namespace optiling {

const uint16_t kFp16ExpBias = 15;
const uint32_t kFp32ExpBias = 127;
const uint16_t kFp16ManLen = 10;
const uint32_t kFp32ManLen = 23;
const uint32_t kFp32SignIndex = 31;
const uint16_t kFp16ManMask = 0x03FF;
const uint16_t kFp16ManHideBit = 0x0400;
const uint16_t kFp16MaxExp = 0x001F;
const uint32_t kFp32MaxMan = 0x7FFFFF;

float Uint16ToFloat(const uint16_t &intVal) {
  float ret;

  uint16_t hfSign = (intVal >> 15) & 1;
  int16_t hfExp = (intVal >> kFp16ManLen) & kFp16MaxExp;
  uint16_t hfMan = ((intVal >> 0) & 0x3FF) | ((((intVal >> 10) & 0x1F) > 0 ? 1 : 0) * 0x400);
  if (hfExp == 0) {
    hfExp = 1;
  }

  while (hfMan && !(hfMan & kFp16ManHideBit)) {
    hfMan <<= 1;
    hfExp--;
  }

  uint32_t sRet, eRet, mRet, fVal;

  sRet = hfSign;
  if (!hfMan) {
    eRet = 0;
    mRet = 0;
  } else {
    eRet = hfExp - kFp16ExpBias + kFp32ExpBias;
    mRet = hfMan & kFp16ManMask;
    mRet = mRet << (kFp32ManLen - kFp16ManLen);
  }
  fVal = ((sRet) << kFp32SignIndex) | ((eRet) << kFp32ManLen) | ((mRet) & kFp32MaxMan);
  ret = *(reinterpret_cast<float *>(&fVal));

  return ret;
}

class OpTilingAttrUtilsTest : public testing::Test {
protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(OpTilingAttrUtilsTest, get_bool_attr_success) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_bool", true);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_bool", "bool", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 1);
}

TEST_F(OpTilingAttrUtilsTest, get_bool_attr_fail_1) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_bool", true);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, nullptr, "bool", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_bool_attr_fail_2) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_bool", true);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_bool", "booooooool", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_bool_attr_fail_3) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_bool", true);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_bool", "booooooool", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_str_attr_success_1) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_str", "12345");
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_str", "str", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 5);
}

TEST_F(OpTilingAttrUtilsTest, get_str_attr_success_2) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_str", "12345");
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_str", "str", attr_data_ptr, "string");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 5);
}

TEST_F(OpTilingAttrUtilsTest, get_str_attr_fail_1) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_str", "12345");
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_str", "str", attr_data_ptr, "stttttr");
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_str_attr_fail_2) {
  Operator op("relu", "Relu");
  op.SetAttr("attr_str", "12345");
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_str", "str", attr_data_ptr, "int");
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_str_attr_fail_3) {
  Operator op("relu", "Relu");
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_str", "str", attr_data_ptr, "int");
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_int_attr_success_1) {
  Operator op("relu", "Relu");
  int32_t attr = -123;
  op.SetAttr("attr_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_int", "int", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, attr);
}

TEST_F(OpTilingAttrUtilsTest, get_int_attr_success_2) {
  Operator op("relu", "Relu");
  int32_t attr = -123;
  op.SetAttr("attr_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_int", "int32", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, attr);
}

TEST_F(OpTilingAttrUtilsTest, get_int_attr_to_uint32_success_1) {
  Operator op("relu", "Relu");
  int32_t attr = 123;
  op.SetAttr("attr_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_int", "int", attr_data_ptr, "uint32");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const uint32_t *data = (const uint32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, attr);
}

TEST_F(OpTilingAttrUtilsTest, get_int_attr_to_uint32_success_2) {
  Operator op("relu", "Relu");
  int32_t attr = 123;
  op.SetAttr("attr_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_int", "int32", attr_data_ptr, "uint");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const uint32_t *data = (const uint32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, attr);
}

TEST_F(OpTilingAttrUtilsTest, get_list_int_attr_success_1) {
  Operator op("relu", "Relu");
  vector<int32_t> attr = {-123, 456};
  op.SetAttr("attr_list_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_int", "list_int", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 8);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, -123);
  EXPECT_EQ(*(data+1), 456);
}

TEST_F(OpTilingAttrUtilsTest, get_list_int_attr_success_2) {
  Operator op("relu", "Relu");
  vector<int32_t> attr = {-123, 456};
  op.SetAttr("attr_list_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_int", "list_int32", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 8);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, -123);
  EXPECT_EQ(*(data+1), 456);
}

TEST_F(OpTilingAttrUtilsTest, get_list_int_attr_to_list_uint32_success_1) {
  Operator op("relu", "Relu");
  vector<int32_t> attr = {-123, 456};
  op.SetAttr("attr_list_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_int", "list_int", attr_data_ptr, "list_uint32");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 8);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, -123);
  EXPECT_EQ(*(data+1), 456);
}

TEST_F(OpTilingAttrUtilsTest, get_list_int_attr_to_list_uint32_success_2) {
  Operator op("relu", "Relu");
  vector<int32_t> attr = {-123, 456};
  op.SetAttr("attr_list_int", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_int", "list_int32", attr_data_ptr, "list_uint");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 8);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, -123);
  EXPECT_EQ(*(data+1), 456);
}

TEST_F(OpTilingAttrUtilsTest, get_float_attr_success_1) {
  Operator op("relu", "Relu");
  float attr = 1.23;
  op.SetAttr("attr_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_float", "float", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const float *data = (const float *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, attr);
}

TEST_F(OpTilingAttrUtilsTest, get_float_attr_success_2) {
  Operator op("relu", "Relu");
  float attr = 1.23;
  op.SetAttr("attr_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_float", "float32", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const float *data = (const float *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, attr);
}

TEST_F(OpTilingAttrUtilsTest, get_list_float_attr_success_1) {
  Operator op("relu", "Relu");
  vector<float> attr = {1.23, 2.34};
  op.SetAttr("attr_list_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_float", "list_float", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 8);
  const float *data = (const float *)attr_data_ptr->GetData();
  cout << *data << endl;
  cout << *(data+1) << endl;
}

TEST_F(OpTilingAttrUtilsTest, get_list_float_attr_success_2) {
  Operator op("relu", "Relu");
  vector<float> attr = {1.23, 2.34};
  op.SetAttr("attr_list_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_float", "list_float32", attr_data_ptr);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 8);
  const float *data = (const float *)attr_data_ptr->GetData();
  cout << *data << endl;
  cout << *(data+1) << endl;
}

TEST_F(OpTilingAttrUtilsTest, get_float_attr_to_float16_success_1) {
  Operator op("relu", "Relu");
  float attr = 1.23;
  op.SetAttr("attr_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_float", "float", attr_data_ptr, "float16");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 2);
  const uint16_t *data = (const uint16_t *)attr_data_ptr->GetData();
  cout << Uint16ToFloat(*data) << endl;
}

TEST_F(OpTilingAttrUtilsTest, get_float_attr_to_float16_success_2) {
  Operator op("relu", "Relu");
  float attr = 1.23;
  op.SetAttr("attr_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_float", "float32", attr_data_ptr, "float16");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 2);
  const uint16_t *data = (const uint16_t *)attr_data_ptr->GetData();
  cout << Uint16ToFloat(*data) << endl;
}

TEST_F(OpTilingAttrUtilsTest, get_list_float_attr_to_float16_success_1) {
  Operator op("relu", "Relu");
  vector<float> attr = {1.23, -2.34};
  op.SetAttr("attr_list_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_float", "list_float", attr_data_ptr, "list_float16");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const uint16_t *data = (const uint16_t *)attr_data_ptr->GetData();
  cout << Uint16ToFloat(*data) << endl;
  cout << Uint16ToFloat(*(data+1)) << endl;
}

TEST_F(OpTilingAttrUtilsTest, get_list_float_attr_to_float16_success_2) {
  Operator op("relu", "Relu");
  vector<float> attr = {1.23, -2.34};
  op.SetAttr("attr_list_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_float", "list_float32", attr_data_ptr, "list_float16");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const uint16_t *data = (const uint16_t *)attr_data_ptr->GetData();
  cout << Uint16ToFloat(*data) << endl;
  cout << Uint16ToFloat(*(data+1)) << endl;
}

TEST_F(OpTilingAttrUtilsTest, get_float_attr_to_int32_success_1) {
  Operator op("relu", "Relu");
  float attr = 3.56;
  op.SetAttr("attr_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_float", "float", attr_data_ptr, "int32");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, 3);
}

TEST_F(OpTilingAttrUtilsTest, get_float_attr_to_int32_success_2) {
  Operator op("relu", "Relu");
  float attr = -3.56;
  op.SetAttr("attr_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_float", "float", attr_data_ptr, "int");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 4);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, -3);
}

TEST_F(OpTilingAttrUtilsTest, get_float_attr_to_int32_fail_1) {
  Operator op("relu", "Relu");
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_float", "float", attr_data_ptr, "int");
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_list_float_attr_to_int32_success_1) {
  Operator op("relu", "Relu");
  vector<float> attr = {1.63, -2.34};
  op.SetAttr("attr_list_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_float", "list_float32", attr_data_ptr, "list_int");
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(attr_data_ptr->GetSize(), 8);
  const int32_t *data = (const int32_t *)attr_data_ptr->GetData();
  EXPECT_EQ(*data, 1);
  EXPECT_EQ(*(data+1), -2);
}

TEST_F(OpTilingAttrUtilsTest, get_list_float_attr_to_int32_fail_1) {
  Operator op("relu", "Relu");
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_float", "list_float32", attr_data_ptr, "list_int");
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(OpTilingAttrUtilsTest, get_list_float_attr_to_int32_fail_2) {
  Operator op("relu", "Relu");
  vector<float> attr;
  op.SetAttr("attr_list_float", attr);
  AttrDataPtr attr_data_ptr = nullptr;
  graphStatus ret = GetOperatorAttrValue(op, "attr_list_float", "list_float32", attr_data_ptr, "list_int");
  EXPECT_EQ(ret, GRAPH_FAILED);
}

}
