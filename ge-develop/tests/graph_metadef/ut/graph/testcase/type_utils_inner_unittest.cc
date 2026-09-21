/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/type_utils_inner.h"
#include <climits>
#include <gtest/gtest.h>
#include "graph_metadef/graph/debug/ge_util.h"

namespace ge {
class UtestTypeUtilsInner : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTypeUtilsInner, IsDataTypeValid) {
  ASSERT_TRUE(TypeUtilsInner::IsDataTypeValid(DT_INT4));
  ASSERT_FALSE(TypeUtilsInner::IsDataTypeValid(DT_MAX));
}

TEST_F(UtestTypeUtilsInner, IsFormatValid) {
  ASSERT_TRUE(TypeUtilsInner::IsFormatValid(FORMAT_NCHW));
  ASSERT_FALSE(TypeUtilsInner::IsFormatValid(FORMAT_END));
}

TEST_F(UtestTypeUtilsInner, IsDataTypeValid2) {
  ASSERT_FALSE(TypeUtilsInner::IsDataTypeValid("MAX"));
  ASSERT_TRUE(TypeUtilsInner::IsDataTypeValid("UINT64"));
  ASSERT_TRUE(TypeUtilsInner::IsDataTypeValid("STRING_REF"));
}

TEST_F(UtestTypeUtilsInner, IsFormatValid2) {
  ASSERT_TRUE(TypeUtilsInner::IsFormatValid("DECONV_SP_STRIDE8_TRANS"));
  ASSERT_FALSE(TypeUtilsInner::IsFormatValid("FORMAT_END"));
}

TEST_F(UtestTypeUtilsInner, SplitFormatFromStr) {
  string primary_format_str;
  int32_t sub_format;
  ASSERT_EQ(TypeUtilsInner::SplitFormatFromStr(":DDD", primary_format_str, sub_format), GRAPH_FAILED);
  ASSERT_EQ(TypeUtilsInner::SplitFormatFromStr(":123", primary_format_str, sub_format), GRAPH_SUCCESS);
  ASSERT_EQ(TypeUtilsInner::SplitFormatFromStr(":012", primary_format_str, sub_format), GRAPH_SUCCESS);
  ASSERT_EQ(TypeUtilsInner::SplitFormatFromStr(":012@34", primary_format_str, sub_format), GRAPH_FAILED);
  ASSERT_EQ(TypeUtilsInner::SplitFormatFromStr(":123456789123456789", primary_format_str, sub_format), GRAPH_FAILED);
  ASSERT_EQ(TypeUtilsInner::SplitFormatFromStr(":65538", primary_format_str, sub_format), GRAPH_FAILED);
}

TEST_F(UtestTypeUtilsInner, CheckUint64MulOverflow) {
  ASSERT_FALSE(TypeUtilsInner::CheckUint64MulOverflow(0x00ULL, 0x00UL));
  ASSERT_FALSE(TypeUtilsInner::CheckUint64MulOverflow(0x02ULL, 0x01UL));
  ASSERT_TRUE(TypeUtilsInner::CheckUint64MulOverflow(0xFFFFFFFFFFFFULL, 0xFFFFFFFUL));
}

TEST_F(UtestTypeUtilsInner, CheckUint64MulOverflow2) {
  ASSERT_FALSE(TypeUtilsInner::CheckUint64MulOverflow(0, 1));
  ASSERT_FALSE(TypeUtilsInner::CheckUint64MulOverflow(1, 1));
  ASSERT_TRUE(TypeUtilsInner::CheckUint64MulOverflow(ULLONG_MAX, 2));
}

TEST_F(UtestTypeUtilsInner, IsInternalFormat) {
  ASSERT_TRUE(TypeUtilsInner::IsInternalFormat(FORMAT_FRACTAL_Z));
  ASSERT_FALSE(TypeUtilsInner::IsInternalFormat(FORMAT_RESERVED));
}

TEST_F(UtestTypeUtilsInner, ImplyTypeToSSerialString) {
  ASSERT_EQ(TypeUtilsInner::ImplyTypeToSerialString(domi::ImplyType::BUILDIN), "buildin");
  ASSERT_EQ(TypeUtilsInner::ImplyTypeToSerialString(static_cast<domi::ImplyType>(30)), "UNDEFINED");
}

TEST_F(UtestTypeUtilsInner, DomiFormatToFormat) {
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(domi::domiTensorFormat_t::DOMI_TENSOR_NDHWC), FORMAT_NDHWC);
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(static_cast<domi::domiTensorFormat_t>(30)), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtilsInner, FmkTypeToSerialString) {
  ASSERT_EQ(TypeUtilsInner::FmkTypeToSerialString(domi::FrameworkType::CAFFE), "caffe");
}

TEST_F(UtestTypeUtilsInner, ImplyTypeToSerialString) {
  ASSERT_EQ(TypeUtilsInner::ImplyTypeToSerialString(domi::ImplyType::BUILDIN), "buildin");
}

TEST_F(UtestTypeUtilsInner, DomiFormatToFormat2) {
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(domi::DOMI_TENSOR_NCHW), FORMAT_NCHW);
  ASSERT_EQ(TypeUtilsInner::DomiFormatToFormat(domi::DOMI_TENSOR_RESERVED), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtilsInner, FmkTypeToSerialString2) {
  ASSERT_EQ(TypeUtilsInner::FmkTypeToSerialString(domi::CAFFE), "caffe");
  ASSERT_EQ(TypeUtilsInner::FmkTypeToSerialString(static_cast<domi::FrameworkType>(domi::FRAMEWORK_RESERVED + 1)), "");
}
}  // namespace ge
