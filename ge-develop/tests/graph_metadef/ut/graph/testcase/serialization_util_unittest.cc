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
#include <iostream>
#include "graph/serialization/utils/serialization_util.h"

namespace ge {
class SerializationUtilUTest : public testing::Test {
 public:
  proto::DataType proto_data_type_;
  DataType ge_data_type_;

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(SerializationUtilUTest, GetComplex32ProtoDataType) {
  SerializationUtil::GeDataTypeToProto(DT_COMPLEX32, proto_data_type_);
  EXPECT_EQ(proto_data_type_, proto::DT_COMPLEX32);
  SerializationUtil::ProtoDataTypeToGe(proto::DT_COMPLEX32, ge_data_type_);
  EXPECT_EQ(ge_data_type_, DT_COMPLEX32);
}
}  // namespace ge
