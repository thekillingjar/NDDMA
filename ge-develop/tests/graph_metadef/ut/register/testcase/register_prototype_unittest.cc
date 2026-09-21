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

#include "register/prototype_pass_registry.h"
namespace ge {
class UtestProtoTypeRegister : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

class RegisterPass : public ProtoTypeBasePass {
 public:
  Status Run(google::protobuf::Message *message) { return SUCCESS; }
};

class RegisterFail : public ProtoTypeBasePass {
 public:
  Status Run(google::protobuf::Message *message) { return FAILED; }
};

REGISTER_PROTOTYPE_PASS("RegisterPass", RegisterPass, domi::CAFFE);
REGISTER_PROTOTYPE_PASS("RegisterPass", RegisterPass, domi::CAFFE);

TEST_F(UtestProtoTypeRegister, register_test) {
  auto pass_vec = ProtoTypePassRegistry::GetInstance().GetCreateFnByType(domi::CAFFE);
  EXPECT_EQ(pass_vec.size(), 1);
}

TEST_F(UtestProtoTypeRegister, register_test_fail) {
  REGISTER_PROTOTYPE_PASS(nullptr, RegisterPass, domi::CAFFE);
  REGISTER_PROTOTYPE_PASS("RegisterFail", RegisterFail, domi::CAFFE);

  ProtoTypePassRegistry::GetInstance().RegisterProtoTypePass(nullptr, nullptr, domi::CAFFE);
  auto pass_vec = ProtoTypePassRegistry::GetInstance().GetCreateFnByType(domi::CAFFE);
  EXPECT_NE(pass_vec.size(), 1);
}
}  // namespace ge
