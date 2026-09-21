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

#include "../graph/ops_stub.h"
#include "graph/operator_factory.h"

#include "macro_utils/dt_public_scope.h"
#include "operator_factory_impl.h"
#include "macro_utils/dt_public_unscope.h"

using namespace ge;
class UtestGeOperatorFactory : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST(UtestGeOperatorFactory, create_operator) {
  Operator acosh = OperatorFactory::CreateOperator("acosh", "Acosh");
  EXPECT_EQ("Acosh", acosh.GetOpType());
  EXPECT_EQ("acosh", acosh.GetName());
  EXPECT_EQ(false, acosh.IsEmpty());
}

TEST(UtestGeOperatorFactory, create_operator_nullptr) {
  Operator abc = OperatorFactory::CreateOperator("abc", "ABC");
  EXPECT_EQ(true, abc.IsEmpty());
}

TEST(UtestGeOperatorFactory, get_infer_shape_func) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  InferShapeFunc infer_shape_func = OperatorFactoryImpl::GetInferShapeFunc("ABC");
  EXPECT_EQ(nullptr, infer_shape_func);
}

TEST(UtestGeOperatorFactory, get_verify_func) {
  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  VerifyFunc verify_func = OperatorFactoryImpl::GetVerifyFunc("ABC");
  EXPECT_EQ(nullptr, verify_func);
}

TEST(UtestGeOperatorFactory, get_ops_type_list) {
  std::vector<std::string> all_ops;
  graphStatus status = OperatorFactory::GetOpsTypeList(all_ops);
  EXPECT_NE(0, all_ops.size());
  EXPECT_EQ(GRAPH_SUCCESS, status);
}

TEST(UtestGeOperatorFactory, is_exist_op) {
  graphStatus status = OperatorFactory::IsExistOp("Acosh");
  EXPECT_EQ(true, status);
  status = OperatorFactory::IsExistOp("ABC");
  EXPECT_EQ(false, status);
}

TEST(UtestGeOperatorFactory, register_func) {
  OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  graphStatus status = OperatorFactoryImpl::RegisterInferShapeFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterInferShapeFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);

  OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  status = OperatorFactoryImpl::RegisterVerifyFunc("test", nullptr);
  EXPECT_EQ(GRAPH_FAILED, status);
  status = OperatorFactoryImpl::RegisterVerifyFunc("ABC", nullptr);
  EXPECT_EQ(GRAPH_SUCCESS, status);
}
/*
TEST(UtestGeOperatorFactory, get_ops_type_list_fail) {
  auto operator_creators_temp = OperatorFactoryImpl::operator_creators_;
  OperatorFactoryImpl::operator_creators_ = nullptr;
  std::vector<std::string> all_ops;
  graphStatus status = OperatorFactoryImpl::GetOpsTypeList(all_ops);
  EXPECT_EQ(GRAPH_FAILED, status);
  OperatorFactoryImpl::operator_creators_ = operator_creators_temp;
}
*/
