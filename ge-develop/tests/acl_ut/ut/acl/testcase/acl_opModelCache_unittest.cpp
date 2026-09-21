/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>

#include <gtest/gtest.h>

#define protected public
#define private public
#include "single_op/op_model_cache.h"
#undef private
#undef protected

#include "acl/acl.h"
#include "framework/runtime/model_v2_executor.h"
using namespace acl; 
using namespace std;

class UTEST_ACL_OpModelCache : public testing::Test {
protected:
    void SetUp() {}
    void TearDown() {}

    OpModelCache cache_;
};

TEST_F(UTEST_ACL_OpModelCache, Test)
{
    OpModelDef def;
    def.opType = "testOp";
    def.modelPath = "modelPath";
    def.opModelId = 1UL;
    OpModel model;
    EXPECT_NE(cache_.GetOpModel(def, model), ACL_SUCCESS);

    EXPECT_EQ(cache_.Add(def.opModelId, model), ACL_SUCCESS);
    EXPECT_EQ(cache_.GetOpModel(def, model), ACL_SUCCESS);
    EXPECT_EQ(cache_.Delete(def, true), ACL_SUCCESS);

    def.modelPath = "another_modelPath";
    def.opModelId = 2UL;
    EXPECT_EQ(cache_.Add(def.opModelId, model), ACL_SUCCESS);
    EXPECT_EQ(cache_.Delete(def, false), ACL_SUCCESS);
}

TEST_F(UTEST_ACL_OpModelCache, TestRTV2Executor)
{
  OpModelDef def;
  def.opType = "testOp";
  def.modelPath = "modelPath";
  def.opModelId = 1UL;
  OpModel model;
  EXPECT_EQ(cache_.Add(def.opModelId, model), ACL_SUCCESS);
  std::shared_ptr<gert::StreamExecutor> executor = nullptr;
  EXPECT_EQ(cache_.UpdateCachedExecutor(model.opModelId, executor), ACL_SUCCESS);

  const aclrtStream stream = (aclrtStream) 0x1234;
  EXPECT_EQ(cache_.CleanCachedExecutor(stream), ACL_SUCCESS);
}