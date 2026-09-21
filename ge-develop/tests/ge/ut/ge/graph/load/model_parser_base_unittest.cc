/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include "macro_utils/dt_public_scope.h"
#include "common/helper/model_parser_base.h"
#include "framework/omg/ge_init.h"
#include "common/model/ge_model.h"
#include "graph/buffer/buffer_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "macro_utils/dt_public_unscope.h"

#include "proto/task.pb.h"

using namespace std;

namespace ge {
class UtestModelParserBase : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestModelParserBase, LoadFromFile) {
  ModelParserBase base;
  ge::ModelData model_data;
  EXPECT_EQ(base.LoadFromFile("/tmp/123test", 1, model_data), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  EXPECT_EQ(base.LoadFromFile("", 1, model_data), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
}

TEST_F(UtestModelParserBase, LoadFromFile_s) {
  system("touch test_xxx");
  ModelParserBase base;
  ge::ModelData model_data;
  
  EXPECT_EQ(base.LoadFromFile("./test_xxx", 1, model_data), ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  system("rm -f ./test_xxx");
}

TEST_F(UtestModelParserBase, ParseModelContent) {
  ModelParserBase base;
  ge::ModelData model;
  uint8_t *modelData = nullptr;
  uint64_t model_len;
  EXPECT_EQ(base.ParseModelContent(model, modelData, model_len), PARAM_INVALID);
}

TEST_F(UtestModelParserBase, GetModelInputDesc) {
  ModelParserBase base;
  uint8_t *data = nullptr;
  size_t size = 0U;
  ge::ModelInOutInfo info;
  EXPECT_EQ(base.GetModelInputDesc(data, size, info), PARAM_INVALID);
}

TEST_F(UtestModelParserBase, GetModelOutputDesc) {
  ModelParserBase base;
  uint8_t *data = nullptr;
  size_t size = 0U;
  ge::ModelInOutInfo info;
  EXPECT_EQ(base.GetModelOutputDesc(data, size, info), PARAM_INVALID);
}

TEST_F(UtestModelParserBase, GetDynamicBatch) {
  ModelParserBase base;
  uint8_t *data = nullptr;
  size_t size = 0U;
  ge::ModelInOutInfo info;
  EXPECT_EQ(base.GetDynamicBatch(data, size, info), PARAM_INVALID);
}

TEST_F(UtestModelParserBase, GetDynamicHW) {
  ModelParserBase base;
  uint8_t *data = nullptr;
  size_t size = 0U;
  ModelInOutInfo info;
  EXPECT_EQ(base.GetDynamicHW(data, size, info), PARAM_INVALID);
}

TEST_F(UtestModelParserBase, GetDynamicDims) {
  ModelParserBase base;
  uint8_t *data = nullptr;
  size_t size = 0U;
  ModelInOutInfo info;
  EXPECT_EQ(base.GetDynamicDims(data, size, info), PARAM_INVALID);
}

TEST_F(UtestModelParserBase, GetDataNameOrder) {
  ModelParserBase base;
  uint8_t *data = nullptr;
  size_t size = 0U;
  ModelInOutInfo info;
  EXPECT_EQ(base.GetDataNameOrder(data, size, info), PARAM_INVALID);
}

TEST_F(UtestModelParserBase, GetDynamicOutShape) {
  ModelParserBase base;
  uint8_t *data = nullptr;
  size_t size = 0U;
  ModelInOutInfo info;
  EXPECT_EQ(base.GetDynamicOutShape(data, size, info), PARAM_INVALID);
}
}  // namespace ge
