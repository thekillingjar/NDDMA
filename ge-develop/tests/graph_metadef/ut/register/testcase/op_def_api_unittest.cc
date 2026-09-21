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
#include <vector>
#include <string>
#include "register/op_def_registry.h"

namespace ge {

static ge::graphStatus InferShape4AddAscendC(gert::InferShapeContext *context) {
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeRange4AddAscendC(gert::InferShapeRangeContext *context) {
  return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType4AddAscendC(gert::InferDataTypeContext *context) {
  return GRAPH_SUCCESS;
}

}  // namespace ge

namespace optiling {

static ge::graphStatus TilingAscendCAdd(gert::TilingContext *context) {
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus check_op_support(const ge::Operator &op, ge::AscendString &result) {
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus get_op_support(const ge::Operator &op, ge::AscendString &result) {
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus op_select_format(const ge::Operator &op, ge::AscendString &result) {
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus get_op_specific_info(const ge::Operator &op, ge::AscendString &result) {
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus generalize_config(const ge::Operator &op, const ge::AscendString &generalize_config,
                             ge::AscendString &generalize_para) {
  return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {

namespace {

class OpDefAPIUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpDefAPIUT, APITest) {
  OpDef opDef("Test");
  opDef.SetInferShape(ge::InferShape4AddAscendC);
  opDef.SetInferShapeRange(ge::InferShapeRange4AddAscendC);
  opDef.SetInferDataType(ge::InferDataType4AddAscendC);
  opDef.AICore().SetTiling(optiling::TilingAscendCAdd);
  opDef.AICore()
      .SetCheckSupport(optiling::check_op_support)
      .SetOpSelectFormat(optiling::op_select_format)
      .SetOpSupportInfo(optiling::get_op_support)
      .SetOpSpecInfo(optiling::get_op_specific_info)
      .SetParamGeneralize(optiling::generalize_config);
  EXPECT_EQ(opDef.GetInferShape(), ge::InferShape4AddAscendC);
  EXPECT_EQ(opDef.GetInferShapeRange(), ge::InferShapeRange4AddAscendC);
  EXPECT_EQ(opDef.GetInferDataType(), ge::InferDataType4AddAscendC);
  EXPECT_EQ(opDef.AICore().GetTiling(), optiling::TilingAscendCAdd);
  EXPECT_EQ(opDef.AICore().GetCheckSupport(), optiling::check_op_support);
  EXPECT_EQ(opDef.AICore().GetOpSelectFormat(), optiling::op_select_format);
  EXPECT_EQ(opDef.AICore().GetOpSupportInfo(), optiling::get_op_support);
  EXPECT_EQ(opDef.AICore().GetOpSpecInfo(), optiling::get_op_specific_info);
  EXPECT_EQ(opDef.AICore().GetParamGeneralize(), optiling::generalize_config);
}

}  // namespace
}  // namespace ops
