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
#include "register/opdef/op_def_impl.h"

namespace ops {

namespace {

class OpDefParamUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpDefParamUT, ParamTest) {
  OpParamDef param("test");
  OpParamDef param2("test");
  OpParamDef param3("test3");
  EXPECT_EQ(param == param2, true);
  EXPECT_EQ(param == param3, false);
  OpParamTrunk desc;
  desc.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_NCHW})
      .ValueDepend(Option::REQUIRED)
      .IgnoreContiguous()
      .AutoContiguous();
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED);
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::OPTIONAL);
  desc.Input("x3")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .Scalar();
  desc.Input("x4")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ScalarList();
  desc.Output("y")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED)
      .IgnoreContiguous()
      .AutoContiguous()
      .OutputShapeDependOnCompute();
  EXPECT_EQ(desc.Input("x1").GetParamName(), "x1");
  EXPECT_EQ(desc.Input("x1").GetParamType(), Option::OPTIONAL);
  EXPECT_EQ(desc.Input("x1").GetDataTypes().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetFormats().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetUnknownShapeFormats().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetUnknownShapeFormats()[0], ge::FORMAT_NCHW);
  EXPECT_EQ(desc.Input("x1").GetValueDepend(), "required");
  EXPECT_EQ(desc.Input("x1").GetIgnoreContiguous(), true);
  EXPECT_EQ(desc.Input("x1").GetAutoContiguous(), true);
  EXPECT_EQ(desc.Input("x2").GetValueDepend(), "optional");
  EXPECT_EQ(desc.Input("x2").GetIgnoreContiguous(), false);
  EXPECT_EQ(desc.Input("x2").GetAutoContiguous(), false);
  EXPECT_EQ(desc.Output("y").GetIgnoreContiguous(), true);
  EXPECT_EQ(desc.Output("y").GetAutoContiguous(), true);
  EXPECT_EQ(desc.GetInputs().size(), 4);
  EXPECT_EQ(desc.GetOutputs().size(), 1);
  EXPECT_EQ(desc.Input("x1").IsScalar(), false);
  EXPECT_EQ(desc.Input("x1").IsScalarList(), false);
  EXPECT_EQ(desc.Input("x3").IsScalar(), true);
  EXPECT_EQ(desc.Input("x3").IsScalarList(), false);
  EXPECT_EQ(desc.Input("x4").IsScalar(), false);
  EXPECT_EQ(desc.Input("x4").IsScalarList(), true);
  EXPECT_EQ(desc.Output("y").IsOutputShapeDependOnCompute(), true);
}

TEST_F(OpDefParamUT, DependParamTest) {
  OpParamTrunk desc;
  desc.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_NCHW})
      .ValueDepend(Option::REQUIRED)
      .IgnoreContiguous()
      .AutoContiguous();
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED);
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::VIRTUAL, DependScope::TILING)
      .ValueDepend(Option::OPTIONAL, (DependScope)5)
      .ValueDepend(Option::OPTIONAL, DependScope::TILING);
  desc.Output("y")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .IgnoreContiguous()
      .AutoContiguous()
      .OutputShapeDependOnCompute();
  EXPECT_EQ(desc.Input("x1").GetParamName(), "x1");
  EXPECT_EQ(desc.Input("x1").GetParamType(), Option::OPTIONAL);
  EXPECT_EQ(desc.Input("x1").GetDataTypes().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetFormats().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetUnknownShapeFormats().size(), 1);
  EXPECT_EQ(desc.Input("x1").GetUnknownShapeFormats()[0], ge::FORMAT_NCHW);
  EXPECT_EQ(desc.Input("x1").GetValueDepend(), "required");
  EXPECT_EQ(desc.Input("x1").GetDependScope(), DependScope::ALL);
  EXPECT_EQ(desc.Input("x1").GetIgnoreContiguous(), true);
  EXPECT_EQ(desc.Input("x1").GetAutoContiguous(), true);
  EXPECT_EQ(desc.Input("x2").GetValueDepend(), "optional");
  EXPECT_EQ(desc.Input("x2").GetDependScope(), DependScope::TILING);
  EXPECT_EQ(desc.Input("x2").GetIgnoreContiguous(), false);
  EXPECT_EQ(desc.Input("x2").GetAutoContiguous(), false);
  EXPECT_EQ(desc.Output("y").GetIgnoreContiguous(), true);
  EXPECT_EQ(desc.Output("y").GetAutoContiguous(), true);
  EXPECT_EQ(desc.GetInputs().size(), 2);
  EXPECT_EQ(desc.GetOutputs().size(), 1);
  EXPECT_EQ(desc.Input("x1").IsScalar(), false);
  EXPECT_EQ(desc.Input("x1").IsScalarList(), false);
  EXPECT_EQ(desc.Output("y").IsOutputShapeDependOnCompute(), true);
}
TEST_F(OpDefParamUT, FollowParamTest) {
  OpParamTrunk desc;
  desc.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_NCHW})
      .ValueDepend(Option::REQUIRED)
      .IgnoreContiguous()
      .AutoContiguous();
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .Follow("x1")
      .ValueDepend(Option::REQUIRED);
  desc.Output("y")
      .ParamType(Option::OPTIONAL)
      .Follow("x2")
      .IgnoreContiguous()
      .AutoContiguous()
      .OutputShapeDependOnCompute();
  desc.FollowDataImpl();
  EXPECT_EQ(desc.Output("y").GetIgnoreContiguous(), true);
  EXPECT_EQ(desc.Output("y").GetAutoContiguous(), true);
  EXPECT_EQ(desc.GetInputs().size(), 2);
  EXPECT_EQ(desc.GetOutputs().size(), 1);
  EXPECT_EQ(desc.Output("y").IsOutputShapeDependOnCompute(), true);
  EXPECT_EQ(desc.Output("y").GetFollowName(), "x1");
  EXPECT_EQ(desc.Output("y").GetFollowType(), FollowType::ALL);
}
TEST_F(OpDefParamUT, FollowListParamTest) {
  OpParamTrunk desc;
  desc.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataTypeList({ge::DT_FLOAT16})
      .FormatList({ge::FORMAT_ND});
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .Follow("x1", FollowType::DTYPE)
      .FormatList({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED);
  desc.Output("y")
      .ParamType(Option::OPTIONAL)
      .Follow("x2", (FollowType)5)
      .Follow("x2", FollowType::DTYPE);
  desc.Output("x1")
      .ParamType(Option::OPTIONAL)
      .Follow("x1");

  desc.FollowDataImpl();
  auto flwMap = desc.GetFollowMap();
  auto shpMap = desc.GetShapeMap();
  auto dtpMap = desc.GetDtypeMap();
  EXPECT_EQ(desc.Output("y").GetFollowType(), FollowType::DTYPE);
  EXPECT_EQ(desc.GetParamDef("y", OpDef::PortStat::OUT).GetFollowType(), FollowType::DTYPE);
  EXPECT_EQ(desc.Output("x1").GetFollowType(), FollowType::ALL);
}
TEST_F(OpDefParamUT, CommentTest) {
  OpParamTrunk desc;
  desc.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataTypeList({ge::DT_FLOAT16})
      .Comment("comment of param x1")
      .FormatList({ge::FORMAT_ND});
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataTypeList({ge::DT_FLOAT16})
      .Comment("")
      .Comment("comment of param x2")
      .FormatList({ge::FORMAT_ND});
  desc.Output("y")
      .ParamType(Option::OPTIONAL)
      .DataTypeList({ge::DT_FLOAT16})
      .Comment("comment of param y")
      .FormatList({ge::FORMAT_ND});

  EXPECT_EQ(desc.Input("x1").GetComment(), "comment of param x1");
  EXPECT_EQ(desc.Input("x2").GetComment(), "comment of param x2");
  EXPECT_EQ(desc.Output("y").GetComment(), "comment of param y");
}
TEST_F(OpDefParamUT, ForBinQueryTest) {
  OpParamTrunk desc;
  desc.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT})
      .DataTypeForBinQuery({})
      .DataTypeForBinQuery({ge::DT_FLOAT})
      .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT})
      .FormatList({ge::FORMAT_NC, ge::FORMAT_ND})
      .FormatForBinQuery({})
      .FormatForBinQuery({ge::FORMAT_ND})
      .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND});
  desc.Input("x2")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
      .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT})
      .Format({ge::FORMAT_NC, ge::FORMAT_ND})
      .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND});
  desc.Output("y")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
      .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT})
      .Format({ge::FORMAT_NC, ge::FORMAT_ND})
      .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND});
  EXPECT_EQ(desc.Input("x1").GetDataTypesForBin()[0], ge::DT_FLOAT);
  EXPECT_EQ(desc.Input("x2").GetFormatsForBin()[0], ge::FORMAT_ND);
  EXPECT_EQ(desc.Output("y").GetFormatsForBin().size(), 2);
}
}  // namespace
}  // namespace ops
