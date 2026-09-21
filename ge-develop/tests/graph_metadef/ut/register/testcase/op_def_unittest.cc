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

namespace ops {

namespace {

class OpDefUT : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(OpDefUT, Construct) {
  OpDef opDef("Test");
  opDef.Input("x1").DataType({ge::DT_FLOAT16}).InitValue({ScalarType::UINT64, 1u})
                   .InitValue({{ScalarType::FLOAT16, 1.1}});
  opDef.Input("x2").DataType({ge::DT_FLOAT16}).Scalar().To("x3");
  opDef.Input("x3").DataType({ge::DT_FLOAT}).Version(1).InitValue({{ScalarType::FLOAT32, 1.1}})
                   .InitValue({ScalarType::UINT32, 1u});
  opDef.Input("x4").DataType({ge::DT_FLOAT}).ScalarList().To(ge::DT_INT32);
  opDef.Output("y").DataType({ge::DT_FLOAT16}).InitValue({ScalarType::INT64, 1});
  opDef.SetInferShape(ge::InferShape4AddAscendC);
  opDef.SetInferShapeRange(ge::InferShapeRange4AddAscendC);
  opDef.SetInferDataType(ge::InferDataType4AddAscendC);
  OpAICoreConfig aicConfig;
  aicConfig.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_ND})
      .UnknownShapeFormat({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED)
      .InitValue(0);
  opDef.AICore().AddConfig("ascend310p", aicConfig);
  aicConfig.ExtendCfgInfo("rangeLimit.value", "limited");
  EXPECT_EQ(ge::AscendString("Test"), opDef.GetOpType());
  std::vector<OpParamDef> inputs = opDef.GetMergeInputs(aicConfig);
  EXPECT_EQ(inputs.size(), 4);
  OpParamDef param = inputs[0];
  auto initValueList = param.GetInitValueList();
  auto originTypes = param.GetOriginDataTypes();
  EXPECT_EQ(param.GetParamName(), ge::AscendString("x1"));
  EXPECT_EQ(param.GetParamType(), Option::OPTIONAL);
  EXPECT_EQ(param.GetDataTypes()[0], ge::DT_FLOAT);
  EXPECT_EQ(param.GetFormats()[0], ge::FORMAT_ND);
  EXPECT_EQ(param.GetUnknownShapeFormats()[0], ge::FORMAT_ND);
  EXPECT_EQ(param.GetValueDepend(), ge::AscendString("required"));
  EXPECT_EQ(param.GetInitValue().value_u64, 0);
  EXPECT_EQ(param.GetInitValueType(), InitValueType::INIT_VALUE_UINT64_T);
  EXPECT_EQ(inputs[1].IsScalar(), true);
  EXPECT_EQ(inputs[3].IsScalarList(), true);
  EXPECT_EQ(inputs[1].GetDataTypes()[0], ge::DT_FLOAT);
  EXPECT_EQ(inputs[3].GetDataTypes()[0], ge::DT_INT32);
  EXPECT_EQ(inputs[3].GetScalarType(), ge::DT_INT32);
  std::vector<OpParamDef> outputs = opDef.GetMergeOutputs(aicConfig);
  EXPECT_EQ(outputs.size(), 1);
  OpParamDef paramOut = outputs[0];
  EXPECT_EQ(paramOut.GetParamType(), Option::REQUIRED);
  EXPECT_EQ(paramOut.GetDataTypes()[0], ge::DT_FLOAT16);
  EXPECT_EQ(paramOut.GetFormats()[0], ge::FORMAT_ND);
  aicConfig.Input("x1")
      .DataType({ge::DT_FLOAT})
      .Format({ge::FORMAT_NCHW});
  inputs = opDef.GetMergeInputs(aicConfig);
  EXPECT_EQ(inputs.size(), 4);
  param = inputs[0];
  EXPECT_EQ(param.GetDataTypes().size(), 1);
  EXPECT_EQ(param.GetFormats().size(), 1);
  EXPECT_EQ(inputs[2].GetVersion(), 1);
}

TEST_F(OpDefUT, ListParamTest) {
  OpDef opDef("Test");
  opDef.Input("x1").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
  opDef.Input("x2").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
  opDef.Input("x3").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT}).Version(1).FormatList({ge::FORMAT_ND});
  opDef.Input("x4").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Scalar().FormatList({ge::FORMAT_ND}).To(ge::DT_FLOAT);
  opDef.Output("y").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT}).FormatList({ge::FORMAT_ND});
  opDef.SetInferShape(ge::InferShape4AddAscendC);
  opDef.SetInferShapeRange(ge::InferShapeRange4AddAscendC);
  opDef.SetInferDataType(ge::InferDataType4AddAscendC);

  OpAICoreConfig aicConfig;
  aicConfig.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT})
      .FormatList({ge::FORMAT_ND})
      .ValueDepend(Option::REQUIRED)
      .InitValue(0);
  opDef.AICore().AddConfig("ascend310p", aicConfig);
  aicConfig.ExtendCfgInfo("rangeLimit.value", "limited");
  EXPECT_EQ(ge::AscendString("Test"), opDef.GetOpType());
  std::vector<OpParamDef> inputs = opDef.GetMergeInputs(aicConfig);
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(inputs[i].GetDataTypes().size(), 16);
    EXPECT_EQ(inputs[i].GetFormats().size(), 16);
    for (size_t j = 0; j < inputs[i].GetFormats().size(); ++j) {
      EXPECT_EQ(inputs[i].GetFormats()[j], ge::FORMAT_ND);
    }
  }
  for (uint32_t i = 0; i < 8; ++i) {
    EXPECT_EQ(inputs[0].GetDataTypes()[i], ge::DT_FLOAT16);
  }
  for (uint32_t i = 8; i < 16; ++i) {
    EXPECT_EQ(inputs[0].GetDataTypes()[i], ge::DT_FLOAT);
  }
  std::vector<OpParamDef> outputs = opDef.GetMergeOutputs(aicConfig);
  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(outputs[i].GetDataTypes().size(), 16);
    EXPECT_EQ(outputs[i].GetFormats().size(), 16);
    for (size_t j = 0; j < outputs[i].GetFormats().size(); ++j) {
      EXPECT_EQ(outputs[i].GetFormats()[j], ge::FORMAT_ND);
    }
  }
}

TEST_F(OpDefUT, ListParamTest1) {
  OpDef opDef("Test");
  opDef.Input("x1").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT});
  opDef.Input("x2").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT}).ScalarList().To("x1");
  opDef.Input("x3").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT}).Version(1);
  opDef.Input("x4").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Scalar().To(ge::DT_FLOAT);
  opDef.Output("y").DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT});
  opDef.SetInferShape(ge::InferShape4AddAscendC);
  opDef.SetInferShapeRange(ge::InferShapeRange4AddAscendC);
  opDef.SetInferDataType(ge::InferDataType4AddAscendC);

  OpAICoreConfig aicConfig;
  aicConfig.Input("x1")
      .ParamType(Option::OPTIONAL)
      .DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT})
      .ValueDepend(Option::REQUIRED)
      .InitValue(0);
  opDef.AICore().AddConfig("ascend310p", aicConfig);
  aicConfig.ExtendCfgInfo("rangeLimit.value", "limited");
  EXPECT_EQ(ge::AscendString("Test"), opDef.GetOpType());
  std::vector<OpParamDef> inputs = opDef.GetMergeInputs(aicConfig);
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(inputs[i].GetDataTypes().size(), 8);
    EXPECT_EQ(inputs[i].GetFormats().size(), 8);
    for (size_t j = 0; j < inputs[i].GetFormats().size(); ++j) {
      EXPECT_EQ(inputs[i].GetFormats()[j], ge::FORMAT_ND);
    }
  }
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(inputs[0].GetDataTypes()[i], ge::DT_FLOAT16);
  }
  for (uint32_t i = 8; i < 8; ++i) {
    EXPECT_EQ(inputs[0].GetDataTypes()[i], ge::DT_FLOAT);
  }
  std::vector<OpParamDef> outputs = opDef.GetMergeOutputs(aicConfig);
  for (size_t i = 0; i < outputs.size(); ++i) {
    EXPECT_EQ(outputs[i].GetDataTypes().size(), 8);
    EXPECT_EQ(outputs[i].GetFormats().size(), 8);
    for (size_t j = 0; j < outputs[i].GetFormats().size(); ++j) {
      EXPECT_EQ(outputs[i].GetFormats()[j], ge::FORMAT_ND);
    }
  }
}

TEST_F(OpDefUT, MC2Test) {
  OpDef opDef("Test");
  opDef.Input("x1").DataType({ge::DT_FLOAT16});
  opDef.Output("y").DataType({ge::DT_FLOAT16});
  opDef.Attr("group1").AttrType(REQUIRED).String();
  opDef.Attr("group1").AttrType(REQUIRED).String();
  opDef.MC2().HcclGroup("group1");
  std::vector<ge::AscendString> groups = opDef.MC2().GetHcclGroups();
  EXPECT_EQ(groups.size(), 1);
  opDef.MC2().HcclGroup({"group1", "group2"});
  groups = opDef.MC2().GetHcclGroups();
  EXPECT_EQ(groups.size(), 2);
  opDef.MC2().HcclGroup("group2");
  groups = opDef.MC2().GetHcclGroups();
  EXPECT_EQ(groups.size(), 2);

  EXPECT_EQ(opDef.MC2().GetHcclServerType(), HcclServerType::MAX);
  opDef.MC2().HcclServerType(HcclServerType::AICPU, "ascend910b");
  EXPECT_EQ(opDef.MC2().GetHcclServerType("ascend910c"), HcclServerType::MAX);
  EXPECT_EQ(opDef.MC2().GetHcclServerType("ascend910b"), HcclServerType::AICPU);
  EXPECT_EQ(opDef.MC2().GetHcclServerType(), HcclServerType::AICPU);
  opDef.MC2().HcclServerType(HcclServerType::AICORE);
  EXPECT_EQ(opDef.MC2().GetHcclServerType("ascend910c"), HcclServerType::AICORE);
  EXPECT_EQ(opDef.MC2().GetHcclServerType("ascend910b"), HcclServerType::AICPU);
}

TEST_F(OpDefUT, CommentTest) {
  OpDef opDef("Test");
  opDef.Comment(CommentSection::BRIEF, "")
      .Comment(CommentSection::CATEGORY, "ca tgg")
      .Comment(CommentSection::CATEGORY, "catg")
      .Comment(CommentSection::BRIEF, "Brie\nf cmt")
      .Comment(CommentSection::BRIEF, "Brief cmt")
      .Comment(CommentSection::CONSTRAINTS, "Constr\naints cmt 1")
      .Comment(CommentSection::CONSTRAINTS, "Constraints cmt 2")
      .Comment(CommentSection::RESTRICTIONS, "Restrictions cmt")
      .Comment(CommentSection::RESTRICTIONS, "Restriction\ns cmt")
      .Comment(CommentSection::THIRDPARTYFWKCOMPAT, "ThirdParn\nyFwkCopat cmt")
      .Comment(CommentSection::THIRDPARTYFWKCOMPAT, "ThirdPartyFwkCopat cmt")
      .Comment(CommentSection::SEE, "See cmt")
      .Comment(CommentSection::SEE, "Seen\n cmt")
      .Comment(CommentSection::SECTION_MAX, "Seen\n cmt");
  EXPECT_EQ(opDef.GetCateGory(), "catg");
  EXPECT_EQ(opDef.GetBrief().size(), 2);
  EXPECT_EQ(opDef.GetConstraints().size(), 2);
  EXPECT_EQ(opDef.GetRestrictions().size(), 2);
  EXPECT_EQ(opDef.GetSee().size(), 2);
  EXPECT_EQ(opDef.GetThirdPartyFwkCopat().size(), 2);
  EXPECT_EQ(opDef.GetConstraints().at(1), "Constraints cmt 2");
}

TEST_F(OpDefUT, ForBinQueryTest) {
  OpDef opDef("Test");
  opDef.Input("x")
      .ParamType(Option::REQUIRED)
      .DataTypeList({ge::DT_FLOAT16, ge::DT_FLOAT})
      .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT})
      .FormatList({ge::FORMAT_NC, ge::FORMAT_ND})
      .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND});
  opDef.Input("y")
      .ParamType(Option::REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
      .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT})
      .Format({ge::FORMAT_NC, ge::FORMAT_ND})
      .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND});
  opDef.Output("z")
      .ParamType(Option::REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
      .DataTypeForBinQuery({ge::DT_FLOAT, ge::DT_FLOAT})
      .Format({ge::FORMAT_NC, ge::FORMAT_ND})
      .FormatForBinQuery({ge::FORMAT_ND, ge::FORMAT_ND})
      .Follow("x");
  opDef.AICore()
      .AddConfig("ascend910");
  
  auto aicoreMap = opDef.AICore().GetAICoreConfigs();
  auto aicore = aicoreMap["ascend910"];
  std::vector<OpParamDef> inputs = opDef.GetMergeInputs(aicore);
  std::vector<OpParamDef> outputs = opDef.GetMergeOutputs(aicore);
  EXPECT_EQ(inputs[0].GetDataTypesForBin().size(), 8);
  EXPECT_EQ(inputs[0].GetFormatsForBin().size(), 8);
}

TEST_F(OpDefUT, ParamUnalignTest) {
  OpDef opDef("Test");
  opDef.Input("x")
      .ParamType(Option::REQUIRED)
      .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
      .Format({ge::FORMAT_NC});
  opDef.Output("z")
      .ParamType(Option::REQUIRED)
      .Follow("x");
  opDef.AICore()
      .AddConfig("ascend910");
  
  auto aicoreMap = opDef.AICore().GetAICoreConfigs();
  auto aicore = aicoreMap["ascend910"];
  std::vector<OpParamDef> inputs = opDef.GetMergeInputs(aicore);
  std::vector<OpParamDef> outputs = opDef.GetMergeOutputs(aicore);
  EXPECT_EQ(inputs.size(), 1);
  EXPECT_EQ(inputs[0].GetDataTypes().size(), 0);
}

TEST_F(OpDefUT, TestFormatCheckAndEnableCallBack) {
  OpDef opDef("Test");
  opDef.Input("x").DataType({ge::DT_FLOAT16});
  opDef.Output("y").DataType({ge::DT_FLOAT16});
  opDef.FormatMatchMode(ops::FormatCheckOption::DEFAULT);
  EXPECT_EQ(opDef.GetFormatMatchMode(), ops::FormatCheckOption::DEFAULT);
  opDef.FormatMatchMode(ops::FormatCheckOption::STRICT);
  EXPECT_EQ(opDef.GetFormatMatchMode(), ops::FormatCheckOption::STRICT);
  EXPECT_EQ(opDef.IsEnableFallBack(), false);
  opDef.EnableFallBack();
  EXPECT_EQ(opDef.IsEnableFallBack(), true);
}

}  // namespace
}  // namespace ops
