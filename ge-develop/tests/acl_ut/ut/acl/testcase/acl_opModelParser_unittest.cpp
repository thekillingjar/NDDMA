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
#include <gmock/gmock.h>
#include <cstdint>

#define protected public
#define private public
#include "single_op/op_model_parser.h"
#undef private
#undef protected

#include "acl/acl.h"
#include "single_op/op_model_parser.h"
#include "framework/common/helper/om_file_helper.h"
#include "framework/common/types.h"
#include "acl_stub.h"
#include "graph/utils/attr_utils.h"
#include "graph/ge_tensor.h"


using namespace std;
using namespace testing;
using namespace acl;

class OpModelParserTest : public testing::Test {
protected:
    void SetUp() {
        MockFunctionTest::aclStubInstance().ResetToDefaultMock();
    }
    void TearDown() {
        Mock::VerifyAndClear((void *)(&MockFunctionTest::aclStubInstance()));
    }
};
//
ge::GeAttrValue::NAMED_ATTRS value1;
vector<ge::GeAttrValue::NAMED_ATTRS> g_value{value1};

std::map<string, AnyValue> GetAllAttrsWithSingleOp()
{
    AnyValue attr;
    std::string name = "ATTR_MODEL_test";
    std::map<string, AnyValue> m;
    m.insert(std::make_pair(name, attr));
    std::string name_single_op = "ATTR_MODEL__single_op_scene";
    m.insert(std::make_pair(name_single_op, attr));
    return m;
}

TEST_F(OpModelParserTest, TestDeserializeModel)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Init(_, _))
        .WillOnce(Return(1))
        .WillRepeatedly(Return(0));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetModelPartition(_, _))
        .WillOnce(Return(1))
        .WillRepeatedly(Return(0));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Load(_, _, _))
        .WillOnce(Return(1))
        .WillRepeatedly(Return(0));

    OpModel opModel;
    ge::Model geModel;
    opModel.size = sizeof(struct ge::ModelFileHeader) + 128;
    ge::ModelFileHeader header;
    header.length = 128;
    opModel.data = std::shared_ptr<void>(&header, [](void *) {});

    ASSERT_NE(OpModelParser::DeserializeModel(opModel, geModel), ACL_SUCCESS);
    ASSERT_NE(OpModelParser::DeserializeModel(opModel, geModel), ACL_SUCCESS);
    ASSERT_NE(OpModelParser::DeserializeModel(opModel, geModel), ACL_SUCCESS);
    ASSERT_EQ(OpModelParser::DeserializeModel(opModel, geModel), ACL_SUCCESS);
    header.model_num = 3;
    opModel.data = std::shared_ptr<void>(&header, [](void *) {});
    ASSERT_EQ(OpModelParser::DeserializeModel(opModel, geModel), ACL_SUCCESS);
}

TEST_F(OpModelParserTest, TestParseModelContent)
{
    OpModel opModel;
    ge::Model geModel;

    ge::ModelFileHeader header;
    header.length = 128;
    opModel.data = std::shared_ptr<void>(&header, [](void *) {});

    uint64_t modelSize;
    uint8_t *modelData = nullptr;
    opModel.size = 0;
    EXPECT_NE(OpModelParser::ParseModelContent(opModel, modelSize, modelData), ACL_SUCCESS);

    opModel.size = sizeof(struct ge::ModelFileHeader) + 129;
    EXPECT_NE(OpModelParser::ParseModelContent(opModel, modelSize, modelData), ACL_SUCCESS);

    opModel.size = sizeof(struct ge::ModelFileHeader) + 128;
    EXPECT_EQ(OpModelParser::ParseModelContent(opModel, modelSize, modelData), ACL_SUCCESS);
}

bool GetBool_invoke(ge::AttrUtils::ConstAttrHolderAdapter obj, const string &name, bool &value)
{
    (void) obj;
    (void) name;
    value = false;
    return true;
}

bool GetBool_invoke1(ge::AttrUtils::ConstAttrHolderAdapter obj, const string &name, bool &value)
{
    (void) obj;
    (void) name;
    value = true;
    return true;
}

TEST_F(OpModelParserTest, TestToModelConfig)
{
    OpModelDef opModelDef;
    ge::Model geModel;
    ASSERT_EQ(OpModelParser::ToModelConfig(geModel, opModelDef), ACL_ERROR_MODEL_MISSING_ATTR);
}

TEST_F(OpModelParserTest, ParseGeTensorDescTest1)
{
    vector<ge::GeTensorDesc> geTensorDescList;
    vector<aclTensorDesc> output;
    string opName = "TransData";
    OpModelParser::ParseGeTensorDesc(geTensorDescList, output, opName);
    EXPECT_EQ(output.size(), 0);

    ge::GeTensorDesc desc;
    geTensorDescList.emplace_back(desc);
    OpModelParser::ParseGeTensorDesc(geTensorDescList, output, opName);
    EXPECT_EQ(output.size(), 1);
    opName = "Add";
    EXPECT_EQ(OpModelParser::ParseGeTensorDesc(geTensorDescList, output, opName), ACL_SUCCESS);
}

TEST_F(OpModelParserTest, ParseGeTensorDescTest2)
{
    vector<ge::GeTensorDesc> geTensorDescList;
    vector<aclTensorDesc> output;
    string opName = "TransData";
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), HasAttr(_, _))
        .WillRepeatedly(Return(true));
    OpModelParser::ParseGeTensorDesc(geTensorDescList, output, opName);
    EXPECT_EQ(output.size(), 0);

    ge::GeTensorDesc desc;
    geTensorDescList.emplace_back(desc);
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBool(_, _, _))
        .WillOnce(Invoke(GetBool_invoke1))
        .WillRepeatedly(Return(true));
    OpModelParser::ParseGeTensorDesc(geTensorDescList, output, opName);
    EXPECT_EQ(output.size(), 1);
    opName = "Add";
    EXPECT_EQ(OpModelParser::ParseGeTensorDesc(geTensorDescList, output, opName), ACL_SUCCESS);
}

TEST_F(OpModelParserTest, TestParseOpAttrs)
{
    ge::Model model;
    aclopAttr attr;
    OpModelParser::ParseOpAttrs(model, attr);
    EXPECT_TRUE(attr.Attrs().empty());
}

TEST_F(OpModelParserTest, TestParseOpAttrsWithSingleOpScene)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetAllAttrs())
        .WillOnce(Invoke(GetAllAttrsWithSingleOp));
    ge::Model model;
    aclopAttr attr;
    OpModelParser::ParseOpAttrs(model, attr);
}

bool GetInt_invoke(ge::AttrUtils::ConstAttrHolderAdapter obj, const string &name, int32_t& value)
{
    (void) obj;
    (void) name;
    value = 1;
    return true;
}

bool GetListNamedAttrs_invoke(ge::AttrUtils::ConstAttrHolderAdapter obj, const string &name, vector<ge::GeAttrValue::NAMED_ATTRS> &value)
{
    (void) obj;
    (void) name;
    value = g_value;
    return true;
}


TEST_F(OpModelParserTest, TestParseOpModel)
{
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), Init(_, _))
        .WillOnce(Return(ACL_ERROR_DESERIALIZE_MODEL))
        .WillRepeatedly(Return(ACL_SUCCESS));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
        .WillOnce(Return(false))
        .WillRepeatedly(Return(true));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), HasAttr(_, _))
        .WillRepeatedly(Return(true));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBool(_, _, _))
        .WillOnce(Invoke(GetBool_invoke))
        .WillRepeatedly(Return(true));


    OpModelDef opModelDef;
    OpModel opModel;
    opModel.size = sizeof(struct ge::ModelFileHeader) + 128;
    ge::ModelFileHeader header;
    header.length = 128;
    opModel.data = std::shared_ptr<void>(&header, [](void *) {});
    ASSERT_NE(OpModelParser::ParseOpModel(opModel, opModelDef), ACL_SUCCESS);
    ASSERT_NE(OpModelParser::ParseOpModel(opModel, opModelDef), ACL_SUCCESS);
    ASSERT_NE(OpModelParser::ParseOpModel(opModel, opModelDef), ACL_SUCCESS);
    ASSERT_EQ(OpModelParser::ParseOpModel(opModel, opModelDef), ACL_SUCCESS);

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetInt(_, _, _))
        .WillRepeatedly(Invoke(GetInt_invoke));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetBool(_, _, _))
        .WillRepeatedly(Invoke(GetBool_invoke1));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListNamedAttrs(_, _, _))
        .WillRepeatedly(Invoke(GetListNamedAttrs_invoke));
    opModelDef.inputDescArr.emplace_back();
    opModelDef.outputDescArr.emplace_back();
    ASSERT_EQ(OpModelParser::ParseOpModel(opModel, opModelDef), ACL_ERROR_PARSE_MODEL);
}

TEST_F(OpModelParserTest, TestToModelConfig1)
{
    OpModelDef opModelDef;
    ge::Model geModel;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetInt(_, _, _))
        .WillRepeatedly(Invoke(GetInt_invoke));
    OpModelParser::ToModelConfig(geModel, opModelDef);
}

TEST_F(OpModelParserTest, TestToModelConfig2)
{
    OpModelDef opModelDef;
    ge::Model geModel;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
        .WillRepeatedly(Return(true));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetInt(_, _, _))
        .WillRepeatedly(Invoke(GetInt_invoke));
    OpModelParser::ToModelConfig(geModel, opModelDef);
}

TEST_F(OpModelParserTest, TestToModelConfig3)
{
    OpModelDef opModelDef;
    ge::Model geModel;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
        .WillRepeatedly(Return(true));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetInt(_, _, _))
        .WillRepeatedly(Invoke(GetInt_invoke));

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), HasAttr(_, _))
        .WillRepeatedly(Return(true));
    OpModelParser::ToModelConfig(geModel, opModelDef);
}


TEST_F(OpModelParserTest, TestToModelConfig4)
{
    OpModelDef opModelDef;
    ge::Model geModel;

    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListTensor(_, _, _))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), HasAttr(_, _))
        .WillRepeatedly(Return(true));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetInt(_, _, _))
        .WillRepeatedly(Invoke(GetInt_invoke));
    EXPECT_CALL(MockFunctionTest::aclStubInstance(), GetListNamedAttrs(_, _, _))
        .WillRepeatedly(Invoke(GetListNamedAttrs_invoke));
    opModelDef.inputDescArr.emplace_back();
    opModelDef.outputDescArr.emplace_back();
    OpModelParser::ToModelConfig(geModel, opModelDef);
}