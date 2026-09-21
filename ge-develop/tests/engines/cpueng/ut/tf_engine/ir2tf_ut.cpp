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
#include "common/util/util.h"
#define private public
#include "ir2tf/ir2tf_base_parser.h"
#undef private
#include "graph/compute_graph.h"
#include "graph/operator.h"
#include "graph/utils/attr_utils.h"

using namespace aicpu;
using namespace testing;
using namespace ge;
using namespace std;
using namespace domi;

class Ir2tfBaseParserTEST : public testing::Test {
    using Ir2tfBaseParserPtr = std::shared_ptr<Ir2tfBaseParser>;
public:
    Ir2tfBaseParserPtr instance_;
protected:
    static void SetUpTestCase() {
        cout << "Ir2tfBaseParserTEST SetUP" << endl;
    }

    static void TearDownTestCase() {
        cout << "Ir2tfBaseParserTEST TearDown" << endl;
    }

    virtual void SetUp() {
        cout << "Ir2tfBaseParserTEST SetUp" << endl;
        instance_ = Ir2tfBaseParser::Instance();
    }

    virtual void TearDown() {
    }
};
TEST_F(Ir2tfBaseParserTEST, Test_Instance_001){
    ASSERT_NE(nullptr, instance_);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_ConcatV2_SUCCESS){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("concatV2", "ConcatV2");
    string inName = "x";
    string outName = "y";
    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddDynamicInputDesc(inName, static_cast<uint32_t>(3), true);
    opDescPtr->AddDynamicOutputDesc(outName, static_cast<uint32_t>(3), true);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_INPUT_TD_NUM(inName), 3);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_OUTPUT_TD_NUM(outName), 3);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    int dIndex = 0;
    int oIndex = 0;
    ge::AttrUtils::GetInt(opDescPtr, DYNAMIC_INPUT_TD_NUM(inName), dIndex);
    ge::AttrUtils::GetInt(opDescPtr, DYNAMIC_OUTPUT_TD_NUM(outName), oIndex);
    ASSERT_EQ(dIndex, 3);
    ASSERT_EQ(oIndex, 3);
    ASSERT_EQ("ConcatV2", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_FIFOQueue_SUCCESS){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("fifo_queue", "FIFOQueue");
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("FIFOQueueV2", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_attrsMapDescTestSrc_SUCCESS){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("extdesc_type_test", "attrsMapDescTestSrc");
    opDescPtr->SetAttr("T1", GeAttrValue::CreateFrom<int64_t>(4));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("attrsMapDescTestDst", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_attrExtDescTestSrc_SUCCESS){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("extdesc_type_test", "attrExtDescTestSrc");
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("attrExtDescTestDst", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_ConcatV2_FAILED01){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("concatV3", "ConcatV3");
    string inName = "x";
    string outName = "y";
    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddDynamicInputDesc(inName, static_cast<uint32_t>(3), true);
    opDescPtr->AddDynamicOutputDesc(outName, static_cast<uint32_t>(3), true);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_INPUT_TD_NUM(inName), 3);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_OUTPUT_TD_NUM(outName), 3);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    ASSERT_EQ(instance_->ParseNodeDef(*nodePtr.get(), &nodeDef), ErrorCode::IR2TF_CONFIG_INVALID);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_ConcatV2_FAILED02){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("concatV4", "ConcatV4");
    string inName = "x";
    string outName = "y";
    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddDynamicInputDesc(inName, static_cast<uint32_t>(3), true);
    opDescPtr->AddDynamicOutputDesc(outName, static_cast<uint32_t>(3), true);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_INPUT_TD_NUM(inName), 3);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_OUTPUT_TD_NUM(outName), 3);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    ASSERT_EQ(instance_->ParseNodeDef(*nodePtr.get(), &nodeDef), ErrorCode::IR2TF_CONFIG_INVALID);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_ConcatV2_FAILED03){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("concatV5", "ConcatV5");
    string inName = "x";
    string outName = "y";
    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddDynamicInputDesc(inName, static_cast<uint32_t>(3), true);
    opDescPtr->AddDynamicOutputDesc(outName, static_cast<uint32_t>(3), true);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_INPUT_TD_NUM(inName), 3);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_OUTPUT_TD_NUM(outName), 3);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    ASSERT_EQ(instance_->ParseNodeDef(*nodePtr.get(), &nodeDef), ErrorCode::IR2TF_CONFIG_INVALID);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_ConcatV2_FAILED04){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("concatV6", "ConcatV6");
    string inName = "x";
    string outName = "y";
    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddDynamicInputDesc(inName, static_cast<uint32_t>(3), true);
    opDescPtr->AddDynamicOutputDesc(outName, static_cast<uint32_t>(3), true);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_INPUT_TD_NUM(inName), 3);
    ge::AttrUtils::SetInt(opDescPtr, DYNAMIC_OUTPUT_TD_NUM(outName), 3);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    ASSERT_EQ(instance_->ParseNodeDef(*nodePtr.get(), &nodeDef), ErrorCode::IR2TF_CONFIG_INVALID);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_Add){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("add", "Add");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddInputDesc("input_y", yGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    // to cover tensor attr, add tensor attr
    auto tensor1 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto tensor2 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT32));
    auto tensor3 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_BOOL));
    auto tensor4 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT64));
    auto tensor5 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_FLOAT16));
    auto tensor6 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8));

    opDescPtr->SetAttr("testTensor1", GeAttrValue::CreateFrom<GeTensorPtr>(tensor1));
    opDescPtr->SetAttr("testTensor2", GeAttrValue::CreateFrom<GeTensorPtr>(tensor2));
    opDescPtr->SetAttr("testTensor3", GeAttrValue::CreateFrom<GeTensorPtr>(tensor3));
    opDescPtr->SetAttr("testTensor4", GeAttrValue::CreateFrom<GeTensorPtr>(tensor4));
    opDescPtr->SetAttr("testTensor5", GeAttrValue::CreateFrom<GeTensorPtr>(tensor5));
    opDescPtr->SetAttr("testTensor6", GeAttrValue::CreateFrom<GeTensorPtr>(tensor6));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("Add", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseOutputArgs_FAILED){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("add", "Add");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddInputDesc("input_y", yGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    // to cover tensor attr, add tensor attr
    auto tensor1 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto tensor2 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT32));
    auto tensor3 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_BOOL));
    auto tensor4 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT64));
    auto tensor5 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_FLOAT16));
    auto tensor6 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8));

    opDescPtr->SetAttr("testTensor1", GeAttrValue::CreateFrom<GeTensorPtr>(tensor1));
    opDescPtr->SetAttr("testTensor2", GeAttrValue::CreateFrom<GeTensorPtr>(tensor2));
    opDescPtr->SetAttr("testTensor3", GeAttrValue::CreateFrom<GeTensorPtr>(tensor3));
    opDescPtr->SetAttr("testTensor4", GeAttrValue::CreateFrom<GeTensorPtr>(tensor4));
    opDescPtr->SetAttr("testTensor5", GeAttrValue::CreateFrom<GeTensorPtr>(tensor5));
    opDescPtr->SetAttr("testTensor6", GeAttrValue::CreateFrom<GeTensorPtr>(tensor6));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);
    string opName = "Ad1";
    std::unordered_map<std::string, std::pair<int, int>> outputs;
    ASSERT_EQ(instance_->ParseOutputArgs(nodePtr, opName, outputs), ErrorCode::OP_NOT_EXIST_IN_IR2TF_CONFIG_FILE);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_Mul){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("mul", "Mul");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc zGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddInputDesc("input_y", yGeTensor);
    opDescPtr->AddOutputDesc("output_z", zGeTensor);
    // to cover tensor attr, add tensor attr
    auto tensor1 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto tensor2 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT32));
    auto tensor3 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_BOOL));
    auto tensor4 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT64));
    auto tensor5 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_FLOAT16));
    auto tensor6 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8));

    opDescPtr->SetAttr("testTensor1", GeAttrValue::CreateFrom<GeTensorPtr>(tensor1));
    opDescPtr->SetAttr("testTensor2", GeAttrValue::CreateFrom<GeTensorPtr>(tensor2));
    opDescPtr->SetAttr("testTensor3", GeAttrValue::CreateFrom<GeTensorPtr>(tensor3));
    opDescPtr->SetAttr("testTensor4", GeAttrValue::CreateFrom<GeTensorPtr>(tensor4));
    opDescPtr->SetAttr("testTensor5", GeAttrValue::CreateFrom<GeTensorPtr>(tensor5));
    opDescPtr->SetAttr("testTensor6", GeAttrValue::CreateFrom<GeTensorPtr>(tensor6));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    ASSERT_EQ(instance_->ParseNodeDef(*nodePtr.get(), &nodeDef), ErrorCode::IR2TF_SRC_ATTR_MISSING);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseNodeDef_UnknownOp){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("unknownOp", "UnknownOp");
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);

    ASSERT_EQ("UnknownOp", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_CreateIrParser_Cast){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("cast", "Cast");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_y", yGeTensor);
    opDescPtr->SetAttr("truncate", GeAttrValue::CreateFrom<bool>(true));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("Cast", nodeDef.op());
    ASSERT_EQ(true, nodeDef.mutable_attr()->find("truncate")->second.b());
}

TEST_F(Ir2tfBaseParserTEST, Test_CreateIrParser_Assert){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("input_data", "Assert");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_condition", xGeTensor);
    opDescPtr->AddDynamicInputDesc("input_data", 2);
    opDescPtr->SetAttr("summarize", GeAttrValue::CreateFrom<int64_t>(4));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("Assert", nodeDef.op());
    ASSERT_EQ(4, nodeDef.mutable_attr()->find("summarize")->second.i());
    cout << "=====debug string()=====: " << nodeDef.DebugString() << endl;
    if (nodeDef.mutable_attr()->find("T") != nodeDef.mutable_attr()->end()) {
        ASSERT_EQ(true, nodeDef.mutable_attr()->find("T")->second.has_list());
        for (const auto& type : nodeDef.attr().at("T").list().type()) {
            ASSERT_EQ(1, static_cast<domi::tensorflow::DataType>(type));
        }
    }
}

TEST_F(Ir2tfBaseParserTEST, Test_IrParser_ReadAttrValue){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("assert", "Assert");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_condition", xGeTensor);
    opDescPtr->AddInputDesc("input_data", yGeTensor);
    opDescPtr->SetAttr("summarize", GeAttrValue::CreateFrom<int64_t>(4));
    std::vector<float> attrList;
    attrList.push_back(1.0);
    attrList.push_back(2.0);
    opDescPtr->SetAttr("list", GeAttrValue::CreateFrom<std::vector<float>>(attrList));
    opDescPtr->SetAttr("testBoolAttr", GeAttrValue::CreateFrom<bool>(true));
    opDescPtr->SetAttr("testStrAttr", GeAttrValue::CreateFrom<std::string>("test"));
    opDescPtr->SetAttr("testFloatAttr", GeAttrValue::CreateFrom<float>(1.0));
    opDescPtr->SetAttr("testIntAttr", GeAttrValue::CreateFrom<int64_t>(2));
    std::vector<std::string> strList;
    strList.push_back("0001");
    strList.push_back("0002");
    opDescPtr->SetAttr("listStrAttr", GeAttrValue::CreateFrom<std::vector<std::string>>(strList));

    auto tensor1 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    std::vector<GeTensorPtr> tensorList;
    tensorList.push_back(tensor1);
    tensorList.push_back(tensor1);
    opDescPtr->SetAttr("tensorListAttr", GeAttrValue::CreateFrom<std::vector<GeTensorPtr>>(tensorList));

    std::vector<std::vector<int64_t>> shapeList;
    vector<int64_t> dimVec;
    dimVec.push_back(1);
    dimVec.push_back(2);
    shapeList.push_back(dimVec);
    vector<int64_t> dimVec1;
    dimVec1.push_back(3);
    dimVec1.push_back(4);
    shapeList.push_back(dimVec1);
    opDescPtr->SetAttr("testShapeList", GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(shapeList));

    std::vector<DataType> typeList;
    typeList.push_back(DT_INT32);
    typeList.push_back(DT_INT64);
    typeList.push_back(DT_FLOAT);
    opDescPtr->SetAttr("testTypeList", GeAttrValue::CreateFrom<std::vector<DataType>>(typeList));

    DataType geType = DT_INT64;
    opDescPtr->SetAttr("testGeType", GeAttrValue::CreateFrom<DataType>(geType));

    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("Assert", nodeDef.op());
    ASSERT_EQ(4, nodeDef.mutable_attr()->find("summarize")->second.i());
    ASSERT_EQ(true, nodeDef.mutable_attr()->find("list")->second.has_list());
    ASSERT_EQ(true, nodeDef.mutable_attr()->find("testBoolAttr")->second.b());
    ASSERT_EQ("test", nodeDef.mutable_attr()->find("testStrAttr")->second.s());
    ASSERT_EQ(1.0, nodeDef.mutable_attr()->find("testFloatAttr")->second.f());
    ASSERT_EQ(2, nodeDef.mutable_attr()->find("testIntAttr")->second.i());
    ASSERT_EQ(tensorflow::DataType::DT_INT64, nodeDef.mutable_attr()->find("testGeType")->second.type());

    auto tfShapeList = nodeDef.mutable_attr()->find("testShapeList")->second.list().shape();
    ASSERT_EQ(1, tfShapeList[0].dim()[0].size());
    ASSERT_EQ(2, tfShapeList[0].dim()[1].size());
    ASSERT_EQ(3, tfShapeList[1].dim()[0].size());
    ASSERT_EQ(4, tfShapeList[1].dim()[1].size());

    auto tfTypeList = nodeDef.attr().at("testTypeList").list().type();
    ASSERT_EQ(tensorflow::DataType::DT_INT32, tfTypeList[0]);
    ASSERT_EQ(tensorflow::DataType::DT_INT64, tfTypeList[1]);
    ASSERT_EQ(tensorflow::DataType::DT_FLOAT, tfTypeList[2]);

    auto tfStrList = nodeDef.mutable_attr()->find("listStrAttr")->second.list().s();
    ASSERT_EQ("0001", tfStrList[0]);
    ASSERT_EQ("0002", tfStrList[1]);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseOutputList_NotSetTFAttr){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("CholeskyGrad", "CholeskyGrad");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc gradGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("x", xGeTensor);
    opDescPtr->AddInputDesc("grad", gradGeTensor);
    opDescPtr->AddDynamicOutputDesc("y", 2);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("CholeskyGrad", nodeDef.op());
    if (nodeDef.mutable_attr()->find("T") != nodeDef.mutable_attr()->end()) {
        ASSERT_EQ(1, nodeDef.mutable_attr()->find("T")->second.type());
    }
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseOutputArgs_Qr){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("qr", "Qr");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("x", xGeTensor);
    opDescPtr->AddDynamicOutputDesc("q", 2);
    opDescPtr->AddDynamicOutputDesc("r", 2);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    std::unordered_map<std::string, std::pair<int, int>> rangeMap;
    instance_->ParseOutputArgs(nodePtr, "Qr", rangeMap);
    auto iterQ = rangeMap.find("q");
    ASSERT_EQ(true, iterQ != rangeMap.end());
    ASSERT_EQ(0, iterQ->second.first);
    ASSERT_EQ(2, iterQ->second.second);
    auto iterR = rangeMap.find("r");
    ASSERT_EQ(true, iterR != rangeMap.end());
    ASSERT_EQ(2, iterR->second.first);
    ASSERT_EQ(4, iterR->second.second);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseOutputArgs_CholeskyGrad){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("CholeskyGrad", "CholeskyGrad");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc gradGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("x", xGeTensor);
    opDescPtr->AddInputDesc("grad", gradGeTensor);
    opDescPtr->AddDynamicOutputDesc("y", 2);
    opDescPtr->SetAttr("num", GeAttrValue::CreateFrom<int64_t>(3));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    std::unordered_map<std::string, std::pair<int, int>> rangeMap;
    instance_->ParseOutputArgs(nodePtr, "CholeskyGrad", rangeMap);
    auto iter = rangeMap.find("output");
    ASSERT_EQ(true, iter != rangeMap.end());
    ASSERT_EQ(0, iter->second.first);
    ASSERT_EQ(3, iter->second.second);
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseShape){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("qr", "Qr");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("x", xGeTensor);
    opDescPtr->AddDynamicOutputDesc("q", 2);
    opDescPtr->AddDynamicOutputDesc("r", 2);

    std::vector<int64_t> shapeList;
    shapeList.push_back(1);
    shapeList.push_back(2);
    shapeList.push_back(3);
    opDescPtr->SetAttr("element_shape", GeAttrValue::CreateFrom<std::vector<int64_t>>(shapeList));

    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("Qr", nodeDef.op());
    if (nodeDef.mutable_attr()->find("T") != nodeDef.mutable_attr()->end()) {
        ASSERT_EQ(true, nodeDef.mutable_attr()->find("T")->second.has_list());
    }
    if (nodeDef.mutable_attr()->find("type") != nodeDef.mutable_attr()->end()) {
        ASSERT_EQ(true, nodeDef.mutable_attr()->find("type")->second.has_list());
    }
    for (const auto& type : nodeDef.attr().at("T").list().type()) {
        ASSERT_EQ(1, static_cast<domi::tensorflow::DataType>(type));
    }

    for (const auto& type : nodeDef.attr().at("type").list().type()) {
        ASSERT_EQ(1, static_cast<domi::tensorflow::DataType>(type));
    }

    auto tfShape = nodeDef.attr().at("element_shape").shape().dim();
    ASSERT_EQ(1, tfShape[0].size());
    ASSERT_EQ(2, tfShape[1].size());
    ASSERT_EQ(3, tfShape[2].size());
}

TEST_F(Ir2tfBaseParserTEST, Test_ParseUnknownShape){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("ResizeArea", "ResizeArea");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_y", yGeTensor);

    std::vector<int64_t> shapeList;
    shapeList.push_back(0);
    opDescPtr->SetAttr("testShape", GeAttrValue::CreateFrom<std::vector<int64_t>>(shapeList));

    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("ResizeArea", nodeDef.op());

    ASSERT_EQ(true, nodeDef.attr().at("testShape").shape().unknown_rank());
}

TEST_F(Ir2tfBaseParserTEST, Test_Blacklist){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("CountUpTo", "CountUpTo");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    opDescPtr->AddInputDesc("input_x", xGeTensor);
    opDescPtr->AddOutputDesc("output_y", yGeTensor);

    opDescPtr->SetAttr("attrs1", GeAttrValue::CreateFrom<bool>(true));
    opDescPtr->SetAttr("attrs2", GeAttrValue::CreateFrom<float>(1.0));
    opDescPtr->SetAttr("testStrAttr", GeAttrValue::CreateFrom<std::string>("test"));

    std::vector<int64_t> intList;
    intList.push_back(1);
    intList.push_back(2);
    opDescPtr->SetAttr("attrs3", GeAttrValue::CreateFrom<std::vector<int64_t>>(intList));

    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("CountUpTo", nodeDef.op());

    auto attrMap = nodeDef.mutable_attr();
    ASSERT_EQ(attrMap->end(), nodeDef.mutable_attr()->find("attrs1"));
    ASSERT_EQ(attrMap->end(), nodeDef.mutable_attr()->find("attrs2"));
    ASSERT_EQ(true, nodeDef.mutable_attr()->find("attrs3")->second.has_list());
    ASSERT_EQ("test", nodeDef.mutable_attr()->find("testStrAttr")->second.s());
}

TEST_F(Ir2tfBaseParserTEST, Test_SetTfDefaultAttrType_AsString){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("AsString", "AsString");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_STRING);
    opDescPtr->AddInputDesc("x", xGeTensor);
    opDescPtr->AddOutputDesc("y", yGeTensor);
    opDescPtr->SetAttr("shortest", GeAttrValue::CreateFrom<bool>(false));
    opDescPtr->SetAttr("width", GeAttrValue::CreateFrom<int64_t>(-1));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("AsString", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_SetTfDefaultAttrType_MutableDenseHashTable){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test2");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("MutableDenseHashTable", "MutableDenseHashTable");

    GeTensorDesc x1GeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc x2GeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("empty_key", x1GeTensor);
    opDescPtr->AddInputDesc("deleted_key", x2GeTensor);
    opDescPtr->AddOutputDesc("handle", yGeTensor);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("MutableDenseHashTableV2", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_SetTfAttrEnum_DropOutGenMaskV4){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test2");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("DropOutGenMaskV4", "DropOutGenMaskV4");

    GeTensorDesc x1GeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    GeTensorDesc x2GeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_FLOAT);
    GeTensorDesc yGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_UINT8);
    opDescPtr->AddInputDesc("shape", x1GeTensor);
    opDescPtr->AddInputDesc("prob", x2GeTensor);
    opDescPtr->AddOutputDesc("y", yGeTensor);
    opDescPtr->SetAttr("seed", GeAttrValue::CreateFrom<int64_t>(0));
    opDescPtr->SetAttr("seed2", GeAttrValue::CreateFrom<int64_t>(0));
    opDescPtr->SetAttr("dtype", GeAttrValue::CreateFrom<DataType>(DT_BOOL));
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("DropOutGenMaskV4", nodeDef.op());
}

TEST_F(Ir2tfBaseParserTEST, Test_SetTfListShape_OptionalGetValue){
    domi::tensorflow::NodeDef nodeDef;
    //to construct instance nodePtr, nodeDef
    ComputeGraphPtr graphPtr = std::make_shared<ComputeGraph>("test");
    OpDescPtr opDescPtr = std::make_shared<OpDesc>("OptionalGetValue", "OptionalGetValue");

    GeTensorDesc xGeTensor = GeTensorDesc(GeShape(), FORMAT_NCHW, DT_INT32);
    opDescPtr->AddInputDesc("optional", xGeTensor);
    opDescPtr->AddDynamicOutputDesc("components", 2);

    std::vector<std::vector<int64_t>> shapeList;
    shapeList.emplace_back(std::vector<int64_t>({-1,2}));
    shapeList.emplace_back(std::vector<int64_t>({4,5}));
    ge::AttrUtils::SetListListInt(opDescPtr, "output_shapes", shapeList);

    std::vector<DataType> dtypeList = {DT_INT32, DT_INT32};
    ge::AttrUtils::SetListDataType(opDescPtr, "output_types", dtypeList);
    NodePtr nodePtr = graphPtr->AddNode(opDescPtr);

    instance_->ParseNodeDef(*nodePtr.get(), &nodeDef);
    ASSERT_EQ("OptionalGetValue", nodeDef.op());
}
