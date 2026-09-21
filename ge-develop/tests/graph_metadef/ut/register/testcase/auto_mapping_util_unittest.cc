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
#include "graph_builder_utils.h"
#include "graph_metadef/register/register.h"
#include <google/protobuf/message.h>
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_op_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/type_utils.h"
#include "register/op_registry.h"
#include "graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "proto/tensorflow/node_def.pb.h"
#include "register/auto_mapping_util.h"
#include "register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"

using namespace ge;
using namespace domi;
class AutoMappingUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

void CreateTFGraphDef(domi::tensorflow::GraphDef &graph_def) {
  // 1. add node
  auto placeholder0 = graph_def.add_node();
  auto placeholder1 = graph_def.add_node();
  auto add0 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();

  // 2. set info
  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");

  add0->set_name("add0");
  add0->set_op("Add");
  add1->set_name("add1");
  add1->set_op("Add");
  add2->set_name("add2");
  add2->set_op("Add");

  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");

  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  // 3. add edges
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");

  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");

  add1->add_input("mul0");
  add1->add_input("placeholder1");

  add2->add_input("mul1");
  add2->add_input("mul0");

  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
}

TEST_F(AutoMappingUtils, FindAttrValueFalse) {
    domi::tensorflow::GraphDef graph_def;
    domi::tensorflow::AttrValue attr_num;
    CreateTFGraphDef(graph_def);
    bool ret;
    domi::tensorflow::NodeDef *node0 = nullptr;
    ret = ge::AutoMappingUtil::FindAttrValue(node0, string(""), attr_num);
    EXPECT_FALSE(ret);

    domi::tensorflow::NodeDef node1;
    ret = ge::AutoMappingUtil::FindAttrValue(&node1, string(""), attr_num);
    EXPECT_FALSE(ret);

    const domi::tensorflow::NodeDef *node2 = graph_def.mutable_node(0);
    ret = ge::AutoMappingUtil::FindAttrValue(node2, node2->name(), attr_num);
    EXPECT_FALSE(ret);
}

TEST_F(AutoMappingUtils, ConvertShape) {
  domi::tensorflow::TensorShapeProto shape;
  vector<int64_t> shape_dims;

  shape.set_unknown_rank(true);
  ge::AutoMappingUtil::ConvertShape(shape, shape_dims);
  EXPECT_EQ(shape_dims, ge::UNKNOWN_SHAPE);

  shape.set_unknown_rank(false);
  shape.add_dim();
  ge::AutoMappingUtil::ConvertShape(shape, shape_dims);
  EXPECT_NE(shape_dims, ge::UNKNOWN_SHAPE);
}

TEST_F(AutoMappingUtils, ConvertTensor) {
  ge::graphStatus ret;
  domi::tensorflow::TensorProto tensor;
  ge::GeTensorPtr weight;

  tensor.set_dtype(domi::tensorflow::DataType_INT_MAX_SENTINEL_DO_NOT_USE_);
  ret = ge::AutoMappingUtil::ConvertTensor(tensor, weight);
  EXPECT_EQ(ret, GRAPH_FAILED);

  tensor.set_dtype(domi::tensorflow::DT_UINT16_REF);
  ret = ge::AutoMappingUtil::ConvertTensor(tensor, weight);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  tensor.set_dtype(domi::tensorflow::DT_UINT8);
  ret = ge::AutoMappingUtil::ConvertTensor(tensor, weight);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(AutoMappingUtils, ConvertTensorList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<ge::GeTensorPtr> vec;

  list.add_tensor();
  ge::AutoMappingUtil::ConvertTensorList(list, vec);
  EXPECT_EQ(vec.empty(), true);
}

TEST_F(AutoMappingUtils, ConvertFunc) {
  EXPECT_NO_THROW(
    domi::tensorflow::NameAttrList tf_func;
    ge::NamedAttrs ge_func;
    const int32_t kInvalidFuncRecursiveDepth = 31;

    tf_func.set_name("test_fun");
    ge::AutoMappingUtil::ConvertFunc(tf_func, ge_func);
    ge::AutoMappingUtil::ConvertFunc(tf_func, ge_func, kInvalidFuncRecursiveDepth);
  );
}

TEST_F(AutoMappingUtils, ConvertDataTypeList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<ge::DataType> vec;

  list.add_type(domi::tensorflow::DT_INT16);
  ge::AutoMappingUtil::ConvertDataTypeList(list, vec);
  EXPECT_EQ(vec.empty(), false);
}

TEST_F(AutoMappingUtils, ConvertShapeList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<vector<int64_t>> vec;

  list.add_shape();
  ge::AutoMappingUtil::ConvertShapeList(list, vec);
  EXPECT_EQ(vec.empty(), false);
}

TEST_F(AutoMappingUtils, ConvertFuncList) {
  domi::tensorflow::AttrValue_ListValue list;
  std::vector<ge::NamedAttrs> vec;
  const int32_t kInvalidFuncRecursiveDepth = 31;

  list.add_func();
  ge::AutoMappingUtil::ConvertFuncList(list, vec, kInvalidFuncRecursiveDepth);
  EXPECT_EQ(vec.empty(), true);

  ge::AutoMappingUtil::ConvertFuncList(list, vec);
  EXPECT_EQ(vec.empty(), false);
}

const float FLOAT_TEST_NUM = 3.14;
const double DOUBLE_TEST_NUM = 3.1415;
const int INT_TEST_NUM = 0;
const unsigned int UNSIGNED_INT_TEST_NUM = 0;

TEST_F(AutoMappingUtils, CopyAttrValueInputTest) {
    ut::GraphBuilder builder("graph");
    NodePtr node_src = builder.AddNode("ParseSingleNode", "ParseSingleType", 3, 0, FORMAT_ALL);

    ge::Operator op_src = OpDescUtils::CreateOperatorFromNode(node_src);
    ge::Operator op_dst = ge::Operator("ParseSingleExample");
    std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
    std::vector<domi::DynamicInputOutputInfo> value;
    EXPECT_EQ(op_dst.GetInputsSize(), 0);

    const char_t *testName1 = "type_str";
    const char_t *testPort1 = "port_str";
    AttrUtils::SetStr(node_src->GetOpDesc(), testName1, "str_shapes");
    op_desc_dst->AddRegisterInputName(testPort1);
    domi::DynamicInputOutputInfo input1(kInput, testPort1, strlen(testPort1), testName1, strlen(testName1));
    value.push_back(input1);

    const char_t *testName2 = "type_int";
    const char_t *testPort2 = "port_int";
    AttrUtils::SetInt(node_src->GetOpDesc(), testName2, INT_TEST_NUM);
    op_desc_dst->AddRegisterInputName(testPort2);
    domi::DynamicInputOutputInfo input2(kInput, testPort2, strlen(testPort2), testName2, strlen(testName2));
    value.push_back(input2);

    const char_t *testName3 = "type_float";
    const char_t *testPort3 = "port_float";
    AttrUtils::SetFloat(node_src->GetOpDesc(), testName3, FLOAT_TEST_NUM);
    op_desc_dst->AddRegisterInputName(testPort3);
    domi::DynamicInputOutputInfo input3(kInput, testPort3, strlen(testPort3), testName3, strlen(testName3));
    value.push_back(input3);

    const char_t *listName = "Name_inputlist1";
    const char_t *listPort = "port_inputlist1";
    vector<DataType> InListDataType = {DT_STRING, DT_INT32, DT_FLOAT};
    AttrUtils::SetListDataType(node_src->GetOpDesc(), listName, InListDataType);
    op_desc_dst->AddRegisterInputName(listPort);
    domi::DynamicInputOutputInfo input4(kInput, listPort, strlen(listPort), listName, strlen(listName));
    value.push_back(input4);

    auto ret = domi::AutoMappingByOpFnDynamic(op_src, op_dst, value);
    EXPECT_EQ(ret, domi::SUCCESS);
    EXPECT_EQ(op_dst.GetInputsSize(), 3);
}

TEST_F(AutoMappingUtils, CopyAttrValueInputListTest) {
    ut::GraphBuilder builder("graph");
    NodePtr node_src = builder.AddNode("ParseSingleNode", "ParseSingleType", 6, 0, FORMAT_ALL);

    ge::Operator op_src = OpDescUtils::CreateOperatorFromNode(node_src);
    ge::Operator op_dst = ge::Operator("ParseSingleExample");
    std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
    std::vector<domi::DynamicInputOutputInfo> value;
    EXPECT_EQ(op_dst.GetInputsSize(), 0);

    const char_t *testlistName1 = "listName_str";
    const char_t *testlistPort1 = "listport_str";
    vector<std::string> attrStrList = {"image/class/lable","image/encode", "image/format"};
    AttrUtils::SetListStr(node_src->GetOpDesc(), testlistName1, attrStrList);
    op_desc_dst->AddRegisterInputName(testlistPort1);
    domi::DynamicInputOutputInfo input1(kInput, testlistPort1, strlen(testlistPort1), testlistName1, strlen(testlistName1));
    value.push_back(input1);

    const char_t *testlistName2 = "listName_Int";
    const char_t *testlistPort2 = "listport_Int";
    vector<int64_t> attrIntList = {0, 1, 2, 3};
    AttrUtils::SetListInt(node_src->GetOpDesc(), testlistName2, attrIntList);
    op_desc_dst->AddRegisterInputName(testlistPort2);
    domi::DynamicInputOutputInfo input2(kInput, testlistPort2, strlen(testlistPort2), testlistName2, strlen(testlistName2));
    value.push_back(input2);

    const char_t *testlistName3 = "listName_Float";
    const char_t *testlistPort3 = "listport_Float";
    vector<float> attrFloatList = {0.0, 0.1, 0.2, 0.3};
    AttrUtils::SetListFloat(node_src->GetOpDesc(), testlistName3, attrFloatList);
    op_desc_dst->AddRegisterInputName(testlistPort3);
    domi::DynamicInputOutputInfo input3(kInput, testlistPort3, strlen(testlistPort3), testlistName3, strlen(testlistName3));
    value.push_back(input3);

    const char_t *testlistName4 = "listName_Bool";
    const char_t *testlistPort4 = "listport_Bool";
    vector<bool> attrBoolList = {true, false, false, true};
    AttrUtils::SetListBool(node_src->GetOpDesc(), testlistName4, attrBoolList);
    op_desc_dst->AddRegisterInputName(testlistPort4);
    domi::DynamicInputOutputInfo input4(kInput, testlistPort4, strlen(testlistPort4), testlistName4, strlen(testlistName4));
    value.push_back(input4);

    const char_t *testlistName5 = "listName_NamedAttrs";
    const char_t *testlistPort5 = "listport_NamedAttrs";
    NamedAttrs name1; NamedAttrs name2;
    vector<NamedAttrs> attrNamedAttrsList = {name1, name2};
    AttrUtils::SetListNamedAttrs(node_src->GetOpDesc(), testlistName5, attrNamedAttrsList);
    op_desc_dst->AddRegisterInputName(testlistPort5);
    domi::DynamicInputOutputInfo input5(kInput, testlistPort5, strlen(testlistPort5), testlistName5, strlen(testlistName5));
    value.push_back(input5);

    const char_t *testlistName6 = "listName_Int";
    const char_t *testlistPort6 = "listport_Int";
    vector<vector<int64_t>> attrIntListList = {attrIntList, attrIntList};
    AttrUtils::SetListListInt(node_src->GetOpDesc(), testlistName6, attrIntListList);
    op_desc_dst->AddRegisterInputName(testlistPort6);
    domi::DynamicInputOutputInfo input6(kInput, testlistPort6, strlen(testlistPort6), testlistName6, strlen(testlistName6));
    value.push_back(input6);

    const char_t *testlist_ListName = "listName_ListData";
    const char_t *testlist_ListPort = "listport_ListData";
    vector<DataType> InListDataType = {DT_STRING, DT_INT32, DT_FLOAT, DT_BOOL, DT_UNDEFINED, DT_UNDEFINED};
    AttrUtils::SetListDataType(node_src->GetOpDesc(), testlist_ListName, InListDataType);
    op_desc_dst->AddRegisterInputName(testlist_ListPort);
    domi::DynamicInputOutputInfo input7(kInput, testlist_ListPort, strlen(testlist_ListPort), testlist_ListName, strlen(testlist_ListName));
    value.push_back(input7);

    auto ret = domi::AutoMappingByOpFnDynamic(op_src, op_dst, value);
    EXPECT_EQ(ret, domi::SUCCESS);
    EXPECT_EQ(op_dst.GetInputsSize(), 6);
}

TEST_F(AutoMappingUtils, CopyAttrValueOutputTest) {
    ut::GraphBuilder builder("graph");
    NodePtr node_src = builder.AddNode("ParseSingleNode", "ParseSingleType", 0, 4, FORMAT_ALL);

    ge::Operator op_src = OpDescUtils::CreateOperatorFromNode(node_src);
    ge::Operator op_dst = ge::Operator("ParseSingleExample");
    std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
    std::vector<domi::DynamicInputOutputInfo> value;
    EXPECT_EQ(op_dst.GetInputsSize(), 0);

    const char_t *testName1 = "Name_attrBool";
    const char_t *testPort1 = "port_attrBool";
    AttrUtils::SetBool(node_src->GetOpDesc(), testName1, true);
    op_desc_dst->AddRegisterOutputName(testPort1);
    domi::DynamicInputOutputInfo output1(kOutput, testPort1, strlen(testPort1), testName1, strlen(testName1));
    value.push_back(output1);

    const char_t *testName2 = "Name_attrName";
    const char_t *testPort2 = "port_attrName";
    NamedAttrs NamedAttr; NamedAttr.SetName("NamedAttr");
    AttrUtils::SetNamedAttrs(node_src->GetOpDesc(), testName2, NamedAttr);
    op_desc_dst->AddRegisterOutputName(testPort2);
    domi::DynamicInputOutputInfo output2(kOutput, testPort2, strlen(testPort2), testName2, strlen(testName2));
    value.push_back(output2);

    const char_t *testName3 = "Name_attrDataType";
    const char_t *testPort3 = "port_attrDataType";
    AttrUtils::SetDataType(node_src->GetOpDesc(), testName3, DT_INT16);
    op_desc_dst->AddRegisterOutputName(testPort3);
    domi::DynamicInputOutputInfo output3(domi::kOutput, testPort3, strlen(testPort3), testName3, strlen(testName3));
    value.push_back(output3);

    const char_t *testName4 = "Name_attrGraph";
    const char_t *testPort4 = "port_attrGraph";
    ComputeGraphPtr graph = builder.GetGraph();
    AttrUtils::SetGraph(node_src->GetOpDesc(), testName4, graph);
    op_desc_dst->AddRegisterOutputName(testPort4);
    domi::DynamicInputOutputInfo output4(kOutput, testPort4, strlen(testPort4), testName4, strlen(testName4));
    value.push_back(output4);

    const char_t *testName5 = "Name_attrDataTypeList";
    const char_t *testPort5 = "port_attrDataTypeList";
    vector<DataType> OutListDataType = {DT_BOOL, DT_STRING, DT_INT16, DT_RESOURCE};
    AttrUtils::SetListDataType(node_src->GetOpDesc(), testName5, OutListDataType);
    op_desc_dst->AddRegisterOutputName(testPort5);
    domi::DynamicInputOutputInfo output5(kOutput, testPort5, strlen(testPort5), testName5, strlen(testName5));
    value.push_back(output5);

    auto ret = domi::AutoMappingByOpFnDynamic(op_src, op_dst, value);
    EXPECT_EQ(ret, domi::SUCCESS);
    EXPECT_EQ(op_dst.GetOutputsSize(), 4);
}

TEST_F(AutoMappingUtils, ConvertValueTest) {
  ge::NamedAttrs ge_func;
  std::string convertName = "convertName";
  domi::tensorflow::AttrValue value;

  value.set_s(std::string("valueString"));
  value.set_has_s();
  auto op_desc = std::make_shared<OpDesc>();
  ge::AutoMappingUtil::ConvertValue(convertName, value, op_desc, 0);
  std::string valueStr;
  ge::AttrUtils::GetStr(op_desc, convertName, valueStr);
  EXPECT_EQ(valueStr=="valueString", true);

  ge::AutoMappingUtil::ConvertValue(convertName, value, ge_func, 0);
  ge::AttrUtils::GetStr(ge_func, convertName, valueStr);
  EXPECT_EQ(valueStr=="valueString", true);
}

