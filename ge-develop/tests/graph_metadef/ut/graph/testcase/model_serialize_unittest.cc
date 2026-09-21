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
#include <queue>
#include <map>

#include <utility>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <string>
#include <sstream>
#include <functional>

#include "graph/ge_attr_value.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/op_desc.h"
#include "graph/compute_graph.h"
#include "graph_builder_utils.h"
#include "graph/node.h"
#include "graph/normal_graph/node_impl.h"
#include "test_std_structs.h"
#include "graph/operator_factory.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator_reg.h"

#include "graph/utils/node_utils.h"
#include "proto/ge_ir.pb.h"
#include "proto/om.pb.h"
#include "attribute_group/attr_group_base.h"
#include "test_structs.h"

using namespace ge;
using namespace std;

using std::vector;
using std::string;
namespace {
static ComputeGraphPtr BuildSubSubComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("subsubgraph");
    auto data = builder.AddNode("sub_sub_Data", "sub_sub_Data", 0, 1);
    auto netoutput = builder.AddNode("sub_sub_Netoutput", "sub_sub_NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    return graph;
}

static ComputeGraphPtr BuildSubComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
    auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
    auto partitioned_call = builder.AddNode("PartitionedCall", "PartitionedCall", 1, 1);
    partitioned_call->GetOpDesc()->AddSubgraphName("subsubgraph");
    partitioned_call->GetOpDesc()->SetSubgraphInstanceName(0, "subsubgraph");
    auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, partitioned_call, 0);
    builder.AddDataEdge(partitioned_call, 0, netoutput, 0);
    auto subgraph = builder.GetGraph();
    partitioned_call->SetOwnerComputeGraph(subgraph);
    ComputeGraphPtr subsubgraph = BuildSubSubComputeGraph();
    subsubgraph->SetParentGraph(subgraph);
    subsubgraph->SetParentNode(partitioned_call);
    return subgraph;
}
// construct graph which contains subgraph
static ComputeGraphPtr BuildComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto data = builder.AddNode("Data", "Data", 0, 1);
    auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
    transdata->GetOpDesc()->AddSubgraphName("subgraph");
    transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, transdata, 0);
    builder.AddDataEdge(transdata, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    // add subgraph
    transdata->SetOwnerComputeGraph(graph);
    ComputeGraphPtr subgraph = BuildSubComputeGraph();
    subgraph->SetParentGraph(graph);
    subgraph->SetParentNode(transdata);

    auto partitioned_call = subgraph->FindNode("PartitionedCall");
    auto sub_sub_graph = ge::NodeUtils::GetSubgraph(*partitioned_call, 0U);

    graph->AddSubgraph("subgraph", subgraph);
    graph->AddSubgraph("partitioned_call", sub_sub_graph);
    return graph;
}

TEST(UTEST_ge_model_serialize, GetAllSubGraphsRecursivelySuccess)
{
    Model model("model_name", "custom version3.0");
    ComputeGraphPtr cgp = BuildComputeGraph();
    model.SetGraph(cgp);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_NE(buffer.GetSize(), 0);
}
}

bool LinkEdge(NodePtr srcNode, int32_t srcIndex, NodePtr dstNode, int32_t dstIndex)
{
    if (srcIndex >= 0) {
        auto srcAnchor = srcNode->GetOutDataAnchor(srcIndex);
        auto dstAnchor = dstNode->GetInDataAnchor(dstIndex);
        srcAnchor->LinkTo(dstAnchor);
    } else {
        auto srcAnchor = srcNode->GetOutControlAnchor();
        auto dstAnchor = dstNode->GetInControlAnchor();
        srcAnchor->LinkTo(dstAnchor);
    }
    return true;
}

NodePtr CreateNode(OpDescPtr op, ComputeGraphPtr ownerGraph)
{
    return ownerGraph->AddNode(op);
}

void CompareShape(const vector<int64_t>& shape1, const vector<int64_t>& shape2)
{
    EXPECT_EQ(shape1.size(), shape2.size());
    if (shape1.size() == shape2.size()) {
        for (size_t i = 0; i < shape1.size(); i++) {
            EXPECT_EQ(shape1[i], shape2[i]);
        }
    }
}

template<typename T>
void CompareList(const vector<T>& val1, const vector<T>& val2)
{
    EXPECT_EQ(val1.size(), val2.size());
    if (val1.size() == val2.size()) {
        for (size_t i = 0; i < val1.size(); i++) {
            EXPECT_EQ(val1[i], val2[i]);
        }
    }
}

static bool NamedAttrsSimpleCmp(const GeAttrValue& left, const GeAttrValue& right)
{
    GeAttrValue::NamedAttrs val1, val2;
    left.GetValue<GeAttrValue::NamedAttrs>(val1);
    right.GetValue<GeAttrValue::NamedAttrs>(val2);
    if (val1.GetName() != val2.GetName()) {
        return false;
    }
    auto attrs1 = val1.GetAllAttrs();
    auto attrs2 = val2.GetAllAttrs();
    if (attrs1.size() != attrs1.size()) {
        return false;
    }

    for (auto it: attrs1) {
        auto it2 = attrs2.find(it.first);
        if (it2 == attrs2.end()) { // simple check
            return false;
        }
        if(it.second.GetValueType() != it2->second.GetValueType()){
            return false;
        }
        switch (it.second.GetValueType()){
            case GeAttrValue::VT_INT:{
                int64_t i1 = 0, i2 = 0;
                it.second.GetValue<GeAttrValue::INT>(i1);
                it2->second.GetValue<GeAttrValue::INT>(i2);
                if(i1 != i2){
                    return false;
                }
            }
            case GeAttrValue::VT_FLOAT:{
                GeAttrValue::FLOAT i1 = 0, i2 = 0;
                it.second.GetValue<GeAttrValue::FLOAT>(i1);
                it2->second.GetValue<GeAttrValue::FLOAT>(i2);
                if(fabs(i1 - i2) > FLT_EPSILON){
                    return false;
                }
            }
            case GeAttrValue::VT_STRING:{
                string i1, i2;
                it.second.GetValue<GeAttrValue::STR>(i1);
                it2->second.GetValue<GeAttrValue::STR>(i2);
                if(i1 != i2){
                    return false;
                }
            }
            case GeAttrValue::VT_BOOL:{
                bool i1 = false, i2 = false;
                it.second.GetValue<GeAttrValue::BOOL>(i1);
                it2->second.GetValue<GeAttrValue::BOOL>(i2);
                if(i1 != i2){
                    return false;
                }
            }
            default: {
                continue;
            }
        }
    }
    return true;
}

static GeAttrValue::NamedAttrs CreateNamedAttrs(const string& name, std::map<string, GeAttrValue> map)
{
    GeAttrValue::NamedAttrs namedAttrs;
    namedAttrs.SetName(name);
    for(auto it :map){
        namedAttrs.SetAttr(it.first, it.second);
    }
    return namedAttrs;
}

static ComputeGraphPtr CreateGraph_1_1_224_224(float *tensor_data) {
  ut::GraphBuilder builder("graph1");
  auto data1 = builder.AddNode("data1", "Data", {}, {"y"});
  AttrUtils::SetInt(data1->GetOpDesc(), "index", 0);
  auto const1 = builder.AddNode("const1", "Const", {}, {"y"});
  GeTensorDesc const1_td;
  const1_td.SetShape(GeShape({1, 1, 224, 224}));
  const1_td.SetOriginShape(GeShape({1, 1, 224, 224}));
  const1_td.SetFormat(FORMAT_NCHW);
  const1_td.SetOriginFormat(FORMAT_NCHW);
  const1_td.SetDataType(DT_FLOAT);
  const1_td.SetOriginDataType(DT_FLOAT);
  GeTensor tensor(const1_td);
  tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data), sizeof(float) * 224 * 224);
  AttrUtils::SetTensor(const1->GetOpDesc(), "value", tensor);
  auto add1 = builder.AddNode("add1", "Add", {"x1", "x2"}, {"y"});
  auto netoutput1 = builder.AddNode("NetOutputNode", "NetOutput", {"x"}, {});
  ge::AttrUtils::SetListListInt(add1->GetOpDesc()->MutableOutputDesc(0), "list_list_i", {{1, 0, 0, 0}});
  ge::AttrUtils::SetListInt(add1->GetOpDesc(), "list_i", {1});
  ge::AttrUtils::SetListStr(add1->GetOpDesc(), "list_s", {"1"});
  ge::AttrUtils::SetListFloat(add1->GetOpDesc(), "list_f", {1.0});
  ge::AttrUtils::SetListBool(add1->GetOpDesc(), "list_b", {false});
  builder.AddDataEdge(data1, 0, add1, 0);
  builder.AddDataEdge(const1, 0, add1, 1);
  builder.AddDataEdge(add1, 0, netoutput1, 0);

  return builder.GetGraph();
}

TEST(UTEST_ge_model_serialize, simple)
{
    Model model("model_name", "custom version3.0");
    model.SetAttr("model_key1", GeAttrValue::CreateFrom<GeAttrValue::INT>(123));
    model.SetAttr("model_key2", GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(456.78f));
    model.SetAttr("model_key3", GeAttrValue::CreateFrom<GeAttrValue::STR>("abcd"));
    model.SetAttr("model_key4", GeAttrValue::CreateFrom<GeAttrValue::LIST_INT>({123, 456}));
    model.SetAttr("model_key5", GeAttrValue::CreateFrom<GeAttrValue::LIST_FLOAT>({456.78f, 998.90f}));
    model.SetAttr("model_key6", GeAttrValue::CreateFrom<GeAttrValue::LIST_STR>({"abcd", "happy"}));
    model.SetAttr("model_key7", GeAttrValue::CreateFrom<GeAttrValue::BOOL>(false));
    model.SetAttr("model_key8", GeAttrValue::CreateFrom<GeAttrValue::LIST_BOOL>({true, false}));

    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto inputOp = std::make_shared<OpDesc>("input", "Input");
    inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    // w1
    auto w1Op = std::make_shared<OpDesc>("w1", "ConstOp");
    w1Op->AddOutputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
    auto w1 = CreateNode(w1Op, computeGraph);

    // node1
    auto node1Op = std::make_shared<OpDesc>("node1", "Conv2D");
    node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
    node1Op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto node1 = CreateNode(node1Op, computeGraph);

    // Attr set
    node1Op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
    node1Op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
    auto namedAttrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
            CreateNamedAttrs("my_name", {{"int_val",
                    GeAttrValue::CreateFrom<int64_t>(
                            123)},
                   {"str_val",
                    GeAttrValue::CreateFrom<string>(
                            "abc")},
                   {"float_val",
                    GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(
                            345.345)}}));

    node1Op->SetAttr("node_key3", std::move(namedAttrs1));
    auto listNamedAttrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
            {CreateNamedAttrs("my_name",
                                    {{"int_val",
                                      GeAttrValue::CreateFrom<int64_t>(
                                              123)},
                                     {"float_val",
                                      GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(
                                              345.345)}}),
             CreateNamedAttrs("my_name2",
                                    {{"str_val",
                                      GeAttrValue::CreateFrom<string>(
                                              "abc")},
                                     {"float_val",
                                      GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(
                                              345.345)}})});
    node1Op->SetAttr("node_key4", std::move(listNamedAttrs));
    // tensor
    auto tensorData1 = "qwertyui";
    auto tensor1 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8), (uint8_t*) tensorData1,
                                            8);
    auto tensorData2 = "asdfqwertyui";
    auto tensor2 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({3, 2, 2}), FORMAT_ND, DT_UINT8), (uint8_t*) tensorData2,
                                            12);
    auto tensorData3 = "ghjkasdfqwertyui";
    auto tensor3 = std::make_shared<GeTensor>(GeTensorDesc(GeShape({4, 2, 2}), FORMAT_ND, DT_UINT16), (uint8_t*) tensorData3,
                                            16);
    node1Op->SetAttr("node_key5", GeAttrValue::CreateFrom<GeAttrValue::TENSOR>(tensor1));
    node1Op->SetAttr("node_key6", GeAttrValue::CreateFrom<GeAttrValue::LIST_TENSOR>({tensor2, tensor3}));

    auto tensorDesc = GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT16);
    TensorUtils::SetSize(tensorDesc, 100);
    node1Op->SetAttr("node_key7", GeAttrValue::CreateFrom<GeAttrValue::TENSOR_DESC>(tensorDesc));
    node1Op->SetAttr("node_key8", GeAttrValue::CreateFrom<GeAttrValue::LIST_TENSOR_DESC>(
            {GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT32), GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT32),
             GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT64), GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT64),
             GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_BOOL),
             GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_DOUBLE)}));

    LinkEdge(input, 0, node1, 0);
    LinkEdge(w1, 0, node1, 1);

    model.SetGraph(computeGraph);

    Buffer buffer;
    ASSERT_EQ(model.Save(buffer), GRAPH_SUCCESS);
    EXPECT_TRUE(buffer.GetData() != nullptr);

    Model model2;
    ASSERT_EQ(Model::Load(buffer.GetData(), buffer.GetSize(), model2), GRAPH_SUCCESS);
    EXPECT_EQ(model2.GetName(), "model_name");
    GeAttrValue::INT modelVal1;
    AttrUtils::GetInt(&model2, "model_key1", modelVal1);
    EXPECT_EQ(modelVal1, 123);

    GeAttrValue::FLOAT modelVal2;
    AttrUtils::GetFloat(&model2, "model_key2", modelVal2);
    EXPECT_EQ(modelVal2, (float) 456.78f);

    GeAttrValue::STR modelVal3;
    AttrUtils::GetStr(&model2, "model_key3", modelVal3);
    EXPECT_EQ(modelVal3, "abcd");

    GeAttrValue::LIST_INT modelVal4;
    AttrUtils::GetListInt(&model2, "model_key4", modelVal4);
    CompareList(modelVal4, {123, 456});

    GeAttrValue::LIST_FLOAT modelVal5;
    AttrUtils::GetListFloat(&model2, "model_key5", modelVal5);
    CompareList(modelVal5, {456.78f, 998.90f});

    GeAttrValue::LIST_STR modelVal6;
    AttrUtils::GetListStr(&model2, "model_key6", modelVal6);
    CompareList(modelVal6, {"abcd", "happy"});

    GeAttrValue::BOOL modelVal7;
    EXPECT_EQ(AttrUtils::GetBool(&model2, "model_key7", modelVal7), true);
    EXPECT_EQ(modelVal7, false);

    GeAttrValue::LIST_BOOL modelVal8;
    AttrUtils::GetListBool(&model2, "model_key8", modelVal8);
    CompareList(modelVal8, {true, false});

    const auto& s_graph = model2.GetGraph();
    ASSERT_TRUE(s_graph != nullptr);
    auto s_nodes = s_graph->GetDirectNode();
    ASSERT_EQ(3, s_nodes.size());

    auto s_input = s_nodes.at(0);
    auto s_w1 = s_nodes.at(1);
    auto s_nod1 = s_nodes.at(2);
    {
        auto s_op = s_input->GetOpDesc();
        EXPECT_EQ(s_op->GetName(), "input");
        EXPECT_EQ(s_op->GetType(), "Input");
        auto s_input_descs = s_op->GetAllInputsDesc();
        ASSERT_EQ(s_input_descs.size(), 0);
        auto s_output_descs = s_op->GetAllOutputsDesc();
        ASSERT_EQ(s_output_descs.size(), 1);
        auto desc1 = s_output_descs.at(0);
        EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
        EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
        CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

        auto outAnchor = s_input->GetOutDataAnchor(0);
        auto peerAnchors = outAnchor->GetPeerInDataAnchors();
        ASSERT_EQ(peerAnchors.size(), 1);
        auto peerAnchor = peerAnchors.at(0);
        ASSERT_EQ(peerAnchor->GetIdx(), 0);
        ASSERT_EQ(peerAnchor->GetOwnerNode(), s_nod1);
    }

    {
        auto s_op = s_w1->GetOpDesc();
        EXPECT_EQ(s_op->GetName(), "w1");
        EXPECT_EQ(s_op->GetType(), "ConstOp");
        auto s_input_descs = s_op->GetAllInputsDesc();
        ASSERT_EQ(s_input_descs.size(), 0);
        auto s_output_descs = s_op->GetAllOutputsDesc();
        ASSERT_EQ(s_output_descs.size(), 1);
        auto desc1 = s_output_descs.at(0);
        EXPECT_EQ(desc1.GetFormat(), FORMAT_NC1HWC0);
        EXPECT_EQ(desc1.GetDataType(), DT_FLOAT16);
        CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

        auto outAnchor = s_w1->GetOutDataAnchor(0);
        auto peerAnchors = outAnchor->GetPeerInDataAnchors();
        ASSERT_EQ(peerAnchors.size(), 1);
        auto peerAnchor = peerAnchors.at(0);
        ASSERT_EQ(peerAnchor->GetIdx(), 1);
        ASSERT_EQ(peerAnchor->GetOwnerNode(), s_nod1);
    }
    {
        auto s_op = s_nod1->GetOpDesc();
        EXPECT_EQ(s_op->GetName(), "node1");
        EXPECT_EQ(s_op->GetType(), "Conv2D");
        auto s_input_descs = s_op->GetAllInputsDesc();
        ASSERT_EQ(s_input_descs.size(), 2);

        auto desc1 = s_input_descs.at(0);
        EXPECT_EQ(desc1.GetFormat(), FORMAT_NCHW);
        EXPECT_EQ(desc1.GetDataType(), DT_FLOAT);
        CompareShape(desc1.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

        auto desc2 = s_input_descs.at(1);
        EXPECT_EQ(desc2.GetFormat(), FORMAT_NC1HWC0);
        EXPECT_EQ(desc2.GetDataType(), DT_FLOAT16);
        CompareShape(desc2.GetShape().GetDims(), vector<int64_t>{12, 2, 64, 64, 16});

        auto s_output_descs = s_op->GetAllOutputsDesc();
        ASSERT_EQ(s_output_descs.size(), 1);
        auto desc3 = s_output_descs.at(0);
        EXPECT_EQ(desc3.GetFormat(), FORMAT_NCHW);
        EXPECT_EQ(desc3.GetDataType(), DT_FLOAT);
        CompareShape(desc3.GetShape().GetDims(), vector<int64_t>{12, 32, 64, 64});

        auto outAnchor = s_nod1->GetOutDataAnchor(0);
        auto peerAnchors = outAnchor->GetPeerInDataAnchors();
        ASSERT_EQ(peerAnchors.size(), 0);

        // node attrs
        GeAttrValue::BYTES nodeVal1;
        AttrUtils::GetBytes(s_op, "node_key1", nodeVal1);
        ASSERT_EQ(nodeVal1.GetSize(), 10);

        GeAttrValue::LIST_BYTES nodeVal2;
        AttrUtils::GetListBytes(s_op, "node_key2", nodeVal2);
        ASSERT_EQ(nodeVal2.size(), 2);
        ASSERT_EQ(nodeVal2[0].GetSize(), 20);
        ASSERT_EQ(nodeVal2[1].GetSize(), 30);

        GeAttrValue s_namedAttrs;
        s_op->GetAttr("node_key3", s_namedAttrs);
        EXPECT_TRUE(NamedAttrsSimpleCmp(s_namedAttrs, namedAttrs1));

        GeAttrValue s_listNamedAttrs;
        s_op->GetAttr("node_key4", s_listNamedAttrs);
        EXPECT_TRUE(NamedAttrsSimpleCmp(s_listNamedAttrs, listNamedAttrs));

        ConstGeTensorPtr s_tensor;
        AttrUtils::GetTensor(s_op, "node_key5", s_tensor);
        ASSERT_TRUE(s_tensor == nullptr);

        GeTensorDesc s_tensorDesc;
        AttrUtils::GetTensorDesc(s_op, "node_key7", s_tensorDesc);
        EXPECT_EQ(s_tensorDesc.GetFormat(), FORMAT_ND);
        EXPECT_EQ(s_tensorDesc.GetDataType(), DT_FLOAT);
        int64_t size = -1;
        TensorUtils::GetSize(s_tensorDesc, size);
        EXPECT_EQ(size, 0);
    }
}

TEST(UTEST_ge_model_serialize, OpDescAsAttrValue)
{
    // node1Op
    auto node1Op = std::make_shared<OpDesc>("node1", "Conv2D");
    node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    node1Op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
    node1Op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    // Attr set
    node1Op->SetAttr("node_key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(Buffer(10)));
    node1Op->SetAttr("node_key2", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({Buffer(20), Buffer(30)}));
    auto namedAttrs1 = GeAttrValue::CreateFrom<GeAttrValue::NAMED_ATTRS>(
            CreateNamedAttrs("my_name", {{"int_val",
                                               GeAttrValue::CreateFrom<int64_t>(
                                                       123)},
                                              {"str_val",
                                               GeAttrValue::CreateFrom<string>(
                                                       "abc")},
                                              {"float_val",
                                               GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(
                                                       345.345)}}));

    node1Op->SetAttr("node_key3", std::move(namedAttrs1));
    auto listNamedAttrs = GeAttrValue::CreateFrom<GeAttrValue::LIST_NAMED_ATTRS>(
            {CreateNamedAttrs("my_name",
                                   {{"int_val",
                                     GeAttrValue::CreateFrom<int64_t>(
                                             123)},
                                    {"float_val",
                                     GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(
                                             345.345)}}),
             CreateNamedAttrs("my_name2",
                                   {{"str_val",
                                     GeAttrValue::CreateFrom<string>(
                                             "abc")},
                                    {"float_val",
                                     GeAttrValue::CreateFrom<GeAttrValue::FLOAT>(
                                             345.345)}})});
    node1Op->SetAttr("node_key4", std::move(listNamedAttrs));


    Model model;
    EXPECT_TRUE(AttrUtils::SetListInt(&model, "my_key2",{123}));
    EXPECT_TRUE(AttrUtils::SetListBytes(&model, "my_key3",{Buffer(100)}));
}

TEST(UTEST_ge_model_serialize, test_subGraph)
{
  Model model("model_name", "custom version3.0");
  {
    auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
    inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(inputOp, computeGraph);
    model.SetGraph(computeGraph);

    auto subComputeGraph = std::make_shared<ComputeGraph>("sub_graph");
    // input
    auto subGraphInputOp = std::make_shared<OpDesc>("sub_graph_test", "TestOp2");
    subGraphInputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto subGraphInput = CreateNode(subGraphInputOp, subComputeGraph);

    AttrUtils::SetGraph(inputOp, "sub_graph", subComputeGraph);
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
}

REG_OP(MatMul)
  .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT32}))
  .ATTR(transpose_x1, Bool, false)
  .ATTR(transpose_x2, Bool, false)
  .OP_END_FACTORY_REG(MatMul)

TEST(UTEST_ge_model_serialize, test_ir_definitions)
{
  Model model("model_name", "custom version3.0");
  auto op = ge::OperatorFactory::CreateOperator("MatMul", "MatMul");
  auto op_desc_origin = ge::OpDescUtils::GetOpDescFromOperator(op);
  EXPECT_NE(op_desc_origin, nullptr);
  auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
  CreateNode(op_desc_origin, computeGraph);
  model.SetGraph(computeGraph);
  auto graph = GraphUtilsEx::CreateGraphFromComputeGraph(computeGraph);
  auto node = graph.GetDirectNode();

  ModelSerialize serialize;
  proto::ModelDef model_def;
  ASSERT_EQ(serialize.SerializeModel(model, false, model_def), SUCCESS); //success
  GraphUtils::WriteProtoToTextFile(model_def, "./ir_definitions.txt");
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  auto state = GraphUtils::LoadGEGraph("./ir_definitions.txt", *com_graph1);
  ASSERT_EQ(state, true);
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 1);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == "MatMul") {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      EXPECT_FALSE(op_desc->GetIrAttrNames().empty());
      EXPECT_FALSE(op_desc->GetIrInputs().empty());
      EXPECT_EQ(op_desc->GetIrAttrNames().size(), op_desc_origin->GetIrAttrNames().size());
      EXPECT_EQ(op_desc->GetIrInputs().size(), op_desc_origin->GetIrInputs().size());
    }
  }
  system("rm -rf ./ir_definitions.txt");
}

TEST(UTEST_ge_model_serialize, test_large_model)
{
  Model model("model_name/main_model", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name/main_graph");
    // input
    auto input_op = std::make_shared<OpDesc>("test/const", CONSTANT);
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    GeTensor ge_tensor;
    auto aligned_ptr = std::make_shared<AlignedPtr>(4294967296U);  // 4g
    auto ptr = aligned_ptr->MutableGet();
    *ptr = 7;
    *(ptr + 10) = 8;
    *(ptr + 4294967295) = 9;
    ge_tensor.SetData(aligned_ptr, 4294967296);
    AttrUtils::SetTensor(input_op, ATTR_NAME_WEIGHTS, ge_tensor);
    model.SetGraph(compute_graph);

    auto sub_compute_graph = std::make_shared<ComputeGraph>("graph_name/sub_graph");
    auto sub_graph_input_op = std::make_shared<OpDesc>("sub_graph_test", "TestOp2");
    sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input = CreateNode(sub_graph_input_op, sub_compute_graph);

    std::string sub_graph = "graph_name/sub_graph";
    input_op->AddSubgraphName(sub_graph);
    input_op->SetSubgraphInstanceName(0, sub_graph);
    sub_compute_graph->SetParentNode(input);
    sub_compute_graph->SetParentGraph(compute_graph);
    compute_graph->AddSubgraph(sub_compute_graph);
  }

  ModelSerialize serialize;
  Model model_back;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_NE(buffer.GetSize(), 0);// failed
  proto::ModelDef model_def;
  ASSERT_EQ(serialize.UnserializeModel(buffer.GetData(), buffer.GetSize(), model_back), true);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 2);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), true);
      ASSERT_EQ((ge_tensor_ptr == nullptr), false);
      const TensorData tensor_data = ge_tensor_ptr->GetData();
      const uint8_t *buff = tensor_data.GetData();
      ASSERT_EQ((buff == nullptr), false);
      ASSERT_EQ(buff[0], 7);
      ASSERT_EQ(buff[10], 8);
      ASSERT_EQ(buff[4294967295], 9); // value is ok for def serialize
    }
  }
  system("rm -rf ./air_weight");
}

TEST(UTEST_ge_model_serialize, test_large_model_with_16_subgraph_multi_thread)
{
  Model model("model_name/main_model", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("main_graph");
    // input
    auto input1 = std::make_shared<OpDesc>("input1", "Data");
    auto input2 = std::make_shared<OpDesc>("input2", "Data");
    auto input3 = std::make_shared<OpDesc>("input3", "Data");
    input1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto input_data1 = CreateNode(input1, compute_graph);
    auto input_data2 = CreateNode(input2, compute_graph);
    auto input_data3 = CreateNode(input3, compute_graph);
    auto case_op = std::make_shared<OpDesc>("case", "Case");
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto case_node = CreateNode(case_op, compute_graph);
    auto output_op = std::make_shared<OpDesc>("output", "NetOutput");
    output_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto output_node = CreateNode(output_op, compute_graph);
    ge::GraphUtils::AddEdge(input_data1->GetOutAnchor(0), case_node->GetInAnchor(0));
    ge::GraphUtils::AddEdge(input_data2->GetOutAnchor(0), case_node->GetInAnchor(1));
    ge::GraphUtils::AddEdge(input_data3->GetOutAnchor(0), case_node->GetInAnchor(2));
    ge::GraphUtils::AddEdge(case_node->GetOutAnchor(0), output_node->GetInAnchor(0));
    model.SetGraph(compute_graph);
    for (auto i = 0UL; i < 16; i++) {
        std::string subgraph_name = "subgraph" + std::to_string(i);
        auto sub_compute_graph = std::make_shared<ComputeGraph>(subgraph_name);
        auto sub_graph_input_op = std::make_shared<OpDesc>("data1", "DATA");
        sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_graph_input_op1 = std::make_shared<OpDesc>("data2", "DATA");
        sub_graph_input_op1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_input_node = CreateNode(sub_graph_input_op, sub_compute_graph);
        auto sub_input_node1 = CreateNode(sub_graph_input_op1, sub_compute_graph);
        auto sub_graph_add_op = std::make_shared<OpDesc>("add_sub", "ADD");
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_add_node = CreateNode(sub_graph_add_op, sub_compute_graph);
        auto sub_output_op = std::make_shared<OpDesc>("sub_output", "NetOutput");
        sub_output_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_output_node = CreateNode(sub_output_op, sub_compute_graph);
        ge::GraphUtils::AddEdge(sub_input_node->GetOutAnchor(0), sub_add_node->GetInAnchor(0));
        ge::GraphUtils::AddEdge(sub_input_node1->GetOutAnchor(0), sub_add_node->GetInAnchor(1));
        ge::GraphUtils::AddEdge(sub_add_node->GetOutAnchor(0), sub_output_node->GetInAnchor(0));
        case_op->AddSubgraphName(subgraph_name);
        case_op->SetSubgraphInstanceName(i, subgraph_name);
        sub_compute_graph->SetParentNode(case_node);
        sub_compute_graph->SetParentGraph(compute_graph);
        compute_graph->AddSubgraph(sub_compute_graph);
    }
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_NE(buffer.GetSize(), 0);// failed
  Model model_back;
  ASSERT_EQ(Model::LoadWithMultiThread(buffer.GetData(), buffer.GetSize(), model_back), SUCCESS);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllSubgraphs().size(), 16);
  for (auto &sub_graph : com_graph1->GetAllSubgraphs()) {
    ASSERT_EQ((sub_graph == nullptr), false);
    EXPECT_EQ(sub_graph->GetDirectNodesSize(), 4);
    for (auto &nodes : sub_graph->GetDirectNode()) {
       ASSERT_EQ((nodes == nullptr), false);
    }
  }
}

TEST(UTEST_ge_model_serialize, test_large_model_with_16_subgraph_single_thread)
{
  Model model("model_name/main_model", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("main_graph");
    // input
    auto input1 = std::make_shared<OpDesc>("input1", "Data");
    auto input2 = std::make_shared<OpDesc>("input2", "Data");
    auto input3 = std::make_shared<OpDesc>("input3", "Data");
    input1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto input_data1 = CreateNode(input1, compute_graph);
    auto input_data2 = CreateNode(input2, compute_graph);
    auto input_data3 = CreateNode(input3, compute_graph);
    auto case_op = std::make_shared<OpDesc>("case", "Case");
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto case_node = CreateNode(case_op, compute_graph);
    auto output_op = std::make_shared<OpDesc>("output", "NetOutput");
    output_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto output_node = CreateNode(output_op, compute_graph);
    ge::GraphUtils::AddEdge(input_data1->GetOutAnchor(0), case_node->GetInAnchor(0));
    ge::GraphUtils::AddEdge(input_data2->GetOutAnchor(0), case_node->GetInAnchor(1));
    ge::GraphUtils::AddEdge(input_data3->GetOutAnchor(0), case_node->GetInAnchor(2));
    ge::GraphUtils::AddEdge(case_node->GetOutAnchor(0), output_node->GetInAnchor(0));
    model.SetGraph(compute_graph);
    for (auto i = 0UL; i < 16; i++) {
        std::string subgraph_name = "subgraph" + std::to_string(i);
        auto sub_compute_graph = std::make_shared<ComputeGraph>(subgraph_name);
        auto sub_graph_input_op = std::make_shared<OpDesc>("data1", "DATA");
        sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_graph_input_op1 = std::make_shared<OpDesc>("data2", "DATA");
        sub_graph_input_op1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_input_node = CreateNode(sub_graph_input_op, sub_compute_graph);
        auto sub_input_node1 = CreateNode(sub_graph_input_op1, sub_compute_graph);
        auto sub_graph_add_op = std::make_shared<OpDesc>("add_sub", "ADD");
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_add_node = CreateNode(sub_graph_add_op, sub_compute_graph);
        auto sub_output_op = std::make_shared<OpDesc>("sub_output", "NetOutput");
        sub_output_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_output_node = CreateNode(sub_output_op, sub_compute_graph);
        ge::GraphUtils::AddEdge(sub_input_node->GetOutAnchor(0), sub_add_node->GetInAnchor(0));
        ge::GraphUtils::AddEdge(sub_input_node1->GetOutAnchor(0), sub_add_node->GetInAnchor(1));
        ge::GraphUtils::AddEdge(sub_add_node->GetOutAnchor(0), sub_output_node->GetInAnchor(0));
        case_op->AddSubgraphName(subgraph_name);
        case_op->SetSubgraphInstanceName(i, subgraph_name);
        sub_compute_graph->SetParentNode(case_node);
        sub_compute_graph->SetParentGraph(compute_graph);
        compute_graph->AddSubgraph(sub_compute_graph);
    }
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_NE(buffer.GetSize(), 0);// failed
  Model model_back;
  ASSERT_EQ(Model::Load(buffer.GetData(), buffer.GetSize(), model_back), SUCCESS);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllSubgraphs().size(), 16);
  for (auto &sub_graph : com_graph1->GetAllSubgraphs()) {
    ASSERT_EQ((sub_graph == nullptr), false);
    EXPECT_EQ(sub_graph->GetDirectNodesSize(), 4);
    for (auto &nodes : sub_graph->GetDirectNode()) {
       ASSERT_EQ((nodes == nullptr), false);
    }
  }
}

TEST(UTEST_ge_model_serialize, test_large_model_with_30_subgraph)
{
  Model model("model_name/main_model", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("main_graph");
    // input
    auto input1 = std::make_shared<OpDesc>("input1", "Data");
    auto input2 = std::make_shared<OpDesc>("input2", "Data");
    auto input3 = std::make_shared<OpDesc>("input3", "Data");
    input1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto input_data1 = CreateNode(input1, compute_graph);
    auto input_data2 = CreateNode(input2, compute_graph);
    auto input_data3 = CreateNode(input3, compute_graph);
    auto case_op = std::make_shared<OpDesc>("case", "Case");
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto case_node = CreateNode(case_op, compute_graph);
    auto output_op = std::make_shared<OpDesc>("output", "NetOutput");
    output_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto output_node = CreateNode(output_op, compute_graph);
    ge::GraphUtils::AddEdge(input_data1->GetOutAnchor(0), case_node->GetInAnchor(0));
    ge::GraphUtils::AddEdge(input_data2->GetOutAnchor(0), case_node->GetInAnchor(1));
    ge::GraphUtils::AddEdge(input_data3->GetOutAnchor(0), case_node->GetInAnchor(2));
    ge::GraphUtils::AddEdge(case_node->GetOutAnchor(0), output_node->GetInAnchor(0));
    model.SetGraph(compute_graph);
    for (auto i = 0UL; i < 30; i++) {
        std::string subgraph_name = "subgraph" + std::to_string(i);
        auto sub_compute_graph = std::make_shared<ComputeGraph>(subgraph_name);
        auto sub_graph_input_op = std::make_shared<OpDesc>("data1", "DATA");
        sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_graph_input_op1 = std::make_shared<OpDesc>("data2", "DATA");
        sub_graph_input_op1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_input_node = CreateNode(sub_graph_input_op, sub_compute_graph);
        auto sub_input_node1 = CreateNode(sub_graph_input_op1, sub_compute_graph);
        auto sub_graph_add_op = std::make_shared<OpDesc>("add_sub", "ADD");
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_add_node = CreateNode(sub_graph_add_op, sub_compute_graph);
        auto sub_output_op = std::make_shared<OpDesc>("sub_output", "NetOutput");
        sub_output_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_output_node = CreateNode(sub_output_op, sub_compute_graph);
        ge::GraphUtils::AddEdge(sub_input_node->GetOutAnchor(0), sub_add_node->GetInAnchor(0));
        ge::GraphUtils::AddEdge(sub_input_node1->GetOutAnchor(0), sub_add_node->GetInAnchor(1));
        ge::GraphUtils::AddEdge(sub_add_node->GetOutAnchor(0), sub_output_node->GetInAnchor(0));
        case_op->AddSubgraphName(subgraph_name);
        case_op->SetSubgraphInstanceName(i, subgraph_name);
        sub_compute_graph->SetParentNode(case_node);
        sub_compute_graph->SetParentGraph(compute_graph);
        compute_graph->AddSubgraph(sub_compute_graph);
    }
  }

  ModelSerialize serialize;
  Model model_back;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_NE(buffer.GetSize(), 0);// failed
  proto::ModelDef model_def;
  ASSERT_EQ(serialize.UnserializeModel(buffer.GetData(), buffer.GetSize(), model_back), true);
  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllSubgraphs().size(), 30);
  for (auto &sub_graph : com_graph1->GetAllSubgraphs()) {
    ASSERT_EQ((sub_graph == nullptr), false);
    EXPECT_EQ(sub_graph->GetDirectNodesSize(), 4);
    for (auto &nodes : sub_graph->GetDirectNode()) {
       ASSERT_EQ((nodes == nullptr), false);
    }
  }
}

TEST(UTEST_ge_model_serialize, test_large_model_with_subgraph_error)
{
  Model model("model_name/main_model", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("main_graph");
    // input
    auto input1 = std::make_shared<OpDesc>("input1", "Data");
    auto input2 = std::make_shared<OpDesc>("input2", "Data");
    auto input3 = std::make_shared<OpDesc>("input3", "Data");
    input1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input2->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    input3->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto input_data1 = CreateNode(input1, compute_graph);
    auto input_data2 = CreateNode(input2, compute_graph);
    auto input_data3 = CreateNode(input3, compute_graph);
    auto case_op = std::make_shared<OpDesc>("case", "Case");
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    case_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto case_node = CreateNode(case_op, compute_graph);
    auto output_op = std::make_shared<OpDesc>("output", "NetOutput");
    output_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
    auto output_node = CreateNode(output_op, compute_graph);
    ge::GraphUtils::AddEdge(input_data1->GetOutAnchor(0), case_node->GetInAnchor(0));
    ge::GraphUtils::AddEdge(input_data2->GetOutAnchor(0), case_node->GetInAnchor(1));
    ge::GraphUtils::AddEdge(input_data3->GetOutAnchor(0), case_node->GetInAnchor(2));
    ge::GraphUtils::AddEdge(case_node->GetOutAnchor(0), output_node->GetInAnchor(0));
    model.SetGraph(compute_graph);
    for (auto i = 0UL; i < 3; i++) {
        std::string subgraph_name = "subgraph" + std::to_string(i);
        auto sub_compute_graph = std::make_shared<ComputeGraph>(subgraph_name);
        auto sub_graph_input_op = std::make_shared<OpDesc>("data1", "DATA");
        sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_graph_input_op1 = std::make_shared<OpDesc>("subgraph/const1", CONSTANT);
        sub_graph_input_op1->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_input_op1->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        GeTensor ge_tensor;
        auto aligned_ptr = std::make_shared<AlignedPtr>(64);  // 500m
        auto ptr = aligned_ptr->MutableGet();
        *ptr = 7;
        *(ptr + 10) = 8;
        ge_tensor.SetData(aligned_ptr, 64);
        AttrUtils::SetTensor(sub_graph_input_op1, ATTR_NAME_WEIGHTS, ge_tensor);
        AttrUtils::SetStr(sub_graph_input_op1, ATTR_NAME_LOCATION, "file_path");
        AttrUtils::SetInt(sub_graph_input_op1, ATTR_NAME_LENGTH, 20);

        auto sub_input_node = CreateNode(sub_graph_input_op, sub_compute_graph);
        auto sub_input_node1 = CreateNode(sub_graph_input_op1, sub_compute_graph);
        auto sub_graph_add_op = std::make_shared<OpDesc>("add_sub", "ADD");
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        sub_graph_add_op->AddOutputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_add_node = CreateNode(sub_graph_add_op, sub_compute_graph);
        auto sub_output_op = std::make_shared<OpDesc>("sub_output", "NetOutput");
        sub_output_op->AddInputDesc(GeTensorDesc(GeShape({1, 3, 2, 2}), FORMAT_NCHW, DT_FLOAT));
        auto sub_output_node = CreateNode(sub_output_op, sub_compute_graph);
        ge::GraphUtils::AddEdge(sub_input_node->GetOutAnchor(0), sub_add_node->GetInAnchor(0));
        ge::GraphUtils::AddEdge(sub_input_node1->GetOutAnchor(0), sub_add_node->GetInAnchor(1));
        ge::GraphUtils::AddEdge(sub_add_node->GetOutAnchor(0), sub_output_node->GetInAnchor(0));
        case_op->AddSubgraphName(subgraph_name);
        case_op->SetSubgraphInstanceName(i, subgraph_name);
        sub_compute_graph->SetParentNode(case_node);
        sub_compute_graph->SetParentGraph(compute_graph);
        compute_graph->AddSubgraph(sub_compute_graph);
    }
  }

  ModelSerialize serialize;
  Model model_back;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_NE(buffer.GetSize(), 0);// failed
  proto::ModelDef model_def;
  ASSERT_EQ(serialize.UnserializeModel(buffer.GetData(), buffer.GetSize(), model_back), false);
}

TEST(UTEST_ge_model_serialize, test_large_model_lots_const)
{
  Model model("model_name/main_model", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name/main_graph");
    // input
    for (int i = 0; i < 4; i++) {
      std::string inpu_node_name = "test/const" + std::to_string(i);
      auto input_op = std::make_shared<OpDesc>(inpu_node_name, CONSTANT);
      input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
      auto input = CreateNode(input_op, compute_graph);
      GeTensor ge_tensor;
      auto aligned_ptr = std::make_shared<AlignedPtr>(536870912U);  // 500m
      auto ptr = aligned_ptr->MutableGet();
      *ptr = 7;
      *(ptr + 10) = 8;
      *(ptr + 536870910) = 9;
      ge_tensor.SetData(aligned_ptr, 536870912);
      AttrUtils::SetTensor(input_op, ATTR_NAME_WEIGHTS, ge_tensor);
    }

    model.SetGraph(compute_graph);

    auto sub_compute_graph = std::make_shared<ComputeGraph>("sub_graph");
    auto sub_graph_input_op = std::make_shared<OpDesc>("sub_graph_test", "TestOp2");
    sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input = CreateNode(sub_graph_input_op, sub_compute_graph);

    auto parent_input_op = std::make_shared<OpDesc>("parenttest", "TestOp2");
    parent_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto parent_input = CreateNode(parent_input_op, compute_graph);
    for (int i = 0; i < 4; i++) {
      std::string inpu_node_name = "subgraph/const" + std::to_string(i);
      auto sub_input_op = std::make_shared<OpDesc>(inpu_node_name, CONSTANT);
      sub_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
      auto sub_input = CreateNode(sub_input_op, sub_compute_graph);
      GeTensor ge_tensor;
      auto aligned_ptr = std::make_shared<AlignedPtr>(536870912U);  // 500m
      auto ptr = aligned_ptr->MutableGet();
      *ptr = 7;
      *(ptr + 10) = 8;
      *(ptr + 536870910) = 9;
      ge_tensor.SetData(aligned_ptr, 536870912);
      AttrUtils::SetTensor(sub_input_op, ATTR_NAME_WEIGHTS, ge_tensor);
    }
    std::string sub_graph = "sub_graph";
    parent_input_op->AddSubgraphName(sub_graph);
    parent_input_op->SetSubgraphInstanceName(0, sub_graph);
    sub_compute_graph->SetParentNode(parent_input);
    sub_compute_graph->SetParentGraph(compute_graph);
    compute_graph->AddSubgraph(sub_compute_graph);
  }
  ModelSerialize serialize;
  Model model_back;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_NE(buffer.GetSize(), 0);// failed
  proto::ModelDef model_def;
  ASSERT_EQ(serialize.UnserializeModel(buffer.GetData(), buffer.GetSize(), model_back), true);

  ComputeGraphPtr com_graph1 = std::make_shared<ComputeGraph>("TestGraph1");
  com_graph1 = model_back.GetGraph();
  ASSERT_EQ(com_graph1->GetAllNodesSize(), 10);
  for (auto &node_ptr : com_graph1->GetAllNodes()) {
    ASSERT_EQ((node_ptr == nullptr), false);
    if (node_ptr->GetType() == CONSTANT) {
      auto op_desc = node_ptr->GetOpDesc();
      ASSERT_EQ((op_desc == nullptr), false);
      ConstGeTensorPtr ge_tensor_ptr;
      ASSERT_EQ(AttrUtils::GetTensor(op_desc, ATTR_NAME_WEIGHTS, ge_tensor_ptr), true);
      ASSERT_EQ((ge_tensor_ptr == nullptr), false);
      const TensorData tensor_data = ge_tensor_ptr->GetData();
      const uint8_t *buff = tensor_data.GetData();
      ASSERT_EQ((buff == nullptr), false);
      ASSERT_EQ(buff[0], 7);
      ASSERT_EQ(buff[10], 8);
      ASSERT_EQ(buff[536870910], 9); // value is ok for def serialize
    }
  }
  auto sub_graph = com_graph1->GetSubgraph("sub_graph");
  ASSERT_EQ((sub_graph == nullptr), false);
  ASSERT_EQ(sub_graph->GetAllNodesSize(), 5);
  system("rm -rf ./air_weight");
}

TEST(UTEST_ge_model_serialize, test_listSubGraph)
{
    Model model("model_name", "custom version3.0");
    {
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
        auto input = CreateNode(inputOp, computeGraph);
        model.SetGraph(computeGraph);

        auto subComputeGraph1 = std::make_shared<ComputeGraph>("sub_graph1");
        // input
        auto subGraphInputOp1 = std::make_shared<OpDesc>("sub_graph_test1", "TestOp2");
        subGraphInputOp1->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
        auto subGraphInput1 = CreateNode(subGraphInputOp1, subComputeGraph1);

        auto subComputeGraph2 = std::make_shared<ComputeGraph>("sub_graph2");
        // input
        auto subGraphInputOp2 = std::make_shared<OpDesc>("sub_graph_test2", "TestOp2");
        subGraphInputOp2->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
        auto subGraphInput2 = CreateNode(subGraphInputOp2, subComputeGraph2);

        AttrUtils::SetListGraph(inputOp, "sub_graph", vector<ComputeGraphPtr>{subComputeGraph1, subComputeGraph2});
    }

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
}

TEST(UTEST_ge_model_serialize, test_Format)
{
    Model model("model_name", "custom version3.0");
    {
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_ND, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1HWC0, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1C0HWPAD, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC1C0, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FSR_NCHW, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_DECONV, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_BN_WEIGHT, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_CHWN, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FILTER_HWCK, DT_FLOAT));
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z_C04, DT_FLOAT));
        auto input = CreateNode(inputOp, computeGraph);
        model.SetGraph(computeGraph);
    }
    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
}

TEST(UTEST_ge_model_serialize, test_ControlEdge)
{
    Model model("model_name", "custom version3.0");
    {
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");
        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
        auto input = CreateNode(inputOp, computeGraph);
        // sink
        auto sinkOp = std::make_shared<OpDesc>("test2", "Sink");
        auto sink = CreateNode(sinkOp, computeGraph);
        LinkEdge(sink, -1, input, -1);

        // sink2
        auto sinkOp2 = std::make_shared<OpDesc>("test3", "Sink");
        auto sink2 = CreateNode(sinkOp2, computeGraph);
        LinkEdge(sink2, -1, input, -1);

        // dest
        auto destOp = std::make_shared<OpDesc>("test4", "Dest");
        auto dest = CreateNode(destOp, computeGraph);
        LinkEdge(input, -1, dest, -1);

        computeGraph->AddInputNode(sink);
        computeGraph->AddInputNode(sink2);
        computeGraph->AddOutputNode(dest);

        model.SetGraph(computeGraph);
    }
    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
}

TEST(UTEST_ge_model_serialize, test_invalid_Model)
{
    {// empty graph
        Model model("model_name", "custom version3.0");
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

        ModelSerialize serialize;
        auto buffer = serialize.SerializeModel(model);
        EXPECT_EQ(buffer.GetSize(), 0);
    }
}

TEST(UTEST_ge_model_serialize, test_invalid_Attrs)
{
    {// valid test
        Model model("model_name", "custom version3.0");
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

        GeAttrValue::NamedAttrs namedAttrs;
        namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::INT>(10));
        AttrUtils::SetNamedAttrs(inputOp, "key", namedAttrs);

        auto input = CreateNode(inputOp, computeGraph);
        model.SetGraph(computeGraph);

        ModelSerialize serialize;
        auto buffer = serialize.SerializeModel(model);
        EXPECT_GE(buffer.GetSize(), 0);
    }
    {// none type
        Model model("model_name", "custom version3.0");
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

        GeAttrValue::NamedAttrs namedAttrs;
        EXPECT_EQ(namedAttrs.SetAttr("key1", GeAttrValue()), GRAPH_FAILED);
    }
    {// bytes attr len is 0
        Model model("model_name", "custom version3.0");
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

        GeAttrValue::NamedAttrs namedAttrs;
        namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::BYTES>(GeAttrValue::BYTES(0)));
        AttrUtils::SetNamedAttrs(inputOp, "key", namedAttrs);

        auto input = CreateNode(inputOp, computeGraph);
        model.SetGraph(computeGraph);

        ModelSerialize serialize;
        auto buffer = serialize.SerializeModel(model);
        EXPECT_GE(buffer.GetSize(), 0);
    }
    {// invalid list bytes attr
        Model model("model_name", "custom version3.0");
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

        GeAttrValue::NamedAttrs namedAttrs;
        namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::LIST_BYTES>({GeAttrValue::BYTES(0)}));
        AttrUtils::SetNamedAttrs(inputOp, "key", namedAttrs);

        auto input = CreateNode(inputOp, computeGraph);
        model.SetGraph(computeGraph);

        ModelSerialize serialize;
        auto buffer = serialize.SerializeModel(model);
        EXPECT_GE(buffer.GetSize(), 0);
    }
    {// invalid graph attr
        Model model("model_name", "custom version3.0");
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

        GeAttrValue::NamedAttrs namedAttrs;
        EXPECT_EQ(namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::GRAPH>(nullptr)), GRAPH_SUCCESS);
        GeAttrValue value;
        EXPECT_EQ(namedAttrs.GetAttr("key1", value), GRAPH_SUCCESS);
        EXPECT_FALSE(value.IsEmpty());
    }
    {// invalid list graph attr
        Model model("model_name", "custom version3.0");
        auto computeGraph = std::make_shared<ComputeGraph>("graph_name");

        // input
        auto inputOp = std::make_shared<OpDesc>("test", "TestOp");
        inputOp->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

        GeAttrValue::NamedAttrs namedAttrs;
        EXPECT_EQ(namedAttrs.SetAttr("key1", GeAttrValue::CreateFrom<GeAttrValue::LIST_GRAPH>({nullptr})), GRAPH_SUCCESS);
        GeAttrValue value;
        EXPECT_EQ(namedAttrs.GetAttr("key1", value), GRAPH_SUCCESS);
        EXPECT_FALSE(value.IsEmpty());
    }
}

TEST(UTEST_ge_model_serialize, test_ModelSerializeImp_Invalid_Param)
{
    ModelSerializeImp imp;
    EXPECT_FALSE(imp.SerializeModel(Model(), nullptr));
    EXPECT_FALSE(imp.SerializeNode(nullptr, nullptr));

    auto graph = std::make_shared<ComputeGraph>("test_graph");
    auto node = graph->AddNode(std::make_shared<OpDesc>());
    node->GetOpDesc() = nullptr;
    proto::ModelDef modelDef;
    Model model;
    model.SetGraph(graph);
    EXPECT_TRUE(imp.SerializeModel(model, &modelDef));
}

TEST(UTEST_ge_model_serialize, test_parse_node_false)
{
    ModelSerializeImp imp;
    string node_index = "invalid_index";
    string node_name = "name";
    int32_t index = 1;
    EXPECT_EQ(imp.ParseNodeIndex(node_index, node_name, index), false);
}

TEST(UTEST_ge_model_unserialize, test_invalid_Attr)
{
    {   // invalid graph
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        auto graphAttr = attrDef->mutable_g();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        ComputeGraphPtr graphAttrNew;
        EXPECT_TRUE(AttrUtils::GetGraph(nodes.at(0)->GetOpDesc(), "key1", graphAttrNew));
        ASSERT_TRUE(graphAttrNew != nullptr);
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(graphAttrNew, "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
    {   // invalid list graph
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_GRAPH);
        auto graphAttr = attrDef->mutable_list()->add_g();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        vector<ComputeGraphPtr> graphListAttr;
        EXPECT_TRUE(AttrUtils::GetListGraph(nodes.at(0)->GetOpDesc(), "key1", graphListAttr));
        ASSERT_EQ(graphListAttr.size(), 1);
        ASSERT_TRUE(graphListAttr[0] != nullptr);
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(graphListAttr[0], "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
    {   // invalid namedAttrs
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        auto graphAttr = attrDef->mutable_func();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        GeAttrValue::NAMED_ATTRS namedAttrs;
        EXPECT_TRUE(AttrUtils::GetNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", namedAttrs));
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(namedAttrs, "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
    {   // invalid list namedAttrs
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_NAMED_ATTRS);
        auto graphAttr = attrDef->mutable_list()->add_na();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        GeAttrValue::LIST_NAMED_ATTRS namedAttrs;
        EXPECT_TRUE(AttrUtils::GetListNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", namedAttrs));
        ASSERT_EQ(namedAttrs.size(), 1);
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(namedAttrs.at(0), "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
    {   // invalid tensorDesc
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        auto graphAttr = attrDef->mutable_td();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        GeTensorDesc tensorDesc;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensorDesc));
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(tensorDesc, "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
    {   // invalid list tensorDesc
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR_DESC);
        auto graphAttr = attrDef->mutable_list()->add_td();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        vector<GeTensorDesc> tensorDesc;
        EXPECT_TRUE(AttrUtils::GetListTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensorDesc));
        ASSERT_EQ(tensorDesc.size(), 1);
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(tensorDesc.at(0), "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
    {   // invalid tensor
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        auto graphAttr = attrDef->mutable_t()->mutable_desc();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        ConstGeTensorPtr tensor;
        EXPECT_TRUE(AttrUtils::GetTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor->GetTensorDesc(), "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
    {   // invalid list tensor
        proto::ModelDef modeDeff;
        auto attrs = modeDeff.add_graph()->add_op()->mutable_attr(); // node attr

        proto::AttrDef* attrDef = &(*attrs)["key1"];
        attrDef->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR);
        auto graphAttr = attrDef->mutable_list()->add_t()->mutable_desc();
        auto attrsOfGraph = graphAttr->mutable_attr();
        auto tensorVal = (*attrsOfGraph)["key2"].mutable_td();
        tensorVal->set_dtype(proto::DT_INT8);
        tensorVal->set_layout("invalidLayout");

        ModelSerializeImp imp;
        Model model;
        EXPECT_TRUE(imp.UnserializeModel(model, modeDeff));
        auto graph = model.GetGraph();
        ASSERT_TRUE(graph != nullptr);
        auto nodes = graph->GetAllNodes();
        ASSERT_EQ(nodes.size(), 1);
        vector<ConstGeTensorPtr> tensor;
        EXPECT_TRUE(AttrUtils::GetListTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
        ASSERT_EQ(tensor.size(), 1);
        GeTensorDesc tensorDesc1;
        EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor.at(0)->GetTensorDesc(), "key2", tensorDesc1));
        EXPECT_EQ(tensorDesc1.GetFormat(), FORMAT_RESERVED);
        EXPECT_EQ(tensorDesc1.GetDataType(), DT_INT8);
    }
}
TEST(UTEST_ge_model_unserialize, RebuildOwnershipTest)
{
    ge::ModelSerializeImp serialize_imp;
    float tensor_data[224 * 224] = {1.0f};
    ComputeGraphPtr compute_graph = CreateGraph_1_1_224_224(tensor_data);
    std::map<std::string, ComputeGraphPtr> subgraphs;
    bool ret = serialize_imp.RebuildOwnership(compute_graph, subgraphs);
    EXPECT_EQ(ret, true);
}

TEST(UTEST_ge_model_unserialize, UnserializeModelTest)
{
    ge::ModelSerialize serialize;
    ge::proto::ModelDef model_def;
    Model model;
    bool ret = serialize.UnserializeModel(model_def, model);
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, SerializeGraphGraphIsNull)
{
    ge::ModelSerializeImp model_serialize_imp;
    ConstComputeGraphPtr graph;
    proto::GraphDef *graph_proto = nullptr;
    bool is_dump = true;
    bool ret = model_serialize_imp.SerializeGraph(graph, graph_proto, is_dump);
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, SerializeAllAttrsFromAnyMapMutableAttrIsNull)
{
    ge::ModelSerializeImp model_serialize_imp;
    std::map<std::string, AnyValue> attr_map;
    google::protobuf::Map<std::string, ::ge::proto::AttrDef> *mutable_attr = nullptr;
    bool ret = model_serialize_imp.SerializeAllAttrsFromAnyMap(attr_map, mutable_attr);
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, DeserializeAllAttrsToAttrHolderHolderIsNull)
{
    ge::ModelSerializeImp model_serialize_imp;
    google::protobuf::Map<std::string, ::ge::proto::AttrDef> proto_attr_map;
    AttrHolder *attr_holder = nullptr;
    bool ret = model_serialize_imp.DeserializeAllAttrsToAttrHolder(proto_attr_map, attr_holder);
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, UnserializeModelDataIsNull)
{
    ge::ModelSerialize serialize;
    uint8_t *data = nullptr;
    size_t len = 1;
    Model model;
    bool ret = serialize.UnserializeModel(data, len, model);
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, HandleNodeNameEdgesSrcNodeIsNull)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test1");
    auto dst_node = builder.AddNode("dst_node", "NetOutput", 1, 0);
    NodeNameNodeReq node_req("src_node", 1, -1, dst_node, 0, "dst_node");
    model_impl.node_input_node_names_.push_back(node_req);
    bool ret = model_impl.HandleNodeNameRef();
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, HandleNodeNameEdgesSrcAnchorIsNull)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test1");
    auto dst_node = builder.AddNode("dst_node", "NetOutput", 1, 0);
    NodeNameNodeReq node_req("src_node", 1, -1, dst_node, 0, "dst_node");
    model_impl.node_input_node_names_.push_back(node_req);
    model_impl.node_map_.insert(pair<std::string, NodePtr>("src_node", dst_node));
    bool ret = model_impl.HandleNodeNameRef();
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, HandleNodeNameControlEdgeSuccess)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test1");
    auto dst_node = builder.AddNode("dst_node", "NetOutput", 1, 0);
    NodeNameNodeReq node_req("src_node", -1, -1, dst_node, 0, "dst_node");
    model_impl.node_input_node_names_.push_back(node_req);
    model_impl.node_map_.insert(pair<std::string, NodePtr>("src_node", dst_node));
    bool ret = model_impl.HandleNodeNameRef();
    EXPECT_EQ(ret, true);
}

TEST(UTEST_ge_model_unserialize, HandleNodeNameGraphInputNodeMapIsNull)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test");
    auto graph = builder.GetGraph();
    NodeNameGraphReq graph_req("node1", 1, graph);
    model_impl.graph_input_node_names_.push_back(graph_req);
    bool ret = model_impl.HandleNodeNameRef();
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, HandleNodeNameGraphInputFail)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test");
    auto graph = builder.GetGraph();
    auto node1 = builder.AddNode("node1", "NetOutput", 1, 0);
    NodeNameGraphReq graph_req("node1", 1, graph);
    model_impl.graph_input_node_names_.push_back(graph_req);
    model_impl.node_map_.insert(pair<std::string, NodePtr>("node1", nullptr));
    bool ret = model_impl.HandleNodeNameRef();
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, HandleNodeNameGraphOutputNodeMapIsNull)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test");
    auto graph = builder.GetGraph();
    NodeNameGraphReq graph_req("node1", 1, graph);
    model_impl.graph_output_node_names_.push_back(graph_req);
    bool ret = model_impl.HandleNodeNameRef();
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, HandleNodeNameGraphOutputSuccess)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test");
    auto graph = builder.GetGraph();
    auto node1 = builder.AddNode("node1", "Data", 1, 0);
    NodeNameGraphReq graph_req("node1", 1, graph);
    model_impl.graph_output_node_names_.push_back(graph_req);
    model_impl.node_map_.insert(pair<std::string, NodePtr>("node1", node1));
    bool ret = model_impl.HandleNodeNameRef();
    EXPECT_EQ(ret, true);
}

TEST(UTEST_ge_model_unserialize, UnserializeGraphFail)
{
    ge::ModelSerializeImp model_impl;
    auto builder = ut::GraphBuilder("test");
    auto graph = builder.GetGraph();
    NodeNameGraphReq graph_req("node1", 1, graph);
    model_impl.graph_input_node_names_.push_back(graph_req);
    proto::GraphDef graph_proto;
    bool ret = model_impl.UnserializeGraph(graph, graph_proto);
    EXPECT_EQ(ret, false);
}

TEST(UTEST_ge_model_unserialize, SerializeAttrGroupFailed)
{
  ge::ModelSerializeImp model_serialize_imp;
  std::shared_ptr<ComputeGraph> graph = std::make_shared<ComputeGraph>("test_graph");
  graph->GetOrCreateAttrsGroup<TestAttrGroup>()->status = GRAPH_FAILED;
  proto::GraphDef graph_proto;
  bool is_dump = true;
  bool ret = model_serialize_imp.SerializeGraph(graph, &graph_proto, is_dump);
  EXPECT_EQ(ret, false);
  Model model;
  model.GetOrCreateAttrsGroup<TestAttrGroup>()->status = GRAPH_FAILED;
  proto::ModelDef model_proto;
  model_serialize_imp.SerializeModel(model, &model_proto);
  EXPECT_EQ(ret, false);

  GeShape shape({64, 32, 16, 64});
  GeTensorDesc desc(shape, FORMAT_NCHW, DT_FLOAT16);
  desc.GetOrCreateAttrsGroup<TestAttrGroup>()->status = GRAPH_FAILED;

  proto::TensorDescriptor desc_proto;
  // 异常场景无返回值供校验
  GeTensorSerializeUtils::GeTensorDescAsProto(desc, &desc_proto);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->GetOrCreateAttrsGroup<TestAttrGroup>()->status = GRAPH_FAILED;
  proto::OpDef op_def;
  EXPECT_EQ(model_serialize_imp.SerializeOpDesc(op_desc, &op_def), true);
}