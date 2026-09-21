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
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "graph/model_serialize.h"
#include "graph/detail/model_serialize_imp.h"
#include "graph/normal_graph/node_impl.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "macro_utils/dt_public_unscope.h"

#include "proto/ge_ir.pb.h"

using namespace ge;
using std::string;
using std::vector;

void LinkEdge(NodePtr src_node, int32_t src_index, NodePtr dst_node, int32_t dst_index) {
  if (src_index >= 0) {
    auto src_anchor = src_node->GetOutDataAnchor(src_index);
    auto dst_anchor = dst_node->GetInDataAnchor(dst_index);
    src_anchor->LinkTo(dst_anchor);
  } else {
    auto src_anchor = src_node->GetOutControlAnchor();
    auto dst_anchor = dst_node->GetInControlAnchor();
    src_anchor->LinkTo(dst_anchor);
  }
}

NodePtr CreateNode(OpDescPtr op, ComputeGraphPtr owner_graph) { return owner_graph->AddNode(op); }

void CompareShape(const vector<int64_t> &shape1, const vector<int64_t> &shape2) {
  EXPECT_EQ(shape1.size(), shape2.size());
  if (shape1.size() == shape2.size()) {
    for (size_t i = 0; i < shape1.size(); i++) {
      EXPECT_EQ(shape1[i], shape2[i]);
    }
  }
}

template <typename T>
void CompareList(const vector<T> &val1, const vector<T> &val2) {
  EXPECT_EQ(val1.size(), val2.size());
  if (val1.size() == val2.size()) {
    for (size_t i = 0; i < val1.size(); i++) {
      EXPECT_EQ(val1[i], val2[i]);
    }
  }
}

static bool NamedAttrsSimpleCmp(const GeAttrValue &left, const GeAttrValue &right) {
  NamedAttrs val1, val2;
  left.GetValue<NamedAttrs>(val1);
  right.GetValue<NamedAttrs>(val2);
  if (val1.GetName() != val2.GetName()) {
    return false;
  }
  auto attrs1 = val1.GetAllAttrs();
  auto attrs2 = val2.GetAllAttrs();
  if (attrs1.size() != attrs1.size()) {
    return false;
  }

  for (auto it : attrs1) {
    auto it2 = attrs2.find(it.first);
    if (it2 == attrs2.end()) {  // simple check
      return false;
    }
    if (it.second.GetValueType() != it2->second.GetValueType()) {
      return false;
    }
    switch (it.second.GetValueType()) {
      case GeAttrValue::VT_INT: {
        int64_t i1 = 0, i2 = 0;
        it.second.GetValue<int64_t>(i1);
        it2->second.GetValue<int64_t>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_FLOAT: {
        float i1 = 0, i2 = 0;
        it.second.GetValue<float>(i1);
        it2->second.GetValue<float>(i2);
        if (fabs(i1 - i2) > FLT_EPSILON) {
          return false;
        }
      }
      case GeAttrValue::VT_STRING: {
        string i1, i2;
        it.second.GetValue<std::string>(i1);
        it2->second.GetValue<std::string>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      case GeAttrValue::VT_BOOL: {
        bool i1 = false, i2 = false;
        it.second.GetValue<bool>(i1);
        it2->second.GetValue<bool>(i2);
        if (i1 != i2) {
          return false;
        }
      }
      default: {
        return true;
      }
    }
  }
  return true;
}

static NamedAttrs CreateNamedAttrs(const string &name, std::map<string, GeAttrValue> map) {
  NamedAttrs named_attrs;
  named_attrs.SetName(name);
  for (auto it : map) {
    named_attrs.SetAttr(it.first, it.second);
  }
  return named_attrs;
}

TEST(UtestGeModelSerialize, simple) {
  Model model("model_name", "custom version3.0");
  model.SetAttr("model_key1", GeAttrValue::CreateFrom<int64_t>(123));
  model.SetAttr("model_key2", GeAttrValue::CreateFrom<float>(456.78f));
  model.SetAttr("model_key3", GeAttrValue::CreateFrom<std::string>("abcd"));
  model.SetAttr("model_key4", GeAttrValue::CreateFrom<std::vector<int64_t>>({123, 456}));
  model.SetAttr("model_key5", GeAttrValue::CreateFrom<std::vector<float>>({456.78f, 998.90f}));
  model.SetAttr("model_key6", GeAttrValue::CreateFrom<std::vector<std::string>>({"abcd", "happy"}));
  model.SetAttr("model_key7", GeAttrValue::CreateFrom<bool>(false));
  model.SetAttr("model_key8", GeAttrValue::CreateFrom<std::vector<bool>>({true, false}));

  auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

  // input
  auto input_op = std::make_shared<OpDesc>("input", "Input");
  input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto input = CreateNode(input_op, compute_graph);
  // w1
  auto w1_op = std::make_shared<OpDesc>("w1", "ConstOp");
  w1_op->AddOutputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  auto w1 = CreateNode(w1_op, compute_graph);

  // node1
  auto node1_op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  auto node1 = CreateNode(node1_op, compute_graph);

  // Attr set
  node1_op->SetAttr("node_key1", GeAttrValue::CreateFrom<Buffer>(Buffer(10)));
  node1_op->SetAttr("node_key2", GeAttrValue::CreateFrom<std::vector<Buffer>>({Buffer(20), Buffer(30)}));
  auto named_attrs1 = GeAttrValue::CreateFrom<NamedAttrs>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<float>(345.345)}}));

  node1_op->SetAttr("node_key3", std::move(named_attrs1));
  auto list_named_attrs = GeAttrValue::CreateFrom<std::vector<NamedAttrs>>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<float>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<float>(345.345)}})});
  node1_op->SetAttr("node_key4", std::move(list_named_attrs));
  // tensor
  auto tensor_data1 = "qwertyui";
  GeTensor tensor1(GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT8), (uint8_t *)tensor_data1, 8);
  auto tensor_data2 = "asdfqwertyui";
  GeTensor tensor2(GeTensorDesc(GeShape({3, 2, 2}), FORMAT_ND, DT_UINT8), (uint8_t *)tensor_data2, 12);
  auto tensor_data3 = "ghjkasdfqwertyui";
  GeTensor tensor3(GeTensorDesc(GeShape({4, 2, 2}), FORMAT_ND, DT_UINT16), (uint8_t *)tensor_data3, 16);
  node1_op->SetAttr("node_key5", GeAttrValue::CreateFrom<GeTensor>(tensor1));
  node1_op->SetAttr("node_key6", GeAttrValue::CreateFrom<std::vector<GeTensor>>({tensor2, tensor3}));

  auto tensor_desc = GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT16);
  TensorUtils::SetSize(tensor_desc, 100);
  node1_op->SetAttr("node_key7", GeAttrValue::CreateFrom<GeTensorDesc>(tensor_desc));
  node1_op->SetAttr("node_key8", GeAttrValue::CreateFrom<std::vector<GeTensorDesc>>(
                                    {GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT32),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT32),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_INT64),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_UINT64),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_BOOL),
                                     GeTensorDesc(GeShape({2, 2, 2}), FORMAT_NCHW, DT_DOUBLE)}));

  LinkEdge(input, 0, node1, 0);
  LinkEdge(w1, 0, node1, 1);

  model.SetGraph(compute_graph);

  Buffer buffer;
  ASSERT_EQ(model.Save(buffer), GRAPH_SUCCESS);
  EXPECT_TRUE(buffer.GetData() != nullptr);

  Model model2;
  ASSERT_EQ(Model::Load(buffer.GetData(), buffer.GetSize(), model2), GRAPH_SUCCESS);
  EXPECT_EQ(model2.GetName(), "model_name");
  int64_t model_val1;
  AttrUtils::GetInt(&model2, "model_key1", model_val1);
  EXPECT_EQ(model_val1, 123);

  float model_val2;
  AttrUtils::GetFloat(&model2, "model_key2", model_val2);
  EXPECT_EQ(model_val2, (float)456.78f);

  std::string model_val3;
  AttrUtils::GetStr(&model2, "model_key3", model_val3);
  EXPECT_EQ(model_val3, "abcd");

  std::vector<int64_t> model_val4;
  AttrUtils::GetListInt(&model2, "model_key4", model_val4);
  CompareList(model_val4, {123, 456});

  std::vector<float> model_val5;
  AttrUtils::GetListFloat(&model2, "model_key5", model_val5);
  CompareList(model_val5, {456.78f, 998.90f});

  std::vector<std::string> model_val6;
  AttrUtils::GetListStr(&model2, "model_key6", model_val6);
  CompareList(model_val6, {"abcd", "happy"});

  bool model_val7;
  EXPECT_EQ(AttrUtils::GetBool(&model2, "model_key7", model_val7), true);
  EXPECT_EQ(model_val7, false);

  std::vector<bool> model_val8;
  AttrUtils::GetListBool(&model2, "model_key8", model_val8);
  CompareList(model_val8, {true, false});

  auto graph2 = model2.GetGraph();
  const auto &s_graph = graph2;
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

    auto out_anchor = s_input->GetOutDataAnchor(0);
    auto peer_anchors = out_anchor->GetPeerInDataAnchors();
    ASSERT_EQ(peer_anchors.size(), 1);
    auto peer_anchor = peer_anchors.at(0);
    ASSERT_EQ(peer_anchor->GetIdx(), 0);
    ASSERT_EQ(peer_anchor->GetOwnerNode(), s_nod1);
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

    auto out_anchor = s_w1->GetOutDataAnchor(0);
    auto peer_anchors = out_anchor->GetPeerInDataAnchors();
    ASSERT_EQ(peer_anchors.size(), 1);
    auto peer_anchor = peer_anchors.at(0);
    ASSERT_EQ(peer_anchor->GetIdx(), 1);
    ASSERT_EQ(peer_anchor->GetOwnerNode(), s_nod1);
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

    auto out_anchor = s_nod1->GetOutDataAnchor(0);
    auto peer_anchors = out_anchor->GetPeerInDataAnchors();
    ASSERT_EQ(peer_anchors.size(), 0);

    // node attrs
    Buffer node_val1;
    AttrUtils::GetBytes(s_op, "node_key1", node_val1);
    ASSERT_EQ(node_val1.GetSize(), 10);

    std::vector<Buffer> node_val2;
    AttrUtils::GetListBytes(s_op, "node_key2", node_val2);
    ASSERT_EQ(node_val2.size(), 2);
    ASSERT_EQ(node_val2[0].GetSize(), 20);
    ASSERT_EQ(node_val2[1].GetSize(), 30);

    GeAttrValue s_named_attrs;
    s_op->GetAttr("node_key3", s_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_named_attrs, named_attrs1));

    GeAttrValue s_list_named_attrs;
    s_op->GetAttr("node_key4", s_list_named_attrs);
    EXPECT_TRUE(NamedAttrsSimpleCmp(s_list_named_attrs, list_named_attrs));

    ConstGeTensorPtr s_tensor;
    AttrUtils::GetTensor(s_op, "node_key5", s_tensor);
    ASSERT_TRUE(s_tensor != nullptr);
    string str((char *)s_tensor->GetData().data(), s_tensor->GetData().size());
    EXPECT_EQ(str, "qwertyui");

    vector<ConstGeTensorPtr> s_list_tensor;
    AttrUtils::GetListTensor(s_op, "node_key6", s_list_tensor);
    ASSERT_EQ(s_list_tensor.size(), 2);
    string str2((char *)s_list_tensor[0]->GetData().data(), s_list_tensor[0]->GetData().size());
    EXPECT_EQ(str2, "asdfqwertyui");
    string str3((char *)s_list_tensor[1]->GetData().data(), s_list_tensor[1]->GetData().size());
    EXPECT_EQ(str3, "ghjkasdfqwertyui");

    GeTensorDesc s_tensor_desc;
    AttrUtils::GetTensorDesc(s_op, "node_key7", s_tensor_desc);
    EXPECT_EQ(s_tensor_desc.GetFormat(), FORMAT_NCHW);
    EXPECT_EQ(s_tensor_desc.GetDataType(), DT_INT16);
    int64_t size = 0;
    TensorUtils::GetSize(s_tensor_desc, size);
    EXPECT_EQ(size, 100);

    vector<GeTensorDesc> s_list_tensor_desc;
    AttrUtils::GetListTensorDesc(s_op, "node_key8", s_list_tensor_desc);
    ASSERT_EQ(s_list_tensor_desc.size(), 6);
    EXPECT_EQ(s_list_tensor_desc[0].GetDataType(), DT_INT32);
    EXPECT_EQ(s_list_tensor_desc[1].GetDataType(), DT_UINT32);
    EXPECT_EQ(s_list_tensor_desc[2].GetDataType(), DT_INT64);
    EXPECT_EQ(s_list_tensor_desc[3].GetDataType(), DT_UINT64);
    EXPECT_EQ(s_list_tensor_desc[4].GetDataType(), DT_BOOL);
    EXPECT_EQ(s_list_tensor_desc[5].GetDataType(), DT_DOUBLE);
  }
}

TEST(UtestGeModelSerialize, opdesc_as_attr_value) {
  // node1_op
  auto node1_op = std::make_shared<OpDesc>("node1", "Conv2D");
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
  node1_op->AddInputDesc(GeTensorDesc(GeShape({12, 2, 64, 64, 16}), FORMAT_NC1HWC0, DT_FLOAT16));
  node1_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

  // Attr set
  node1_op->SetAttr("node_key1", GeAttrValue::CreateFrom<Buffer>(Buffer(10)));
  node1_op->SetAttr("node_key2", GeAttrValue::CreateFrom<std::vector<Buffer>>({Buffer(20), Buffer(30)}));
  auto named_attrs1 = GeAttrValue::CreateFrom<NamedAttrs>(
      CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                   {"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                   {"float_val", GeAttrValue::CreateFrom<float>(345.345)}}));

  node1_op->SetAttr("node_key3", std::move(named_attrs1));
  auto list_named_attrs = GeAttrValue::CreateFrom<std::vector<NamedAttrs>>(
      {CreateNamedAttrs("my_name", {{"int_val", GeAttrValue::CreateFrom<int64_t>(123)},
                                    {"float_val", GeAttrValue::CreateFrom<float>(345.345)}}),
       CreateNamedAttrs("my_name2", {{"str_val", GeAttrValue::CreateFrom<string>("abc")},
                                     {"float_val", GeAttrValue::CreateFrom<float>(345.345)}})});
  node1_op->SetAttr("node_key4", std::move(list_named_attrs));

  Model model;
  EXPECT_TRUE(AttrUtils::SetListInt(&model, "my_key2", {123}));
  EXPECT_TRUE(AttrUtils::SetListBytes(&model, "my_key3", {Buffer(100)}));
}

TEST(UtestGeModelSerialize, test_sub_graph) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    auto sub_compute_graph = std::make_shared<ComputeGraph>("sub_graph");
    // input
    auto sub_graph_input_op = std::make_shared<OpDesc>("sub_graph_test", "TestOp2");
    sub_graph_input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input = CreateNode(sub_graph_input_op, sub_compute_graph);

    AttrUtils::SetGraph(input_op, "sub_graph", sub_compute_graph);
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
}

TEST(UtestGeModelSerialize, test_list_sub_graph) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    auto sub_compute_graph1 = std::make_shared<ComputeGraph>("sub_graph1");
    // input
    auto sub_graph_input_op1 = std::make_shared<OpDesc>("sub_graph_test1", "TestOp2");
    sub_graph_input_op1->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input1 = CreateNode(sub_graph_input_op1, sub_compute_graph1);

    auto sub_compute_graph2 = std::make_shared<ComputeGraph>("sub_graph2");
    // input
    auto sub_graph_input_op2 = std::make_shared<OpDesc>("sub_graph_test2", "TestOp2");
    sub_graph_input_op2->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto sub_graph_input2 = CreateNode(sub_graph_input_op2, sub_compute_graph2);

    AttrUtils::SetListGraph(input_op, "sub_graph", vector<ComputeGraphPtr>{sub_compute_graph1, sub_compute_graph2});
  }

  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
}

TEST(UtestGeModelSerialize, test_format) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_ND, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1HWC0, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NC1C0HWPAD, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NHWC1C0, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FSR_NCHW, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_DECONV, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_BN_WEIGHT, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_CHWN, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FILTER_HWCK, DT_FLOAT));
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_FRACTAL_Z_C04, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  ASSERT_GE(buffer.GetSize(), 0);
}

TEST(UtestGeModelSerialize, test_control_edge) {
  Model model("model_name", "custom version3.0");
  {
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");
    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddInputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    // sink
    auto sink_op = std::make_shared<OpDesc>("test2", "Sink");
    auto sink = CreateNode(sink_op, compute_graph);
    LinkEdge(sink, -1, input, -1);

    // sink2
    auto sink_op2 = std::make_shared<OpDesc>("test3", "Sink");
    auto sink2 = CreateNode(sink_op2, compute_graph);
    LinkEdge(sink2, -1, input, -1);

    // dest
    auto dest_op = std::make_shared<OpDesc>("test4", "Dest");
    auto dest = CreateNode(dest_op, compute_graph);
    LinkEdge(input, -1, dest, -1);

    compute_graph->AddInputNode(sink);
    compute_graph->AddInputNode(sink2);
    compute_graph->AddOutputNode(dest);

    model.SetGraph(compute_graph);
  }
  ModelSerialize serialize;
  auto buffer = serialize.SerializeModel(model);
  EXPECT_GE(buffer.GetSize(), 0);
}

TEST(UtestGeModelSerialize, test_invalid_model) {
  {  // empty graph
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_EQ(buffer.GetSize(), 0);
  }
}

TEST(UtestGeModelSerialize, test_invalid_tensor_desc) {
  {  // valid test
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));
    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // invalid format
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_RESERVED, DT_FLOAT));  // invalid format
    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
  }
  {  // DT_UNDEFINED datatype
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_UNDEFINED));
    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    ASSERT_GE(buffer.GetSize(), 0);
  }
}

TEST(UtestGeModelSerialize, test_invalid_attrs) {
  {  // valid test
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    NamedAttrs named_attrs;
    named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<int64_t>(10));
    AttrUtils::SetNamedAttrs(input_op, "key", named_attrs);

    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // none type
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    NamedAttrs named_attrs;
    EXPECT_EQ(named_attrs.SetAttr("key1", GeAttrValue()), GRAPH_FAILED);
  }
  {  // bytes attr len is 0
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    NamedAttrs named_attrs;
    named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<Buffer>(Buffer(512)));
    AttrUtils::SetNamedAttrs(input_op, "key", named_attrs);

    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 512);
  }
  {  // invalid list bytes attr
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    NamedAttrs named_attrs;
    named_attrs.SetAttr("key1", GeAttrValue::CreateFrom<std::vector<Buffer>>({Buffer(0)}));
    AttrUtils::SetNamedAttrs(input_op, "key", named_attrs);

    auto input = CreateNode(input_op, compute_graph);
    model.SetGraph(compute_graph);

    ModelSerialize serialize;
    auto buffer = serialize.SerializeModel(model);
    EXPECT_GE(buffer.GetSize(), 0);
  }
  {  // invalid graph attr
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    NamedAttrs named_attrs;
    EXPECT_EQ(named_attrs.SetAttr("key1",
                                  GeAttrValue::CreateFrom<proto::GraphDef>(proto::GraphDef())), GRAPH_SUCCESS);
    GeAttrValue value;
    EXPECT_EQ(named_attrs.GetAttr("key1", value), GRAPH_SUCCESS);
    EXPECT_TRUE(!value.IsEmpty());
  }
  {  // invalid list graph attr
    Model model("model_name", "custom version3.0");
    auto compute_graph = std::make_shared<ComputeGraph>("graph_name");

    // input
    auto input_op = std::make_shared<OpDesc>("test", "TestOp");
    input_op->AddOutputDesc(GeTensorDesc(GeShape({12, 32, 64, 64}), FORMAT_NCHW, DT_FLOAT));

    NamedAttrs named_attrs;
    EXPECT_EQ(named_attrs.
              SetAttr("key1",
                      GeAttrValue::CreateFrom<std::vector<proto::GraphDef>>({proto::GraphDef()})), GRAPH_SUCCESS);
    GeAttrValue value;
    EXPECT_EQ(named_attrs.GetAttr("key1", value), GRAPH_SUCCESS);
    EXPECT_TRUE(!value.IsEmpty());
  }
}

TEST(UtestGeModelSerialize, test_model_serialize_imp_invalid_param) {
  ModelSerializeImp imp;
  EXPECT_FALSE(imp.SerializeModel(Model(), nullptr));
  EXPECT_FALSE(imp.SerializeNode(nullptr, nullptr));

  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto node = graph->AddNode(std::make_shared<OpDesc>());
  node->impl_->op_ = nullptr;
  ge::proto::ModelDef model_def;
  Model model;
  model.SetGraph(graph);
  EXPECT_FALSE(imp.SerializeModel(model, &model_def));
}

TEST(UTEST_ge_model_unserialize, test_invalid_TensorDesc) {
  {  // valid
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_td();
    tensor_desc_attr->set_layout("NCHW");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
  }
  {  // invalid layout
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_td();
    tensor_desc_attr->set_layout("InvalidLayout");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    GeTensorDesc tensor_desc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(model, "key1", tensor_desc));
    EXPECT_EQ(tensor_desc.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc.GetDataType(), DT_INT8);
  }
  {  // invalid datatype
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_td();  // tensor desc
    tensor_desc_attr->set_layout("NHWC");
    tensor_desc_attr->set_dtype((ge::proto::DataType)100);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    GeTensorDesc tensor_desc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(model, "key1", tensor_desc));
    EXPECT_EQ(tensor_desc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(tensor_desc.GetDataType(), DT_UNDEFINED);
  }
  {  // invalid datatype
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.mutable_attr();

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_t()->mutable_desc();  // tensor
    tensor_desc_attr->set_layout("NHWC");
    tensor_desc_attr->set_dtype((ge::proto::DataType)100);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(model, "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensor_desc = tensor->GetTensorDesc();
    EXPECT_EQ(tensor_desc.GetFormat(), FORMAT_NHWC);
    EXPECT_EQ(tensor_desc.GetDataType(), DT_UNDEFINED);
  }
  {  // invalid attrmap
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->mutable_attr();  // graph attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_t()->mutable_desc();  // tensor
    tensor_desc_attr->set_layout("NCHW");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);
    auto attrs1 = tensor_desc_attr->mutable_attr();
    auto attr1 = (*attrs1)["key2"];  // empty attr

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(graph, "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensor_desc = tensor->GetTensorDesc();
    GeAttrValue attr_value;
    EXPECT_NE(tensor_desc.GetAttr("key2", attr_value), GRAPH_SUCCESS);
    EXPECT_EQ(attr_value.GetValueType(), GeAttrValue::VT_NONE);
  }
  {  // invalid attrmap2
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto tensor_desc_attr = attr_def->mutable_t()->mutable_desc();  // tensor
    tensor_desc_attr->set_layout("NCHW");
    tensor_desc_attr->set_dtype(ge::proto::DataType::DT_INT8);

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_TRUE(tensor != nullptr);
    auto tensor_desc = tensor->GetTensorDesc();
    GeAttrValue attr_value;
    EXPECT_NE(tensor_desc.GetAttr("key2", attr_value), GRAPH_SUCCESS);
    EXPECT_EQ(attr_value.GetValueType(), GeAttrValue::VT_NONE);
  }
}
TEST(UTEST_ge_model_unserialize, test_invalid_attr) {
  {  // invalid graph
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_g();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ComputeGraphPtr graph_attr_new;
    EXPECT_TRUE(AttrUtils::GetGraph(nodes.at(0)->GetOpDesc(), "key1", graph_attr_new));
    ASSERT_TRUE(graph_attr_new != nullptr);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(graph_attr_new, "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list graph
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_GRAPH);
    auto graph_attr = attr_def->mutable_list()->add_g();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ComputeGraphPtr> graph_list_attr;
    EXPECT_TRUE(AttrUtils::GetListGraph(nodes.at(0)->GetOpDesc(), "key1", graph_list_attr));
    ASSERT_EQ(graph_list_attr.size(), 1);
    ASSERT_TRUE(graph_list_attr[0] != nullptr);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(graph_list_attr[0], "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid named_attrs
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_func();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    NamedAttrs named_attrs;
    EXPECT_TRUE(AttrUtils::GetNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", named_attrs));
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(named_attrs, "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list named_attrs
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_NAMED_ATTRS);
    auto graph_attr = attr_def->mutable_list()->add_na();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    std::vector<NamedAttrs> named_attrs;
    EXPECT_TRUE(AttrUtils::GetListNamedAttrs(nodes.at(0)->GetOpDesc(), "key1", named_attrs));
    ASSERT_EQ(named_attrs.size(), 1);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(named_attrs.at(0), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid tensor_desc
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_td();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    GeTensorDesc tensor_desc;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensor_desc));
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor_desc, "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor_desc
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR_DESC);
    auto graph_attr = attr_def->mutable_list()->add_td();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<GeTensorDesc> tensor_desc;
    EXPECT_TRUE(AttrUtils::GetListTensorDesc(nodes.at(0)->GetOpDesc(), "key1", tensor_desc));
    ASSERT_EQ(tensor_desc.size(), 1);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor_desc.at(0), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid tensor
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    auto graph_attr = attr_def->mutable_t()->mutable_desc();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    ConstGeTensorPtr tensor;
    EXPECT_TRUE(AttrUtils::GetTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor->GetTensorDesc(), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor
    ge::proto::ModelDef mode_def;
    auto attrs = mode_def.add_graph()->add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR);
    auto graph_attr = attr_def->mutable_list()->add_t()->mutable_desc();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Model model;
    EXPECT_TRUE(imp.UnserializeModel(model, mode_def));
    auto graph = model.GetGraph();
    ASSERT_TRUE(graph != nullptr);
    auto nodes = graph->GetAllNodes();
    ASSERT_EQ(nodes.size(), 1);
    vector<ConstGeTensorPtr> tensor;
    EXPECT_TRUE(AttrUtils::GetListTensor(nodes.at(0)->GetOpDesc(), "key1", tensor));
    ASSERT_EQ(tensor.size(), 1);
    GeTensorDesc tensor_desc1;
    EXPECT_TRUE(AttrUtils::GetTensorDesc(tensor.at(0)->GetTensorDesc(), "key2", tensor_desc1));
    EXPECT_EQ(tensor_desc1.GetFormat(), FORMAT_RESERVED);
    EXPECT_EQ(tensor_desc1.GetDataType(), DT_INT8);
  }
  {  // invalid list tensor
    ge::proto::GraphDef graph_def;
    auto attrs = graph_def.add_op()->mutable_attr();  // node attr

    ge::proto::AttrDef *attr_def = &(*attrs)["key1"];
    attr_def->mutable_list()->set_val_type(ge::proto::AttrDef_ListValue_ListValueType_VT_LIST_TENSOR);
    auto graph_attr = attr_def->mutable_list()->add_t()->mutable_desc();
    auto attrs_of_graph = graph_attr->mutable_attr();
    auto tensor_val = (*attrs_of_graph)["key2"].mutable_td();
    tensor_val->set_dtype(ge::proto::DT_INT8);
    tensor_val->set_layout("invalidLayout");

    ModelSerializeImp imp;
    Buffer buffer(graph_def.ByteSizeLong());
    graph_def.SerializeToArray(buffer.GetData(), static_cast<int>(buffer.GetSize()));
  }
}

