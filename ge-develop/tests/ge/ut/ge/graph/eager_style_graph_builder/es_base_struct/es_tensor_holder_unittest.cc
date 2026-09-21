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
#include "es_graph_builder.h"
#include "es_tensor_holder.h"
#include "es_c_graph_builder.h"
#include "es_c_tensor_holder.h"
#include "graph/utils/node_adapter.h"
#include "graph/symbolizer/symbolic.h"
#include "attribute_group/attr_group_symbolic_desc.h"
using namespace ge::es;
class EsTensorHolderLLT : public ::testing::Test {
 protected:
  void SetUp() override {
    builder = new EsGraphBuilder("tensor_holder_test");
    tensor = builder->CreateScalar(int64_t(123));
    tensor_produer = ge::NodeAdapter::GNode2Node(*tensor.GetProducer());
  }
  void TearDown() override {
    delete builder;
  }
  ge::GeTensorDescPtr EsTensorToGeTensorDesc(const EsTensorHolder &tsh) {
    return tensor_produer->GetOpDesc()->MutableOutputDesc(tsh.GetCTensorHolder()->GetOutIndex());
  }
  EsGraphBuilder *builder;
  EsTensorHolder tensor{nullptr};
  ge::NodePtr tensor_produer{nullptr};
};

TEST_F(EsTensorHolderLLT, SetDataType) {
  EXPECT_EQ(EsTensorToGeTensorDesc(tensor)->GetDataType(), ge::DT_INT64);
  tensor.SetDataType(ge::DT_INT32);
  EXPECT_EQ(EsTensorToGeTensorDesc(tensor)->GetDataType(), ge::DT_INT32);
}

TEST_F(EsTensorHolderLLT, SetFormat) {
  EXPECT_EQ(EsTensorToGeTensorDesc(tensor)->GetFormat(), ge::FORMAT_ND);
  tensor.SetFormat(ge::FORMAT_NCHW);
  EXPECT_EQ(EsTensorToGeTensorDesc(tensor)->GetFormat(), ge::FORMAT_NCHW);
}

TEST_F(EsTensorHolderLLT, SetShape) {
  EXPECT_EQ(EsTensorToGeTensorDesc(tensor)->GetShape(), ge::GeShape{});
  std::vector<int64_t> shape = {2, 3, 4};
  tensor.SetShape(shape);
  EXPECT_EQ(EsTensorToGeTensorDesc(tensor)->GetShape().GetDims(), shape);
}

TEST_F(EsTensorHolderLLT, SetSymbolShape) {
  EXPECT_EQ(EsTensorToGeTensorDesc(tensor)->GetAttrsGroup<ge::SymbolicDescAttr>(), nullptr);
  std::vector<const char *> sym_shape = {"s0", "s1"};
  tensor.SetOriginSymbolShape(sym_shape);
  auto attr_group_sym = EsTensorToGeTensorDesc(tensor)->GetAttrsGroup<ge::SymbolicDescAttr>();
  EXPECT_NE(attr_group_sym, nullptr);
  std::vector<ge::Expression> golden = {ge::Expression::Parse(sym_shape[0]), ge::Expression::Parse(sym_shape[1])};
  EXPECT_EQ(attr_group_sym->symbolic_tensor.GetOriginSymbolShape().GetDims(), golden);
}

TEST_F(EsTensorHolderLLT, GetCTensorHolder) {
  EXPECT_NE(tensor.GetCTensorHolder(), nullptr);
}

TEST_F(EsTensorHolderLLT, OperatorAssignment) {
  EsTensorHolder t2 = tensor;
  EXPECT_EQ(t2.GetCTensorHolder(), tensor.GetCTensorHolder());
}

TEST_F(EsTensorHolderLLT, LiveLife) {
  EsTensorHolder t{nullptr};
  {
    auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_test");
    t = graph_builder->CreateScalar(int64_t(123));
    EXPECT_NE(t.GetCTensorHolder(), nullptr);
    EXPECT_EQ(EsTensorToGeTensorDesc(t)->GetFormat(), ge::FORMAT_ND);
  }
  EXPECT_NE(t.GetCTensorHolder(), nullptr);
  //  注意：这里不进行实际的内存访问，因为会导致未定义行为, 因为已经是野指针
  //  EXPECT_EQ(EsTensorToGeTensorDesc(t)->GetFormat(), ge::FORMAT_ND);
}

// 用例：测试EsSetAttrForTensor
TEST_F(EsTensorHolderLLT, SetAttrForTensor_Int64) {
  EXPECT_EQ(tensor.SetAttr("attr_int64", int64_t(123)), ge::SUCCESS);
  auto td = EsTensorToGeTensorDesc(tensor);
  ge::AnyValue av;
  EXPECT_EQ(td->GetAttr("attr_int64", av), ge::GRAPH_SUCCESS);
  int64_t v = 0;
  av.GetValue(v);
  EXPECT_EQ(v, 123);
  EXPECT_EQ(tensor.SetAttr("attr_int64", int64_t(456)), ge::SUCCESS);
  EXPECT_EQ(td->GetAttr("attr_int64", av), ge::GRAPH_SUCCESS);
  av.GetValue(v);
  EXPECT_EQ(v, 456);
}

TEST_F(EsTensorHolderLLT, SetAttrForTensor_String) {
  EXPECT_EQ(tensor.SetAttr("attr_str", "abc"), ge::SUCCESS);
  auto td = EsTensorToGeTensorDesc(tensor);
  ge::AnyValue av;
  EXPECT_EQ(td->GetAttr("attr_str", av), ge::GRAPH_SUCCESS);
  std::string v;
  av.GetValue(v);
  EXPECT_EQ(v, "abc");
  EXPECT_EQ(tensor.SetAttr("attr_str", "def"), ge::SUCCESS);
  EXPECT_EQ(td->GetAttr("attr_str", av), ge::GRAPH_SUCCESS);
  av.GetValue(v);
  EXPECT_EQ(v, "def");
}

TEST_F(EsTensorHolderLLT, SetAttrForTensor_Bool) {
  EXPECT_EQ(tensor.SetAttr("attr_bool", true), ge::SUCCESS);
  auto td = EsTensorToGeTensorDesc(tensor);
  ge::AnyValue av;
  EXPECT_EQ(td->GetAttr("attr_bool", av), ge::GRAPH_SUCCESS);
  bool v = false;
  av.GetValue(v);
  EXPECT_EQ(v, true);
  EXPECT_EQ(tensor.SetAttr("attr_bool", false), ge::SUCCESS);
  EXPECT_EQ(td->GetAttr("attr_bool", av), ge::GRAPH_SUCCESS);
  av.GetValue(v);
  EXPECT_EQ(v, false);
}

// 用例：测试EsSetAttrForNode
TEST_F(EsTensorHolderLLT, SetAttrForNode_Int64) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test");
  auto t = graph_builder->CreateScalar(int64_t(321));
  EXPECT_EQ(t.SetAttrForNode("node_attr_int64", int64_t(789)), ge::SUCCESS);
  auto esb_tensor = t.GetCTensorHolder();
  auto node = esb_tensor->GetProducer();
  int64_t v = 0;
  EXPECT_EQ(node.GetAttr("node_attr_int64", v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(v, 789);
  EXPECT_EQ(t.SetAttrForNode("node_attr_int64", int64_t(100)), ge::SUCCESS);
  EXPECT_EQ(node.GetAttr("node_attr_int64", v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(v, 100);

  // 另外一种设置属性的方式
  auto node_ptr = t.GetProducer();
  EXPECT_TRUE(node_ptr != nullptr);
  EXPECT_EQ(node_ptr->SetAttr("node_attr_int64", v), ge::SUCCESS);
  int64_t v_get = 0;
  EXPECT_EQ(node.GetAttr("node_attr_int64", v_get), ge::GRAPH_SUCCESS);
  EXPECT_EQ(v_get, v);
}

TEST_F(EsTensorHolderLLT, SetAttrForNode_String) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test2");
  auto t = graph_builder->CreateScalar(int64_t(111));
  EXPECT_EQ(t.SetAttrForNode("node_attr_str", "hello"), ge::SUCCESS);
  auto esb_tensor = t.GetCTensorHolder();
  auto node = esb_tensor->GetProducer();
  ge::AscendString v;
  EXPECT_EQ(node.GetAttr("node_attr_str", v), ge::GRAPH_SUCCESS);

  EXPECT_EQ(v, "hello");
  EXPECT_EQ(t.SetAttrForNode("node_attr_str", "world"), ge::SUCCESS);
  EXPECT_EQ(node.GetAttr("node_attr_str", v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(v, "world");
}

TEST_F(EsTensorHolderLLT, SetAttrForNode_Bool) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test3");
  auto t = graph_builder->CreateScalar(int64_t(222));
  EXPECT_EQ(t.SetAttrForNode("node_attr_bool", true), ge::SUCCESS);
  auto esb_tensor = t.GetCTensorHolder();
  auto node = esb_tensor->GetProducer();
  bool v = false;
  EXPECT_EQ(node.GetAttr("node_attr_bool", v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(v, true);
  EXPECT_EQ(t.SetAttrForNode("node_attr_bool", false), ge::SUCCESS);
  EXPECT_EQ(node.GetAttr("node_attr_bool", v), ge::GRAPH_SUCCESS);
  EXPECT_EQ(v, false);
}

TEST_F(EsTensorHolderLLT, OperatorAdd) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test4");
  auto t1 = graph_builder->CreateScalar(int64_t(111));
  auto t2 = graph_builder->CreateScalar(int64_t(222));
  auto t3 = t1 + t2;
  EXPECT_EQ(t3.GetCTensorHolder(), nullptr);
}

TEST_F(EsTensorHolderLLT, OperatorSub) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test4");
  auto t1 = graph_builder->CreateScalar(int64_t(111));
  auto t2 = graph_builder->CreateScalar(int64_t(222));
  auto t3 = t1 - t2;
  EXPECT_EQ(t3.GetCTensorHolder(), nullptr);
}
TEST_F(EsTensorHolderLLT, OperatorMul) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test4");
  auto t1 = graph_builder->CreateScalar(int64_t(111));
  auto t2 = graph_builder->CreateScalar(int64_t(222));
  auto t3 = t1 * t2;
  EXPECT_EQ(t3.GetCTensorHolder(), nullptr);
}
TEST_F(EsTensorHolderLLT, OperatorDiv) {
  auto graph_builder = ge::ComGraphMakeUnique<EsGraphBuilder>("tensor_holder_node_test4");
  auto t1 = graph_builder->CreateScalar(int64_t(111));
  auto t2 = graph_builder->CreateScalar(int64_t(222));
  auto t3 = t1 / t2;
  EXPECT_EQ(t3.GetCTensorHolder(), nullptr);
}

TEST_F(EsTensorHolderLLT, AddControlEdge_success) {
  auto tensor0 = builder->CreateInput(0);
  auto tensor1 = builder->CreateInput(1);
  auto tensor2 = builder->CreateInput(2);
  std::vector<EsTensorHolder> ctrl_ins = {tensor1, tensor2};
  EXPECT_EQ(ge::GRAPH_SUCCESS, tensor0.AddControlEdge(ctrl_ins));
  auto srd_node = tensor0.GetProducer();
  auto src_node_ctrl_ins = srd_node->GetInControlNodes();
  EXPECT_EQ(2, src_node_ctrl_ins.size());
  ge::AscendString ctrl_in_name;
  EXPECT_EQ(ge::GRAPH_SUCCESS, src_node_ctrl_ins.at(0)->GetName(ctrl_in_name));
  EXPECT_EQ(ge::AscendString("input_1"), ctrl_in_name.GetString());
  EXPECT_EQ(ge::GRAPH_SUCCESS, src_node_ctrl_ins.at(1)->GetName(ctrl_in_name));
  EXPECT_EQ(ge::AscendString("input_2"), ctrl_in_name.GetString());
}

TEST_F(EsTensorHolderLLT, AddControlEdge_fail) {
  auto tensor_invalid = ge::ComGraphMakeUnique<EsTensorHolder>();
  auto tensor1 = builder->CreateInput(1);
  auto tensor2 = builder->CreateInput(2);
  std::vector<EsTensorHolder> ctrl_ins = {tensor1, tensor2};
  EXPECT_EQ(ge::PARAM_INVALID, tensor_invalid->AddControlEdge(ctrl_ins));
}

TEST_F(EsTensorHolderLLT, ProducerOutIndex) {
  auto tensor_invalid = ge::ComGraphMakeUnique<EsTensorHolder>();
  auto tensor1 = builder->CreateInput(1);
  auto out_index = tensor1.GetProducerOutIndex();
  EXPECT_EQ(out_index, 0);
}