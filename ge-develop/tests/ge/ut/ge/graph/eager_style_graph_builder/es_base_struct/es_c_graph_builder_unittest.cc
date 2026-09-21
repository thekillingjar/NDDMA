/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "es_c_graph_builder.h"
#include "compliant_node_builder.h"
#include <gtest/gtest.h>
#include "common/summary_checker.h"
#include "common/topo_checker.h"
#include "utils/graph_utils_ex.h"
#include "es_graph_builder.h"
using namespace ge::es;

class EsCGraphBuilderLLT : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(EsCGraphBuilderLLT, Test_StoreSubgraph) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", "Data");
  std::vector<EsTensorHolder> outputs1 = {t1};
  auto graph = builder.BuildAndReset(outputs1);
  EXPECT_NE(graph, nullptr);

  EsGraphBuilder subgraph_builder("subgraph");
  auto t2 = subgraph_builder.CreateInput(0, "input0", "Data");
  std::vector<EsTensorHolder> outputs2 = {t2};
  auto subgraph = subgraph_builder.BuildAndReset(outputs2);
  EXPECT_NE(subgraph, nullptr);

  auto c_builder = builder.GetCGraphBuilder();
  auto ret_subgraph = c_builder->AddResource(std::move(subgraph));
  EXPECT_NE(ret_subgraph, nullptr);
  ge::AscendString subgraph_name;
  EXPECT_EQ(ge::GRAPH_SUCCESS, ret_subgraph->GetName(subgraph_name));
  EXPECT_EQ(ge::AscendString("subgraph"), subgraph_name);
}

TEST_F(EsCGraphBuilderLLT, Test_StoreAttrTensor) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", "Data");
  EXPECT_NE(t1.GetProducer(), nullptr);
  std::vector<int64_t> data = {1};
  auto tensor = CreateTensor<int64_t>(data.data(), nullptr, 0, ge::DT_INT64);
  EXPECT_NE(tensor, nullptr);
  auto es_c_tensor = builder.GetCGraphBuilder()->AddResource(std::move(tensor));
  EXPECT_NE(es_c_tensor, nullptr);
  auto ret_tensor = static_cast<ge::Tensor*>(static_cast<void *>(es_c_tensor));
  auto tensor_data = reinterpret_cast<const int64_t *>(ret_tensor->GetData());
  EXPECT_NE(tensor_data, nullptr);
  EXPECT_EQ(tensor_data[0], 1);
}

TEST_F(EsCGraphBuilderLLT, Test_CreateDynamicTensorHolderFromNode) {
  EsGraphBuilder builder("test_graph");
  auto t1 = builder.CreateInput(0, "input0", "Data");
  std::vector<EsTensorHolder> outputs1 = {t1};
  auto c_builder = builder.GetCGraphBuilder();
  auto node = ge::es::CompliantNodeBuilder(c_builder->GetGraph()).OpType("test_node")
      .Name( c_builder->GenerateNodeName("test_node").GetString())
      .IrDefInputs({
          {"x", ge::es::CompliantNodeBuilder::kEsIrInputRequired, ""},
      })
      .IrDefOutputs({
          {"y", ge::es::CompliantNodeBuilder::kEsIrOutputDynamic, ""},
      })
      .InstanceDynamicOutputNum("y", static_cast<int32_t>(3))
      .Build();
  auto ret_dynamic_outputs = c_builder->CreateDynamicTensorHolderFromNode(node, 1, 3);
  EXPECT_NE(ret_dynamic_outputs, nullptr);
  EXPECT_EQ(ret_dynamic_outputs->size(), 3);
}

TEST_F(EsCGraphBuilderLLT, Test_AddGraphInput_failed) {
  EsGraphBuilder builder("test_graph");
  auto c_builder = builder.GetCGraphBuilder();
  EXPECT_EQ(c_builder->AddGraphInput(0, "test", "invalid_type"), nullptr);
}

TEST_F(EsCGraphBuilderLLT, Test_AddResource) {
  EsGraphBuilder builder("test_graph");
  std::vector<int64_t> data{1, 3, 4};
  auto tensor = CreateTensor<int64_t>(data.data(), nullptr, 0, ge::DT_INT64);
  auto req_tensor_stored = builder.GetCGraphBuilder()->AddResource(
      std::unique_ptr<ge::Tensor>(static_cast<ge::Tensor *>(static_cast<void *>(tensor.release()))));
  EXPECT_NE(req_tensor_stored, nullptr);
}