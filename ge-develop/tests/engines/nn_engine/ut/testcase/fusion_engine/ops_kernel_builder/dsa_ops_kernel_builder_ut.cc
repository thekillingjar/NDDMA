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
#include <nlohmann/json.hpp>
#include "all_ops_stub.h"
#include <iostream>

#include <list>

#define private public
#define protected public
#include "ops_kernel_builder/dsa_ops_kernel_builder.h"
#include "graph/node.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/compute_graph.h"
#include "common/constants_define.h"
#include "common/aicore_util_attr_define.h"
#include "rt_error_codes.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"

using namespace std;
using namespace fe;
using namespace ge;
using OpsKernelBuilderPtr = shared_ptr<DsaOpsKernelBuilder>;

class DsaOpsKernelBuilderTest : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "DSAOpsKernelBuilderTest SetUP" << endl;
  }
  static void TearDownTestCase() {
    cout << "DSAOpsKernelBuilderTest SetUP" << endl;
  }
  // Some expensive resource shared by all tests.
  virtual void SetUp() {
    dsa_ops_kernel_builder_ptr_ = std::shared_ptr<DsaOpsKernelBuilder>(new DsaOpsKernelBuilder());
    std::map<std::string, std::string> options;
    dsa_ops_kernel_builder_ptr_->Initialize(options);
    context_ = CreateContext();
  }
  virtual void TearDown() {
    cout << "a test Tear Down" << endl;
    dsa_ops_kernel_builder_ptr_->Finalize();
  }

 private:
  static RunContext CreateContext()
  {
    RunContext context;
    context.dataMemSize = 100;
    context.dataMemBase = (uint8_t *) (intptr_t) 1000;
    context.weightMemSize = 200;
    context.weightMemBase = (uint8_t *) (intptr_t) 1100;
    context.weightsBuffer = Buffer(20);

    return context;
  }

  OpDescPtr GreateOpDesc() {
    vector<int64_t> dims = {1, 2, 3, 4};
    GeShape shape(dims);
    shared_ptr<ge::GeTensorDesc> tensor_desc_ptr = make_shared<ge::GeTensorDesc>();
    tensor_desc_ptr->SetShape(shape);
    tensor_desc_ptr->SetDataType(ge::DT_FLOAT);
    tensor_desc_ptr->SetFormat(ge::FORMAT_NCHW);

    OpDescPtr op_desc_ptr = std::make_shared<OpDesc>("test_op_desc", "conv");
    op_desc_ptr->SetId(123456);
    op_desc_ptr->AddInputDesc(tensor_desc_ptr->Clone());
    op_desc_ptr->AddOutputDesc(tensor_desc_ptr->Clone());

    return op_desc_ptr;
  }

  uint64_t GetSeed(uint32_t seed0, uint32_t seed1) {
    uint64_t seed_value = seed1;
    seed_value = seed_value << 32 | seed0;
    return seed_value;
  }

  bool IsSameAddr(const std::string& addr, int64_t offset) {
    uint64_t real_addr = *reinterpret_cast<const uint64_t*>(addr.c_str());
    uint64_t expect_addr = reinterpret_cast<const uint64_t>(context_.dataMemBase) + offset;
    return real_addr == expect_addr;
  }

  void ExpectSameAddr(const std::string& addr, int64_t offset) {
    uint64_t real_addr = *reinterpret_cast<const uint64_t*>(addr.c_str());
    uint64_t expect_addr = reinterpret_cast<const uint64_t>(context_.dataMemBase) + offset;
    EXPECT_EQ(real_addr, expect_addr);
  }

  bool IsSameValue(const std::string& addr, float value) {
    float real_addr = *reinterpret_cast<const float*>(addr.c_str());
    EXPECT_FLOAT_EQ(real_addr, value);
    return real_addr == value;
  }

  bool IsSameValue(const std::string& addr, int64_t value) {
    int64_t real_addr = *reinterpret_cast<const int64_t*>(addr.c_str());
    EXPECT_EQ(real_addr, value);
    return real_addr == value;
  }

  bool IsSameAddr(uint64_t addr, int64_t offset) {
    uint64_t real_addr = addr;
    uint64_t expect_addr = reinterpret_cast<const uint64_t>(context_.dataMemBase) + offset;
    return real_addr == expect_addr;
  }
  static void SetConstInputsValue(const ge::NodePtr &node) {
    auto op = ge::OpDescUtils::CreateOperatorFromNode(node);
    std::vector<std::string> value_list;
    auto input_desc_size = node->GetOpDesc()->GetAllInputsSize();
    for (size_t input_idx = 0; input_idx < input_desc_size; input_idx++) {
      std::string value;
      auto const_tensor = ge::OpDescUtils::GetInputConstData(op, input_idx);
      if (const_tensor == nullptr) {
        value_list.emplace_back(value);
        continue;
      }
      value.assign(reinterpret_cast<const char *>(const_tensor->GetData().GetData()), const_tensor->GetData().GetSize());
      value_list.emplace_back(value);
    }
    (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), kOpConstValueList, value_list);
  }
  static void SetWeight(ge::NodePtr node, int index, float w) {
    float data[] = {w};
    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT16);
    ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data, sizeof(data));
    map<int, ge::GeTensorPtr> weights = {{index, tensor}};
    ge::OpDescUtils::SetWeights(*node, weights);
  }

  static void SetWeight(ge::NodePtr node, int index, int64_t w) {
    int64_t data[] = {w};
    ge::GeTensorDesc tensor_desc(ge::GeShape({1}), ge::FORMAT_ND, ge::DT_INT64);
    ge::GeTensorPtr tensor = std::make_shared<ge::GeTensor>(tensor_desc, (uint8_t *) data, sizeof(data));
    map<int, ge::GeTensorPtr> weights = {{index, tensor}};
    ge::OpDescUtils::SetWeights(*node, weights);
  }

 private:
  OpsKernelBuilderPtr dsa_ops_kernel_builder_ptr_;
  RunContext context_;
  const uint32_t DSA_INT32 = 0b000;
  const uint32_t DSA_INT64 = 0b001;
  const uint32_t DSA_UINT32 = 0b010;
  const uint32_t DSA_UINT64 = 0b011;
  const uint32_t DSA_BF16 = 0b100;
  const uint32_t DSA_FLOAT16 = 0b101;
  const uint32_t DSA_FLOAT = 0b110;

  const uint32_t DSA_BITMASK = 0b000;
  const uint32_t DSA_UNIFORM = 0b001;
  const uint32_t DSA_NORMAL = 0b010;
  const uint32_t DSA_TRUNCATED_NORMAL = 0b011;

  const uint32_t VLD_TRUNCATED_NORMAL = 0b11000;
  const uint32_t VLD_NORMAL = 0b11000;
  const uint32_t VLD_BITMASK = 0b00001;
  const uint32_t VLD_UNIFORM = 0b00110;

  const uint32_t DSA_DATA_VALUE = 0b1;
  const uint32_t DSA_DATA_ADDR = 0b0;
};

TEST_F(DsaOpsKernelBuilderTest, calcoprunningparam_success_1) {
  ge::ComputeGraphPtr graph = make_shared<ge::ComputeGraph>("test_graph");
  OpDescPtr op_desc_ptr = GreateOpDesc();
  NodePtr node = graph->AddNode(op_desc_ptr);

  Status status = dsa_ops_kernel_builder_ptr_->CalcOpRunningParam(*node);
  EXPECT_EQ(fe::SUCCESS, status);
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_normal_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1U);
  auto dsa_task_def = task_defs[0].dsa_task();
  EXPECT_EQ(dsa_task_def.start(), 1U);
  EXPECT_EQ(dsa_task_def.distribution_type(), DSA_NORMAL);
  EXPECT_EQ(dsa_task_def.data_type(), DSA_FLOAT16);
  EXPECT_EQ(dsa_task_def.alg_type(), 0U);
  EXPECT_EQ(dsa_task_def.input_vld(), VLD_TRUNCATED_NORMAL);
  EXPECT_EQ(dsa_task_def.input_value_addr_flag(), 0U);
  EXPECT_EQ(dsa_task_def.input1_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.input2_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.seed_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.random_count_value_or_ptr(), DSA_DATA_ADDR);

  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().output_addr(), 2048+1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_philox_count_addr(), 1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_input_addr(), 2048));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().random_count_value_or_addr(), 0));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().seed_value_or_addr(), 32));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input1_value_or_addr(), 64));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input2_value_or_addr(), 128));
}

TEST_F(DsaOpsKernelBuilderTest, gen_ffts_task_normal_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
   std::vector<domi::TaskDef> task_defs;
  (void)ge::AttrUtils::SetBool(src_op, "_ffts_plus", true);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_truncated_normal_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomTruncatedNormal", "DSARandomTruncatedNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  src_op->SetAttr("seed0", GeAttrValue::CreateFrom<GeAttrValue::INT>(1));
  src_op->SetAttr("seed1", GeAttrValue::CreateFrom<GeAttrValue::INT>(2));
  auto src_node = graph->AddNode(src_op);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1U);
  auto dsa_task_def = task_defs[0].dsa_task();
  EXPECT_EQ(dsa_task_def.start(), 1U);
  EXPECT_EQ(dsa_task_def.distribution_type(), DSA_TRUNCATED_NORMAL);
  EXPECT_EQ(dsa_task_def.data_type(), DSA_FLOAT16);
  EXPECT_EQ(dsa_task_def.alg_type(), 0U);
  EXPECT_EQ(dsa_task_def.input_vld(), VLD_TRUNCATED_NORMAL);
  EXPECT_EQ(dsa_task_def.input_value_addr_flag(), 0U);
  EXPECT_EQ(dsa_task_def.input1_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.input2_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.seed_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.random_count_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().output_addr(), 2048+1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_philox_count_addr(), 1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_input_addr(), 2048));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().random_count_value_or_addr(), 0));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().seed_value_or_addr(), 32));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input1_value_or_addr(), 64));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input2_value_or_addr(), 128));
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_uniform_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1U);
  auto dsa_task_def = task_defs[0].dsa_task();
  EXPECT_EQ(dsa_task_def.start(), 1U);
  EXPECT_EQ(dsa_task_def.distribution_type(), DSA_UNIFORM);
  EXPECT_EQ(dsa_task_def.data_type(), DSA_FLOAT16);
  EXPECT_EQ(dsa_task_def.alg_type(), 0U);
  EXPECT_EQ(dsa_task_def.input_vld(), VLD_UNIFORM);
  EXPECT_EQ(dsa_task_def.input_value_addr_flag(), 0U);
  EXPECT_EQ(dsa_task_def.input1_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.input2_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.seed_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.random_count_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().output_addr(), 2048+1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_philox_count_addr(), 1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_input_addr(), 2048));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().random_count_value_or_addr(), 0));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().seed_value_or_addr(), 32));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input1_value_or_addr(), 64));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input2_value_or_addr(), 128));
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_uniform_with_constant_input0) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
  GeTensorDesc src_tensor_desc(GeShape({1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({1}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);
  SetWeight(src_node, 0, 512L);
  SetWeight(src_node, 2, 0.24f);
  SetConstInputsValue(src_node);
  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1U);
  auto dsa_task_def = task_defs[0].dsa_task();
  EXPECT_EQ(dsa_task_def.start(), 1U);
  EXPECT_EQ(dsa_task_def.distribution_type(), DSA_UNIFORM);
  EXPECT_EQ(dsa_task_def.data_type(), DSA_FLOAT16);
  EXPECT_EQ(dsa_task_def.alg_type(), 0U);
  EXPECT_EQ(dsa_task_def.input_vld(), VLD_UNIFORM);
  EXPECT_EQ(dsa_task_def.input_value_addr_flag(), 0b1000010U);
  EXPECT_EQ(dsa_task_def.input1_value_or_ptr(), DSA_DATA_VALUE);
  EXPECT_EQ(dsa_task_def.input2_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.random_count_value_or_ptr(), DSA_DATA_VALUE);
  EXPECT_EQ(dsa_task_def.seed_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().output_addr(), 2048+1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_philox_count_addr(), 1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_input_addr(), 2048));
  EXPECT_TRUE(IsSameValue(dsa_task_def.args().random_count_value_or_addr(), 512L));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().seed_value_or_addr(), 32));
  EXPECT_TRUE(IsSameValue(dsa_task_def.args().input1_value_or_addr(), 0.24f));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input2_value_or_addr(), 128));
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_uniform_with_constant_input1) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr random_uniform = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
  OpDescPtr count = std::make_shared<OpDesc>("count", "Data");
  OpDescPtr seed = std::make_shared<OpDesc>("seed", "Data");
  OpDescPtr low = std::make_shared<OpDesc>("low", "Const");
  OpDescPtr high = std::make_shared<OpDesc>("DSARandomUniform", "Data");

  GeTensorDesc input_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_FLOAT);
  input_desc.SetOriginShape(GeShape({1}));
  input_desc.SetOriginFormat(ge::FORMAT_ND);
  GeTensorDesc seed_desc(GeShape({1}), ge::FORMAT_ND, ge::DT_INT32);
  seed_desc.SetOriginShape(GeShape({1}));
  seed_desc.SetOriginFormat(ge::FORMAT_ND);

  random_uniform->AddInputDesc("count", seed_desc);
  random_uniform->AddInputDesc("seed", seed_desc);
  random_uniform->AddInputDesc("low", input_desc);
  random_uniform->AddInputDesc("high", input_desc);
  random_uniform->AddOutputDesc(input_desc);

  count->AddOutputDesc(seed_desc);
  seed->AddOutputDesc(seed_desc);
  low->AddOutputDesc(input_desc);
  high->AddOutputDesc(input_desc);

  random_uniform->SetWorkspace({1024, 2048});
  random_uniform->SetOutputOffset({2048+1024});
  random_uniform->SetInputOffset({0, 32, 64, 128});

  auto random_uniform_node = graph->AddNode(random_uniform);
  auto count_node = graph->AddNode(count);
  auto seed_node = graph->AddNode(seed);
  auto low_node = graph->AddNode(low);
  auto high_node = graph->AddNode(high);


  GraphUtils::AddEdge(count_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(seed_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(low_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(2));
  GraphUtils::AddEdge(high_node->GetOutDataAnchor(0), random_uniform_node->GetInDataAnchor(3));

  float low_value = 0.24f;
  GeTensor const_low(input_desc, (uint8_t*)(&low_value), sizeof(low_value));
  ge::AttrUtils::SetTensor(low, "value", const_low);

  std::vector<domi::TaskDef> task_defs;
  SetConstInputsValue(random_uniform_node);
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*random_uniform_node, context_, task_defs);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1U);
  auto dsa_task_def = task_defs[0].dsa_task();
  EXPECT_EQ(dsa_task_def.start(), 1U);
  EXPECT_EQ(dsa_task_def.distribution_type(), DSA_UNIFORM);
  EXPECT_EQ(dsa_task_def.data_type(), DSA_FLOAT);
  EXPECT_EQ(dsa_task_def.alg_type(), 0U);
  EXPECT_EQ(dsa_task_def.input_vld(), VLD_UNIFORM);
  EXPECT_EQ(dsa_task_def.input_value_addr_flag(), 0b0000010U);
  EXPECT_EQ(dsa_task_def.input1_value_or_ptr(), DSA_DATA_VALUE);
  EXPECT_EQ(dsa_task_def.input2_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.random_count_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.seed_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().output_addr(), 2048+1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_philox_count_addr(), 1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_input_addr(), 2048));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().random_count_value_or_addr(), 0));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().seed_value_or_addr(), 32));
  EXPECT_TRUE(IsSameValue(dsa_task_def.args().input1_value_or_addr(), 0.24f));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input2_value_or_addr(), 128));
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_uniform_with_all_constant_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUniform", "DSARandomUniform");
  GeTensorDesc src_tensor_desc(GeShape({1}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({1}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);
  SetWeight(src_node, 0, 256L);
  SetWeight(src_node, 1, 512L);
  SetWeight(src_node, 2, 0.12f);
  SetWeight(src_node, 3, 0.24f);
  SetConstInputsValue(src_node);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1U);
  auto dsa_task_def = task_defs[0].dsa_task();
  EXPECT_EQ(dsa_task_def.start(), 1U);
  EXPECT_EQ(dsa_task_def.distribution_type(), DSA_UNIFORM);
  EXPECT_EQ(dsa_task_def.data_type(), DSA_FLOAT16);
  EXPECT_EQ(dsa_task_def.alg_type(), 0U);
  EXPECT_EQ(dsa_task_def.input_vld(), VLD_UNIFORM);
  EXPECT_EQ(dsa_task_def.input_value_addr_flag(), 0b1100110U);
  EXPECT_EQ(dsa_task_def.input1_value_or_ptr(), DSA_DATA_VALUE);
  EXPECT_EQ(dsa_task_def.input2_value_or_ptr(), DSA_DATA_VALUE);
  EXPECT_EQ(dsa_task_def.random_count_value_or_ptr(), DSA_DATA_VALUE);
  EXPECT_EQ(dsa_task_def.seed_value_or_ptr(), DSA_DATA_VALUE);
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().output_addr(), 2048+1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_philox_count_addr(), 1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_input_addr(), 2048));
  EXPECT_TRUE(IsSameValue(dsa_task_def.args().random_count_value_or_addr(), 256L));
  EXPECT_TRUE(IsSameValue(dsa_task_def.args().seed_value_or_addr(), 512L));
  EXPECT_TRUE(IsSameValue(dsa_task_def.args().input1_value_or_addr(), 0.12f));
  EXPECT_TRUE(IsSameValue(dsa_task_def.args().input2_value_or_addr(), 0.24f));
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_bitmask_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSAGenBitMask", "DSAGenBitMask");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64});
  auto src_node = graph->AddNode(src_op);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(task_defs.size(), 1U);
  auto dsa_task_def = task_defs[0].dsa_task();
  EXPECT_EQ(dsa_task_def.start(), 1U);
  EXPECT_EQ(dsa_task_def.distribution_type(), DSA_BITMASK);
  EXPECT_EQ(dsa_task_def.data_type(), DSA_FLOAT16);
  EXPECT_EQ(dsa_task_def.alg_type(), 0U);
  EXPECT_EQ(dsa_task_def.input_vld(), VLD_BITMASK);
  EXPECT_EQ(dsa_task_def.input_value_addr_flag(), 0U);
  EXPECT_EQ(dsa_task_def.input1_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.input2_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.seed_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_EQ(dsa_task_def.random_count_value_or_ptr(), DSA_DATA_ADDR);
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().output_addr(), 2048+1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_philox_count_addr(), 1024));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().workspace_input_addr(), 2048));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().random_count_value_or_addr(), 0));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().seed_value_or_addr(), 32));
  EXPECT_TRUE(IsSameAddr(dsa_task_def.args().input1_value_or_addr(), 64));
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_unknown_optype) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomUnknown", "DSARandomUnknown");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64});
  src_op->SetAttr("seed0", GeAttrValue::CreateFrom<GeAttrValue::INT>(1));
  src_op->SetAttr("seed1", GeAttrValue::CreateFrom<GeAttrValue::INT>(2));
  auto src_node = graph->AddNode(src_op);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::FAILED);
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_unsupported_dtype) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_INT8);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64});
  src_op->SetAttr("seed0", GeAttrValue::CreateFrom<GeAttrValue::INT>(1));
  src_op->SetAttr("seed1", GeAttrValue::CreateFrom<GeAttrValue::INT>(2));
  auto src_node = graph->AddNode(src_op);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::FAILED);
}

TEST_F(DsaOpsKernelBuilderTest, gen_task_invalid_input_count) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_INT32);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32});
  src_op->SetAttr("seed0", GeAttrValue::CreateFrom<GeAttrValue::INT>(1));
  src_op->SetAttr("seed1", GeAttrValue::CreateFrom<GeAttrValue::INT>(2));
  auto src_node = graph->AddNode(src_op);

  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  ASSERT_EQ(ret, ge::FAILED);
}

TEST_F(DsaOpsKernelBuilderTest, generate_dsa_static_const_as_value_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const3 = std::make_shared<OpDesc>("const3", "Const");
  OpDescPtr const4 = std::make_shared<OpDesc>("const4", "Const");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  const1->AddOutputDesc(src_tensor_desc);
  const2->AddOutputDesc(src_tensor_desc);
  const3->AddOutputDesc(src_tensor_desc);
  const4->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto const1_node = graph->AddNode(const1);
  auto const2_node = graph->AddNode(const2);
  auto const3_node = graph->AddNode(const3);
  auto const4_node = graph->AddNode(const4);
  auto src_node = graph->AddNode(src_op);

  ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(3));
  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::FAILED);
}

TEST_F(DsaOpsKernelBuilderTest, generate_dsa_dynamic_const_as_addr_input) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr const1 = std::make_shared<OpDesc>("const1", "Const");
  OpDescPtr const2 = std::make_shared<OpDesc>("const2", "Const");
  OpDescPtr const3 = std::make_shared<OpDesc>("const3", "Const");
  OpDescPtr const4 = std::make_shared<OpDesc>("const4", "Const");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  const1->AddOutputDesc(src_tensor_desc);
  const2->AddOutputDesc(src_tensor_desc);
  const3->AddOutputDesc(src_tensor_desc);
  const4->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->SetWorkspace({1024, 2048});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto const1_node = graph->AddNode(const1);
  auto const2_node = graph->AddNode(const2);
  auto const3_node = graph->AddNode(const3);
  auto const4_node = graph->AddNode(const4);
  auto src_node = graph->AddNode(src_op);

  ge::GraphUtils::AddEdge(const1_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(const2_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(const3_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(const4_node->GetOutDataAnchor(0),
                          src_node->GetInDataAnchor(3));
  (void)ge::AttrUtils::SetBool(graph, "_graph_unknown_flag", true);
  std::vector<domi::TaskDef> task_defs;
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(DsaOpsKernelBuilderTest, workspace_size_1_generate_dsa_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test_graph_input");
  OpDescPtr src_op = std::make_shared<OpDesc>("DSARandomNormal", "DSARandomNormal");
  GeTensorDesc src_tensor_desc(GeShape({100, 2, 3, 512, 4}), ge::FORMAT_NC1HWC0, ge::DT_FLOAT16);
  src_tensor_desc.SetOriginShape(GeShape({10, 11, 12, 13}));
  src_tensor_desc.SetOriginFormat(ge::FORMAT_NHWC);
  src_op->AddOutputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
  src_op->AddInputDesc(src_tensor_desc);
   std::vector<domi::TaskDef> task_defs;
  (void)ge::AttrUtils::SetBool(src_op, "_ffts_plus", false);
  src_op->SetWorkspace({1024});
  src_op->SetOutputOffset({2048+1024});
  src_op->SetInputOffset({0, 32, 64, 128});
  auto src_node = graph->AddNode(src_op);
  Status ret = dsa_ops_kernel_builder_ptr_->GenerateTask(*src_node, context_, task_defs);
  EXPECT_EQ(ret, ge::SUCCESS);
}
