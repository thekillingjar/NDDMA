
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_op_types.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/debug/ge_attr_define.h"
#include "../../eager_style_graph_builder/all_ops_cpp.h"
#include "../../eager_style_graph_builder/esb_graph.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_lowerer/asc_overrides.h"
#include "lowering/lowerings.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"
#include "../../eager_style_graph_builder/compliant_op_desc_builder.h"
#include "pattern_fusion/flatten_concat_pass.h"
#include "pattern_fusion/flatten_split_pass.h"
#include "pattern_fusion/pattern_fusion.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include <gtest/gtest.h>
#include "op_creator_register.h"
#include "pattern_fusion/gather_forward_fusion_pass.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/asc_backend_post_processor.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "base/att_const_values.h"
#include "depends/runtime/src/runtime_stub.h"
#include "platform_context.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "ascgen_log.h"

using namespace std;
using namespace testing;
namespace ge {
class PatternFusionBeforeAutoFuseV2UT : public testing::Test {
  public:
  protected:
    void SetUp() override {
      dlog_setlevel(0, 3, 0);
      es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
      RegisterAllOpCreator();
    }
    void TearDown() override {
       dlog_setlevel(0, 3, 0);
    }
    std::unique_ptr<es::Graph> es_graph_;
};
   
template <typename T>
es::Tensor CreateConstTensor(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims, std::vector<T> value) {
  auto result = es::FileConstant(graph, dims, dtype);
  GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
  GeTensorPtr tensor =
      std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(value.data()), sizeof(T) * value.size());
  AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
  return result;
}

uint8_t CountAscSubgraphNode(const NodePtr & AscNode , const string &node_type) {
  const auto attr = AscNode->GetOpDesc()->GetAttrsGroup<ge::AutoFuseAttrs>();
  uint8_t count = 0;
  for (const auto &node : attr->GetAscGraph()->GetAllNodes()) {
    if (node->GetType() == node_type) {
      count++;
    }
  }
  return count;
}

REG_OP(Relu)
  .INPUT(x, TensorType::UnaryDataType())
  .OUTPUT(y, TensorType::UnaryDataType())
  .ATTR(base, Float, -1.0)
  .ATTR(scale, Float, 1.0)
  .ATTR(shift, Float, 0.0)
  .OP_END_FACTORY_REG(Relu)
REG_OP(Exp)
  .INPUT(x, TensorType::UnaryDataType())
  .OUTPUT(y, TensorType::UnaryDataType())
  .ATTR(base, Float, -1.0)
  .OP_END_FACTORY_REG(Exp)
REG_OP(Abs)
  .INPUT(x, TensorType::UnaryDataType())
  .OUTPUT(y, TensorType::UnaryDataType())
  .ATTR(base, Float, -1.0)
  .ATTR(scale, Float, 1.0)
  .ATTR(shift, Float, 0.0)
  .OP_END_FACTORY_REG(Abs)

const auto ReluInfer = [](Operator &op) {
  return GRAPH_SUCCESS;
};

INFER_FUNC_REG(Relu, ReluInfer);
INFER_FUNC_REG(Abs, ReluInfer);
INFER_FUNC_REG(Exp, ReluInfer);

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_Tail_A3) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(abs1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(relu1, data1, axis);
    gather.SetSymbolShape({"s0", "s1", "s3", "s4"});
    auto relu2 = es::Relu(gather);
    relu2.SetSymbolShape({"s0", "s1", "s3", "s4"});
    auto abs2 = es::Abs(relu2);
    abs2.SetSymbolShape({"s0", "s1", "s3", "s4"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 3);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 2);
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A3) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(abs1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(relu1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu2 = es::Relu(gather);
    relu2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs2 = es::Abs(relu2);
    abs2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 2);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 0);
  ASSERT_EQ(asc_node_count_after_autofuse, 2);
}


TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_Tail_A5_1) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(abs1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(relu1, data1, axis);
    gather.SetSymbolShape({"s0", "s1", "s3", "s4"});
    auto relu2 = es::Relu(gather);
    relu2.SetSymbolShape({"s0", "s1", "s3", "s4"});
    auto abs2 = es::Abs(relu2);
    abs2.SetSymbolShape({"s0", "s1", "s3", "s4"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 2);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_Tail_A5_2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"s5", "s6", "s7"});
    auto axis1 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    axis1.SetSymbolShape({});
    auto axis2 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    axis2.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(data2);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    abs2.SetInputSymbolShape({"s0", "s1", "s2"});
    auto add1 = es::Add(abs1, abs2);
    add1.SetSymbolShape({"s0", "s1", "s2"});
    add1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(add1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather1 = es::GatherV2(relu1, data1, axis1);
    gather1.SetSymbolShape({"s0", "s1", "s3", "s4"});
    auto exp1 = es::Exp(gather1);
    exp1.SetSymbolShape({"s0", "s1", "s3", "s4"});
    exp1.SetInputSymbolShape({"s0", "s1", "s3", "s4"});
    auto abs3 = es::Abs(exp1);
    abs3.SetSymbolShape({"s0", "s1", "s3", "s4"});
    abs3.SetInputSymbolShape({"s0", "s1", "s3", "s4"});
    auto gather2 = es::GatherV2(abs3, data3, axis2);
    gather2.SetSymbolShape({"s0", "s1", "s3", "s5", "s6", "s7"});
    auto relu2 = es::Relu(gather2);
    relu2.SetSymbolShape({"s0", "s1", "s3", "s5", "s6", "s7"});
    auto abs4 = es::Abs(relu2);
    abs4.SetSymbolShape({"s0", "s1", "s3", "s5", "s6", "s7"});
    es_graph_->SetOutput(abs4, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  nodeptr = graph->FindNode("data3");
  ASSERT_NE(nodeptr, nullptr);
  tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);


  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);

  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 5);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 2);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}


TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_1) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(abs1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(relu1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu2 = es::Relu(gather);
    relu2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs2 = es::Abs(relu2);
    abs2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 2);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_2) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto exp1 = es::Exp(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    exp1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto abs1 = es::Abs(exp1);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(abs1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto exp2 = es::Exp(gather);
    exp2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu1 = es::Relu(exp2);
    relu1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs2 = es::Abs(relu1);
    abs2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 4);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_3) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu1 = es::Relu(gather);
    relu1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs1 = es::Abs(relu1);
    abs1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 2);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_4) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto exp1 = es::Exp(data0);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    exp1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(exp1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu1 = es::Relu(gather);
    relu1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs1 = es::Abs(relu1);
    abs1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs1, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 3);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_5) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(abs1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu1 = es::Relu(gather);
    relu1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    relu1.SetInputSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs2 = es::Abs(relu1);
    abs2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    abs2.SetInputSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 2);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_6) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(abs1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu1 = es::Relu(gather);
    relu1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs2 = es::Abs(relu1);
    abs2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs2, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 2);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_7) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto mul1 = es::Mul(abs1, data2);
    mul1.SetSymbolShape({"s0", "s1", "s2"});
    mul1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(mul1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(relu1);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    abs2.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(abs2, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu2 = es::Relu(gather);
    relu2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs3 = es::Abs(relu2);
    abs3.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs3, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);


  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);

  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 3);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 2);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_8) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(data2);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    abs2.SetInputSymbolShape({"s0", "s1", "s2"});
    auto add1 = es::Add(abs1, abs2);
    add1.SetSymbolShape({"s0", "s1", "s2"});
    add1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(add1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Exp(relu1);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    exp1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(exp1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu2 = es::Relu(gather);
    relu2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs3 = es::Abs(relu2);
    abs3.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(abs3, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);


  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);

  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 4);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 2);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_9) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"s5", "s6", "s7"});
    auto axis1 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis1.SetSymbolShape({});
    auto axis2 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    axis2.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(data2);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    abs2.SetInputSymbolShape({"s0", "s1", "s2"});
    auto add1 = es::Add(abs1, abs2);
    add1.SetSymbolShape({"s0", "s1", "s2"});
    add1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(add1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather1 = es::GatherV2(relu1, data1, axis1);
    gather1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto exp1 = es::Exp(gather1);
    exp1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs3 = es::Abs(exp1);
    abs3.SetSymbolShape({"s3", "s4", "s1", "s2"});
    abs3.SetInputSymbolShape({"s3", "s4", "s1", "s2"});
    auto gather2 = es::GatherV2(abs3, data3, axis2);
    gather2.SetSymbolShape({"s3", "s5", "s6", "s7", "s1", "s2"});
    auto relu2 = es::Relu(gather2);
    relu2.SetSymbolShape({"s3", "s5", "s6", "s7", "s1", "s2"});
    auto abs4 = es::Abs(relu2);
    abs4.SetSymbolShape({"s3", "s5", "s6", "s7", "s1", "s2"});
    es_graph_->SetOutput(abs4, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  nodeptr = graph->FindNode("data3");
  ASSERT_NE(nodeptr, nullptr);
  tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);


  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);

  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 5);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 2);
  ASSERT_EQ(asc_node_count_after_autofuse, 3);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_10) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"s5", "s6", "s7"});
    auto data4 = es_graph_->CreateInput(4, "data4", nullptr);
    data4.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto axis1 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis1.SetSymbolShape({});
    auto axis2 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    axis2.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto abs2 = es::Abs(data2);
    abs2.SetSymbolShape({"s0", "s1", "s2"});
    abs2.SetInputSymbolShape({"s0", "s1", "s2"});
    auto add1 = es::Add(abs1, abs2);
    add1.SetSymbolShape({"s0", "s1", "s2"});
    add1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(add1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather1 = es::GatherV2(relu1, data1, axis1);
    gather1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto exp1 = es::Exp(gather1);
    exp1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto mul1 = es::Mul(exp1, data4);
    mul1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs3 = es::Abs(mul1);
    abs3.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto gather2 = es::GatherV2(abs3, data3, axis2);
    gather2.SetSymbolShape({"s3", "s5", "s6", "s7", "s1", "s2"});
    auto relu2 = es::Relu(gather2);
    relu2.SetSymbolShape({"s3", "s5", "s6", "s7", "s1", "s2"});
    auto abs4 = es::Abs(relu2);
    abs4.SetSymbolShape({"s3", "s5", "s6", "s7", "s1", "s2"});
    es_graph_->SetOutput(abs4, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  nodeptr = graph->FindNode("data3");
  ASSERT_NE(nodeptr, nullptr);
  tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);


  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);

  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 6);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 2);
  ASSERT_EQ(asc_node_count_after_autofuse, 3);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_11) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"s3", "s4"});
    auto data4 = es_graph_->CreateInput(4, "data4", nullptr);
    data4.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto axis1 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis1.SetSymbolShape({});
    auto axis2 = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis2.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto relu1 = es::Relu(abs1);
    relu1.SetSymbolShape({"s0", "s1", "s2"});
    relu1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Exp(relu1);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    exp1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather1=es::GatherV2(exp1, data1, axis1);
    gather1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs2 = es::Abs(gather1);
    abs2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto gather2 = es::GatherV2(relu1, data3, axis2);
    gather2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto add1 = es::Add(abs2, gather2);
    add1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(add1, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  nodeptr = graph->FindNode("data3");
  ASSERT_NE(nodeptr, nullptr);
  tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);


  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);

  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 5);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 2);
  ASSERT_EQ(asc_node_count_after_autofuse, 3);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_12) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"s0", "s1", "s2"});
    abs1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto exp1 = es::Exp(abs1);
    exp1.SetSymbolShape({"s0", "s1", "s2"});
    exp1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(exp1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto relu1 = es::Relu(gather);
    relu1.SetSymbolShape({"s3", "s4", "s1", "s2"});
    auto abs2 = es::Abs(gather);
    abs2.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(relu1, 0);
    es_graph_->SetOutput(abs2, 1);
  }();


  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 4);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 1);
  ASSERT_EQ(asc_node_count_after_autofuse, 1);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_13) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto cast1=es::Cast(data0, DT_FLOAT);
    cast1.SetSymbolShape({"s0", "s1", "s2"});
    cast1.SetInputSymbolShape({"s0", "s1", "s2"});
    auto gather = es::GatherV2(cast1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(gather, 0);
  }();


  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  nodeptr = graph->FindNode("data0");
  ASSERT_NE(nodeptr, nullptr);
  tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  nodeptr = graph->FindNode("Cast_1");
  ASSERT_NE(nodeptr, nullptr);
  tmp_desc = nodeptr->GetOpDesc()->MutableInputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 2);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 0);
  ASSERT_EQ(asc_node_count_after_autofuse, 0);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_14) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto squeeze_axis = std::vector<int64_t>({0});
    auto squeeze1=es::Squeeze(data0, squeeze_axis);
    squeeze1.SetSymbolShape({"s0", "s1", "s2"});
    squeeze1.SetInputSymbolShape({"s0", "s1", "s2", "s5"});
    auto gather = es::GatherV2(squeeze1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(gather, 0);
  }();

  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 1);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 0);
  ASSERT_EQ(asc_node_count_after_autofuse, 0);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}

TEST_F(PatternFusionBeforeAutoFuseV2UT, GatherForward_NonTail_A5_15) {
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
  RuntimeStub::SetInstance(stub_v2);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConstTensor(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{0});
    axis.SetSymbolShape({});
    auto squeeze_axis = std::vector<int64_t>({0});
    auto unsqueeze1=es::Unsqueeze(data0, squeeze_axis);
    unsqueeze1.SetSymbolShape({"s0", "s1", "s2"});
    unsqueeze1.SetInputSymbolShape({"s0", "s1"});
    auto gather = es::GatherV2(unsqueeze1, data1, axis);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(gather, 0);
  }();


  auto graph_es = es_graph_->Build();
  auto graph = GraphUtilsEx::GetComputeGraph(*graph_es);
  auto nodeptr = graph->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  GatherForwardFusionPass GatherForwardFusionPassTest;
  auto result = GatherForwardFusionPassTest.Run(graph);
  ASSERT_EQ(result, GRAPH_SUCCESS);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(graph), GRAPH_SUCCESS);
  ASSERT_EQ(asc_adapt::GeFallback(graph), GRAPH_SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  size_t asc_node_count_after_lowering = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      asc_node_count_after_lowering++;
    }
  }
  ASSERT_EQ(asc_node_count_after_lowering, 1);
  EXPECT_EQ(fusion_strategy_solver.Fuse(graph), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(graph), GRAPH_SUCCESS);
  AscBackendPostProcessor post_processor;
  EXPECT_EQ(post_processor.Do(graph), SUCCESS);
  size_t asc_node_count_after_autofuse_gather = 0;
  size_t asc_node_count_after_autofuse = 0;
  for (auto &node : graph->GetDirectNode()) {
    if (node->GetType() == kAscBackendType) {
      if (CountAscSubgraphNode(node, att::kGather) == 1) {
        asc_node_count_after_autofuse_gather++;
      }
      asc_node_count_after_autofuse++;
    }
  }
  ASSERT_EQ(asc_node_count_after_autofuse_gather, 0);
  ASSERT_EQ(asc_node_count_after_autofuse, 0);
  SetCurShapeEnvContext(nullptr);
  ge::PlatformContext::GetInstance().Reset();
  RuntimeStub::Reset();
}
}