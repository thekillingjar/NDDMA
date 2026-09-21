
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <chrono>
#include <gtest/gtest.h>

#include "graph/ascendc_ir/ascendc_ir_core/ascendc_ir.h"
#include "graph/attribute_group/attr_group_symbolic_desc.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_op_types.h"
#include "attribute_group/attr_group_shape_env.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/node_adapter.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"

#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "lowering/lowerings.h"
#include "lowering/asc_ir_lowerer.h"
#include "can_fuse/fusion_strategy_solver.h"
#include "can_fuse/backend/fusion_decider_registry.h"
#include "can_fuse/backend/asc_backend_fusion_decider.h"
#include "post_process/scheduler_adapter/adaption_fallback_load.h"
#include "utils/autofuse_attrs.h"
#include "util/mem_utils.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "platform_context.h"

#include "expression/testcase/source_stub.h"
#include "depends/runtime/src/runtime_stub.h"
#include "all_ops_cpp.h"
#include "compliant_op_desc_builder.h"
#include "esb_graph.h"
#include "op_creator_register.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;
namespace {
REGISTER_LOWERING(DynamicQuantStub) {
  (void)loop::Store(node->GetOutDataAnchor(0), loop::Abs(loop::Load(node->GetInDataAnchor(0))));
  (void)loop::Store(node->GetOutDataAnchor(1), loop::Exp(loop::Load(node->GetInDataAnchor(0))));
  return GRAPH_SUCCESS;
}

std::string ReadableComputeGraph(const ComputeGraphPtr &graph, bool only_can_reached = true) {
  std::stringstream ss;
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  ss << "ComputeGraph(" << graph->GetName() << ")" << std::endl;
  std::set<NodePtr> can_reached;
  std::stack<NodePtr> stack;
  auto sink = graph->FindFirstNodeMatchType(NETOUTPUT);
  can_reached.insert(sink);
  if (sink != nullptr) {
    stack.push(sink);
    while (!stack.empty()) {
      auto current = stack.top();
      stack.pop();
      for (auto &in_node : current->GetInAllNodes()) {
        if (can_reached.insert(in_node).second) {
          stack.push(in_node);
        }
      }
    }
  }
  std::vector<std::string> unused_nodes;
  for (const auto &node : graph->GetAllNodes()) {
    if (only_can_reached && can_reached.find(node) == can_reached.end()) {
      unused_nodes.emplace_back(node->GetName());
      continue;
    }
    std::vector<std::string> input_names;
    std::vector<std::string> control_names;
    for (auto &anchor : node->GetAllInDataAnchors()) {
      auto peer = anchor->GetPeerOutAnchor();
      if (peer == nullptr) {
        continue;
      }
      input_names.emplace_back(anchor_name[peer]);
    }
    for (auto &in_control : node->GetInControlNodes()) {
      control_names.emplace_back(in_control->GetName());
    }
    std::vector<std::string> output_names;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
    }
    if (output_names.size() > 1U) {
      ss << loop::StrJoin(output_names) << " = ";
    } else if (!output_names.empty()) {
      ss << output_names[0] << " = ";
    }
    if (control_names.empty()) {
      ss << "ge." << node->GetType() << "(" << node->GetName() << ", " << loop::StrJoin(input_names) << ")"
         << std::endl;
    } else {
      ss << "ge." << node->GetType() << "(" << node->GetName() << ", " << loop::StrJoin(input_names) << ", "
         << loop::StrJoin(control_names) << ")" << std::endl;
    }
  }
  ss << "ununsed nodes: " << loop::StrJoin(unused_nodes) << std::endl;
  return ss.str();
}

template <typename T>
es::Tensor CreateConst(es::Graph &graph, ge::DataType dtype, const std::vector<int64_t> &dims, std::vector<T> value) {
  auto result = es::FileConstant(graph, dims, dtype);
  GeTensorDesc desc(GeShape(dims), ge::FORMAT_ND, dtype);
  GeTensorPtr tensor =
      std::make_shared<GeTensor>(desc, reinterpret_cast<uint8_t *>(value.data()), sizeof(T) * value.size());
  AttrUtils::SetTensor(result.GetEsbTensor()->GetProducer()->GetOpDesc(), "value", tensor);
  result.GetEsbTensor()->GetProducer()->GetOpDesc()->SetType(ge::CONSTANT);
  return result;
}
}  // namespace

class LoopAscIrLowerPrunerUTV2 : public testing::Test {
public:
protected:
  void SetUp() override {
    AutoFuseConfig::MutableConfig().GetMutableFusionStrategySolver().max_fusion_size = 64U;
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
    dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
    RuntimeStub::SetInstance(stub_v2);
  }
  void TearDown() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
  }
  std::unique_ptr<es::Graph> es_graph_;
};


TEST_F(LoopAscIrLowerPrunerUTV2, GraphLiftingAfterSplitLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "16"});
    auto split_dim = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto split = es::Split(split_dim,data0,2);
    split[0].SetSymbolShape({"192", "32", "16"});
    split[1].SetSymbolShape({"192", "32", "16"});
    auto abs = es::Abs(split[0]);
    abs.SetSymbolShape({"192", "32", "16"});
    auto perms0 = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose0 = es::Transpose(split[0],perms0);
    transpose0.SetSymbolShape({"192", "32", "16"});
    auto transpose1 = es::TransposeD(split[1],{2,1,0});
    transpose1.SetSymbolShape({"192", "32", "16"});
    es_graph_->SetOutput(abs, 0);
    es_graph_->SetOutput(transpose0, 1);
    es_graph_->SetOutput(transpose1, 2);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindFirstNodeMatchType("Split")->GetOpDesc()->SetType("Split");
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  EXPECT_EQ(asc_adapt::GeFallback(cg),SUCCESS);
  FusionStrategySolver fusion_strategy_solver;
  FusionDeciderRegistry::Instance().Register(std::unique_ptr<FusionDecider>(new AscBackendFusionDecider()));
  EXPECT_EQ(fusion_strategy_solver.Fuse(cg), SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  std::string expected1 = R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Const(FileConstant_0, [])
[tmp2, tmp3, tmp4] = ge.AscBackend(fused_graph_1582, [tmp0], [FileConstant_0])
tmp5 = ge.TransposeD(TransposeD_5, [tmp2])
tmp6 = ge.Const(FileConstant_3, [])
tmp7 = ge.Transpose(Transpose_4, [tmp3, tmp6])
[tmp8, tmp9, tmp10] = ge.NetOutput(NetOutput, [tmp4, tmp7, tmp5])
ununsed nodes: []
)";

  std::string expected2 = R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Const(FileConstant_0, [])
tmp3 = ge.TransposeD(TransposeD_5, [], [Split_1, Abs_2])
[tmp4, tmp5] = ge.Split(Split_1, [tmp1, tmp0])
tmp5 = ge.Abs(Abs_2, [tmp4])
tmp6 = ge.Const(FileConstant_3, [])
tmp7 = ge.Transpose(Transpose_4, [tmp4, tmp6], [Split_1, Abs_2])
[tmp8, tmp9, tmp10] = ge.NetOutput(NetOutput, [tmp5, tmp7, tmp3], [Split_1, Split_1])
ununsed nodes: []
)";

  std::string actual = ReadableComputeGraph(cg);

  // EXPECT_TRUE(actual == expected1 || actual == expected2)
  //     << "Actual output:\n" << actual
  //     << "\n\nDid not match either expected version.";
}

TEST_F(LoopAscIrLowerPrunerUTV2, SimpleSkipLifting) {
  AutoFuseConfig::MutableLoweringConfig().experimental_disable_lifting = true;
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(x);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(x, [])
tmp1 = ge.Abs(Abs_0, [tmp0])
tmp2 = ge.AscBackend(autofuse_pointwise_0_Abs, [tmp0])
tmp3 = ge.NetOutput(NetOutput, [tmp2])
ununsed nodes: []
)");
  AutoFuseConfig::MutableLoweringConfig().experimental_disable_lifting = false;
}

TEST_F(LoopAscIrLowerPrunerUTV2, TestNoExtraDataOutputAfterCanFuseLiftingBothWithMultiOutNode) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0"});
    auto relu = es::Relu(abs);
    relu.SetSymbolShape({"s0"});
    auto maximum = es::Maximum(relu, data1);
    maximum.SetSymbolShape({"s0"});
    auto square_diff = es::SquaredDifference(relu, maximum);
    square_diff.SetSymbolShape({"s0"});
    auto concat = es::ConcatD({relu, data2}, 0);  // no sym for trigger realize
    es_graph_->SetOutput(square_diff, 0);
    es_graph_->SetOutput(concat, 1);
  }();
  ge::PlatformContext::GetInstance().Reset();
  auto stub_v2 = std::make_shared<ge::RuntimeStubV2Common>();
  ge::RuntimeStub::SetInstance(stub_v2);
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ge::AscIrLowerer lowerer;
  ASSERT_EQ(lowerer.Lowering(cg), GRAPH_SUCCESS);
  ge::RuntimeStub::Reset();
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Data(data1, [])
tmp2 = ge.Data(data2, [])
tmp3 = ge.Abs(Abs_0, [tmp0])
tmp4 = ge.Relu(Relu_1, [tmp3])
tmp5 = ge.AscBackend(autofuse_pointwise_0_Abs_Relu, [tmp0])
tmp6 = ge.Maximum(Maximum_2, [tmp1])
tmp7 = ge.SquaredDifference(SquaredDifference_3, [tmp6])
tmp8 = ge.AscBackend(autofuse_pointwise_1_Maximum_SquaredDifference, [tmp5, tmp1])
tmp9 = ge.ConcatD(ConcatD_4, [tmp5, tmp2])
[tmp10, tmp11] = ge.NetOutput(NetOutput, [tmp8, tmp9])
ununsed nodes: []
)");
  ASSERT_EQ(lowerer.Lifting(cg), GRAPH_SUCCESS);
  EXPECT_EQ(ReadableComputeGraph(cg, false), R"(ComputeGraph(graph)
tmp0 = ge.Data(data0, [])
tmp1 = ge.Data(data1, [])
tmp2 = ge.Data(data2, [])
tmp3 = ge.AscBackend(autofuse_pointwise_0_Abs_Relu, [tmp0])
tmp4 = ge.AscBackend(autofuse_pointwise_1_Maximum_SquaredDifference, [tmp3, tmp1])
tmp5 = ge.ConcatD(ConcatD_4, [tmp3, tmp2])
[tmp6, tmp7] = ge.NetOutput(NetOutput, [tmp4, tmp5])
ununsed nodes: []
)");
}
}  // namespace ge
