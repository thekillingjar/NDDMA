
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

#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_lowerer/asc_overrides.h"
#include "lowering/lowerings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"
#include "backend/backend_spec.h"
#include "platform_context.h"

#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "compliant_op_desc_builder.h"
#include "depends/runtime/src/runtime_stub.h"
#include <gtest/gtest.h>

using namespace std;
using namespace testing;

namespace ge {
using namespace autofuse;
namespace {
std::string GetAscTensorLoop(const OutDataAnchorPtr &anchor) {
  auto attr = anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(anchor->GetIdx())->GetAttrsGroup<AscTensorAttr>();
  if (attr == nullptr || (attr->axis.empty() && attr->repeats.empty() && attr->strides.empty())) {
    return "";
  }
  std::stringstream ss;
  const static auto kExpressionStr = [](const Expression &e) { return std::string(e.Str().get()); };
  ss << "axis = " << loop::StrJoin(attr->axis, [](const int64_t &e) { return std::to_string(e); });
  ss << ", repeats = " << loop::StrJoin(attr->repeats, kExpressionStr);
  ss << ", strides = " << loop::StrJoin(attr->strides, kExpressionStr);
  return ss.str();
}

std::string ReadableAscGraph(const AscGraph &asc_graph, bool trip_scope = true) {
  std::stringstream ss;
  std::map<OutDataAnchorPtr, std::string> anchor_name;
  ss << "AscGraph(" << asc_graph.GetName() << ", axis="
     << loop::StrJoin(asc_graph.GetAllAxis(),
                      [](const AxisPtr &axis) { return std::to_string(axis->id) + ":" + axis->size.Str().get(); })
     << ")" << std::endl;
  for (const auto &node : asc_graph.GetAllNodes()) {
    std::vector<std::string> input_names;
    for (auto &anchor : node->GetAllInDataAnchors()) {
      auto peer = anchor->GetPeerOutAnchor();
      if (peer == nullptr) {
        continue;
      }
      input_names.emplace_back(anchor_name[peer]);
    }
    std::vector<std::string> output_names;
    std::map<std::string, std::string> output_loop;
    for (auto &anchor : node->GetAllOutDataAnchors()) {
      output_names.emplace_back("tmp" + std::to_string(anchor_name.size()));
      anchor_name[anchor] = output_names.back();
      auto loop = GetAscTensorLoop(anchor);
      if (!loop.empty()) {
        output_loop[output_names.back()] = loop;
      }
    }
    if (output_names.size() > 1U) {
      ss << loop::StrJoin(output_names) << " = ";
    } else if (!output_names.empty()) {
      ss << output_names[0] << " = ";
    }
    std::string name = node->GetName();
    if (trip_scope) {
      auto pos = name.find_last_of('/');
      if (pos != std::string::npos) {
        name = name.substr(pos + 1);
      }
    }
    ss << "ascir." << node->GetType() << "(" << name << ", " << loop::StrJoin(input_names) << ")" << std::endl;
    for (auto &loop : output_loop) {
      ss << loop.first << ".attr = {" << loop.second << "}" << std::endl;
    }
  }
  return ss.str();
}

REGISTER_LOWERING(SquaredDifferenceStub) {
  auto box0 = loop::Load(node->GetInDataAnchor(0));
  auto box1 = loop::Load(node->GetInDataAnchor(1));
  auto sub_box = loop::Sub(box0, box1);
  auto mul_box = loop::Mul(sub_box, sub_box);
  loop::Store(node->GetOutDataAnchor(0), mul_box);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(SoftmaxStub) {
  auto box = loop::Load(node->GetInDataAnchor(0));
  auto max = loop::ReduceThenBroadcast(loop::ReduceType::MAX, box, -1);
  auto sub = loop::Sub(box, max);
  auto exp = loop::Exp(sub);
  auto sum = loop::ReduceThenBroadcast(loop::ReduceType::SUM, exp, -1);
  auto div = loop::Div(exp, sum);
  loop::Store(node->GetOutDataAnchor(0), div);
  return GRAPH_SUCCESS;
}

REGISTER_LOWERING(LogAddExpStub) {
  auto box0 = loop::Load(node->GetInDataAnchor(0));
  auto box1 = loop::Load(node->GetInDataAnchor(1));
  auto log_box = loop::Log(box0);
  auto add_box = loop::Add(log_box, box1);
  auto exp_box = loop::Exp(add_box);
  loop::Store(node->GetOutDataAnchor(0), exp_box);
  return GRAPH_SUCCESS;
}

}  // namespace
class LoopLoweringToAscBackendUT : public testing::Test {
 public:
 protected:
  void SetUp() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {}
  std::unique_ptr<es::Graph> es_graph_;
};

class LoopLoweringToAscBackendUTV1 : public testing::Test {
public:
protected:
  void SetUp() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
    es_graph_ = std::unique_ptr<es::Graph>(new es::Graph("graph"));
    RegisterAllOpCreator();
  }
  void TearDown() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
    RuntimeStub::Reset();
    ge::PlatformContext::GetInstance().Reset();
  }
  std::unique_ptr<es::Graph> es_graph_;
};

TEST_F(LoopLoweringToAscBackendUT, SimpleAddAbsExpWithCse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data0);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(add);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto log_add_exp = cg->FindNode("Exp_2");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(log_add_exp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Add(Add_0, [tmp1, tmp1])
tmp3 = ascir.Abs(Abs_0, [tmp2])
tmp4 = ascir.Exp(Exp_0, [tmp3])
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleAddAbsExpNoCse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data0);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(add);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto log_add_exp = cg->FindNode("Exp_2");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(log_add_exp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph", false);
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Abs(Abs_0, [tmp4])
tmp6 = ascir.Exp(Exp_0, [tmp5])
tmp7 = ascir.Store(Store_0, [tmp6])
tmp7.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp8 = ascir.Output(Output_0, [tmp7])
)");
}

TEST_F(LoopLoweringToAscBackendUT, ComplexCseOnlyComputNode) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto sd1 = es::SquaredDifference(data0, data1);
    auto sd2 = es::SquaredDifference(data0, data1);
    sd1.SetSymbolShape({"s0", "s1", "s2"});
    sd2.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(sd1, sd2);
    add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindNode("SquaredDifference_0")->GetOpDesc()->SetType("SquaredDifferenceStub");
  cg->FindNode("SquaredDifference_1")->GetOpDesc()->SetType("SquaredDifferenceStub");
  auto add = cg->FindNode("Add_2");

  LoweringConfig config;
  config.max_loop_loads = 8U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Sub(Sub_0, [tmp1, tmp3])
tmp5 = ascir.Mul(Mul_0, [tmp4, tmp4])
tmp6 = ascir.Add(Add_0, [tmp5, tmp5])
tmp7 = ascir.Store(Store_0, [tmp6])
tmp7.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp8 = ascir.Output(Output_0, [tmp7])
)");
}

TEST_F(LoopLoweringToAscBackendUT, ComplexCseBothComputeAndLoad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto sd1 = es::SquaredDifference(data0, data0);
    auto sd2 = es::SquaredDifference(data0, data0);
    sd1.SetSymbolShape({"s0", "s1", "s2"});
    sd2.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(sd1, sd2);
    add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  cg->FindNode("SquaredDifference_0")->GetOpDesc()->SetType("SquaredDifferenceStub");
  cg->FindNode("SquaredDifference_1")->GetOpDesc()->SetType("SquaredDifferenceStub");
  auto add = cg->FindNode("Add_2");

  LoweringConfig config;
  config.max_loop_loads = 8U;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg, config), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Sub(Sub_0, [tmp1, tmp1])
tmp3 = ascir.Mul(Mul_0, [tmp2, tmp2])
tmp4 = ascir.Add(Add_0, [tmp3, tmp3])
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleCseLoad) {
  auto g = std::make_shared<loop::AscOverrides>("graph");
  auto loop_axis = loop::LoopAxis::FromDims({Symbol(2), Symbol(3)});
  g->SetLoopAxis(loop_axis);
  auto src0 = es_graph_->CreateInput(0, "data0", nullptr);
  auto src1 = es_graph_->CreateInput(1, "data1", nullptr);
  g->SetBufferSrc("buffer", src0.GetEsbTensor()->GetAnchor().get());
  g->SetBufferSrc("buffer1", src1.GetEsbTensor()->GetAnchor().get());

  std::vector<Expression> repeats;
  std::vector<Expression> strides;
  for (auto &repeat : loop_axis.repeats) {
    repeats.emplace_back(repeat);
    strides.emplace_back(Symbol(1));
  }
  loop::TensorLoopDesc desc(repeats, strides);
  auto data0 = g->Load("buffer", desc, Symbol(0));
  auto data1 = g->Load("buffer", desc, Symbol(0));
  EXPECT_EQ(data0, data1);

  auto data2 = g->Load("buffer1", desc, Symbol(0));
  EXPECT_NE(data0, data2);  // different buffer name

  auto data3 = g->Load("buffer", desc, Symbol(1));
  EXPECT_NE(data0, data3);  // different loop offset

  desc.repeats[0] = Symbol("s1");
  auto data4 = g->Load("buffer", desc, Symbol(1));
  EXPECT_NE(data4, data3);  // different loop repeats

  desc.strides[0] = Symbol(2);
  auto data5 = g->Load("buffer", desc, Symbol(0));
  EXPECT_NE(data5, data4);  // different loop strides
}

TEST_F(LoopLoweringToAscBackendUT, SimpleCseScalar) {
  auto g = std::make_shared<loop::AscOverrides>("graph");
  auto scalar0 = g->Scalar("2", ge::DT_FLOAT);
  auto scalar1 = g->Scalar("2", ge::DT_FLOAT);
  EXPECT_EQ(scalar0, scalar1);
  auto scalar2 = g->Scalar("3", ge::DT_FLOAT);
  EXPECT_NE(scalar0, scalar2);  // different value
  auto scalar3 = g->Scalar("2", ge::DT_INT32);
  EXPECT_NE(scalar0, scalar3);  // different data type
}

template <typename T>
bool VectorEq(const std::vector<T> &x, const std::vector<T> &y) {
  if (x.size() != y.size()) {
    return false;
  }
  for (size_t i = 0U; i < x.size(); i++) {
    if (x[i] != y[i]) {
      return false;
    }
  }
  return true;
}

TEST_F(LoopLoweringToAscBackendUT, TestTensorLoopCorrectWithNewAxisBroadcast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s1", "s0"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s2", "s1", "s0"});
    auto add1 = es::Add(data0, data1);
    add1.SetSymbolShape({"s1", "s0"});
    auto add2 = es::Add(add1, data2);
    add2.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(add2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_1");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s2, 1:s1, 2:s0])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, s0], strides = [0, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [1, s1, s0], strides = [0, s0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Data(Data_2, [])
tmp5.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp6 = ascir.Load(Load_2, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp7 = ascir.Add(Add_1, [tmp4, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestTensorLoopCorrectWithNormBroadcast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "1", "s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "s1", "s0"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s2", "s1", "1"});
    auto add1 = es::Add(data0, data1);
    add1.SetSymbolShape({"s2", "s1", "s0"});
    auto add2 = es::Add(add1, data2);
    add2.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(add2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_1");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s2, 1:s1, 2:s0])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, s0], strides = [0, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [1, s1, s0], strides = [0, s0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Data(Data_2, [])
tmp5.attr = {axis = [0, 1, 2], repeats = [s2, s1, 1], strides = [s1, 1, 0]}
tmp6 = ascir.Load(Load_2, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp7 = ascir.Add(Add_1, [tmp4, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStubSqueezeLower) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s1"});
    auto squeezed = es::Squeeze(data0, {1});
    squeezed.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(squeezed, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squeezed = cg->FindNode("Squeeze_0");
  ASSERT_NE(squeezed, nullptr);

  auto load = loop::Load(squeezed->GetInDataAnchor(0));
  auto asc_kernel = loop::Store(squeezed->GetOutDataAnchor(0), loop::Squeeze(load, {1}));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStubUnSqueezeLower) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto squeezed = es::Unsqueeze(data0, {1});
    squeezed.SetSymbolShape({"s0", "1", "s1"});
    es_graph_->SetOutput(squeezed, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto unsquee = cg->FindNode("Unsqueeze_0");
  ASSERT_NE(unsquee, nullptr);

  auto load = loop::Load(unsquee->GetInDataAnchor(0));
  auto asc_kernel = loop::Store(unsquee->GetOutDataAnchor(0), loop::Unsqueeze(load, {1}));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:1, 2:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, 1, s1], strides = [s1, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, 1, s1], strides = [s1, 0, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, 1, s1], strides = [s1, 0, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestTensorLoopCorrectWithNormAndNewAxisBroadcast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "s1", "s0"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s2", "1", "1"});
    auto add1 = es::Add(data0, data1);
    add1.SetSymbolShape({"1", "s1", "s0"});
    auto add2 = es::Add(add1, data2);
    add2.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(add2, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto add = cg->FindNode("Add_1");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(add->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s2, 1:s1, 2:s0])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, s0], strides = [0, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [1, s1, s0], strides = [0, s0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Data(Data_2, [])
tmp5.attr = {axis = [0, 1, 2], repeats = [s2, 1, 1], strides = [1, 0, 0]}
tmp6 = ascir.Load(Load_2, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp7 = ascir.Add(Add_1, [tmp4, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, OptimizeReduceToStoreAsKeepDimAndSrcDimIsOne) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "1", "s2"});
    auto sum = es::ReduceSumD(abs, {1}, true);
    sum.SetSymbolShape({"s0", "1", "s2"});
    es_graph_->SetOutput(sum, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto sum = cg->FindNode("ReduceSumD_1");
  ASSERT_NE(sum, nullptr);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kPointwise);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ReduceSumD_1_out0_graph, axis=[0:s0, 1:1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, OptimizeReduceToSqueezeAsKeepDimAndSrcDimIsOne) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "1", "s2"});
    auto sum = es::ReduceSumD(abs, {1}, false);
    sum.SetSymbolShape({"s0", "s2"});
    es_graph_->SetOutput(sum, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kPointwise);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ReduceSumD_1_out0_graph, axis=[0:s0, 1:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s2], strides = [s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s2], strides = [s2, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s2], strides = [s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringDim0Concat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"2", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"3", "s1", "s2"});
    auto concat = es::ConcatD({data0, data1}, 0);
    concat.SetSymbolShape({"5", "s1", "s2"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto concat = cg->FindNode("ConcatD_0");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kConcat);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ConcatD_0_out0_graph, axis=[0:5, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [2, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [2, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [3, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [3, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Concat(Concat_0, [tmp1, tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [5, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [5, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringDim1Concat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "3", "s2"});
    auto concat = es::ConcatD({data0, data1}, 0);
    concat.SetSymbolShape({"s0", "5", "s2"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto concat = cg->FindNode("ConcatD_0");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kConcat);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ConcatD_0_out0_graph, axis=[0:s0, 1:5, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, 2, s2], strides = [(2 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, 2, s2], strides = [(2 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, 3, s2], strides = [(3 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, 3, s2], strides = [(3 * s2), s2, 1]}
tmp4 = ascir.Concat(Concat_0, [tmp1, tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, 5, s2], strides = [(5 * s2), s2, 1]}
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, 5, s2], strides = [(5 * s2), s2, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringDimEndConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "3"});
    auto concat = es::ConcatD({data0, data1}, 0);
    concat.SetSymbolShape({"s0", "s1", "5"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto concat = cg->FindNode("ConcatD_0");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kConcat);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ConcatD_0_out0_graph, axis=[0:s0, 1:s1, 2:5])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp4 = ascir.Concat(Concat_0, [tmp1, tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, 5], strides = [(5 * s1), 5, 1]}
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, 5], strides = [(5 * s1), 5, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, RealizedBeforeConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "2"});
    data0 = es::Abs(data0);
    data0.SetSymbolShape({"s0", "s1", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "3"});
    data1 = es::Abs(data1);
    data1.SetSymbolShape({"s0", "s1", "3"});
    auto concat = es::ConcatD({data0, data1}, 0);
    concat.SetSymbolShape({"s0", "s1", "5"});
    es_graph_->SetOutput(concat, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto concat = cg->FindNode("ConcatD_2");
  ASSERT_NE(concat, nullptr);

  auto abs0 = cg->FindNode("Abs_0");
  ASSERT_NE(abs0, nullptr);
  auto abs1 = cg->FindNode("Abs_1");
  ASSERT_NE(abs1, nullptr);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto fused_abs0 = cg->FindNode("autofuse_pointwise_0_Abs");
  ASSERT_NE(fused_abs0, nullptr);
  auto fused_abs1 = cg->FindNode("autofuse_pointwise_1_Abs");
  ASSERT_NE(fused_abs1, nullptr);

  EXPECT_EQ(ReadableAscGraph(*fused_abs0->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>()->GetAscGraph()),
            R"(AscGraph(Abs_0_out0_graph, axis=[0:s0, 1:s1, 2:2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
  EXPECT_EQ(ReadableAscGraph(*fused_abs1->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>()->GetAscGraph()),
            R"(AscGraph(Abs_1_out0_graph, axis=[0:s0, 1:s1, 2:3])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");

  auto asc_node = cg->FindNode("autofuse_concat_2_ConcatD");
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kConcat);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ConcatD_2_out0_graph, axis=[0:s0, 1:s1, 2:5])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp4 = ascir.Concat(Concat_0, [tmp1, tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, 5], strides = [(5 * s1), 5, 1]}
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, 5], strides = [(5 * s1), 5, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, RealizedAfterConcat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "3"});
    auto concat = es::ConcatD({data0, data1}, 0);
    concat.SetSymbolShape({"s0", "s1", "5"});
    auto abs = es::Abs(concat);
    abs.SetSymbolShape({"s0", "s1", "5"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto concat = cg->FindNode("ConcatD_0");
  ASSERT_NE(concat, nullptr);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kConcat);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ConcatD_0_out0_graph, axis=[0:s0, 1:s1, 2:5])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, 2], strides = [(2 * s1), 2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, 3], strides = [(3 * s1), 3, 1]}
tmp4 = ascir.Concat(Concat_0, [tmp1, tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, 5], strides = [(5 * s1), 5, 1]}
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, 5], strides = [(5 * s1), 5, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, BiasAddLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0"});
    auto bias_add = es::BiasAdd(data0, data1, "NCHW");
    bias_add.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(bias_add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto bias_add = cg->FindFirstNodeMatchType("BiasAdd");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(bias_add->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, 1, 1], strides = [1, 0, 0]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleAddTileExpWithCse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s2"});
    auto add = es::Add(data0, data0);
    add.SetSymbolShape({"s0", "1", "s2"});
    auto multiples = es_graph_->CreateVector({1, 5, 1});
    multiples.SetSymbolShape({"3"});
    auto tile = es::Tile(add, multiples);
    tile.SetSymbolShape({"s0", "5", "s2"});
    auto exp = es::Exp(tile);
    exp.SetSymbolShape({"s0", "5", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto log_add_exp = cg->FindNode("Exp_3");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(log_add_exp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:5, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, 5, s2], strides = [(5 * s2), s2, 1]}
tmp2 = ascir.Add(Add_0, [tmp1, tmp1])
tmp3 = ascir.Exp(Exp_0, [tmp2])
tmp4 = ascir.Store(Store_0, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, 5, s2], strides = [(5 * s2), s2, 1]}
tmp5 = ascir.Output(Output_0, [tmp4])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleAddTileConcatWithCse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2", "s2"});
    auto add = es::Add(data0, data0);
    add.SetSymbolShape({"s0", "2", "s2"});
    auto multiples = es_graph_->CreateVector({1, 2, 1});
    multiples.SetSymbolShape({"3"});
    auto tile = es::Tile(add, multiples);
    tile.SetSymbolShape({"s0", "4", "s2"});
    es_graph_->SetOutput(tile, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto log_add_exp = cg->FindNode("Tile_2");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(log_add_exp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:4, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, 2, s2], strides = [(2 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, 2, s2], strides = [(2 * s2), s2, 1]}
tmp2 = ascir.Concat(Concat_0, [tmp1, tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, 4, s2], strides = [(4 * s2), s2, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, 4, s2], strides = [(4 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleTileExpScalarWithCse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s2"});
    auto multiples = es_graph_->CreateVector({1, 5, 1});
    multiples.SetSymbolShape({"3"});
    auto tile = es::Tile(data0, multiples);
    tile.SetSymbolShape({"1", "5", "s2"});
    auto exp = es::Exp(tile);
    exp.SetSymbolShape({"1", "5", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto log_add_exp = cg->FindNode("Exp_2");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(log_add_exp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:1, 1:5, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, s2], strides = [0, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [1, 5, s2], strides = [0, s2, 1]}
tmp2 = ascir.Exp(Exp_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [1, 5, s2], strides = [0, s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleAddTileDExpNoCse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "s2"});
    auto add = es::Add(data0, data0);
    add.SetSymbolShape({"s0", "1", "s2"});
    auto tile = es::TileD(add, {1, 5, 1});
    tile.SetSymbolShape({"s0", "5", "s2"});
    auto exp = es::Exp(tile);
    exp.SetSymbolShape({"s0", "5", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto log_add_exp = cg->FindNode("Exp_2");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(log_add_exp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph", false);
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:5, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, 5, s2], strides = [(5 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, 5, s2], strides = [(5 * s2), s2, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Exp(Exp_0, [tmp4])
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s0, 5, s2], strides = [(5 * s2), s2, 1]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleTileDExpScalarNoCse) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s2"});
    auto tile = es::TileD(data0, {1, 5, 1});
    tile.SetSymbolShape({"1", "5", "s2"});
    auto exp = es::Exp(tile);
    exp.SetSymbolShape({"1", "5", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto log_add_exp = cg->FindNode("Exp_1");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(log_add_exp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph", false);
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:1, 1:5, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, s2], strides = [0, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [1, 5, s2], strides = [0, s2, 1]}
tmp2 = ascir.Exp(Exp_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [1, 5, s2], strides = [0, s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, ZerosLikeLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0"});
    auto bias_add = es::BiasAdd(data0, data1, "NCHW");
    bias_add.SetSymbolShape({"s0", "s1", "s2"});
    auto zeroslike = es::ZerosLike(bias_add);
    zeroslike.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(zeroslike, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto zeros_like = cg->FindFirstNodeMatchType("ZerosLike");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(zeros_like->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Scalar(Scalar_0, [])
tmp1 = ascir.Store(Store_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Output(Output_0, [tmp1])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleAddN) {
  [this]() {
    std::vector<es::Tensor> datas = {};
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1"});
    datas.emplace_back(data0);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "s1", "s0"});
    datas.emplace_back(data1);
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s2", "1", "1"});
    datas.emplace_back(data2);
    auto addN = es::AddN(datas, datas.size());
    addN.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(addN, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto addN = cg->FindFirstNodeMatchType("AddN");
  addN->GetOpDesc()->SetType("AddN");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(addN->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph", false);
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s2, 1:s1, 2:s0])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, 1], strides = [0, 0, 0]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [1, s1, s0], strides = [0, s0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Data(Data_2, [])
tmp5.attr = {axis = [0, 1, 2], repeats = [s2, 1, 1], strides = [1, 0, 0]}
tmp6 = ascir.Load(Load_2, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp7 = ascir.Add(Add_1, [tmp4, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}
TEST_F(LoopLoweringToAscBackendUT, SimpleSquare) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto square1 = es::Square(data0);
    square1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(square1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto square = cg->FindNode("Square_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(square->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Mul(Mul_0, [tmp1, tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}
TEST_F(LoopLoweringToAscBackendUT, SimpleSquaredDifference) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto square_diff1 = es::SquaredDifference(data0, data1);
    square_diff1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(square_diff1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto square_diff = cg->FindNode("SquaredDifference_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(square_diff->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Sub(Sub_0, [tmp1, tmp3])
tmp5 = ascir.Mul(Mul_0, [tmp4, tmp4])
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SquaredDifferenceWithBroadcast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s2", "s1", "s0"});
    auto sd1 = es::SquaredDifference(data0, data1);
    sd1.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(sd1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("SquaredDifference_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s2, 1:s1, 2:s0])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, s1, s0], strides = [0, s0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp4 = ascir.Sub(Sub_0, [tmp1, tmp3])
tmp5 = ascir.Mul(Mul_0, [tmp4, tmp4])
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeBmmV2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"bs", "m", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"bs", "128", "1"});
    auto mm = es::BatchMatMulV2(data0, data1);
    mm.SetSymbolShape({"bs", "m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_matmul = true;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(asc_kernel.Readable(), R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.Permute(tmp1, "[d0, d1, d2]->[d0, d2, d1]")
tmp3 = ops.Broadcast(tmp2, "[d0, 1, d2]->[d0, d1, d2]")
tmp4 = ops.Mul(tmp0, tmp3)
tmp5 = ops.StoreReduction("BatchMatMulV2_0:0", ops.Sum(tmp4, "[d0, d1, d2]->[d0, d1, 1]"))
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:bs, 1:m, 2:128])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [bs, 1, 128], strides = [128, 0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Sum(Sum_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeBmmV2TransposeB) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"bs", "m", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"bs", "1", "128"});
    auto mm = es::BatchMatMulV2(data0, data1, nullptr, nullptr, false, true);
    mm.SetSymbolShape({"bs", "m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(asc_kernel.Readable(), R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.Broadcast(tmp1, "[d0, 1, d2]->[d0, d1, d2]")
tmp3 = ops.Mul(tmp0, tmp2)
tmp4 = ops.StoreReduction("BatchMatMulV2_0:0", ops.Sum(tmp3, "[d0, d1, d2]->[d0, d1, 1]"))
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:bs, 1:m, 2:128])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [bs, 1, 128], strides = [128, 0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Sum(Sum_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeBmmV2K1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"bs", "m", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"bs", "1", "1"});
    auto mm = es::BatchMatMulV2(data0, data1, nullptr, nullptr, false, true);
    mm.SetSymbolShape({"bs", "m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(!asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(asc_kernel.Readable(), R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.Broadcast(tmp1, "[d0, 1, d2]->[d0, d1, d2]")
tmp3 = ops.Mul(tmp0, tmp2)
tmp4 = ops.Store("BatchMatMulV2_0:0", tmp3)
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:bs, 1:m, 2:1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [bs, 1, 1], strides = [1, 0, 0]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeBmmV2AndFuseWithBefore) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"bs", "m", "128"});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"bs", "m", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"bs", "128", "1"});
    auto abs2 = es::Abs(data1);
    abs2.SetSymbolShape({"bs", "128", "1"});
    auto mm = es::BatchMatMulV2(abs1, abs2);
    mm.SetSymbolShape({"bs", "m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_2");
  ASSERT_NE(mm, nullptr);

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(asc_kernel.Readable(), R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Abs(tmp0)
tmp2 = ops.Store("Abs_0:0", tmp1)
tmp3 = ops.Load("data1:0")
tmp4 = ops.Abs(tmp3)
tmp5 = ops.Store("Abs_1:0", tmp4)
tmp6 = ops.Permute(tmp5, "[d0, d1, d2]->[d0, d2, d1]")
tmp7 = ops.Broadcast(tmp6, "[d0, 1, d2]->[d0, d1, d2]")
tmp8 = ops.Mul(tmp2, tmp7)
tmp9 = ops.StoreReduction("BatchMatMulV2_2:0", ops.Sum(tmp8, "[d0, d1, d2]->[d0, d1, 1]"))
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:bs, 1:m, 2:128])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Data(Data_1, [])
tmp3.attr = {axis = [0, 1, 2], repeats = [bs, 1, 128], strides = [128, 0, 1]}
tmp4 = ascir.Load(Load_1, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [bs, m, 128], strides = [(128 * m), 128, 1]}
tmp5 = ascir.Abs(Abs_1, [tmp4])
tmp6 = ascir.Mul(Mul_0, [tmp2, tmp5])
tmp7 = ascir.Sum(Sum_0, [tmp6])
tmp7.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [bs, m, 1], strides = [m, 1, 0]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeBmmV2AsSymbolicK) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "4", "1"});
    auto mm = es::BatchMatMulV2(data0, data1);
    mm.SetSymbolShape({"4", "4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeBmmV2AsOversizeK) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4", "1024"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "1024", "1"});
    auto mm = es::BatchMatMulV2(data0, data1);
    mm.SetSymbolShape({"4", "4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeBmmV2AsTransposeA) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "128", "1"});
    auto mm = es::BatchMatMulV2(data0, data1, nullptr, nullptr, true, false);
    mm.SetSymbolShape({"4", "4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeBmmV2AsWithBias) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "128", "1"});
    auto mm = es::BatchMatMulV2(data0, data1, data0);
    mm.SetSymbolShape({"4", "4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeBmmV2AsWithOffsetW) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "128", "1"});
    auto mm = es::BatchMatMulV2(data0, data1, nullptr, data0);
    mm.SetSymbolShape({"4", "4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeBmmV2AsWithOffset) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "128", "1"});
    auto mm = es::BatchMatMulV2(data0, data1, nullptr, nullptr, false, false, 2);
    mm.SetSymbolShape({"4", "4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("BatchMatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SimpleCast) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto cast = es::Cast(data0, ge::DataType::DT_FLOAT16);
    cast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(cast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cast = cg->FindNode("Cast_0");
  auto cast_desc = cast->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(cast_desc, nullptr);
  cast_desc->SetDataType(DT_FLOAT16);
  cast_desc->SetOriginDataType(DT_FLOAT16);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(cast->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  auto asc_cast = asc_graph->Graph().FindNode("graph/Cast_0");
  ASSERT_NE(asc_cast, nullptr);
  EXPECT_EQ(asc_cast->outputs[0].attr.dtype, DT_FLOAT16);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Cast(Cast_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleCastBool) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto cast = es::Cast(data0, ge::DataType::DT_BOOL);
    cast.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(cast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto cast = cg->FindNode("Cast_0");
  auto cast_desc = cast->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(cast_desc, nullptr);
  cast_desc->SetDataType(DT_BOOL);
  cast_desc->SetOriginDataType(DT_BOOL);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(cast->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SimpleSquareSumV1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<int64_t> axis = {0, -2};
    auto squaresumv1 = es::SquareSumV1(x, axis, true);
    squaresumv1.SetSymbolShape({"1", "1", "s2"});
    es_graph_->SetOutput(squaresumv1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squaresumv1 = cg->FindNode("SquareSumV1_0");
  ASSERT_NE(squaresumv1, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(squaresumv1->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Mul(Mul_0, [tmp1, tmp1])
tmp3 = ascir.Sum(Sum_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [1, 1, s2], strides = [0, 0, 1]}
tmp4 = ascir.Store(Store_0, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [1, 1, s2], strides = [0, 0, 1]}
tmp5 = ascir.Output(Output_0, [tmp4])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleSquareSumV1AllOne) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"1", "s0", "1", "s1", "s2"});
    std::vector<int64_t> axis = {0, 2};
    auto squaresumv1 = es::SquareSumV1(x, axis, true);
    squaresumv1.SetSymbolShape({"1", "1", "s2"});
    es_graph_->SetOutput(squaresumv1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto squaresumv1 = cg->FindNode("SquareSumV1_0");
  ASSERT_NE(squaresumv1, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(squaresumv1->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:1, 1:1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [1, 1, s2], strides = [0, 0, 1]}
tmp2 = ascir.Mul(Mul_0, [tmp1, tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [1, 1, s2], strides = [0, 0, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleBiasAddGrad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto add = es::BiasAddGrad(data0, "NHWC");
    add.SetSymbolShape({"s2"});
    es_graph_->SetOutput(add, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("BiasAddGrad_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Sum(Sum_0, [tmp1])
tmp2.attr = {axis = [0, 1], repeats = [1, s1], strides = [0, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [1, s1], strides = [0, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleBroadcastTo) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"1", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"3"});
    auto tensor = es::BroadcastTo(data0, data1);
    tensor.SetSymbolShape({"16", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("BroadcastTo_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:16, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, s1, s2], strides = [0, s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [16, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [16, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleLayerNormBetaGammaBackpropV2) {
  [this]() {
    auto dy = es_graph_->CreateInput(0, "dy", nullptr);
    dy.SetSymbolShape({"1024", "1024"});
    auto res_for_gamma = es_graph_->CreateInput(1, "res_for_gamma", nullptr);
    res_for_gamma.SetSymbolShape({"1024", "1024"});
    vector<int64_t> shape_gamma = {1024};
    auto tmp = es::LayerNormBetaGammaBackpropV2(dy, res_for_gamma, shape_gamma);
    tmp.pd_gamma.SetSymbolShape({"1024"});
    tmp.pd_beta.SetSymbolShape({"1024"});
    es_graph_->SetOutput(tmp.pd_gamma, 0);
    es_graph_->SetOutput(tmp.pd_beta, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("LayerNormBetaGammaBackpropV2_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:1024, 1:1024])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [1024, 1024], strides = [1024, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [1024, 1024], strides = [1024, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [1024, 1024], strides = [1024, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [1024, 1024], strides = [1024, 1]}
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Sum(Sum_0, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [1, 1024], strides = [0, 1]}
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1], repeats = [1, 1024], strides = [0, 1]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  auto asc_graph1 = asc_kernel1.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph1, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph1->Graph()), R"(AscGraph(graph, axis=[0:1024, 1:1024])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [1024, 1024], strides = [1024, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [1024, 1024], strides = [1024, 1]}
tmp2 = ascir.Sum(Sum_0, [tmp1])
tmp2.attr = {axis = [0, 1], repeats = [1, 1024], strides = [0, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [1, 1024], strides = [0, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReluGrad) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"gradients:0\")\n"
"tmp1 = ops.Load(\"features:0\")\n"
"tmp2 = ops.Scalar(\"DT_FLOAT(0.00000000000000000000000000000000000001175494350822)\")\n"
"tmp3 = ops.Scalar(\"DT_FLOAT(274877906944.000)\")\n"
"tmp4 = ops.Scalar(\"DT_FLOAT(17592186044416.000)\")\n"
"tmp5 = ops.Scalar(\"DT_FLOAT(17592186044416.000)\")\n"
"tmp6 = ops.Broadcast(tmp2, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
"tmp8 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp9 = ops.Broadcast(tmp5, \"[]->[d0, d1, d2]\")\n"
"tmp10 = ops.Scalar(\"DT_FLOAT(0)\")\n"
"tmp11 = ops.Broadcast(tmp10, \"[]->[d0, d1, d2]\")\n"
"tmp12 = ops.Minimum(tmp1, tmp6)\n"
"tmp13 = ops.Maximum(tmp12, tmp11)\n"
"tmp14 = ops.Mul(tmp13, tmp7)\n"
"tmp15 = ops.Mul(tmp14, tmp9)\n"
"tmp16 = ops.Mul(tmp15, tmp9)\n"
"tmp17 = ops.Mul(tmp0, tmp16)\n"
"tmp18 = ops.Store(\"ReluGrad_0:0\", tmp17)\n");
}

//TEST_F(LoopLoweringToAscBackendUT, SimpleReluGradInt64) {
//  [this]() {
//    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
//    gradients.SetSymbolShape({"s0", "s1", "s2"});
//    auto features = es_graph_->CreateInput(1, "features", nullptr);
//    features.SetSymbolShape({"s0", "s1", "s2"});
//    auto tensor = es::ReluGrad(gradients, features);
//    tensor.SetSymbolShape({"s0", "s1", "s2"});
//    es_graph_->SetOutput(tensor, 0);
//  }();
//
//  auto graph = es_graph_->Build();
//  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
//  auto node = cg->FindNode("ReluGrad_0");
//  ASSERT_NE(node, nullptr);
//
//  auto nodeptr1 = cg->FindNode("gradients");
//  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
//  tmp_desc->SetDataType(DT_INT64);
//  tmp_desc->SetOriginDataType(DT_INT64);
//  auto nodeptr2 = cg->FindNode("features");
//  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
//  tmp_desc2->SetDataType(DT_INT64);
//  tmp_desc2->SetOriginDataType(DT_INT64);
//
//  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
//  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
//  ASSERT_FALSE(kernel.IsExternKernel());
//  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"gradients:0\")\n"
//"tmp1 = ops.Load(\"features:0\")\n"
//"tmp2 = ops.Scalar(\"DT_INT64(0)\")\n"
//"tmp3 = ops.Scalar(\"DT_INT64(0)\")\n"
//"tmp4 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
//"tmp5 = ops.Broadcast(tmp3, \"[]->[d0, d1, d2]\")\n"
//"tmp6 = ops.Le(tmp1, tmp5)\n"
//"tmp7 = ops.Where(tmp6, tmp5, tmp0)\n"
//"tmp8 = ops.Store(\"ReluGrad_0:0\", tmp7)\n");
//}

TEST_F(LoopLoweringToAscBackendUT, SimpleReluGradInt32) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  auto nodeptr1 = cg->FindNode("gradients");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT32);
  tmp_desc->SetOriginDataType(DT_INT32);
  auto nodeptr2 = cg->FindNode("features");
  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_INT32);
  tmp_desc2->SetOriginDataType(DT_INT32);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_FALSE(kernel.IsExternKernel());
  EXPECT_EQ(kernel.Readable(), "tmp0 = ops.Load(\"gradients:0\")\n"
"tmp1 = ops.Load(\"features:0\")\n"
"tmp2 = ops.Scalar(\"DT_INT32(1)\")\n"
"tmp3 = ops.Scalar(\"DT_INT32(1)\")\n"
"tmp4 = ops.Scalar(\"DT_INT32(1)\")\n"
"tmp5 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
"tmp6 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
"tmp7 = ops.Broadcast(tmp4, \"[]->[d0, d1, d2]\")\n"
"tmp8 = ops.Scalar(\"DT_INT32(0)\")\n"
"tmp9 = ops.Broadcast(tmp8, \"[]->[d0, d1, d2]\")\n"
"tmp10 = ops.Minimum(tmp1, tmp5)\n"
"tmp11 = ops.Maximum(tmp10, tmp9)\n"
"tmp12 = ops.Mul(tmp11, tmp6)\n"
"tmp13 = ops.Mul(tmp12, tmp7)\n"
"tmp14 = ops.Mul(tmp0, tmp13)\n"
"tmp15 = ops.Store(\"ReluGrad_0:0\", tmp14)\n");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReluGradUINT8) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  auto nodeptr1 = cg->FindNode("gradients");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_UINT8);
  tmp_desc->SetOriginDataType(DT_UINT8);
  auto nodeptr2 = cg->FindNode("features");
  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_UINT8);
  tmp_desc2->SetOriginDataType(DT_UINT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReluGradBF16) {
  [this]() {
    auto gradients = es_graph_->CreateInput(0, "gradients", nullptr);
    gradients.SetSymbolShape({"s0", "s1", "s2"});
    auto features = es_graph_->CreateInput(1, "features", nullptr);
    features.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::ReluGrad(gradients, features);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("ReluGrad_0");
  ASSERT_NE(node, nullptr);

  auto nodeptr1 = cg->FindNode("gradients");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_BF16);
  tmp_desc->SetOriginDataType(DT_BF16);
  auto nodeptr2 = cg->FindNode("features");
  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_BF16);
  tmp_desc2->SetOriginDataType(DT_BF16);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  ASSERT_TRUE(kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SimpleRsqrtGrad) {
  [this]() {
    auto y = es_graph_->CreateInput(0, "y", nullptr);
    y.SetSymbolShape({"s0", "s1", "s2"});
    auto dy = es_graph_->CreateInput(1, "dy", nullptr);
    dy.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::RsqrtGrad(y, dy);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("RsqrtGrad_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Scalar(Scalar_0, [])
tmp5 = ascir.Mul(Mul_0, [tmp1, tmp1])
tmp6 = ascir.Mul(Mul_1, [tmp1, tmp5])
tmp7 = ascir.Mul(Mul_2, [tmp3, tmp6])
tmp8 = ascir.Mul(Mul_3, [tmp4, tmp7])
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleSigmoidGrad) {
  [this]() {
    auto y = es_graph_->CreateInput(0, "y", nullptr);
    y.SetSymbolShape({"s0", "s1", "s2"});
    auto dy = es_graph_->CreateInput(1, "dy", nullptr);
    dy.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::SigmoidGrad(y, dy);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("SigmoidGrad_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Sub(Sub_0, [tmp2, tmp1])
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Data(Data_1, [])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Load(Load_1, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp7 = ascir.Mul(Mul_1, [tmp4, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleAddV2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto tmp = es::AddV2(data0, data1);
    tmp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tmp = cg->FindNode("AddV2_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");

  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Add(Add_0, [tmp1, tmp3])
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleMuls) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::Muls(x, 1.5);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Muls_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Mul(Mul_0, [tmp1, tmp2])
tmp4 = ascir.Store(Store_0, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Output(Output_0, [tmp4])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleAxpy) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto tensor = es::Axpy(x, x, 0.8);
    tensor.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tensor, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Axpy_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Axpy(Axpy_0, [tmp1, tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringMatmul_extern) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"4", "5"});
    data1.SetSymbolShape({"5", "6"});
    auto mm = es::MatMul(data0, data1);
    mm.SetSymbolShape({"4", "6"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mm = cg->FindNode("MatMul_0");
  ASSERT_NE(mm, nullptr);
  auto tmp_desc = mm->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);

  auto data0 = cg->FindNode("data0");
  auto tmp_desc0 = data0->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc0->SetDataType(DT_FLOAT16);
  tmp_desc0->SetOriginDataType(DT_FLOAT16);
  auto data1 = cg->FindNode("data1");
  auto tmp_desc1 = data1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_FLOAT16);
  tmp_desc1->SetOriginDataType(DT_FLOAT16);

  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_matmul = false;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  AutoFuseConfig::MutableLoweringConfig().experimental_lowering_matmul = true;
}

TEST_F(LoopLoweringToAscBackendUT, SetOpOutputDtypeByInferDtype) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto abs = cg->FindNode("Abs_0");
  auto data = cg->FindNode("data0");
  ASSERT_NE(data, nullptr);
  auto input0_desc = data->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_FLOAT16);
  input0_desc->SetOriginDataType(DT_FLOAT16);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(abs->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  auto asc_abs = asc_graph->Graph().FindNode("graph/Abs_0");
  ASSERT_NE(asc_abs, nullptr);
  EXPECT_EQ(asc_abs->outputs[0].attr.dtype, DT_FLOAT16);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Cast(Cast_0, [tmp2])
tmp4 = ascir.Store(Store_0, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Output(Output_0, [tmp4])
)");
}

EsbTensor *EsPartitionedCall(EsbTensor *x1) {
  GE_ASSERT_NOTNULL(x1);
  auto &graph = x1->GetOwner();

  auto desc = ge::CompliantOpDescBuilder()
                  .OpType(PARTITIONEDCALL)
                  .Name(("PartitionedCall_" + std::to_string(graph.NextNodeIndex())).c_str())
                  .IrDefInputs({
                      {"x1", ge::kIrInputRequired, ""},
                  })
                  .IrDefOutputs({
                      {"y", ge::kIrOutputRequired, ""},
                  })
                  .IrDefAttrs({})
                  .Build();
  GE_ASSERT_NOTNULL(desc);
  auto node = graph.GetComputeGraph()->AddNode(desc);
  GE_ASSERT_NOTNULL(node);

  GE_ASSERT_GRAPH_SUCCESS(ge::GraphUtils::AddEdge(x1->GetAnchor(), node->GetInDataAnchor(0)));
  return graph.GetEsbTensorFromNode(std::move(node), 0);
}

es::Tensor PartitionedCall(es::Tensor &t) {
  auto out = EsPartitionedCall(t.GetEsbTensor());
  return out;
}

TEST_F(LoopLoweringToAscBackendUT, SimpleSubgraph) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data0);
    add.SetSymbolShape({"s0", "s1", "s2"});
    // auto abs = es::Abs(add);
    auto abs = PartitionedCall(add);
    abs.SetSymbolShape({"s0", "s1", "s2"});
    auto exp = es::Exp(abs);
    exp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(exp, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto subgraph_name = "subgraph1";
  auto sub_graph_ = std::unique_ptr<es::Graph>(new es::Graph(subgraph_name));
  auto sub_data_s0 = sub_graph_->CreateInput(0, "data0", nullptr);
  sub_data_s0.SetSymbolShape({"s0", "s1", "s2"});
  auto sub_add = es::Add(sub_data_s0, sub_data_s0);
  sub_add.SetSymbolShape({"s0", "s1", "s2"});
  auto sub_abs = es::Abs(sub_add);
  sub_abs.SetSymbolShape({"s0", "s1", "s2"});
  sub_graph_->SetOutput(sub_abs, 0);
  auto sub_graph_cg = GraphUtilsEx::GetComputeGraph(*sub_graph_->Build());

  // add subgraph
  auto parent_node = cg->FindNode("PartitionedCall_1");
  ASSERT_NE(parent_node, nullptr);
  ASSERT_EQ(parent_node->GetOpDesc()->AddSubgraphName(subgraph_name), GRAPH_SUCCESS);
  ASSERT_EQ(parent_node->GetOpDesc()->SetSubgraphInstanceName(0, subgraph_name), GRAPH_SUCCESS);
  sub_graph_cg->SetParentNode(parent_node);
  sub_graph_cg->SetParentGraph(cg);
  ASSERT_EQ(cg->AddSubgraph(sub_graph_cg), GRAPH_SUCCESS);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto exp = cg->FindNode("Exp_2");
  ASSERT_NE(exp, nullptr);
  auto fused_exp = cg->FindNode("autofuse_pointwise_1_Exp");
  ASSERT_NE(fused_exp, nullptr);
  EXPECT_EQ(ReadableAscGraph(*fused_exp->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>()->GetAscGraph()),
            R"(AscGraph(Exp_2_out0_graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Exp(Exp_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");

  auto sg_abs = sub_graph_cg->FindNode("Abs_1");
  ASSERT_NE(sg_abs, nullptr);
  auto fused_sg_abs = sub_graph_cg->FindNode("autofuse_pointwise_2_Add_Abs");
  ASSERT_NE(fused_sg_abs, nullptr);
  EXPECT_EQ(ReadableAscGraph(*fused_sg_abs->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>()->GetAscGraph()),
            R"(AscGraph(Abs_1_out0_graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Add(Add_0, [tmp1, tmp1])
tmp3 = ascir.Abs(Abs_0, [tmp2])
tmp4 = ascir.Store(Store_0, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Output(Output_0, [tmp4])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringSoftmaxLikeExample) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1"});
    auto softmax = es::SoftmaxV2(data, {-1});
    softmax.SetSymbolShape({"s0", "s1"});
    auto abs = es::Abs(softmax);
    abs.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(abs, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto softmax = cg->FindNode("SoftmaxV2_0");
  softmax->GetOpDesc()->SetType("SoftmaxStub");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(softmax->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  auto fused_softmax = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(fused_softmax, nullptr);
  auto fuse_attr = fused_softmax->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fuse_attr, nullptr);
  EXPECT_EQ(fuse_attr->GetFuseType(), loop::FuseType::kReduction);
  EXPECT_EQ(ReadableAscGraph(*fuse_attr->GetAscGraph()),
          R"(AscGraph(SoftmaxV2_0_out0_graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Max(Max_0, [tmp1])
tmp2.attr = {axis = [0, 1], repeats = [s0, 1], strides = [1, 0]}
tmp3 = ascir.Broadcast(Broadcast_0, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp4 = ascir.Sub(Sub_0, [tmp1, tmp3])
tmp5 = ascir.Exp(Exp_0, [tmp4])
tmp6 = ascir.Sum(Sum_0, [tmp5])
tmp6.attr = {axis = [0, 1], repeats = [s0, 1], strides = [1, 0]}
tmp7 = ascir.Broadcast(Broadcast_1, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp8 = ascir.Div(Div_0, [tmp5, tmp7])
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, AutoConvertInputBoolToUint8) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1", "s2"});
    auto cast = es::Cast(data0, DT_FLOAT16);
    cast.SetSymbolShape({"s1", "s2"});
    es_graph_->SetOutput(cast, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto data0 = cg->FindNode("data0");
  ASSERT_NE(data0, nullptr);
  auto input0_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  input0_desc->SetDataType(DT_BOOL);
  input0_desc->SetOriginDataType(DT_BOOL);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto cast = cg->FindNode("Cast_0");
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(cast->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  auto asc_node = asc_graph->Graph().FindNode("graph/Cast_0");
  ASSERT_NE(asc_node, nullptr);
  EXPECT_EQ(asc_node->outputs[0].attr.dtype, DT_FLOAT16);
}

TEST_F(LoopLoweringToAscBackendUT, SimpleUnsqueeze) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "2"});
    auto unsqueeze = es::Unsqueeze(data0, {0, 1});
    unsqueeze.SetSymbolShape({"1", "1", "s0", "2"});
    es_graph_->SetOutput(unsqueeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Unsqueeze_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:1, 1:1, 2:s0, 3:2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, s0, 2], strides = [0, 0, 2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, s0, 2], strides = [0, 0, 2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, s0, 2], strides = [0, 0, 2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleSqueeze) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "1", "2", "1", "1"});
    auto squeeze = es::Squeeze(data0, {1, 3, -1});
    squeeze.SetSymbolShape({"s0", "2"});
    es_graph_->SetOutput(squeeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Squeeze_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, 2], strides = [2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, 2], strides = [2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1], repeats = [s0, 2], strides = [2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
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

TEST_F(LoopLoweringToAscBackendUT, SimpleFill) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto value = CreateConst(*es_graph_, DT_FLOAT, {}, std::vector<float>{1});
    value.SetSymbolShape({});
    auto tmp = es::Fill(data0, value);
    tmp.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp, 0);

    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto value1 = CreateConst(*es_graph_, ge::DT_INT8, {}, std::vector<int8_t>{1});
    value1.SetSymbolShape({});
    auto tmp1 = es::Fill(data1, value1);
    tmp1.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp1, 1);

    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1", "s2"});
    auto value2 = CreateConst(*es_graph_, ge::DT_INT32, {}, std::vector<int32_t>{1});
    value2.SetSymbolShape({});
    auto tmp2 = es::Fill(data2, value2);
    tmp2.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp2, 2);

    auto data3 = es_graph_->CreateInput(3, "data3", nullptr);
    data3.SetSymbolShape({"s0", "s1", "s2"});
    auto value3 = CreateConst(*es_graph_, ge::DT_UINT8, {}, std::vector<uint8_t>{1});
    value3.SetSymbolShape({});
    auto tmp3 = es::Fill(data3, value3);
    tmp3.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp3, 3);

    auto data4 = es_graph_->CreateInput(4, "data4", nullptr);
    data4.SetSymbolShape({"s0", "s1", "s2"});
    auto value4 = CreateConst(*es_graph_, ge::DT_INT16, {}, std::vector<int16_t>{1});
    value4.SetSymbolShape({});
    auto tmp4 = es::Fill(data4, value4);
    tmp4.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp4, 4);

    auto data5 = es_graph_->CreateInput(5, "data5", nullptr);
    data5.SetSymbolShape({"s0", "s1", "s2"});
    auto value5 = CreateConst(*es_graph_, ge::DT_UINT16, {}, std::vector<uint16_t>{1});
    value5.SetSymbolShape({});
    auto tmp5 = es::Fill(data5, value5);
    tmp5.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp5, 5);

    auto data6 = es_graph_->CreateInput(6, "data6", nullptr);
    data6.SetSymbolShape({"s0", "s1", "s2"});
    auto value6 = CreateConst(*es_graph_, ge::DT_UINT32, {}, std::vector<uint32_t>{1});
    value6.SetSymbolShape({});
    auto tmp6 = es::Fill(data6, value6);
    tmp6.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp6, 6);

    auto data7 = es_graph_->CreateInput(7, "data7", nullptr);
    data7.SetSymbolShape({"s0", "s1", "s2"});
    auto value7 = CreateConst(*es_graph_, ge::DT_FLOAT16, {}, std::vector<int16_t>{1});
    value7.SetSymbolShape({});
    auto tmp7 = es::Fill(data7, value7);
    tmp7.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp7, 7);

    auto data8 = es_graph_->CreateInput(8, "data8", nullptr);
    data8.SetSymbolShape({"s0", "s1", "s2"});
    auto value8 = CreateConst(*es_graph_, DT_FLOAT, {}, std::vector<float>{1.0f/0.0f});
    value8.SetSymbolShape({});
    auto tmp8 = es::Fill(data8, value8);
    tmp8.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(tmp8, 8);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto tmp = cg->FindNode("Fill_1");
  auto tmp1 = cg->FindNode("Fill_3");
  auto tmp_desc1 = tmp1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(ge::DT_INT8);
  tmp_desc1->SetOriginDataType(ge::DT_INT8);
  auto tmp2 = cg->FindNode("Fill_5");
  auto tmp_desc2 = tmp2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(ge::DT_INT32);
  tmp_desc2->SetOriginDataType(ge::DT_INT32);
  auto tmp3 = cg->FindNode("Fill_7");
  auto tmp_desc3 = tmp3->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(ge::DT_UINT8);
  tmp_desc3->SetOriginDataType(ge::DT_UINT8);
  auto tmp4 = cg->FindNode("Fill_9");
  auto tmp_desc4 = tmp4->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc4->SetDataType(ge::DT_INT16);
  tmp_desc4->SetOriginDataType(ge::DT_INT16);
  auto tmp5 = cg->FindNode("Fill_11");
  auto tmp_desc5 = tmp5->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc5->SetDataType(ge::DT_UINT16);
  tmp_desc5->SetOriginDataType(ge::DT_UINT16);
  auto tmp6 = cg->FindNode("Fill_13");
  auto tmp_desc6 = tmp6->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc6->SetDataType(ge::DT_UINT32);
  tmp_desc6->SetOriginDataType(ge::DT_UINT32);
  auto tmp7 = cg->FindNode("Fill_15");
  auto tmp_desc7 = tmp7->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc7->SetDataType(ge::DT_FLOAT16);
  tmp_desc7->SetOriginDataType(ge::DT_FLOAT16);
  auto tmp8 = cg->FindNode("Fill_17");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  vector<NodePtr> nodes = {tmp, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6};
  for (auto node : nodes) {
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
    ASSERT_NE(asc_graph, nullptr);
    EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Scalar(Scalar_0, [])
tmp1 = ascir.Store(Store_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Output(Output_0, [tmp1])
)");
  }

    loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(tmp7->GetOutDataAnchor(0));
    auto asc_graph1 = asc_kernel1.Realize<loop::AscOverrides>("graph");
    ASSERT_NE(asc_graph1, nullptr);
    EXPECT_EQ(ReadableAscGraph(asc_graph1->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [1, 1, 1], strides = [0, 0, 0]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Cast(Cast_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");

  loop::KernelBox asc_kernel2 = ge::loop::GetKernelBox(tmp8->GetOutDataAnchor(0));
  auto asc_graph2 = asc_kernel2.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph2, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, TestSliceLoweringStaticSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    auto offset = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{0, 0, 1});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{-1, -1, 32});
    auto slice = es::Slice(data0, offset, size);
    slice.SetSymbolShape({"192", "64", "32"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("Slice_2");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:192, 1:64, 2:32])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [192, 64, 32], strides = [2112, 33, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [192, 64, 32], strides = [2048, 32, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [192, 64, 32], strides = [2048, 32, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestSliceLoweringDynamicSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto offset = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{20, 10, 1});
    auto size = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{50, 50, 32});
    auto slice = es::Slice(data0, offset, size);
    slice.SetSymbolShape({"o0", "o1", "o2"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("Slice_2");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:o0, 1:o1, 2:o2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestSliceDLoweringDynamicSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<int64_t> offsets = {20, 10, 1};
    std::vector<int64_t> sizes = {50, 0, 32};
    auto slice = es::SliceD(data0, offsets, sizes);
    slice.SetSymbolShape({"o0", "o1", "o2"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto slice = cg->FindNode("SliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:o0, 1:o1, 2:o2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestSliceDLoweringStaticSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> offsets = {20, 10, 1};
    std::vector<int64_t> sizes = {50, 50, 32};
    auto slice = es::SliceD(data0, offsets, sizes);
    slice.SetSymbolShape({"50", "50", "32"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto slice = cg->FindNode("SliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:50, 1:50, 2:32])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [50, 50, 32], strides = [2112, 33, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [50, 50, 32], strides = [1600, 32, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [50, 50, 32], strides = [1600, 32, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceLoweringStaticSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    auto begin = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{0, 0, 1});
    auto end = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{60, 60, 10});
    auto stride = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{2, 1, 1});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"192", "64", "33"});
    auto slice = es::StridedSlice(abs, begin, end, stride);
    slice.SetSymbolShape({"30", "60", "9"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSlice_4");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  auto load_node = asc_graph->Graph().FindNode("graph/Load_0");
  ASSERT_NE(load_node, nullptr);
  ASSERT_NE(load_node->GetOpDesc(), nullptr);
  const auto &attr = load_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  ASSERT_NE(attr, nullptr);
  auto load_attr = dynamic_cast<ascir_op::Load::AscLoadIrAttrDef *>(attr->ir_attr.get());
  ASSERT_NE(load_attr, nullptr);
  Expression load_offset;
  ASSERT_EQ(load_attr->GetOffset(load_offset), 0);
  EXPECT_EQ(load_offset, Symbol(1));
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:30, 1:60, 2:9])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [4224, 33, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceLoweringDynamicSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto begin = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{0, 0, 1});
    auto end = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{60, 60, 10});
    auto stride = CreateConst(*es_graph_, ge::DT_INT32, {3}, std::vector<int32_t>{1, 2, 1});
    auto slice = es::StridedSlice(data0, begin, end, stride);
    slice.SetSymbolShape({"o0", "o1", "o2"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSlice_3");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:o0, 1:o1, 2:o2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(s1 * s2), (2 * s2), 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceDLoweringStaticSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 10};
    std::vector<int64_t> stride = {2, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"30", "60", "9"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:30, 1:60, 2:9])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [4224, 33, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceDEndStrideIsNotOne) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 11};
    std::vector<int64_t> stride = {2, 1, 2};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"30", "60", "5"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceDLoweringDynamicSuccess) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 10};
    std::vector<int64_t> stride = {2, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"o0", "o1", "o2"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:o0, 1:o1, 2:o2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(2 * s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceDLoweringStaticSuccessWithMask) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> begin = {0, 1};
    std::vector<int64_t> end = {60, 10};
    std::vector<int64_t> stride = {1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride, 1, 0, 2, 1, 0);
    slice.SetSymbolShape({"30", "60", "9"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:30, 1:60, 2:9])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [2112, 33, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestSplitDLoweringSplitInputDimNotEqOutputDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "30"});
  }();
  auto desc = ge::CompliantOpDescBuilder().OpType("SplitD")
  .Name("SplitD")
  .IrDefInputs({
  {"x", ge::kIrInputRequired, ""},
  })
  .IrDefOutputs({
  {"y", ge::kIrOutputDynamic, ""},
  })
  .InstanceDynamicOutputNum("y", 3)
  .IrDefAttrs({
    {
    "split_dim",
    ge::kAttrRequired,
    "VT_INT",
    ge::AnyValue::CreateFrom(static_cast<int64_t>(-1))
    },
  {
  "num_split",
  ge::kAttrRequired,
  "VT_INT",
  ge::AnyValue::CreateFrom(static_cast<int64_t>(3))
  },
  })
  .Build();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->AddNode(desc);
  auto x = cg->FindNode("data0");
  ASSERT_NE(x, nullptr);
  ASSERT_EQ(ge::GraphUtils::AddEdge(x->GetOutDataAnchor(0), node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  auto esb_graph = es_graph_->GetEsbGraph();
  auto split_out0 = esb_graph->GetEsbTensorFromNode(node, 0);
  auto split_out1 = esb_graph->GetEsbTensorFromNode(node, 1);
  auto split_out2 = esb_graph->GetEsbTensorFromNode(node, 2);
  split_out0->SetSymbolShape({Symbol(192), Symbol(64), Symbol(10), Symbol(10)});
  split_out1->SetSymbolShape({Symbol(192), Symbol(64), Symbol(10), Symbol(10)});
  split_out2->SetSymbolShape({Symbol(192), Symbol(64), Symbol(10), Symbol(10)});
  es_graph_->SetOutput(split_out0, 0);
  es_graph_->SetOutput(split_out1, 1);
  es_graph_->SetOutput(split_out2, 2);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto split = cg->FindNode("SplitD");
  ASSERT_NE(split, nullptr);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  for (const auto &node_out : split->GetAllOutDataAnchors()) {
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(node_out);
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
    ASSERT_EQ(asc_graph, nullptr);
  }
}

TEST_F(LoopLoweringToAscBackendUTV1, TestSplitDLoweringSplitOutputEndDimHasOne) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "3"});
  }();
  auto desc = ge::CompliantOpDescBuilder().OpType("SplitD")
  .Name("SplitD")
  .IrDefInputs({
  {"x", ge::kIrInputRequired, ""},
  })
  .IrDefOutputs({
  {"y", ge::kIrOutputDynamic, ""},
  })
  .InstanceDynamicOutputNum("y", 3)
  .IrDefAttrs({
    {
    "split_dim",
    ge::kAttrRequired,
    "VT_INT",
    ge::AnyValue::CreateFrom(static_cast<int64_t>(2))
    },
  {
  "num_split",
  ge::kAttrRequired,
  "VT_INT",
  ge::AnyValue::CreateFrom(static_cast<int64_t>(3))
  },
  })
  .Build();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->AddNode(desc);
  auto x = cg->FindNode("data0");
  ASSERT_NE(x, nullptr);
  ASSERT_EQ(ge::GraphUtils::AddEdge(x->GetOutDataAnchor(0), node->GetInDataAnchor(0)), GRAPH_SUCCESS);
  auto esb_graph = es_graph_->GetEsbGraph();
  auto split_out0 = esb_graph->GetEsbTensorFromNode(node, 0);
  auto split_out1 = esb_graph->GetEsbTensorFromNode(node, 1);
  auto split_out2 = esb_graph->GetEsbTensorFromNode(node, 2);
  split_out0->SetSymbolShape({Symbol(192), Symbol(64), Symbol(1)});
  split_out1->SetSymbolShape({Symbol(192), Symbol(64), Symbol(1)});
  split_out2->SetSymbolShape({Symbol(192), Symbol(64), Symbol(1)});
  es_graph_->SetOutput(split_out0, 0);
  es_graph_->SetOutput(split_out1, 1);
  es_graph_->SetOutput(split_out2, 2);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto split = cg->FindNode("SplitD");
  ASSERT_NE(split, nullptr);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  for (const auto &node_out : split->GetAllOutDataAnchors()) {
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(node_out);
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
    ASSERT_EQ(asc_graph, nullptr);
  }
}

TEST_F(LoopLoweringToAscBackendUT, SimpleLog) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto log = es::Log(x, 2, 2, 1);

    log.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(log, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Log_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Mul(Mul_0, [tmp1, tmp2])
tmp4 = ascir.Scalar(Scalar_1, [])
tmp5 = ascir.Add(Add_0, [tmp3, tmp4])
tmp6 = ascir.Ln(Ln_0, [tmp5])
tmp7 = ascir.Scalar(Scalar_2, [])
tmp8 = ascir.Mul(Mul_1, [tmp6, tmp7])
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleLogDefault) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto log = es::Log(x);

    log.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(log, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Log_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Ln(Ln_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"5"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "1", "s1", "s2", "1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:1, 2:s1, 3:s2, 4:1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2, 3, 4], repeats = [s0, 1, s1, s2, 1], strides = [(s1 * s2), 0, s2, 1, 0]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2, 3, 4], repeats = [s0, 1, s1, s2, 1], strides = [(s1 * s2), 0, s2, 1, 0]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2, 3, 4], repeats = [s0, 1, s1, s2, 1], strides = [(s1 * s2), 0, s2, 1, 0]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape2) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "1", "s1", "1", "1"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"2"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape3) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(3), Symbol(2), Symbol(3)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"4"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(6), Symbol(3)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);

//   EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:2, 1:6, 2:3])
// tmp0 = ascir.Data(Data_0, [])
// tmp0.attr = {axis = [0, 1, 2], repeats = [2, 6, 3], strides = [18, 3, 1]}
// tmp1 = ascir.Load(Load_0, [tmp0])
// tmp1.attr = {axis = [0, 1, 2], repeats = [2, 6, 3], strides = [18, 3, 1]}
// tmp2 = ascir.Store(Store_0, [tmp1])
// tmp2.attr = {axis = [0, 1, 2], repeats = [2, 6, 3], strides = [18, 3, 1]}
// tmp3 = ascir.Output(Output_0, [tmp2])
// )");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape4) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(6), Symbol(3)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"4"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(2), Symbol(3), Symbol(2), Symbol(3)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);

//   EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:2, 1:3, 2:2, 3:3])
// tmp0 = ascir.Data(Data_0, [])
// tmp0.attr = {axis = [0, 1, 2, 3], repeats = [2, 3, 2, 3], strides = [18, 6, 3, 1]}
// tmp1 = ascir.Load(Load_0, [tmp0])
// tmp1.attr = {axis = [0, 1, 2, 3], repeats = [2, 3, 2, 3], strides = [18, 6, 3, 1]}
// tmp2 = ascir.Store(Store_0, [tmp1])
// tmp2.attr = {axis = [0, 1, 2, 3], repeats = [2, 3, 2, 3], strides = [18, 6, 3, 1]}
// tmp3 = ascir.Output(Output_0, [tmp2])
// )");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape5) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(2), Symbol(2)});
    auto abs = es::Abs(x);
    abs.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(2), Symbol(2)});
    auto abs1 = es::Abs(abs);
    abs1.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(2), Symbol(2)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"2"});
    auto reshape = es::Reshape(abs1, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(4)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_2");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);

//   EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:4, 1:4])
// tmp0 = ascir.Data(Data_0, [])
// tmp0.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
// tmp1 = ascir.Load(Load_0, [tmp0])
// tmp1.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
// tmp2 = ascir.Abs(Abs_0, [tmp1])
// tmp3 = ascir.Abs(Abs_1, [tmp2])
// tmp4 = ascir.Store(Store_0, [tmp3])
// tmp4.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
// tmp5 = ascir.Output(Output_0, [tmp4])
// )");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape6) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(4), Symbol(3)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"4"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(2), Symbol(2), Symbol(3)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);

//   EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:4, 1:2, 2:2, 3:3])
// tmp0 = ascir.Data(Data_0, [])
// tmp0.attr = {axis = [0, 1, 2, 3], repeats = [4, 2, 2, 3], strides = [12, 6, 3, 1]}
// tmp1 = ascir.Load(Load_0, [tmp0])
// tmp1.attr = {axis = [0, 1, 2, 3], repeats = [4, 2, 2, 3], strides = [12, 6, 3, 1]}
// tmp2 = ascir.Store(Store_0, [tmp1])
// tmp2.attr = {axis = [0, 1, 2, 3], repeats = [4, 2, 2, 3], strides = [12, 6, 3, 1]}
// tmp3 = ascir.Output(Output_0, [tmp2])
// )");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape7) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(2), Symbol(2), Symbol(3)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"4"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(4), Symbol(4), Symbol(3)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);

//   EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:4, 1:4, 2:3])
// tmp0 = ascir.Data(Data_0, [])
// tmp0.attr = {axis = [0, 1, 2], repeats = [4, 4, 3], strides = [12, 3, 1]}
// tmp1 = ascir.Load(Load_0, [tmp0])
// tmp1.attr = {axis = [0, 1, 2], repeats = [4, 4, 3], strides = [12, 3, 1]}
// tmp2 = ascir.Store(Store_0, [tmp1])
// tmp2.attr = {axis = [0, 1, 2], repeats = [4, 4, 3], strides = [12, 3, 1]}
// tmp3 = ascir.Output(Output_0, [tmp2])
// )");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshape8) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.GetEsbTensor()->SetSymbolShape({Symbol(128), Symbol(16)});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"3"});
    auto reshape = es::Reshape(x, shape);
    reshape.GetEsbTensor()->SetSymbolShape({Symbol(128), Symbol(1), Symbol(16)});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:128, 1:1, 2:16])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [128, 1, 16], strides = [16, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [128, 1, 16], strides = [16, 0, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [128, 1, 16], strides = [16, 0, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshapeFailure1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"3"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1", "s3"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshapeFailure2) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"4"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1", "s3", "s4"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshapeFailure3) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"2"});
    auto reshape = es::Reshape(x, shape);
    reshape.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, SimpleReshapeFailure4) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto shape = es_graph_->CreateInput(1, "shape", nullptr);
    shape.SetSymbolShape({"2"});
    auto reshape = es::Reshape(x, shape, 0, 1);
    reshape.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(reshape, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Reshape_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceLoweringStrideLessZero) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 10};
    std::vector<int64_t> stride = {2, -1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"o0", "o1", "o2"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  
  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceLoweringInDimNotEqualOutDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 10};
    std::vector<int64_t> stride = {2, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"o0", "o1", "o2", "o3"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceDLoweringBeginLessZero) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> begin = {0, -64, -32};
    std::vector<int64_t> end = {60, 60, 10};
    std::vector<int64_t> stride = {2, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"30", "60", "9"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_NE(asc_graph, nullptr);
  const auto res = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res, R"(AscGraph(graph, axis=[0:30, 1:60, 2:9])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [4224, 33, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [30, 60, 9], strides = [540, 9, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, TestStridedSliceDLoweringOutputEndDimIsOne) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"192", "64", "33"});
    std::vector<int64_t> begin = {0, 0, 1};
    std::vector<int64_t> end = {60, 60, 2};
    std::vector<int64_t> stride = {2, 1, 1};
    auto slice = es::StridedSliceD(data0, begin, end, stride);
    slice.SetSymbolShape({"30", "60", "1"});
    es_graph_->SetOutput(slice, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  auto slice = cg->FindNode("StridedSliceD_0");
  ASSERT_NE(slice, nullptr);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(slice->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  ASSERT_EQ(asc_graph, nullptr);
}


TEST_F(LoopLoweringToAscBackendUT, SimpleBitwiseAndFloat) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::BitwiseAnd(data0, data1);
    rst.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("BitwiseAnd_0");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, SimpleBitwiseAndInt64) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0, s1, s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0, s1, s2"});
    auto rst = es::BitwiseAnd(data0, data1);
    rst.SetSymbolShape({"s0, s1, s2"});
    es_graph_->SetOutput(rst, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("BitwiseAnd_0");
  auto nodeptr1 = cg->FindNode("data0");
  auto nodeptr2 = cg->FindNode("data1");
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT32);
  tmp_desc->SetOriginDataType(DT_INT32);
  auto tmp_desc1 = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetDataType(DT_INT32);
  tmp_desc1->SetOriginDataType(DT_INT32);
  auto tmp_desc2 = nodeptr2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_INT32);
  tmp_desc2->SetOriginDataType(DT_INT32);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0], repeats = [s0], strides = [1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0], repeats = [s0], strides = [1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0], repeats = [s0], strides = [1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0], repeats = [s0], strides = [1]}
tmp4 = ascir.BitwiseAnd(BitwiseAnd_0, [tmp1, tmp3])
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0], repeats = [s0], strides = [1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleLeakyRelu) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto unsqueeze = es::LeakyRelu(data0, 1.5f);
    unsqueeze.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(unsqueeze, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("LeakyRelu_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.LeakyRelu(LeakyRelu_0, [tmp1])
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleApplyAdagradD) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto accum = es_graph_->CreateInput(1, "accum", nullptr);
    accum.SetSymbolShape({"s0", "s1", "s2"});
    auto lr = es_graph_->CreateInput(2, "lr", nullptr);
    lr.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdagradD(var, accum, lr, grad);
    rst.accum.SetSymbolShape({"s0", "s1", "s2"});
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.accum, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdagradD_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Mul(Mul_0, [tmp3, tmp3])
tmp5 = ascir.Add(Add_0, [tmp1, tmp4])
tmp6 = ascir.Data(Data_2, [])
tmp6.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp7 = ascir.Load(Load_2, [tmp6])
tmp7.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp8 = ascir.Data(Data_3, [])
tmp8.attr = {axis = [0, 1, 2], repeats = [1, 1, 1], strides = [0, 0, 0]}
tmp9 = ascir.Load(Load_3, [tmp8])
tmp9.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp10 = ascir.Sqrt(Sqrt_0, [tmp5])
tmp11 = ascir.Mul(Mul_1, [tmp9, tmp3])
tmp12 = ascir.Div(Div_0, [tmp11, tmp10])
tmp13 = ascir.Sub(Sub_0, [tmp7, tmp12])
tmp14 = ascir.Store(Store_0, [tmp13])
tmp14.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp15 = ascir.Output(Output_0, [tmp14])
)");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  auto asc_graph1 = asc_kernel1.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph1, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph1->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Mul(Mul_0, [tmp3, tmp3])
tmp5 = ascir.Add(Add_0, [tmp1, tmp4])
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleApplyAdagradD1) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto accum = es_graph_->CreateInput(1, "accum", nullptr);
    accum.SetSymbolShape({"s0", "s1", "s2"});
    auto lr = es_graph_->CreateInput(2, "lr", nullptr);
    lr.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdagradD(var, accum, lr, grad, false);
    rst.accum.SetSymbolShape({"s0", "s1", "s2"});
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.accum, 1);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdagradD_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Data(Data_3, [])
tmp6.attr = {axis = [0, 1, 2], repeats = [1, 1, 1], strides = [0, 0, 0]}
tmp7 = ascir.Load(Load_3, [tmp6])
tmp7.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp8 = ascir.Sqrt(Sqrt_0, [tmp1])
tmp9 = ascir.Mul(Mul_0, [tmp7, tmp3])
tmp10 = ascir.Div(Div_0, [tmp9, tmp8])
tmp11 = ascir.Sub(Sub_0, [tmp5, tmp10])
tmp12 = ascir.Store(Store_0, [tmp11])
tmp12.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp13 = ascir.Output(Output_0, [tmp12])
)");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  auto asc_graph1 = asc_kernel1.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph1, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph1->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleApplyAdamD) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto m = es_graph_->CreateInput(1, "m", nullptr);
    m.SetSymbolShape({"s0", "s1", "s2"});
    auto v = es_graph_->CreateInput(2, "v", nullptr);
    v.SetSymbolShape({"s0", "s1", "s2"});
    auto beta1_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.08});
    beta1_power.SetSymbolShape({});
    auto beta2_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.25});
    beta2_power.SetSymbolShape({});
    auto lr = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.13});
    lr.SetSymbolShape({});
    auto beta1 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.77});
    beta1.SetSymbolShape({});
    auto beta2 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.05});
    beta2.SetSymbolShape({});
    auto epsilon = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.53});
    epsilon.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdamD(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad);
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    rst.m.SetSymbolShape({"s0", "s1", "s2"});
    rst.v.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.m, 1);
    es_graph_->SetOutput(rst.v, 2);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdamD_6");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Scalar(Scalar_0, [])
tmp7 = ascir.Scalar(Scalar_1, [])
tmp8 = ascir.Scalar(Scalar_2, [])
tmp9 = ascir.Data(Data_3, [])
tmp9.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp10 = ascir.Load(Load_3, [tmp9])
tmp10.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp11 = ascir.Sub(Sub_0, [tmp10, tmp3])
tmp12 = ascir.Mul(Mul_0, [tmp6, tmp11])
tmp13 = ascir.Add(Add_0, [tmp3, tmp12])
tmp14 = ascir.Scalar(Scalar_3, [])
tmp15 = ascir.Mul(Mul_1, [tmp10, tmp10])
tmp16 = ascir.Sub(Sub_1, [tmp15, tmp5])
tmp17 = ascir.Mul(Mul_2, [tmp7, tmp16])
tmp18 = ascir.Add(Add_1, [tmp5, tmp17])
tmp19 = ascir.Sqrt(Sqrt_0, [tmp18])
tmp20 = ascir.Add(Add_2, [tmp8, tmp19])
tmp21 = ascir.Div(Div_0, [tmp13, tmp20])
tmp22 = ascir.Mul(Mul_3, [tmp14, tmp21])
tmp23 = ascir.Sub(Sub_2, [tmp1, tmp22])
tmp24 = ascir.Store(Store_0, [tmp23])
tmp24.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp25 = ascir.Output(Output_0, [tmp24])
)");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  auto asc_graph1 = asc_kernel1.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph1, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph1->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Data(Data_1, [])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Load(Load_1, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Sub(Sub_0, [tmp4, tmp1])
tmp6 = ascir.Mul(Mul_0, [tmp2, tmp5])
tmp7 = ascir.Add(Add_0, [tmp1, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");

  loop::KernelBox asc_kernel2 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(2));
  auto asc_graph2 = asc_kernel2.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph2, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph2->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Data(Data_1, [])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Load(Load_1, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Mul(Mul_0, [tmp4, tmp4])
tmp6 = ascir.Sub(Sub_0, [tmp5, tmp1])
tmp7 = ascir.Mul(Mul_1, [tmp2, tmp6])
tmp8 = ascir.Add(Add_0, [tmp1, tmp7])
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleApplyAdamD1) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto m = es_graph_->CreateInput(1, "m", nullptr);
    m.SetSymbolShape({"s0", "s1", "s2"});
    auto v = es_graph_->CreateInput(2, "v", nullptr);
    v.SetSymbolShape({"s0", "s1", "s2"});
    auto beta1_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.08});
    beta1_power.SetSymbolShape({});
    auto beta2_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.25});
    beta2_power.SetSymbolShape({});
    auto lr = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.13});
    lr.SetSymbolShape({});
    auto beta1 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.77});
    beta1.SetSymbolShape({});
    auto beta2 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.05});
    beta2.SetSymbolShape({});
    auto epsilon = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.53});
    epsilon.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdamD(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, false, true);
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    rst.m.SetSymbolShape({"s0", "s1", "s2"});
    rst.v.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.m, 1);
    es_graph_->SetOutput(rst.v, 2);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdamD_6");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Scalar(Scalar_0, [])
tmp7 = ascir.Scalar(Scalar_1, [])
tmp8 = ascir.Scalar(Scalar_2, [])
tmp9 = ascir.Scalar(Scalar_3, [])
tmp10 = ascir.Data(Data_3, [])
tmp10.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp11 = ascir.Load(Load_3, [tmp10])
tmp11.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp12 = ascir.Sub(Sub_0, [tmp11, tmp3])
tmp13 = ascir.Mul(Mul_0, [tmp7, tmp12])
tmp14 = ascir.Add(Add_0, [tmp3, tmp13])
tmp15 = ascir.Scalar(Scalar_4, [])
tmp16 = ascir.Mul(Mul_1, [tmp11, tmp11])
tmp17 = ascir.Sub(Sub_1, [tmp16, tmp5])
tmp18 = ascir.Mul(Mul_2, [tmp8, tmp17])
tmp19 = ascir.Add(Add_1, [tmp5, tmp18])
tmp20 = ascir.Sqrt(Sqrt_0, [tmp19])
tmp21 = ascir.Add(Add_2, [tmp9, tmp20])
tmp22 = ascir.Mul(Mul_3, [tmp7, tmp11])
tmp23 = ascir.Mul(Mul_4, [tmp14, tmp6])
tmp24 = ascir.Add(Add_3, [tmp23, tmp22])
tmp25 = ascir.Div(Div_0, [tmp24, tmp21])
tmp26 = ascir.Mul(Mul_5, [tmp15, tmp25])
tmp27 = ascir.Sub(Sub_2, [tmp1, tmp26])
tmp28 = ascir.Store(Store_0, [tmp27])
tmp28.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp29 = ascir.Output(Output_0, [tmp28])
)");

  loop::KernelBox asc_kernel1 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(1));
  auto asc_graph1 = asc_kernel1.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph1, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph1->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Data(Data_1, [])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Load(Load_1, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Sub(Sub_0, [tmp4, tmp1])
tmp6 = ascir.Mul(Mul_0, [tmp2, tmp5])
tmp7 = ascir.Add(Add_0, [tmp1, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");

  loop::KernelBox asc_kernel2 = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(2));
  auto asc_graph2 = asc_kernel2.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph2, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph2->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Data(Data_1, [])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Load(Load_1, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Mul(Mul_0, [tmp4, tmp4])
tmp6 = ascir.Sub(Sub_0, [tmp5, tmp1])
tmp7 = ascir.Mul(Mul_1, [tmp2, tmp6])
tmp8 = ascir.Add(Add_0, [tmp1, tmp7])
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleApplyAdamD2_beta1_power_eq_1) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto m = es_graph_->CreateInput(1, "m", nullptr);
    m.SetSymbolShape({"s0", "s1", "s2"});
    auto v = es_graph_->CreateInput(2, "v", nullptr);
    v.SetSymbolShape({"s0", "s1", "s2"});
    auto beta1_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{1.0});
    beta1_power.SetSymbolShape({});
    auto beta2_power = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.25});
    beta2_power.SetSymbolShape({});
    auto lr = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.13});
    lr.SetSymbolShape({});
    auto beta1 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.77});
    beta1.SetSymbolShape({});
    auto beta2 = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.05});
    beta2.SetSymbolShape({});
    auto epsilon = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{0.53});
    epsilon.SetSymbolShape({});
    auto grad = es_graph_->CreateInput(3, "grad", nullptr);
    grad.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyAdamD(var, m, v, beta1_power, beta2_power, lr, beta1, beta2, epsilon, grad, false, true);
    rst.var.SetSymbolShape({"s0", "s1", "s2"});
    rst.m.SetSymbolShape({"s0", "s1", "s2"});
    rst.v.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst.var, 0);
    es_graph_->SetOutput(rst.m, 1);
    es_graph_->SetOutput(rst.v, 2);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyAdamD_6");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_EQ(asc_graph, nullptr);
}

TEST_F(LoopLoweringToAscBackendUT, SimpleApplyGradientDescent) {
  [this]() {
    auto var = es_graph_->CreateInput(0, "var", nullptr);
    var.SetSymbolShape({"s0", "s1", "s2"});
    auto alpha = es_graph_->CreateInput(1, "alpha", nullptr);
    alpha.SetSymbolShape({});
    auto delta = es_graph_->CreateInput(2, "delta", nullptr);
    delta.SetSymbolShape({"s0", "s1", "s2"});
    auto rst = es::ApplyGradientDescent(var, alpha, delta);
    rst.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(rst, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("ApplyGradientDescent_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2], repeats = [1, 1, 1], strides = [0, 0, 0]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp6 = ascir.Mul(Mul_0, [tmp5, tmp3])
tmp7 = ascir.Sub(Sub_0, [tmp1, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleExpandDims) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"28", "28"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT32, {1}, std::vector<int32_t>{1});
    auto axis1 = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{1});
    auto expanddims = es::ExpandDims(data0, axis);
    expanddims.SetSymbolShape({"28", "1", "28"});
    auto expanddims1 = es::ExpandDims(data0, axis1);
    expanddims1.SetSymbolShape({"28", "1", "28"});
    es_graph_->SetOutput(expanddims, 0);
    es_graph_->SetOutput(expanddims1, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto expanddims = cg->FindNode("ExpandDims_2");
  auto expanddims1 = cg->FindNode("ExpandDims_3");
  ASSERT_NE(expanddims, nullptr);
  ASSERT_NE(expanddims1, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(expanddims->GetOutDataAnchor(0));
  auto asc_graph = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:28, 1:1, 2:28])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [28, 1, 28], strides = [28, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [28, 1, 28], strides = [28, 0, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [28, 1, 28], strides = [28, 0, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");

  auto kernel1 = ge::loop::GetKernelBox(expanddims1->GetOutDataAnchor(0));
  auto asc_graph1 = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph1, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph1->Graph()), R"(AscGraph(graph, axis=[0:28, 1:1, 2:28])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [28, 1, 28], strides = [28, 0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [28, 1, 28], strides = [28, 0, 1]}
tmp2 = ascir.Store(Store_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [28, 1, 28], strides = [28, 0, 1]}
tmp3 = ascir.Output(Output_0, [tmp2])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleSelect) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1"});
    auto select = es::Select(data0, data1, data2);
    select.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(select, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Select_0");
  auto nodeptr1 = cg->FindNode("data0");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_UINT8);
  tmp_desc->SetOriginDataType(DT_UINT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, 1], strides = [1, 0]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp6 = ascir.Where(Where_0, [tmp1, tmp3, tmp5])
tmp7 = ascir.Store(Store_0, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp8 = ascir.Output(Output_0, [tmp7])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringDimEndGatherV2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"s0", "s1", "s3", "s4"});
    es_graph_->SetOutput(gather, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto gather = cg->FindNode("GatherV2_1");
  auto nodeptr = cg->FindNode("data1");
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  ASSERT_NE(gather, nullptr);
  ge::AttrUtils::SetInt(gather->GetOpDesc(), "_op_impl_mode_enum", 0x4);
  ge::AttrUtils::SetInt(gather->GetOpDesc(), "batch_dims", 0x0);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(gather->GetOpDesc(), "_op_impl_mode_enum", op_impl_mode);
  EXPECT_TRUE(op_impl_mode == 4);

  int64_t batch_dims = -1;
  ge::AttrUtils::GetInt(gather->GetOpDesc(), "batch_dims", batch_dims);
  EXPECT_TRUE(batch_dims == 0);

  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(gather->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsRealized());
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  auto gather_node = asc_graph->Graph().FindNode("graph/Gather_0");
  ASSERT_NE(gather_node, nullptr);
  ASSERT_NE(gather_node->GetOpDesc(), nullptr);

  const auto &attr = gather_node->GetOpDesc()->GetAttrsGroup<AscNodeAttr>();
  ASSERT_NE(attr, nullptr);
  auto gather_attr = dynamic_cast<ascir_op::Gather::AscGatherIrAttrDef *>(attr->ir_attr.get());
  ASSERT_NE(gather_attr, nullptr);
  int64_t gather_axis;
  ASSERT_EQ(gather_attr->GetAxis(gather_axis), 0);
  EXPECT_EQ(gather_axis, 2);
  std::cout << ReadableAscGraph(asc_graph->Graph()) << std::endl;
}

TEST_F(LoopLoweringToAscBackendUT, LoweringDim0GatherV2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
    axis.SetSymbolShape({});
    auto gather = es::GatherV2(data0, data1, axis);
    gather.SetSymbolShape({"s0", "s1", "s3", "s4"});
    es_graph_->SetOutput(gather, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto gather = cg->FindNode("GatherV2_1");
  ASSERT_NE(gather, nullptr);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  ge::AttrUtils::SetInt(asc_node->GetOpDesc(), "_op_impl_mode_enum", 0x4);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(asc_node->GetOpDesc(), "_op_impl_mode_enum", op_impl_mode);
  EXPECT_TRUE(op_impl_mode == 4);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kGather);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(GatherV2_1_out0_graph, axis=[0:s0, 1:s1, 2:s3, 3:s4])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Data(Data_1, [])
tmp1.attr = {axis = [0, 1], repeats = [s3, s4], strides = [s4, 1]}
tmp2 = ascir.Gather(Gather_0, [tmp0, tmp1])
tmp2.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s3, s4], strides = [(s1 * s3 * s4), (s3 * s4), s4, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s3, s4], strides = [(s1 * s3 * s4), (s3 * s4), s4, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringDim0Gather) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s3", "s4"});
    auto gather = es::Gather(data0, data1);
    gather.SetSymbolShape({"s3", "s4", "s1", "s2"});
    es_graph_->SetOutput(gather, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto gather = cg->FindNode("Gather_0");
  ASSERT_NE(gather, nullptr);
  auto nodeptr = cg->FindNode("data1");
  ASSERT_NE(nodeptr, nullptr);
  auto tmp_desc = nodeptr->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_INT64);
  tmp_desc->SetOriginDataType(DT_INT64);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);

  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);
  
  ge::AttrUtils::SetInt(asc_node->GetOpDesc(), "_op_impl_mode_enum", 0x4);

  int64_t op_impl_mode = -1;
  ge::AttrUtils::GetInt(asc_node->GetOpDesc(), "_op_impl_mode_enum", op_impl_mode);
  EXPECT_TRUE(op_impl_mode == 4);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kGather);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(Gather_0_out0_graph, axis=[0:s3, 1:s4, 2:s1, 3:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Data(Data_1, [])
tmp1.attr = {axis = [0, 1], repeats = [s3, s4], strides = [s4, 1]}
tmp2 = ascir.Gather(Gather_0, [tmp0, tmp1])
tmp2.attr = {axis = [0, 1, 2, 3], repeats = [s3, s4, s1, s2], strides = [(s1 * s2 * s4), (s1 * s2), s2, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2, 3], repeats = [s3, s4, s1, s2], strides = [(s1 * s2 * s4), (s1 * s2), s2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringSelectV2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1"});
    auto select = es::SelectV2(data0, data1, data2);
    select.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(select, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("SelectV2_0");
  ASSERT_NE(nodeptr, nullptr);
  auto nodeptr1 = cg->FindNode("data0");
  ASSERT_NE(nodeptr1, nullptr);
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_UINT8);
  tmp_desc->SetOriginDataType(DT_UINT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [1, s1], strides = [0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp6 = ascir.Where(Where_0, [tmp1, tmp3, tmp5])
tmp7 = ascir.Store(Store_0, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp8 = ascir.Output(Output_0, [tmp7])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringElu) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto elu = es::Elu(data0, 0.5, 0.8, 0.8);
    elu.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(elu, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Elu_0");
  auto nodeptr1 = cg->FindNode("data0");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Scalar(Scalar_1, [])
tmp4 = ascir.Scalar(Scalar_2, [])
tmp5 = ascir.Scalar(Scalar_3, [])
tmp6 = ascir.Minimum(Minimum_0, [tmp1, tmp2])
tmp7 = ascir.Mul(Mul_0, [tmp6, tmp5])
tmp8 = ascir.Exp(Exp_0, [tmp7])
tmp9 = ascir.Sub(Sub_0, [tmp8, tmp3])
tmp10 = ascir.Maximum(Maximum_0, [tmp1, tmp2])
tmp11 = ascir.Mul(Mul_1, [tmp9, tmp4])
tmp12 = ascir.Add(Add_0, [tmp11, tmp10])
tmp13 = ascir.Mul(Mul_2, [tmp12, tmp5])
tmp14 = ascir.Store(Store_0, [tmp13])
tmp14.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp15 = ascir.Output(Output_0, [tmp14])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringEluGrad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto elu_grad = es::EluGrad(data0, data1);
    elu_grad.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(elu_grad, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("EluGrad_0");
  ASSERT_NE(nullptr, nodeptr);
  auto nodeptr1 = cg->FindNode("data0");
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp4 = ascir.Scalar(Scalar_0, [])
tmp5 = ascir.Scalar(Scalar_1, [])
tmp6 = ascir.Minimum(Minimum_0, [tmp3, tmp4])
tmp7 = ascir.Add(Add_0, [tmp6, tmp5])
tmp8 = ascir.Mul(Mul_0, [tmp7, tmp1])
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringTanhGrad) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto tanhgrad = es::TanhGrad(data0, data1);
    tanhgrad.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(tanhgrad, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("TanhGrad_0");
  ASSERT_NE(nullptr, nodeptr);
  auto nodeptr1 = cg->FindNode("data0");
  ASSERT_NE(nullptr, nodeptr1);
  auto tmp_desc = nodeptr1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(DT_FLOAT);
  tmp_desc->SetOriginDataType(DT_FLOAT);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp4 = ascir.Scalar(Scalar_0, [])
tmp5 = ascir.Mul(Mul_0, [tmp1, tmp1])
tmp6 = ascir.Sub(Sub_0, [tmp4, tmp5])
tmp7 = ascir.Mul(Mul_1, [tmp3, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, FusedMulAddNLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"s0", "s1"});
    auto data2 = es_graph_->CreateInput(2, "data2", nullptr);
    data2.SetSymbolShape({"s0", "s1"});
    auto fused_mul_addn = es::FusedMulAddN(data0, data1, data2);
    fused_mul_addn.SetSymbolShape({"s0", "s1"});
    es_graph_->SetOutput(fused_mul_addn, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto fused_mul_addn = cg->FindNode("FusedMulAddN_0");
  ASSERT_NE(fused_mul_addn, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto asc_kernel = ge::loop::GetKernelBox(fused_mul_addn->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp6 = ascir.Mul(Mul_0, [tmp1, tmp5])
tmp7 = ascir.Add(Add_0, [tmp6, tmp3])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, L2LossLowering) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1"});
    auto l2_loss = es::L2Loss(data0);
    l2_loss.SetSymbolShape({});
    es_graph_->SetOutput(l2_loss, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto l2_loss = cg->FindNode("L2Loss_0");
  ASSERT_NE(l2_loss, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto asc_kernel = ge::loop::GetKernelBox(l2_loss->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [s0, s1], strides = [s1, 1]}
tmp2 = ascir.Mul(Mul_0, [tmp1, tmp1])
tmp3 = ascir.Scalar(Scalar_0, [])
tmp4 = ascir.Mul(Mul_1, [tmp2, tmp3])
tmp5 = ascir.Sum(Sum_0, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [1, 1], strides = [0, 0]}
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1], repeats = [1, 1], strides = [0, 0]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleZeroslikeBool) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"28", "28"});
    auto expanddims = es::ZerosLike(data0);
    expanddims.SetSymbolShape({"28", "28"});
    es_graph_->SetOutput(expanddims, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto data0 = cg->FindNode("data0");
  auto data_desc1 = data0->GetOpDesc()->MutableInputDesc(0);
  data_desc1->SetDataType(ge::DT_BOOL);
  data_desc1->SetOriginDataType(ge::DT_BOOL);
  auto data_desc = data0->GetOpDesc()->MutableOutputDesc(0);
  data_desc->SetDataType(ge::DT_BOOL);
  data_desc->SetOriginDataType(ge::DT_BOOL);
  auto tmp = cg->FindNode("ZerosLike_0");
  auto tmp_desc1 = tmp->GetOpDesc()->MutableInputDesc(0);
  tmp_desc1->SetDataType(ge::DT_BOOL);
  tmp_desc1->SetOriginDataType(ge::DT_BOOL);
  auto tmp_desc = tmp->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc->SetDataType(ge::DT_BOOL);
  tmp_desc->SetOriginDataType(ge::DT_BOOL);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));
  auto asc_graph = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:28, 1:28])
tmp0 = ascir.Scalar(Scalar_0, [])
tmp1 = ascir.Store(Store_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [28, 28], strides = [28, 1]}
tmp2 = ascir.Output(Output_0, [tmp1])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleClipByValueConstScalar) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"28", "28"});
    auto min = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{1.0});
    min.SetSymbolShape({});
    auto max = CreateConst(*es_graph_, ge::DT_FLOAT, {}, std::vector<float>{2.0});
    max.SetSymbolShape({});
    auto expanddims = es::ClipByValue(data0, min, max);
    expanddims.SetSymbolShape({"28", "28"});
    es_graph_->SetOutput(expanddims, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto expanddims = cg->FindNode("ClipByValue_2");
  ASSERT_NE(expanddims, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(expanddims->GetOutDataAnchor(0));
  auto asc_graph = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:28, 1:28])
tmp0 = ascir.Scalar(Scalar_0, [])
tmp1 = ascir.Scalar(Scalar_1, [])
tmp2 = ascir.Data(Data_0, [])
tmp2.attr = {axis = [0, 1], repeats = [28, 28], strides = [28, 1]}
tmp3 = ascir.Load(Load_0, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [28, 28], strides = [28, 1]}
tmp4 = ascir.Minimum(Minimum_0, [tmp3, tmp1])
tmp5 = ascir.Maximum(Maximum_0, [tmp4, tmp0])
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1], repeats = [28, 28], strides = [28, 1]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, ClipByValueWithTensorInput) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"2", "28"});
    auto min = es_graph_->CreateInput(1, "data1", nullptr);
    min.SetSymbolShape({"28"});
    auto max = es_graph_->CreateInput(2, "data2", nullptr);
    max.SetSymbolShape({"28"});
    auto expanddims = es::ClipByValue(data0, min, max);
    expanddims.SetSymbolShape({"2", "28"});
    es_graph_->SetOutput(expanddims, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto expanddims = cg->FindNode("ClipByValue_0");
  ASSERT_NE(expanddims, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(expanddims->GetOutDataAnchor(0));
  auto asc_graph = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:2, 1:28])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [1, 28], strides = [0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [2, 28], strides = [28, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [1, 28], strides = [0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [2, 28], strides = [28, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [2, 28], strides = [28, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [2, 28], strides = [28, 1]}
tmp6 = ascir.Minimum(Minimum_0, [tmp5, tmp3])
tmp7 = ascir.Maximum(Maximum_0, [tmp6, tmp1])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1], repeats = [2, 28], strides = [28, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleBNInference) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto mean = es_graph_->CreateInput(1, "mean", nullptr);
    mean.SetSymbolShape({"s3"});
    auto variance = es_graph_->CreateInput(2, "variance", nullptr);
    variance.SetSymbolShape({"s3"});
    auto bninferenced = es::BNInferenceD(x, mean, variance);
    bninferenced.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(bninferenced, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("BNInferenceD_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  auto asc_graph = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2, 3:s3])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp6 = ascir.Add(Add_0, [tmp1, tmp3])
tmp7 = ascir.Mul(Mul_0, [tmp5, tmp6])
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleBNInference1) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto mean = es_graph_->CreateInput(1, "mean", nullptr);
    mean.SetSymbolShape({"s3"});
    auto variance = es_graph_->CreateInput(2, "variance", nullptr);
    variance.SetSymbolShape({"s3"});
    auto scale = es_graph_->CreateInput(3, "scale", nullptr);
    scale.SetSymbolShape({"s3"});
    auto bninferenced = es::BNInferenceD(x, mean, variance, scale);
    bninferenced.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(bninferenced, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("BNInferenceD_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  auto asc_graph = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2, 3:s3])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp6 = ascir.Data(Data_3, [])
tmp6.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp7 = ascir.Load(Load_3, [tmp6])
tmp7.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp8 = ascir.Add(Add_0, [tmp1, tmp3])
tmp9 = ascir.Mul(Mul_0, [tmp5, tmp8])
tmp10 = ascir.Mul(Mul_1, [tmp9, tmp7])
tmp11 = ascir.Store(Store_0, [tmp10])
tmp11.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp12 = ascir.Output(Output_0, [tmp11])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleBNInference2) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2", "s3"});
    auto mean = es_graph_->CreateInput(1, "mean", nullptr);
    mean.SetSymbolShape({"s3"});
    auto variance = es_graph_->CreateInput(2, "variance", nullptr);
    variance.SetSymbolShape({"s3"});
    auto scale = es_graph_->CreateInput(3, "scale", nullptr);
    scale.SetSymbolShape({"s3"});
    auto b = es_graph_->CreateInput(4, "b", nullptr);
    b.SetSymbolShape({"s3"});
    auto bninferenced = es::BNInferenceD(x, mean, variance, scale, b);
    bninferenced.SetSymbolShape({"s0", "s1", "s2", "s3"});
    es_graph_->SetOutput(bninferenced, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto node = cg->FindNode("BNInferenceD_0");
  ASSERT_NE(node, nullptr);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto kernel = ge::loop::GetKernelBox(node->GetOutDataAnchor(0));
  auto asc_graph = kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2, 3:s3])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp6 = ascir.Data(Data_3, [])
tmp6.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp7 = ascir.Load(Load_3, [tmp6])
tmp7.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp8 = ascir.Data(Data_4, [])
tmp8.attr = {axis = [0, 1, 2, 3], repeats = [1, 1, 1, s3], strides = [0, 0, 0, 1]}
tmp9 = ascir.Load(Load_4, [tmp8])
tmp9.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp10 = ascir.Add(Add_0, [tmp1, tmp3])
tmp11 = ascir.Mul(Mul_0, [tmp5, tmp10])
tmp12 = ascir.Mul(Mul_1, [tmp11, tmp7])
tmp13 = ascir.Add(Add_1, [tmp12, tmp9])
tmp14 = ascir.Store(Store_0, [tmp13])
tmp14.attr = {axis = [0, 1, 2, 3], repeats = [s0, s1, s2, s3], strides = [(s1 * s2 * s3), (s2 * s3), s3, 1]}
tmp15 = ascir.Output(Output_0, [tmp14])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SimpleLog1p) {
  [this]() {
    auto x = es_graph_->CreateInput(0, "x", nullptr);
    x.SetSymbolShape({"s0", "s1", "s2"});
    auto log1p = es::Log1p(x);

    log1p.SetSymbolShape({"s0", "s1", "s2"});
    es_graph_->SetOutput(log1p, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto nodeptr = cg->FindNode("Log1p_0");

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(nodeptr->GetOutDataAnchor(0));
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);

  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Scalar(Scalar_0, [])
tmp3 = ascir.Scalar(Scalar_1, [])
tmp4 = ascir.Scalar(Scalar_2, [])
tmp5 = ascir.Add(Add_0, [tmp1, tmp2])
tmp6 = ascir.Add(Add_1, [tmp5, tmp3])
tmp7 = ascir.Div(Div_0, [tmp1, tmp6])
tmp8 = ascir.Ln(Ln_0, [tmp5])
tmp9 = ascir.Mul(Mul_0, [tmp8, tmp7])
tmp10 = ascir.Ne(Ne_0, [tmp5, tmp2])
tmp11 = ascir.Where(Where_0, [tmp10, tmp9, tmp1])
tmp12 = ascir.Ne(Ne_1, [tmp5, tmp4])
tmp13 = ascir.Where(Where_1, [tmp12, tmp11, tmp7])
tmp14 = ascir.Store(Store_0, [tmp13])
tmp14.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp15 = ascir.Output(Output_0, [tmp14])
)");
}

#define TEST_F_LOWER_ASCBACKEND_INST1(LOOPOP, ESOP)                                                    \
  TEST_F(LoopLoweringToAscBackendUT, Simple##LOOPOP) {                                                 \
    [this]() {                                                                                         \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                                        \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto tmp = es::ESOP(data0);                                                                      \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                                          \
      es_graph_->SetOutput(tmp, 0);                                                                    \
    }();                                                                                               \
    auto graph = es_graph_->Build();                                                                   \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                                   \
    auto tmp = cg->FindNode(#ESOP "_0");                                                               \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                                      \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));                     \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");                                  \
    ASSERT_NE(asc_graph, nullptr);                                                                     \
    EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()),                                                    \
              "AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])\n"                                             \
              "tmp0 = ascir.Data(Data_0, [])\n"                                                        \
              "tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp1 = ascir.Load(Load_0, [tmp0])\n"                                                    \
              "tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp2 = ascir." #LOOPOP "(" #LOOPOP                                                      \
              "_0, [tmp1])\n"                                                                          \
              "tmp3 = ascir.Store(Store_0, [tmp2])\n"                                                  \
              "tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp4 = ascir.Output(Output_0, [tmp3])\n");                                              \
  }

#define TEST_F_LOWER_ASCBACKEND_INST1_WITH_DTYPE(LOOPOP, ESOP, OP_DTYPE)                               \
  TEST_F(LoopLoweringToAscBackendUT, Simple##LOOPOP) {                                                 \
    [this]() {                                                                                         \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                                        \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto tmp = es::ESOP(data0);                                                                      \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                                          \
      es_graph_->SetOutput(tmp, 0);                                                                    \
    }();                                                                                               \
    auto graph = es_graph_->Build();                                                                   \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                                   \
    auto tmp = cg->FindNode(#ESOP "_0");                                                               \
    auto tmp_desc = tmp->GetOpDesc()->MutableOutputDesc(0);                                            \
    tmp_desc->SetDataType(OP_DTYPE);                                                                   \
    tmp_desc->SetOriginDataType(OP_DTYPE);                                                             \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                                      \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));                     \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");                                  \
    ASSERT_NE(asc_graph, nullptr);                                                                     \
    EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()),                                                    \
              "AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])\n"                                             \
              "tmp0 = ascir.Data(Data_0, [])\n"                                                        \
              "tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp1 = ascir.Load(Load_0, [tmp0])\n"                                                    \
              "tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp2 = ascir." #LOOPOP "(" #LOOPOP                                                      \
              "_0, [tmp1])\n"                                                                          \
              "tmp3 = ascir.Store(Store_0, [tmp2])\n"                                                  \
              "tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp4 = ascir.Output(Output_0, [tmp3])\n");                                              \
  }

#define TEST_F_LOWER_ASCBACKEND_INST2(LOOPOP, ESOP)                                                    \
  TEST_F(LoopLoweringToAscBackendUT, Simple##LOOPOP) {                                                 \
    [this]() {                                                                                         \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                                        \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto data1 = es_graph_->CreateInput(1, "data1", nullptr);                                        \
      data1.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto tmp = es::ESOP(data0, data1);                                                               \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                                          \
      es_graph_->SetOutput(tmp, 0);                                                                    \
    }();                                                                                               \
                                                                                                       \
    auto graph = es_graph_->Build();                                                                   \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                                   \
    auto tmp = cg->FindNode(#ESOP "_0");                                                               \
                                                                                                       \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                                      \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));                     \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");                                  \
                                                                                                       \
    ASSERT_NE(asc_graph, nullptr);                                                                     \
    EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()),                                                    \
              "AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])\n"                                             \
              "tmp0 = ascir.Data(Data_0, [])\n"                                                        \
              "tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp1 = ascir.Load(Load_0, [tmp0])\n"                                                    \
              "tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp2 = ascir.Data(Data_1, [])\n"                                                        \
              "tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp3 = ascir.Load(Load_1, [tmp2])\n"                                                    \
              "tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp4 = ascir." #LOOPOP "(" #LOOPOP                                                      \
              "_0, [tmp1, tmp3])\n"                                                                    \
              "tmp5 = ascir.Store(Store_0, [tmp4])\n"                                                  \
              "tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp6 = ascir.Output(Output_0, [tmp5])\n");                                              \
  }

#define TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(LOOPOP, ESOP, OP_DTYPE)                               \
  TEST_F(LoopLoweringToAscBackendUT, Simple##LOOPOP) {                                                 \
    [this]() {                                                                                         \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                                        \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto data1 = es_graph_->CreateInput(1, "data1", nullptr);                                        \
      data1.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto tmp = es::ESOP(data0, data1);                                                               \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                                          \
      es_graph_->SetOutput(tmp, 0);                                                                    \
    }();                                                                                               \
                                                                                                       \
    auto graph = es_graph_->Build();                                                                   \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                                   \
    auto tmp = cg->FindNode(#ESOP "_0");                                                               \
    auto tmp_desc = tmp->GetOpDesc()->MutableOutputDesc(0);                                            \
    tmp_desc->SetDataType(OP_DTYPE);                                                                   \
    tmp_desc->SetOriginDataType(OP_DTYPE);                                                             \
                                                                                                       \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                                      \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));                     \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");                                  \
                                                                                                       \
    ASSERT_NE(asc_graph, nullptr);                                                                     \
    EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()),                                                    \
              "AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])\n"                                             \
              "tmp0 = ascir.Data(Data_0, [])\n"                                                        \
              "tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp1 = ascir.Load(Load_0, [tmp0])\n"                                                    \
              "tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp2 = ascir.Data(Data_1, [])\n"                                                        \
              "tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp3 = ascir.Load(Load_1, [tmp2])\n"                                                    \
              "tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp4 = ascir." #LOOPOP "(" #LOOPOP                                                      \
              "_0, [tmp1, tmp3])\n"                                                                    \
              "tmp5 = ascir.Store(Store_0, [tmp4])\n"                                                  \
              "tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp6 = ascir.Output(Output_0, [tmp5])\n");                                              \
  }

#define TEST_F_LOWER_ASCBACKEND_INST3(LOOPOP, ESOP)                                                    \
  TEST_F(LoopLoweringToAscBackendUT, Simple##ESOP) {                                                   \
    [this]() {                                                                                         \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                                        \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto data1 = es_graph_->CreateInput(1, "data1", nullptr);                                        \
      data1.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto data2 = es_graph_->CreateInput(2, "data2", nullptr);                                        \
      data2.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto tmp = es::ESOP(data0, data1, data2);                                                        \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                                          \
      es_graph_->SetOutput(tmp, 0);                                                                    \
    }();                                                                                               \
                                                                                                       \
    auto graph = es_graph_->Build();                                                                   \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                                   \
    auto tmp = cg->FindNode(#ESOP "_0");                                                               \
                                                                                                       \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                                      \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(tmp->GetOutDataAnchor(0));                     \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");                                  \
                                                                                                       \
    ASSERT_NE(asc_graph, nullptr);                                                                     \
    EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()),                                                    \
              "AscGraph(graph, axis=[0:s0, 1:s1, 2:s2])\n"                                             \
              "tmp0 = ascir.Data(Data_0, [])\n"                                                        \
              "tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp1 = ascir.Load(Load_0, [tmp0])\n"                                                    \
              "tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp2 = ascir.Data(Data_1, [])\n"                                                        \
              "tmp2.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp3 = ascir.Load(Load_1, [tmp2])\n"                                                    \
              "tmp3.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp4 = ascir.Data(Data_2, [])\n"                                                        \
              "tmp4.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp5 = ascir.Load(Load_2, [tmp4])\n"                                                    \
              "tmp5.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp6 = ascir." #LOOPOP "(" #LOOPOP                                                      \
              "_0, [tmp1, tmp3, tmp5])\n"                                                              \
              "tmp7 = ascir.Store(Store_0, [tmp6])\n"                                                  \
              "tmp7.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp8 = ascir.Output(Output_0, [tmp7])\n");                                              \
  }

#define TEST_F_NOTIMPLEMENT_INST1(LOOPOP, ESOP)                                     \
  REGISTER_POINTWISE_LOWER(ESOP, loop::LOOPOP);                                     \
  TEST_F(LoopLoweringToAscBackendUT, Simple##LOOPOP) {                              \
    [this]() {                                                                      \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                     \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                     \
      auto tmp = es::ESOP(data0);                                                   \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                       \
      es_graph_->SetOutput(tmp, 0);                                                 \
    }();                                                                            \
                                                                                    \
    auto graph = es_graph_->Build();                                                \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                \
    auto acos = cg->FindNode(#ESOP "_0");                                           \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                   \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(acos->GetOutDataAnchor(0)); \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");               \
    ASSERT_EQ(asc_graph, nullptr);                                                  \
  }

#define TEST_F_NOTIMPLEMENT_INST2(LOOPOP, ESOP)                                     \
  REGISTER_POINTWISE_LOWER(ESOP, loop::LOOPOP);                                     \
  TEST_F(LoopLoweringToAscBackendUT, Simple##LOOPOP) {                              \
    [this]() {                                                                      \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                     \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                     \
      auto data1 = es_graph_->CreateInput(1, "data1", nullptr);                     \
      data1.SetSymbolShape({"s0", "s1", "s2"});                                     \
      auto tmp = es::ESOP(data0, data1);                                            \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                       \
      es_graph_->SetOutput(tmp, 0);                                                 \
    }();                                                                            \
                                                                                    \
    auto graph = es_graph_->Build();                                                \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                \
    auto acos = cg->FindNode(#ESOP "_0");                                           \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                   \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(acos->GetOutDataAnchor(0)); \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");               \
    ASSERT_EQ(asc_graph, nullptr);                                                  \
  }

#define TEST_F_NOTIMPLEMENT_INST3(LOOPOP, ESOP)                                     \
  REGISTER_POINTWISE_LOWER(ESOP, loop::LOOPOP);                                     \
  TEST_F(LoopLoweringToAscBackendUT, Simple##LOOPOP) {                              \
    [this]() {                                                                      \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                     \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                     \
      auto data1 = es_graph_->CreateInput(1, "data1", nullptr);                     \
      data1.SetSymbolShape({"s0", "s1", "s2"});                                     \
      auto data2 = es_graph_->CreateInput(2, "data2", nullptr);                     \
      data2.SetSymbolShape({"s0", "s1", "s2"});                                     \
      auto tmp = es::ESOP(data0, data1, data2);                                     \
      tmp.SetSymbolShape({"s0", "s1", "s2"});                                       \
      es_graph_->SetOutput(tmp, 0);                                                 \
    }();                                                                            \
                                                                                    \
    auto graph = es_graph_->Build();                                                \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                \
    auto acos = cg->FindNode(#ESOP "_0");                                           \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                   \
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(acos->GetOutDataAnchor(0)); \
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");               \
    ASSERT_EQ(asc_graph, nullptr);                                                  \
  }

#define TEST_F_LOWER_REDUCE_ASCBACKEND_INST1(LOOPOP, ESOP)                                             \
  TEST_F(LoopLoweringToAscBackendUT, FusePointwiseWithEndKeepDimReduction##ESOP) {                     \
    [this]() {                                                                                         \
      auto data0 = es_graph_->CreateInput(0, "data0", nullptr);                                        \
      data0.SetSymbolShape({"s0", "s1", "s2"});                                                        \
      auto abs = es::Abs(data0);                                                                       \
      abs.SetSymbolShape({"s0", "s1", "s2"});                                                          \
      auto sum = es::ESOP(abs, {1}, true);                                                             \
      sum.SetSymbolShape({"s0", "1", "s2"});                                                           \
      es_graph_->SetOutput(sum, 0);                                                                    \
    }();                                                                                               \
    auto graph = es_graph_->Build();                                                                   \
    auto cg = GraphUtilsEx::GetComputeGraph(*graph);                                                   \
    ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);                                      \
    ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);                            \
    auto asc_node = cg->FindFirstNodeMatchType("AscBackend");                                          \
    ASSERT_NE(asc_node, nullptr);                                                                      \
    auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();                          \
    ASSERT_NE(fused_attrs, nullptr);                                                                   \
    ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);                                                    \
    EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kReduction);                                 \
    EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()),                                           \
              "AscGraph(" #ESOP                                                                        \
              "_1_out0_graph, axis=[0:s0, 1:s1, 2:s2])\n"                                              \
              "tmp0 = ascir.Data(Data_0, [])\n"                                                        \
              "tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp1 = ascir.Load(Load_0, [tmp0])\n"                                                    \
              "tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n" \
              "tmp2 = ascir.Abs(Abs_0, [tmp1])\n"                                                      \
              "tmp3 = ascir." #LOOPOP "(" #LOOPOP                                                      \
              "_0, [tmp2])\n"                                                                          \
              "tmp3.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}\n"          \
              "tmp4 = ascir.Store(Store_0, [tmp3])\n"                                                  \
              "tmp4.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}\n"          \
              "tmp5 = ascir.Output(Output_0, [tmp4])\n");                                              \
  }

TEST_F_LOWER_ASCBACKEND_INST1(Abs, Abs)
TEST_F_LOWER_ASCBACKEND_INST1(Erf, Erf)
TEST_F_LOWER_ASCBACKEND_INST1(Exp, Exp)
TEST_F_LOWER_ASCBACKEND_INST1_WITH_DTYPE(IsFinite, IsFinite, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST1_WITH_DTYPE(Isnan, IsNan, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST1(LogicalNot, LogicalNot)
TEST_F_LOWER_ASCBACKEND_INST1(Neg, Neg)
TEST_F_LOWER_ASCBACKEND_INST1(Reciprocal, Reciprocal)
TEST_F_LOWER_ASCBACKEND_INST1(Relu, Relu)
TEST_F_LOWER_ASCBACKEND_INST1(Rsqrt, Rsqrt)
TEST_F_LOWER_ASCBACKEND_INST1(Sigmoid, Sigmoid)
TEST_F_LOWER_ASCBACKEND_INST1(Sign, Sign)
TEST_F_LOWER_ASCBACKEND_INST1(Sqrt, Sqrt)
TEST_F_LOWER_ASCBACKEND_INST1(Tanh, Tanh)
TEST_F_LOWER_ASCBACKEND_INST1(Gelu, Gelu)

TEST_F_LOWER_ASCBACKEND_INST2(Add, Add)
TEST_F_LOWER_ASCBACKEND_INST2(Div, Div)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(Eq, Equal, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(Ge, GreaterEqual, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(Gt, Greater, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(Le, LessEqual, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(Lt, Less, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(LogicalAnd, LogicalAnd, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(LogicalOr, LogicalOr, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2(Maximum, Maximum)
TEST_F_LOWER_ASCBACKEND_INST2(Minimum, Minimum)
TEST_F_LOWER_ASCBACKEND_INST2(Mul, Mul)
TEST_F_LOWER_ASCBACKEND_INST2_WITH_DTYPE(Ne, NotEqual, DT_BOOL)
TEST_F_LOWER_ASCBACKEND_INST2(Pow, Pow)
TEST_F_LOWER_ASCBACKEND_INST2(Sub, Sub)
TEST_F_LOWER_ASCBACKEND_INST2(TrueDiv, RealDiv)
TEST_F_LOWER_ASCBACKEND_INST2(FloorDiv, FloorDiv)

TEST_F_NOTIMPLEMENT_INST1(Acos, Acos)
TEST_F_NOTIMPLEMENT_INST1(Acosh, Acosh)
TEST_F_NOTIMPLEMENT_INST1(Asin, Asin)
TEST_F_NOTIMPLEMENT_INST1(Asinh, Asinh)
TEST_F_NOTIMPLEMENT_INST1(Atan, Atan)
TEST_F_NOTIMPLEMENT_INST1(Atanh, Atanh)
TEST_F_NOTIMPLEMENT_INST1(Ceil, Ceil)
TEST_F_NOTIMPLEMENT_INST1(Cos, Cos)
TEST_F_NOTIMPLEMENT_INST1(Cosh, Cosh)
TEST_F_NOTIMPLEMENT_INST1(Erfc, Erfc)
TEST_F_NOTIMPLEMENT_INST1(Erfinv, Erfinv)
TEST_F_NOTIMPLEMENT_INST1(Floor, Floor)
TEST_F_NOTIMPLEMENT_INST1(Identity, Identity)
TEST_F_NOTIMPLEMENT_INST1(IsInf, IsInf)
TEST_F_NOTIMPLEMENT_INST1(Lgamma, Lgamma)
TEST_F_NOTIMPLEMENT_INST1(Round, Round)
TEST_F_NOTIMPLEMENT_INST1(Sin, Sin)
TEST_F_NOTIMPLEMENT_INST1(Sinh, Sinh)
TEST_F_NOTIMPLEMENT_INST1(Tan, Tan)
TEST_F_NOTIMPLEMENT_INST1(Trunc, Trunc)

TEST_F_NOTIMPLEMENT_INST2(Atan2, Atan2)
TEST_F_NOTIMPLEMENT_INST2(BitwiseOr, BitwiseOr)
TEST_F_NOTIMPLEMENT_INST2(BitwiseXor, BitwiseXor)
TEST_F_NOTIMPLEMENT_INST2(NextAfter, NextAfter)
TEST_F_NOTIMPLEMENT_INST2(TruncDiv, TruncateDiv)

TEST_F_NOTIMPLEMENT_INST3(Fma, FusedMulAdd)
TEST_F_NOTIMPLEMENT_INST3(Masked, MaskedFill)

TEST_F_LOWER_REDUCE_ASCBACKEND_INST1(Sum, ReduceSumD)
TEST_F_LOWER_REDUCE_ASCBACKEND_INST1(Max, ReduceMaxD)
TEST_F_LOWER_REDUCE_ASCBACKEND_INST1(Mean, ReduceMeanD)
TEST_F_LOWER_REDUCE_ASCBACKEND_INST1(Min, ReduceMinD)
TEST_F_LOWER_REDUCE_ASCBACKEND_INST1(Prod, ReduceProdD)
}  // namespace ge
