
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
#include "op_creator_register.h"
#include "all_ops_cpp.h"
#include "esb_graph.h"
#include "lowering/asc_lowerer/loop_api.h"
#include "lowering/asc_lowerer/asc_overrides.h"
#include "lowering/lowerings.h"
#include "lowering/op_lowering_impl/lowering_impl.h"
#include "utils/autofuse_attrs.h"
#include "utils/auto_fuse_config.h"
#include "compliant_op_desc_builder.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "common/util/mem_utils.h"
#include "post_process/scheduler_adapter/torch_adaption_fallback_load.h"
#include "expression/testcase/source_stub.h"
#include "depends/runtime/src/runtime_stub.h"

#include "backend/backend_spec.h"
#include "platform_context.h"

#include <regex>
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
  void TearDown() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  }
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
    auto stub_v1 = std::make_shared<RuntimeStub>();
    RuntimeStub::SetInstance(stub_v1);
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
  auto log_add_exp = cg->FindNode("Exp_2");
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
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

TEST_F(LoopLoweringToAscBackendUT, SimpleLogAddExpNoCse) {
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

TEST_F(LoopLoweringToAscBackendUT, FusePointwiseWithEndReductionKeepDim) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    auto abs = es::Abs(data0);
    abs.SetSymbolShape({"s0", "s1", "s2"});
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
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kReduction);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(ReduceSumD_1_out0_graph, axis=[0:s0, 1:s1, 2:s2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Sum(Sum_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp4 = ascir.Store(Store_0, [tmp3])
tmp4.attr = {axis = [0, 1, 2], repeats = [s0, 1, s2], strides = [s2, 0, 1]}
tmp5 = ascir.Output(Output_0, [tmp4])
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

  auto sum = cg->FindNode("ReduceSumD_1");
  ASSERT_NE(sum, nullptr);

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

TEST_F(LoopLoweringToAscBackendUT, LoweringIndicesScalarGatherV2) {
  [this]() {
  auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
  data0.SetSymbolShape({"s0", "s1", "s2"});
  auto data1 = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
  data1.SetSymbolShape({});
  auto axis = CreateConst(*es_graph_, ge::DT_INT64, {1}, std::vector<int64_t>{-1});
  axis.SetSymbolShape({"0"});
  auto gather = es::GatherV2(data0, data1, axis);
  gather.SetSymbolShape({"s0", "s1"});
  es_graph_->SetOutput(gather, 0);
  }();
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto mm = cg->FindNode("GatherV2_2");
  ASSERT_NE(mm, nullptr);
  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, FallBackTransposeOK) {
  [this]() {
    auto data = es_graph_->CreateInput(0, "data0", nullptr);
    data.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose = es::Transpose(data, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(transpose, 0);
  }();
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting());
  SetCurShapeEnvContext(&shape_env);
  auto s1 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s2 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s3 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  NodePtr asc_node = nullptr;
  for (const auto &node : cg->GetAllNodes()) {
    if(node->GetType() == "AscBackend") {
      asc_node = node;
    }
  }
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  // 获取实际生成的图描述字符串
  std::string actual = ReadableAscGraph(*fused_attrs->GetAscGraph());
  // 使用正则表达式替换Transpose节点中动态生成的编号为固定值（11）
  cout << actual << endl;
  std::regex load_transpose_regex(R"(Load_(\d+)_(\d+))");
  // 替换为 Load_X_11
  std::string sanitized_actual = std::regex_replace(actual, load_transpose_regex, "Load_$1_11");
  EXPECT_EQ(sanitized_actual,
    "AscGraph(Transpose_1_out0_graph, axis=[0:s2, 1:s1, 2:s0])\n"
    "tmp0 = ascir.Data(Data_0, [])\n"
    "tmp0.attr = {axis = [2, 1, 0], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n"
    "tmp1 = ascir.Load(Load_0, [tmp0])\n"
    "tmp1.attr = {axis = [2, 1, 0], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}\n"
    "tmp2 = ascir.Transpose(Load_0_11, [tmp1])\n"
    "tmp2.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}\n"
    "tmp3 = ascir.Store(Store_0, [tmp2])\n"
    "tmp3.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}\n"
    "tmp4 = ascir.Output(Output_0, [tmp3])\n");
  SetCurShapeEnvContext(nullptr);
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
}

TEST_F(LoopLoweringToAscBackendUT, FallBackTransposeOK1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"s0", "s1", "s2"});
    data1.SetSymbolShape({"s0", "s1", "s2"});
    auto add = es::Add(data0, data1);
    add.SetSymbolShape({"s0", "s1", "s2"});
    auto perms = CreateConst(*es_graph_, ge::DT_INT64, {3}, std::vector<int64_t>{2, 1, 0});
    auto transpose = es::Transpose(add, perms);
    transpose.SetSymbolShape({"s2", "s1", "s0"});
    es_graph_->SetOutput(transpose, 0);
  }();
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = true;
  auto shape_env = ShapeEnvAttr(ShapeEnvSetting());
  SetCurShapeEnvContext(&shape_env);
  auto s1 = shape_env.CreateSymbol(2, MakeShared<GraphInputShapeSourceStub>(0, 0));
  auto s2 = shape_env.CreateSymbol(3, MakeShared<GraphInputShapeSourceStub>(0, 1));
  auto s3 = shape_env.CreateSymbol(4, MakeShared<GraphInputShapeSourceStub>(0, 2));
  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  NodePtr asc_node = nullptr;
  for (const auto &node : cg->GetAllNodes()) {
    if(node->GetType() == "AscBackend") {
      asc_node = node;
    }
  }
  ASSERT_NE(asc_node, nullptr);
  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  ASSERT_EQ(asc_adapt::GeFallback(cg), GRAPH_SUCCESS);
  // 获取实际生成的图描述字符串
  std::string actual = ReadableAscGraph(*fused_attrs->GetAscGraph());
  // 使用正则表达式替换Transpose节点中动态生成的编号为固定值（11）
  std::regex load_transpose_regex(R"(Load_(\d+)_(\d+))");
  // 替换为 Load_X_11
  std::string sanitized_actual = std::regex_replace(actual, load_transpose_regex, "Load_$1_11");
  cout << sanitized_actual;
  EXPECT_EQ(sanitized_actual, R"(AscGraph(Transpose_2_out0_graph, axis=[0:s2, 1:s1, 2:s0])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [2, 1, 0], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [2, 1, 0], repeats = [s0, s1, s2], strides = [(s1 * s2), s2, 1]}
tmp2 = ascir.Transpose(Load_0_11, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [s2, s1, s0], strides = [(s0 * s1), s0, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)");
  SetCurShapeEnvContext(nullptr);
  AutoFuseConfig::MutableConfig().MutableLoweringConfig().experimental_lowering_transpose = false;
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

TEST_F(LoopLoweringToAscBackendUT, LoweringMatmul) {
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
  tmp_desc0->SetShape(GeShape({4, 5}));
  tmp_desc0->SetOriginShape(GeShape({4, 5}));
  auto data1 = cg->FindNode("data1");
  auto tmp_desc1 = data1->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc1->SetShape(GeShape({5, 6}));
  tmp_desc1->SetOriginShape(GeShape({5, 6}));

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kCube);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(MatMul_0_out0_graph, axis=[0:4, 1:6])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [4, 5], strides = [5, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [4, 5], strides = [5, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [5, 6], strides = [6, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [5, 6], strides = [6, 1]}
tmp4 = ascir.MatMul(MatMul_0, [tmp1, tmp3])
tmp4.attr = {axis = [0, 1], repeats = [4, 6], strides = [6, 1]}
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [4, 6], strides = [6, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringMatmulBias) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto bias = es_graph_->CreateInput(2, "bias", nullptr);
    data0.SetSymbolShape({"4", "4"});
    data1.SetSymbolShape({"4", "4"});
    bias.SetSymbolShape({"4", "4"});
    auto mm = es::MatMul(data0, data1, bias);
    mm.SetSymbolShape({"4", "4"});
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

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kCube);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(MatMul_0_out0_graph, axis=[0:4, 1:4])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp6 = ascir.MatMulBias(MatMulBias_0, [tmp1, tmp3, tmp5])
tmp6.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp7 = ascir.Store(Store_0, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp8 = ascir.Output(Output_0, [tmp7])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringMatmulOffset) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto offset_w = es_graph_->CreateInput(2, "offset_w", nullptr);
    data0.SetSymbolShape({"4", "4"});
    data1.SetSymbolShape({"4", "4"});
    offset_w.SetSymbolShape({"4", "4"});
    auto mm = es::MatMulV2(data0, data1, nullptr, offset_w);
    mm.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(mm, 0);
}();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mm = cg->FindNode("MatMulV2_0");
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
  auto data3 = cg->FindNode("offset_w");
  auto tmp_desc3 = data3->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kCube);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(MatMulV2_0_out0_graph, axis=[0:4, 1:4])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp6 = ascir.MatMulOffset(MatMulOffset_0, [tmp1, tmp3, tmp5])
tmp6.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp7 = ascir.Store(Store_0, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp8 = ascir.Output(Output_0, [tmp7])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringMatmulBiasOffset) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto bias = es_graph_->CreateInput(2, "bias", nullptr);
    auto offset_w = es_graph_->CreateInput(3, "offset_w", nullptr);
    data0.SetSymbolShape({"4", "4"});
    data1.SetSymbolShape({"4", "4"});
    bias.SetSymbolShape({"4", "4"});
    offset_w.SetSymbolShape({"4", "4"});
    auto mm = es::MatMulV2(data0, data1, bias, offset_w);
    mm.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mm = cg->FindNode("MatMulV2_0");
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

  auto data2 = cg->FindNode("bias");
  auto tmp_desc2 = data2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_FLOAT);
  tmp_desc2->SetOriginDataType(DT_FLOAT);
  auto data3 = cg->FindNode("offset_w");
  auto tmp_desc3 = data3->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kCube);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(MatMulV2_0_out0_graph, axis=[0:4, 1:4])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp6 = ascir.Data(Data_3, [])
tmp6.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp7 = ascir.Load(Load_3, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp8 = ascir.MatMulOffsetBias(MatMulOffsetBias_0, [tmp1, tmp3, tmp5, tmp7])
tmp8.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringBatchMatmulBiasOffset) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    auto bias = es_graph_->CreateInput(2, "bias", nullptr);
    auto offset_w = es_graph_->CreateInput(3, "offset_w", nullptr);
    data0.SetSymbolShape({"4", "4"});
    data1.SetSymbolShape({"4", "4"});
    bias.SetSymbolShape({"4", "4"});
    offset_w.SetSymbolShape({"4", "4"});
    auto mm = es::BatchMatMulV2(data0, data1, bias, offset_w);
    mm.SetSymbolShape({"4", "4"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);

  auto mm = cg->FindNode("BatchMatMulV2_0");
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

  auto data2 = cg->FindNode("bias");
  auto tmp_desc2 = data2->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc2->SetDataType(DT_FLOAT);
  tmp_desc2->SetOriginDataType(DT_FLOAT);
  auto data3 = cg->FindNode("offset_w");
  auto tmp_desc3 = data3->GetOpDesc()->MutableOutputDesc(0);
  tmp_desc3->SetDataType(DT_INT8);
  tmp_desc3->SetOriginDataType(DT_INT8);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kCube);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(BatchMatMulV2_0_out0_graph, axis=[0:4, 1:4])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp4 = ascir.Data(Data_2, [])
tmp4.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp5 = ascir.Load(Load_2, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp6 = ascir.Data(Data_3, [])
tmp6.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp7 = ascir.Load(Load_3, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp8 = ascir.BatchMatMulOffsetBias(BatchMatMulOffsetBias_0, [tmp1, tmp3, tmp5, tmp7])
tmp8.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp9 = ascir.Store(Store_0, [tmp8])
tmp9.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp10 = ascir.Output(Output_0, [tmp9])
)");
}

TEST_F(LoopLoweringToAscBackendUT, LoweringMatmulAxisNotAlign) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data0.SetSymbolShape({"4"});
    data1.SetSymbolShape({"4", "4"});
    auto mm = es::MatMul(data0, data1);
    mm.SetSymbolShape({"4", "4"});
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

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);

  auto asc_node = cg->FindFirstNodeMatchType("AscBackend");
  ASSERT_NE(asc_node, nullptr);

  auto fused_attrs = asc_node->GetOpDesc()->GetAttrsGroup<AutoFuseAttrs>();
  ASSERT_NE(fused_attrs, nullptr);
  ASSERT_NE(fused_attrs->GetAscGraph(), nullptr);
  EXPECT_EQ(fused_attrs->GetFuseType(), loop::FuseType::kCube);

  EXPECT_EQ(ReadableAscGraph(*fused_attrs->GetAscGraph()), R"(AscGraph(MatMul_0_out0_graph, axis=[0:4, 1:4])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [1, 4], strides = [0, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [1, 4], strides = [0, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp4 = ascir.MatMul(MatMul_0, [tmp1, tmp3])
tmp4.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [4, 4], strides = [4, 1]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeMMV2) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"m", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "1"});
    auto mm = es::MatMulV2(data0, data1);
    mm.SetSymbolShape({"m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(asc_kernel.Readable(), R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.Permute(tmp1, "[d0, d1]->[d1, d0]")
tmp3 = ops.Broadcast(tmp2, "[1, d1]->[d0, d1]")
tmp4 = ops.Mul(tmp0, tmp3)
tmp5 = ops.StoreReduction("MatMulV2_0:0", ops.Sum(tmp4, "[d0, d1]->[d0, 1]"))
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:m, 1:128])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [1, 128], strides = [0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Sum(Sum_0, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeMMV2TransposeB) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"m", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "128"});
    auto mm = es::MatMulV2(data0, data1, nullptr, nullptr, false, true);
    mm.SetSymbolShape({"m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(asc_kernel.Readable(), R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.Broadcast(tmp1, "[1, d1]->[d0, d1]")
tmp3 = ops.Mul(tmp0, tmp2)
tmp4 = ops.StoreReduction("MatMulV2_0:0", ops.Sum(tmp3, "[d0, d1]->[d0, 1]"))
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:m, 1:128])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [1, 128], strides = [0, 1]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Sum(Sum_0, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp6 = ascir.Store(Store_0, [tmp5])
tmp6.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp7 = ascir.Output(Output_0, [tmp6])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeMMV2K1) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"m", "1"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1", "1"});
    auto mm = es::MatMulV2(data0, data1, nullptr, nullptr, false, true);
    mm.SetSymbolShape({"m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_TRUE(!asc_kernel.IsReduction());
  ASSERT_TRUE(asc_kernel.IsRealized());
  ASSERT_EQ(asc_kernel.Readable(), R"(tmp0 = ops.Load("data0:0")
tmp1 = ops.Load("data1:0")
tmp2 = ops.Broadcast(tmp1, "[1, d1]->[d0, d1]")
tmp3 = ops.Mul(tmp0, tmp2)
tmp4 = ops.Store("MatMulV2_0:0", tmp3)
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:m, 1:1])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp2 = ascir.Data(Data_1, [])
tmp2.attr = {axis = [0, 1], repeats = [1, 1], strides = [0, 0]}
tmp3 = ascir.Load(Load_1, [tmp2])
tmp3.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp4 = ascir.Mul(Mul_0, [tmp1, tmp3])
tmp5 = ascir.Store(Store_0, [tmp4])
tmp5.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp6 = ascir.Output(Output_0, [tmp5])
)");
}

TEST_F(LoopLoweringToAscBackendUT, VectorizeMMV2AndFuseWithBefore) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"m", "128"});
    auto abs1 = es::Abs(data0);
    abs1.SetSymbolShape({"m", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "1"});
    auto abs2 = es::Abs(data1);
    abs2.SetSymbolShape({"128", "1"});
    auto mm = es::MatMulV2(abs1, abs2);
    mm.SetSymbolShape({"m", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_2");
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
tmp6 = ops.Permute(tmp5, "[d0, d1]->[d1, d0]")
tmp7 = ops.Broadcast(tmp6, "[1, d1]->[d0, d1]")
tmp8 = ops.Mul(tmp2, tmp7)
tmp9 = ops.StoreReduction("MatMulV2_2:0", ops.Sum(tmp8, "[d0, d1]->[d0, 1]"))
)");

  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  EXPECT_EQ(ReadableAscGraph(asc_graph->Graph()), R"(AscGraph(graph, axis=[0:m, 1:128])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp2 = ascir.Abs(Abs_0, [tmp1])
tmp3 = ascir.Data(Data_1, [])
tmp3.attr = {axis = [0, 1], repeats = [1, 128], strides = [0, 1]}
tmp4 = ascir.Load(Load_1, [tmp3])
tmp4.attr = {axis = [0, 1], repeats = [m, 128], strides = [128, 1]}
tmp5 = ascir.Abs(Abs_1, [tmp4])
tmp6 = ascir.Mul(Mul_0, [tmp2, tmp5])
tmp7 = ascir.Sum(Sum_0, [tmp6])
tmp7.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp8 = ascir.Store(Store_0, [tmp7])
tmp8.attr = {axis = [0, 1], repeats = [m, 1], strides = [1, 0]}
tmp9 = ascir.Output(Output_0, [tmp8])
)");
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeMMV2AsSymbolicK) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "4"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"4", "1"});
    auto mm = es::MatMulV2(data0, data1);
    mm.SetSymbolShape({"4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeMMV2AsOversizeK) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "1024"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"1024", "1"});
    auto mm = es::MatMulV2(data0, data1);
    mm.SetSymbolShape({"4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeMMV2AsTransposeA) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "1"});
    auto mm = es::MatMulV2(data0, data1, nullptr, nullptr, true, false);
    mm.SetSymbolShape({"4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeMMV2AsWithBias) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "1"});
    auto mm = es::MatMulV2(data0, data1, data0);
    mm.SetSymbolShape({"4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeMMV2AsWithOffsetW) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "1"});
    auto mm = es::MatMulV2(data0, data1, nullptr, data0);
    mm.SetSymbolShape({"4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
}

TEST_F(LoopLoweringToAscBackendUT, SkipVectorizeMMV2AsWithOffset) {
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"4", "128"});
    auto data1 = es_graph_->CreateInput(1, "data1", nullptr);
    data1.SetSymbolShape({"128", "1"});
    auto mm = es::MatMulV2(data0, data1, nullptr, nullptr, false, false, 2);
    mm.SetSymbolShape({"4", "1"});
    es_graph_->SetOutput(mm, 0);
  }();

  auto graph = es_graph_->Build();
  auto cg = GraphUtilsEx::GetComputeGraph(*graph);
  auto mm = cg->FindNode("MatMulV2_0");

  auto origin = AutoFuseConfig::LoweringConfig().max_k_for_vectorize_mm;
  GE_MAKE_GUARD(config, [origin]() { AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = origin; });
  AutoFuseConfig::MutableLoweringConfig().max_k_for_vectorize_mm = 256;
  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(mm->GetOutDataAnchor(0));
  ASSERT_FALSE(asc_kernel.IsExternKernel());
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
}  // namespace ge
