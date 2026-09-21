
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
#include "utils/auto_fuse_config.h"
#include "compliant_op_desc_builder.h"
#include "graph/attribute_group/attr_group_shape_env.h"
#include "post_process/scheduler_adapter/torch_adaption_fallback_load.h"
#include "expression/testcase/source_stub.h"
#include "depends/runtime/src/runtime_stub.h"

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

class LoopLoweringToAscBackendUTV2 : public testing::Test {
public:
protected:
  void SetUp() override {
    dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
    ge::PlatformContext::GetInstance().Reset();
    auto stub_v2 = std::make_shared<RuntimeStubV2Common>();
    RuntimeStub::SetInstance(stub_v2);
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

TEST_F(LoopLoweringToAscBackendUTV2, TestSplitDLoweringDynamicSuccess) {
  dlog_setlevel(0, 0, 1);
  setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1" , 1);
  [this]() {
    auto data0 = es_graph_->CreateInput(0, "data0", nullptr);
    data0.SetSymbolShape({"o0", "(3 * o1)", "o2"});
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
    ge::AnyValue::CreateFrom(static_cast<int64_t>(1))
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
  split_out0->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
  split_out1->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
  split_out2->SetSymbolShape({Symbol("o0"), Symbol("o1"), Symbol("o2")});
  es_graph_->SetOutput(split_out0, 0);
  es_graph_->SetOutput(split_out1, 1);
  es_graph_->SetOutput(split_out2, 2);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto split = cg->FindNode("SplitD");
  ASSERT_NE(split, nullptr);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  std::string expected_res = R"(AscGraph(graph, axis=[0:o0, 1:(3 * o1), 2:o2])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [o0, (3 * o1), o2], strides = [(3 * o1 * o2), o2, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [o0, (3 * o1), o2], strides = [(3 * o1 * o2), o2, 1]}
tmp2 = ascir.Split(Split_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [o0, o1, o2], strides = [(o1 * o2), o2, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)";
  size_t idx = 0U;
  for (const auto &node_out : split->GetAllOutDataAnchors()) {
    loop::KernelBox asc_kernel = ge::loop::GetKernelBox(node_out);
    auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
    ASSERT_NE(asc_graph, nullptr);
    const auto res = ReadableAscGraph(asc_graph->Graph());
    EXPECT_EQ(res, expected_res);
  }
}

TEST_F(LoopLoweringToAscBackendUTV2, TestSplitDLoweringStaticSuccess) {
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
  split_out0->SetSymbolShape({Symbol(192), Symbol(64), Symbol(10)});
  split_out1->SetSymbolShape({Symbol(192), Symbol(64), Symbol(10)});
  split_out2->SetSymbolShape({Symbol(192), Symbol(64), Symbol(10)});
  es_graph_->SetOutput(split_out0, 0);
  es_graph_->SetOutput(split_out1, 1);
  es_graph_->SetOutput(split_out2, 2);

  ASSERT_EQ(LoweringManager::LoweringGraph(cg), GRAPH_SUCCESS);
  auto split = cg->FindNode("SplitD");
  ASSERT_NE(split, nullptr);
  ASSERT_EQ(LoweringManager::FusedLoopToAscBackendOp(cg), GRAPH_SUCCESS);
  auto range_out = split->GetAllOutDataAnchors();
  std::vector<OutDataAnchorPtr> node_outputs;
  for (const auto e: range_out) {
    node_outputs.emplace_back(e);
  }
  loop::KernelBox asc_kernel = ge::loop::GetKernelBox(node_outputs[0]);
  std::string expected = R"(AscGraph(graph, axis=[0:192, 1:64, 2:30])
tmp0 = ascir.Data(Data_0, [])
tmp0.attr = {axis = [0, 1, 2], repeats = [192, 64, 30], strides = [1920, 30, 1]}
tmp1 = ascir.Load(Load_0, [tmp0])
tmp1.attr = {axis = [0, 1, 2], repeats = [192, 64, 30], strides = [1920, 30, 1]}
tmp2 = ascir.Split(Split_0, [tmp1])
tmp2.attr = {axis = [0, 1, 2], repeats = [192, 64, 10], strides = [640, 10, 1]}
tmp3 = ascir.Store(Store_0, [tmp2])
tmp3.attr = {axis = [0, 1, 2], repeats = [192, 64, 10], strides = [640, 10, 1]}
tmp4 = ascir.Output(Output_0, [tmp3])
)";
  auto asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  const auto res0 = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res0, expected);
  asc_kernel = ge::loop::GetKernelBox(node_outputs[1]);
  asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  const auto res1 = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res1, expected);
  asc_kernel = ge::loop::GetKernelBox(node_outputs[2]);
  asc_graph = asc_kernel.Realize<loop::AscOverrides>("graph");
  ASSERT_NE(asc_graph, nullptr);
  const auto res2 = ReadableAscGraph(asc_graph->Graph());
  EXPECT_EQ(res2, expected);
}


}  // namespace ge
