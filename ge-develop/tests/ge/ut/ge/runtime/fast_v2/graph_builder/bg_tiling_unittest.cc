/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_builder/bg_tiling.h"
#include "graph_builder/bg_platform.h"
#include "graph_builder/bg_core_type.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "register/node_converter_registry.h"
#include "exe_graph_comparer.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/share_graph.h"
#include "engine/aicore/fe_rt2_common.h"
#include "common/topo_checker.h"
#include "common/summary_checker.h"
#include "macro_utils/dt_public_scope.h"
#include "register/op_impl_registry.h"
#include "macro_utils/dt_public_unscope.h"
#include "register/op_tiling_registry.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/const_data_helper.h"
#include "kernel/common_kernel_impl/tiling.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/ge_local_context.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace gert {
namespace bg {
namespace {
using namespace ge;
void SetTensorDesc(ge::GeTensorDesc &desc) {
  desc.SetShape(GeShape({8, 3, 224, 224}));
  desc.SetOriginShape(GeShape({8, 3, 224, 224}));
  desc.SetDataType(DT_FLOAT);
  desc.SetOriginDataType(DT_FLOAT);
  desc.SetFormat(ge::FORMAT_NCHW);
  desc.SetOriginFormat(ge::FORMAT_NCHW);
}
ComputeGraphPtr BuildGraphWithManyInputs(const std::string &node_type, const size_t input_num) {
  DEF_GRAPH(g1) {
    // 第一个 Data -> Add -> NetOutput
    CHAIN(NODE("data0", "Data")
          ->NODE(node_type, node_type)
          ->NODE("NetOutput", "NetOutput"));

    // data1 ~ data64 全部连到同一个 Add
    for (size_t i = 1; i < input_num; ++i) {
      CHAIN(NODE(("data" + std::to_string(i)).c_str(), "Data")
            ->EDGE(0, i)
            ->NODE(node_type));
    }
  };

  auto graph = ToComputeGraph(g1);

  for (size_t i = 0; i < input_num; ++i) {
    auto data = graph->FindNode("data" + std::to_string(i));
    GE_ASSERT_NOTNULL(data);

    auto &out_desc = *data->GetOpDesc()->MutableOutputDesc(0);
    SetTensorDesc(out_desc);

    AttrUtils::SetInt(data->GetOpDesc(), "index", i);
  }

  auto add = graph->FindNode(node_type);
  GE_ASSERT_NOTNULL(add);

  // Add 输出
  SetTensorDesc(*add->GetOpDesc()->MutableOutputDesc(0));

  // Add 的 65 个输入
  for (size_t i = 0; i < input_num; ++i) {
    SetTensorDesc(*add->GetOpDesc()->MutableInputDesc(i));
  }

  return graph;
}

void AddCompiledJson(const ge::NodePtr &node) {
  constexpr char const *kStubJson =
      "{\"vars\": {\"srcFormat\": \"NCHW\", \"dstFormat\": \"NC1HWC0\", \"dType\": \"float16\", \"ub_size\": 126464, "
      "\"block_dim\": 32, \"input_size\": 0, \"hidden_size\": 0, \"group\": 1}}";
  AttrUtils::SetStr(node->GetOpDesc(), "compile_info_json", kStubJson);
  AttrUtils::SetInt(node->GetOpDesc(), "op_para_size", 2048);
}
}  // namespace

class BgTilingUT : public BgTestAutoCreateFrame {
 public:
  void TilingTopoCorrect(ge::ExecuteGraph *exe_graph, const std::vector<ValueHolderPtr> &tiling_rets,
                         const std::vector<ValueHolderPtr> &io_shapes, const ValueHolderPtr &platform) {
    for (const auto &tiling_ret : tiling_rets) {
      ASSERT_NE(tiling_ret, nullptr);
    }
    std::vector<FastSrcNode> expect_from;
    for (const auto &io_shape : io_shapes) {
      expect_from.emplace_back(io_shape);
    }
    expect_from.emplace_back("TilingParse");
    expect_from.emplace_back(platform);
    // UT中未执行CEM，因此PrepareTilingFwkData还在main图上，需要将校验对象InnerData替换为PrepareTilingFwkData
    expect_from.emplace_back("PrepareTilingFwkData");
    expect_from.emplace_back("InnerData");
    expect_from.emplace_back("InnerData");
    ASSERT_EQ(FastNodeTopoChecker(tiling_rets[0]).StrictConnectFrom(expect_from), "success");
    auto tiling_parse_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "TilingParse");
    ASSERT_NE(tiling_parse_node, nullptr);
    ASSERT_EQ(FastNodeTopoChecker(tiling_parse_node).StrictConnectFrom({{"Const"}, {"Data"}, {"Const"}, {"Const"}}),
              "success");
    auto find_tiling_func_node =
        ge::ExecuteGraphUtils::FindFirstNodeMatchType(init_frame_->GetExecuteGraph().get(), "FindTilingFunc");
    ASSERT_NE(find_tiling_func_node, nullptr);
    ASSERT_EQ(FastNodeTopoChecker(find_tiling_func_node).StrictConnectFrom({{"Const"}, {"GetSpaceRegistry"}}),
              "success");
    auto tiling_fwk_data_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "PrepareTilingFwkData");
    // Check "FindTilingFunc" in init graph link to "PrepareTilingFwkData" in main graph (ut only)
    ConnectFromInitToMain(find_tiling_func_node, 0, tiling_fwk_data_node, 0);
  }

 protected:
  void SetUp() override {
    ValueHolder::PushGraphFrame();
    auto init = ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    auto main = ValueHolder::CreateVoid<bg::ValueHolder>("Main", {});
    auto de_init = ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});

    ValueHolder::PushGraphFrame(init, "Init");
    init_frame_ = ValueHolder::PopGraphFrame({}, {});

    ValueHolder::PushGraphFrame(de_init, "DeInit");
    de_init_frame_ = ValueHolder::PopGraphFrame();

    ValueHolder::PushGraphFrame(main, "Main");
    auto launch_arg_output =
        bg::ValueHolder::CreateDataOutput("AllocLaunchArg", {}, static_cast<size_t>(AllocLaunchArgOutputs::kNum));
    fake_launch_arg_ = launch_arg_output[static_cast<size_t>(AllocLaunchArgOutputs::kRtArg)];
  }

  void TearDown() override {
    BgTest::TearDown();
    init_frame_.reset();
    de_init_frame_.reset();
  }

  std::unique_ptr<GraphFrame> init_frame_;
  std::unique_ptr<GraphFrame> de_init_frame_;
  ValueHolderPtr fake_launch_arg_;
};


TEST_F(BgTilingUT, BgTiling_Ok_CachableTilingUnsupported) {
  // FakeAddN 由于没有 IR，因此在检查 IsDataDependent 时预期会失败
  IMPL_OP(FakeAddN).InputsDataDependency({0});
  SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2(true);

  constexpr int64_t kInputNum = 2UL;
  std::vector<ValueHolderPtr> input_shapes;
  for (int64_t i = 0; i < kInputNum; ++i) {
    input_shapes.emplace_back(ValueHolder::CreateFeed(i));
  }
  auto out_shape = ValueHolder::CreateFeed(2);
  auto platform = ValueHolder::CreateFeed(3);
  auto graph = BuildGraphWithManyInputs("FakeAddN", kInputNum);
  ASSERT_NE(graph, nullptr);
  auto add_node = graph->FindNode("FakeAddN");
  ASSERT_NE(add_node, nullptr);
  AddCompiledJson(add_node);

  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("FakeAddN", false).Build();
  bg::LowerConstDataNode(global_data);
  auto tiling_rets = Tiling(add_node, input_shapes, {out_shape}, {platform, global_data, fake_launch_arg_});
  auto frame = ValueHolder::PopGraphFrame();
  ASSERT_NE(frame, nullptr);
  auto exe_graph = frame->GetExecuteGraph().get();
  ASSERT_NE(exe_graph, nullptr);
  ge::DumpGraph(exe_graph, "TilingUT");

  // check main 图中节点数量
  ASSERT_EQ(ExeGraphSummaryChecker(exe_graph).StrictAllNodeTypes({{"Data", 6},
                                                                  {"Tiling", 1},
                                                                  {"TilingAppendDfxInfo", 1},
                                                                  {"TilingParse", 1},
                                                                  {"InnerData", 3},
                                                                  {"SplitRtStreams", 1},
                                                                  {"Const", 10},
                                                                  {"CalcTensorSizeFromStorage", 1},
                                                                  {"PrepareTilingFwkData", 1},
                                                                  {"AllocLaunchArg", 1}}),
            "success");
  ge::DumpGraph(init_frame_->GetExecuteGraph().get(), "TilingUTInit");

  // check init 图中节点数量
  ASSERT_EQ(ExeGraphSummaryChecker(init_frame_->GetExecuteGraph().get())
                .StrictAllNodeTypes({
                    {"InnerNetOutput", 1},
                    {"FindTilingFunc", 1},
                    {"ConstData", 3},
                    {"Data", 1},
                    {"Const", 6},
                    {"SplitRtStreams", 1},
                    {"GetSpaceRegistry", 1},
                }),
            "success");
  // check topo连接关系，包括子图内部以及子图之间
  std::vector<ValueHolderPtr> io_shapes(input_shapes.begin(), input_shapes.end());
  io_shapes.emplace_back(out_shape);
  TilingTopoCorrect(exe_graph, tiling_rets, io_shapes, platform);
  ASSERT_EQ(tiling_rets.size(), static_cast<size_t>(kernel::TilingExOutputIndex::kNum));
}


}  // namespace bg
}  // namespace gert
