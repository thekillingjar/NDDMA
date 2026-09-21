/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define private public
#include "base/registry/op_impl_space_registry_v2.h"
#undef private
#include "engine/aicore/converter/aclnn_node_converter.h"
#include "engine/node_converter_utils.h"
#include <gtest/gtest.h>
#include "engine/aicore/fe_rt2_common.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_builder/value_holder_generator.h"
#include "lowering/placement/placed_lowering_result.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "register/op_impl_registry.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "aicore/converter/bg_kernel_launch.h"
#include "ge/ut/ge/runtime/fast_v2/common/const_data_helper.h"
#include "runtime/subscriber/global_dumper.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
using namespace ge;
using namespace gert::bg;

namespace gert {
extern ge::Status AddDataNodeForAtomic(ge::ComputeGraphPtr &graph, ge::NodePtr &clean_node, size_t output_size);
extern ge::NodePtr BuildAtomicNode(const ge::NodePtr &origin_node,
                        const gert::bg::AtomicLoweringArg &atomic_lowering_arg,
                        std::vector<gert::bg::ValueHolderPtr> &output_clean_sizes,
                        std::vector<gert::bg::ValueHolderPtr> &output_clean_addrs,
                        ComputeGraphPtr &graph);

namespace {
ge::graphStatus InferShapeStub(InferShapeContext *context) { return SUCCESS;}
IMPL_OP(Conv2d).InferShape(InferShapeStub);
IMPL_OP(Relu).InferShape(InferShapeStub);
UINT32 OpExecuteFuncStub(gert::OpExecuteContext *tiling_context) {
  return 1;
}

UINT32 OpExecutePrepareStub(gert::OpExecutePrepareContext *op_exe_prepare_context) {
  return 1;
}

UINT32 OpExecuteLaunchStub(gert::OpExecuteLaunchContext *op_exe_launch_context) {
  return 1;
}
} // namespace

class AclnnNodeConverterST : public bg::BgTestAutoCreate3StageFrame {
 protected:
  void SetUp() override {
    //BgTest::SetUp();
    BgTestAutoCreate3StageFrame::SetUp();
  }
  void TearDown() override {
    BgTestAutoCreate3StageFrame::TearDown();
  }
};

TEST_F(AclnnNodeConverterST, TestHgl) {
  auto graph = ShareGraph::AicoreGraph();
  auto add_node = graph->FindNode("add1");
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};

  graph->FindNode("data1")->GetOpDesc()->SetExtAttr("_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr("_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));
  auto data_guarder = bg::DevMemValueHolder::CreateSingleDataOutput("FreeAdd", {add_input.input_addrs[1]}, 0);
  add_input.input_addrs[0]->SetGuarder(data_guarder);


  gert::GlobalDumper::GetInstance()->SetEnableFlags(
  gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));

  gert::OpImplSpaceRegistryV2Ptr space_registry_stub = std::make_shared<gert::OpImplSpaceRegistryV2>();
  auto op_impl_func = space_registry_stub->CreateOrGetOpImpl("Add");
  op_impl_func->op_execute_func = OpExecuteFuncStub;

  auto space_registry_array = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  space_registry_array->at(static_cast<size_t>(ge::OppImplVersion::kOpp)) = space_registry_stub;
  global_data.SetSpaceRegistriesV2(*space_registry_array);
  auto add_ret = LoweringAclnnNode(add_node, add_input);
}

TEST_F(AclnnNodeConverterST, TestHgl2Stages) {
  auto graph = ShareGraph::AicoreGraph();
  auto add_node = graph->FindNode("add1");
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);

  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};

  graph->FindNode("data1")->GetOpDesc()->SetExtAttr("_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data1"), std::move(data1_ret)));
  graph->FindNode("data2")->GetOpDesc()->SetExtAttr("_lowering_result", gert::PlacedLoweringResult(graph->FindNode("data2"), std::move(data2_ret)));
  auto data_guarder = bg::DevMemValueHolder::CreateSingleDataOutput("FreeAdd", {add_input.input_addrs[1]}, 0);
  add_input.input_addrs[0]->SetGuarder(data_guarder);


  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));

  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_jsoncompile_info_json";
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), optiling::COMPILE_INFO_JSON, compile_info_json);

  auto space_registry_stub = std::make_shared<gert::OpImplSpaceRegistryV2>();
  auto op_impl_func = space_registry_stub->CreateOrGetOpImpl("Add");
  op_impl_func->op_execute_func = nullptr;
  op_impl_func->op_execute_prepare_func = OpExecutePrepareStub;
  op_impl_func->op_execute_launch_func = OpExecuteLaunchStub;
  auto space_registry_array = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  space_registry_array->at(static_cast<size_t>(ge::OppImplVersion::kOpp)) = space_registry_stub;
  global_data.SetSpaceRegistriesV2(*space_registry_array);
  auto add_ret = LoweringAclnnNode(add_node, add_input);
}
}