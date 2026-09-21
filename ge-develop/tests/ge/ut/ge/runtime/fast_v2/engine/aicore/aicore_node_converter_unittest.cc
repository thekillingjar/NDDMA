/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/aicore/converter/aicore_node_converter.h"
#include "engine/node_converter_utils.h"
#include <gtest/gtest.h>
#include "engine/aicore/fe_rt2_common.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph_builder/exe_graph_comparer.h"
#include "graph_builder/value_holder_generator.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "register/op_impl_registry.h"
#include "register/op_tiling_info.h"
#include "aicore/converter/bg_kernel_launch.h"
#include "common/const_data_helper.h"
#include "runtime/subscriber/global_dumper.h"
#include "graph/utils/execute_graph_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/ge_local_context.h"

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

/*
                            g1

┌────────┐  (0,0)   ┌──────────────┐  (0,0)   ┌───────────┐
│ data_a │ ───────> │ conv2d_fused │ ───────> │ netoutput │
└────────┘          └──────────────┘          └───────────┘
                          |
                          |_origin_fuse_graph属性
                          |
                          |
-----------------------------------------------------------------------------
|                            fuse_origin_graph                              |
|                                                                           |
|┌───────┐  (0,0)   ┌────────┐  (0,0)   ┌──────┐  (0,0)   ┌───────────────┐ |
|│ data1 │ ───────> │ conv2d │ ───────> │ relu │ ───────> │ netoutput_sub │ |
|└───────┘          └────────┘          └──────┘          └───────────────┘ |
-----------------------------------------------------------------------------
*/
ge::ComputeGraphPtr BuildGraphWithUBfusion() {
   std::vector<int64_t> shape = {2,2,3,2};  // NCHW
   std::vector<int64_t> unknown_shape = {2,2,-1,2};  // NCHW

  auto data1 = OP_CFG("Data")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Attr("OwnerGraphIsUnknown", true)
        .Build("data1");

  vector<int64_t> test_int64_list_attr = {1,2,3};
  auto conv2d = OP_CFG("Conv2d")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("string_attr", "test")
        .Attr("int32_attr", (int32_t)1)
        .Attr("uint32_attr", (uint32_t)1)
        .Attr("data_format", "NHWC")  // attr on operator
        .Attr("dilations", test_int64_list_attr)
        .Attr("groups", (int32_t)1)
        .Attr("offset_x", (int32_t)1)
        .Build("conv2d");
  conv2d->SetOpEngineName("AIcoreEngine");
  conv2d->SetOpKernelLibName("AIcoreEngine");

  auto relu = OP_CFG("Relu")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Build("relu");
  relu->SetOpEngineName("AIcoreEngine");
  relu->SetOpKernelLibName("AIcoreEngine");

  auto netoutput_sub = OP_CFG("_RetVal")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Build("netoutput_sub");

  DEF_GRAPH(fuse_origin_graph) {
    CHAIN(NODE(data1)->NODE(conv2d)->NODE(relu)->NODE(netoutput_sub));
  };

  auto data_a = OP_CFG("Data")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr(ATTR_NAME_INDEX, 0)
        .Build("data_a");

  auto conv2d_fused = OP_CFG("Conv2d")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(1)
        .OutCnt(1)
        .Attr("data_format", "NHWC")  // attr on operator
        .Attr("dilations", test_int64_list_attr)
        .Attr("groups", (int32_t)1)
        .Attr("offset_x", (int32_t)1)
        .Attr("_original_fusion_graph", fuse_origin_graph)
        .Build("conv2d_fused");
  conv2d_fused->SetOpEngineName("AIcoreEngine");
  conv2d_fused->SetOpKernelLibName("AIcoreEngine");  // fake op can not do that?
  

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_a)->NODE(conv2d_fused)->NODE("netoutput", "NetOutput"));
  };

  auto graph = ToGeGraph(g1);
  auto compute_graph = ge::GraphUtilsEx::GetComputeGraph(graph);
  auto conv2d_fused_node = compute_graph->FindNode("conv2d_fused");
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 100}, {1, 100}, {1, 100}, {1, 100}};
  for (auto &output_tensor : conv2d_fused_node->GetOpDesc()->GetAllOutputsDescPtr()) {
    output_tensor->SetShapeRange(shape_range);
  }
  auto fused_graph = ToGeGraph(fuse_origin_graph);
  auto fused_compute_graph = ge::GraphUtilsEx::GetComputeGraph(fused_graph);
  fused_compute_graph->SetGraphUnknownFlag(true);
  auto netoutput_sub_node = fused_compute_graph->FindNode("netoutput_sub");
  AttrUtils::SetGraph(conv2d_fused_node->GetOpDesc(), "_original_fusion_graph", fused_compute_graph);
  AddCompileResult(conv2d_fused_node, false);
  return compute_graph;
}
} // namespace

class AicoreNodeConverterUT : public bg::BgTestAutoCreate3StageFrame {
 protected:
  void SetUp() override {
    //BgTest::SetUp();
    BgTestAutoCreate3StageFrame::SetUp();
  }
  void TearDown() override {
    BgTestAutoCreate3StageFrame::TearDown();
  }
  void TestAicoreNodeConvert(int type) {
    auto graph = ShareGraph::AicoreGraph();
    auto add_node = graph->FindNode("add1");
    std::vector<std::string> dfx_opts = {kOpDfxPrintf, kOpDfxAssert};
    (void)ge::AttrUtils::SetListStr(add_node->GetOpDesc(), gert::kOpDfxOptions, dfx_opts);
    (void)ge::AttrUtils::SetInt(add_node->GetOpDesc(), gert::kOpDfxBufferSize, 12345);
    if (type > 0) {
      (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixVectorCore);
      std::vector<uint32_t> core_num_v = {15, 8, 7};
      (void)ge::AttrUtils::SetListInt(add_node->GetOpDesc(), kMixCoreNumVec, core_num_v);
      add_node->GetOpDesc()->SetAttachedStreamId(0);
      std::vector<int32_t> notify_id_v = {0, 1};
      (void)ge::AttrUtils::SetListInt(add_node->GetOpDesc(), ge::RECV_ATTR_NOTIFY_ID, notify_id_v);
    }
    if (type == 3) {
      (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::ATTR_NAME_CUBE_VECTOR_CORE_TYPE, kCoreTypeMixAICore);
    }
    if (type == 4) {
      (void)ge::AttrUtils::SetBool(add_node->GetOpDesc(), kDisableMixVectorCore, true);
    }
    if (type == 5) {
      std::vector<uint32_t> core_num_v = {15, 8};
      (void)ge::AttrUtils::SetListInt(add_node->GetOpDesc(), kMixCoreNumVec, core_num_v);
    }
    if (type == 6) {
      add_node->GetOpDesc()->SetAttachedStreamId(-1);
    }
    if (type == 7) {
      (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_op_aicore_num", "2");
      (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_op_vectorcore_num", "4");
    }
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
    bg::ValueHolder::PushGraphFrame();
    auto root_frame = bg::ValueHolder::GetCurrentFrame();
    (void)root_frame;
    auto init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("Init", {});
    bg::ValueHolder::PushGraphFrame(init_node, "Init");
    auto init_frame = bg::ValueHolder::PopGraphFrame();
    (void)init_frame;
    auto de_init_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>("DeInit", {});
    bg::ValueHolder::PushGraphFrame(de_init_node, "DeInit");
    auto de_init_frame = bg::ValueHolder::PopGraphFrame();
    (void)de_init_frame;

    auto main_node = bg::ValueHolder::CreateVoid<bg::ValueHolder>(GetExecuteGraphTypeStr(ExecuteGraphType::kMain), {});
    bg::ValueHolder::PushGraphFrame(main_node, "Main");
    auto root_model = GeModelBuilder(graph).BuildGeRootModel();
    auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build(3);
    std::vector<bg::ValueHolderPtr> notifies;
    int64_t block_dim = 5;
    auto notify = bg::ValueHolder::CreateConst(&block_dim, sizeof(block_dim));
    notifies.emplace_back(notify);
    notifies.emplace_back(notify);
    global_data.SetRtNotifies(notifies);

    bg::LowerConstDataNode(global_data);
    LowerInput data_input = {{}, {}, &global_data};

    auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
    auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
    ASSERT_TRUE(data1_ret.result.IsSuccess());
    ASSERT_TRUE(data2_ret.result.IsSuccess());

    LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                            {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                            &global_data};

    gert::GlobalDumper::GetInstance()->SetEnableFlags(
        gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
    auto add_ret = LoweringAiCoreNode(add_node, add_input);
    ASSERT_TRUE(add_ret.result.IsSuccess());
    ASSERT_EQ(add_ret.out_addrs.size(), 1);
    ASSERT_EQ(add_ret.out_shapes.size(), 1);
    ASSERT_EQ(add_ret.order_holders.size(), 1);

    auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
    ASSERT_NE(exe_graph, nullptr);
    // graph compare

    auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs),
                                                        add_ret.order_holders)->GetExecuteGraph();
    ASSERT_NE(execute_graph, nullptr);
    DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
    gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
  }
};
TEST_F(AicoreNodeConverterUT, ConvertGeneralAicoreNode) {
  auto graph = ShareGraph::AicoreGraph();
  auto add_node = graph->FindNode("add1");
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_op_aicore_num", "2");
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_op_vectorcore_num", "4");
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

  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}
TEST_F(AicoreNodeConverterUT, ConvertSoftAicoreNode) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto add_node = graph->FindNode("add1");
  auto compile_result = global_data.FindCompiledResult(add_node);
  // 0. alloc rt arg
  const domi::TaskDef *task_def = GetTaskDef(add_node, compile_result, TaskDefType::kAICore);
  auto task_def_t = const_cast<domi::TaskDef*>(task_def);
  uint16_t args_offset[9] = {0};
  task_def_t->mutable_kernel()->mutable_context()->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  task_def_t->mutable_kernel()->set_args_size(64);
  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  std::initializer_list<int64_t> shape = {2, 2, 2, 2};
  auto op_desc = graph->FindNode("add1")->GetOpDesc();
  for (size_t i = 0; i < op_desc->GetInputsSize(); ++i) {
    op_desc->MutableInputDesc(i)->SetFormat(FORMAT_NCHW);
    op_desc->MutableInputDesc(i)->SetOriginFormat(FORMAT_NCHW);
    op_desc->MutableInputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableInputDesc(i)->SetDataType(DT_FLOAT);
    op_desc->MutableInputDesc(i)->SetOriginDataType(DT_FLOAT);
  }
  for (size_t i = 0; i < op_desc->GetOutputsSize(); ++i) {
    op_desc->MutableOutputDesc(i)->SetFormat(FORMAT_NCHW);
    op_desc->MutableOutputDesc(i)->SetOriginFormat(FORMAT_NCHW);
    op_desc->MutableOutputDesc(i)->SetShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetOriginShape(GeShape(shape));
    op_desc->MutableOutputDesc(i)->SetDataType(DT_FLOAT);
    op_desc->MutableOutputDesc(i)->SetOriginDataType(DT_FLOAT);
  }
  (void)ge::AttrUtils::SetBool(op_desc, kStaticToDynamicSoftSyncOp, true);
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertAtomicAicoreNode) {
  auto graph = ShareGraph::BuildAtomicAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("TransData", true).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());

  LowerInput inputs = {data1_ret.out_shapes, data1_ret.out_addrs, &global_data};
  auto ret = LoweringAiCoreNode(graph->FindNode("trans1"), inputs);
  ASSERT_TRUE(ret.result.IsSuccess());
  // todo check graph

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(ret.out_addrs), ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  // GraphUtils::DumpGEGraph(execute_graph, "", true, "./atomic_exe_graph.txt");
  DumpGraph(execute_graph.get(), "AtomicAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertStaticMemSetAicoreNode) {
  auto graph = ShareGraph::BuildMemSetAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("TransData", true).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());

  LowerInput inputs = {data1_ret.out_shapes, data1_ret.out_addrs, &global_data};
  auto trans1 = graph->FindNode("trans1");
  trans1->GetOpDesc()->DelAttr( "_atomic_compile_info_json");
  auto ret = LoweringAiCoreNode(trans1, inputs);
  ASSERT_TRUE(ret.result.IsSuccess());
  // todo check graph

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(ret.out_addrs), ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  // GraphUtils::DumpGEGraph(execute_graph, "", true, "./atomic_exe_graph.txt");
  DumpGraph(execute_graph.get(), "MemSetAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertDynamicMemSetAicoreNode) {
  auto graph = ShareGraph::BuildMemSetAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("TransData", true).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());

  LowerInput inputs = {data1_ret.out_shapes, data1_ret.out_addrs, &global_data};
  auto trans1 = graph->FindNode("trans1");
  auto ret = LoweringAiCoreNode(trans1, inputs);
  ASSERT_TRUE(ret.result.IsSuccess());
  // todo check graph

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(ret.out_addrs), ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  // GraphUtils::DumpGEGraph(execute_graph, "", true, "./atomic_exe_graph.txt");
  DumpGraph(execute_graph.get(), "MemSetAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertStaticNode) {
  auto graph = ShareGraph::AicoreStaticGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);
  FastNodeTopoChecker checker(add_ret.out_addrs[0]);
  ge::DumpGraph(add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr(), "demo");
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SelectL2Allocator", 0}, {"CalcTensorSizeFromStorage", 0}}), true),
            "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<FastSrcNode>({{"FreeMemory", 0}, {"LaunchKernelWithFlag", 13}})),
            "success");
  EXPECT_EQ(checker.InChecker().DataFromByType("CalcTensorSizeFromStorage").DataFromByType("Const").Result(),
            "success");

  auto shape_node = add_ret.out_shapes[0]->GetFastNode();
  EXPECT_EQ(shape_node->GetType(), "Const");

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  auto launch_node = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "LaunchKernelWithFlag");
  ASSERT_NE(launch_node, nullptr);
  FastNodeTopoChecker launch_checker(launch_node);
  EXPECT_EQ(launch_checker.InChecker().DataFromByType("Const").Result(), "success");
  EXPECT_EQ(launch_checker.InChecker().DataFromByType("AllocMemHbm").Result(), "success");
  EXPECT_EQ(launch_checker.InChecker().DataFromByType("AllocBatchHbm").Result(), "success");

  auto workspace_addr = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "AllocBatchHbm");
  ASSERT_NE(workspace_addr, nullptr);
  FastNodeTopoChecker workspace_checker(launch_node);
  EXPECT_EQ(workspace_checker.InChecker().DataFromByType("Const").Result(), "success");
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "InferShape"), nullptr);
  EXPECT_EQ(ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "Tiling"), nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "StaticAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConstructUbFusionNodeInferShapeOk) {
  auto graph = BuildGraphWithUBfusion();
  auto ub_fusion_node = graph->FindNode("conv2d_fused");
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Conv2d", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data_a_ret = LoweringDataNode(graph->FindNode("data_a"), data_input);
  ASSERT_TRUE(data_a_ret.result.IsSuccess());
  ASSERT_EQ(data_a_ret.out_shapes.size(), 1);
  LowerInput conv2d_input = {{data_a_ret.out_shapes[0]},
                           {data_a_ret.out_addrs[0]},
                           &global_data};

  auto conv_ret = LoweringAiCoreNode(ub_fusion_node, conv2d_input);
  ASSERT_TRUE(conv_ret.result.IsSuccess());
  ASSERT_EQ(conv_ret.out_addrs.size(), 1);
  ASSERT_EQ(conv_ret.out_shapes.size(), 1);
  ASSERT_EQ(conv_ret.order_holders.size(), 1);

  auto exe_graph = conv_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  auto conv_tiling = ge::ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "CacheableTiling");
  ASSERT_NE(conv_tiling, nullptr);
  auto relu_infershape = conv_tiling->GetInDataNodes().at(1);
  // check relu infershape as input of conv_tiling.
  // which mean conv2d is unfold as conv2d->relu
  ASSERT_EQ(relu_infershape->GetType(), "InferShape");
  ASSERT_TRUE(relu_infershape->GetName().find("relu") != std::string::npos);
}

TEST_F(AicoreNodeConverterUT, ConvertAbnormalAicoreNode) {
  auto graph = ShareGraph::AicoreNoCompileResultGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build(1);
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};

  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);
  // No compile result
  ASSERT_FALSE(add_ret.result.IsSuccess());

  // // holder is nullptr
  add_input.input_shapes.emplace_back(nullptr);
  add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);
  ASSERT_FALSE(add_ret.result.IsSuccess());
}

TEST_F(AicoreNodeConverterUT, AddDataNodeForCleanAtomicTestOK) {
  auto atomic_op_desc = std::make_shared<ge::OpDesc>("AtomicClean", "DynamicAtomicAddrClean");
  ASSERT_NE(atomic_op_desc, nullptr);
  atomic_op_desc->AppendIrInput("workspace", ge::kIrInputRequired);
  atomic_op_desc->AddInputDesc("workspace", ge::GeTensorDesc());

  const size_t output_size = 5UL;
  atomic_op_desc->AppendIrInput("output", ge::kIrInputDynamic);
  for (size_t i = 0U; i < output_size; ++i) {
    atomic_op_desc->AddInputDesc("output" + std::to_string(i + 1), ge::GeTensorDesc());
  }

  auto graph = std::make_shared<ge::ComputeGraph>("graph");
  ASSERT_NE(graph, nullptr);
  auto clean_node = graph->AddNode(atomic_op_desc);
  ASSERT_NE(clean_node, nullptr);
  EXPECT_EQ(AddDataNodeForAtomic(graph, clean_node, output_size), ge::SUCCESS);
  clean_node = graph->FindNode("AtomicClean");
  ASSERT_NE(clean_node, nullptr);
  EXPECT_EQ(clean_node->GetInDataNodesAndAnchors().size(), output_size + 1UL);
}

TEST_F(AicoreNodeConverterUT, ConvertThirdClassAicoreNode) {
  auto graph = ShareGraph::AicoreGraph();
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
  auto add_node = graph->FindNode("add1");
  ge::AttrUtils::SetInt(add_node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, 3);
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 2);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertAicoreNodeWithOverflow) {
  auto graph = ShareGraph::AicoreGraph();
  auto add_node = graph->FindNode("add1");
  AttrUtils::SetBool(add_node->GetOpDesc(), "globalworkspace_type", true);
  AttrUtils::SetInt(add_node->GetOpDesc(), "globalworkspace_size", 32U);
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

  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare
  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertPartSupportAicoreNode1) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, true).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  (void)ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::kPartiallySupported, true);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::kAICpuKernelLibName, "aicpu_tf_kernel");
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  FastNode *aicore_launch_node = nullptr;
  FastNode *cpu_launch_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
    if (node->GetType() == "LaunchKernelWithHandle") {
      aicore_launch_node = node;
    } else if (node->GetType() == "AicpuLaunchTfKernel") {
      cpu_launch_node = node;
    }
  }
  ASSERT_NE(aicore_launch_node, nullptr);
  ASSERT_NE(cpu_launch_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralPartSupportAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertPartSupportAicoreNode2) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false, true).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  (void)ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::kPartiallySupported, true);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::kAICpuKernelLibName, "aicpu_tf_kernel");
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);

  FastNode *aicore_launch_node = nullptr;
  FastNode *cpu_launch_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
    if (node->GetType() == "LaunchKernelWithFlag") {
      aicore_launch_node = node;
    } else if (node->GetType() == "AicpuLaunchTfKernel") {
      cpu_launch_node = node;
    }
  }
  ASSERT_NE(aicore_launch_node, nullptr);
  ASSERT_NE(cpu_launch_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralPartSupportAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertPartSupportAicoreNode3) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  (void)ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::kPartiallySupported, true);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::kAICpuKernelLibName, "aicpu_tf_kernel");
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  FastNode *aicore_launch_node = nullptr;
  FastNode *cpu_launch_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
  if (node->GetType() == "LaunchKernelWithHandle") {
    aicore_launch_node = node;
  } else if (node->GetType() == "AicpuLaunchTfKernel") {
    cpu_launch_node = node;
  }
  }
  ASSERT_NE(aicore_launch_node, nullptr);
  ASSERT_EQ(cpu_launch_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralPartSupportAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertPartSupportAicoreNode4) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false, true).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  (void)ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::kPartiallySupported, true);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::kAICpuKernelLibName, "aicpu_tf_kernel");
  auto ori_ptr = NodeConverterRegistry::GetInstance().FindRegisterData(ge::kEngineNameAiCpuTf);
  NodeConverterRegistry::ConverterRegisterData ori_data = *ori_ptr;
  NodeConverterRegistry::ConverterRegisterData data;
  data.converter = nullptr;
  NodeConverterRegistry::GetInstance().Register(ge::kEngineNameAiCpuTf, data);
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  FastNode *aicore_launch_node = nullptr;
  FastNode *cpu_launch_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
  if (node->GetType() == "LaunchKernelWithHandle") {
    aicore_launch_node = node;
  } else if (node->GetType() == "AicpuLaunchTfKernel") {
    cpu_launch_node = node;
  }
  }
  ASSERT_NE(aicore_launch_node, nullptr);
  ASSERT_EQ(cpu_launch_node, nullptr);
  NodeConverterRegistry::GetInstance().Register(ge::kEngineNameAiCpuTf, ori_data);
  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralPartSupportAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertPartSupportAicoreNode5) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false, true).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  (void)add_node->GetOpDesc()->DelAttr("compile_info_json");
  (void)ge::AttrUtils::SetBool(add_node->GetOpDesc(), ge::kPartiallySupported, true);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), ge::kAICpuKernelLibName, "aicpu_tf_kernel");
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);

  FastNode *aicore_launch_node = nullptr;
  FastNode *cpu_launch_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
  if (node->GetType() == "LaunchKernelWithFlag") {
  aicore_launch_node = node;
  } else if (node->GetType() == "AicpuLaunchTfKernel") {
  cpu_launch_node = node;
  }
  }
  ASSERT_NE(aicore_launch_node, nullptr);
  ASSERT_EQ(cpu_launch_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
}

TEST_F(AicoreNodeConverterUT, AicoreEmptyTensorTest1) {
    auto graph = ShareGraph::AicoreGraph();
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

    auto add_node = graph->FindNode("add1");
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 10}, {1, 10}, {1, 10}, {1, 10}};
    for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
      output_tensor->SetShapeRange(shape_range);
    }

    auto add_ret = LoweringAiCoreNode(add_node, add_input);
    ASSERT_TRUE(add_ret.result.IsSuccess());
    ASSERT_EQ(add_ret.out_addrs.size(), 1);
    ASSERT_EQ(add_ret.out_shapes.size(), 1);
    ASSERT_EQ(add_ret.order_holders.size(), 1);

    auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
    ASSERT_NE(exe_graph, nullptr);
    // graph compare

    FastNode *check_shape_node = nullptr;
    for (auto &node : exe_graph->GetAllNodes()) {
      if (node->GetType() == "CheckOutputShapesEmpty") {
        check_shape_node = node;
        break;
      }
    }
    ASSERT_EQ(check_shape_node, nullptr);

    auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
    ASSERT_NE(execute_graph, nullptr);
    DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, AicoreEmptyTensorTest2) {
    auto graph = ShareGraph::AicoreGraph();
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

    auto add_node = graph->FindNode("add1");
    std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 10}, {1, 10}, {0, 10}, {1, 10}};
    for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
      output_tensor->SetShapeRange(shape_range);
    }

    auto add_ret = LoweringAiCoreNode(add_node, add_input);
    ASSERT_TRUE(add_ret.result.IsSuccess());
    ASSERT_EQ(add_ret.out_addrs.size(), 1);
    ASSERT_EQ(add_ret.out_shapes.size(), 1);
    ASSERT_EQ(add_ret.order_holders.size(), 1);

    auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
    ASSERT_NE(exe_graph, nullptr);
    // graph compare

    FastNode * check_shape_node = nullptr;
    for (auto &node : exe_graph->GetAllNodes()) {
      if (node->GetType() == "CheckOutputShapesEmpty") {
        check_shape_node = node;
        break;
      }
    }
    ASSERT_NE(check_shape_node, nullptr);

    auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
    ASSERT_NE(execute_graph, nullptr);
    DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, AicoreEmptyTensorTest3) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 10}, {1, 10}, {1, 10}, {1, 10}};
  for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
    output_tensor->SetShapeRange(shape_range);
  }
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);

  FastNode * check_shape_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
    if (node->GetType() == "CheckOutputShapesEmpty") {
      check_shape_node = node;
      break;
    }
  }
  ASSERT_EQ(check_shape_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralPartSupportAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, AicoreEmptyTensorTest4) {
  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 10}, {1, 10}, {0, 10}, {1, 10}};
  for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
    output_tensor->SetShapeRange(shape_range);
  }
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);

  FastNode * check_shape_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
    if (node->GetType() == "CheckOutputShapesEmpty") {
      check_shape_node = node;
      break;
    }
  }
  ASSERT_NE(check_shape_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralPartSupportAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, AicoreEmptyTensorTest5) {
  auto graph = ShareGraph::AicoreGraph();
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

  auto add_node = graph->FindNode("add1");
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 10}, {1, 10}, {1, 10}, {1, 10}};
  for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
    output_tensor->SetShape(GeShape({-2}));
    output_tensor->SetShapeRange(shape_range);
  }

  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  FastNode * check_shape_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
    if (node->GetType() == "CheckOutputShapesEmpty") {
      check_shape_node = node;
      break;
    }
  }
  ASSERT_NE(check_shape_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, AicoreEmptyTensorTest6) {
  auto graph = ShareGraph::AicoreGraph();
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
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{}};
  auto add_node = graph->FindNode("add1");
  for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
    output_tensor->SetShape(GeShape({1,10}));
    output_tensor->SetShapeRange(shape_range);
  }

  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  FastNode * check_shape_node = nullptr;
  for (auto &node : exe_graph->GetAllNodes()) {
    if (node->GetType() == "CheckOutputShapesEmpty") {
      check_shape_node = node;
      break;
    }
  }
  ASSERT_NE(check_shape_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, AicoreEmptyTensorTest7) {
  const auto back_options = ge::GetThreadLocalContext().GetAllGraphOptions();
  auto new_options = back_options;
  new_options[ge::OPTION_ALL_TENSOR_NOT_EMPTY] = "1";
  ge::GetThreadLocalContext().SetGraphOption(new_options);

  auto graph = ShareGraph::AicoreGraph();
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

  auto add_node = graph->FindNode("add1");
  std::vector<std::pair<int64_t, int64_t>> shape_range = {{1, 10}, {1, 10}, {0, 10}, {1, 10}};
  for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
    output_tensor->SetShapeRange(shape_range);
  }

  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  FastNode * check_shape_node = ExecuteGraphUtils::FindFirstNodeMatchType(exe_graph, "CheckOutputShapesEmpty");
  ASSERT_EQ(check_shape_node, nullptr);

  auto execute_graph = bg::ValueHolder::PopGraphFrame(ConvertDevMemValueHoldersToValueHolders(add_ret.out_addrs), add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralNoEmptyTensorAiCoreExe");
}

TEST_F(AicoreNodeConverterUT, ConvertStaticNodeReuseBinary) {
  auto graph = ShareGraph::AicoreStaticGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  for (const auto &node : graph->GetDirectNode()) {
    for (auto &output_tensor : node->GetOpDesc()->GetAllOutputsDescPtr()) {
      output_tensor->SetShapeRange({{1, 100}});
    }
    if (node->GetName() == "add1") {
      std::shared_ptr<optiling::utils::OpRunInfo> tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
      vector<int64_t> work_space_vec = {1,3,5};
      tiling_info->SetWorkspaces(work_space_vec);
      node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
    }
  }
  bg::LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  LowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                          {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                          &global_data};
  auto add_node = graph->FindNode("add1");
  for (auto &output_tensor : add_node->GetOpDesc()->GetAllOutputsDescPtr()) {
    output_tensor->SetShapeRange({{1, 100}});
  }
  auto add_ret = LoweringAiCoreNode(add_node, add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);
  FastNodeTopoChecker checker(add_ret.out_addrs[0]);
  EXPECT_EQ(checker.StrictConnectFrom(std::vector<FastSrcNode>({{"SelectL2Allocator", 0}, {"CalcTensorSizeFromStorage", 0}}), true),
            "success");
  EXPECT_EQ(checker.StrictConnectTo(0, std::vector<FastSrcNode>({{"FreeMemory", 0}, {"LaunchKernelWithFlag", 13}})),
            "success");
  EXPECT_EQ(checker.InChecker().DataFromByType("CalcTensorSizeFromStorage").DataFromByType("Const").Result(),
            "success");
}

TEST_F(AicoreNodeConverterUT, ConvertEnableVectorNode) {
  TestAicoreNodeConvert(0);
  TestAicoreNodeConvert(1);
  TestAicoreNodeConvert(2);
  TestAicoreNodeConvert(3);
  TestAicoreNodeConvert(4);
  TestAicoreNodeConvert(5);
  TestAicoreNodeConvert(6);
  TestAicoreNodeConvert(7);
}
}  // namespace gert
