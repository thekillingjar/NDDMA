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

#include "macro_utils/dt_public_scope.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "graph/op_desc.h"
#include "macro_utils/dt_public_unscope.h"

#include "engine/aicore/converter/aicore_node_converter.h"
#include "engine/aicore/converter/aicore_ffts_node_converter.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/converter/aicore_ffts_node_converter.h"
#include "register/ffts_node_converter_registry.h"
#include "register/ffts_node_calculater_registry.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "register/op_impl_registry.h"
#include "register/op_tiling_info.h"
#include "aicore/converter/bg_kernel_launch.h"
#include "framework/runtime/gert_const_types.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph_builder/bg_tiling.h"
#include "register/op_impl_registry.h"
#include "aicore/converter/bg_kernel_launch.h"
#include "engine/ffts_plus/converter/ffts_plus_common.h"
#include "exe_graph/lowering/lowering_definitions.h"
#include "graph_builder/bg_memory.h"
#include "graph/utils/graph_utils_ex.h"
#include "common/sgt_slice_type.h"
#include "faker/global_data_faker.h"
#include "runtime/subscriber/global_dumper.h"
#include "framework/runtime/gert_const_types.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "lowering/model_converter.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph/utils/graph_dump_utils.h"
#include "engine/aicore/converter/autofuse_node_converter.h"

using namespace ge;
using namespace gert::bg;
namespace gert {
namespace bg {
inline void LowerConstDataNode(LoweringGlobalData &global_data) {
  size_t const_data_num = static_cast<size_t>(ConstDataType::kTypeEnd);
  auto const_data_outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    std::vector<bg::ValueHolderPtr> const_datas;
    for (size_t i = 0U; i < const_data_num; ++i) {
      auto const_data_holder = bg::ValueHolder::CreateConstData(static_cast<int64_t>(i));
      const_datas.emplace_back(const_data_holder);
    }
    return const_datas;
  });
  for (size_t i = 0U; i < const_data_num; ++i) {
    const auto &const_data_name = GetConstDataTypeStr(static_cast<ConstDataType>(i));
    global_data.SetUniqueValueHolder(const_data_name, const_data_outputs[i]);
  }
}
}
namespace {
ge::graphStatus InferShapeStub(InferShapeContext *context) { return SUCCESS;}
IMPL_OP(Conv2d).InferShape(InferShapeStub);
IMPL_OP(Relu).InferShape(InferShapeStub);
} // namespace
extern LowerResult LoweringStaticAicoreNode(const ge::NodePtr &node, const LowerInput &lower_input);
inline void LowerConstDataNode(LoweringGlobalData &global_data) {
  size_t const_data_num = static_cast<size_t>(ConstDataType::kTypeEnd);
  auto const_data_outputs = bg::FrameSelector::OnInitRoot([&]() -> std::vector<bg::ValueHolderPtr> {
    std::vector<bg::ValueHolderPtr> const_datas;
    for (size_t i = 0U; i < const_data_num; ++i) {
      auto const_data_holder = bg::ValueHolder::CreateConstData(static_cast<int64_t>(i));
      const_datas.emplace_back(const_data_holder);
    }
    return const_datas;
  });
  for (size_t i = 0U; i < const_data_num; ++i) {
    const auto &const_data_name = GetConstDataTypeStr(static_cast<ConstDataType>(i));
    global_data.SetUniqueValueHolder(const_data_name, const_data_outputs[i]);
  }
}

class AicoreNodeConverterST : public bg::BgTestAutoCreate3StageFrame {
public:
  bg::ValueHolderPtr node_para_;
  std::unique_ptr<uint8_t[]> host_holder_;

  void CreateNodeParam(ge::NodePtr &node, LoweringGlobalData &global_data) {
    auto add_desc = node->GetOpDesc();
    (void)ge::AttrUtils::SetStr(add_desc, ge::kAttrCalcArgsSizeFunc, ge::kEngineNameAiCore);
    size_t size = 0;
    size_t pre_size = 0;
    // todo get node calculate size interface
    std::unique_ptr<uint8_t[]> pre_args_data = nullptr;
    auto cal_func = GetNodeCalculater(node);
    ASSERT_NE(cal_func, nullptr);
    cal_func(node, &global_data, size, pre_size, pre_args_data);
    ASSERT_NE(pre_args_data, nullptr);
    NodeMemPara node_para;
    node_para.size = size;
    host_holder_ = ge::MakeUnique<uint8_t[]>(size);
    ASSERT_NE(host_holder_, nullptr);
    memcpy(host_holder_.get(), pre_args_data.get(), pre_size);
    node_para.dev_addr = host_holder_.get();
    node_para.host_addr = host_holder_.get();
    node_para_ = bg::ValueHolder::CreateConst(&node_para, sizeof(node_para));
  }

 protected:
  void SetUp() override {
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
TEST_F(AicoreNodeConverterST, ConvertGeneralAicoreNode) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry();
  auto conv2d_op_impl= const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
      DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Conv2d"));
  auto conv2d_infer_func = conv2d_op_impl->infer_shape;
  conv2d_op_impl->infer_shape = InferShapeStub;

  auto relu_op_impl = const_cast<OpImplKernelRegistry::OpImplFunctionsV2 *>(
      DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->GetOpImpl("Relu"));
  auto relu_infer_func = relu_op_impl->infer_shape;
  relu_op_impl->infer_shape = InferShapeStub;

  auto graph = ShareGraph::AicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("Add", false).Build();
  LowerConstDataNode(global_data);
  auto add_node = graph->FindNode("add1");
  auto compile_result = global_data.FindCompiledResult(add_node);
  // 0. alloc rt arg
  const domi::TaskDef *task_def = GetTaskDef(add_node, compile_result, TaskDefType::kAICore);
  auto task_def_t = const_cast<domi::TaskDef*>(task_def);
  uint16_t args_offset[9] = {0};
  task_def_t->mutable_kernel()->mutable_context()->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  task_def_t->mutable_kernel()->set_args_size(64);
  char args[64] = {0};
  task_def_t->mutable_kernel()->set_args(args, 64);
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
  auto add_ret = LoweringStaticAicoreNode(graph->FindNode("add1"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);

  auto exe_graph = add_ret.out_addrs[0]->GetFastNode()->GetExtendInfo()->GetOwnerGraphBarePtr();
  ASSERT_NE(exe_graph, nullptr);
  // graph compare

  std::vector<ValueHolderPtr> out_addrs;
  for (const auto &addr : add_ret.out_addrs) {
    out_addrs.emplace_back(addr);
  }
  auto execute_graph = bg::ValueHolder::PopGraphFrame(out_addrs, add_ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  DumpGraph(execute_graph.get(), "GeneralAiCoreExe");

  conv2d_op_impl->infer_shape = conv2d_infer_func;
  relu_op_impl->infer_shape = relu_infer_func;
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}

TEST_F(AicoreNodeConverterST, ConvertStaticMemSetAicoreNode) {
  auto graph = ShareGraph::BuildMemSetAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("TransData", true).Build();
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());

  LowerInput inputs = {data1_ret.out_shapes, data1_ret.out_addrs, &global_data};
  auto ret = LoweringAiCoreNode(graph->FindNode("trans1"), inputs);
  ASSERT_TRUE(ret.result.IsSuccess());
  // todo check graph

  std::vector<ValueHolderPtr> out_addrs;
  for (const auto &addr : ret.out_addrs) {
    out_addrs.emplace_back(addr);
  }
  auto execute_graph = bg::ValueHolder::PopGraphFrame(out_addrs, ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  // GraphUtils::DumpGEGraph(execute_graph, "", true, "./atomic_exe_graph.txt");
  DumpGraph(execute_graph.get(), "MemSetAiCoreExe");
}

TEST_F(AicoreNodeConverterST, ConvertDynamicMemSetAicoreNode) {
  auto graph = ShareGraph::BuildMemSetAicoreGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithHandleAiCore("TransData", true).Build();
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  auto trans_node = graph->FindNode("trans1");
  ge::GeTensorDescPtr tensor_ptr = trans_node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(tensor_ptr, nullptr);
  (void)tensor_ptr->SetShapeRange({{0, 1}, {0, 10}});
  LowerInput inputs = {data1_ret.out_shapes, data1_ret.out_addrs, &global_data};
  auto ret = LoweringAiCoreNode(graph->FindNode("trans1"), inputs);
  ASSERT_TRUE(ret.result.IsSuccess());
  // todo check graph

  std::vector<ValueHolderPtr> out_addrs;
  for (const auto &addr : ret.out_addrs) {
    out_addrs.emplace_back(addr);
  }
  auto execute_graph = bg::ValueHolder::PopGraphFrame(out_addrs, ret.order_holders)->GetExecuteGraph();
  ASSERT_NE(execute_graph, nullptr);
  // GraphUtils::DumpGEGraph(execute_graph, "", true, "./atomic_exe_graph.txt");
  DumpGraph(execute_graph.get(), "MemSetAiCoreExe");
}

TEST_F(AicoreNodeConverterST, ConvertStaticNodeReuseBinary) {
  auto graph = ShareGraph::AicoreStaticGraph();
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).FakeWithoutHandleAiCore("Add", false).Build();
  for (const ge::NodePtr &node : graph->GetDirectNode()) {
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
  auto add_ret = LoweringAiCoreNode(graph->FindNode("add1"), add_input);
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

TEST_F(AicoreNodeConverterST, ConvertPartSupportAicoreNode) {
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

TEST_F(AicoreNodeConverterST, ConvertEnableVectorNode) {
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
