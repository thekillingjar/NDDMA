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
#include "runtime/model_v2_executor.h"
#include "framework/common/ge_types.h"
#include "engine/aicore/fe_rt2_common.h"
#include "exe_graph/runtime/kernel_context.h"
#include <gtest/gtest.h>
#include "lowering/model_converter.h"
#include "exe_graph/runtime/tiling_context.h"
#include "lowering/graph_converter.h"
#include "register/op_impl_registry.h"
#include "register/op_tiling_registry.h"
// stub and faker
#include "faker/global_data_faker.h"
#include "common/share_graph.h"
#include "common/sgt_slice_type.h"
#include "faker/ge_model_builder.h"
#include "faker/fake_value.h"
#include "faker/magic_ops.h"
#include "faker/kernel_run_context_facker.h"
#include "stub/gert_runtime_stub.h"
#include "check/executor_statistician.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph_builder/bg_tiling.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_dump_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "framework/executor/ge_executor.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/load/model_manager/task_info/task_info.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "graph/operator_factory_impl.h"
#include "kernel/common_kernel_impl/sink_node_bin.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "engine/aicore/kernel/mixl2_update_kernel.h"
#include "engine/aicore/kernel/aicore_update_kernel.h"
#include "engine/aicore/kernel/rt_ffts_plus_launch_args.h"
#include "kernel/memory/single_stream_l2_allocator.h"
#include "runtime/subscriber/global_dumper.h"

#include "macro_utils/dt_public_scope.h"
#include "common/dump/exception_dumper.h"
#include "subscriber/dumper/executor_dumper.h"
#include "subscriber/profiler/cann_profiler_v2.h"
#include "macro_utils/dt_public_unscope.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "../aicore_exe_graph_check.h"
using namespace std;
using namespace testing;
using namespace ge;

namespace gert {
namespace kernel {
}
using namespace kernel;
const std::string kAtomicCtxIdList = "_atomic_context_id_list";
namespace {
ge::graphStatus TilingTestSuccess(TilingContext *tiling_context) {
  tiling_context->SetTilingKey(0);
  tiling_context->SetBlockDim(32);
  TilingData *tiling_data = tiling_context->GetRawTilingData();
  GELOGD("TilingTestSuccess tiling_data:%p.", tiling_data);
  if (tiling_data == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Tiling data is nullptr");
    return ge::GRAPH_FAILED;
  }
  auto workspaces = tiling_context->GetWorkspaceSizes(2);
  workspaces[0] = 4096;
  workspaces[1] = 6904;
  int64_t data = 100;
  tiling_data->Append<int64_t>(data);
  tiling_data->SetDataSize(sizeof(int64_t));

  return ge::GRAPH_SUCCESS;
}
ge::graphStatus TilingParseTEST(gert::KernelContext *kernel_context) {
  (void)kernel_context;
  GELOGD("TilingParseTEST");
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeTest1(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto input_shape_1 = *context->GetInputShape(1);
  auto output_shape = context->GetOutputShape(0);
  if (input_shape_0.GetDimNum() != input_shape_1.GetDimNum()) {
    GELOGE(ge::PARAM_INVALID, "Add param invalid, input_shape_0.GetDimNum() is %zu,  input_shape_1.GetDimNum() is %zu",
           input_shape_0.GetDimNum(), input_shape_1.GetDimNum());
  }
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeTest1 %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, std::max(input_shape_0.GetDim(i), input_shape_1.GetDim(i)));
    GELOGD("InferShapeTest1 index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferShapeTest2(InferShapeContext *context) {
  auto input_shape_0 = *context->GetInputShape(0);
  auto output_shape = context->GetOutputShape(0);
  output_shape->SetDimNum(input_shape_0.GetDimNum());
  GELOGD("InferShapeTest2 %zu", input_shape_0.GetDimNum());
  for (size_t i = 0; i < input_shape_0.GetDimNum(); ++i) {
    output_shape->SetDim(i, input_shape_0.GetDim(i));
    GELOGD("InferShapeTest2 index:%zu, val:%u.", i, output_shape->GetDim(i));
  }
  return ge::GRAPH_SUCCESS;
}
}  // namespace

class MixL2LoweringST : public testing::Test {
  void SetUp() override {}
  void TearDown() override {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
 public:
  KernelRegistryImpl &registry = KernelRegistryImpl::GetInstance();
  std::unordered_map<std::string, std::string> _check_kernel_map;
  void TestMixL2Lowering(ComputeGraphPtr &graph, LoweringGlobalData &global_data, bool expect) {
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    (void)bg::ValueHolder::PopGraphFrame();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
    auto exe_graph = GraphConverter()
        .SetModelDescHolder(&model_desc_holder)
        .ConvertComputeGraphToExecuteGraph(graph, global_data);
    if (expect) {
      ASSERT_NE(exe_graph, nullptr);
    } else {
      ASSERT_EQ(exe_graph, nullptr);
      return;
    }
    ge::DumpGraph(exe_graph.get(), "LoweringMixL2ExeGraph");
  }
  void TestMixL2SingleLowering(GeRootModelPtr &root_model, LoweringGlobalData &global_data, std::vector<JudgeInfo> &judge_v,
                               bool expect, bool atomic, size_t in_num = 2, size_t out_num = 1) {
    auto graph = root_model->GetRootGraph();
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
    auto graph_convert = GraphConverter().SetModelDescHolder(&model_desc_holder);
    auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(graph, global_data);
    if (expect) {
      ASSERT_NE(exe_graph, nullptr);
    } else {
      ASSERT_EQ(exe_graph, nullptr);
      return;
    }
    for (auto &judge_info : judge_v) {
      ASSERT_EQ(Judge2KernelPriority(exe_graph, judge_info), true);
    }
    ge::DumpGraph(exe_graph.get(), "LoweringMixL2Node");

    auto model_executor = ModelV2Executor::Create(exe_graph, root_model);
    ASSERT_NE(model_executor, nullptr);

    auto ess = StartExecutorStatistician(model_executor);

    EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
    FakeTensors inputs = FakeTensors({4, 4, 4, 4}, in_num);
    FakeTensors outputs = FakeTensors({4, 4, 4, 4}, out_num);

    rtStream_t stream;
    ASSERT_EQ(rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIORITY_DEFAULT)), RT_ERROR_NONE);
    auto stream_value = FakeValue<uint64_t>(reinterpret_cast<uint64_t>(stream));

    ess->Clear();
    ASSERT_EQ(model_executor->Execute({stream_value.value}, inputs.GetTensorList(), inputs.size(),
                                      outputs.GetTensorList(), outputs.size()),
              ge::GRAPH_SUCCESS);
    ess->PrintExecutionSummary();
    if (atomic) {
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType("DynamicAtomicAddrClean", "FFTSUpdateAtomicArgs"), 1);
    }
    for (auto x : _check_kernel_map) {
      EXPECT_EQ(ess->GetExecuteCountByNodeTypeAndKernelType(x.first, x.second), 1);
    }
    Shape expect_out_shape = {4, 4, 4, 4};
    EXPECT_EQ(outputs.GetTensorList()[0]->GetShape().GetStorageShape(), expect_out_shape);

    ASSERT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
    rtStreamDestroy(stream);
  }
};
/***********************************************************************************************************************
 *
 * Data           Data
 *   \             /
 *    \           /
 *     \         /                                       Data      Data
 *      \       /                                          \        /
 *      onv2d_sqrt                                          \      /
 *          |                                                 conv2d
 *          |                                                   |
 *          |                                                 sqrt
 *      NetOutput                                               |
 *                                                        NetOutput
 *
 *
 **********************************************************************************************************************/
static void BuildMixL2NodeGraph(ComputeGraphPtr &root_graph, ge::NodePtr &node, bool single) {

  DEF_GRAPH(fused_graph) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto ret_val_0 = OP_CFG(FRAMEWORKOP).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Attr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal");

    auto conv = OP_CFG("CONV2D_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto sqrt = OP_CFG("SQRT_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_in_0", data_0)
              ->EDGE(0, 0)
              ->NODE("conv2d", conv)
              ->EDGE(0, 0)
              ->NODE("sqrt", sqrt)
              ->EDGE(0, 0)
              ->NODE("retVal", ret_val_0));
    CHAIN(NODE("_arg_in_1", data_1)
              ->EDGE(0, 1)
              ->NODE("conv2d", conv));
  };
  auto origin_fused_graph = ToComputeGraph(fused_graph);
  std::vector<int64_t> unknown_shape = {4,4,-1,4};
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto fused_conv = OP_CFG("CONV2D_T")
        .TensorDesc(FORMAT_NCHW, DT_FLOAT, unknown_shape)
        .InCnt(2)
        .OutCnt(1)
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
        .Build("Conv2D_Sqrt");
    CHAIN(NODE("_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE(fused_conv)
              ->EDGE(0, 0)
              ->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE(fused_conv));
  };
  uint32_t mem_offset = 0U;
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);
  AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);
  node = root_graph->FindNode("Conv2D_Sqrt");
  auto conv2d_desc = node->GetOpDesc();
  for (auto out_desc : conv2d_desc->GetAllOutputsDescPtr()) {
    out_desc->SetDataType(DT_FLOAT);
  }
  std::string kernel_bin_id = "test_kernel_bin_id";
  (void)ge::AttrUtils::SetStr(conv2d_desc, kAttrKernelBinId, kernel_bin_id);
  SetGraphOutShapeRange(root_graph);
  AttrUtils::SetGraph(conv2d_desc, "_original_fusion_graph", origin_fused_graph);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrLowingFunc, ge::kFFTSMixL2LowerFunc);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrCalcArgsSizeFunc, ge::kFFTSMixL2CalcFunc);
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);
  conv2d_desc->MutableInputDesc(0)->SetShape(GeShape(unknown_shape));

  vector<int64_t> workspace_bytes = { 200, 300, 400};
  conv2d_desc->SetWorkspaceBytes(workspace_bytes);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("s_mix_aictbeKernel", std::move(test_bin));
  conv2d_desc->SetExtAttr(std::string("_mix_aic") + OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  conv2d_desc->SetExtAttr(std::string("_mix_aiv") + OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  conv2d_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  std::vector<int64_t> clean_output_indexes = {0};
  (void)ge::AttrUtils::SetListInt(conv2d_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, clean_output_indexes);
  int64_t atom_max_size = 32;
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxAtomicCleanTilingSize, atom_max_size);
  std::string op_compile_info_json = R"({"_workspace_size_list":[32], "vars":{"ub_size": 12, "core_num": 2}})";
  ge::AttrUtils::SetStr(conv2d_desc, optiling::ATOMIC_COMPILE_INFO_JSON, op_compile_info_json);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIC");
  AttrUtils::SetBool(conv2d_desc, "support_dynamicshape", true);
  AttrUtils::SetInt(conv2d_desc, "op_para_size", 32);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);

  std::vector<std::string> names_prefix;
  names_prefix.emplace_back("_mix_aic");
  if (!single) {
    names_prefix.emplace_back("_mix_aiv");
    AttrUtils::SetStr(node->GetOpDesc(), "_mix_aic_kernel_list_first_name", "aic");
    AttrUtils::SetStr(node->GetOpDesc(), "_mix_aiv_kernel_list_first_name", "aiv");
  }
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
}

static void BuildMixL2NewNodeGraph(ComputeGraphPtr &root_graph, ge::NodePtr &node) {

  DEF_GRAPH(fused_graph) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto ret_val_0 = OP_CFG(FRAMEWORKOP).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0)
        .Attr(ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal");

    auto conv = OP_CFG("CONV2D_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto sqrt = OP_CFG("SQRT_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_in_0", data_0)
              ->EDGE(0, 0)
              ->NODE("conv2d", conv)
              ->EDGE(0, 0)
              ->NODE("sqrt", sqrt)
              ->EDGE(0, 0)
              ->NODE("retVal", ret_val_0));
    CHAIN(NODE("_arg_in_1", data_1)
              ->EDGE(0, 1)
              ->NODE("conv2d", conv));
  };
  auto origin_fused_graph = ToComputeGraph(fused_graph);

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto fused_conv = OP_CFG("CONV2D_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("Conv2D_Sqrt", fused_conv)
              ->EDGE(0, 0)
              ->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("Conv2D_Sqrt", fused_conv));
  };
  uint32_t mem_offset = 0U;
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);
  AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);
  node = root_graph->FindNode("Conv2D_Sqrt");
  auto conv2d_desc = node->GetOpDesc();
  AttrUtils::SetGraph(conv2d_desc, "_original_fusion_graph", origin_fused_graph);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrLowingFunc, ge::kFFTSMixL2LowerFunc);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrCalcArgsSizeFunc, ge::kFFTSMixL2CalcFunc);
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);

  vector<int64_t> workspace_bytes = { 200, 300, 400};
  conv2d_desc->SetWorkspaceBytes(workspace_bytes);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("s_mix_enhancedtbeKernel", std::move(test_bin));
  conv2d_desc->SetExtAttr(std::string("_mix_enhanced") + OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetBool(conv2d_desc, "support_dynamicshape", true);
  AttrUtils::SetBool(conv2d_desc, "_mix_with_enhanced_kernel", true);
  AttrUtils::SetInt(conv2d_desc, "op_para_size", 32);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  std::vector<std::string> names_prefix;
  names_prefix.emplace_back("_mix_enhanced");
  AttrUtils::SetStr(node->GetOpDesc(), "_mix_enhanced_kernel_list_first_name", "_mix_enhanced");
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
}

static void BuildMixL2NewDynNodeGraph(ComputeGraphPtr &root_graph, ge::NodePtr &node) {
  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto data_3 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 3);
    auto data_4 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 4);
    auto data_5 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 5);
    auto data_6 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 6);
    auto ifa_node = OP_CFG("IFA_T")
        .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
        .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
        .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("IFA_NODE", ifa_node)
              ->EDGE(0, 0)
              ->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", data_1)->EDGE(0, 1)->NODE("IFA_NODE", ifa_node));
    CHAIN(NODE("_arg_2", data_2)->EDGE(0, 2)->NODE("IFA_NODE", ifa_node));
    CHAIN(NODE("_arg_3", data_3)->EDGE(0, 3)->NODE("IFA_NODE", ifa_node));
    CHAIN(NODE("_arg_4", data_4)->EDGE(0, 4)->NODE("IFA_NODE", ifa_node));
    CHAIN(NODE("_arg_5", data_5)->EDGE(0, 5)->NODE("IFA_NODE", ifa_node));
    CHAIN(NODE("_arg_6", data_6)->EDGE(0, 6)->NODE("IFA_NODE", ifa_node));
    CHAIN(NODE("IFA_NODE", ifa_node)->EDGE(1, 1)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("IFA_NODE", ifa_node)->EDGE(2, 2)->NODE("Node_Output", NETOUTPUT));
  };
  uint32_t mem_offset = 0U;
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);
  AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);
  AttrUtils::SetInt(root_graph->FindNode("_arg_2")->GetOpDesc(), "index", 2);
  AttrUtils::SetInt(root_graph->FindNode("_arg_3")->GetOpDesc(), "index", 3);
  AttrUtils::SetInt(root_graph->FindNode("_arg_4")->GetOpDesc(), "index", 4);
  AttrUtils::SetInt(root_graph->FindNode("_arg_5")->GetOpDesc(), "index", 5);
  AttrUtils::SetInt(root_graph->FindNode("_arg_6")->GetOpDesc(), "index", 6);
  node = root_graph->FindNode("IFA_NODE");
  auto ifa_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(ifa_desc, ge::kAttrLowingFunc, ge::kFFTSMixL2LowerFunc);
  (void)ge::AttrUtils::SetStr(ifa_desc, ge::kAttrCalcArgsSizeFunc, ge::kFFTSMixL2CalcFunc);
  (void)ge::AttrUtils::SetInt(ifa_desc, bg::kMaxTilingSize, 50);

  for (size_t i = 0; i < ifa_desc->GetAllInputsDesc().size(); ++i) {
    ifa_desc->MutableInputDesc(i)->SetShape(GeShape({4, -1, -1, 4}));
  }

  for (size_t i = 0; i < ifa_desc->GetAllOutputsDesc().size(); ++i) {
    ifa_desc->MutableOutputDesc(i)->SetShape(GeShape({4, -1, -1, 4}));
  }
  vector<int64_t> workspace_bytes = { 200, 300, 400};
  ifa_desc->SetWorkspaceBytes(workspace_bytes);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("s_mix_enhancedtbeKernel", std::move(test_bin));
  ifa_desc->SetExtAttr(std::string("_mix_enhanced") + OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(ifa_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetBool(ifa_desc, "support_dynamicshape", true);
  AttrUtils::SetBool(ifa_desc, "_mix_with_enhanced_kernel", true);
  AttrUtils::SetInt(ifa_desc, "op_para_size", 32);
  (void)ge::AttrUtils::SetStr(ifa_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(ifa_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  std::vector<std::string> names_prefix;
  names_prefix.emplace_back("_mix_enhanced");
  AttrUtils::SetStr(node->GetOpDesc(), "_mix_enhanced_kernel_list_first_name", "_mix_enhanced");
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), ge::ATTR_NAME_KERNEL_NAMES_PREFIX, names_prefix);
}

static bool RemoveAnchor(const ge::AnchorPtr &outputanchor, const ge::AnchorPtr &inputanchor) {
  if (outputanchor == nullptr || inputanchor == nullptr) {
    return false;
  }
  if (outputanchor->IsLinkedWith(inputanchor)) {
    ge::GraphUtils::RemoveEdge(outputanchor, inputanchor);
    return true;
  }
  return false;
}

TEST_F(MixL2LoweringST, lowering_mix_l2_1_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, true);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  ctx_def_0.mutable_mix_aic_aiv_ctx()->set_args_format("{i_instance0}{i_instance1}{}{o_instance0}{}");
  (void)ctx_def_0.mutable_label_ctx();
  uint32_t need_mode = 1U;
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kNeedModeAddr, need_mode);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_memcheck", true);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), "op_para_size", 512);

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  auto op_impl_func = space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T");
  op_impl_func->infer_shape = InferShapeTest1;
  op_impl_func->tiling = TilingTestSuccess;
  op_impl_func->tiling_parse = TilingParseTEST;
  space_registry_v2_array[0]->CreateOrGetOpImpl("SQRT_T")->infer_shape = InferShapeTest2;

  auto infer_fun = [](Operator &op) -> graphStatus {
    const char_t *name = "__output0";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferShapeFunc("CONV2D_T", infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc("SQRT_T", infer_fun);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("CONV2D_T", "FFTSUpdateMixL2Args");
  std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, false);

  // third op
  int32_t unknown_shape_type_val = ge::DEPEND_SHAPE_RANGE;
  (void) ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);
  node->GetOpDesc()->DelAttr(ge::kAttrLowingFunc);
  auto exe_graph = GraphConverter().ConvertComputeGraphToExecuteGraph(root_graph, global_data);
  ASSERT_EQ(exe_graph, nullptr);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_2_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  auto op_impl_func = space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T");
  op_impl_func->infer_shape = InferShapeTest1;
  op_impl_func->tiling = TilingTestSuccess;
  op_impl_func->tiling_parse = TilingParseTEST;
  space_registry_v2_array[0]->CreateOrGetOpImpl("SQRT_T")->infer_shape = InferShapeTest2;

  auto infer_fun = [](Operator &op) -> graphStatus {
    const char_t *name = "__output0";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferShapeFunc("CONV2D_T", infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc("SQRT_T", infer_fun);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("CONV2D_T", "FFTSUpdateMixL2Args");
  std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, true);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_with_optional_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  auto op_impl_func = space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T");
  op_impl_func->infer_shape = InferShapeTest1;
  op_impl_func->tiling = TilingTestSuccess;
  op_impl_func->tiling_parse = TilingParseTEST;
  space_registry_v2_array[0]->CreateOrGetOpImpl("SQRT_T")->infer_shape = InferShapeTest2;

  auto infer_fun = [](Operator &op) -> graphStatus {
    const char_t *name = "__output0";
    op.UpdateOutputDesc(name, op.GetInputDesc(0));
    return GRAPH_SUCCESS;
  };
  OperatorFactoryImpl::RegisterInferShapeFunc("CONV2D_T", infer_fun);
  OperatorFactoryImpl::RegisterInferShapeFunc("SQRT_T", infer_fun);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 7);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  NodeUtils::AppendInputAnchor(node, 4);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("CONV2D_T", "FFTSUpdateMixL2Args");
  std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, true);
}


TEST_F(MixL2LoweringST, lowering_static_mix_l2_2_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(out_desc, nullptr);
  out_desc->SetShape(GeShape({4, 4, 4, 4}));
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, false);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("CONV2D_T", "FFTSUpdateMixL2Args");
  std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, true);
}

TEST_F(MixL2LoweringST, lowering_static_mix_l2_reuse_binary_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info;
  tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  tiling_info->AddTilingData("666");
  node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), "_tiling_data_str", "666");
  node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_ATOMIC_OP_RUN_INFO, tiling_info);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(out_desc, nullptr);
  out_desc->SetShape(GeShape({4, 4, 4, 4}));
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, false);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("CONV2D_T", "FFTSUpdateMixL2Args");
std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, true);
}

TEST_F(MixL2LoweringST, lowering_static_mix_l2_moe_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(out_desc, nullptr);
  out_desc->SetShape(GeShape({4, 4, 4, 4}));
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, false);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), ge::ATTR_NAME_DYNAMIC_TILING_DEPEND_OP, true);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("CONV2D_T", "TilingLegacy");
  _check_kernel_map.emplace("DynamicAtomicAddrClean", "TilingLegacy");
  _check_kernel_map.emplace("CONV2D_T", "FFTSUpdateMixL2Args");
  auto op_impl_func = space_registry_v2_array[0]->CreateOrGetOpImpl("CONV2D_T");
  op_impl_func->tiling = TilingTestSuccess;
  op_impl_func->tiling_parse = TilingParseTEST;
  space_registry_v2_array[0]->CreateOrGetOpImpl("DynamicAtomicAddrClean")->tiling = TilingTestSuccess;
  space_registry_v2_array[0]->CreateOrGetOpImpl("DynamicAtomicAddrClean")->tiling_parse = TilingParseTEST;
  std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, true);
}

TEST_F(MixL2LoweringST, lowering_static_mix_l2_new_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NewNodeGraph(root_graph, node);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  (void)ctx_def_0.mutable_label_ctx();

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(out_desc, nullptr);
  out_desc->SetShape(GeShape({4, 4, 4, 4}));
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, false);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("CONV2D_T", "FFTSUpdateMixL2Args");
  std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, false);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_dynamic_new_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NewDynNodeGraph(root_graph, node);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  (void)ctx_def_0.mutable_label_ctx();

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto op_impl_func = space_registry_v2_array[0]->CreateOrGetOpImpl("IFA_T");
  op_impl_func->infer_shape = InferShapeTest1;
  op_impl_func->tiling = TilingTestSuccess;
  op_impl_func->tiling_parse = TilingParseTEST;

  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("IFA_T", "TilingLegacy");
  _check_kernel_map.emplace("IFA_T", "FFTSUpdateMixL2Args");
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<std::vector<int64_t>> dyn_in_vv = {{2,3}, {4,5,6}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 10);
  std::vector<uint32_t> insert_pos_vec = {7, 8, 9};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kInputInsertOptPosList, insert_pos_vec);
  std::vector<JudgeInfo> judge_v;
  JudgeInfo info = {"IFA_NODE", "FFTSUpdateMixL2Args", "IFA_NODE", "LaunchFFTSPlusTaskNoCopy"};
  judge_v.emplace_back(info);
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, false, 7, 3);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_dynamic_new_kernel_success1) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NewDynNodeGraph(root_graph, node);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  (void)ctx_def_0.mutable_label_ctx();

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  OpImplSpaceRegistryV2Array space_registry_v2_array;
  space_registry_v2_array[0] = DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  global_data.SetSpaceRegistriesV2(space_registry_v2_array);
  auto op_impl_func = space_registry_v2_array[0]->CreateOrGetOpImpl("IFA_T");
  op_impl_func->infer_shape = InferShapeTest1;
  op_impl_func->tiling = TilingTestSuccess;
  op_impl_func->tiling_parse = TilingParseTEST;

  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  _check_kernel_map.clear();
  _check_kernel_map.emplace("IFA_T", "TilingLegacy");
  _check_kernel_map.emplace("IFA_T", "FFTSUpdateMixL2Args");
  _check_kernel_map.emplace("IFA_T", "ExpandDfxWorkspaceSize");
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<std::vector<int64_t>> dyn_in_vv = {{2,3}, {5,6}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 10);
  std::vector<std::string> dfx_opts = {kOpDfxPrintf, kOpDfxAssert};
  (void)ge::AttrUtils::SetListStr(node->GetOpDesc(), gert::kOpDfxOptions, dfx_opts);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), gert::kOpDfxBufferSize, 12345);
  NodeUtils::AppendInputAnchor(node, 3);
  std::vector<JudgeInfo> judge_v;
  TestMixL2SingleLowering(root_model, global_data, judge_v, true, false, 7, 3);
}

TEST_F(MixL2LoweringST, sink_mixl2_get_kernel_info) {
  gert::kernel::AICoreSinkRet sink_ret;
  rtKernelDetailInfo_t rt_kernel_info_1;
  rt_kernel_info_1.functionInfoNum = 2;
  rt_kernel_info_1.functionInfo[0].pcAddr = &rt_kernel_info_1;
  rt_kernel_info_1.functionInfo[0].prefetchCnt = 10;
  rt_kernel_info_1.functionInfo[0].mixType = 3;
  rt_kernel_info_1.functionInfo[1].pcAddr = &rt_kernel_info_1;
  rt_kernel_info_1.functionInfo[1].prefetchCnt = 10;
  rt_kernel_info_1.functionInfo[1].mixType = 3;
  EXPECT_EQ(kernel::ParseKernelInfo(rt_kernel_info_1, rt_kernel_info_1, &sink_ret), ge::SUCCESS);
  EXPECT_EQ(sink_ret.aic_icache_prefetch_cnt, 10);
  EXPECT_EQ(sink_ret.aiv_icache_prefetch_cnt, 10);

  rtKernelDetailInfo_t rt_kernel_info_2;
  rt_kernel_info_2.functionInfoNum = 1;
  rt_kernel_info_2.functionInfo[0].pcAddr = &rt_kernel_info_2;
  rt_kernel_info_2.functionInfo[0].prefetchCnt = 10;
  rt_kernel_info_2.functionInfo[0].mixType = 2;

  EXPECT_EQ(kernel::ParseKernelInfo(rt_kernel_info_2, rt_kernel_info_2, &sink_ret), ge::SUCCESS);
  EXPECT_EQ(sink_ret.aiv_icache_prefetch_cnt, 10);

  rtKernelDetailInfo_t rt_kernel_info_3;
  rt_kernel_info_3.functionInfoNum = 1;
  rt_kernel_info_3.functionInfo[0].pcAddr = &rt_kernel_info_2;
  rt_kernel_info_3.functionInfo[0].prefetchCnt = 10;
  rt_kernel_info_3.functionInfo[0].mixType = 10;
  EXPECT_EQ(kernel::ParseKernelInfo(rt_kernel_info_3, rt_kernel_info_3, &sink_ret), ge::SUCCESS);

  rtKernelDetailInfo_t rt_kernel_info_4;
  rt_kernel_info_4.functionInfoNum = 3;
  EXPECT_EQ(kernel::CheckKernelInfoValid(rt_kernel_info_4), false);
}

TEST_F(MixL2LoweringST, test_mixl2_update_args) {
GlobalDumper::GetInstance()->SetEnableFlags(
    BuiltInSubscriberUtil::BuildEnableFlags<DumpType>({DumpType::kLiteExceptionDump}));
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  // update args
  constexpr uint64_t stub_mem_hbm_addr = 0x22;
  memory::CachingMemAllocator stub_allocator{0, 1};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 3);
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  GertTensorData work_space_gtd =
      GertTensorData(ms_block.GetAddr(), ms_block.GetSize(), TensorPlacement::kOnDeviceHbm, 0);
  auto work_space = ContinuousVector::Create<GertTensorData *>(3);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(3);
  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &work_space_gtd;
  }

  uint32_t io_addr_num = 3;
  AICoreSinkRet sink_ret;
  auto device_data = std::vector<int8_t>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  GertTensorData tensor_data = {(uint8_t *)device_data.data(), 0U, kTensorPlacementEnd, -1};

  auto run_context = BuildKernelRunContext(static_cast<size_t>(MixL2ArgsInKey::kNUM) + 5, static_cast<size_t>(MixL2ArgsOutKey::kNUM));
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::WORKSPACE)].Set(work_space_vector, nullptr);
  uint32_t need_mode = 1;
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::NEED_MODE_ADDR)]
      .Set(reinterpret_cast<void *>(need_mode), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::SINK_RET)].Set(&sink_ret, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_ADDR_NUM)]
      .Set(reinterpret_cast<void *>(io_addr_num), nullptr);
  uint32_t need_assert = 1;
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::HAS_ASSERT)].Set(reinterpret_cast<void *>(need_assert), nullptr);
  gert::StorageShape in_shape = {{5,2,3,4}, {5, 1, 1, 1, 1}};
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START)].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 1].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 2].Set(&in_shape, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 3].Set(&tensor_data, nullptr);
  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};

  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 4].Set(&in_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 5].Set(&out_data, nullptr);

  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc{};
  node_desc.max_tiling_data = 64;
  node_desc.max_tail_tiling_data = 63;
  node_desc.input_num = 2;
  node_desc.output_num = 1;
  node_desc.addr_num = 3;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  size_t size = 0;
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(node, node_desc, size);

  NodeMemPara node_para;
  node_para.size = size;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);

  run_context.value_holder[static_cast<size_t>(MixL2ArgsInKey::IO_START) + 7].Set(&node_para, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->outputs_creator(nullptr, run_context),
  ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->run_func(run_context),
          ge::GRAPH_SUCCESS);
  EXPECT_FALSE(registry.FindKernelFuncs("FFTSUpdateMixL2Args")->trace_printer(run_context).empty());
  delete[] mem_1;
  delete[] mem_2;

  // profiling
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 8;
  run_context.value_holder[3].Set(&task_inf, nullptr);
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  size_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set((void*)ctx_id, nullptr);
  ProfNodeAdditionInfo info;
  CannProfilingInfoWrapper prof_info(&info);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->profiling_info_filler(run_context, prof_info),
            ge::GRAPH_SUCCESS);
  ASSERT_EQ(prof_info.add_infos_[0]->node_basic_info.data.nodeBasicInfo.blockDim, 131076);  // 0x0204
}

TEST_F(MixL2LoweringST, test_MixL2UpdateGeDataDumpInfo) {
  auto context = BuildKernelRunContext(static_cast<size_t>(MixL2DataDumpKey::RESERVED) + 3, 0);
  uint32_t ctxid = 1;
  uint32_t in_num = 1;
  uint32_t out_num = 1;
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::CONTEXT_ID)]
      .Set(reinterpret_cast<void *>(ctxid), nullptr);
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::IN_NUM)].Set(reinterpret_cast<void *>(in_num), nullptr);
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::OUT_NUM)].Set(reinterpret_cast<void *>(out_num), nullptr);

  auto device_data1 = std::vector<int8_t>{10};
  GertTensorData in_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};
  GertTensorData out_data = {(uint8_t *)device_data1.data(), 0U, kTensorPlacementEnd, -1};

  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::IO_START)].Set(&in_data, nullptr);
  context.value_holder[static_cast<size_t>(MixL2DataDumpKey::IO_START) + 1].Set(&out_data, nullptr);

  ASSERT_NE(registry.FindKernelFuncs("MixL2UpdateDataDumpInfo"), nullptr);
  NodeDumpUnit dump_unit;
  gert::ExecutorDataDumpInfoWrapper wrapper(&dump_unit);
  auto ret = registry.FindKernelFuncs("MixL2UpdateDataDumpInfo")->data_dump_info_filler(context, wrapper);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateDataDumpInfo")->run_func(context),
      ge::GRAPH_SUCCESS);
}

TEST_F(MixL2LoweringST, test_MixL2UpdateGeExceptionDumpInfo) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NodeGraph(root_graph, node, false);
  auto context = BuildKernelRunContext(static_cast<size_t>(MixL2ExceptionDumpKey::RESERVED), 0);
  size_t ws_size = 2;
  auto work_space = ContinuousVector::Create<GertTensorData *>(ws_size);
  auto work_space_vector = reinterpret_cast<ContinuousVector *>(work_space.get());
  work_space_vector->SetSize(ws_size);
  auto work_space_ptr = reinterpret_cast<GertTensorData **>(work_space_vector->MutableData());
  constexpr uint64_t stub_mem_hbm_addr = 0x22;
  memory::CachingMemAllocator stub_allocator{0, 1};
  memory::SingleStreamL2Allocator single_stream_l2_allocator{&stub_allocator};
  ge::MemBlock mem_block(stub_allocator, reinterpret_cast<void *>(stub_mem_hbm_addr), 100);
  memory::MultiStreamMemBlock ms_block;
  ms_block.ReInit(&single_stream_l2_allocator, &mem_block, memory::BlockAllocType(memory::BlockAllocType::kNorm, 0U));
  GertTensorData work_space_gtd =
      GertTensorData(ms_block.GetAddr(), ms_block.GetSize(), TensorPlacement::kOnDeviceHbm, 0);
  for (size_t i = 0; i < work_space_vector->GetSize(); i++) {
    work_space_ptr[i] = &work_space_gtd;
  }
  context.value_holder[static_cast<size_t>(MixL2ExceptionDumpKey::WORKSPACE)].Set(work_space_vector, nullptr);

  RtFFTSKernelLaunchArgs::ComputeNodeDesc node_desc;
  node_desc.max_tiling_data = 63;
  node_desc.max_tail_tiling_data = 0;
  node_desc.addr_num = 0;
  node_desc.input_num = 2;
  node_desc.output_num = 1;
  node_desc.workspace_cap = 8;
  node_desc.thread_num_max = 1;
  size_t size = 0;
  auto rt_arg = RtFFTSKernelLaunchArgs::Create(node, node_desc, size);

  NodeMemPara node_para;
  char *mem_1 = new char[size];
  char *mem_2 = new char[size];
  node_para.host_addr = (void*)mem_1;
  node_para.dev_addr = (void*)mem_2;
  (void)memcpy_s(node_para.host_addr, node_para.size, rt_arg.get(), node_para.size);
  context.value_holder[static_cast<size_t>(MixL2ExceptionDumpKey::ARGS_PARA)].Set(&node_para, nullptr);

  ExceptionDumpUint dump_unit;
  ExecutorExceptionDumpInfoWrapper wrapper(&dump_unit);
  ASSERT_NE(registry.FindKernelFuncs("MixL2UpdateExceptionDumpInfo"), nullptr);
  auto ret = registry.FindKernelFuncs("MixL2UpdateExceptionDumpInfo")->exception_dump_info_filler(context, wrapper);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateExceptionDumpInfo")->run_func(context),
      ge::GRAPH_SUCCESS);
  delete[] mem_1;
  delete[] mem_2;
}

TEST_F(MixL2LoweringST, lowering_mix_l2_2_kernel_dump_success) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  TestMixL2Lowering(root_graph, global_data, true);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}

TEST_F(MixL2LoweringST, test_mix_ratio_update_context) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 1;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 2;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 1;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)].Set((void*)is_mix_ratio, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MixL2LoweringST, test_mix_ratio_update_context_2) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 1;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 0;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 1;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)].Set((void*)is_mix_ratio, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MixL2LoweringST, test_mix_ratio_update_context_3) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 1;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 1;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 0;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)].Set((void*)is_mix_ratio, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_SUCCESS);
}

TEST_F(MixL2LoweringST, test_mix_ratio_update_context_failed) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 0;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(1);
  auto tilings_ptr = reinterpret_cast<uint64_t *>(tiling_vec->MutableData());
  tilings_ptr[0] = 1;
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(1);
  auto cration_ptr = reinterpret_cast<int32_t *>(cration_vec->MutableData());
  cration_ptr[0] = 2;
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(1);
  auto vration_ptr = reinterpret_cast<int32_t *>(vration_vec->MutableData());
  vration_ptr[0] = 1;
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)].Set((void*)is_mix_ratio, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(MixL2LoweringST, test_mix_ratio_update_context_failed_02) {
  AICoreSubTaskFlush flush_data{};
  flush_data.input_addr_vv[0][0] = 0x1000;
  flush_data.input_addr_vv[0][1] = 0x1001;
  flush_data.output_addr_vv[0][0] = 0x2000;
  flush_data.output_addr_vv[0][1] = 0x2001;
  auto run_context = KernelRunContextFaker().KernelIONum(static_cast<size_t>(MixL2UpdateKey::RESERVED), 1).
                     NodeIoNum(2,2).IrInputNum(2)
      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NC1HWC0)
                         .Build();
  uint32_t blk_dim = 4;
  uint32_t schem = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::FLUSH_DATA)].Set(&flush_data, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::BLOCK_DIM)]
      .Set(reinterpret_cast<void *>(blk_dim), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::SCHEDULE_MODE)]
      .Set(reinterpret_cast<void *>(schem), nullptr);
  uint32_t ctx_id = 0;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CTX_ID)].Set(reinterpret_cast<void *>(ctx_id), nullptr);
  uint16_t tiling_key = 0;
  auto tilings = ContinuousVector::Create<uint64_t>(1);
  auto tiling_vec = reinterpret_cast<ContinuousVector *>(tilings.get());
  tiling_vec->SetSize(0);
  auto crations = ContinuousVector::Create<int32_t>(1);
  auto cration_vec = reinterpret_cast<ContinuousVector *>(crations.get());
  cration_vec->SetSize(0);
  auto vrations = ContinuousVector::Create<int32_t>(1);
  auto vration_vec = reinterpret_cast<ContinuousVector *>(vrations.get());
  vration_vec->SetSize(0);
  bool is_mix_ratio = true;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::MIX_RATIO)].Set((void*)is_mix_ratio, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_KEY)]
      .Set(reinterpret_cast<void *>(tiling_key), nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TILING_VEC)].Set(tiling_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::CRATIO_VEC)].Set(cration_vec, nullptr);
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::VRATIO_VEC)].Set(vration_vec, nullptr);
  size_t descBufLen = sizeof(rtFftsPlusComCtx_t) * static_cast<size_t>(1);
  size_t total_size = sizeof(TransTaskInfo) + descBufLen + sizeof(rtFftsPlusSqe_t) ;
  auto holder = ge::MakeUnique<uint8_t[]>(total_size);
  TransTaskInfo *task_info_ptr = reinterpret_cast<TransTaskInfo*>(holder.get());
  size_t buf_offset = sizeof(rtFftsPlusSqe_t);
  task_info_ptr->offsets[static_cast<size_t>(InfoStType::kDescBuf)] = buf_offset;
  task_info_ptr->rt_task_info.descBufLen = descBufLen;
  auto *buff_ptr = &task_info_ptr->args[buf_offset];
  auto context = reinterpret_cast<rtFftsPlusMixAicAivCtx_t*>(buff_ptr);
  context->contextType = RT_CTX_TYPE_MIX_AIC;
  context->nonTailBlockdim = 4;
  context->nonTailBlockRatioN = 2;
  rtFftsPlusTaskInfo_t task_inf;
  auto *const ffts_plus_sqe = ge::PtrToPtr<uint8_t, rtFftsPlusSqe_t>(task_info_ptr->args);
  task_inf.fftsPlusSqe = ffts_plus_sqe;
  task_inf.descBuf = &task_info_ptr->args[buf_offset];
  ffts_plus_sqe->totalContextNum = 1;
  run_context.value_holder[static_cast<size_t>(MixL2UpdateKey::TASK_INFO)].Set(&task_inf, nullptr);
  ASSERT_EQ(registry.FindKernelFuncs("MixL2UpdateContext")->run_func(run_context), ge::GRAPH_FAILED);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_mix_ratio_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NewDynNodeGraph(root_graph, node);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  (void)ctx_def_0.mutable_label_ctx();

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(IFA_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<std::vector<int64_t>> dyn_in_vv = {{2,3}, {4,5,6}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 10);

  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_mix_dynamic_ratio", true);
  ge::GeAttrValue::NAMED_ATTRS tiling_with_ratio;
  std::vector<std::string> tiling_key_vec = {"0", "1", "2"};
  std::vector<int64_t> c_ratio_vec = {0, 1, 2};
  std::vector<int64_t> v_ratio_vec = {1, 1, 1};
  tiling_with_ratio.SetAttr("mix_tiling_key", ge::GeAttrValue::CreateFrom<std::vector<std::string>>(tiling_key_vec));
  tiling_with_ratio.SetAttr("mix_tiling_c_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(c_ratio_vec));
  tiling_with_ratio.SetAttr("mix_tiling_v_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(v_ratio_vec));
  (void)ge::AttrUtils::SetNamedAttrs(node->GetOpDesc(), "mix_tiling_with_ratio_attr", tiling_with_ratio);
  NodeUtils::AppendInputAnchor(node, 3);
  TestMixL2Lowering(root_graph, global_data, true);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_mix_ratio_kernel_failed) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  BuildMixL2NewDynNodeGraph(root_graph, node);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  (void)ctx_def_0.mutable_label_ctx();

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(IFA_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kAttrDynamicParamMode, kFoldedWithDesc);
  std::vector<std::vector<int64_t>> dyn_in_vv = {{2,3}, {4,5,6}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicInputsIndexes, dyn_in_vv);
  std::vector<std::vector<int64_t>> dyn_out_vv = {{1}};
  (void)ge::AttrUtils::SetListListInt(node->GetOpDesc(), kDynamicOutputsIndexes, dyn_out_vv);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 10);

  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), "_mix_dynamic_ratio", true);
  ge::GeAttrValue::NAMED_ATTRS tiling_with_ratio;
  std::vector<std::string> tiling_key_vec = {};
  std::vector<int64_t> c_ratio_vec = {};
  std::vector<int64_t> v_ratio_vec = {};
  tiling_with_ratio.SetAttr("mix_tiling_key", ge::GeAttrValue::CreateFrom<std::vector<std::string>>(tiling_key_vec));
  tiling_with_ratio.SetAttr("mix_tiling_c_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(c_ratio_vec));
  tiling_with_ratio.SetAttr("mix_tiling_v_ratio", ge::GeAttrValue::CreateFrom<std::vector<int64_t>>(v_ratio_vec));
  (void)ge::AttrUtils::SetNamedAttrs(node->GetOpDesc(), "mix_tiling_with_ratio_attr", tiling_with_ratio);
  NodeUtils::AppendInputAnchor(node, 3);
  TestMixL2Lowering(root_graph, global_data, false);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_with_output_optional_success) {
gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 7);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalOutputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllOutputSize, 2);
  auto op_desc = node->GetOpDesc();
  auto output_desc_ptr = op_desc->MutableOutputDesc(1);
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                              static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  NodeUtils::AppendInputAnchor(node, 4);
  TestMixL2Lowering(root_graph, global_data, true);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_with_output_optional_anchor_null_success) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 7);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalOutputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllOutputSize, 2);
  auto op_desc = node->GetOpDesc();
  auto output_desc_ptr = op_desc->MutableOutputDesc(1);
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                              static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  NodeUtils::AppendInputAnchor(node, 4);
  const auto &out_data_anchor = node->GetOutDataAnchor(0);
  for (const auto &in_data_anchor : out_data_anchor->GetPeerInDataAnchors()) {
      RemoveAnchor(out_data_anchor, in_data_anchor);
  }
  TestMixL2Lowering(root_graph, global_data, true);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}

TEST_F(MixL2LoweringST, lowering_mix_l2_with_output_optional_failed) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kExceptionDump}));
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NodeGraph(root_graph, node, false);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(2);

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalInputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllInputSize, 7);
  (void)ge::AttrUtils::SetStr(node->GetOpDesc(), kOptionalOutputMode, kGenPlaceHolder);
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), kOpKernelAllOutputSize, 0);
  auto op_desc = node->GetOpDesc();
  auto output_desc_ptr = op_desc->MutableOutputDesc(1);
  (void)ge::AttrUtils::SetInt(output_desc_ptr, ge::ATTR_NAME_MEMORY_SIZE_CALC_TYPE,
                              static_cast<int64_t>(ge::MemorySizeCalcType::ALWAYS_EMPTY));
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  NodeUtils::AppendInputAnchor(node, 4);
  TestMixL2Lowering(root_graph, global_data, false);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}
}  // namespace gert
