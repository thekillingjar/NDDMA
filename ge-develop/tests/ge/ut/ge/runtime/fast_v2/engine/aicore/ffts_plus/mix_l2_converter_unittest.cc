/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "engine/aicore/converter/aicore_ffts_node_converter.h"
#include "lowering/graph_converter.h"
#include <gtest/gtest.h>
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "graph/utils/graph_utils.h"
#include "graph_builder/exe_graph_comparer.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "common/topo_checker.h"
#include "common/sgt_slice_type.h"
#include "register/op_impl_registry.h"
#include "lowering/graph_converter.h"
#include "aicore/converter/bg_kernel_launch.h"
#include "op_tiling/op_tiling_constants.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "graph_builder/bg_tiling.h"
#include "runtime/subscriber/global_dumper.h"
#include "register/op_tiling_info.h"
#include "register/op_tiling/op_tiling_constants.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace ge;
using namespace gert::bg;

const std::string kAtomicCtxIdList = "_atomic_context_id_list";
namespace gert {
class MixL2LoweringUT : public testing::Test {
 public:
  void TearDown() override {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
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
  SetGraphOutShapeRange(root_graph);
  node = root_graph->FindNode("Conv2D_Sqrt");
  auto conv2d_desc = node->GetOpDesc();
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
  int64_t atom_max_size = 512;
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxAtomicCleanTilingSize, atom_max_size);
  std::string op_compile_info_json = R"({"_workspace_size_list":[32], "vars":{"ub_size": 12, "core_num": 2}})";
  ge::AttrUtils::SetStr(conv2d_desc, optiling::ATOMIC_COMPILE_INFO_JSON, op_compile_info_json);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "FFTS_BINARY_MAGIC_ELF_MIX_AIC");
  AttrUtils::SetBool(conv2d_desc, "support_dynamicshape", true);
  AttrUtils::SetInt(conv2d_desc, "op_para_size", 512);
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
  AttrUtils::SetInt(conv2d_desc, "op_para_size", 512);
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

  for (auto in_desc : ifa_desc->GetAllOutputsDesc()) {
    in_desc.SetShape(GeShape({4, -1, -1, 4}));
  }

  auto out_desc = node->GetOpDesc()->MutableOutputDesc(0);
  ASSERT_NE(out_desc, nullptr);
  out_desc->SetShape(GeShape({4, 4, 4, 4}));

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
  AttrUtils::SetInt(ifa_desc, "op_para_size", 512);
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

TEST_F(MixL2LoweringUT, lowering_mix_l2_1_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NodeGraph(root_graph, node, true);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  ctx_def_0.mutable_mix_aic_aiv_ctx()->set_args_format("{i_instance0}{i_instance1}{o_instance0}");

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);
  // third op
  int32_t unknown_shape_type_val = ge::DEPEND_SHAPE_RANGE;
  (void) ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  TestMixL2Lowering(root_graph, global_data, true);

  node->GetOpDesc()->DelAttr(ge::kAttrLowingFunc);
  auto exe_graph = GraphConverter().ConvertComputeGraphToExecuteGraph(root_graph, global_data);
  ASSERT_EQ(exe_graph, nullptr);
}
TEST_F(MixL2LoweringUT, lowering_mix_l2_2_kernel_success) {
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
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  TestMixL2Lowering(root_graph, global_data, true);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}

TEST_F(MixL2LoweringUT, lowering_mix_l2_with_optional_success) {
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
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  NodeUtils::AppendInputAnchor(node, 4);
  TestMixL2Lowering(root_graph, global_data, true);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}

TEST_F(MixL2LoweringUT, lowering_static_mix_l2_2_kernel_success) {
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
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, false);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  TestMixL2Lowering(root_graph, global_data, true);
}

TEST_F(MixL2LoweringUT, lowering_static_mix_l2_reuse_binary_kernel_success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
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

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, false);
  std::vector<uint32_t> atomic_ctx_id_vec = {0};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kAtomicCtxIdList, atomic_ctx_id_vec);
  TestMixL2Lowering(root_graph, global_data, true);
}

TEST_F(MixL2LoweringUT, lowering_static_mix_l2_new_kernel_success) {
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

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, false);
  TestMixL2Lowering(root_graph, global_data, true);
}

/**
 * 用例描述： 校验Conv2D_Sqrt的ub子图的计算节点对应的执行节点的优先级为Conv2D_Sqrt的id * 10 + 9
 * todo dead loop? :561
 */
TEST_F(MixL2LoweringUT, Node_With_Original_Fusion_Graph_Set_Priority_Success) {
  ComputeGraphPtr root_graph;
  ge::NodePtr node = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildMixL2NodeGraph(root_graph, node, true);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef *ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);
  // third op
  int32_t unknown_shape_type_val = ge::DEPEND_SHAPE_RANGE;
  (void)ge::AttrUtils::SetInt(node->GetOpDesc(), ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, unknown_shape_type_val);

  LoweringGlobalData global_data;
  global_data.AddCompiledResult(node, {{task_def}});
    auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  IMPL_OP(CONV2D_T);
  IMPL_OP(SQRT_T);
  (void)ge::AttrUtils::SetBool(node->GetOpDesc(), kUnknownShapeFromFe, true);
  ASSERT_NE(root_graph, nullptr);
  root_graph->TopologicalSorting();
  (void)bg::ValueHolder::PopGraphFrame();
  ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
  model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
  auto exe_graph = GraphConverter()
      .SetModelDescHolder(&model_desc_holder)
      .ConvertComputeGraphToExecuteGraph(root_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  ge::FastNode* conv2d_compatible_infershape_node = nullptr;
  ge::FastNode* conv2d_sqrt_launch_node = nullptr;
  for (const auto node : exe_graph->GetAllNodes()) {
    if (node != nullptr && node->GetName().find("CompatibleInferShape_conv2d") != std::string::npos) {
      conv2d_compatible_infershape_node = node;
    }
    if (node != nullptr && node->GetType() == "LaunchFFTSPlusTaskNoCopy") {
      conv2d_sqrt_launch_node = node;
    }
  }
  ASSERT_NE(conv2d_compatible_infershape_node, nullptr);
  ASSERT_NE(conv2d_sqrt_launch_node, nullptr);
  ge::NodePtr conv2d_sqrt_node = nullptr;
  for (const auto &node : root_graph->GetAllNodes()) {
    if (node != nullptr && node->GetName() == "Conv2D_Sqrt") {
      conv2d_sqrt_node = node;
      break;
    }
  }
  ASSERT_NE(conv2d_sqrt_node, nullptr);
  int64_t compatible_infershape_node_priority = -1;
  (void)ge::AttrUtils::GetInt(conv2d_compatible_infershape_node->GetOpDescBarePtr(), "priority",
                              compatible_infershape_node_priority);
  int64_t conv2d_sqrt_launch_node_priority = -1;
  (void)ge::AttrUtils::GetInt(conv2d_sqrt_launch_node->GetOpDescBarePtr(), "priority", conv2d_sqrt_launch_node_priority);
  ASSERT_NE(conv2d_sqrt_node->GetOpDesc(), nullptr);
  const int64_t kPriorityExpansion = 10;
  const int64_t expect_priority =
      (conv2d_sqrt_node->GetOpDesc()->GetId() * kPriorityExpansion) + (kPriorityExpansion - 1);
  ASSERT_EQ(compatible_infershape_node_priority, expect_priority);
  ASSERT_EQ(conv2d_sqrt_launch_node_priority, expect_priority);
  ASSERT_NE(exe_graph, nullptr);
}

TEST_F(MixL2LoweringUT, lowering_mix_l2_dynamic_new_kernel_success) {
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

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  LoweringGlobalData global_data;
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
  std::vector<uint32_t> insert_pos_vec = {7, 8, 9};
  (void)ge::AttrUtils::SetListInt(node->GetOpDesc(), kInputInsertOptPosList, insert_pos_vec);
  TestMixL2Lowering(root_graph, global_data, true);
}

TEST_F(MixL2LoweringUT, lowering_mix_l2_mix_ratio_kernel_success) {
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

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  LoweringGlobalData global_data;
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

TEST_F(MixL2LoweringUT, lowering_mix_l2_mix_ratio_kernel_failed) {
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

  uint32_t data_type = 0;
  domi::AdditionalDataDef *additional_data_def = ffts_plus_task_def.add_additional_data();
  additional_data_def->set_data_type(data_type);
  additional_data_def->add_context_id(0);

  domi::FftsPlusSqeDef* ffts_plus_sqe = ffts_plus_task_def.mutable_ffts_plus_sqe();
  ffts_plus_sqe->set_ready_context_num(1);
  ffts_plus_sqe->set_total_context_num(1);

  LoweringGlobalData global_data;
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

TEST_F(MixL2LoweringUT, lowering_mix_l2_with_output_optional_success) {
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

TEST_F(MixL2LoweringUT, lowering_mix_l2_with_output_optional_anchor_null_success) {
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

TEST_F(MixL2LoweringUT, lowering_mix_l2_with_output_optional_failed) {
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
