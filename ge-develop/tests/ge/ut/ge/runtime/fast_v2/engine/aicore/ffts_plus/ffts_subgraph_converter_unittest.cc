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
#include "engine/ffts_plus/converter/ffts_plus_common.h"
#include "engine/aicore/converter/aicore_ffts_node_converter.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "engine/gelocal/inputs_converter.h"
#include "engine/node_converter_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "engine/ffts_plus/converter/ffts_plus_proto_transfer.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
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
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "runtime/subscriber/global_dumper.h"
#include "graph/utils/graph_dump_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace ge;
using namespace gert::bg;

namespace gert {

const std::string kFFTSAiCoreLowerFuncTest = "ffts_ai_core_lower_func_test";
LowerResult LoweringFFTSAiCoreNodeTest(const ge::NodePtr &node, const LowerInput &lower_input) {
  return {HyperStatus::ErrorStatus(static_cast<const char *>("TEST failed.")), {}, {}, {}};
}
REGISTER_NODE_CONVERTER_PLACEMENT(kFFTSAiCoreLowerFuncTest.c_str(), kOnDeviceHbm, LoweringFFTSAiCoreNodeTest);

class FFTSGraphLoweringUT : public testing::Test {
 public:
  void SetUp() override {
  }
  void TearDown() override {
    while (bg::ValueHolder::PopGraphFrame() != nullptr) {
    }
  }
  void TestFFTSSingleLowering(ComputeGraphPtr &graph, LoweringGlobalData &global_data, bool expect) {
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
    ge::DumpGraph(exe_graph.get(), "LoweringSingleFFTSGraph");
  }
  void TestFFTSLowering(ComputeGraphPtr &graph, LoweringGlobalData &global_data) {
    ASSERT_NE(graph, nullptr);
    graph->TopologicalSorting();
    (void)bg::ValueHolder::PopGraphFrame();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
    auto exe_graph = GraphConverter()
        .SetModelDescHolder(&model_desc_holder)
        .ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_NE(exe_graph, nullptr);
    ge::DumpGraph(exe_graph.get(), "LoweringFFTSGraph");
  }
  void TestFFTSSingleLoweringFail(ComputeGraphPtr &graph, LoweringGlobalData &global_data) {
    ASSERT_NE(graph, nullptr);
    (void)bg::ValueHolder::PopGraphFrame();
    ModelDescHolder model_desc_holder = ModelDescHolderFaker().Build();
    model_desc_holder.SetSpaceRegistry(SpaceRegistryFaker().Build());
    auto exe_graph = GraphConverter()
        .SetModelDescHolder(&model_desc_holder)
        .ConvertComputeGraphToExecuteGraph(graph, global_data);
    ASSERT_EQ(exe_graph, nullptr);
  }
};

static void BuildFftsPlusSingleOpGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &ffts_plus_graph,
                                       TBEKernelStore *kernel_store = nullptr) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
  };
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);

  AttrUtils::SetStr(root_graph->FindNode("PartitionedCall_0")->GetOpDesc(), ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH,
                    "ffts_plus");
  AttrUtils::SetInt(root_graph->FindNode("_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);

  DEF_GRAPH(g2) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto conv_0 = OP_CFG(CONV2D)
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("sgt_graph/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT));
    CHAIN(NODE("sgt_graph/_arg_1", data_1)->EDGE(0, 1)->NODE("sgt_graph/Conv2D", conv_0));
  };
  ffts_plus_graph = ToComputeGraph(g2);
  auto conv2d_desc = ffts_plus_graph->FindNode("sgt_graph/Conv2D")->GetOpDesc();
  AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_1")->GetOpDesc(), "index", 1);
  conv2d_desc->SetOpKernelLibName(ge::kEngineNameAiCore);
  (void)ge::AttrUtils::SetBool(conv2d_desc, kUnknownShapeFromFe, true);
  std::vector<int32_t> unknown_axis = {0};
  ge::AttrUtils::SetListInt(conv2d_desc, "unknown_axis_index", unknown_axis);
  int64_t param_k = 0;
  int64_t param_b = 0;
  (void)ge::AttrUtils::SetInt(conv2d_desc, "param_k", param_k);
  (void)ge::AttrUtils::SetInt(conv2d_desc, "param_b", param_b);

  std::vector<uint32_t> input_tensor_indexes;
  std::vector<uint32_t> output_tensor_indexes;
  (void)ge::AttrUtils::SetListInt(conv2d_desc, ge::kInputTensorIndexs, input_tensor_indexes);
  (void)ge::AttrUtils::SetListInt(conv2d_desc, ge::kOutputTensorIndexs, output_tensor_indexes);
  (void)ge::AttrUtils::SetInt(conv2d_desc, bg::kMaxTilingSize, 50);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("sgt/test", std::move(test_bin));
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(conv2d_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  conv2d_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  conv2d_desc->SetExtAttr(EXT_ATTR_ATOMIC_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::ATOMIC_ATTR_TVM_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  AttrUtils::SetBool(conv2d_desc, "_kernel_list_first_name", true);

  ffts_plus_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(ffts_plus_graph, mem_offset);
  AddPartitionedCall(root_graph, "PartitionedCall_0", ffts_plus_graph);
}

extern ge::graphStatus LoadSgtKernelBinToOpDesc(const ge::NodePtr &node, const ge::ComputeGraphPtr &graph,
                                                 const ge::GeModelPtr &ge_model, const ModelTaskType task_type);
TEST_F(FFTSGraphLoweringUT, load_ffts_bin_success) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusSingleOpGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  ge::GeModelPtr ge_model = std::make_shared<ge::GeModel>();
  auto part_node = root_graph->FindNode("PartitionedCall_0");
  auto ret = LoadSgtKernelBinToOpDesc(part_node, root_graph, ge_model, ModelTaskType::MODEL_TASK_FFTS_PLUS);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(FFTSGraphLoweringUT, ffts_plus_lowering_single_fail) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusSingleOpGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  auto part_node = root_graph->FindNode("PartitionedCall_0");
  (void)ge::AttrUtils::SetStr(part_node->GetOpDesc(), ge::kAttrLowingFunc, ge::kFFTSGraphLowerFunc);

  auto conv_node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
  conv_node->GetOpDesc()->SetOpKernelLibName(ge::kEngineNameDvpp);
  ASSERT_NE(root_graph, nullptr);
  root_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(part_node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);
  TestFFTSSingleLowering(root_graph, global_data, false);
}

TEST_F(FFTSGraphLoweringUT, ffts_plus_lowering_single_failed) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusSingleOpGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  auto part_node = root_graph->FindNode("PartitionedCall_0");
  (void)ge::AttrUtils::SetStr(part_node->GetOpDesc(), ge::kAttrLowingFunc, ge::kFFTSGraphLowerFunc);
  ASSERT_NE(root_graph, nullptr);
  root_graph->TopologicalSorting();
  auto root_model = GeModelBuilder(root_graph).BuildGeRootModel();
  LoweringGlobalData global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(part_node, {{task_def}});
  auto space_registry_array = OpImplSpaceRegistryV2Array();
  space_registry_array[static_cast<size_t>(gert::OppImplVersionTag::kOpp)] = SpaceRegistryFaker().Build();
  global_data.SetSpaceRegistriesV2(space_registry_array);

  auto conv2d_desc = ffts_plus_graph->FindNode("sgt_graph/Conv2D")->GetOpDesc();
  conv2d_desc->SetOpKernelLibName(ge::kEngineNameDvpp);
  TestFFTSSingleLoweringFail(root_graph, global_data);

  global_data = GlobalDataFaker(root_model).Build();
  global_data.AddCompiledResult(part_node, {{task_def}});
  global_data.SetSpaceRegistriesV2(space_registry_array);
  (void)ge::AttrUtils::SetStr(conv2d_desc, ge::kAttrLowingFunc, kFFTSAiCoreLowerFuncTest);
  TestFFTSSingleLoweringFail(root_graph, global_data);
}

TEST_F(FFTSGraphLoweringUT, ffts_plus_proto_trans_test) {
  ComputeGraphPtr ffts_plus_graph = nullptr;
  ComputeGraphPtr root_graph = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusSingleOpGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto part_node = root_graph->FindNode("PartitionedCall_0");
  auto part_desc = part_node->GetOpDesc();

  auto &ctx_def_0 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_0.set_context_id(0);
  ctx_def_0.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICORE));
  auto aic_ctx_def = ctx_def_0.mutable_aic_aiv_ctx();
  aic_ctx_def->set_atm(0);
  aic_ctx_def->set_thread_dim(0);
  aic_ctx_def->add_successor_list(1);

  auto &ctx_def_1 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_1.set_context_id(1);
  ctx_def_1.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_PERSISTENT_CACHE));
  auto cache_ctx_def = ctx_def_1.mutable_cache_persist_ctx();
  cache_ctx_def->add_successor_list(2);

  auto &ctx_def_2 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_2.set_context_id(2);
  ctx_def_2.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_NOTIFY_WAIT));
  auto notify_ctx_def = ctx_def_2.mutable_notify_ctx();
  notify_ctx_def->set_atm(1);
  notify_ctx_def->add_successor_list(2);

  auto &ctx_def_3 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_3.set_context_id(3);
  ctx_def_3.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_SDMA));
  auto sdma_ctx_def = ctx_def_3.mutable_sdma_ctx();
  sdma_ctx_def->set_atm(1);
  sdma_ctx_def->add_successor_list(2);

  auto &ctx_def_4 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_4.set_context_id(4);
  ctx_def_4.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_WRITE_VALUE));
  auto write_ctx_def = ctx_def_4.mutable_write_value_ctx();
  write_ctx_def->set_atm(1);
  write_ctx_def->add_successor_list(2);

  auto &ctx_def_5 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_5.set_context_id(5);
  ctx_def_5.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  auto aicpu_ctx_def = ctx_def_5.mutable_aicpu_ctx();
  aicpu_ctx_def->set_atm(1);
  aicpu_ctx_def->set_thread_dim(0);
  aicpu_ctx_def->add_successor_list(2);

  auto &ctx_def_6 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_6.set_context_id(6);
  ctx_def_6.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_FLUSH_DATA));
  auto data_ctx_def = ctx_def_6.mutable_data_ctx();
  data_ctx_def->set_atm(1);
  data_ctx_def->add_successor_list(2);

  auto &ctx_def_7 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_7.set_context_id(7);
  ctx_def_7.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_START));
  (void)ctx_def_7.mutable_at_start_ctx();

  auto &ctx_def_8 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_8.set_context_id(8);
  ctx_def_8.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_END));
  (void)ctx_def_8.mutable_at_end_ctx();

  auto &ctx_def_9 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_9.set_context_id(9);
  ctx_def_9.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_LABEL));
  (void)ctx_def_9.mutable_label_ctx();

  auto &ctx_def_10 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_10.set_context_id(10);
  ctx_def_10.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_CASE_SWITCH));
  (void)ctx_def_10.mutable_case_switch_ctx();

  auto &ctx_def_11 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_11.set_context_id(11);
  ctx_def_11.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  auto mix_ctx_def = ctx_def_11.mutable_mix_aic_aiv_ctx();
  mix_ctx_def->set_atm(1);
  mix_ctx_def->set_thread_dim(0);
  mix_ctx_def->add_successor_list(2);

  auto &ctx_def_12 = *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_12.set_context_id(12);
  ctx_def_12.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_COND_SWITCH));

  auto conv_node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
  auto conv_desc = conv_node->GetOpDesc();
  const auto find_node_handle = [&conv_desc](const uint32_t op_index) -> ge::OpDescPtr {
    return conv_desc;
  };
  std::vector<uintptr_t> io_addrs;
  std::vector<size_t> mode_addr_idx;
  const ge::RuntimeParam runtime_param;
  std::vector<void *> ext_args;
  FftsPlusProtoTransfer ffts_proto_transfer(0U, io_addrs, runtime_param, ext_args, mode_addr_idx);
  ffts_proto_transfer.SetFindNodeHandle(find_node_handle);
  size_t total_size = 0;
  auto trans_task_info = ffts_proto_transfer.Transfer(part_desc, ffts_plus_task_def, total_size);
  ASSERT_NE(trans_task_info, nullptr);
  aic_ctx_def->set_atm(0);
  aic_ctx_def->set_thread_dim(2);
  aic_ctx_def->add_kernel_name("kernel1");
  notify_ctx_def->set_atm(0);
  sdma_ctx_def->set_atm(0);
  write_ctx_def->set_atm(0);
  aicpu_ctx_def->set_atm(0);
  data_ctx_def->set_atm(0);
  mix_ctx_def->set_atm(0);
  mix_ctx_def->set_thread_dim(2);
  auto trans_task_info1 = ffts_proto_transfer.Transfer(part_desc, ffts_plus_task_def, total_size);
  ASSERT_NE(trans_task_info1, nullptr);

  aic_ctx_def->set_atm(1);
  aic_ctx_def->add_kernel_name("kernel2");
  mix_ctx_def->set_atm(1);
  mix_ctx_def->add_kernel_name("kernel1");
  mix_ctx_def->add_kernel_name("kernel2");
  mix_ctx_def->add_kernel_name("kernel3");
  mix_ctx_def->add_kernel_name("kernel4");
  auto trans_task_info2 = ffts_proto_transfer.Transfer(part_desc, ffts_plus_task_def, total_size);
  ASSERT_NE(trans_task_info2, nullptr);
}

TEST_F(FFTSGraphLoweringUT, ffts_plus_proto_trans_aicpu_test) {
  ComputeGraphPtr ffts_plus_graph = nullptr;
  ComputeGraphPtr root_graph = nullptr;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusSingleOpGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  auto conv_node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
  auto conv_desc = conv_node->GetOpDesc();
// Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();

  auto &case_defalut_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusCaseDefaultDef(ffts_plus_graph, case_defalut_ctx_def, "sgt_graph/Conv2D");

  string args(64, '1');
  auto &ctx_def_3= *ffts_plus_task_def.add_ffts_plus_ctx();
  ctx_def_3.set_context_id(2);
  ctx_def_3.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  auto aicpu_ctx_def3 = ctx_def_3.mutable_aicpu_ctx();
  aicpu_ctx_def3->set_atm(1);
  aicpu_ctx_def3->set_kernel_type(0);
  domi::aicpuKernelDef *kernel_def3 = aicpu_ctx_def3->mutable_kernel();
  kernel_def3->set_so_name("kernel_so_name");
  kernel_def3->set_kernel_name("funcName");
  kernel_def3->set_args(args.data(), 64);
  kernel_def3->set_args_size(64);

  const auto find_node_handle = [&conv_desc](const uint32_t op_index) -> ge::OpDescPtr {
    return conv_desc;
  };
  std::vector<uintptr_t> io_addrs;
  std::vector<size_t> mode_addr_idx;
  const ge::RuntimeParam runtime_param;
  std::vector<void *> ext_args;
  FftsPlusProtoTransfer ffts_proto_transfer(0U, io_addrs, runtime_param, ext_args, mode_addr_idx);
  ffts_proto_transfer.SetFindNodeHandle(find_node_handle);
  size_t total_size = 0;
  auto trans_task_info = ffts_proto_transfer.Transfer(conv_desc, ffts_plus_task_def, total_size);
  ASSERT_NE(trans_task_info, nullptr);
}
}  // namespace gert
