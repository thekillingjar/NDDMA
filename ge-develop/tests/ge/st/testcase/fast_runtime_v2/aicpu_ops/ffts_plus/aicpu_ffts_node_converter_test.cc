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

#include "framework/common/ge_inner_error_codes.h"
#include "engine/aicpu/converter/aicpu_ffts_node_converter.h"
#include "register/ffts_node_converter_registry.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "common/sgt_slice_type.h"
#include "engine/gelocal/inputs_converter.h"
#include "graph/utils/graph_utils.h"
#include "common/share_graph.h"
#include "faker/global_data_faker.h"
#include "common/bg_test.h"
#include "faker/aicpu_taskdef_faker.h"
#include "framework/common/ge_types.h"
#include "mmpa/mmpa_api.h"
#include "register/kernel_registry.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "graph_builder/bg_tiling.h"
#include "op_tiling/op_tiling_constants.h"
#include "graph_tuner/rt2_src/graph_tunner_rt2_stub.h"
#include "framework/runtime/gert_const_types.h"
#include "exe_graph/lowering/frame_selector.h"
#include "graph_builder/bg_memory.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/ffts_mem_allocator.h"

using namespace ge;
namespace gert {
namespace {
  const std::string kEngineNameAicpuFfts = "ffts_plus_aicpu_ascend";
  const std::string kEngineNameAicputfFfts = "ffts_plus_aicpu_tf";
  const std::string kFFTSAiCoreLowerFunc = "ffts_ai_core_lower_func";
}
class AicpuFFTSNodeConverterST : public bg::BgTestAutoCreate3StageFrame {};
void LowerConstDataNode(LoweringGlobalData &global_data) {
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
/***********************************************************************************************************************
 *
 * Data0            Data1
 *   \             /
 *    \           /
 *     \         /                                      Data0      Data1
 *      \       /                                          \        /
 *   PartitionedCall                                        \      /
 *          |                                                 Add
 *          |                                                  |
 *          |                                                  |
 *          |                                              NetOutput
 *          |                                                 
 *          |                                                
 *      NetOutput                                             
 *                                                           
 **********************************************************************************************************************/
static void BuildFftsPlusGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &ffts_plus_graph,
                               TBEKernelStore *kernel_store = nullptr) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
  };
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);

  AttrUtils::SetStr(root_graph->FindNode("PartitionedCall_0")->GetOpDesc(), ge::ATTR_NAME_FFTS_PLUS_SUB_GRAPH, "ffts_plus");
  auto data_0_desc = root_graph->FindNode("_arg_0")->GetOpDesc();
  AttrUtils::SetInt(data_0_desc, "index", 0);
  AttrUtils::SetInt(root_graph->FindNode("_arg_1")->GetOpDesc(), "index", 1);

  DEF_GRAPH(g2) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto add_0 = OP_CFG("ADD_T").Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
    CHAIN(NODE("sgt_graph/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Add", add_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT));
    CHAIN(NODE("sgt_graph/_arg_1", data_1)->EDGE(0, 1)->NODE("sgt_graph/Add", add_0));
  };
  ffts_plus_graph = ToComputeGraph(g2);
  auto add_desc = ffts_plus_graph->FindNode("sgt_graph/Add")->GetOpDesc();
  AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_0")->GetOpDesc(), "index", 0);
  AttrUtils::SetInt(ffts_plus_graph->FindNode("sgt_graph/_arg_1")->GetOpDesc(), "index", 1);

  std::vector<int32_t> unknown_axis = {0};
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();

  (void)add_desc->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetStr(add_desc, "_ge_attr_lowering_func", kEngineNameAicpuFfts);

  string compile_info_key = "compile_info_key";
  string compile_info_json = "{\"_workspace_size_list\":[]}";
  std::vector<char> test_bin(64, '\0');
  ge::TBEKernelPtr test_kernel = MakeShared<ge::OpKernelBin>("sgt/test", std::move(test_bin));
  (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(add_desc, optiling::COMPILE_INFO_JSON, compile_info_json);
  add_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, test_kernel);
  (void)ge::AttrUtils::SetStr(add_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AIVEC");
  AttrUtils::SetBool(add_desc, "_kernel_list_first_name", true);

  ffts_plus_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(ffts_plus_graph, mem_offset);
  AddPartitionedCall(root_graph, "PartitionedCall_0", ffts_plus_graph);
}

TEST_F(AicpuFFTSNodeConverterST, CalcAICpuCCArgsMemSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).Build();
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_args", "aicpu_args");
  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_ext_info", "aicpu_ext_info");
  size_t total_size = 0;
  size_t pre_data_size = 0;
  std::unique_ptr<uint8_t[]> pre_data_ptr(new uint8_t[10]);
  auto ret = CalcAICpuCCArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(pre_data_size, 0);
  ASSERT_EQ(total_size, 768);
}

TEST_F(AicpuFFTSNodeConverterST, CalcAICpuCCArgsMemAbnormal) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).Build();
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  size_t total_size = 0;
  size_t pre_data_size = 0;
  std::unique_ptr<uint8_t[]> pre_data_ptr(new uint8_t[10]);
  auto ret = CalcAICpuCCArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_FAILED);
  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_args", "aicpu_args");
  ret = CalcAICpuCCArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_FAILED); 
  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_ext_info", "aicpu_ext_info");
  ret = CalcAICpuCCArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(AicpuFFTSNodeConverterST, CalcAICpuTfArgsMemSuccess) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).Build();
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_args", "aicpu_args");
  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_ext_info", "aicpu_ext_info");
  size_t total_size = 0;
  size_t pre_data_size = 0;
  std::unique_ptr<uint8_t[]> pre_data_ptr(new uint8_t[10]);
  auto ret = CalcAICpuTfArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
  ASSERT_EQ(pre_data_size, 0);
  std::cout << "total_size" << std::endl;
  ASSERT_EQ(total_size, 1536);
}

TEST_F(AicpuFFTSNodeConverterST, CalcAICpuTfArgsMemAbnormal) {
  auto graph = ShareGraph::BuildSingleNodeGraph();
  auto add_op_desc = graph->FindNode("add1")->GetOpDesc();
  add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  auto global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker.SetNeedMemcpy(true)).Build();
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};

  auto data1_ret = LoweringDataNode(graph->FindNode("data1"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("data2"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  size_t total_size = 0;
  size_t pre_data_size = 0;
  std::unique_ptr<uint8_t[]> pre_data_ptr(new uint8_t[10]);
  auto ret = CalcAICpuTfArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_FAILED);
  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_args", "aicpu_args");
  ret = CalcAICpuTfArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_FAILED); 
  (void)ge::AttrUtils::SetStr(add_op_desc, "_aicpu_ffts_ext_info", "aicpu_ext_info");
  ret = CalcAICpuTfArgsMem(graph->FindNode("add1"), &global_data, total_size, pre_data_size, pre_data_ptr);
  ASSERT_EQ(ret, ge::GRAPH_SUCCESS);
}

void TestAicpuFFTSConvertNoRefNode(std::string node_type, LowerResult &add_ret) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic1_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic1_ctx_def, "sgt_graph/Add");

  auto part_node = root_graph->FindNode("PartitionedCall_0");
  (void)ge::AttrUtils::SetStr(part_node->GetOpDesc(), "_ge_attr_lowering_func", "ffts_graph_lower_func");
  LoweringGlobalData global_data;
  auto graph = ffts_plus_graph;
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker).Build();
  global_data.AddCompiledResult(part_node, {{task_def}});
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("sgt_graph/_arg_0"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("sgt_graph/_arg_1"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  rtFftsPlusTaskInfo_t task_info;
  bg::ValueHolderPtr ffts_task = bg::ValueHolder::CreateConst(&task_info, sizeof(rtFftsPlusTaskInfo_t));
  uint32_t val = 20U;
  bg::ValueHolderPtr thread_dim = bg::ValueHolder::CreateConst(&val, sizeof(val));
  std::vector<uint32_t> mem_pool_types = {0, 1};
  uint32_t window_size = 20U;
  auto window_size_holder = bg::ValueHolder::CreateConst(&window_size, sizeof(window_size));
  auto ffts_allocator = CreateFftsMemAllocator(window_size_holder, global_data);
  FFTSLowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                              {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                              mem_pool_types,
                              &global_data,
                              ffts_task,
                              thread_dim,
                              window_size_holder,
                              nullptr,
                              ffts_allocator,
                              tune::FFTSNodeThreadV2};
  auto add_op_desc = graph->FindNode("sgt_graph/Add")->GetOpDesc();
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  auto add_node = graph->FindNode("sgt_graph/Add");
  (void)add_node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_aicpu_ffts_args", "aicpu_args");
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_aicpu_ffts_ext_info", "aicpu_ext_info");
  if (node_type == "tf") {
    add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  } else if (node_type == "cc") {
    add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  }

  auto add_ret = LoweringFFTSAiCpuNode(graph->FindNode("sgt_graph/Add"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);
}

void TestAicpuFFTSConvertWithRefNode(std::string node_type, LowerResult &add_ret) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);
  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic1_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic1_ctx_def, "sgt_graph/Add");

  auto part_node = root_graph->FindNode("PartitionedCall_0");
  (void)ge::AttrUtils::SetStr(part_node->GetOpDesc(), "_ge_attr_lowering_func", "ffts_graph_lower_func");
  LoweringGlobalData global_data;
  auto graph = ffts_plus_graph;
  AiCpuCCTaskDefFaker aicpu_task_def_faker;
  auto root_model = GeModelBuilder(graph).BuildGeRootModel();
  global_data = GlobalDataFaker(root_model).AddTaskDef("Add", aicpu_task_def_faker).Build();
  global_data.AddCompiledResult(part_node, {{task_def}});
  LowerConstDataNode(global_data);
  LowerInput data_input = {{}, {}, &global_data};
  auto data1_ret = LoweringDataNode(graph->FindNode("sgt_graph/_arg_0"), data_input);
  auto data2_ret = LoweringDataNode(graph->FindNode("sgt_graph/_arg_1"), data_input);
  ASSERT_TRUE(data1_ret.result.IsSuccess());
  ASSERT_TRUE(data2_ret.result.IsSuccess());

  rtFftsPlusTaskInfo_t task_info;
  bg::ValueHolderPtr ffts_task = bg::ValueHolder::CreateConst(&task_info, sizeof(rtFftsPlusTaskInfo_t));
  uint32_t val = 20U;
  bg::ValueHolderPtr thread_dim = bg::ValueHolder::CreateConst(&val, sizeof(val));
  std::vector<uint32_t> mem_pool_types = {0, 1};
  uint32_t window_size = 20U;
  auto window_size_holder = bg::ValueHolder::CreateConst(&window_size, sizeof(window_size));
  auto ffts_allocator = CreateFftsMemAllocator(window_size_holder, global_data);
  FFTSLowerInput add_input = {{data1_ret.out_shapes[0], data2_ret.out_shapes[0]},
                              {data1_ret.out_addrs[0], data2_ret.out_addrs[0]},
                              mem_pool_types,
                              &global_data,
                              ffts_task,
                              thread_dim,
                              window_size_holder,
                              nullptr,
                              ffts_allocator,
                              tune::FFTSNodeThreadV2};
  auto add_op_desc = graph->FindNode("sgt_graph/Add")->GetOpDesc();
  ffts::ThreadSliceMapPtr slice_info_ptr = make_shared<ffts::ThreadSliceMap>();
  auto add_node = graph->FindNode("sgt_graph/Add");
  (void)add_node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_aicpu_ffts_args", "aicpu_args");
  (void)ge::AttrUtils::SetStr(add_node->GetOpDesc(), "_aicpu_ffts_ext_info", "aicpu_ext_info");
  if (node_type == "tf") {
    add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpuTf.c_str());
  } else if (node_type == "cc") {
    add_op_desc->SetOpKernelLibName(ge::kEngineNameAiCpu.c_str());
  }

  auto add_ret = LoweringFFTSAiCpuNode(graph->FindNode("sgt_graph/Add"), add_input);
  ASSERT_TRUE(add_ret.result.IsSuccess());
  ASSERT_EQ(add_ret.out_addrs.size(), 1);
  ASSERT_EQ(add_ret.out_shapes.size(), 1);
  ASSERT_EQ(add_ret.order_holders.size(), 1);
}

TEST_F(AicpuFFTSNodeConverterST, ConvertAicpuCCFFTSNode_NoRefNode) {
  LowerResult add_ret;
  TestAicpuFFTSConvertNoRefNode("cc", add_ret);
}

TEST_F(AicpuFFTSNodeConverterST, ConvertAicpuCCFFTSNode_WithRefNode) {
  LowerResult add_ret;
  TestAicpuFFTSConvertWithRefNode("cc", add_ret);
}

TEST_F(AicpuFFTSNodeConverterST, ConvertAicpuTfFFTSNode_NoRefNode) {
  LowerResult add_ret;
  TestAicpuFFTSConvertNoRefNode("tf", add_ret);
}

TEST_F(AicpuFFTSNodeConverterST, ConvertAicpuTfFFTSNode_WithRefNode) {
  LowerResult add_ret;
  TestAicpuFFTSConvertWithRefNode("tf", add_ret);
}
}  // namespace gert
