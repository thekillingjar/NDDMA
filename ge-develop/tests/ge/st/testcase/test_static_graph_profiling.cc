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

#include "common/profiling/profiling_properties.h"
#include "common/global_variables/diagnose_switch.h"
#include "framework/runtime/subscriber/global_profiler.h"
#include "framework/ge_runtime_stub/include/common/share_graph.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "framework/ge_runtime_stub/include/faker/ge_model_builder.h"
#include "framework/ge_runtime_stub/include/faker/aicore_taskdef_faker.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "depends/profiler/src/profiling_auto_checker.h"
#include "runtime/v2/lowering/model_converter.h"
#include "runtime/v2/lowering/graph_converter.h"
#include "graph/manager/graph_manager.h"
#include "graph/execute/model_executor.h"
#include "graph/load/model_manager/model_manager.h"
#include "framework/ge_runtime_stub/include/faker/global_data_faker.h"
#include "framework/ge_graph_dsl/include/ge_graph_dsl/graph_dsl.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "utils/bench_env.h"
#include "init_ge.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "register/op_tiling_registry.h"
#include "graph/manager/mem_manager.h"

namespace ge {
class StaticGraphProfilingSt : public testing::Test {
 protected:
  void SetUp() override {
    ReInitGe();
    BenchEnv::Init();
    RTS_STUB_SETUP();
    optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(CONV2D);
    optiling::OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(CONV2D);
    const auto op_tiling_func_aic = [](const Operator &op, const optiling::OpCompileInfoV2 &,
                                       optiling::OpRunInfoV2 &op_run_info) {
      static uint32_t tiling_key = 0;
      op_run_info.SetTilingKey(tiling_key % 2U);
      tiling_key++;

      const std::vector<char> tiling_data(126, 0);
      op_run_info.AddTilingData(tiling_data.data(), tiling_data.size());
      op_run_info.SetWorkspaces({100, 100});
      op_run_info.SetBlockDim(666);
      return true;
    };
    REGISTER_OP_TILING_UNIQ_V2(Conv2D, op_tiling_func_aic, 201);
    optiling::OpTilingRegistryInterf_V2(CONV2D, op_tiling_func_aic);

    optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(RELU);
    optiling::OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(RELU);
    const auto op_tiling_func_mix = [](const Operator &op, const optiling::OpCompileInfoV2 &,
                                       optiling::OpRunInfoV2 &op_run_info) {
      static uint32_t tiling_key = 0;
      op_run_info.SetTilingKey(tiling_key % 4U);
      tiling_key++;

      op_run_info.SetWorkspaces({100, 128, 96});
      return true;
    };
    REGISTER_OP_TILING_UNIQ_V2(ReLU, op_tiling_func_mix, 201);
    optiling::OpTilingRegistryInterf_V2(RELU, op_tiling_func_mix);
  }
  void TearDown() override {
    gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(0UL);
    gert::GlobalDumper::GetInstance()->SetEnableFlags(0);
    RTS_STUB_TEARDOWN();
    optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(CONV2D);
    optiling::OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(CONV2D);
    optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(RELU);
    optiling::OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(RELU);
    ge::MemManager::Instance().Finalize();
  }
};

void TestLoadStaticModel() {
  auto compute_graph = gert::ShareGraph::BuildWithKnownSubgraph();
  for (auto &node : compute_graph->GetAllNodes()) {
    if (node->GetName() == "conv2d") {
      ge::AttrUtils::SetInt(node->GetOpDesc(), "_logic_stream_id", 100);
    }
    ge::AttrUtils::SetInt(node->GetOpDesc(), "_op_impl_mode_enum", 0x40);
  }
  gert::GertRuntimeStub runtime_stub;
  compute_graph->TopologicalSorting();
  auto root_model = gert::GeModelBuilder(compute_graph).BuildGeRootModel();
  auto faker = gert::GlobalDataFaker(root_model);
  auto global_data = faker.FakeWithoutHandleAiCore("Conv2d", false).Build();
  gert::ModelDescHolder model_desc_holder;
  model_desc_holder.SetSpaceRegistry(gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry());
  auto graph_convert = gert::GraphConverter().SetModelDescHolder(&model_desc_holder);
  auto exe_graph = graph_convert.ConvertComputeGraphToExecuteGraph(compute_graph, global_data);
  ASSERT_NE(exe_graph, nullptr);
  auto model_executor = gert::ModelV2Executor::Create(exe_graph, root_model);
  ASSERT_NE(model_executor, nullptr);
  ge::diagnoseSwitch::EnableTaskTimeProfiling();
  EXPECT_EQ(model_executor->Load(), ge::GRAPH_SUCCESS);
  EXPECT_EQ(model_executor->UnLoad(), ge::GRAPH_SUCCESS);
}

/**
 * 用例描述：静态子图加载，上报logic stream info
 *
 * 预置条件：NA
 *
 * 测试步骤：
 * 1. 构造包含静态子图的计算图
 * 2. 给计算图上的算子打logic_stream_id属性
 * 3. 打开profiling的task time l0开关
 * 4. 加载
 *
 * 预期结果：
 * 1. 上报profiling的logic stream info
 */
TEST_F(StaticGraphProfilingSt, DavinciModelProfiling_ReportLogicStreamInfo_WithTaskTimeL0SwitchOn) {
  size_t records = 0UL;
  auto check_func = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) -> int32_t {
    if (type == ge::InfoType::kInfo) {
      auto additional_info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      if (additional_info->type == MSPROF_REPORT_MODEL_LOGIC_STREAM_TYPE) {
        ++records;
        auto logic_stream_info = reinterpret_cast<MsprofLogicStreamInfo *>(additional_info->data);
        logic_stream_info->physicStreamNum = 1;
        logic_stream_info->logicStreamId = 100;
      }
    }
    return 0;
  };
  AutoProfilingTestWithExpectedFunc(TestLoadStaticModel, check_func);
  EXPECT_EQ(records, 1);
}

void SetAicAivOpKernel(const ComputeGraphPtr &graph, const std::string name, TBEKernelStore *kernel_store = nullptr) {
  const auto &node = graph->FindNode(name);
  EXPECT_NE(node, nullptr);
  const auto &op_desc = node->GetOpDesc();
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  std::vector<char> aic_kernel_bin(64, '\0');
  std::vector<char> aiv_kernel_bin(64, '\0');
  std::vector<std::string> thread_kernel_names = {"aictest", "aivtest"};
  (void)AttrUtils::SetListStr(op_desc, "_thread_kernelname", thread_kernel_names);

  std::vector<TBEKernelPtr> tbe_kernel_vec{
      std::make_shared<ge::OpKernelBin>(thread_kernel_names[0], std::move(aic_kernel_bin)),
      std::make_shared<ge::OpKernelBin>(thread_kernel_names[1], std::move(aiv_kernel_bin))};
  if (kernel_store == nullptr) {
    op_desc->SetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tbe_kernel_vec);
  } else {
    kernel_store->AddTBEKernel(tbe_kernel_vec[0]);
    kernel_store->AddTBEKernel(tbe_kernel_vec[1]);
  }

  std::vector<string> bin_file_keys{op_desc->GetName() + "_aic", op_desc->GetName() + "_aiv"};
  (void)AttrUtils::SetListStr(op_desc, "_register_stub_func", bin_file_keys);
  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_MODE, 1);
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_SCOPE_ID, 1);
  // Init Binary Magic
  std::vector<std::string> json_list{"RT_DEV_BINARY_MAGIC_ELF_AIVEC", "RT_DEV_BINARY_MAGIC_ELF_AICUBE"};
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_magic", json_list);
  // Init meta data
  std::vector<std::string> meta_data_list{"AIVEC_META_DATA", "AICUBE_META_DATA"};
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_metadata", meta_data_list);
}

/***********************************************************************************************************************
 *
 * Data    Data    Data
 *   \      |       /
 *    \     |      /
 *     \    |     /                                       Data  Data  Const
 *      \   |    /                                          \    /   /
 *   PartitionedCall                                        \   /   /
 *          |                                          Data  Conv2D
 *          |                                              \   |
 *          |                                               \  |
 *          |                                                Add
 *          |                                                  |
 *          |                                                Relu
 *      NetOutput                                             |
 *                                                            |
 *                                                        NetOutput
 *
 *
 **********************************************************************************************************************/
void BuildFftsPlusGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &ffts_plus_graph,
                        TBEKernelStore *kernel_store = nullptr, bool is_mixl2 = false) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
    CHAIN(NODE("_arg_2", DATA)->NODE("PartitionedCall_0"));
  };
  root_graph = ToComputeGraph(g1);
  SetUnknownOpKernel(root_graph, mem_offset, true);

  DEF_GRAPH(g2) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto conv_0 = OP_CFG(CONV2D)
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    auto add_0 = OP_CFG(ADD)
                     .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU))
                     .Attr("_is_blocking_op", true);
    CHAIN(NODE("sgt_graph/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Add", add_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Relu", relu_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT));
    auto const_0 = OP_CFG(CONSTANT);
    CHAIN(NODE("sgt_graph/_arg_1", data_1)->EDGE(0, 1)->NODE("sgt_graph/Conv2D", conv_0));
    CHAIN(NODE("sgt_graph/_const_0", const_0)->EDGE(0, 2)->NODE("sgt_graph/Conv2D", conv_0));
    CHAIN(NODE("sgt_graph/_arg_2", data_2)->EDGE(0, 1)->NODE("sgt_graph/Add", add_0));
  };
  ffts_plus_graph = ToComputeGraph(g2);

  auto const_node = ffts_plus_graph->FindNode("sgt_graph/_const_0");
  std::vector<int64_t> data(4,1);
  GeTensor weight;
  weight.SetData((uint8_t *)data.data(), data.size() * sizeof(int64_t));
  GeTensorDesc weight_desc;
  weight_desc.SetShape(GeShape({4}));
  weight_desc.SetOriginShape(GeShape({4}));
  weight.SetTensorDesc(weight_desc);
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", weight);

  SetUnknownOpKernel(ffts_plus_graph, mem_offset);
  AddPartitionedCall(root_graph, "PartitionedCall_0", ffts_plus_graph);
  SetAicAivOpKernel(ffts_plus_graph, "sgt_graph/Conv2D", kernel_store);
}

void TestStaticFftsPlusGraphLoad() {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  aic_ctx_def.set_uniq_ctx_name("testtesttest1");
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  aicpu_ctx_def.set_uniq_ctx_name("testtesttest2");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");
  aicpu_ctx_def_relu.set_uniq_ctx_name("testtesttest3");
  ffts_plus_task_def.set_addr_size(1000);

  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kTaskTime}));
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}
}  // namespace ge
