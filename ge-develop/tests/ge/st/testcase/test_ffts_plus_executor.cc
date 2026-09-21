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
#include <gmock/gmock.h>

#include "macro_utils/dt_public_scope.h"
#include "register/op_tiling_registry.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "framework/common/types.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "framework/executor/ge_executor.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/execute/model_executor.h"
#include "common/dump/dump_manager.h"
#include "graph/ge_context.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "common/profiling/profiling_properties.h"
#include "register/op_impl_kernel_registry.h"
#include "register/kernel_registry.h"
#include "common/sgt_slice_type.h"
#include "faker/space_registry_faker.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/manager/mem_manager.h"
#include "engine/aicore/fe_rt2_common.h"
#include "macro_utils/dt_public_unscope.h"

#include "ge_graph_dsl/graph_dsl.h"
#include "utils/bench_env.h"
#include "init_ge.h"
#include "common/ge_common/scope_guard.h"
#include "common/error_tracking/error_tracking.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace std;
using namespace testing;
using namespace optiling;

namespace ge {
class FftsPlusTest : public testing::Test {
 protected:
  void SetUp() {
    char runtime2_env[MMPA_MAX_PATH] = {'0'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    ReInitGe();
    BenchEnv::Init();
    RTS_STUB_SETUP();
    OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(CONV2D);
    OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(CONV2D);
    const auto op_tiling_func_aic = [](const Operator &op, const OpCompileInfoV2 &, OpRunInfoV2 &op_run_info) {
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
    OpTilingRegistryInterf_V2(CONV2D, op_tiling_func_aic);

    OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(RELU);
    OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(RELU);
    const auto op_tiling_func_mix = [](const Operator &op, const OpCompileInfoV2 &, OpRunInfoV2 &op_run_info) {
      static uint32_t tiling_key = 0;
      op_run_info.SetTilingKey(tiling_key % 4U);
      tiling_key++;

      op_run_info.SetWorkspaces({100, 128, 96});
      return true;
    };
    REGISTER_OP_TILING_UNIQ_V2(ReLU, op_tiling_func_mix, 201);
    OpTilingRegistryInterf_V2(RELU, op_tiling_func_mix);
  }

  void TearDown() {
    RTS_STUB_TEARDOWN();
    OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(CONV2D);
    OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(CONV2D);
    OpTilingFuncRegistry::RegisteredOpFuncInfo().erase(RELU);
    OpTilingRegistryInterf_V2::RegisteredOpInterf().erase(RELU);
    char runtime2_env[MMPA_MAX_PATH] = {'1'};
    mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
    ge::MemManager::Instance().Finalize();
  }
};

/***********************************************************************************************************************
 *                                     Data     Data       Data
 * Data    Data    Constant              |        |         |
 *   \       |       /                   |        |         |
 *    \      |      /               TransData  TransData    |
 *     \     |     /                     \        |        /                   Data  Data  Data
 *      \    |    /                       \       |       /                       \    |    /
 *    PartitionedCall                      \      |      /                         \   |   /
 *           |                              \     |     /                           Conv2D
 *           |                             PartitionedCall                             |
 *           |                                    |                                    |
 *           |                                    |                                  Relu
 *       NetOutput                                |                                    |
 *                                                |                                    |
 *                                            TransData                            GatherV2
 *                                                |                                    |
 *                                                |                                    |
 *                                                |                                NetOutput
 *                                            NetOutput
***********************************************************************************************************************/
static void BuildFftsDynamicGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &dsp_graph, ComputeGraphPtr &ffts_graph,
                                  bool is_ffts_dynamic, uint32_t &mem_offset) {
  int64_t max_size = 1;
  GeTensorDesc tensor_desc(GeShape(), FORMAT_ND, DT_INT64);
  GeTensorPtr const_tensor = MakeShared<GeTensor>(tensor_desc, reinterpret_cast<uint8_t *>(&max_size), sizeof(int64_t));
  const auto const_op = OP_CFG(CONSTANTOP).OutCnt(1).Weight(const_tensor);

  DEF_GRAPH(g1) {
                  CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
                  CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
                  CHAIN(NODE("weight", const_op)->NODE("PartitionedCall_0"));
                };
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);
  const auto root_call_0 = root_graph->FindNode("PartitionedCall_0");
  EXPECT_NE(root_call_0, nullptr);

  DEF_GRAPH(g2) {
                  const auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
                  const auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
                  const auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
                  CHAIN(NODE("dsp_graph/_arg_0", data_0)->EDGE(0, 0)->
                      NODE("dsp_graph/trans_TransData_0", IDENTITY)->EDGE(0, 0)->
                      NODE("dsp_graph/PartitionedCall_0", PARTITIONEDCALL)->EDGE(0, 0)->
                      NODE("dsp_graph/trans_TransData_2", IDENTITY)->EDGE(0, 0)->
                      NODE("dsp_graph/Node_Output", NETOUTPUT)
                  );
                  CHAIN(NODE("dsp_graph/_arg_1", data_1)->EDGE(0, 0)->
                      NODE("dsp_graph/trans_TransData_1", IDENTITY)->EDGE(0, 1)->
                      NODE("dsp_graph/PartitionedCall_0")
                  );
                  CHAIN(NODE("dsp_graph/_arg_2", data_2)->EDGE(0, 2)->NODE("dsp_graph/PartitionedCall_0"));
                };
  dsp_graph = ToComputeGraph(g2);
  dsp_graph->SetGraphUnknownFlag(is_ffts_dynamic);
  SetUnknownOpKernel(dsp_graph, mem_offset);
  AddPartitionedCall(root_graph, "PartitionedCall_0", dsp_graph);

  const auto ffts_call_node = dsp_graph->FindNode("dsp_graph/PartitionedCall_0");
  EXPECT_NE(ffts_call_node, nullptr);
  AttrUtils::SetBool(ffts_call_node->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);

  DEF_GRAPH(g3) {
                  const auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
                  const auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
                  const auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
                  const auto conv_0 = OP_CFG(CONV2D).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                      .Attr("_mix_with_enhanced_kernel", true);
                  const auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIV")
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
                  const auto gather = OP_CFG(GATHERV2).Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AICPU")
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU))
                      .Attr("_AllShape", true);
                  CHAIN(NODE("sgt_graph/_arg_0", data_0)->EDGE(0, 0)->
                      NODE("sgt_graph/Conv2D", conv_0)->EDGE(0, 0)->
                      NODE("sgt_graph/Relu", relu_0)->EDGE(0, 0)->
                      NODE("sgt_graph/Gather", gather)->EDGE(0, 0)->
                      NODE("sgt_graph/Node_Output", NETOUTPUT)
                  );
                  CHAIN(NODE("sgt_graph/_arg_1", data_1)->EDGE(0, 1)->NODE("sgt_graph/Conv2D", conv_0));
                  CHAIN(NODE("sgt_graph/_arg_2", data_2)->EDGE(0, 2)->NODE("sgt_graph/Conv2D", conv_0));
                };
  ffts_graph = ToComputeGraph(g3);
  ffts_graph->SetGraphUnknownFlag(is_ffts_dynamic);
  ffts_graph->TopologicalSorting();
  SetUnknownOpKernel(ffts_graph, mem_offset);
  AddPartitionedCall(dsp_graph, "dsp_graph/PartitionedCall_0", ffts_graph);

  const auto ffts_conv0 = ffts_graph->FindNode("sgt_graph/Conv2D");
  EXPECT_NE(ffts_conv0, nullptr);
  const auto ffts_relu0 = ffts_graph->FindNode("sgt_graph/Relu");
  EXPECT_NE(ffts_relu0, nullptr);
  const auto ffts_gather = ffts_graph->FindNode("sgt_graph/Gather");
  EXPECT_NE(ffts_gather, nullptr);

  InitFftsThreadSliceMap(ffts_conv0->GetOpDesc());
  InitFftsThreadSliceMap(ffts_relu0->GetOpDesc());
  InitFftsThreadSliceMap(ffts_gather->GetOpDesc());

  std::vector<char> conv_bin(64, '\0');
  TBEKernelPtr conv_kernel = MakeShared<ge::OpKernelBin>("sgt/conv", std::move(conv_bin));
  ffts_conv0->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, conv_kernel);
  ffts_conv0->GetOpDesc()->SetOpInferDepends({"__input2"}); // Not IR Build... Test Data Callback.
  AttrUtils::SetStr(ffts_conv0->GetOpDesc(), ffts_conv0->GetName() + "_kernelname", "sgt/conv");

  std::vector<char> relu_bin(64, '\0');
  TBEKernelPtr relu_kernel = MakeShared<ge::OpKernelBin>("sgt/relu", std::move(relu_bin));
  ffts_relu0->GetOpDesc()->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, relu_kernel);
  AttrUtils::SetStr(ffts_relu0->GetOpDesc(), ffts_relu0->GetName() + "_kernelname", "sgt/relu");

  const auto &sgt_nodes = ffts_graph->GetDirectNode();
  std::for_each(sgt_nodes.begin(), sgt_nodes.end(), [](const NodePtr &n) {
    (void)AttrUtils::SetBool(n->GetOpDesc(), "_kernel_list_first_name", true); // for call rtallkernel register
    (void)AttrUtils::SetInt(n->GetOpDesc(), "op_para_size", 2); // for tiling
  });
}

static void SetAicAivOpKernel(const ComputeGraphPtr &graph, const std::string name,
                              TBEKernelStore *kernel_store = nullptr) {
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
      std::make_shared<ge::OpKernelBin>(thread_kernel_names[1], std::move(aiv_kernel_bin))
  };
  if (kernel_store == nullptr) {
    op_desc->SetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tbe_kernel_vec);
  } else {
    kernel_store->AddTBEKernel(tbe_kernel_vec[0]);
    kernel_store->AddTBEKernel(tbe_kernel_vec[1]);
  }

  std::vector<string> bin_file_keys{ op_desc->GetName() + "_aic", op_desc->GetName() + "_aiv" };
  (void)AttrUtils::SetListStr(op_desc, "_register_stub_func", bin_file_keys);
  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_MODE, 1);
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_SCOPE_ID, 1);
  (void)ge::AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_aic");
  // Init Binary Magic
  std::vector<std::string> json_list{ "RT_DEV_BINARY_MAGIC_ELF_AIVEC", "RT_DEV_BINARY_MAGIC_ELF_AICUBE" };
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_magic", json_list);
  // Init meta data
  std::vector<std::string> meta_data_list{ "AIVEC_META_DATA", "AICUBE_META_DATA" };
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_metadata", meta_data_list);
}

static void SetManualAicAivOpKernel(const ComputeGraphPtr &graph, const std::string name,
                              TBEKernelStore *kernel_store = nullptr) {
  const auto &node = graph->FindNode(name);
  EXPECT_NE(node, nullptr);
  const auto &op_desc = node->GetOpDesc();
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  std::vector<char> conv_bin(64, '\0');
  TBEKernelPtr conv_kernel = MakeShared<ge::OpKernelBin>("sgt/conv", std::move(conv_bin));
  if (kernel_store == nullptr) {
    op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, conv_kernel);
  } else {
    kernel_store->AddTBEKernel(conv_kernel);
  }
  (void)AttrUtils::SetStr(op_desc, "_kernelname", conv_kernel->GetName());
  // Init Binary Magic
  std::vector<std::string> json_list{ "RT_DEV_BINARY_MAGIC_ELF_AIVEC", "RT_DEV_BINARY_MAGIC_ELF_AICUBE" };
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_magic", json_list);
  // Init meta data
  std::vector<std::string> meta_data_list{ "AIVEC_META_DATA", "AICUBE_META_DATA" };
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_metadata", meta_data_list);
}

static void SetMixL2AicAivOpKernel(const ComputeGraphPtr &graph, const std::string name,
                                   TBEKernelStore *kernel_store = nullptr) {
  const auto &node = graph->FindNode(name);
  EXPECT_NE(node, nullptr);
  const auto &op_desc = node->GetOpDesc();
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  std::vector<char> aic_kernel_bin(64, '\0');
  std::vector<char> aiv_kernel_bin(64, '\0');
  std::vector<std::string> thread_kernel_names = {"mixaic_a", "mixaiv_a"};
  (void)AttrUtils::SetListStr(op_desc, "_thread_kernelname", thread_kernel_names);

  std::vector<TBEKernelPtr> tbe_kernel_vec{
      std::make_shared<ge::OpKernelBin>(thread_kernel_names[0], std::move(aic_kernel_bin)),
      std::make_shared<ge::OpKernelBin>(thread_kernel_names[1], std::move(aiv_kernel_bin))
  };
  if (kernel_store == nullptr) {
    op_desc->SetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tbe_kernel_vec);
  } else {
    kernel_store->AddTBEKernel(tbe_kernel_vec[0]);
    kernel_store->AddTBEKernel(tbe_kernel_vec[1]);
  }

  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  (void)AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);

  std::vector<string> bin_file_keys{ op_desc->GetName() + "_aic", op_desc->GetName() + "_aiv" };
  (void)AttrUtils::SetListStr(op_desc, "_register_stub_func", bin_file_keys);
  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_MODE, 1);
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_SCOPE_ID, 1);
  // Init Binary Magic
  std::vector<std::string> json_list{ "RT_DEV_BINARY_MAGIC_ELF_AIVEC", "RT_DEV_BINARY_MAGIC_ELF_AICUBE" };
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_magic", json_list);
  // Init meta data
  std::vector<std::string> meta_data_list{ "AIVEC_META_DATA", "AICUBE_META_DATA" };
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
static void BuildFftsPlusGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &ffts_plus_graph,
                               TBEKernelStore *kernel_store = nullptr, bool is_mixl2 = false, bool is_auto = true) {
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
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                      .Attr("_mix_with_enhanced_kernel", true);
                  auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
                  auto add_0 = OP_CFG(ADD).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU)).Attr("_is_blocking_op", true);
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
  SetUnknownOpKernel(ffts_plus_graph, mem_offset);
  AddPartitionedCall(root_graph, "PartitionedCall_0", ffts_plus_graph);
  (void)AttrUtils::SetListInt(ffts_plus_graph->FindNode("sgt_graph/Conv2D")->GetOpDesc(), gert::kContextIdList, {0, 1, 2, 3});

  if (is_mixl2) {
    SetMixL2AicAivOpKernel(ffts_plus_graph, "sgt_graph/Conv2D", kernel_store);
  } else {
    if (is_auto) {
      SetAicAivOpKernel(ffts_plus_graph, "sgt_graph/Conv2D", kernel_store);
    } else {
      SetManualAicAivOpKernel(ffts_plus_graph, "sgt_graph/Conv2D", kernel_store);
    }
  }
}

static void RunPureStaticFftsPlusGraph(const ComputeGraphPtr &root_graph, const ComputeGraphPtr &ffts_plus_graph,
                                       const std::shared_ptr<domi::ModelTaskDef> &model_task_def) {
  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

static void RunDynamicStaticFftsPlusGraph(const ComputeGraphPtr &root_graph,
                                          const ComputeGraphPtr &ffts_plus_parent_graph,
                                          const std::shared_ptr<domi::ModelTaskDef> &model_task_def) {
  // Callback for Execute.
  std::mutex run_mutex;
  std::condition_variable model_run_cv;
  Status run_status = FAILED;
  const auto call_back = [&](Status status, std::vector<gert::Tensor> &outputs) {
    std::unique_lock<std::mutex> lock(run_mutex);
    run_status = status;
    model_run_cv.notify_one();
  };

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetModelName(root_graph->GetName());
  GeModelPtr ge_sub_model = MakeShared<GeModel>();
  ge_sub_model->SetModelTaskDef(model_task_def);

  ge_sub_model->SetGraph(ffts_plus_parent_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(ffts_plus_parent_graph->GetName(), ge_sub_model);

  AttrUtils::SetInt(ge_sub_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_sub_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_sub_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->IncreaseLoadCount();
  graph_node->SetAsync(true);
  graph_node->Lock();

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  // Execute for Dynamic Graph.
  std::vector<gert::Tensor> inputs(2);
  TensorCheckUtils::ConstructGertTensor(inputs[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(inputs[1], {1}, DT_INT64, FORMAT_ND);

  GEThreadLocalContext context;
  error_message::ErrorManagerContext error_context;
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  ASSERT_TRUE(arg != nullptr);
  arg->graph_node = graph_node;
  arg->graph_id = graph_id;
  arg->session_id = 2001;
  arg->error_context = error_context;
  arg->input_tensor = std::move(inputs);
  arg->context = context;
  arg->callback = call_back;
  EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

  std::unique_lock<std::mutex> lock(run_mutex);
  EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
  EXPECT_EQ(run_status, SUCCESS);
  // unload
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
  ge_root_model->ClearAllModelId();
}

TEST_F(FftsPlusTest, ffts_plus_dynamic_static_subgraph) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr dsp_graph;
  ComputeGraphPtr sgt_graph;
  uint32_t mem_offset = 0U;
  BuildFftsDynamicGraph(root_graph, dsp_graph, sgt_graph, false, mem_offset);
  // Set ffts's parent to be static
  dsp_graph->SetGraphUnknownFlag(false);
  const auto ffts_call_node = dsp_graph->FindNode("dsp_graph/PartitionedCall_0");
  ASSERT_EQ(dsp_graph->TopologicalSorting(), GRAPH_SUCCESS);
  // Build FftsPlusTaskDef.
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  domi::TaskDef &task_def = *model_def->add_task();
  InitFftsplusTaskDef(sgt_graph, task_def);
  // Add dsa ctx
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto dsa_def = ffts_plus_task_def->add_ffts_plus_ctx();
  InitDsaDef(sgt_graph, *dsa_def, 2U, "sgt_graph/Conv2D", true);
  // Run
  RunDynamicStaticFftsPlusGraph(root_graph, dsp_graph, model_def);
}

static void BuildDSAGraph(ComputeGraphPtr &root_graph) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
                  CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
                  CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
                  CHAIN(NODE("_arg_2", DATA)->NODE("PartitionedCall_0"));
                };
  root_graph = ToComputeGraph(g1);
  SetUnknownOpKernel(root_graph, mem_offset, true);
}

static void BuildDSAUnknownGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &sub_graph) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
                  CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
                  CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
                  CHAIN(NODE("_arg_2", DATA)->NODE("PartitionedCall_0"));
                };
  root_graph = ToComputeGraph(g1);
  root_graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(root_graph, mem_offset, true);


  DEF_GRAPH(g2) { // Known Graph
                  auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
                  auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
                  auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);

                  CHAIN(NODE("data0", data_0)->EDGE(0, 0)->NODE("memcpy0", MEMCPYASYNC)->EDGE(0, 0)->NODE("dsa", "DSARandomUniform")
                            ->EDGE(0, 0)->NODE("memcpy3", MEMCPYASYNC)->NODE("sub_Node_Output", NETOUTPUT));
                  CHAIN(NODE("data1", data_1)->EDGE(0, 0)->NODE("memcpy1", MEMCPYASYNC)->EDGE(0, 1)->NODE("dsa", "DSARandomUniform"));
                  CHAIN(NODE("data2", data_2)->EDGE(0, 0)->NODE("memcpy2", MEMCPYASYNC)->EDGE(0, 2)->NODE("dsa", "DSARandomUniform"));
                };
  sub_graph = ToComputeGraph(g2);
  sub_graph->SetGraphUnknownFlag(false);
  SetUnknownOpKernel(sub_graph, mem_offset, true);
  AddPartitionedCall(root_graph, "PartitionedCall_0", sub_graph);
}

TEST_F(FftsPlusTest, dsa_graph) {
  ComputeGraphPtr root_graph, sub_graph;
  BuildDSAUnknownGraph(root_graph, sub_graph);

  // Build GeModel.
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(root_graph);

  const auto model_task_def = MakeShared<domi::ModelTaskDef>();
  const auto ge_model = MakeShared<GeModel>();
  ge_root_model->SetSubgraphInstanceNameToModel(sub_graph->GetName(), ge_model);
  ge_model->SetGraph(sub_graph);
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetName(sub_graph->GetName());

  InitDSATaskDef(sub_graph, *model_task_def, "dsa", true);
  InitMemcpyAsyncDef(sub_graph, *model_task_def, "memcpy0");
  InitMemcpyAsyncDef(sub_graph, *model_task_def, "memcpy1");
  InitMemcpyAsyncDef(sub_graph, *model_task_def, "memcpy2");
  InitMemcpyAsyncDef(sub_graph, *model_task_def, "memcpy3");

  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);


  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  ASSERT_EQ(model_executor.Initialize({{VARIABLE_MEMORY_MAX_SIZE, "12800"}}, 0), SUCCESS);
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(FftsPlusTest, dsa_graph_set_input1_value) {
  ComputeGraphPtr root_graph;
  BuildDSAGraph(root_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  EXPECT_NE(root_graph, nullptr);

  InitDSATaskDef(root_graph, *model_task_def, "PartitionedCall_0", false);

  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(root_graph);;
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(FftsPlusTest, dsa_graph_with_dump) {
  ComputeGraphPtr root_graph;
  BuildDSAGraph(root_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  EXPECT_NE(root_graph, nullptr);
  InitDSATaskDef(root_graph, *model_task_def, "PartitionedCall_0", false);

  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();

  const uint64_t session_id = GetContext().SessionId();
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  DumpProperties dump_properties;
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);

  DumpManager::GetInstance().RemoveDumpProperties(session_id);
}

TEST_F(FftsPlusTest, ffts_plus_graph_load_success) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  const auto aic_node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");

  domi::FftsPlusCtxDef *fftsplusstartctx = ffts_plus_task_def.add_ffts_plus_ctx();
  fftsplusstartctx->set_op_index(0);
  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_START));
  domi::FftsPlusAtStartCtxDef *startctxdef = fftsplusstartctx->mutable_at_start_ctx();
  InitAtStartCtx(startctxdef);

  domi::FftsPlusCtxDef *fftsplusendctx = ffts_plus_task_def.add_ffts_plus_ctx();
  fftsplusendctx->set_op_index(0);
  fftsplusendctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_END));
  domi::FftsPlusAtEndCtxDef *endctxdef = fftsplusendctx->mutable_at_end_ctx();
  InitAtEndCtx(endctxdef);

  auto &label_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  label_def.set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_LABEL));
  label_def.mutable_label_ctx()->add_successor_list(1);

  domi::FftsPlusCtxDef *switch_ctx = ffts_plus_task_def.add_ffts_plus_ctx();
  switch_ctx->set_context_type(RT_CTX_TYPE_CASE_SWITCH);
  switch_ctx->mutable_case_switch_ctx()->add_successor_list(1);
  switch_ctx->mutable_case_switch_ctx()->set_successor_num(1);

  EXPECT_NE(aic_node, nullptr);
  std::vector<uint32_t> ctx_ids = {20, 21};
  (void)ge::AttrUtils::SetListInt(aic_node->GetOpDescBarePtr()->MutableOutputDesc(0), gert::kTensorCtxId, ctx_ids);
  (void)ge::AttrUtils::SetListInt(aic_node->GetOpDescBarePtr()->MutableOutputDesc(0), ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_OFFSETS, {20, 40, 2, 100, 200});
  std::string json_str =
      "{\"dependencies\":[],\"thread_scopeId\":200,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
      "0,"
      "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
      "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":\"thread_x\",\"oriInputTensorShape\":[],"
      "\"oriOutputTensorShape\":["
      "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[{\"splitCutIndex\":1,\"reduceCutIndex\":2,\"cutId\":3}],"
      "\"atomic_types\":[],\"thread_id\":0,\"same_atomic_clean_"
      "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
      "\"input_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"output_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
      "],\"inputCutList\":[], \"outputCutList\":[]}";
  (void)ge::AttrUtils::SetStr(aic_node->GetOpDesc(), ffts::kAttrSgtJsonInfo, json_str);
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");

  domi::FftsPlusCtxDef *data_ctx1 = ffts_plus_task_def.add_ffts_plus_ctx();
  data_ctx1->set_op_index(0);
  data_ctx1->set_context_id(20);
  data_ctx1->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_INVALIDATE_DATA));
  auto data1 = data_ctx1->mutable_data_ctx();
  data1->set_addr_base(0x10);
  data1->set_addr_offset(20);

  domi::FftsPlusCtxDef *data_ctx2 = ffts_plus_task_def.add_ffts_plus_ctx();
  data_ctx2->set_op_index(0);
  data_ctx2->set_context_id(21);
  data_ctx2->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_INVALIDATE_DATA));
  auto data2 = data_ctx2->mutable_data_ctx();
  data2->set_addr_base(0x10);
  data2->set_addr_offset(120);

  {
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
    RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(root_graph);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    const uint64_t session_id = GetContext().SessionId();
    DumpManager::GetInstance().RemoveDumpProperties(session_id);
    DumpProperties dump_properties;
    dump_properties.is_train_op_debug_ = true;
    DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

    // Test for Load.
    ModelExecutor model_executor;
    ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    ASSERT_EQ(model_executor.Finalize(), SUCCESS);

    DumpManager::GetInstance().RemoveDumpProperties(session_id);
  }

  {
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
    RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

    GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
    ge_root_model->Initialize(root_graph);
    ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

    GraphId graph_id = 1001;
    GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
    graph_node->SetGeRootModel(ge_root_model);;
    graph_node->SetLoadFlag(true);
    graph_node->SetAsync(true);

    const uint64_t session_id = GetContext().SessionId();
    DumpManager::GetInstance().RemoveDumpProperties(session_id);
    DumpProperties dump_properties;
    dump_properties.SetDumpMode("output");
    dump_properties.AddPropertyValue(DUMP_WATCHER_MODEL, {"PartitionedCall_0"});
    DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

    // Test for Load.
    ModelExecutor model_executor;
    ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
    model_executor.StartRunThread();
    EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
    EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
    ASSERT_EQ(model_executor.Finalize(), SUCCESS);

    DumpManager::GetInstance().RemoveDumpProperties(session_id);
  }

}

TEST_F(FftsPlusTest, ffts_plus_error_tracking_test) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  const auto aic_node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
  EXPECT_NE(aic_node, nullptr);
  AttrUtils::SetListStr(aic_node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});
  std::vector<uint32_t> ctx_ids = {20, 21};
  (void)ge::AttrUtils::SetListInt(aic_node->GetOpDescBarePtr()->MutableOutputDesc(0), gert::kTensorCtxId, ctx_ids);
  (void)ge::AttrUtils::SetListInt(aic_node->GetOpDescBarePtr()->MutableOutputDesc(0), ge::ATTR_NAME_FFTS_SUB_TASK_TENSOR_OFFSETS, {20, 40, 2, 100, 200});
  std::string json_str =
      "{\"dependencies\":[],\"thread_scopeId\":200,\"is_first_node_in_topo_order\":false,\"node_num_in_thread_scope\":"
      "0,"
      "\"is_input_node_of_thread_scope\":false,\"is_output_node_of_thread_scope\":false,\"threadMode\":false,\"slice_"
      "instance_num\":0,\"parallel_window_size\":0,\"thread_id\":\"thread_x\",\"oriInputTensorShape\":[],"
      "\"oriOutputTensorShape\":["
      "],\"original_node\":\"\",\"core_num\":[],\"cutType\":[{\"splitCutIndex\":1,\"reduceCutIndex\":2,\"cutId\":3}],"
      "\"atomic_types\":[],\"thread_id\":0,\"same_atomic_clean_"
      "nodes\":[],\"input_axis\":[],\"output_axis\":[],\"input_tensor_indexes\":[],\"output_tensor_indexes\":[],"
      "\"input_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"output_tensor_slice\":[[[{\"lower\":1, \"higher\":2}, {\"lower\":3, \"higher\":4}, {\"lower\":3, \"higher\":4}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]],"
      "[[{\"lower\":9, \"higher\":10}, {\"lower\":11, \"higher\":12}, {\"lower\":11, \"higher\":12}]]],"
      "\"ori_input_tensor_slice\":[],\"ori_output_tensor_slice\":["
      "],\"inputCutList\":[], \"outputCutList\":[]}";
  (void)ge::AttrUtils::SetStr(aic_node->GetOpDesc(), ffts::kAttrSgtJsonInfo, json_str);
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");
  const auto relu_node = ffts_plus_graph->FindNode("sgt_graph/Relu");
  AttrUtils::SetListStr(relu_node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});

  domi::FftsPlusCtxDef *data_ctx1 = ffts_plus_task_def.add_ffts_plus_ctx();
  data_ctx1->set_op_index(0);
  data_ctx1->set_context_id(20);
  data_ctx1->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_INVALIDATE_DATA));
  auto data1 = data_ctx1->mutable_data_ctx();
  data1->set_addr_base(0x10);
  data1->set_addr_offset(20);

  domi::FftsPlusCtxDef *data_ctx2 = ffts_plus_task_def.add_ffts_plus_ctx();
  data_ctx2->set_op_index(0);
  data_ctx2->set_context_id(21);
  data_ctx2->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_INVALIDATE_DATA));
  auto data2 = data_ctx2->mutable_data_ctx();
  data2->set_addr_base(0x10);
  data2->set_addr_offset(120);

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  const uint64_t session_id = GetContext().SessionId();
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  DumpProperties dump_properties;
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  rtExceptionInfo rt_exception_info;
  rt_exception_info.streamid = 0xFFFFFFFFU;
  rt_exception_info.taskid = 0xFFFFFFFFU;
  rt_exception_info.expandInfo.type = RT_EXCEPTION_FFTS_PLUS;
  rt_exception_info.expandInfo.u.fftsPlusInfo.contextId = 0;
  rt_exception_info.expandInfo.u.fftsPlusInfo.threadId = 1;
  ErrorTrackingCallback(&rt_exception_info);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);

  DumpManager::GetInstance().RemoveDumpProperties(session_id);
}

TEST_F(FftsPlusTest, ffts_plus_graph_manual_load_success) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D", true);
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  const uint64_t session_id = GetContext().SessionId();
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  DumpProperties dump_properties;
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);

  DumpManager::GetInstance().RemoveDumpProperties(session_id);
}

TEST_F(FftsPlusTest, ffts_plus_graph_load_success_with_tiling_data) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store, false, false);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  // eliminate circular reference before ffts_plus_graph calls destructor
  GE_MAKE_GUARD(DelExtAttrBefore, [&ffts_plus_graph]() {
    auto node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
    GE_CHECK_NOTNULL_JUST_RETURN(node);
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL_JUST_RETURN(op_desc);
    (void)op_desc->DelExtAttr("memset_node_ptr");
  });
  InitFftsPlusAicCtxDefWithTilingData(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  const uint64_t session_id = GetContext().SessionId();
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  DumpProperties dump_properties;
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);

  DumpManager::GetInstance().RemoveDumpProperties(session_id);
}

TEST_F(FftsPlusTest, ffts_plus_graph_load_with_exceptiondump) {
  const char_t * const kEnvRecordPath = "NPU_COLLECT_PATH";
  char_t npu_collect_path[MMPA_MAX_PATH] = "valid_path";
  mmSetEnv(kEnvRecordPath, &npu_collect_path[0U], MMPA_MAX_PATH);

  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  unsetenv(kEnvRecordPath);
  remove("valid_path");

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(FftsPlusTest, ffts_plus_graph_with_aicpu_load_success) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  auto node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
  AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, {"op1", "op2", "op3"});
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &custom_aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitCustomFftsPlusAicpuCtxDef(ffts_plus_graph, custom_aicpu_ctx_def, "sgt_graph/Relu");
  ProfilingProperties::Instance().SetLoadProfiling(true);
  auto hash_func = [](const char_t * info, size_t length)->uint64_t {return 0;};
  ge::ProfilingTestUtil::Instance().hash_func_ = hash_func;
  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);
  ge::ProfilingTestUtil::Instance().hash_func_ = nullptr;
}

TEST_F(FftsPlusTest, ffts_plus_graph_with_aicpu_load_no_block_sucess) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &custom_aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitCustomFftsPlusAicpuCtxDef(ffts_plus_graph, custom_aicpu_ctx_def, "sgt_graph/Relu");

  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(FftsPlusTest, ffts_plus_graph_with_aicpu_load_failed) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &custom_aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitCustomFftsPlusAicpuCtxDef(ffts_plus_graph, custom_aicpu_ctx_def, "sgt_graph/Relu");

  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_MODE_NO_FFTS);
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_NE(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(FftsPlusTest, ffts_plus_graph_init) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &case_defalut_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusCaseDefaultDef(ffts_plus_graph, case_defalut_ctx_def, "sgt_graph/Conv2D");
  auto &notify_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusNotifyDef(ffts_plus_graph, notify_ctx_def, "sgt_graph/Add");
  auto &write_value_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitWriteValueDef(ffts_plus_graph, write_value_def, "sgt_graph/Relu");
  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);
}

TEST_F(FftsPlusTest, ffts_plus_graph_init_task_def) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);
  AttrUtils::SetInt(ffts_plus_graph->GetParentNode()->GetOpDesc(), "_parallel_group_id", 0);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &sdma_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitSdmaDef(ffts_plus_graph, sdma_def, "sgt_graph/Conv2D");
  auto &data_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitDataCtx(ffts_plus_graph, data_def, "sgt_graph/Add");
  auto &switch_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitCondSwitchCtx(ffts_plus_graph, switch_def, "sgt_graph/Relu");
  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);

  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);
}

TEST_F(FftsPlusTest, ffts_plus_graph_dsa_task_def) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &dsa_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitDsaDef(ffts_plus_graph, dsa_def, 1U, "sgt_graph/Conv2D", true);
  auto &data_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitDataCtx(ffts_plus_graph, data_def, "sgt_graph/Add");
  auto &switch_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitCondSwitchCtx(ffts_plus_graph, switch_def, "sgt_graph/Relu");

  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);

  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);
}

TEST_F(FftsPlusTest, ffts_plus_graph_dsa_task_def_with_ptr) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &dsa_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitDsaDef(ffts_plus_graph, dsa_def, 1U, "sgt_graph/Conv2D", false);
  auto &data_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitDataCtx(ffts_plus_graph, data_def, "sgt_graph/Add");
  auto &switch_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitCondSwitchCtx(ffts_plus_graph, switch_def, "sgt_graph/Relu");

  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);

  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(FftsPlusTest, ffts_plus_graph_dsa_task_def_with_dump) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &dsa_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitDsaDef(ffts_plus_graph, dsa_def, 1U, "sgt_graph/Conv2D", false);
  auto &data_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitDataCtx(ffts_plus_graph, data_def, "sgt_graph/Add");
  auto &switch_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitCondSwitchCtx(ffts_plus_graph, switch_def, "sgt_graph/Relu");


  const uint64_t session_id = GetContext().SessionId();
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  DumpProperties dump_properties;
  dump_properties.is_train_op_debug_ = true;
  DumpManager::GetInstance().AddDumpProperties(session_id, dump_properties);
  DumpManager::GetInstance().GetDumpProperties(session_id).SetDumpList("");
  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);

  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);
  DumpManager::GetInstance().RemoveDumpProperties(session_id);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(FftsPlusTest, ffts_plus_graph_cache_persist) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &cache_persist_ctx = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusCachePersistDef(ffts_plus_graph, cache_persist_ctx, "sgt_graph/Conv2D");
  RunPureStaticFftsPlusGraph(root_graph, ffts_plus_graph, model_task_def);
}

TEST_F(FftsPlusTest, FftsPlusTest_ffts_plus_auto_graph_with_mix_load_fail) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &mix_aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitMixAicAivDef(ffts_plus_graph, mix_aic_ctx_def, "sgt_graph/Conv2D", true);
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_NE(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(FftsPlusTest, FftsPlusTest_ffts_plus_auto_graph_with_mix_load_success) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store, true);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &mix_aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitMixAicAivDef(ffts_plus_graph, mix_aic_ctx_def, "sgt_graph/Conv2D", false);
  mix_aic_ctx_def.mutable_mix_aic_aiv_ctx()->add_task_addr(std::numeric_limits<uint64_t>::max());
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

UINT32 StubTilingFFTSST(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  auto tiling_data = context->GetTilingData<uint64_t>();
  *tiling_data = 100;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParseFFTSST(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void* CompileInfoCreatorFFTSST() {
  auto tmp =  ge::MakeUnique<char>();
  return tmp.get();
}

TEST_F(FftsPlusTest, FftsPlusTest_ffts_plus_graph_mix_prof_Test) {
  auto &space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  auto funcs = space_registry->CreateOrGetOpImpl(RELU);
  funcs->tiling = StubTilingFFTSST;
  funcs->tiling_parse = StubTilingParseFFTSST;
  funcs->compile_info_creator = CompileInfoCreatorFFTSST;
  funcs->compile_info_deleter = nullptr;

  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store, true);

  const auto relu = ffts_plus_graph->FindNode("sgt_graph/Relu");
  const auto op_desc = relu->GetOpDesc();
  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  ffts_plus_task_def.set_addr_size(1000);
  auto &mix_aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitMixAicAivDef(ffts_plus_graph, mix_aic_ctx_def, "sgt_graph/Conv2D");
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  ProfilingProperties::Instance().SetLoadProfiling(true);

  auto func2 = [&](uint32_t moduleId, uint32_t type, void *data, uint32_t len) {
    if (type == ge::InfoType::kCompactInfo) {
      auto node_basic_info = reinterpret_cast<MsprofCompactInfo *>(data);
      if (node_basic_info->data.nodeBasicInfo.taskType == MSPROF_GE_TASK_TYPE_MIX_AIC ||
          node_basic_info->data.nodeBasicInfo.taskType == MSPROF_GE_TASK_TYPE_MIX_AIV) {
        EXPECT_EQ(node_basic_info->data.nodeBasicInfo.blockDim, 0x10006);
      }
    }
    return 0;
  };
  ProfilingTestUtil::Instance().SetProfFunc(func2);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
  ProfilingProperties::Instance().SetLoadProfiling(false);
  ProfilingTestUtil::Instance().Clear();
}

TEST_F(FftsPlusTest, ffts_plus_graph_load_with_level1update) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def= MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  // add cmo ctx
  domi::FftsPlusCtxDef *datactx = ffts_plus_task_def.add_ffts_plus_ctx();
  datactx->set_op_index(0);
  datactx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_FLUSH_DATA));
  domi::FftsPlusDataCtxDef *datadef = datactx->mutable_data_ctx();
  InitDataCtx(datadef);
  datadef->set_addr_base(0x300);

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
  ge_root_model->Initialize(root_graph);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);;
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  const auto davinci_model = ModelManager::GetInstance().GetModel(ge_root_model->GetModelId());
  ASSERT_NE(davinci_model, nullptr);
  davinci_model->SetFeatureBaseRefreshable(true);
  std::vector<uint64_t> inputs{0x300};
  std::vector<uint64_t> outputs{0x100};
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  EXPECT_EQ(davinci_model->UpdateKnownNodeArgs(inputs, outputs), SUCCESS);
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}
} // namespace ge
