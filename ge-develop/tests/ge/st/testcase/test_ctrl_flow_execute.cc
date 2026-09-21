/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ge_graph_dsl/graph_dsl.h"

#include "ge/ut/ge/test_tools_task_info.h"
#include "framework/executor/ge_executor.h"
#include "graph/execute/model_executor.h"
#include "framework/common/helper/om_file_helper.h"
#include "framework/common/helper/model_helper.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"
#include "mmpa/mmpa_api.h"

using namespace std;
using namespace testing;

namespace ge {
class CtrlFlowDynamicTest : public testing::Test {
 protected:
  void SetUp() {
    GeExecutor::Initialize({});

    //NodeExecutorManager::GetInstance().engine_mapping_.clear();
    //auto &engine_mapping = NodeExecutorManager::GetInstance().engine_mapping_;
    //engine_mapping.emplace("DNN_VM_RTS_OP_STORE", NodeExecutorManager::ExecutorType::RTS);
    //engine_mapping.emplace("DNN_VM_GE_LOCAL_OP_STORE", NodeExecutorManager::ExecutorType::GE_LOCAL);
    //
    //NodeExecutorManager::GetInstance().executors_.clear();
    //auto &task_executor = NodeExecutorManager::GetInstance().executors_;
    //task_executor.emplace(NodeExecutorManager::ExecutorType::RTS, std::unique_ptr<NodeExecutor>(new RtsNodeExecutor()));
    //task_executor.emplace(NodeExecutorManager::ExecutorType::GE_LOCAL, std::unique_ptr<NodeExecutor>(new GeLocalNodeExecutor()));
  }
  void TearDown() {
    //NodeExecutorManager::GetInstance().engine_mapping_.clear();
    //NodeExecutorManager::GetInstance().executors_.clear();

    GeExecutor::FinalizeEx();
  }
};

/**
*             |
*           Merge
*          /     \.
*  Active /       \ Active
*        /         \.
*       Add       Sub
*      |   \     /   |
*      |    \ _ /    |
*      |    /   \    |
*      |   /     \   |
*    Switch       Switch
*     |   \       /   |
*     |    \     /    |
*     |    Active     |
*     |     \  /      |
*     |     Less      |
*     |     /   \     |
*     |    /     \    |
*      Data       Data
*/
static void BuildSampleCondGraph(ComputeGraphPtr &graph, uint32_t &mem_offset) {
  DEF_GRAPH(g0) {
    const auto add_node = OP_CFG(IDENTITY).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    const auto sub_node = OP_CFG(IDENTITY).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    const auto less_node = OP_CFG(IDENTITY).Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    CHAIN(NODE("_arg_0", DATA)->NODE("add", add_node)->NODE("merge", STREAMMERGE)->NODE(NODE_NAME_NET_OUTPUT, NETOUTPUT));
    CHAIN(NODE("const_0", CONSTANT)->NODE("add"));
    CHAIN(NODE("_arg_1", DATA)->NODE("sub", sub_node)->NODE("merge"));
    CHAIN(NODE("const_1", CONSTANT)->NODE("sub"));

    const auto switch_t = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    const auto switch_f = OP_CFG(STREAMSWITCH).Attr(ATTR_NAME_STREAM_SWITCH_COND, static_cast<uint32_t>(RT_NOT_EQUAL))
                                              .Attr(ATTR_NAME_SWITCH_DATA_TYPE, static_cast<int64_t>(RT_SWITCH_INT64))
                                              .Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("_arg_0")->EDGE(0, 0)->NODE("less", less_node)->EDGE(0, 0)->NODE("Less/StreamSwitch_t", switch_t)->CTRL_EDGE()->NODE("add"));
    CHAIN(NODE("const_0")->EDGE(0, 1)->NODE("Less/StreamSwitch_t"));
    CHAIN(NODE("_arg_1")->EDGE(0, 1)->NODE("less", less_node)->EDGE(0, 0)->NODE("Less/StreamSwitch_f", switch_f)->CTRL_EDGE()->NODE("sub"));
    CHAIN(NODE("const_1")->EDGE(0, 1)->NODE("Less/StreamSwitch_f"));

    const auto active_s = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{1});
    CHAIN(NODE("less")->CTRL_EDGE()->NODE("Less_StreamActive", active_s)->CTRL_EDGE()->NODE("Less/StreamSwitch_t"));
    CHAIN(NODE("Less_StreamActive")->CTRL_EDGE()->NODE("Less/StreamSwitch_f"));

    const auto active_0 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    const auto active_1 = OP_CFG(STREAMACTIVE).Attr(ATTR_NAME_ACTIVE_STREAM_LIST, std::vector<int64_t>{2});
    CHAIN(NODE("add")->CTRL_EDGE()->NODE("merge_input_0_active", active_0)->CTRL_EDGE()->NODE("merge"));
    CHAIN(NODE("sub")->CTRL_EDGE()->NODE("merge_input_1_active", active_1)->CTRL_EDGE()->NODE("merge"));
  };

  graph = ToComputeGraph(g0);
  graph->SetGraphUnknownFlag(true);
  SetUnknownOpKernel(graph, mem_offset, true);
}

void BuildGraphModel(const ComputeGraphPtr &graph, uint32_t mem_offset, GeModelPtr &ge_model, TBEKernelStore &tbe_kernel_store) {
  InitConstantNode(graph, "const_0", 1);
  InitConstantNode(graph, "const_1", 0);

  std::shared_ptr<domi::ModelTaskDef> model_def = MakeShared<domi::ModelTaskDef>();
  InitKernelTaskDef_TE(graph, *model_def, "less", tbe_kernel_store);

  InitStreamActiveDef(graph, *model_def, "Less_StreamActive");
  InitStreamSwitchDef(graph, *model_def, "Less/StreamSwitch_f");
  InitStreamSwitchDef(graph, *model_def, "Less/StreamSwitch_t");

  InitKernelTaskDef_TE(graph, *model_def, "add", tbe_kernel_store);
  InitKernelTaskDef_TE(graph, *model_def, "sub", tbe_kernel_store);

  InitStreamActiveDef(graph, *model_def, "merge_input_0_active");
  InitStreamActiveDef(graph, *model_def, "merge_input_1_active");
  InitStreamMergeDef(graph, *model_def, "merge");

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetModelTaskDef(model_def);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, mem_offset));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_WEIGHT_SIZE, weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 0));
}

Status OnlineInferDynamic(ComputeGraphPtr &graph, const GeModelPtr &ge_model) {
  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  ModelExecutor model_executor;
  EXPECT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);

  std::vector<gert::Tensor> input_tensors(2);
  TensorCheckUtils::ConstructGertTensor(input_tensors[0], {1}, DT_INT64, FORMAT_ND);
  TensorCheckUtils::ConstructGertTensor(input_tensors[1], {1}, DT_INT64, FORMAT_ND);

  std::mutex run_mutex;
  std::condition_variable model_run_cv;
  Status run_status = FAILED;
  std::vector<gert::Tensor> run_outputs;
  const auto callback = [&](Status status, std::vector<gert::Tensor> &outputs) {
    std::unique_lock<std::mutex> lock(run_mutex);
    run_status = status;
    run_outputs.swap(outputs);
    model_run_cv.notify_one();
  };

  GEThreadLocalContext context;
  error_message::ErrorManagerContext error_context;
  graph_node->Lock();
  std::shared_ptr<RunArgs> arg;
  arg = std::make_shared<RunArgs>();
  if (arg == nullptr) {
    return FAILED;
  }
  arg->graph_node = graph_node;
  arg->graph_id = graph_id;
  arg->session_id = 2001;
  arg->error_context = error_context;
  arg->input_tensor = std::move(input_tensors);
  arg->context = context;
  arg->callback = callback;
  EXPECT_EQ(model_executor.PushRunArgs(arg), SUCCESS);

  std::unique_lock<std::mutex> lock(run_mutex);
  EXPECT_EQ(model_run_cv.wait_for(lock, std::chrono::seconds(10)), std::cv_status::no_timeout);
  EXPECT_EQ(run_status, SUCCESS);
  EXPECT_EQ(run_outputs.size(), 1U);

  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  EXPECT_EQ(model_executor.Finalize(), SUCCESS);
  return run_status;
}

TEST_F(CtrlFlowDynamicTest, ctrl_flow_dynamic_execute) {
  char runtime2_env[MMPA_MAX_PATH] = {'0'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
  uint32_t mem_offset = 0;
  ComputeGraphPtr graph;
  BuildSampleCondGraph(graph, mem_offset);
  EXPECT_NE(graph, nullptr);
  graph->SetGraphUnknownFlag(true);

  GeModelPtr ge_model;
  TBEKernelStore tbe_kernel_store;
  BuildGraphModel(graph, mem_offset, ge_model, tbe_kernel_store);
  EXPECT_NE(ge_model, nullptr);

  {
    // Test LoadModelOnline: V1 control flow dynamic.
    EXPECT_EQ(OnlineInferDynamic(graph, ge_model), SUCCESS);
  }

  // Serialization GeModel to memory.
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelBufferData model_buffer;
  EXPECT_TRUE(tbe_kernel_store.Build());
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  const ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};

  {
    // Test LoadModelOffline: V1 control flow dynamic.
    int64_t arg_0 = 127;
    int64_t arg_1 = 100;
    RunModelData run_input_data;
    run_input_data.blobs.emplace_back(DataBuffer{&arg_0, sizeof(arg_0), false, 0});
    run_input_data.blobs.emplace_back(DataBuffer{&arg_1, sizeof(arg_1), false, 0});

    int64_t arg_3 = 111;
    RunModelData run_output_data;
    run_output_data.blobs.emplace_back(DataBuffer{&arg_3, sizeof(arg_3), false, 0});

    uint32_t model_id = 0;
    GeExecutor ge_executor;
    EXPECT_EQ(ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0U, nullptr, 0U), SUCCESS);
    EXPECT_EQ(ge_executor.ExecModel(model_id, nullptr, run_input_data, run_output_data, true), SUCCESS);
    EXPECT_EQ(ge_executor.UnloadModel(model_id), SUCCESS);
  }
  runtime2_env[0] = {'1'};
  mmSetEnv("ENABLE_RUNTIME_V2", &(runtime2_env[0U]), static_cast<uint32_t>(MMPA_MAX_PATH));
}
} // namespace ge

