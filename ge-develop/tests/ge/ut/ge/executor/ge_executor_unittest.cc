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
#include <memory>

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "runtime/mem.h"
#include "common/util.h"
#include "omg/omg_inner_types.h"
#include "ge_graph_dsl/graph_dsl.h"

#include "macro_utils/dt_public_scope.h"
#include "executor/ge_executor.h"
#include "single_op/stream_resource.h"

#include "common/helper/file_saver.h"
#include "common/profiling/profiling_properties.h"
#include "common/debug/log.h"
#include "common/types.h"
#include "graph/load/graph_loader.h"
#include "graph/load/model_manager/davinci_model.h"
#include "hybrid/hybrid_davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/task_info/fe/kernel_task_info.h"
#include "graph/load/model_manager/task_info/aicpu/kernel_ex_task_info.h"
#include "graph/execute/graph_executor.h"
#include "common/dump/dump_properties.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/graph_manager.h"
#include "graph/utils/graph_utils.h"
#include "proto/ge_ir.pb.h"
#include "graph/manager/graph_var_manager.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "framework/ge_runtime_stub/include/faker/ge_model_builder.h"
#include "graph/passes/graph_builder_utils.h"
#include "common/share_graph.h"
#include "common/memory/tensor_trans_utils.h"
#include "single_op/single_op_model.h"
#include "graph/manager/mem_manager.h"
#include "v2/kernel/memory/caching_mem_allocator.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace std;
using namespace testing;

namespace ge {
class UtestGeExecutor : public testing::Test {
 protected:
  void SetUp() {
    unsetenv("FMK_SYSMODE");
    unsetenv("FMK_DUMP_PATH");
    unsetenv("FMK_USE_FUSION");
    unsetenv("DAVINCI_TIMESTAT_ENABLE");
  }

  void TearDown() {
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
    VarManager::Instance(0)->FreeVarMemory();
  }
};

class DModelListener : public ge::ModelListener {
 public:
  DModelListener() {
  };
  Status OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode,
                       std::vector<gert::Tensor> &outputs) {
    GELOGI("In Call back. OnComputeDone");
    return SUCCESS;
  }
};

namespace {
class MockHybridModelExecutor : public ge::hybrid::HybridDavinciModel {
 public:
  Status Execute(const vector<DataBuffer> &inputs,
                 const vector<GeTensorDesc> &input_desc,
                 vector<DataBuffer> &outputs,
                 vector<GeTensorDesc> &output_desc,
                 const rtStream_t stream) override {
    return SUCCESS;
  }
};
}  // namespace

TEST_F(UtestGeExecutor, finalize) {
  GeExecutor ge_executor;
  Status retStatus;

  GeExecutor::is_inited_ = false;
  retStatus = ge_executor.Finalize();
  EXPECT_EQ(retStatus, SUCCESS);

  GeExecutor::is_inited_ = true;
  ProfilingProperties::Instance().SetLoadProfiling(true);
  retStatus = ge_executor.Finalize();
  EXPECT_EQ(retStatus, SUCCESS);
}

shared_ptr<ge::ModelListener> g_label_call_back(new DModelListener());

TEST_F(UtestGeExecutor, test_single_op_exec) {
  GeExecutor exeutor;
  ModelData model_data;
  string model_name = "1234";

  EXPECT_EQ(exeutor.LoadSingleOp(model_name, model_data, nullptr, nullptr), ACL_ERROR_GE_INTERNAL_ERROR);
  EXPECT_EQ(exeutor.LoadDynamicSingleOp(model_name, model_data, nullptr, nullptr), PARAM_INVALID);
}

TEST_F(UtestGeExecutor, test_ge_initialize) {
  GeExecutor executor;
  EXPECT_EQ(executor.Initialize(), SUCCESS);
  EXPECT_EQ(executor.Initialize(), SUCCESS);
}

TEST_F(UtestGeExecutor, load_data_from_file) {
  GeExecutor ge_executor;
  ModelData model_data;
  Status retStatus;
  string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.LoadDataFromFile(test_smap, model_data);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.LoadDataFromFile(std::string(""), model_data);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);
  EXPECT_EQ(model_data.model_data, nullptr);

  string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);
  retStatus = ge_executor.LoadDataFromFile(test_smap, model_data);
  EXPECT_EQ(retStatus, SUCCESS);
  EXPECT_NE(model_data.model_data, nullptr);

  delete[] static_cast<char *>(model_data.model_data);
  model_data.model_data = nullptr;
}

TEST_F(UtestGeExecutor, load_model_from_data) {
  GeExecutor ge_executor;
  ModelData model_data;
  Status retStatus;
  uint32_t model_id = 0;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0, nullptr, 0);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  model_id = std::numeric_limits<uint32_t>::max();
  retStatus = ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0, nullptr, 0);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_PARAM_INVALID);
  model_data.model_data = (void *) &model_data;
  retStatus = ge_executor.LoadModelFromData(model_id, model_data, nullptr, 0, nullptr, 0);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(UtestGeExecutor, load_model_from_data_with_Arg) {
  GeExecutor ge_executor;
  ModelData model_data;
  Status retStatus;
  uint32_t model_id = 0;
  ModelLoadArg arg;
  gert::RtSession session;
  session.SetExternalVar(nullptr, 0);
  void *external_var = &model_id;
  uint64_t var_size = 99;
  session.GetExternalVar(external_var, var_size);
  EXPECT_EQ(external_var, nullptr);
  EXPECT_EQ(var_size, 0);
  arg.rt_session = &session;
  arg.dev_ptr = nullptr;
  arg.mem_size = 0U;
  arg.weight_ptr = nullptr;
  arg.weight_size = 0U;
  ge_executor.is_inited_ = false;
  retStatus = ge_executor.LoadModelFromDataWithArgs(model_id, model_data, arg);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  model_id = std::numeric_limits<uint32_t>::max();
  retStatus = ge_executor.LoadModelFromDataWithArgs(model_id, model_data, arg);
  EXPECT_NE(retStatus, SUCCESS);
  model_data.model_data = (void *) &model_data;
  retStatus = ge_executor.LoadModelFromDataWithArgs(model_id, model_data, arg);
  EXPECT_NE(retStatus, SUCCESS);
  session.DestroyResources();
}

TEST_F(UtestGeExecutor, load_model_with_Q) {
  GeExecutor ge_executor;
  ModelData model_data;
  uint32_t model_id = 0;
  Status retStatus;

  std::vector<uint32_t> input_queue_ids = {0,1,2};
  std::vector<uint32_t> output_queue_ids = {3,4,5};
  QueueAttrs in_queue_0 = {.queue_id = 0, .device_type = NPU, .device_id = 0};
  QueueAttrs in_queue_1 = {.queue_id = 1, .device_type = NPU, .device_id = 0};
  QueueAttrs in_queue_2 = {.queue_id = 2, .device_type = NPU, .device_id = 0};
  std::vector<QueueAttrs> input_queue_attrs = {in_queue_0, in_queue_1, in_queue_2};
  QueueAttrs out_queue_0 = {.queue_id = 3, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_1 = {.queue_id = 4, .device_type = NPU, .device_id = 0};
  QueueAttrs out_queue_2 = {.queue_id = 5, .device_type = NPU, .device_id = 0};
  std::vector<QueueAttrs> out_queue_attrs = {out_queue_0, out_queue_1, out_queue_2};

  ModelQueueParam model_queue_param{};
  model_queue_param.input_queues = input_queue_ids;
  model_queue_param.output_queues = output_queue_ids;
  model_queue_param.input_queues_attrs = input_queue_attrs;
  model_queue_param.output_queues_attrs = out_queue_attrs;

  ge_executor.is_inited_ = false;
  ModelQueueArg args{input_queue_ids, output_queue_ids, {}};
  retStatus = ge_executor.LoadModelWithQ(model_id, model_data, args);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);
  retStatus = ge_executor.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);
  retStatus = ge_executor.LoadModelWithQueueParam(model_id, model_data, model_queue_param);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.LoadModelWithQ(model_id, model_data, input_queue_ids, output_queue_ids);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);

  retStatus = ge_executor.LoadModelWithQ(model_id, model_data, args);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
  retStatus = ge_executor.LoadModelWithQueueParam(model_id, model_data, model_queue_param);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);

  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);

  shared_ptr<GeRootModel> root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(graph), SUCCESS);

  retStatus = ge_executor.LoadModelWithQ(model_id, root_model, model_queue_param);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  std::shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  model_task_def->add_task()->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_END_GRAPH));
  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);

  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  ModelBufferData model_buffer;
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  
  ge_model->SetGraph(graph);
  EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
  model_data = {model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
  retStatus = ge_executor.LoadModelWithQueueParam(model_id, model_data, model_queue_param);
  EXPECT_NE(retStatus, SUCCESS);
}

TEST_F(UtestGeExecutor, load_model_without_Q) {
  uint32_t model_id = 0;
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);

  shared_ptr<GeRootModel> root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(root_model->Initialize(graph), SUCCESS);
  GeExecutor ge_executor;
  ge_executor.is_inited_ = true;
  Status retStatus = ge_executor.LoadModelWithoutQ(model_id, root_model);
  EXPECT_EQ(retStatus, INTERNAL_ERROR);
}

TEST_F(UtestGeExecutor, exec_model) {
  GeExecutor ge_executor;
  ModelBufferData model_buffer;
  ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
  uint32_t model_id = 0;
  Status retStatus;
  uint64_t *stream = nullptr;

  int64_t arg_0 = 127;
  int64_t arg_1 = 100;
  RunModelData input_data;
  input_data.blobs.emplace_back(DataBuffer{&arg_0, sizeof(arg_0), false, 0});
  input_data.blobs.emplace_back(DataBuffer{&arg_1, sizeof(arg_1), false, 0});

  int64_t arg_3 = 111;
  RunModelData output_data;
  output_data.blobs.emplace_back(DataBuffer{&arg_3, sizeof(arg_3), false, 0});

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.ExecModel(model_id, stream, input_data, output_data, false);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.ExecModel(model_id, stream, input_data, output_data, false);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  input_data.dynamic_batch_size = 4;
  retStatus = ge_executor.ExecModel(model_id, stream, input_data, output_data, false);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestGeExecutor, exec_dynamic_model) {
  GeExecutor ge_executor;
  ModelBufferData model_buffer;
  ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
  uint32_t model_id = 0;
  uint64_t *stream = nullptr;

  int64_t arg_0 = 127;
  int64_t arg_1 = 100;
  RunModelData input_data;
  input_data.blobs.emplace_back(DataBuffer{&arg_0, sizeof(arg_0), false, 0});
  input_data.blobs.emplace_back(DataBuffer{&arg_1, sizeof(arg_1), false, 0});

  RunModelData output_data;
  output_data.blobs.emplace_back(DataBuffer{nullptr, 0, false, 0});

  GeTensorDesc tensor_desc(GeShape(std::vector<int64_t>{}));
  std::vector<GeTensorDesc> input_desc{tensor_desc};
  std::vector<GeTensorDesc> output_desc{tensor_desc};

  ge_executor.is_inited_ = true;
  ModelManager::GetInstance().hybrid_model_map_[model_id] = make_shared<MockHybridModelExecutor>();
  EXPECT_EQ(ge_executor.ExecModel(model_id, stream, input_data, input_desc, output_data, output_desc, true), SUCCESS);
  EXPECT_EQ(ge_executor.ExecModel(model_id, stream, input_data, input_desc, output_data, output_desc, false), SUCCESS);
  ModelManager::GetInstance().hybrid_model_map_.erase(model_id);
}

TEST_F(UtestGeExecutor, get_mem_and_weight_size1) {
  GeExecutor ge_executor;
  ModelData model_data;
  Status retStatus;

  ge_executor.is_inited_ = false;
  size_t weightSize;
  size_t memSize;
  retStatus = ge_executor.GetMemAndWeightSize("", memSize, weightSize);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetMemAndWeightSize("", memSize, weightSize);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_PATH_INVALID);

  string test_smap = "/tmp/" + std::to_string(getpid()) + "_maps";
  string self_smap = "/proc/" + std::to_string(getpid()) + "/maps";
  string copy_smap = "cp -f " + self_smap + " " + test_smap;
  EXPECT_EQ(system(copy_smap.c_str()), 0);
  retStatus = ge_executor.GetMemAndWeightSize(test_smap, memSize, weightSize);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestGeExecutor, get_mem_and_weight_size2) {
  GeExecutor ge_executor;
  ModelData model_data;
  Status retStatus;

  ge_executor.is_inited_ = false;
  size_t weightSize;
  size_t memSize;
  retStatus = ge_executor.GetMemAndWeightSize(&model_data, sizeof(model_data), memSize, weightSize);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetMemAndWeightSize(nullptr, sizeof(model_data), memSize, weightSize);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ADDR_INVALID);

  retStatus = ge_executor.GetMemAndWeightSize(&model_data, sizeof(model_data), memSize, weightSize);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_PARAM_INVALID);

  ModelBufferData model_buffer;
  ModelData model_data2{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", "/tmp/a"};
  retStatus = ge_executor.GetMemAndWeightSize(&model_data2, sizeof(model_data2), memSize, weightSize);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestGeExecutor, load_single_op) {
  GeExecutor ge_executor;
  SingleOp *single_op;
  Status retStatus;

  ModelBufferData model_buffer;
  ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", "/tmp/a"};
  retStatus = ge_executor.LoadSingleOp(std::string("/tmp/a"), model_data, nullptr, &single_op);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_PARAM_INVALID);

  retStatus = ge_executor.ReleaseSingleOpResource(nullptr);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExecutor, execute_async1) {
  GeExecutor ge_executor;
  Status retStatus;
  int64_t arg_0 = 127;
  int64_t arg_1 = 100;
  std::vector<DataBuffer> inputs;
  std::vector<DataBuffer> outputs;
  inputs.emplace_back(DataBuffer{&arg_0, sizeof(arg_0), false, 0});
  outputs.emplace_back(DataBuffer{&arg_1, sizeof(arg_1), false, 0});

  retStatus = ge_executor.ExecuteAsync(nullptr, inputs, outputs);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOp *single_op = new (std::nothrow) SingleOp(res, &stream_mu, stream);
  SingleOpModelParam model_params;
  single_op->impl_->model_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  retStatus = ge_executor.ExecuteAsync(single_op, inputs, outputs);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_PARAM_INVALID);
  delete single_op;
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestGeExecutor, execute_async2) {
  GeExecutor ge_executor;
  Status retStatus;
  int64_t arg_0 = 127;
  int64_t arg_1 = 100;

  std::vector<GeTensorDesc> input_desc;
  std::vector<GeTensorDesc> output_desc;

  std::vector<DataBuffer> inputs;
  std::vector<DataBuffer> outputs;
  inputs.emplace_back(DataBuffer{&arg_0, sizeof(arg_0), false, 0});
  outputs.emplace_back(DataBuffer{&arg_1, sizeof(arg_1), false, 0});

  std::mutex stream_mu_;
  DynamicSingleOp dynamic_single_op(nullptr, 0, &stream_mu_, nullptr);
  retStatus = ge_executor.ExecuteAsync(&dynamic_single_op, input_desc, inputs, output_desc, outputs);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestGeExecutor, clear_custom_aicpu_so) {
  GeExecutor ge_executor;
  Status retStatus;

  ModelManager mm;

  // cust_aicpu_so_ is empty.
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("empty_cust_aicpu"), SUCCESS);

  // deleteCustOp after Launch will deleted.
  uintptr_t resource_id = 1;    // for rtCtxGetCurrent stub
  std::vector<char> kernel_bin(256);
  auto &cust_resource_001 = mm.cust_aicpu_so_[resource_id];
  auto tbe_kernel = std::shared_ptr<OpKernelBin>(new OpKernelBin("deleteCustOp", std::move(kernel_bin)));
  cust_resource_001["deleteCustOp"] = {tbe_kernel, 1UL};

  EXPECT_FALSE(mm.cust_aicpu_so_.empty());
  EXPECT_EQ(mm.LaunchKernelCustAicpuSo("deleteCustOp"), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["deleteCustOp"].launch_count, 0UL);
  EXPECT_EQ(mm.LaunchCustAicpuSo(), SUCCESS);
  EXPECT_EQ(mm.cust_aicpu_so_[resource_id]["deleteCustOp"].launch_count, 1UL);
  uint32_t device_id = 256U;
  retStatus = ge_executor.ClearCustomAicpuSo(device_id);
  EXPECT_EQ(retStatus, SUCCESS);
}

namespace {
class UtestGeExeWithDavModel : public testing::Test {
 public:
  ComputeGraphPtr graph;
  shared_ptr<DavinciModel> model;
  shared_ptr<hybrid::HybridDavinciModel> hybrid_model;
  uint32_t model_id1;
  uint32_t model_id2;

 protected:
  void SetUp() {
    unsetenv("FMK_SYSMODE");
    unsetenv("FMK_DUMP_PATH");
    unsetenv("FMK_USE_FUSION");
    unsetenv("DAVINCI_TIMESTAT_ENABLE");

    VarManager::Instance(0)->Init(0, 0, 0, 0);
    map<string, string> options;
    options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
    VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
    graph = make_shared<ComputeGraph>("default");

    model_id1 = 1;
    model = MakeShared<DavinciModel>(1, g_label_call_back);
    model->SetId(model_id1);
    model->om_name_ = "testom";
    model->name_ = "test";
    ModelManager::GetInstance().InsertModel(model_id1, model);

    model_id2 = 2;
    hybrid_model = MakeShared<hybrid::HybridDavinciModel>();
    model->SetId(model_id2);
    model->om_name_ = "testom_hybrid";
    model->name_ = "test_hybrid";
    ModelManager::GetInstance().InsertModel(model_id2, hybrid_model);

    model->ge_model_ = make_shared<GeModel>();
    model->ge_model_->SetGraph(graph);
    model->runtime_param_.mem_base = 0x08000000;
    model->runtime_param_.mem_size = 5120000;
  }
  void TearDown() {
    ModelManager::GetInstance().DeleteModel(model_id1);
    ModelManager::GetInstance().DeleteModel(model_id2);
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
  }

  static ComputeGraphPtr BuildSubComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
    auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
    auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
    builder.AddDataEdge(data, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    return graph;
  }

  static ComputeGraphPtr BuildComputeGraph() {
    ut::GraphBuilder builder = ut::GraphBuilder("graph");
    auto data = builder.AddNode("Data", "Data", 1, 1);
    auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
    transdata->GetOpDesc()->AddSubgraphName("subgraph");
    transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
    auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 1);
    builder.AddDataEdge(data, 0, transdata, 0);
    builder.AddDataEdge(transdata, 0, netoutput, 0);
    auto graph = builder.GetGraph();
    // add subgraph
    transdata->SetOwnerComputeGraph(graph);
    ComputeGraphPtr subgraph = BuildSubComputeGraph();
    subgraph->SetParentGraph(graph);
    subgraph->SetParentNode(transdata);
    graph->AddSubgraph("subgraph", subgraph);
    auto op_desc = netoutput->GetOpDesc();
    std::vector<std::string> src_name{"out1", "out2"};
    op_desc->SetSrcName(src_name);
    std::vector<int64_t> src_index{0, 1};
    op_desc->SetSrcIndex(src_index);

    ge::GeTensorDesc tensor_desc;
    tensor_desc.SetOriginFormat(ge::FORMAT_NCHW);
    tensor_desc.SetFormat(ge::FORMAT_FRACTAL_Z);
    tensor_desc.SetDataType(ge::DT_FLOAT16);
    tensor_desc.SetShape(ge::GeShape({3, 12, 5, 6}));
    std::vector<std::pair<int64_t, int64_t>> range{{1, 5}, {2, 5}, {3, 6}, {4, 7}};
    tensor_desc.SetShapeRange(range);
    op_desc->AddInputDesc("x", tensor_desc);

    return graph;
  }

  ComputeGraphPtr BuildCaseGraph() {
    ComputeGraphPtr graph = gert::ShareGraph::CaseGraph();
    NodePtr case_node = graph->FindNode("case");
    OpDescPtr op_desc = case_node->GetOpDesc();

    int32_t dynamic_type = ge::DYNAMIC_BATCH;
    AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type);
    uint32_t batch_num = 2U;
    AttrUtils::SetInt(op_desc, ATTR_NAME_BATCH_NUM, batch_num);
    for (uint32_t i = 0U; i < batch_num; ++i) {
      const std::string attr_name = ATTR_NAME_PRED_VALUE + "_" + std::to_string(i);
      std::vector<int64_t> batch_shape = {1, 2};
      AttrUtils::SetListInt(op_desc, attr_name, batch_shape);
    }
    std::vector<std::string> user_designate_shape_order = {"data1", "data2"};
    AttrUtils::SetListStr(op_desc, ATTR_USER_DESIGNEATE_SHAPE_ORDER, user_designate_shape_order);

    return graph;
  }
};
}

TEST_F(UtestGeExeWithDavModel, get_cur_shape) {
  GeExecutor ge_executor;
  Status retStatus;
  std::vector<int64_t> batch_info;
  int32_t dynamic_type;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetCurShape(model_id1, batch_info, dynamic_type);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetCurShape(-1, batch_info, dynamic_type);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  retStatus = ge_executor.GetCurShape(model_id1, batch_info, dynamic_type);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExeWithDavModel, set_synamic_aipp_data) {
  GeExecutor ge_executor;
  Status retStatus;
  void *dynamic_input_addr = nullptr;
  uint64_t length = 0;
  std::vector<kAippDynamicBatchPara> aipp_batch_para;
  kAippDynamicPara aipp_parms;

  retStatus = ge_executor.SetDynamicAippData(model_id1, dynamic_input_addr,
                                             length, aipp_batch_para, aipp_parms);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID);

  dynamic_input_addr = malloc(sizeof(kAippDynamicBatchPara)*2);
  retStatus = ge_executor.SetDynamicAippData(model_id1, dynamic_input_addr,
                                             length, aipp_batch_para, aipp_parms);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_AIPP_BATCH_EMPTY);

  kAippDynamicBatchPara data1;
  aipp_batch_para.push_back(data1);
  retStatus = ge_executor.SetDynamicAippData(model_id1, dynamic_input_addr,
                                             length, aipp_batch_para, aipp_parms);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID);

  length = sizeof(kAippDynamicBatchPara)*2;
  retStatus = ge_executor.SetDynamicAippData(model_id1, dynamic_input_addr,
                                             length, aipp_batch_para, aipp_parms);
  EXPECT_EQ(retStatus, SUCCESS);

  free(dynamic_input_addr);
}

TEST_F(UtestGeExeWithDavModel, unload_model) {
  GeExecutor ge_executor;
  Status retStatus;

  retStatus = ge_executor.UnloadModel(-1);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  retStatus = ge_executor.UnloadModel(model_id2);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge_executor.UnloadModel(model_id1);
  EXPECT_EQ(retStatus, SUCCESS);

  retStatus = ge_executor.UnloadModel(model_id1);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestGeExeWithDavModel, cmd_handler) {
  GeExecutor ge_executor;
  struct Command cmd;
  Status retStatus;

  retStatus = ge_executor.CommandHandle(cmd);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_COMMAND_HANDLE);

  cmd.cmd_type = "dump";
  cmd.cmd_params.push_back(std::string("off"));
  cmd.module_index = 1;
  retStatus = ge_executor.CommandHandle(cmd);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_COMMAND_HANDLE);
}

TEST_F(UtestGeExeWithDavModel, get_max_used_mem) {
  GeExecutor ge_executor;
  uint32_t max_size;
  Status retStatus;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetMaxUsedMemory(model_id1, max_size);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetMaxUsedMemory(model_id1, max_size);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExeWithDavModel, get_used_design_shape_order) {
  GeExecutor ge_executor;
  std::vector<std::string> user_designate_shape_order;
  Status retStatus;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetUserDesignateShapeOrder(model_id1, user_designate_shape_order);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetUserDesignateShapeOrder(-1, user_designate_shape_order);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  retStatus = ge_executor.GetUserDesignateShapeOrder(model_id1, user_designate_shape_order);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExeWithDavModel, get_combine_dynamic_dims) {
  GeExecutor ge_executor;
  std::vector<std::vector<int64_t>> batch_info;
  Status retStatus;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetCombinedDynamicDims(model_id1, batch_info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetCombinedDynamicDims(-1, batch_info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  retStatus = ge_executor.GetCombinedDynamicDims(model_id1, batch_info);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExeWithDavModel, get_aipp_info) {
  GeExecutor ge_executor;
  uint32_t index = 0;
  AippConfigInfo aipp_info;
  Status retStatus;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetAIPPInfo(model_id1, index, aipp_info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetAIPPInfo(model_id1, index, aipp_info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(UtestGeExeWithDavModel, get_aipp_type) {
  GeExecutor ge_executor;
  uint32_t index = 0;
  InputAippType type = DATA_WITHOUT_AIPP;
  size_t aipp_index;
  Status retStatus;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetAippType(model_id1, index, type, aipp_index);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetAippType(model_id1, index, type, aipp_index);
  EXPECT_EQ(retStatus, PARAM_INVALID);
}

TEST_F(UtestGeExeWithDavModel, get_op_attr) {
  OpDescPtr op_desc = CreateOpDesc("test", "test");
  std::vector<std::string> value{"test"};
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, value);

  model->SaveSpecifyAttrValues(op_desc);

  GeExecutor ge_executor;
  GeExecutor::is_inited_ = true;
  std::string attr_value;
  auto ret = ge_executor.GetOpAttr(1, "test", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, attr_value);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(attr_value, "[4]test");
  ret = ge_executor.GetOpAttr(2, "test", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, attr_value);
  EXPECT_EQ(ret, PARAM_INVALID);
  ret = ge_executor.GetOpAttr(3, "test", ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, attr_value);
  EXPECT_EQ(ret, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);
}

TEST_F(UtestGeExeWithDavModel, get_model_attr) {
  OpDescPtr op_desc = CreateOpDesc("add0", "Add");
  std::vector<std::string> value{"66"};
  ge::AttrUtils::SetListStr(op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, value);
  model->SaveSpecifyAttrValues(op_desc);

  GeExecutor ge_executor;
  Status retStatus;
  std::vector<std::string> str_info;

  GeExecutor::is_inited_ = false;
  retStatus = ge_executor.GetModelAttr(model_id1, str_info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  GeExecutor::is_inited_ = true;
  retStatus = ge_executor.GetModelAttr(-1, str_info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  GeExecutor::is_inited_ = true;
  retStatus = ge_executor.GetModelAttr(model_id1, str_info);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExeWithDavModel, get_deviceId_by_modelId) {
  GeExecutor ge_executor;
  uint32_t device_id;
  Status retStatus;

  retStatus = ge_executor.GetDeviceIdByModelId(-1, device_id);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  retStatus = ge_executor.GetDeviceIdByModelId(model_id1, device_id);
  EXPECT_EQ(retStatus, SUCCESS);
  EXPECT_EQ(device_id, model->device_id_);
}

TEST_F(UtestGeExeWithDavModel, get_batch_info_size) {
  GeExecutor ge_executor;
  size_t shape_count;
  Status retStatus;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetBatchInfoSize(model_id1, shape_count);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetBatchInfoSize(model_id1, shape_count);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExeWithDavModel, get_all_aipp_inputOutput_dims) {
  GeExecutor ge_executor;
  OriginInputInfo info;
  Status retStatus;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetOrigInputInfo(model_id1, 0, info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetOrigInputInfo(model_id1, 0, info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_AIPP_NOT_EXIST);
}

TEST_F(UtestGeExeWithDavModel, set_dynamic_batch_size) {
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr data_desc = CreateOpDesc("data", DATA);
  data_desc->AddInputDesc(tensor);
  data_desc->AddOutputDesc(tensor);
  data_desc->SetInputOffset({1024});
  data_desc->SetOutputOffset({1024});
  graph->AddNode(data_desc);

  std::vector<string> inputs = { "NCHW:DT_FLOAT:inputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:outputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);
  std::vector<NodePtr> variable_nodes;
  ASSERT_EQ(model->InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  GeExecutor ge_executor;
  Status retStatus;
  uint64_t length = 0;
  uint64_t batch_size = 4;

  retStatus = ge_executor.SetDynamicBatchSize(model_id1, nullptr, length, batch_size);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID);

  constexpr int SIZE = 32;
  uint64_t *dynamic_input_addr = new uint64_t[SIZE];
  ge_executor.is_inited_ = false;
  retStatus = ge_executor.SetDynamicBatchSize(model_id1, dynamic_input_addr, length, batch_size);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID);

  length = sizeof(uint64_t)*SIZE;
  retStatus = ge_executor.SetDynamicBatchSize(-1, dynamic_input_addr, length, batch_size);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.SetDynamicBatchSize(model_id1, dynamic_input_addr, length, batch_size);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID);

  delete[] dynamic_input_addr;
}

TEST_F(UtestGeExeWithDavModel, set_dynamic_image_size) {
  vector<int64_t> dim(4, 4);
  GeTensorDesc tensor(GeShape(dim), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr data_desc = CreateOpDesc("data", DATA);
  data_desc->AddInputDesc(tensor);
  data_desc->AddOutputDesc(tensor);
  data_desc->SetInputOffset({1024});
  data_desc->SetOutputOffset({1024});
  graph->AddNode(data_desc);

  std::vector<string> inputs = { "NCHW:DT_FLOAT:inputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:outputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);
  std::vector<NodePtr> variable_nodes;
  ASSERT_EQ(model->InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  GeExecutor ge_executor;
  Status retStatus;
  uint64_t length = 0;
  uint64_t height = 0;
  uint64_t width = 0;

  retStatus = ge_executor.SetDynamicImageSize(model_id1, nullptr, length, height, width);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID);

  constexpr int SIZE = 32;
  uint64_t *dynamic_input_addr = new uint64_t[SIZE];
  retStatus = ge_executor.SetDynamicImageSize(model_id1, dynamic_input_addr, length, height, width);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_INPUT_LENGTH_INVALID);

  length = sizeof(uint64_t)*SIZE;
  ge_executor.is_inited_ = false;
  retStatus = ge_executor.SetDynamicImageSize(-1, dynamic_input_addr, length, height, width);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.SetDynamicImageSize(model_id1, dynamic_input_addr, length, height, width);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID);

  delete[] dynamic_input_addr;
}

TEST_F(UtestGeExeWithDavModel, dynamic_dims_test) {
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr data_desc = CreateOpDesc("data", DATA);
  data_desc->AddInputDesc(tensor);
  data_desc->AddOutputDesc(tensor);
  data_desc->SetInputOffset({1024});
  data_desc->SetOutputOffset({1024});
  graph->AddNode(data_desc);

  std::vector<string> inputs = { "NCHW:DT_FLOAT:inputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:outputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);
  std::vector<NodePtr> variable_nodes;
  ASSERT_EQ(model->InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  GeExecutor ge_executor;
  Status retStatus;
  std::vector<uint64_t> dynamic_dims;
  std::vector<uint64_t> cur_dynamic_dims;

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetCurDynamicDims(model_id1, dynamic_dims, cur_dynamic_dims);
  EXPECT_EQ(dynamic_dims.empty(), true);
  EXPECT_EQ(cur_dynamic_dims.empty(), true);

  uint64_t length = 0;
  retStatus = ge_executor.SetDynamicDims(model_id1, nullptr, length, dynamic_dims);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_INPUT_ADDR_INVALID);

  constexpr int SIZE = 32;
  uint64_t *dynamic_input_addr = new uint64_t[SIZE];
  ge_executor.is_inited_ = false;
  retStatus = ge_executor.SetDynamicDims(model_id1, dynamic_input_addr, length, dynamic_dims);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.SetDynamicDims(-1, dynamic_input_addr, length, dynamic_dims);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  retStatus = ge_executor.SetDynamicDims(model_id1, dynamic_input_addr, length, dynamic_dims);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_DYNAMIC_BATCH_SIZE_INVALID);

  delete[] dynamic_input_addr;
}

TEST_F(UtestGeExeWithDavModel, get_and_set_dims_info) {
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr data_desc;
  data_desc = CreateOpDesc("data", DATA);
  data_desc->AddInputDesc(tensor);
  data_desc->AddOutputDesc(tensor);
  data_desc->SetInputOffset({1024});
  data_desc->SetOutputOffset({1024});
  graph->AddNode(data_desc);

  std::vector<string> inputs = { "NCHW:DT_FLOAT:inputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:outputName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);
  std::vector<NodePtr> variable_nodes;
  ASSERT_EQ(model->InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  GeExecutor ge_executor;
  Status retStatus;
  std::vector<InputOutputDims> input_dims;
  std::vector<InputOutputDims> output_dims;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetAllAippInputOutputDims(model_id1, 0, input_dims, output_dims);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetAllAippInputOutputDims(-1, 0, input_dims, output_dims);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_ID_INVALID);

  retStatus = ge_executor.GetAllAippInputOutputDims(model_id1, 0, input_dims, output_dims);
  EXPECT_EQ(retStatus, SUCCESS);
  EXPECT_EQ(input_dims.front().name=="inputName", true);
}

TEST_F(UtestGeExeWithDavModel, get_modelDesc_info) {
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr data_desc;
  data_desc = CreateOpDesc("data", DATA);
  data_desc->AddInputDesc(tensor);
  data_desc->AddOutputDesc(tensor);
  data_desc->SetInputOffset({1024});
  data_desc->SetOutputOffset({1024});
  graph->AddNode(data_desc);

  std::vector<string> inputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_INPUTS, inputs);
  std::vector<string> outputs = { "NCHW:DT_FLOAT:TensorName:TensorSize:3:1,2,8" };
  AttrUtils::SetListStr(data_desc, ATTR_NAME_AIPP_OUTPUTS, outputs);
  std::vector<NodePtr> variable_nodes;
  ASSERT_EQ(model->InitIoNodes(graph, variable_nodes), SUCCESS);
  EXPECT_TRUE(variable_nodes.empty());

  GeExecutor ge_executor;
  Status retStatus;
  std::vector<TensorDesc> input_desc;
  std::vector<TensorDesc> output_desc;

  ge_executor.is_inited_ = false;
  retStatus = ge_executor.GetModelDescInfo(model_id1, input_desc, output_desc, false);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_NOT_INIT);

  ge_executor.is_inited_ = true;
  retStatus = ge_executor.GetModelDescInfo(model_id1, input_desc, output_desc, false);
  EXPECT_EQ(retStatus, SUCCESS);
  EXPECT_EQ(input_desc.front().GetFormat(), FORMAT_NCHW);
}

TEST_F(UtestGeExeWithDavModel, get_model_desc_info_from_mem) {
  GeExecutor ge_executor;
  Status retStatus;
  ModelBufferData model_buffer;
  ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", "/tmp/a"};
  ModelInOutInfo info;
  retStatus = ge_executor.GetModelDescInfoFromMem(model_data, info);
  EXPECT_EQ(retStatus, ACL_ERROR_GE_EXEC_MODEL_DATA_SIZE_INVALID);
}

TEST_F(UtestGeExeWithDavModel, get_model_desc_info_from_mem_success) {
  GeModelPtr ge_model = std::make_shared<GeModel>();
  std::shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  model_task_def->add_task()->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_END_GRAPH));
  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);

  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  ModelBufferData model_buffer;
  ModelHelper model_helper;
  model_helper.SetSaveMode(false);  // Save to buffer.
  ModelInOutInfo info;
  GeExecutor ge_executor;

  {
    // this graph does not contain the opdesc of case type.
    ComputeGraphPtr graph = BuildComputeGraph();
    ge_model->SetGraph(graph);
    EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
    ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
    EXPECT_EQ(ge_executor.GetModelDescInfoFromMem(model_data, info), SUCCESS);
  }

  {
    // this graph mainly tests the opdesc of case type.
    ComputeGraphPtr graph = BuildCaseGraph();
    ge_model->SetGraph(graph);
    EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
    ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
    EXPECT_EQ(ge_executor.GetModelDescInfoFromMem(model_data, info), SUCCESS);
  }

  {
    ComputeGraphPtr graph = BuildCaseGraph();
    OpDescPtr op_desc = graph->FindNode("case")->GetOpDesc();
    int32_t dynamic_type = ge::DYNAMIC_IMAGE;
    AttrUtils::SetInt(op_desc, ATTR_DYNAMIC_TYPE, dynamic_type);
    ge_model->SetGraph(graph);
    EXPECT_EQ(model_helper.SaveToOmModel(ge_model, "file_name_prefix", model_buffer), SUCCESS);
    ModelData model_data{model_buffer.data.get(), static_cast<uint32_t>(model_buffer.length), 0, "", ""};
    EXPECT_EQ(ge_executor.GetModelDescInfoFromMem(model_data, info), SUCCESS);
  }
}

TEST_F(UtestGeExecutor, get_op_desc_info) {
  ComputeGraphPtr graph = make_shared<ComputeGraph>("default");
  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("data", DATA);
  op_desc->AddInputDesc(tensor);
  op_desc->AddOutputDesc(tensor);
  op_desc->SetInputOffset({1024});
  op_desc->SetOutputOffset({1024});
  NodePtr node = graph->AddNode(op_desc);

  GeExecutor ge_executor;
  OpDescInfo op_desc_info;
  Status retStatus;

  retStatus = ge_executor.GetOpDescInfo(0, 0, 0, op_desc_info);
  EXPECT_EQ(retStatus, FAILED);
}

TEST_F(UtestGeExecutor, set_dump) {
  GeExecutor ge_executor;
  DumpConfig dump_config;
  Status retStatus;

  retStatus = ge_executor.SetDump(dump_config);
  EXPECT_EQ(retStatus, SUCCESS);
}

TEST_F(UtestGeExecutor, InitFeatureMapAndP2PMem_failed) {
  DavinciModel model(0, g_label_call_back);
  model.is_feature_map_mem_has_inited_ = true;
  EXPECT_EQ(model.InitFeatureMapAndP2PMem(0U, 0U), PARAM_INVALID);
}

TEST_F(UtestGeExecutor, kernel_ex_InitDumpArgs) {
  DavinciModel model(0, g_label_call_back);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");

  std::map<std::string, std::set<std::string>> model_dump_properties_map;
  std::set<std::string> s;
  model_dump_properties_map[DUMP_ALL_MODEL] = s;
  DumpProperties dp;
  dp.model_dump_properties_map_ = model_dump_properties_map;
  model.SetDumpProperties(dp);

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_NO_THROW(kernel_ex_task_info.InitDumpArgs(nullptr, op_desc));
}

TEST_F(UtestGeExecutor, kernel_ex_InitDumpFlag) {
  DavinciModel model(0, g_label_call_back);
  model.om_name_ = "testom";
  model.name_ = "test";
  OpDescPtr op_desc = CreateOpDesc("test", "test");

  std::map<std::string, std::set<std::string>> model_dump_properties_map;
  std::set<std::string> s;
  model_dump_properties_map[DUMP_ALL_MODEL] = s;
  DumpProperties dp;
  dp.model_dump_properties_map_ = model_dump_properties_map;
  model.SetDumpProperties(dp);

  KernelExTaskInfo kernel_ex_task_info;
  kernel_ex_task_info.davinci_model_ = &model;
  EXPECT_NO_THROW(kernel_ex_task_info.InitDumpFlag(op_desc));
}

TEST_F(UtestGeExecutor, execute_graph_with_stream) {
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  map<string, string> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
  VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("square", "Square");
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_square_123");
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"dump"}));

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_op_index(op_desc->GetId());
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({5120});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 2

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(5120);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({5120});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  setenv("SKT_ENABLE", "1", 1);
  const auto model = MakeShared<DavinciModel>(0, nullptr);
  model->Assign(ge_model);
  EXPECT_EQ(model->Init(), SUCCESS);

  EXPECT_EQ(model->input_addrs_list_.size(), 1);
  EXPECT_EQ(model->output_addrs_list_.size(), 1);
  EXPECT_EQ(model->task_list_.size(), 2);

  OutputData output_data;
  vector<gert::Tensor> outputs;
  EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);

  GraphExecutor graph_executer;
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  std::vector<GeTensor> input_tensor;
  std::vector<GeTensor> output_tensor;
  std::vector<InputOutputDescInfo> output_desc;
  InputOutputDescInfo desc0;
  output_desc.push_back(desc0);

  ge_root_model->SetModelId(1001U);
  ModelManager::GetInstance().InsertModel(ge_root_model->GetModelId(), model);
  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1);
  EXPECT_EQ(graph_executer.ExecuteGraphWithStream(nullptr, graph_node, ge_root_model, input_tensor, output_tensor), PARAM_INVALID);
  std::vector<gert::Tensor> gert_input_tensor;
  std::vector<gert::Tensor> gert_output_tensor;
  EXPECT_EQ(graph_executer.ExecuteGraphWithStream(nullptr, graph_node, ge_root_model, gert_input_tensor,
            gert_output_tensor), PARAM_INVALID);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(ge_root_model->GetModelId()), SUCCESS);
  unsetenv("SKT_ENABLE");
}

TEST_F(UtestGeExecutor, execute_graph_sync) {
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  map<string, string> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
  VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  const char *const kVectorcoreNum = "ge.vectorcoreNum";

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("square", "Square");
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_square_123");
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"dump"}));

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_op_index(op_desc->GetId());
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({5120});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 2

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(5120);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({5120});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  auto mock_listener = MakeShared<GraphModelListener>();
  {
    shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(0, mock_listener);
    model->SetId(0);

    model->Assign(ge_model);
    auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
    graph_options[ge::AICORE_NUM] = "1";
    graph_options[kVectorcoreNum] = "1";
    GetThreadLocalContext().SetGraphOption(graph_options);
    EXPECT_EQ(model->Init(), SUCCESS);

    EXPECT_EQ(model->input_addrs_list_.size(), 1);
    EXPECT_EQ(model->output_addrs_list_.size(), 1);
    EXPECT_EQ(model->task_list_.size(), 2);

    OutputData output_data;
    vector<gert::Tensor> outputs;
    EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);
    ModelManager::GetInstance().InsertModel(0U, model);
  }

  {
    shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(1, mock_listener);
    model->SetId(1);

    model->Assign(ge_model);
    EXPECT_EQ(model->Init(), SUCCESS);

    EXPECT_EQ(model->input_addrs_list_.size(), 1);
    EXPECT_EQ(model->output_addrs_list_.size(), 1);
    EXPECT_EQ(model->task_list_.size(), 2);

    OutputData output_data;
    vector<gert::Tensor> outputs;
    EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);
    ModelManager::GetInstance().InsertModel(1U, model);
  }

  GraphExecutor graph_executer;
  std::vector<gert::Tensor> input_tensor(1);
  TensorCheckUtils::ConstructGertTensor(input_tensor[0], {4, 512}, DT_FLOAT, FORMAT_NCHW);
  std::vector<gert::Tensor> output_tensor;

  {
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetModelId(0);
    ge_root_model->SetModelId(1);
    EXPECT_EQ(graph_executer.ExecuteGraph(0, ge_root_model, input_tensor, output_tensor), SUCCESS);
  }
  output_tensor.clear();
  {
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetModelId(1);
    EXPECT_EQ(graph_executer.ExecuteGraph(0, ge_root_model, input_tensor, output_tensor), GRAPH_SUCCESS);
  }

  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(0U), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(1U), SUCCESS);
}

TEST_F(UtestGeExecutor, execute_graph_sync_gert_tensor) {
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  map<string, string> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
  VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);
  const char *const kVectorcoreNum = "ge.vectorcoreNum";

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("square", "Square");
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_square_123");
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"dump"}));

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_op_index(op_desc->GetId());
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({5120});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 2

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(5120);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({5120});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  auto mock_listener = MakeShared<GraphModelListener>();
  {
    shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(0, mock_listener);
    model->SetId(0);

    model->Assign(ge_model);
    auto graph_options = GetThreadLocalContext().GetAllGraphOptions();
    graph_options[ge::AICORE_NUM] = "1";
    graph_options[kVectorcoreNum] = "1";
    GetThreadLocalContext().SetGraphOption(graph_options);
    EXPECT_EQ(model->Init(), SUCCESS);

    EXPECT_EQ(model->input_addrs_list_.size(), 1);
    EXPECT_EQ(model->output_addrs_list_.size(), 1);
    EXPECT_EQ(model->task_list_.size(), 2);

    OutputData output_data;
    vector<gert::Tensor> outputs;
    EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);
    ModelManager::GetInstance().InsertModel(0U, model);
  }

  {
    shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(1, mock_listener);
    model->SetId(1);

    model->Assign(ge_model);
    EXPECT_EQ(model->Init(), SUCCESS);

    EXPECT_EQ(model->input_addrs_list_.size(), 1);
    EXPECT_EQ(model->output_addrs_list_.size(), 1);
    EXPECT_EQ(model->task_list_.size(), 2);

    OutputData output_data;
    vector<gert::Tensor> outputs;
    EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);
    ModelManager::GetInstance().InsertModel(1U, model);
  }

  GraphExecutor graph_executer;
  InputOutputDescInfo desc0;
  std::vector<InputOutputDescInfo> output_desc{desc0};
  GeTensorDesc td(GeShape({4, 4}), FORMAT_NCHW, DT_FLOAT);
  GeTensor ge_tensor(td);
  vector<uint8_t> data(512, 0);
  ge_tensor.SetData(data);
  std::vector<GeTensor> input_tensor;
  input_tensor.push_back(std::move(ge_tensor));
  std::vector<gert::Tensor> output_tensor;
  std::vector<gert::Tensor> gert_inputs;
  TensorTransUtils::GeTensors2GertTensors(input_tensor, gert_inputs);
  {
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetModelId(0);
    ge_root_model->SetModelId(1);

    EXPECT_EQ(graph_executer.ExecuteGraph(0, ge_root_model, gert_inputs, output_tensor), SUCCESS);
  }
  output_tensor.clear();
  {
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetModelId(1);
    EXPECT_EQ(graph_executer.ExecuteGraph(0, ge_root_model, gert_inputs, output_tensor), GRAPH_SUCCESS);
  }

  {
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetModelId(1);
    EXPECT_NE(graph_executer.ExecuteGraph(0, ge_root_model, gert_inputs, output_tensor), GRAPH_SUCCESS);
  }

  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(0U), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(1U), SUCCESS);
}

TEST_F(UtestGeExecutor, execute_graph_async_multi) {
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  map<string, string> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
  VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("default");

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  {
    OpDescPtr op_desc = CreateOpDesc("data", DATA);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    NodePtr node = graph->AddNode(op_desc);    // op_index = 0
  }

  {
    OpDescPtr op_desc = CreateOpDesc("square", "Square");
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({1024});
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_square_123");
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"dump"}));

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_op_index(op_desc->GetId());
    context->set_kernel_type(2);    // ccKernelType::TE
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    OpDescPtr op_desc = CreateOpDesc("memcpy", MEMCPYASYNC);
    op_desc->AddInputDesc(tensor);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetInputOffset({1024});
    op_desc->SetOutputOffset({5120});
    NodePtr node = graph->AddNode(op_desc);  // op_index = 2

    domi::TaskDef *task_def = model_task_def->add_task();
    task_def->set_stream_id(0);
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_MEMCPY_ASYNC));
    domi::MemcpyAsyncDef *memcpy_async = task_def->mutable_memcpy_async();
    memcpy_async->set_src(1024);
    memcpy_async->set_dst(5120);
    memcpy_async->set_dst_max(512);
    memcpy_async->set_count(1);
    memcpy_async->set_kind(RT_MEMCPY_DEVICE_TO_DEVICE);
    memcpy_async->set_op_index(op_desc->GetId());
  }

  {
    OpDescPtr op_desc = CreateOpDesc("output", NETOUTPUT);
    op_desc->AddInputDesc(tensor);
    op_desc->SetInputOffset({5120});
    op_desc->SetSrcName( { "memcpy" } );
    op_desc->SetSrcIndex( { 0 } );
    NodePtr node = graph->AddNode(op_desc);  // op_index = 3
  }

  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(0, listener);
    model->SetId(0);

    model->Assign(ge_model);
    EXPECT_EQ(model->Init(), SUCCESS);

    EXPECT_EQ(model->input_addrs_list_.size(), 1);
    EXPECT_EQ(model->output_addrs_list_.size(), 1);
    EXPECT_EQ(model->task_list_.size(), 2);

    OutputData output_data;
    vector<gert::Tensor> outputs;
    EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);
    ModelManager::GetInstance().InsertModel(0, model);
  }

  {
    auto listener = MakeShared<RunAsyncListener>();
    shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(1, listener);
    model->SetId(1);

    model->Assign(ge_model);
    EXPECT_EQ(model->Init(), SUCCESS);

    EXPECT_EQ(model->input_addrs_list_.size(), 1);
    EXPECT_EQ(model->output_addrs_list_.size(), 1);
    EXPECT_EQ(model->task_list_.size(), 2);

    OutputData output_data;
    vector<gert::Tensor> outputs;
    EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);
    ModelManager::GetInstance().InsertModel(1, model);
  }

  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(0U), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(1U), SUCCESS);
}

TEST_F(UtestGeExecutor, SetAllocator_NoCoreDump) {
  int32_t fake_stream = 0;
  auto stream = reinterpret_cast<rtStream_t>(fake_stream);
  GeExecutor executor{};
  EXPECT_EQ(executor.SetAllocator(stream, (ge::Allocator *)0x01), SUCCESS);
}

TEST_F(UtestGeExecutor, malloc_after_release_resource) {
  GeExecutor executor{};
  EXPECT_EQ(executor.Initialize(), SUCCESS);
  size_t memory_size = 1024U;
  uint64_t session_id = 0U;
  auto memory_addr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(memory_size, session_id);
  EXPECT_TRUE(memory_addr != nullptr);
  EXPECT_EQ(MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Free(session_id), SUCCESS);
  EXPECT_EQ(GeExecutor::ReleaseResource(), SUCCESS);
  memory_addr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(memory_size, session_id);
  EXPECT_TRUE(memory_addr != nullptr);
  EXPECT_EQ(MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Free(session_id), SUCCESS);
  EXPECT_EQ(executor.Finalize(), SUCCESS);
  memory_addr = MemManager::Instance().SessionScopeMemInstance(RT_MEMORY_HBM).Malloc(memory_size, session_id);
  EXPECT_TRUE(memory_addr == nullptr);
  // repeat finalize, no core dump
  MemManager::Instance().Finalize();
  MemManager::Instance().Finalize();
}

TEST_F(UtestGeExecutor, finalize_after_reset_device) {
  class RuntimeMock : public RuntimeStub {
   public:
    rtError_t rtFree(void *dev_ptr) override {
      EXPECT_TRUE(!device_reset_);
      delete[](uint8_t *) dev_ptr;
      return RT_ERROR_NONE;
    }

    rtError_t rtDeviceReset(int32_t device) override {
      device_reset_ = true;
      return RT_ERROR_NONE;
    }
    bool device_reset_ = false;
  };
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock>());
  GeExecutor executor{};
  EXPECT_EQ(executor.Initialize(), SUCCESS);
  uint32_t device_id = 0U;
  auto allocator = gert::memory::CachingMemAllocator::GetAllocator(device_id);
  EXPECT_TRUE(allocator != nullptr);
  auto block = allocator->Malloc(1024U);
  EXPECT_TRUE(block != nullptr);
  block->Free();

  EXPECT_EQ(GeExecutor::ReleaseResource(), SUCCESS);
  allocator = gert::memory::CachingMemAllocator::GetAllocator(device_id);
  EXPECT_TRUE(allocator != nullptr);
  block = allocator->Malloc(1024U);
  EXPECT_TRUE(block != nullptr);
  auto addr = block->GetAddr();
  block->Free();
  auto block1 = allocator->Malloc(1024U);
  EXPECT_TRUE(block1 != nullptr);
  ASSERT_EQ(block1->GetAddr(), addr);
  block1->Free();
  EXPECT_EQ(GeExecutor::ReleaseResource(), SUCCESS);

  RuntimeStub::GetInstance()->rtDeviceReset(device_id);
  gert::memory::RtsCachingMemAllocator::device_id_to_allocators_.clear();
  // Expect free all memory before Finalize
  EXPECT_EQ(executor.Finalize(), SUCCESS);
  RuntimeStub::SetInstance(nullptr);
}

TEST_F(UtestGeExecutor, finalize_after_reset_device_fail) {
  class RuntimeMock : public RuntimeStub {
   public:
    rtError_t rtFree(void *dev_ptr) override {
      if (device_reset_) {
        return -1;
      }
      delete[](uint8_t *) dev_ptr;
      return RT_ERROR_NONE;
    }

    rtError_t rtDeviceReset(int32_t device) override {
      device_reset_ = true;
      return RT_ERROR_NONE;
    }
    bool device_reset_ = false;
  };
  RuntimeStub::SetInstance(std::make_shared<RuntimeMock>());
  GeExecutor executor{};
  EXPECT_EQ(executor.Initialize(), SUCCESS);
  uint32_t device_id = 0U;
  auto allocator = gert::memory::CachingMemAllocator::GetAllocator(device_id);
  EXPECT_TRUE(allocator != nullptr);
  auto block = allocator->Malloc(1024U);
  EXPECT_TRUE(block != nullptr);
  block->Free();
  EXPECT_EQ(GeExecutor::ReleaseResource(), SUCCESS);

  RuntimeStub::GetInstance()->rtDeviceReset(device_id);

  // Expect free all memory before Finalize
  EXPECT_EQ(executor.Finalize(), SUCCESS);
  RuntimeStub::SetInstance(nullptr);
}

TEST_F(UtestGeExecutor, test_ge_initialize_without_hccl) {
  GeExecutor executor;
  const std::map<std::string, std::string> options({
    {OPTION_EXEC_JOB_ID, "1"},
    {OPTION_EXEC_PROFILING_MODE, ""},
    {OPTION_EXEC_PROFILING_OPTIONS, ""},
    {OPTION_EXEC_HCCL_FLAG, "0"}});
  EXPECT_EQ(executor.Initialize(options), SUCCESS);
}
} // namespace ge

