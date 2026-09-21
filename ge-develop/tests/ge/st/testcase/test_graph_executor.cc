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
#include "graph/utils/graph_utils.h"
#include "proto/ge_ir.pb.h"
#include "graph/manager/graph_var_manager.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "macro_utils/dt_public_unscope.h"
#include "exe_graph/runtime/tensor.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "ge_running_env/ge_running_env_faker.h"
#include "ge_running_env/fake_op.h"
#include "utils/mock_ops_kernel_builder.h"

#include "graph_metadef/depends/checker/tensor_check_utils.h"

using namespace std;
using namespace testing;

namespace ge {
namespace {
static void MockGenerateTask() {
  auto aicore_func = [](const ge::Node &node, RunContext &context, std::vector<domi::TaskDef> &tasks) -> Status {
    auto op_desc = node.GetOpDesc();
    op_desc->SetOpKernelLibName("AIcoreEngine");
    ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    ge::AttrUtils::SetStr(op_desc, ge::ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    ge::OpKernelBinPtr tbeKernelPtr = std::make_shared<ge::OpKernelBin>("test_tvm", std::move(buffer));
    op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
    size_t arg_size = 100;
    std::vector<uint8_t> args(arg_size, 0);
    domi::TaskDef task_def;
    task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    auto kernel_info = task_def.mutable_kernel();
    kernel_info->set_args(args.data(), args.size());
    kernel_info->set_args_size(arg_size);
    kernel_info->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    kernel_info->set_kernel_name(node.GetName());
    kernel_info->set_block_dim(1);
    uint16_t args_offset[2] = {0};
    kernel_info->mutable_context()->set_args_offset(args_offset, 2 * sizeof(uint16_t));
    kernel_info->mutable_context()->set_op_index(node.GetOpDesc()->GetId());

    tasks.emplace_back(task_def);
    return SUCCESS;
  };

  MockForGenerateTask("AiCoreLib", aicore_func);
  MockForGenerateTask("AIcoreEngine", aicore_func);
}
}

class StestGraphExecutor : public testing::Test {
 protected:
  void SetUp() {
    unsetenv("FMK_SYSMODE");
    unsetenv("FMK_DUMP_PATH");
    unsetenv("FMK_USE_FUSION");
    unsetenv("DAVINCI_TIMESTAT_ENABLE");
    MockGenerateTask();
  }

  void TearDown() {
    EXPECT_TRUE(ModelManager::GetInstance().model_map_.empty());
    EXPECT_TRUE(ModelManager::GetInstance().hybrid_model_map_.empty());
  }
};

namespace {
const std::string kAttrNameAtomicWspMode = "wspMode";
const std::string kWspFoldedMode = "folded";
}

TEST_F(StestGraphExecutor, execute_graph_sync_multi) {
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
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"dump"}));

    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, op_desc->GetName() + "_fake_id");
    const char tbeBin[] = "tbe_bin";
    vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
    OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("teslt_tvm", std::move(buffer));
    op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);

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
  {
    shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(1, mock_listener);
    model->SetId(1);

    model->Assign(ge_model);
    DataBuffer data_buf;
    data_buf.data = malloc(20);
    data_buf.length = 20;
    ge_model->SetWeightDataBuf(data_buf);
    ModelParam param;
    param.mem_base = 10000;
    param.weight_base = 10000;
    param.weight_size = 5;
    EXPECT_EQ(model->Init(param), ACL_ERROR_GE_PARAM_INVALID);
    free(data_buf.data);
  }

  GraphExecutor graph_executer;
  std::vector<gert::Tensor> input_tensor(1);
  TensorCheckUtils::ConstructGertTensor(input_tensor[0], {128}, DT_FLOAT, FORMAT_NCHW);
  std::vector<gert::Tensor> output_tensor;

  {
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetModelId(0);
    ge_root_model->SetModelId(1);
    EXPECT_EQ(graph_executer.ExecuteGraph(0, ge_root_model, input_tensor, output_tensor), SUCCESS);
  }

  std::vector<gert::Tensor> output_tensor_pro;
  {
    graph->DelAttr(ATTR_NAME_DEPLOY_INFO);
    const auto ge_root_model = MakeShared<GeRootModel>();
    EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
    ge_root_model->SetModelId(1);
    EXPECT_EQ(graph_executer.ExecuteGraph(0, ge_root_model, input_tensor, output_tensor_pro), SUCCESS);
  }

  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(0U), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(1U), SUCCESS);
}
void AddTask(shared_ptr<domi::ModelTaskDef> &model_task_def,
             const int32_t stream_id,
             const std::string &stub_func_name,
             const size_t args_size,
             const NodePtr &task_node) {
  // add task def
  domi::TaskDef *task_def = model_task_def->add_task();
  task_def->set_stream_id(stream_id);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  // add kernel def
  domi::KernelDef *kernel_def = task_def->mutable_kernel();
  kernel_def->set_stub_func(stub_func_name);
  kernel_def->set_args_size(args_size);
  string args(args_size, '1');
  kernel_def->set_args(args.data(), args.size());
  // add context
  domi::KernelContext *context = kernel_def->mutable_context();
  context->set_op_index(task_node->GetOpDesc()->GetId());
  context->set_kernel_type(static_cast<int32_t>(ge::ccKernelType::TE));
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
}

TEST_F(StestGraphExecutor, execute_graph_sync_multi_atomic_memset) {
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  map<string, string> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
  VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);

  DEF_GRAPH(graph) {
   auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto atomic_memset = OP_CFG(MEMSET)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    CTRL_CHAIN(NODE("memset", atomic_memset)->NODE("atomic_node", fake_type2_op1));
    CHAIN(NODE("args0", data0)->NODE("atomic_node", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  auto memset_node = root_graph->FindFirstNodeMatchType(MEMSET);
  memset_node->GetOpDesc()->SetWorkspace({1024});
  memset_node->GetOpDesc()->SetWorkspaceBytes({64});
  AttrUtils::SetInt(memset_node->GetOpDesc(), ATTR_NAME_NODE_SQE_NUM, 66);
  auto data_node = root_graph->FindFirstNodeMatchType(DATA);
  data_node->GetOpDesc()->SetInputOffset({1024});
  data_node->GetOpDesc()->SetOutputOffset({1024});
  auto atomic_node = root_graph->FindFirstNodeMatchType("AtomicNode");
  const char tbeBin[] = "tbe_bin";
  vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
  OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
  ge::AttrUtils::SetStr(atomic_node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetStr(atomic_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_BIN_ID, "_atomic_node_1_fake_id");
  atomic_node->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
  ge::AttrUtils::SetStr(memset_node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetStr(memset_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_BIN_ID, "_memset_node_1_fake_id");
  memset_node->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
  ge::AttrUtils::SetBool(atomic_node->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  auto netoutput = root_graph->FindFirstNodeMatchType(NETOUTPUT);
  ge::AttrUtils::SetBool(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  netoutput->GetOpDesc()->SetInputOffset({2048});
  netoutput->GetOpDesc()->SetSrcName({"atomic_node"});
  netoutput->GetOpDesc()->SetSrcIndex({0});

  ge::AttrUtils::SetBool(memset_node->GetOpDesc(), GLOBALWORKSPACE_TYPE, true);
  EXPECT_EQ(ge::AttrUtils::SetStr(memset_node->GetOpDesc(), kAttrNameAtomicWspMode, kWspFoldedMode), true);
  EXPECT_EQ(GraphUtils::AddEdge(memset_node->GetOutControlAnchor(), atomic_node->GetInControlAnchor()), SUCCESS);
  // atomic node task
  AddTask(model_task_def, 0, "stub_func", 64UL, atomic_node);
  // memset node task
  AddTask(model_task_def, 0, "stub_func", 32UL, memset_node);

  auto mock_listener = MakeShared<GraphModelListener>();
  shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(0, mock_listener);
  model->SetId(0);

  model->Assign(ge_model);
  EXPECT_EQ(model->Init(), SUCCESS);

  EXPECT_EQ(model->input_addrs_list_.size(), 1);
  EXPECT_EQ(model->output_addrs_list_.size(), 1);
  EXPECT_EQ(model->task_list_.size(), 2);

  OutputData output_data;
  vector<gert::Tensor> outputs;
  EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);

  void *goverflow_addr = nullptr;
  EXPECT_EQ(rtMalloc(&goverflow_addr, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16), SUCCESS);
  model->SetOverflowAddr(goverflow_addr);

  ModelManager::GetInstance().InsertModel(0U, model);

  GraphExecutor graph_executer;
  std::vector<gert::Tensor> input_tensor(1);
  TensorCheckUtils::ConstructGertTensor(input_tensor[0], {4, 4}, DT_FLOAT, FORMAT_NCHW);
  std::vector<gert::Tensor> output_tensor;

  const auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ge_root_model->SetModelId(0);
  EXPECT_EQ(graph_executer.ExecuteGraph(0, ge_root_model, input_tensor, output_tensor), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(0U), SUCCESS);
  EXPECT_EQ(rtFree(goverflow_addr), 0);
}

TEST_F(StestGraphExecutor, execute_graph_async_with_gert_tensor_multi_atomic_memset) {
  VarManager::Instance(0)->Init(0, 0, 0, 0);
  map<string, string> options;
  options[GRAPH_MEMORY_MAX_SIZE] = "1048576";
  VarManager::Instance(0)->SetMemoryMallocSize(options, 1024UL * 1024UL * 1024UL);

  DEF_GRAPH(graph) {
   auto data0 = OP_CFG(DATA)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});
    auto atomic_memset = OP_CFG(MEMSET)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto fake_type2_op1 = OP_CFG("AtomicNode")
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    auto net_output = OP_CFG(NETOUTPUT)
        .InCnt(1)
        .OutCnt(1)
        .TensorDesc(FORMAT_ND, DT_INT32, {16});

    CTRL_CHAIN(NODE("memset", atomic_memset)->NODE("atomic_node", fake_type2_op1));
    CHAIN(NODE("args0", data0)->NODE("atomic_node", fake_type2_op1)->NODE("Node_Output", net_output));
  };

  auto root_graph = ToComputeGraph(graph);
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(root_graph);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);

  shared_ptr<domi::ModelTaskDef> model_task_def = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task_def);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  auto memset_node = root_graph->FindFirstNodeMatchType(MEMSET);
  memset_node->GetOpDesc()->SetWorkspace({1024});
  memset_node->GetOpDesc()->SetWorkspaceBytes({64});
  AttrUtils::SetInt(memset_node->GetOpDesc(), ATTR_NAME_NODE_SQE_NUM, 66);
  auto data_node = root_graph->FindFirstNodeMatchType(DATA);
  data_node->GetOpDesc()->SetInputOffset({1024});
  data_node->GetOpDesc()->SetOutputOffset({1024});
  auto atomic_node = root_graph->FindFirstNodeMatchType("AtomicNode");
  const char tbeBin[] = "tbe_bin";
  vector<char> buffer(tbeBin, tbeBin + strlen(tbeBin));
  OpKernelBinPtr tbeKernelPtr = std::make_shared<OpKernelBin>("test_tvm", std::move(buffer));
  ge::AttrUtils::SetStr(atomic_node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetStr(atomic_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_BIN_ID, "_atomic_node_1_fake_id");
  atomic_node->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
  ge::AttrUtils::SetStr(memset_node->GetOpDesc(), ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  ge::AttrUtils::SetStr(memset_node->GetOpDesc(), ge::ATTR_NAME_KERNEL_BIN_ID, "_memset_node_1_fake_id");
  memset_node->GetOpDesc()->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbeKernelPtr);
  ge::AttrUtils::SetBool(atomic_node->GetOpDesc()->MutableOutputDesc(0), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  auto netoutput = root_graph->FindFirstNodeMatchType(NETOUTPUT);
  ge::AttrUtils::SetBool(netoutput->GetOpDesc()->MutableInputDesc(0), ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE, true);
  netoutput->GetOpDesc()->SetInputOffset({2048});
  netoutput->GetOpDesc()->SetSrcName({"atomic_node"});
  netoutput->GetOpDesc()->SetSrcIndex({0});

  ge::AttrUtils::SetBool(memset_node->GetOpDesc(), GLOBALWORKSPACE_TYPE, true);
  EXPECT_EQ(ge::AttrUtils::SetStr(memset_node->GetOpDesc(), kAttrNameAtomicWspMode, kWspFoldedMode), true);
  EXPECT_EQ(GraphUtils::AddEdge(memset_node->GetOutControlAnchor(), atomic_node->GetInControlAnchor()), SUCCESS);
  // atomic node task
  AddTask(model_task_def, 0, "stub_func", 64UL, atomic_node);
  // memset node task
  AddTask(model_task_def, 0, "stub_func", 32UL, memset_node);

  auto mock_listener = MakeShared<GraphModelListener>();
  shared_ptr<DavinciModel> model = MakeShared<DavinciModel>(0, mock_listener);
  model->SetId(0);

  model->Assign(ge_model);
  EXPECT_EQ(model->Init(), SUCCESS);

  EXPECT_EQ(model->input_addrs_list_.size(), 1);
  EXPECT_EQ(model->output_addrs_list_.size(), 1);
  EXPECT_EQ(model->task_list_.size(), 2);

  OutputData output_data;
  vector<gert::Tensor> outputs;
  EXPECT_EQ(model->GenOutputTensorInfo(output_data, outputs), SUCCESS);

  void *goverflow_addr = nullptr;
  EXPECT_EQ(rtMalloc(&goverflow_addr, static_cast<uint64_t>(16), RT_MEMORY_HBM, GE_MODULE_NAME_U16), SUCCESS);
  model->SetOverflowAddr(goverflow_addr);

  ModelManager::GetInstance().InsertModel(0U, model);
  GraphExecutor graph_executer;
  const auto ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ge_root_model->SetModelId(0);

  std::vector<gert::Tensor> gert_inputs;
  vector<uint8_t> inputs_data(512, 0);
  gert_inputs.resize(1);
  gert_inputs[0] =  {{{4, 4}, {4, 4}},                            // shape
                     {ge::FORMAT_NCHW, ge::FORMAT_NCHW, {}},      // format
                     gert::kOnDeviceHbm,                          // placement
                     ge::DT_FLOAT,                                // data type
                     (void *) inputs_data.data()};

  // 异步执行的output size需要调用时确定?
  std::vector<gert::Tensor> gert_outputs;
  gert_outputs.resize(1);
  vector<uint8_t> outputs_data(512, 0);
  gert_outputs[0].SetSize(4);
  gert_outputs[0].MutableTensorData().SetAddr((void *) inputs_data.data(), nullptr);

  GraphNodePtr graph_node = MakeShared<ge::GraphNode>(1);
  EXPECT_EQ(graph_executer.ExecuteGraphWithStream(nullptr, graph_node, ge_root_model, gert_inputs, gert_outputs), SUCCESS);
  EXPECT_EQ(ModelManager::GetInstance().DeleteModel(0U), SUCCESS);
  EXPECT_EQ(rtFree(goverflow_addr), 0);
}

TEST_F(StestGraphExecutor, execute_graph_async_multi) {
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
    NodePtr node = graph->AddNode(op_desc);  // op_index = 1
    EXPECT_TRUE(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, std::vector<std::string>{"dump"}));
    std::vector<char> kernel_bin(64, '\0');
    TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
    EXPECT_TRUE(op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle));
    EXPECT_TRUE(AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName()));
    AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "te_square_123");
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

TEST_F(StestGraphExecutor, InvalidBrachInfoTest) {
  ComputeGraphPtr sub_graph = MakeShared<ComputeGraph>("sub_graph");
  OpDescPtr sub_netoutput = CreateOpDesc("sub_netoutput", "NetOutput");
  NodePtr sub_netoutput_node = sub_graph->AddNode(sub_netoutput);
  std::string invalid_batch_label_1 = "Batch_a";
  AttrUtils::SetStr(sub_netoutput_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, invalid_batch_label_1);

  ComputeGraphPtr graph = MakeShared<ComputeGraph>("default");
  graph->AddSubGraph(sub_graph);
  OpDescPtr op_case = CreateOpDesc("case", "Case");
  NodePtr node_case = graph->AddNode(op_case);
  node_case->GetOpDesc()->AddSubgraphName("sub_graph");
  node_case->GetOpDesc()->SetSubgraphInstanceName(0, "sub_graph");

  DavinciModel model(0, nullptr);
  auto invalid_op = MakeShared<OpDesc>();
  Status ret = model.GetRealOutputSizeOfCase(graph, 0, node_case);
  EXPECT_NE(ret, SUCCESS);

  std::string invalid_batch_label_2 = "Batch_" + std::to_string(INT64_MAX);
  AttrUtils::SetStr(sub_netoutput_node->GetOpDesc(), ATTR_NAME_BATCH_LABEL, invalid_batch_label_2);
  ret = model.GetRealOutputSizeOfCase(graph, 0, node_case);
  EXPECT_NE(ret, SUCCESS);
}
} // namespace ge
