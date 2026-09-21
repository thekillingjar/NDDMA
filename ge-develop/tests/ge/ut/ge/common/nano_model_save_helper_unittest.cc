/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <gtest/gtest.h>
#include "mmpa/mmpa_api.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "ge_graph_dsl/op_desc/op_desc_cfg_box.h"
#include "ge_graph_dsl/op_desc/op_desc_node_builder.h"

#include "macro_utils/dt_public_scope.h"
#include "framework/common/helper/nano_model_save_helper.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/omg/ge_init.h"
#include "common/model/ge_model.h"
#include "common/share_graph.h"
#include "faker/ge_model_builder.h"
#include "common/helper/model_parser_base.h"
#include "common/preload/model/pre_model_partition_utils.h"
#include "graph/buffer/buffer_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "common/profiling/profiling_manager.h"
#include "graph/manager/graph_var_manager.h"
#include "framework/common/taskdown_common.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/env_path.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "proto/task.pb.h"

using namespace std;

namespace ge {
using namespace hybrid;
namespace {
std::vector<char> CreateStubBin() {
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  std::string op_bin_path = ascend_install_path + "/fwkacllib/lib64/switch_by_index.o";
  std::vector<char> buf;
  std::ifstream file(op_bin_path.c_str(), std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    std::cout << "file:" << op_bin_path << "does not exist or is unaccessible." << std::endl;
    return buf;
  }
  GE_MAKE_GUARD(file_guard, [&file]() {
    (void)file.close();
  });
  const std::streampos begin = file.tellg();
  (void)file.seekg(0, std::ios::end);
  const std::streampos end = file.tellg();
  auto buf_len = static_cast<uint32_t>(end - begin);
  buf.resize(buf_len + 1);
  (void)file.seekg(0, std::ios::beg);
  (void)file.read(const_cast<char *>(buf.data()), buf_len);
  return buf;
}

static GeRootModelPtr GenGeRootModel() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = CreateNode(*graph, "add", ADD, 3, 2);
  OpDescPtr op_desc = add_node->GetOpDesc();
  EXPECT_NE(op_desc, nullptr);

  const std::vector<int64_t> v_memory_type{RT_MEMORY_HBM, RT_MEMORY_L1};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
  std::vector<uint64_t> output_offset{10, 10};
  (void)op_desc->SetExtAttr("task_addr_offset", output_offset);
  op_desc->SetOutputOffset({2048, 2048});
  for (int i = 0; i < 2; ++i) {
    auto output_tensor = op_desc->MutableOutputDesc(i);
    output_tensor->SetShape(GeShape({1, 4, 4, 8}));
  }

  const std::vector<int64_t> v_memory_type1{RT_MEMORY_L1, RT_MEMORY_HBM, RT_MEMORY_HBM};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type1);
  op_desc->SetInputOffset({2048, 2048, 2048});
  std::vector<uint64_t> input_offset{10, 10, 10};
  (void)op_desc->SetExtAttr("task_addr_offset", input_offset);

  std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
  AttrUtils::SetListInt(op_desc, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
  op_desc->SetWorkspaceBytes(std::vector<int64_t>{32});

  op_desc->SetIsInputConst({true});
  std::vector<string> original_op_names = {"conv", "add"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);

  op_desc->AddInputDesc(GeTensorDesc());

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);

  TBEKernelStore tbe_kernel_store;
  const auto kernel = MakeShared<OpKernelBin>("test", CreateStubBin());
  tbe_kernel_store.AddTBEKernel(kernel);
  tbe_kernel_store.PreBuild();
  ge_model->SetTBEKernelStore(tbe_kernel_store);

  std::shared_ptr<domi::ModelTaskDef> model_task = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task);
  
  domi::TaskDef *task_def0 = model_task->add_task();
  task_def0->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  task_def0->set_stream_id(op_desc->GetStreamId());
  domi::KernelDef *kernel_def0 = task_def0->mutable_kernel();
  kernel_def0->set_stub_func("stub_func");
  kernel_def0->set_args_size(64);
  string args(64, '1');
  kernel_def0->set_args(args.data(), 64);
  kernel_def0->set_kernel_name("test");

  domi::KernelContext *context0 = kernel_def0->mutable_context();
  context0->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  context0->set_op_index(op_desc->GetId());
  uint16_t args_offset[9] = {0};
  context0->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName(ge_model->GetName());
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);

  return ge_root_model;
}
}

class UtestNanoModelSaveHelper : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestNanoModelSaveHelper, ModelToOm)
{
  std::string output_file = "outputfile.om";
  ModelBufferData model;
  NanoModelSaveHelper model_save_helper;
  GeRootModelPtr ge_root_model = GenGeRootModel();
  // output_file is null
  EXPECT_NE(model_save_helper.SaveToOmRootModel(ge_root_model, "", model, false), SUCCESS);
  // Unknown shape not support
  EXPECT_NE(model_save_helper.SaveToOmRootModel(ge_root_model, output_file, model, true), SUCCESS);
  // opdesc is nullptr
  EXPECT_NE(model_save_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");
}

TEST_F(UtestNanoModelSaveHelper, RepackSoToOm)
{
  uint32_t mem_offset = 0U;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("add_n", ADDN));   // ccKernelType::TE
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);
  SetKnownOpKernel(graph, mem_offset);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  {
    const auto &node = graph->FindNode("add_n");
    const auto &op_desc = node->GetOpDesc();

    TBEKernelStore tbe_kernel_store;
    const auto kernel = MakeShared<OpKernelBin>("test", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 0
    domi::TaskDef *task_def0 = model_def->add_task();
    task_def0->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def0->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def0 = task_def0->mutable_kernel();
    kernel_def0->set_stub_func("stub_func");
    kernel_def0->set_args_size(64);
    string args(64, '1');
    kernel_def0->set_args(args.data(), 64);
    kernel_def0->set_kernel_name("test");

    domi::KernelContext *context0 = kernel_def0->mutable_context();
    context0->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context0->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context0->set_args_offset(args_offset, 9 * sizeof(uint16_t));

    // task 1
    domi::TaskDef *task_def1 = model_def->add_task();
    task_def1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def1->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def1 = task_def1->mutable_kernel();
    kernel_def1->set_stub_func("stub_func");
    kernel_def1->set_args_size(64);
    kernel_def1->set_args(args.data(), 64);
    kernel_def1->set_kernel_name("test");

    domi::KernelContext *context1 = kernel_def1->mutable_context();
    context1->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context1->set_op_index(op_desc->GetId());
    context1->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string output_file = "";
  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_NE(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }

  output_file = "outputfile.exeom";
  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_EQ(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
}

TEST_F(UtestNanoModelSaveHelper, GenHostFuncExeom)
{
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  uint32_t mem_offset = 0U;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("add_n", ADDN));   // ccKernelType::TE
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);
  SetKnownOpKernel(graph, mem_offset);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  std::string output_file = "outputfile.exeom";

  {
    const auto &node = graph->FindNode("add_n");
    const auto &op_desc = node->GetOpDesc();

    GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 512);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetOutputOffset({8});
    PreModelPartitionUtils::GetInstance().SetZeroCopyTable(1024, 8);

    op_desc->SetAttr("int_test", GeAttrValue::CreateFrom<int64_t>(100));
    op_desc->SetAttr("str_test", GeAttrValue::CreateFrom<std::string>("Hello!"));
    op_desc->SetAttr("float_test", GeAttrValue::CreateFrom<float>(10.101));
    op_desc->SetAttr("list_int_test", GeAttrValue::CreateFrom<std::vector<int64_t>>({1, 2, 3}));
    op_desc->SetAttr("list_float_test", GeAttrValue::CreateFrom<std::vector<float>>({1.2, 3.4, 4.5}));
    op_desc->SetAttr("list_str_test", GeAttrValue::CreateFrom<std::vector<std::string>>({"Hello!", "ABCD"}));
    op_desc->SetAttr("list_bool_test", GeAttrValue::CreateFrom<std::vector<bool>>({true, false}));
    std::vector<std::vector<int64_t>> test_listlist_int;
    std::vector<int64_t> vec1;
    vec1.push_back(1);
    vec1.push_back(2);
    test_listlist_int.push_back(vec1);
    std::vector<int64_t> vec2;
    vec2.push_back(3);
    vec2.push_back(4);
    test_listlist_int.push_back(vec2);
    op_desc->SetAttr("list_list_int_test", GeAttrValue::CreateFrom<std::vector<std::vector<int64_t>>>(test_listlist_int));
    std::vector<std::vector<float>> test_listlist_float;
    std::vector<float> vec3;
    vec3.push_back(1.2);
    vec3.push_back(3.4);
    test_listlist_float.push_back(vec3);
    std::vector<float> vec4;
    vec4.push_back(5.6);
    vec4.push_back(7.8);
    test_listlist_float.push_back(vec4);
    op_desc->SetAttr("list_list_float_test", GeAttrValue::CreateFrom<std::vector<std::vector<float>>>(test_listlist_float));


    TBEKernelStore tbe_kernel_store;
    const auto kernel = MakeShared<OpKernelBin>("test", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 0
    domi::TaskDef *task_def0 = model_def->add_task();
    task_def0->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def0->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def0 = task_def0->mutable_kernel();
    kernel_def0->set_stub_func("stub_func");
    kernel_def0->set_args_size(64);
    string args(64, '1');
    kernel_def0->set_args(args.data(), 64);
    kernel_def0->set_kernel_name("test");

    domi::KernelContext *context0 = kernel_def0->mutable_context();
    context0->set_kernel_type(static_cast<uint32_t>(ccKernelType::AI_CPU));
    context0->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context0->set_args_offset(args_offset, 9 * sizeof(uint16_t));

    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_EQ(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }

  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestNanoModelSaveHelper, GenSwitchByIndexExeom)
{
  /**
   *       Data    Data
   *         \      /
   *          SwitchByIndex
   *           |       \ 
   *        LabelSet   LabelSet
   *           |         \ 
   *      StreamActive   AddN
   *           |
   *          Add
   */

  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/fwkacllib/lib64").c_str(), 1);
  auto label_switch = OP_CFG(LABELSWITCHBYINDEX).Attr(ATTR_NAME_LABEL_SWITCH_LIST, std::vector<uint32_t>{0, 1});
  auto label_set_l0 = OP_CFG(LABELSET).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 0);
  auto label_set_l1 = OP_CFG(LABELSET).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 1);
  auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("index_0", data_0)->EDGE(0, 0)->NODE("switch", label_switch)->EDGE(0, 0)->NODE("label_set_l0", label_set_l0)->NODE("stream", STREAMACTIVE)->NODE("add", ADD));
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 1)->NODE("switch")->EDGE(1, 0)->NODE("label_set_l1", label_set_l1)->NODE("add_n", ADDN));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();

  {
    const auto &node = graph->FindNode("switch");
    const auto &op_desc = node->GetOpDesc();
    EXPECT_TRUE(AttrUtils::SetListInt(op_desc, ATTR_NAME_LABEL_SWITCH_LIST, {0, 1}));

    GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 512);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetOutputOffset({1024});

    // task 0 switch_by_index
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_LABEL_SWITCH_BY_INDEX));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("switch_by_index.o");

    domi::LabelSwitchByIndexDef *switch_task_def = task_def->mutable_label_switch_by_index();
    switch_task_def->set_op_index(op_desc->GetId());
    switch_task_def->set_label_max(2);
  }

  {
    const auto &node = graph->FindNode("label_set_l0");
    const auto &op_desc = node->GetOpDesc();

    // task 1 label_set 0
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("stream");
    const auto &op_desc = node->GetOpDesc();

    // task 2 stream active
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_ACTIVE));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::StreamActiveDef *stream_task_def = task_def->mutable_stream_active();
    stream_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("add");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("add", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 3 add
    domi::TaskDef *task_def0 = model_def->add_task();
    task_def0->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def0->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def0 = task_def0->mutable_kernel();
    kernel_def0->set_stub_func("stub_func");
    kernel_def0->set_args_size(64);
    string args(64, '1');
    kernel_def0->set_args(args.data(), 64);
    kernel_def0->set_kernel_name("add");

    domi::KernelContext *context0 = kernel_def0->mutable_context();
    context0->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context0->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context0->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = graph->FindNode("label_set_l1");
    const auto &op_desc = node->GetOpDesc();

    // task 4 label_set 1
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("add_n");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("add_n", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 5 add_n
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("add_n");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string output_file = "outputfile.exeom";
  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_EQ(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestNanoModelSaveHelper, GenLabelGotoExExeom)
{
  /**
   *       Data
   *         \ 
   *          LabelGotoEx
   *           | 
   *        LabelSet 
   *           | 
   *      StreamActive 
   *           |
   *          AddN
   */
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/fwkacllib/lib64").c_str(), 1);
  auto label_set_l2 = OP_CFG(LABELSET).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 2);
  auto label_gotoex = OP_CFG(LABELGOTOEX).Attr(ATTR_NAME_LABEL_SWITCH_INDEX, 2);
  auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("label_gotoex", label_gotoex)->NODE("label_set_l2", label_set_l2)->NODE("stream", STREAMACTIVE)->NODE("add_n", ADDN));
  };
  ComputeGraphPtr graph = ToComputeGraph(g1);

  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();

  {
    const auto &node = graph->FindNode("label_gotoex");
    const auto &op_desc = node->GetOpDesc();

    GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 512);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetOutputOffset({1024});

    // task 0 label goto
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_LABEL_GOTO));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("switch_by_index.o");

    domi::LabelGotoExDef *label_gotoex_task_def = task_def->mutable_label_goto_ex();
    label_gotoex_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("label_set_l2");
    const auto &op_desc = node->GetOpDesc();

    // task 1 label_set 2
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("stream");
    const auto &op_desc = node->GetOpDesc();

    // task 2 stream active
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_ACTIVE));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::StreamActiveDef *stream_task_def = task_def->mutable_stream_active();
    stream_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("add_n");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("add_n", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 3 add_n
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("add_n");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string output_file = "outputfile.exeom";
  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_EQ(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");

  {
    const auto &node = graph->FindNode("label_set_l2");
    const auto &op_desc = node->GetOpDesc();
    EXPECT_TRUE(AttrUtils::SetInt(op_desc, ATTR_NAME_LABEL_SWITCH_INDEX, 1));
  }

  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_NE(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
  unsetenv("LD_LIBRARY_PATH");
}

TEST_F(UtestNanoModelSaveHelper, GenExeomSubGraph)
{
  DEF_GRAPH(g_main) {
    CHAIN(NODE("input", "Data")->NODE("while", "While")->NODE("NetOutput", "NetOutput"));
  };
  auto main_graph = ToComputeGraph(g_main);
  {
    auto netoutput = main_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"while"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
  }

  DEF_GRAPH(g_cond) {
    CHAIN(NODE("input", "Data")->NODE("foo", "LessThan_5")->NODE("NetOutput", "NetOutput"));
  };
  auto cond_graph = ToComputeGraph(g_cond);
  {
    auto netoutput = cond_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"foo"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
  }

  DEF_GRAPH(g_body) {
    CHAIN(NODE("input", "Data")->NODE("bar", "Add_1")->NODE("NetOutput", "NetOutput"));
  };
  auto body_graph = ToComputeGraph(g_body);
  {
    auto netoutput = body_graph->FindFirstNodeMatchType("NetOutput");
    netoutput->GetOpDesc()->SetSrcName({"bar"});
    netoutput->GetOpDesc()->SetSrcIndex({0});
  }

  auto while_node = main_graph->FindFirstNodeMatchType("While");
  cond_graph->SetParentGraph(main_graph);
  cond_graph->SetParentNode(while_node);
  body_graph->SetParentGraph(main_graph);
  body_graph->SetParentNode(while_node);

  main_graph->AddSubgraph(cond_graph);
  main_graph->AddSubgraph(body_graph);

  while_node->GetOpDesc()->AddSubgraphName("cond");
  while_node->GetOpDesc()->AddSubgraphName("body");
  
  while_node->GetOpDesc()->SetSubgraphInstanceName(0, cond_graph->GetName());
  while_node->GetOpDesc()->SetSubgraphInstanceName(1, body_graph->GetName());

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(main_graph);

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();

  {
    const auto &node = main_graph->FindNode("while");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("while", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("while");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }
  
  std::string output_file = "outputfile.exeom";
  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_EQ(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");
}

TEST_F(UtestNanoModelSaveHelper, GenWindowCacheExeom)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///      Data    Const    Data
  ///        \      |       /
  ///         FillWindowCache
  ///              | 
  ///             Add 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<int64_t> shape = {2, 2};           // NCHW
  auto data_0 = OP_CFG(DATA)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Build("data_0");
  data_0->SetOutputOffset({0});
  PreModelPartitionUtils::GetInstance().SetZeroCopyTable(128, 0);
  TensorUtils::SetSize(*data_0->MutableOutputDesc(0), 8);

  auto data_1 = OP_CFG(CONSTANT)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Build("data_1");
  data_1->SetOutputOffset({0});
  ge::AttrUtils::SetInt(*data_1->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);
  TensorUtils::SetSize(*data_1->MutableOutputDesc(0), 8);

  auto data_2 = OP_CFG(DATA)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .OutCnt(1)
                    .Build("data_2");
  data_2->SetOutputOffset({8});
  PreModelPartitionUtils::GetInstance().SetZeroCopyTable(136, 8);
  TensorUtils::SetSize(*data_2->MutableOutputDesc(0), 8);

  auto fifo = OP_CFG("FillWindowCache")
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(3)
                    .OutCnt(1)
                    .Attr(ATTR_NAME_REFERENCE, true)
                    .Build("fillwindowcache");
  vector<bool> is_input_const_vec = {false, true, false};
  fifo->SetIsInputConst(is_input_const_vec);
  fifo->SetInputOffset({0,1024,8});
  fifo->SetOutputOffset({1024});
  ge::AttrUtils::SetInt(*fifo->MutableOutputDesc(0), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);
  ge::AttrUtils::SetInt(*fifo->MutableInputDesc(1), ATTR_NAME_TENSOR_MEMORY_SCOPE, 2);
  auto output_tensordesc = fifo->MutableOutputDesc(0);
  ge::TensorUtils::SetReuseInput(*output_tensordesc, true);
  ge::TensorUtils::SetReuseInputIndex(*output_tensordesc, 1);
  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(fifo)->EDGE(0, 0)->
    NODE("add", ADD));
    CHAIN(NODE(data_1)->EDGE(0, 1)->NODE("fillwindowcache"));
    CHAIN(NODE(data_2)->EDGE(0, 2)->NODE("fillwindowcache"));
  };

  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();

  {
    const auto &node = graph->FindNode("fillwindowcache");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("fillwindowcache", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 0 fillwindowcache
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("fillwindowcache");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = graph->FindNode("add");
    const auto &op_desc = node->GetOpDesc();
    op_desc->SetInputOffset({0});

    const auto kernel = MakeShared<OpKernelBin>("add", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 1 add
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("add");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  std::string output_file = "outputfile.exeom";
  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_EQ(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");
}

TEST_F(UtestNanoModelSaveHelper, NanoEnablePrefetchTest)
{
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ///      Data  Const
  ///         \   |
  ///          Add
  ///           | 
  ///          Relu
  ///           | 
  ///       NetOutput 
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<int64_t> shape = {2};           // NCHW
  auto data_0 = OP_CFG(DATA).TensorDesc(FORMAT_NCHW, DT_FLOAT, shape).InCnt(1).OutCnt(1).Build("data_0");
  data_0->SetOutputOffset({0});
  PreModelPartitionUtils::GetInstance().SetZeroCopyTable(128, 0);

  int32_t data_value_vec[2] = {2, 4};
  GeTensorDesc data_tensor_desc(GeShape(shape), FORMAT_ND, DT_FLOAT);
  TensorUtils::SetDataOffset(data_tensor_desc, 16);
  TensorUtils::SetWeightSize(data_tensor_desc, 16);
  TensorUtils::SetSize(data_tensor_desc, 16);
  GeTensorPtr data_tensor = make_shared<GeTensor>(data_tensor_desc, (uint8_t *)data_value_vec, 2 * sizeof(int32_t));
  auto const_0 = OP_CFG(CONSTANT).Weight(data_tensor).InCnt(1).OutCnt(1).Build("const_0");;

  auto add = OP_CFG(ADD)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(2)
                    .OutCnt(1)
                    .Attr("_weight_prefetch_type", std::vector<std::string>{"1", "1"})
                    .Attr("_weight_prefetch_src_offset", std::vector<int64_t>{1024,2048})
                    .Attr("_weight_prefetch_dst_offset", std::vector<int64_t>{64, 128})
                    .Attr("_weight_prefetch_data_size", std::vector<int64_t>{16, 16})
                    .Build("add");
  add->SetInputOffset({0,64});
  add->SetOutputOffset({16});
  auto relu = OP_CFG(RELU)
                    .TensorDesc(FORMAT_NCHW, DT_FLOAT, shape)
                    .InCnt(1)
                    .InCnt(1)
                    .Attr("_weight_prefetch_type", std::vector<std::string>{"1"})
                    .Attr("_weight_prefetch_src_offset", std::vector<int64_t>{1024,2048})
                    .Attr("_weight_prefetch_dst_offset", std::vector<int64_t>{64, 128})
                    .Attr("_weight_prefetch_data_size", std::vector<int64_t>{16, 16})
                    .Build("relu");
  relu->SetInputOffset({16});
  relu->SetOutputOffset({32});

  DEF_GRAPH(g1) {
    CHAIN(NODE(data_0)->EDGE(0, 0)->NODE(add)->EDGE(0, 0)->NODE(relu)->EDGE(0, 0)->NODE("output", "NetOutput"));
    CHAIN(NODE(const_0)->EDGE(0, 1)->NODE("add"));
  };

  ComputeGraphPtr graph = ToComputeGraph(g1);
  AttrUtils::SetInt(graph, "globalworkspace_type", 1);
  AttrUtils::SetInt(graph, "globalworkspace_size", 1);
  AttrUtils::SetBool(graph, "_is_graph_prefetch", true);

  ProfilingProperties::Instance().is_load_profiling_ = true;

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  const size_t logic_var_base = VarManager::Instance(graph->GetSessionID())->GetVarMemLogicBase();
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_TASK_GEN_VAR_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, logic_var_base));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120));

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();

  auto netoutput = graph->FindNode("output");
  netoutput->GetOpDesc()->SetSrcName({"relu"});
  netoutput->GetOpDesc()->SetSrcIndex({0});

  {
    const auto &node = graph->FindNode("add");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("add", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 0 add
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("add");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  // task 1 relu
  {
    const auto &node = graph->FindNode("relu");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("relu", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 1 relu
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("relu");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  std::string output_file = "outputfile.exeom";
  {
    ModelBufferData model;
    NanoModelSaveHelper nano_model_helper;
    EXPECT_EQ(nano_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");
}
}  // namespace ge
