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

#include "macro_utils/dt_public_scope.h"
#include "framework/common/helper/pre_model_helper.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "framework/omg/ge_init.h"
#include "common/model/ge_model.h"
#include "common/helper/model_parser_base.h"
#include "graph/buffer/buffer_impl.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
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
constexpr size_t kElfDataIdx = 5UL;
std::vector<char> CreateStubBin(bool isLittleEndian) {
  auto ascend_install_path = ge::EnvPath().GetAscendInstallPath();
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
  std::cout << "[BEFORECHANGE]:" << buf[kElfDataIdx] << std::endl;
  if (isLittleEndian) {
    buf[kElfDataIdx] = 1;
  } else {
    buf[kElfDataIdx] = 2;
  }
  std::cout << "[AFTERCHANGE]:" << buf[kElfDataIdx] << std::endl;
  std::cout << std::string(buf.data()) << std::endl;
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
  const auto kernel = MakeShared<OpKernelBin>("test", std::vector<char>(64, 0));
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

class UtestPreModelSaveHelper : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestPreModelSaveHelper, ModelToOm)
{
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  std::string output_file = "outputfile.om";
  ModelBufferData model;
  PreModelHelper model_save_helper;
  GeRootModelPtr ge_root_model = GenGeRootModel();
  // output_file is null
  EXPECT_NE(model_save_helper.SaveToOmRootModel(ge_root_model, "", model, false), SUCCESS);
  // Unknown shape not support
  EXPECT_NE(model_save_helper.SaveToOmRootModel(ge_root_model, output_file, model, true), SUCCESS);
  // opdesc is nullptr
  EXPECT_NE(model_save_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), SUCCESS);

  system("rm -rf outputfile.exeom");
  system("rm -rf outputfile.dbg");
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
}

TEST_F(UtestPreModelSaveHelper, RepackLSBSoToOm) {
  uint32_t mem_offset = 0U;
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("add_n", ADDN));  // ccKernelType::TE
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
    const auto kernel = MakeShared<OpKernelBin>("test", CreateStubBin(true));
    tbe_kernel_store.AddTBEKernel(kernel);
    tbe_kernel_store.PreBuild();
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

  {
    std::string output_file = "outputfile.exeom";
    ModelBufferData model;
    PreModelHelper pre_model_helper;
    EXPECT_EQ(pre_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);

    output_file = "";
    EXPECT_NE(pre_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
}

TEST_F(UtestPreModelSaveHelper, RepackMSBSoToOm) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->EDGE(0, 0)->NODE("add_n", ADDN));  // ccKernelType::TE
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
    const auto kernel = MakeShared<OpKernelBin>("test", CreateStubBin(false));
    tbe_kernel_store.AddTBEKernel(kernel);
    tbe_kernel_store.PreBuild();
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

  {
    std::string output_file = "outputfile.exeom";
    ModelBufferData model;
    PreModelHelper pre_model_helper;
    EXPECT_NE(pre_model_helper.SaveToExeOmModel(ge_model, output_file, model), SUCCESS);
  }
}
}  // namespace ge
