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
#include <iostream>
#include <fstream>
#include <sstream>
#include "faker/fake_value.h"
#include "faker/global_data_faker.h"
#include "faker/aicpu_taskdef_faker.h"
#include "faker/aicore_taskdef_faker.h"
#include "faker/model_data_faker.h"
#include "faker/ge_model_builder.h"
#include "faker/label_switch_task_def_faker.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/tensor_utils.h"
#include "common/share_graph.h"
#include "mmpa/mmpa_api.h"
#include "securec.h"
#include "graph/operator_reg.h"
#include "graph/utils/op_desc_utils.h"
#include "framework/common/ge_types.h"
#include "stub/hostcpu_mmpa_stub.h"
#include "check/executor_statistician.h"
#include "ge_running_env/path_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "framework/common/helper/nano_model_save_helper.h"
#include "framework/common/helper/pre_model_helper.h"
#include "framework/common/tlv/nano_dbg_desc.h"
#include "common/preload/dbg/nano_dbg_data.h"
#include "common/preload/model/pre_model_partition_utils.h"
#include "ge_local_context.h"
#include "common/path_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/env_path.h"
#include "graph_metadef/common/ge_common/util.h"

namespace ge {
using namespace gert;
class ModelHelperTest : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

std::vector<char> CreateStubBin() {
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
  std::cout << std::string(buf.data()) << std::endl;
  return buf;
}

static void CheckDbgFileTLv(const std::string &filename) {
  std::ifstream fs(filename, std::ifstream::in);
  EXPECT_EQ(fs.is_open(), true);

  std::ostringstream ctx;
  ctx << fs.rdbuf();
  fs.close();

  string  dbg_tlv = ctx.str();
  const char *buff = dbg_tlv.c_str();
  size_t buff_size = dbg_tlv.length();

  size_t len = 0;
  const DbgDataHead *head = (const DbgDataHead *)buff;
  EXPECT_EQ(head->version_id, 0);
  EXPECT_EQ(head->magic, DBG_DATA_HEAD_MAGIC);
  len = len + sizeof(DbgDataHead);

  const TlvHead *tlv = (TlvHead *)head->tlv;
  EXPECT_EQ(tlv->type, DBG_L1_TLV_TYPE_MODEL_NAME);
  len = len + sizeof(TlvHead) + tlv->len;

  tlv = (const TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L1_TLV_TYPE_OP_DESC);
  DbgOpDescTlv1 *op_tlv = (DbgOpDescTlv1 *)tlv->data;
  EXPECT_EQ(op_tlv->num, 3);
  len = len + sizeof(TlvHead) + tlv->len;

  EXPECT_EQ(buff_size, len);
}

TEST_F(ModelHelperTest, SaveToOmRootModel_For_Nano) {
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_env, MMPA_MAX_PATH);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/fwkacllib/lib64").c_str(), 1);
  auto graph = ShareGraph::AtcNanoGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin1"))
                              .FakeTbeBin({"Add"})
                              .BuildGeRootModel();
  auto ge_model_map = ge_root_model->GetSubgraphInstanceNameToModel();
  auto ge_model = ge_model_map[graph->GetName()];
  auto model_task_def = ge_model->GetModelTaskDefPtr();
  model_task_def->mutable_task(0)->mutable_kernel()->set_kernel_name("add1_faked_kernel");
  model_task_def->mutable_task(1)->mutable_kernel()->set_kernel_name("add2_faked_kernel");
  model_task_def->mutable_task(2)->mutable_kernel()->set_kernel_name("add3_faked_kernel");

  graph->FindNode("add1")->GetOpDesc()->SetInputOffset({0, 2048});
  graph->FindNode("add1")->GetOpDesc()->SetOutputOffset({0});
  graph->FindNode("add2")->GetOpDesc()->SetInputOffset({0, 2048});
  graph->FindNode("add2")->GetOpDesc()->SetOutputOffset({0});
  graph->FindNode("add3")->GetOpDesc()->SetInputOffset({0, 2048});
  graph->FindNode("add3")->GetOpDesc()->SetOutputOffset({0});

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "pb_exeom_for_nano");
  std::string output = om_path + ".exeom";

  ModelBufferData model_buff;
  NanoModelSaveHelper helper;
  helper.SetSaveMode(true);
  Status ret = helper.SaveToOmRootModel(ge_root_model, output, model_buff, false);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(IsFile((om_path + ".exeom").c_str()), true);
  EXPECT_EQ(IsFile((om_path + ".dbg").c_str()), true);

  CheckDbgFileTLv((om_path + ".dbg").c_str());
  unsetenv("LD_LIBRARY_PATH");
  mmSetEnv("LD_LIBRARY_PATH", old_env, 1);
}

TEST_F(ModelHelperTest, SaveToOmRootModel_For_NanoHostFunc) {
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_env, MMPA_MAX_PATH);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/fwkacllib/lib64").c_str(), 1);
  auto graph = ShareGraph::AtcNanoGraph();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.AddTaskDef("Add", AiCoreTaskDefFaker("AddStubBin1"))
                              .FakeTbeBin({"Add"})
                              .BuildGeRootModel();

  auto ge_model_map = ge_root_model->GetSubgraphInstanceNameToModel();
  auto model = ge_model_map[graph->GetName()];
  auto model_task_def = model->GetModelTaskDefPtr();
  model_task_def->mutable_task(0)->mutable_kernel()->set_kernel_name("add1_faked_kernel");
  model_task_def->mutable_task(1)->mutable_kernel()->set_kernel_name("add2_faked_kernel");
  model_task_def->mutable_task(2)->mutable_kernel()->set_kernel_name("add3_faked_kernel");

  graph->FindNode("add1")->GetOpDesc()->SetInputOffset({0, 2048});
  graph->FindNode("add1")->GetOpDesc()->SetOutputOffset({0});
  graph->FindNode("add2")->GetOpDesc()->SetInputOffset({0, 2048});
  graph->FindNode("add2")->GetOpDesc()->SetOutputOffset({0});
  graph->FindNode("add3")->GetOpDesc()->SetInputOffset({0, 2048});
  graph->FindNode("add3")->GetOpDesc()->SetOutputOffset({0});

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);

  const auto &node = graph->FindNode("add1");
  const auto &op_desc = node->GetOpDesc();

  op_desc->SetOutputOffset({8});
  op_desc->SetInputOffset({0, 2048});
  PreModelPartitionUtils::GetInstance().SetZeroCopyTable(128, 0);
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

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "pb_exeom_for_nano");
  std::string output = om_path + ".exeom";

  ModelBufferData model_buff;
  NanoModelSaveHelper helper;
  helper.SetSaveMode(true);
  Status ret = helper.SaveToOmRootModel(ge_root_model, output, model_buff, false);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(IsFile((om_path + ".exeom").c_str()), true);
  EXPECT_EQ(IsFile((om_path + ".dbg").c_str()), true);

  CheckDbgFileTLv((om_path + ".dbg").c_str());
  unsetenv("LD_LIBRARY_PATH");
  mmSetEnv("LD_LIBRARY_PATH", old_env, 1);
}

TEST_F(ModelHelperTest, SaveToOmRootModel_For_NanoWHILESwitch) {
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_env, MMPA_MAX_PATH);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/fwkacllib/lib64").c_str(), 1);
  auto graph = ShareGraph::WhileGraph3();
  graph->TopologicalSorting();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();

  auto cond_graph = graph->GetSubgraph("cond_instance");
  auto body_graph = graph->GetSubgraph("body_instance");
  {
    const auto &node = cond_graph->FindNode("LabelSet_0");
    const auto &op_desc = node->GetOpDesc();

    // task 0 label_set 0
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = cond_graph->FindNode("Stream_0");
    const auto &op_desc = node->GetOpDesc();

    // task 1 stream0 active
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_ACTIVE));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::StreamActiveDef *stream_task_def = task_def->mutable_stream_active();
    stream_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = cond_graph->FindNode("LessThan_5");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("LessThan_5", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 2 LessThan_5
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("LessThan_5");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = cond_graph->FindNode("SwitchByIndex");
    const auto &op_desc = node->GetOpDesc();

    GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 512);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetOutputOffset({1024});

    // task 3 switch_by_index
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
    const auto &node = body_graph->FindNode("LabelSet_1");
    const auto &op_desc = node->GetOpDesc();

    // task 4 label_set 1
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = body_graph->FindNode("Stream_1");
    const auto &op_desc = node->GetOpDesc();

    // task 5 stream1 active
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_ACTIVE));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::StreamActiveDef *stream_task_def = task_def->mutable_stream_active();
    stream_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = body_graph->FindNode("add");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("add", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 6 add
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

  {
    const auto &node = body_graph->FindNode("LabelGoto");
    const auto &op_desc = node->GetOpDesc();

    GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 512);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetOutputOffset({1024});

    // task 7 label goto
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
    const auto &node = body_graph->FindNode("LabelSet_2");
    const auto &op_desc = node->GetOpDesc();

    // task 8 label_set 2
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("NetOutput");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("NetOutput", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 9 NetOutput
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("NetOutput");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "pb_exeom_for_nano");
  std::string output = om_path + ".exeom";

  ModelBufferData model_buff;
  NanoModelSaveHelper helper;
  helper.SetSaveMode(true);
  dlog_setlevel(-1, 0, 1);
  Status ret = helper.SaveToExeOmModel(ge_model, output, model_buff);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(IsFile((om_path + ".exeom").c_str()), true);
  EXPECT_EQ(IsFile((om_path + ".dbg").c_str()), true);

  CheckDbgFileTLv((om_path + ".dbg").c_str());
  unsetenv("LD_LIBRARY_PATH");
  mmSetEnv("LD_LIBRARY_PATH", old_env, 1);
}

TEST_F(ModelHelperTest, SaveToOmRootModel_For_NanoIFSwitch) {
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_env, MMPA_MAX_PATH);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/fwkacllib/lib64").c_str(), 1);
  auto graph = ShareGraph::IfGraphWithSwitch();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();

  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(graph);

  const auto model_def = MakeShared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_def);
  TBEKernelStore tbe_kernel_store = ge_model->GetTBEKernelStore();

  auto then_graph = graph->GetSubgraph("then");
  auto else_graph = graph->GetSubgraph("else");
  {
    const auto &node = then_graph->FindNode("SwitchByIndex");
    const auto &op_desc = node->GetOpDesc();

    GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 512);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetOutputOffset({1024});

    // task 0 SwitchByIndex
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
    const auto &node = then_graph->FindNode("LabelSet_0");
    const auto &op_desc = node->GetOpDesc();

    // task 1 label_set 0
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = then_graph->FindNode("Stream_0");
    const auto &op_desc = node->GetOpDesc();

    // task 2 stream0 active
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_ACTIVE));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::StreamActiveDef *stream_task_def = task_def->mutable_stream_active();
    stream_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = then_graph->FindNode("shape");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("shape", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 3 shape
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("shape");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = then_graph->FindNode("LabelGoto");
    const auto &op_desc = node->GetOpDesc();

    GeTensorDesc tensor(GeShape({1, 4, 4, 8}), FORMAT_NCHW, DT_FLOAT);
    TensorUtils::SetSize(tensor, 512);
    op_desc->AddOutputDesc(tensor);
    op_desc->SetOutputOffset({1024});

    // task 4 label goto
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
    const auto &node = else_graph->FindNode("LabelSet_1");
    const auto &op_desc = node->GetOpDesc();

    // task 5 label_set 1
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = else_graph->FindNode("Stream_1");
    const auto &op_desc = node->GetOpDesc();

    // task 6 stream1 active
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_STREAM_ACTIVE));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::StreamActiveDef *stream_task_def = task_def->mutable_stream_active();
    stream_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = else_graph->FindNode("shape");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("shape", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 7 shape
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("shape");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  {
    const auto &node = else_graph->FindNode("LabelSet_2");
    const auto &op_desc = node->GetOpDesc();

    // task 8 label_set 2
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_LABEL_SET));
    task_def->set_stream_id(op_desc->GetStreamId());

    domi::LabelSetDef *label_set_task_def = task_def->mutable_label_set();
    label_set_task_def->set_op_index(op_desc->GetId());
  }

  {
    const auto &node = graph->FindNode("NetOutput");
    const auto &op_desc = node->GetOpDesc();

    const auto kernel = MakeShared<OpKernelBin>("NetOutput", CreateStubBin());
    tbe_kernel_store.AddTBEKernel(kernel);
    ge_model->SetTBEKernelStore(tbe_kernel_store);

    // task 9 NetOutput
    domi::TaskDef *task_def = model_def->add_task();
    task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
    task_def->set_stream_id(op_desc->GetStreamId());
    domi::KernelDef *kernel_def = task_def->mutable_kernel();
    kernel_def->set_stub_func("stub_func");
    kernel_def->set_args_size(64);
    string args(64, '1');
    kernel_def->set_args(args.data(), 64);
    kernel_def->set_kernel_name("NetOutput");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "pb_exeom_for_nano");
  std::string output = om_path + ".exeom";

  ModelBufferData model_buff;
  NanoModelSaveHelper helper;
  helper.SetSaveMode(true);
  Status ret = helper.SaveToExeOmModel(ge_model, output, model_buff);

  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(IsFile((om_path + ".exeom").c_str()), true);
  EXPECT_EQ(IsFile((om_path + ".dbg").c_str()), true);

  CheckDbgFileTLv((om_path + ".dbg").c_str());
  unsetenv("LD_LIBRARY_PATH");
  mmSetEnv("LD_LIBRARY_PATH", old_env, 1);
}

TEST_F(ModelHelperTest, SaveToOmRootModel_For_NanoFIFOtest) {
  auto ascend_install_path = EnvPath().GetAscendInstallPath();
  char old_env[MMPA_MAX_PATH] = {'\0'};
  (void)mmGetEnv("LD_LIBRARY_PATH", old_env, MMPA_MAX_PATH);
  setenv("LD_LIBRARY_PATH", (ascend_install_path + "/fwkacllib/lib64").c_str(), 1);
  auto graph = ShareGraph::GraphWithFifoWindowCache();
  GeModelBuilder builder(graph);
  auto ge_root_model = builder.BuildGeRootModel();
  auto sub_models = ge_root_model->GetSubgraphInstanceNameToModel();
  EXPECT_EQ(sub_models.size(), 1);
  auto ge_model = sub_models[graph->GetName()];
  EXPECT_NE(ge_model, nullptr);

  PreModelPartitionUtils::GetInstance().SetZeroCopyTable(128, 0);
  PreModelPartitionUtils::GetInstance().SetZeroCopyTable(136, 8);

  std::vector<uint64_t> weights_value(64, 1024);
  size_t weight_size = weights_value.size() * sizeof(uint64_t);
  ge_model->SetGraph(graph);

  ge_model->SetWeight(Buffer::CopyFrom((uint8_t *)weights_value.data(), weight_size));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 3));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_EVENT_NUM, 1));
  EXPECT_TRUE(AttrUtils::SetInt(ge_model, ATTR_MODEL_LABEL_NUM, 0));
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
    const auto &node = graph->FindNode("conv");
    const auto &op_desc = node->GetOpDesc();
    op_desc->SetInputOffset({0});
    op_desc->SetOutputOffset({32});
    ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_INPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});
    ge::AttrUtils::SetListInt(op_desc, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, {RT_MEMORY_L1});

    const auto kernel = MakeShared<OpKernelBin>("conv", CreateStubBin());
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
    kernel_def->set_kernel_name("conv");

    domi::KernelContext *context = kernel_def->mutable_context();
    context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
    context->set_op_index(op_desc->GetId());
    uint16_t args_offset[9] = {0};
    context->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  }

  auto om_path = PathJoin(GetRunPath().c_str(), "temp");
  Mkdir(om_path.c_str());
  om_path = PathJoin(om_path.c_str(), "pb_exeom_for_nano");
  std::string output = om_path + ".exeom";

  ModelBufferData model_buff;
  NanoModelSaveHelper helper;
  helper.SetSaveMode(true);
  Status ret = helper.SaveToOmRootModel(ge_root_model, output, model_buff, false);
  dlog_setlevel(-1, 3, 0);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(IsFile((om_path + ".exeom").c_str()), true);
  EXPECT_EQ(IsFile((om_path + ".dbg").c_str()), true);
  unsetenv("LD_LIBRARY_PATH");
  mmSetEnv("LD_LIBRARY_PATH", old_env, 1);
//  int64_t *a = nullptr;
//  EXPECT_EQ(*a, 16);
}

TEST_F(ModelHelperTest, SaveToOm_for_SplitAndUpgraded_Opp) {
  auto options = GetThreadLocalContext().GetAllGlobalOptions();
  std::vector<std::string> paths;
  CreateBuiltInSplitAndUpgradedSo(paths);

  GeModelPtr ge_model = ge::MakeShared<GeModel>();
  ComputeGraphPtr graph = ge::MakeShared<ComputeGraph>("g1");
  ge_model->SetGraph(graph);
  std::string output_file{"split_opp.om"};
  ModelBufferData buffer_data;
  GeRootModelPtr ge_root_model = ge::MakeShared<GeRootModel>();
  ComputeGraphPtr root_graph = ge::MakeShared<ComputeGraph>("subgraph");
  (void)AttrUtils::SetBool(root_graph, ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED, true);
  ge_root_model->SetRootGraph(root_graph);
  EXPECT_EQ(ge_root_model->CheckAndSetNeedSoInOM(), SUCCESS);
  EXPECT_EQ(ge_root_model->GetSoInOmFlag(), 0x8000);

  std::map<string, string> env_options;
  env_options["ge.host_env_os"] = "linux";
  env_options["ge.host_env_cpu"] = "x86_64";
  (void)GetThreadLocalContext().SetGlobalOption(env_options);

  ModelHelper model_helper;
  auto ret = model_helper.SaveToOmModel(ge_model, output_file, buffer_data, ge_root_model);
  EXPECT_NE(ret, SUCCESS);
  (void)GetThreadLocalContext().SetGlobalOption(options);
  for (const auto &path : paths) {
    ge::PathUtils::RemoveDirectories(path);
  }
}
}  // namespace ge
