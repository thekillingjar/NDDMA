/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdio>
#include <fstream>
#include <memory>
#include <gtest/gtest.h>

#include "framework/common/helper/mobile_model_helper.h"
#include "framework/omg/ge_init.h"
#include "framework/common/types.h"

#include "common/env_path.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/model/ge_model.h"
#include "common/opskernel/ops_kernel_info_types.h"

#include "graph/buffer/buffer_impl.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/graph_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_local_context.h"
#include "proto/task.pb.h"

namespace {

std::vector<char> CreateStubBin()
{
  auto ascend_install_path = ge::EnvPath().GetAscendInstallPath();
  GELOGI("ascend_install_path: %s.", ascend_install_path.c_str());
  std::string op_bin_path = ascend_install_path + "/fwkacllib/lib64/switch_by_index.o";
  std::vector<char> buf;
  std::ifstream file(op_bin_path.c_str(), std::ios::binary | std::ios::in);
  if (!file.is_open()) {
    GELOGW("file: %s does not exist or is unaccessible.", op_bin_path.c_str());
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

ge::NodePtr CreateNode(ge::ComputeGraph &graph, const std::string &name,
  const std::string &type, int in_num, int out_num, ge::DataType data_type = ge::DT_INT64)
{
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  ge::GeTensorDesc tensor(ge::GeShape({8}), ge::FORMAT_ND, data_type);
  ge::TensorUtils::SetSize(tensor, 64);
  std::vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(index * 64 + i * 64);
  }
  op_desc->SetInputOffset(input_offset);

  std::vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(index * 64 + in_num * 64 + i * 64);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});

  const static std::set<std::string> kGeLocalTypes{ "Data", "Constant", "NetOutput" };
  op_desc->SetOpKernelLibName((kGeLocalTypes.count(type) > 0U) ? "DNN_VM_GE_LOCAL_OP_STORE" : "DNN_VM_RTS_OP_STORE");
  return graph.AddNode(op_desc);
}

ge::GeRootModelPtr GenGeRootModel(bool is_invaild_data_type = false)
{
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("test");

  ge::NodePtr input0_node = CreateNode(*graph, "input0", "Data", 1, 1);
  ge::OpDescPtr input0_op_desc = input0_node->GetOpDesc();
  EXPECT_NE(input0_op_desc, nullptr);

  ge::NodePtr input1_node = CreateNode(*graph, "input1", "Data", 1, 1);
  ge::OpDescPtr input1_op_desc = input1_node->GetOpDesc();
  EXPECT_NE(input1_op_desc, nullptr);

  // not support data type
  if (is_invaild_data_type) {
    ge::NodePtr input2_node = CreateNode(*graph, "input2", "Data", 1, 1, ge::DT_FLOAT4_E1M2);
    ge::OpDescPtr input2_op_desc = input2_node->GetOpDesc();
    EXPECT_NE(input2_op_desc, nullptr);
  }

  ge::NodePtr add_node = CreateNode(*graph, "add", "Add", 2, 1);
  ge::OpDescPtr add_op_desc = add_node->GetOpDesc();
  EXPECT_NE(add_op_desc, nullptr);

  std::string test_float_attr_str = "test_float_attr";
  float test_float_value = 1.0;
  ge::AttrUtils::SetFloat(add_op_desc, test_float_attr_str, test_float_value);

  std::string test_bytes_attr_str = "test_bytes_attr";
  ge::Buffer test_bytes_buffer(10, 1);
  ge::AttrUtils::SetBytes(add_op_desc, test_bytes_attr_str, test_bytes_buffer);

  std::string test_list_attr_str = "test_list_attr";
  std::vector<int> test_list_attr_value = {1, 2, 3, 4};
  ge::AttrUtils::SetListInt(add_op_desc, test_list_attr_str, test_list_attr_value);

  std::string test_list_list_int_attr_str = "test_list_list_int_attr";
  std::vector<std::vector<int64_t>> test_list_list_int_value = {{1, 2, 3}, {1, 2, 3}};
  ge::AttrUtils::SetListListInt(add_op_desc,
    test_list_list_int_attr_str, test_list_list_int_value);

  std::string test_list_list_float_attr_str = "test_list_list_float_attr";
  std::vector<std::vector<float>> test_list_list_float_value = {{1.1, 2.34, 3.4}, {1.1, 2.8, 3.5}};
  ge::AttrUtils::SetListListFloat(add_op_desc,
    test_list_list_float_attr_str, test_list_list_float_value);

  ge::GeAttrValue::NAMED_ATTRS test_named_attrs;
  std::string named_attr_str0 = "named_attr_test_str";
  std::string named_attr_str_value = "testtest";
  ge::AttrUtils::SetStr(test_named_attrs, named_attr_str0, named_attr_str_value);
  std::string named_attr_str1 = "named_attr_test_int";
  int named_attr_int_value = 1232;
  ge::AttrUtils::SetInt(test_named_attrs, named_attr_str1, named_attr_int_value);
  std::string named_attr = "named_attr";
  (void)ge::AttrUtils::SetNamedAttrs(add_op_desc, named_attr, test_named_attrs);
  
  std::string dynamic_inputs_indexes_str = "_dynamic_inputs_indexes";
  std::vector<std::vector<int64_t>> dynamic_inputs_indexes = {{0, 1}};
  ge::AttrUtils::SetListListInt(add_op_desc,
    dynamic_inputs_indexes_str, dynamic_inputs_indexes);

  std::string dynamic_outputs_indexes_str = "_dynamic_outputs_indexes";
  std::vector<std::vector<int64_t>> dynamic_outputs_indexes = {{0}};
  ge::AttrUtils::SetListListInt(add_op_desc,
    dynamic_outputs_indexes_str, dynamic_outputs_indexes);

  const std::string kernel_name = "test";
  std::string kernel_name_str = "_kernelname";
  ge::AttrUtils::SetStr(add_op_desc, kernel_name_str, kernel_name);

  ge::NodePtr netoutput_node = CreateNode(*graph, "output0", "NetOutput", 1, 1);
  ge::OpDescPtr netoutput_op_desc = netoutput_node->GetOpDesc();
  EXPECT_NE(netoutput_op_desc, nullptr);

  ge::GeModelPtr ge_model = std::make_shared<ge::GeModel>();
  ge_model->SetGraph(graph);
  ge_model->SetName("test_model");
  
  ge::TBEKernelStore tbe_kernel_store;
  const auto kernel = ge::MakeShared<ge::OpKernelBin>(kernel_name, CreateStubBin());
  tbe_kernel_store.AddTBEKernel(kernel);
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  
  std::shared_ptr<domi::ModelTaskDef> model_task = std::make_shared<domi::ModelTaskDef>();
  ge_model->SetModelTaskDef(model_task);
  
  domi::TaskDef *task_def0 = model_task->add_task();
  task_def0->set_type(static_cast<uint32_t>(ge::ModelTaskType::MODEL_TASK_KERNEL));
  task_def0->set_stream_id(add_op_desc->GetStreamId());
  domi::KernelDef *kernel_def0 = task_def0->mutable_kernel();
  kernel_def0->set_stub_func("stub_func_add");
  kernel_def0->set_args_size(64);
  std::string args(64, '1');
  kernel_def0->set_args(args.data(), 64);
  kernel_def0->set_kernel_name("test");

  domi::KernelContext *context0 = kernel_def0->mutable_context();
  context0->set_kernel_type(0);
  context0->set_op_index(add_op_desc->GetId());
  uint16_t args_offset[9] = {0};
  context0->set_args_offset(args_offset, 9 * sizeof(uint16_t));
  std::string args_format = "{i_desc0}{o_desc0}";
  context0->set_args_format(args_format);

  ge::GeRootModelPtr ge_root_model = std::make_shared<ge::GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), ge::SUCCESS);
  ge_root_model->SetModelName(ge_model->GetName());
  ge_root_model->SetSubgraphInstanceNameToModel(graph->GetName(), ge_model);
  return ge_root_model;
}

} // namespace

namespace ge {

class UtestMobileModelHelper : public testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

TEST_F(UtestMobileModelHelper, ModelToMobile)
{
  std::string output_file = "mobile_model.om";
  ModelBufferData model;
  MobileModelHelper model_save_helper;
  GeRootModelPtr ge_root_model = GenGeRootModel();
  EXPECT_EQ(model_save_helper.SaveToOmRootModel(ge_root_model, output_file, model, false), ge::SUCCESS);
  EXPECT_NE(model_save_helper.SaveToOmRootModel(ge_root_model, output_file, model, true), ge::SUCCESS);
  GeRootModelPtr ge_root_model_invaild_data_type = GenGeRootModel(true);
  EXPECT_NE(model_save_helper.SaveToOmRootModel(
    ge_root_model_invaild_data_type, output_file, model, false), ge::SUCCESS);
  system("rm -rf mobile_model.omc");
}

} // namespace ge
