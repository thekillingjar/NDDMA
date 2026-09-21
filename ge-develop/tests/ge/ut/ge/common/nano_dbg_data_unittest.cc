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
#include "common/preload/dbg/nano_dbg_data.h"
#include "common/debug/log.h"
#include "framework/common/types.h"
#include "framework/common/tlv/nano_dbg_desc.h"
#include "common/ge_inner_error_codes.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge_graph_dsl/graph_dsl.h"
#include "macro_utils/dt_public_unscope.h"
#include "graph/passes/graph_builder_utils.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
class UTEST_nano_dbg_data : public testing::Test {
 protected:
  void SetUp() {
    GenGeModel();
  }
  void TearDown() {}
  void GenGeModel()
  {
    auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_INT64, {1});
    auto data2 = OP_CFG(CONSTANT).TensorDesc(FORMAT_ND, DT_INT64, {1});
    auto data3 = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_INT64, {1});
    DEF_GRAPH(g1) {
      CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("add", "Add")->EDGE(0, 0)->NODE("NetOutput1", "NetOutput"));
      CHAIN(NODE("data2", data2)->EDGE(0, 1)->NODE("add", "Add")->EDGE(1, 0)->NODE("NetOutput2", "NetOutput"));
      CHAIN(NODE("data3", data3)->EDGE(0, 2)->NODE("add", "Add"));
    };
    graph_ = ToComputeGraph(g1);
    op_desc_ = graph_->FindNode("add")->GetOpDesc();
    EXPECT_NE(op_desc_, nullptr);
    const int input_num = 3;

    const std::vector<int64_t> v_memory_type{RT_MEMORY_HBM, RT_MEMORY_L1};
    AttrUtils::SetListInt(op_desc_, ATTR_NAME_OUTPUT_MEM_TYPE_LIST, v_memory_type);
    std::vector<uint64_t> output_offset{10, 10};
    (void)op_desc_->SetExtAttr("task_addr_offset", output_offset);
    op_desc_->SetOutputOffset({2048, 2048});
    for (int i = 0; i < 2; ++i) {
      auto output_tensor = op_desc_->MutableOutputDesc(i);
      output_tensor->SetShape(GeShape({1,4,4,8}));
      output_tensor->SetOriginShape(GeShape());
      output_tensor->SetDataType(DT_INT64);
      TensorUtils::SetSize(*output_tensor, 64);
    }

    for (int i = 0; i < input_num; i++) {
      auto input_desc = op_desc_->MutableInputDesc(i);
      int64_t input_size = 8;
      AttrUtils::SetInt(input_desc, ATTR_NAME_INPUT_ORIGIN_SIZE, input_size);
      input_desc->SetShape(GeShape());
      input_desc->SetDataType(DT_INT64);
      TensorUtils::SetSize(*input_desc, 64);
    }
    const std::vector<int64_t> v_memory_type1{RT_MEMORY_L1, RT_MEMORY_HBM, RT_MEMORY_HBM};
    AttrUtils::SetListInt(op_desc_, ATTR_NAME_INPUT_MEM_TYPE_LIST, v_memory_type1);
    op_desc_->SetInputOffset({2048, 2048, 2048});
    std::vector<uint64_t> input_offset{10, 10, 10};
    (void)op_desc_->SetExtAttr("task_addr_offset", input_offset);

    std::vector<int64_t> tvm_workspace_memory_type = {ge::AicpuWorkSpaceType::CUST_LOG};
    AttrUtils::SetListInt(op_desc_, ATTR_NAME_AICPU_WORKSPACE_TYPE, tvm_workspace_memory_type);
    op_desc_->SetWorkspace({});
    op_desc_->SetWorkspaceBytes(std::vector<int64_t>{32});

    const static std::set<std::string> kGeLocalTypes{ DATA, CONSTANT, VARIABLE, NETOUTPUT };
    op_desc_->SetOpKernelLibName((kGeLocalTypes.count(ADD) > 0U) ? "DNN_VM_GE_LOCAL_OP_STORE" : "DNN_VM_RTS_OP_STORE");

    op_desc_->SetIsInputConst({true});
    std::vector<string> original_op_names = {"conv", "add"};
    AttrUtils::SetListStr(op_desc_, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_op_names);

    ge_model_ = std::make_shared<GeModel>();
    ge_model_->SetGraph(graph_);

    model_task_ = std::make_shared<domi::ModelTaskDef>();
    (void)model_task_->add_task();
    ge_model_->SetModelTaskDef(model_task_);
  }

 public:
  GeModelPtr ge_model_;
  OpDescPtr op_desc_;
 private:
  ComputeGraphPtr graph_;
  std::shared_ptr<domi::ModelTaskDef> model_task_;

};

TEST_F(UTEST_nano_dbg_data, dsa_dbg) {
  domi::TaskDef *task_def = ge_model_->GetModelTaskDefPtr()->mutable_task(0);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_DSA));
  task_def->set_stream_id(0);
  task_def->mutable_dsa_task()->set_op_index(op_desc_->GetId());

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(dbg.op_list_.size(), 1);
}

TEST_F(UTEST_nano_dbg_data, kernel_ex_dbg) {
  domi::TaskDef *task_def = ge_model_->GetModelTaskDefPtr()->mutable_task(0);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL_EX));
  task_def->set_stream_id(0);
  task_def->mutable_kernel_ex()->set_op_index(op_desc_->GetId());

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(dbg.op_list_.size(), 1);
}

TEST_F(UTEST_nano_dbg_data, kernel_dbg_check) {
  domi::TaskDef *task_def = ge_model_->GetModelTaskDefPtr()->mutable_task(0);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  task_def->set_stream_id(0);
  task_def->mutable_kernel()->set_block_dim(10);
  task_def->mutable_kernel()->mutable_context()->set_op_index(op_desc_->GetId());

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(dbg.op_list_.size(), 1);

  NanoDbgOpDesc &op = dbg.op_list_[0];
  EXPECT_EQ(op.op_name, "add");
  EXPECT_EQ(op.op_type, "Add");
  EXPECT_EQ(op.block_dim, 10);

  EXPECT_EQ(op.input_list.size(), 2);
  EXPECT_EQ(op.input_list[0].shape_dims.size(), 0);
  EXPECT_EQ(op.input_list[0].data_type, DT_INT64);
  EXPECT_EQ(op.input_list[0].addr_type, toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);
  EXPECT_NE(op.input_list[0].addr, 0);
  EXPECT_EQ(op.input_list[0].size, 8);
  EXPECT_EQ(op.input_list[0].offset, 10);

  EXPECT_EQ(op.output_list.size(), 1);
  EXPECT_EQ(op.output_list[0].shape_dims.size(), 4);
  EXPECT_EQ(op.output_list[0].original_shape_dims.size(), 0);
  EXPECT_EQ(op.output_list[0].data_type, DT_INT64);
  EXPECT_EQ(op.output_list[0].addr_type, toolkit::aicpu::dump::AddressType::NANO_IO_ADDR);
  EXPECT_NE(op.output_list[0].addr, 0);
  EXPECT_EQ(op.output_list[0].size, 1024);
  EXPECT_EQ(op.output_list[0].offset, 10);

  EXPECT_EQ(op.buffer_list.size(), 2);
  EXPECT_EQ(op.buffer_list[0].type, toolkit::aicpu::dump::L1);
  EXPECT_EQ(op.buffer_list[0].addr, 0);

  EXPECT_EQ(op.workspace_list.size(), 1);
  EXPECT_EQ(op.workspace_list[0].type, toolkit::aicpu::dump::Workspace::LOG);
  EXPECT_EQ(op.workspace_list[0].size, 32);

  EXPECT_EQ(op.mem_info_list.size(), 1);
  EXPECT_EQ(op.mem_info_list[0].input_mem_size, 192);
  EXPECT_EQ(op.mem_info_list[0].output_mem_size, 128);
  EXPECT_EQ(op.mem_info_list[0].weight_mem_size, 64);
  EXPECT_EQ(op.mem_info_list[0].total_mem_size, 384);
}

TEST_F(UTEST_nano_dbg_data, kernel_tlv_check) {
  domi::TaskDef *task_def = ge_model_->GetModelTaskDefPtr()->mutable_task(0);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  task_def->set_stream_id(0);
  task_def->mutable_kernel()->set_block_dim(10);
  task_def->mutable_kernel()->mutable_context()->set_op_index(op_desc_->GetId());

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
  EXPECT_EQ(dbg.op_list_.size(), 1);

  EXPECT_NE(dbg.GetDbgDataSize(), 0);
  DbgDataHead *head = (DbgDataHead *)dbg.GetDbgData();
  EXPECT_EQ(head->version_id, 0);
  EXPECT_EQ(head->magic, DBG_DATA_HEAD_MAGIC);

  // l1 tlv
  TlvHead *tlv = (TlvHead *)head->tlv;
  EXPECT_EQ(tlv->type, DBG_L1_TLV_TYPE_MODEL_NAME);

  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L1_TLV_TYPE_OP_DESC);
  DbgOpDescTlv1 *op_tlv = (DbgOpDescTlv1 *)tlv->data;
  EXPECT_EQ(op_tlv->num, 1);
  DbgOpDescParamTlv1 *op_param = (DbgOpDescParamTlv1 *)op_tlv->param;
  EXPECT_EQ(op_param->block_dim, 10);

  // l2 tlv 0
  tlv = (TlvHead *)op_param->l2_tlv;
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_OP_NAME);
  EXPECT_EQ(memcmp(tlv->data, "add", tlv->len), 0);

  // l2 tlv 1
  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_OP_TYPE);
  EXPECT_EQ(memcmp(tlv->data, "Add", tlv->len), 0);

  // l2 tlv 2
  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_ORI_OP_NAME);

  // l2 tlv 4
  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_INPUT_DESC);
  DbgInputDescTlv2 *input_tlv = (DbgInputDescTlv2 *)tlv->data;
  EXPECT_EQ(input_tlv->num, 2);
  DbgInputDescParamTlv2 *input_param = (DbgInputDescParamTlv2 *)input_tlv->param;
  EXPECT_EQ(input_param->data_type, DT_INT64);
  EXPECT_EQ(input_param->addr_type, toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);
  EXPECT_NE(input_param->addr, 0);
  EXPECT_EQ(input_param->size, 8);
  EXPECT_EQ(input_param->offset, 10);

  // l2 tlv 5
  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_OUTPUT_DESC);
  DbgOutputDescTlv2 *output_tlv = (DbgOutputDescTlv2 *)tlv->data;
  EXPECT_EQ(output_tlv->num, 1);
  DbgOutputDescParamTlv2 *output_param = (DbgOutputDescParamTlv2 *)output_tlv->param;
  EXPECT_EQ(output_param->data_type, DT_INT64);
  EXPECT_EQ(output_param->addr_type, toolkit::aicpu::dump::AddressType::NANO_IO_ADDR);
  EXPECT_NE(output_param->addr, 0);
  EXPECT_EQ(output_param->size, 1024);
  EXPECT_EQ(output_param->offset, 10);

  // l2 tlv 6
  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_WORKSPACE_DESC);
  DbgWorkspaceDescTlv2 *workspace_tlv = (DbgWorkspaceDescTlv2 *)tlv->data;
  EXPECT_EQ(workspace_tlv->num, 1);
  DbgWorkspaceDescParamTlv2 *workspace_param = (DbgWorkspaceDescParamTlv2 *)workspace_tlv->param;
  EXPECT_EQ(workspace_param->type, toolkit::aicpu::dump::L1);
  EXPECT_EQ(workspace_param->data_addr, 0);
  EXPECT_EQ(workspace_param->size, 32);

  // l2 tlv 7
  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_OP_BUF);
  DbgOpBufTlv2 *buffer_tlv = (DbgOpBufTlv2 *)tlv->data;
  EXPECT_EQ(buffer_tlv->num, 2);
  DbgOpBufParamTlv2 *buffer_param = (DbgOpBufParamTlv2 *)buffer_tlv->param;
  EXPECT_EQ(buffer_param->type, toolkit::aicpu::dump::Workspace::LOG);

  // l2 tlv 8
  tlv = (TlvHead *)(tlv->data + tlv->len);
  EXPECT_EQ(tlv->type, DBG_L2_TLV_TYPE_MEM_INFO);
  DbgOpMemInfosTlv2 *mem_tlv = (DbgOpMemInfosTlv2 *)tlv->data;
  EXPECT_EQ(mem_tlv->num, 1);
  DbgOpMemInfoTlv2 *mem_param = (DbgOpMemInfoTlv2 *)mem_tlv->param;
  EXPECT_EQ(mem_param->input_mem_size, 192);
  EXPECT_EQ(mem_param->output_mem_size, 128);
  EXPECT_EQ(mem_param->weight_mem_size, 64);
  EXPECT_EQ(mem_param->total_mem_size, 384);
}

TEST_F(UTEST_nano_dbg_data, empty_kernel) {
  GeModelPtr ge_model_empty = std::make_shared<GeModel>();

  auto data1 = OP_CFG(DATA).TensorDesc(FORMAT_ND, DT_INT64, {1});
  DEF_GRAPH(g1) {
    CHAIN(NODE("data1", data1)->EDGE(0, 0)->NODE("add", "Add")->EDGE(0, 0)->NODE("NetOutput1", "NetOutput"));
  };

  ComputeGraphPtr graph_tmp = ToComputeGraph(g1);
  ge_model_empty->SetGraph(graph_tmp);
  std::shared_ptr<domi::ModelTaskDef> model_task_tmp = std::make_shared<domi::ModelTaskDef>();
  ge_model_empty->SetModelTaskDef(model_task_tmp);

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_empty, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_nano_dbg_data, task_all_kernel_dbg_data) {
  domi::TaskDef *task_def = ge_model_->GetModelTaskDefPtr()->mutable_task(0);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def->set_stream_id(0);
  task_def->mutable_kernel_with_handle()->set_block_dim(10);
  task_def->mutable_kernel_with_handle()->mutable_context()->set_op_index(op_desc_->GetId());

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_nano_dbg_data, task_map_not_find) {
  domi::TaskDef *task_def = ge_model_->GetModelTaskDefPtr()->mutable_task(0);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_EVENT_RECORD));
  task_def->set_stream_id(0);
  task_def->mutable_kernel()->set_block_dim(10);
  task_def->mutable_kernel()->mutable_context()->set_op_index(op_desc_->GetId());

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
}

TEST_F(UTEST_nano_dbg_data, empty_tlv_list) {
  NanoDbgOpDesc dbg_empty_op = {};
  domi::TaskDef *task_def = ge_model_->GetModelTaskDefPtr()->mutable_task(0);
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def->set_stream_id(0);
  task_def->mutable_kernel()->set_block_dim(10);
  task_def->mutable_kernel()->mutable_context()->set_op_index(op_desc_->GetId());

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[1] = 1;
  NanoDbgData dbg(ge_model_, zerocopy_info);

  EXPECT_EQ(dbg.SaveDbgMemInfoTlv(dbg_empty_op.mem_info_list), ge::SUCCESS);
  EXPECT_EQ(dbg.SaveDbgBufTlv(dbg_empty_op.buffer_list), ge::SUCCESS);
  EXPECT_EQ(dbg.SaveDbgInputDescTlv(dbg_empty_op.input_list), ge::SUCCESS);
  EXPECT_EQ(dbg.SaveDbgOutputDescTlv(dbg_empty_op.output_list), ge::SUCCESS);
  EXPECT_EQ(dbg.SaveDbgWorkspaceDescTlv(dbg_empty_op.workspace_list), ge::SUCCESS);
  EXPECT_EQ(dbg.SaveDbgVecTlv(dbg_empty_op.original_op_names, DBG_L2_TLV_TYPE_ORI_OP_NAME), ge::SUCCESS);

  // 空vector<int>需要拷贝tlv头，未初始化dbg buf，拷贝内存不足报错
  auto shape = GeShape();
  EXPECT_NE(dbg.SaveDbgVecTlv(shape.GetDims(), DBG_INPUT_DESC_L3_TLV_TYPE_SHAPE_DIMS), ge::SUCCESS);
}

TEST_F(UTEST_nano_dbg_data, AddDbg) {
  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[0x10] = 0;
  zerocopy_info[0x12] = 8;
  NanoDbgData dbg(ge_model_, zerocopy_info);

  const auto &node = ge_model_->GetGraph()->FindNode("add");
  const auto &op_desc = node->GetOpDesc();
  vector<int64_t> inputoffset = {};
  inputoffset.push_back(0x11);
  inputoffset.push_back(0);
  inputoffset.push_back(0x10);
  vector<int64_t> outputoffset = {};
  outputoffset.push_back(0x8);
  outputoffset.push_back(0x13);
  op_desc->SetInputOffset(inputoffset);
  op_desc->SetOutputOffset(outputoffset);

  dbg.input_mem_types_[0].push_back(toolkit::aicpu::dump::AddressType::NANO_IO_ADDR);
  dbg.input_mem_types_[0].push_back(toolkit::aicpu::dump::AddressType::NANO_IO_ADDR);
  dbg.input_mem_types_[0].push_back(toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);
  dbg.input_mem_types_[1].push_back(toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);
  dbg.input_mem_types_[1].push_back(toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);
  dbg.input_mem_types_[1].push_back(toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);

  dbg.output_mem_types_[0].push_back(toolkit::aicpu::dump::AddressType::NANO_IO_ADDR);
  dbg.output_mem_types_[0].push_back(toolkit::aicpu::dump::AddressType::NANO_IO_ADDR);
  dbg.output_mem_types_[1].push_back(toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);
  dbg.output_mem_types_[1].push_back(toolkit::aicpu::dump::AddressType::NANO_WEIGHT_ADDR);

  NanoDbgOpDesc dbg_op = {};

  EXPECT_EQ(dbg.AddDbgInput(op_desc, dbg_op, 1), ge::SUCCESS);
  EXPECT_EQ(dbg.AddDbgInput(op_desc, dbg_op, 0), ge::SUCCESS);
  EXPECT_EQ(dbg.AddDbgOutput(op_desc, dbg_op, 1), ge::SUCCESS);
  EXPECT_EQ(dbg.AddDbgOutput(op_desc, dbg_op, 0), ge::SUCCESS);
}

///
///  net_output
///     |
///    add1  add2
///     |    /
///     var1
///
ComputeGraphPtr BuildSuspendedGraph() {
  auto builder = ut::GraphBuilder("g1");
  auto var1 = builder.AddNode("var1", VARIABLEV2, 0, 2);
  auto add1 = builder.AddNode("add1", ADD, 2, 1);
  auto add2 = builder.AddNode("add2", ADD, 2, 1);
  auto net_output = builder.AddNode("net_output", NETOUTPUT, 1, 0);

  builder.AddDataEdge(var1, 0, add1, 0);
  builder.AddDataEdge(var1, 1, add2, 0);
  builder.AddDataEdge(add1, 0, net_output, 0);
  return builder.GetGraph();
}

TEST_F(UTEST_nano_dbg_data, DbgSuspended) {
  auto graph = BuildSuspendedGraph();
  auto add1_op = graph->FindNode("add1")->GetOpDesc();
  auto add2_op = graph->FindNode("add2")->GetOpDesc();
  auto data_op = graph->FindNode("var1")->GetOpDesc();
  EXPECT_NE(add1_op, nullptr);
  EXPECT_NE(add2_op, nullptr);
  EXPECT_NE(data_op, nullptr);

  std::vector<uint64_t> output_offset{10};
  std::vector<uint64_t> input_offset{10, 10};

  (void)add1_op->SetExtAttr("task_addr_offset", output_offset);
  add1_op->SetOutputOffset({2048});
  auto output_tensor = add1_op->MutableOutputDesc(0);
  output_tensor->SetShape(GeShape({1,4,4,8}));
  output_tensor->SetOriginShape(GeShape());
  output_tensor->SetDataType(DT_INT64);
  TensorUtils::SetSize(*output_tensor, 64);

  add1_op->SetInputOffset({2048, 2048});
  (void)add1_op->SetExtAttr("task_addr_offset", input_offset);
  for (int i = 0; i < 2; i++) {
    auto input_desc = add1_op->MutableInputDesc(i);
    int64_t input_size = 8;
    AttrUtils::SetInt(input_desc, ATTR_NAME_INPUT_ORIGIN_SIZE, input_size);
    input_desc->SetShape(GeShape());
    input_desc->SetDataType(DT_INT64);
    TensorUtils::SetSize(*input_desc, 64);
  }

  (void)add2_op->SetExtAttr("task_addr_offset", output_offset);
  add2_op->SetOutputOffset({2048});
  auto output2_tensor = add2_op->MutableOutputDesc(0);
  output2_tensor->SetShape(GeShape({1,4,4,8}));
  output2_tensor->SetOriginShape(GeShape());
  output2_tensor->SetDataType(DT_INT64);
  TensorUtils::SetSize(*output2_tensor, 64);

  add2_op->SetInputOffset({2048, 2048});
  (void)add2_op->SetExtAttr("task_addr_offset", input_offset);
  for (int i = 0; i < 2; i++) {
    auto input_desc = add2_op->MutableInputDesc(i);
    int64_t input_size = 8;
    AttrUtils::SetInt(input_desc, ATTR_NAME_INPUT_ORIGIN_SIZE, input_size);
    input_desc->SetShape(GeShape());
    input_desc->SetDataType(DT_INT64);
    TensorUtils::SetSize(*input_desc, 64);
  }


  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);

  std::shared_ptr<domi::ModelTaskDef> model_task = std::make_shared<domi::ModelTaskDef>();
  //add two task for add1&add2
  domi::TaskDef *task1 = model_task->add_task();
  domi::TaskDef *task2 = model_task->add_task();
  task1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task1->mutable_kernel_with_handle()->set_block_dim(10);
  task1->mutable_kernel_with_handle()->mutable_context()->set_op_index(add1_op->GetId());
  task2->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task2->mutable_kernel_with_handle()->set_block_dim(10);
  task2->mutable_kernel_with_handle()->mutable_context()->set_op_index(add2_op->GetId());

  ge_model->SetModelTaskDef(model_task);

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[0x10] = 0;
  zerocopy_info[0x12] = 8;
  NanoDbgData dbg(ge_model, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
}

///
///  net_output  net_output
///     |        /
///      splitv
///        |
///       data
///
ComputeGraphPtr BuildOneInputMultiOutputGraph() {
  auto builder = ut::GraphBuilder("g1");
  auto data = builder.AddNode("data", DATA, 0, 1);
  auto splitv = builder.AddNode("splitv", SPLITV, 1, 2);
  auto net_output0 = builder.AddNode("net_output0", NETOUTPUT, 1, 0);
  auto net_output1 = builder.AddNode("net_output1", NETOUTPUT, 1, 0);

  builder.AddDataEdge(data, 0, splitv, 0);
  builder.AddDataEdge(splitv, 0, net_output0, 0);
  builder.AddDataEdge(splitv, 1, net_output1, 0);
  return builder.GetGraph();
}

TEST_F(UTEST_nano_dbg_data, DbgOneInputMultiOutput) {
  auto graph = BuildOneInputMultiOutputGraph();
  auto splitv = graph->FindNode("splitv")->GetOpDesc();
  EXPECT_NE(splitv, nullptr);

  std::vector<uint64_t> task_addr_offset{10, 10};

  (void)splitv->SetExtAttr("task_addr_offset", task_addr_offset);
  splitv->SetInputOffset({2048});
  splitv->SetOutputOffset({2048, 2048});

  auto input_desc = splitv->MutableInputDesc(0);
  int64_t input_size = 8;
  AttrUtils::SetInt(input_desc, ATTR_NAME_INPUT_ORIGIN_SIZE, input_size);
  input_desc->SetShape(GeShape());
  input_desc->SetDataType(DT_INT64);
  TensorUtils::SetSize(*input_desc, 64);

  for (int i = 0; i < 2; i++) {
    auto output_desc = splitv->MutableOutputDesc(i);
    output_desc->SetShape(GeShape());
    output_desc->SetOriginShape(GeShape());
    output_desc->SetDataType(DT_INT64);
    TensorUtils::SetSize(*output_desc, 64);
  }

  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetGraph(graph);

  std::shared_ptr<domi::ModelTaskDef> model_task = std::make_shared<domi::ModelTaskDef>();
  domi::TaskDef *task1 = model_task->add_task();
  task1->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task1->mutable_kernel_with_handle()->set_block_dim(10);
  task1->mutable_kernel_with_handle()->mutable_context()->set_op_index(splitv->GetId());
  ge_model->SetModelTaskDef(model_task);

  std::unordered_map<int64_t, uint32_t> zerocopy_info;
  zerocopy_info[0x10] = 0;
  zerocopy_info[0x12] = 8;
  NanoDbgData dbg(ge_model, zerocopy_info);
  auto ret = dbg.Init();
  EXPECT_EQ(ret, ge::SUCCESS);
}
}  // namespace ge
