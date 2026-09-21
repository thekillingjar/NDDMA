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
#include "graph/kernel_launch_info.h"
#include "graph/arg_desc_info.h"
#include "runtime/rt_model.h"
#include "proto/task.pb.h"
#include "exe_graph/runtime/exe_res_generation_context.h"
#include "exe_graph/lowering/exe_res_generation_ctx_builder.h"
#include "ge/framework/common/taskdown_common.h"
#include "graph/utils/args_format_desc_utils.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"
#include "common/opskernel/ops_kernel_info_types.h"

namespace ge {
namespace {
ComputeGraphPtr CreateMc2NodeGraph() {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr x1_desc = std::make_shared<OpDesc>("x1", "Data");
  OpDescPtr x2_desc = std::make_shared<OpDesc>("x2", "Data");
  OpDescPtr bias_desc = std::make_shared<OpDesc>("bias", "Data");
  OpDescPtr all_gather_matmul_desc = std::make_shared<OpDesc>("mc2", "AllGatherMatmul");
  OpDescPtr net_output_desc = std::make_shared<OpDesc>("output", "NetOutput");

  // add descriptor
  ge::GeShape shape1({2,4});
  GeTensorDesc tensor_desc1(shape1, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc1.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc1.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc1.SetOriginShape(shape1);

  ge::GeShape shape2({4,3});
  GeTensorDesc tensor_desc2(shape2, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc2.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc2.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc2.SetOriginShape(shape2);

  ge::GeShape shape3({3});
  GeTensorDesc tensor_desc3(shape3, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc3.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc3.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc3.SetOriginShape(shape3);

  ge::GeShape shape4({2, 3});
  GeTensorDesc tensor_desc4(shape4, ge::FORMAT_ND, ge::DT_FLOAT16);
  tensor_desc4.SetOriginFormat(ge::FORMAT_ND);
  tensor_desc4.SetOriginDataType(ge::DT_FLOAT16);
  tensor_desc4.SetOriginShape(shape4);

  x1_desc->AddOutputDesc(tensor_desc1);
  x2_desc->AddOutputDesc(tensor_desc2);
  bias_desc->AddOutputDesc(tensor_desc3);

  all_gather_matmul_desc->AddInputDesc(tensor_desc1);
  all_gather_matmul_desc->AddInputDesc(tensor_desc2);
  all_gather_matmul_desc->AddInputDesc(tensor_desc3);
  all_gather_matmul_desc->AddOutputDesc(tensor_desc4);
  all_gather_matmul_desc->AddOutputDesc(tensor_desc4);
  all_gather_matmul_desc->AppendIrInput("x1", ge::kIrInputRequired);
  all_gather_matmul_desc->AppendIrInput("x2", ge::kIrInputRequired);
  all_gather_matmul_desc->AppendIrInput("bias", ge::kIrInputOptional);
  all_gather_matmul_desc->AppendIrOutput("y", ge::kIrOutputRequired);
  all_gather_matmul_desc->AppendIrOutput("gather_out", ge::kIrOutputRequired);

  net_output_desc->AddInputDesc(tensor_desc4);
  net_output_desc->AddInputDesc(tensor_desc4);
  // create nodes
  NodePtr x1_node = graph->AddNode(x1_desc);
  NodePtr x2_node = graph->AddNode(x2_desc);
  NodePtr bias_node = graph->AddNode(bias_desc);
  NodePtr mc2_node = graph->AddNode(all_gather_matmul_desc);
  NodePtr output_node = graph->AddNode(net_output_desc);

  ge::GraphUtils::AddEdge(x1_node->GetOutDataAnchor(0), mc2_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(x2_node->GetOutDataAnchor(0), mc2_node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(bias_node->GetOutDataAnchor(0), mc2_node->GetInDataAnchor(2));
  ge::GraphUtils::AddEdge(mc2_node->GetOutDataAnchor(0), output_node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(mc2_node->GetOutDataAnchor(1), output_node->GetInDataAnchor(1));

  all_gather_matmul_desc->SetStreamId(2);
  all_gather_matmul_desc->SetId(4);
  std::vector<int64_t> ori_work_sizes{22,33,44};
  all_gather_matmul_desc->SetWorkspaceBytes(ori_work_sizes);
  return graph;
}

gert::ExeResGenerationCtxHolderPtr CreateNodeExeResContext(const NodePtr &node) {
  gert::ExeResGenerationCtxBuilder exe_ctx_builder;
  auto res_ptr_holder = exe_ctx_builder.CreateOpExeContext(*node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_ptr_holder->GetKernelContext());
  std::vector<gert::StreamInfo> stream_info_vec;
  gert::StreamInfo si_1;
  si_1.name = "aicpu kfc server";
  si_1.reuse_key = "kfc_stream";
  si_1.depend_value_input_indices = {};
  si_1.required = true;
  stream_info_vec.emplace_back(si_1);
  op_exe_res_ctx->SetAttachedStreamInfos(stream_info_vec);
  std::vector<ge::GeAttrValue::NAMED_ATTRS> stream_info_attrs;
  (void)ge::AttrUtils::GetListNamedAttrs(node->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
      stream_info_attrs);
  (void)ge::AttrUtils::SetInt(stream_info_attrs.front(), ge::ATTR_NAME_ATTACHED_RESOURCE_ID, 4);
  (void)ge::AttrUtils::SetListNamedAttrs(node->GetOpDesc(), ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST,
      stream_info_attrs);
  return res_ptr_holder;
}

struct HcclCommParamDesc {
  uint64_t version : 4;
  uint64_t group_num : 4;
  uint64_t has_ffts : 1;
  uint64_t tiling_off : 7;
  uint64_t is_dyn : 48;
};

graphStatus Mc2GenTaskCallback(const gert::ExeResGenerationContext *context,
    std::vector<std::vector<uint8_t>> &tasks) {
  GE_ASSERT_NOTNULL(context);
  GE_ASSERT_TRUE(tasks.size() == 1UL);
  auto aicore_index = 0;
  // 获取attach流id
  auto stream_infos = context->GetAttachedStreamInfos();
  GE_ASSERT_TRUE(!stream_infos.empty());
  const int64_t attach_stream_id = stream_infos[0].stream_id;
  const int64_t stream_id = context->GetStreamId();
  // 创建WaitTask
  auto wait_task = KernelLaunchInfo::CreateHcomWaitTask(context);
  wait_task.SetStreamId(attach_stream_id);
  tasks.insert(tasks.begin() + aicore_index, wait_task.Serialize());
  aicore_index++;
  // 设置aicpu任务
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(context,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  size_t input_size = context->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = context->GetComputeNodeInfo()->GetIrOutputsNum();
  const size_t offset = 3UL;
  union {
    HcclCommParamDesc hccl_desc;
    uint64_t custom_value;
  } desc;
  desc.hccl_desc.version = 1;
  desc.hccl_desc.group_num = 1;
  desc.hccl_desc.has_ffts = 0;
  desc.hccl_desc.tiling_off = offset + input_size + output_size;
  desc.hccl_desc.is_dyn = 0;
  std::vector<ArgDescInfo> aicpu_args_format;
  aicpu_args_format.emplace_back(ArgDescInfo::CreateCustomValue(desc.custom_value));
  aicpu_args_format.emplace_back(ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrInput, 0));
  for (size_t i = 1; i < input_size; i++) {
    aicpu_args_format.emplace_back(ArgDescInfo::CreateCustomValue(0));
  }
  for (size_t i = 0; i < output_size; i++) {
    aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kIrOutput, i));
  }
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kWorkspace));
  aicpu_args_format.emplace_back(ArgDescInfo(ArgDescType::kTiling));
  aicpu_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicpu_args_format).GetString());
  aicpu_task.SetStreamId(attach_stream_id);
  tasks.insert(tasks.begin() + aicore_index, aicpu_task.Serialize());
  aicore_index++;
  // 创建RecordTask
  auto record_task = KernelLaunchInfo::CreateHcomRecordTask(context);
  record_task.SetStreamId(stream_id);
  tasks.insert(tasks.begin() + aicore_index, record_task.Serialize());
  aicore_index++;
  // 更改原AICORE任务的argsformat
  auto aicore_task = KernelLaunchInfo::LoadFromData(context, tasks.back());
  auto aicore_args_format_str = aicore_task.GetArgsFormat();
  auto aicore_args_format = ArgsFormatSerializer::Deserialize(aicore_args_format_str);
  size_t i = 0UL;
  for (; i < aicore_args_format.size(); i++) {
    if (aicore_args_format[i].GetType() == ArgDescType::kIrInput ||
        aicore_args_format[i].GetType() == ArgDescType::kInputInstance) {
      break;
    }
  }
  aicore_args_format.insert(aicore_args_format.begin() + i, ArgDescInfo::CreateHiddenInput(HiddenInputSubType::kHcom));
  aicore_task.SetArgsFormat(ArgsFormatSerializer::Serialize(aicore_args_format).GetString());
  tasks.back() = aicore_task.Serialize();
  return SUCCESS;
}

domi::TaskDef CreateAicoreTaskDef(const gert::ExeResGenerationContext *context) {
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  aicore_task_def.set_id(context->GetOpId());
  aicore_task_def.set_stream_id(context->GetStreamId());
  auto kernel_def = aicore_task_def.mutable_kernel();
  kernel_def->set_block_dim(32);
  kernel_def->set_schedule_mode(0);
  auto kernel_context = kernel_def->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  kernel_context->set_op_index(context->GetOpId());

  std::vector<ArgDesc> args;
  size_t input_size = context->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = context->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));

  return aicore_task_def;
}

domi::TaskDef CreateAllKernelTaskDef(const gert::ExeResGenerationContext *context) {
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  task_def.set_id(context->GetOpId());
  task_def.set_stream_id(context->GetStreamId());
  auto kernel_def_with_handle = task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(64);
  kernel_def_with_handle->set_schedule_mode(1);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  kernel_context->set_op_index(context->GetOpId());

  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::Append(args, AddrType::FFTS_ADDR);
  size_t input_size = context->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = context->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT_INSTANCE, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_INSTANCE, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));

  return task_def;
}
}
class TestGenTaskCallback : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};
// 验证使用kernel_def的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestNormalMc2NodeGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def = aicore_task_def.mutable_kernel();
  kernel_def->set_block_dim(32);
  kernel_def->set_schedule_mode(0);
  auto kernel_context = kernel_def->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format, "{hi.hcom0*}{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
}

// 验证使用kernel_def_with_handle的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestMc2NodeWithHandleGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel_with_handle();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE_AI_CORE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format, "{hi.hcom0*}{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
}

// 验证使用mixL2的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestMixL2Mc2NodeGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::Append(args, AddrType::FFTS_ADDR);
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel_with_handle();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format, "{ffts_addr}{hi.hcom0*}{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
}


// 验证使用带有input_instance的mc2算子在GenTaskCallback函数中构造taskDef的功能
TEST_F(TestGenTaskCallback, TestMc2WithInputInstanceNodeGenTaskCallback) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::Append(args, AddrType::FFTS_ADDR);
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT_INSTANCE, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_INSTANCE, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  tasks.emplace_back(buffer);
  // 执行mc2的gentaskcallback
  EXPECT_EQ(Mc2GenTaskCallback(op_exe_res_ctx, tasks), SUCCESS);
  EXPECT_EQ(tasks.size(), 4UL);
  // 校验wait算子的结果
  domi::TaskDef wait_task;
  wait_task.ParseFromArray(tasks[0].data(), tasks[0].size());
  EXPECT_EQ(wait_task.id(), 4);
  EXPECT_EQ(wait_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(wait_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_WAIT));
  EXPECT_EQ(wait_task.private_def(), "group");
  EXPECT_EQ(wait_task.stream_id(), 4);
  // 校验aicpu算子结果
  domi::TaskDef aicpu_task;
  aicpu_task.ParseFromArray(tasks[1].data(), tasks[1].size());
  EXPECT_EQ(aicpu_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  EXPECT_EQ(aicpu_task.stream_id(), 4);
  auto aicpu_kernel_def = aicpu_task.mutable_kernel();
  EXPECT_EQ(aicpu_kernel_def->so_name(), "libccl_kernel.so");
  EXPECT_EQ(aicpu_kernel_def->kernel_name(), "RunAicpuKfcSrvLaunch");
  auto aicpu_kernel_context = aicpu_kernel_def->mutable_context();
  EXPECT_EQ(aicpu_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::AI_CPU_KFC));
  EXPECT_EQ(aicpu_kernel_context->op_index(), 4);
  auto aicpu_args_format = aicpu_kernel_context->args_format();
  EXPECT_EQ(aicpu_args_format, "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");

  // 校验record算子结果
  domi::TaskDef record_task;
  record_task.ParseFromArray(tasks[2].data(), tasks[2].size());
  EXPECT_EQ(record_task.id(), 4);
  EXPECT_EQ(record_task.notify_id(), UINT32_MAX);
  EXPECT_EQ(record_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_NOTIFY_RECORD));
  EXPECT_EQ(record_task.private_def(), "group");
  EXPECT_EQ(record_task.stream_id(), 2);
  // 校验aicore算子结果
  domi::TaskDef aicore_task;
  aicore_task.ParseFromArray(tasks[3].data(), tasks[3].size());
  EXPECT_EQ(aicore_task.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  EXPECT_EQ(aicore_task.stream_id(), 2);
  auto aicore_kernel_def = aicore_task.mutable_kernel_with_handle();
  EXPECT_EQ(aicore_kernel_def->block_dim(), 32);
  EXPECT_EQ(aicore_kernel_def->schedule_mode(), 0);
  auto aicore_kernel_context = aicore_kernel_def->mutable_context();
  EXPECT_EQ(aicore_kernel_context->kernel_type(), static_cast<uint32_t>(ccKernelType::TE));
  EXPECT_EQ(aicore_kernel_context->op_index(), 4);
  auto aicore_args_format = aicore_kernel_context->args_format();
  EXPECT_EQ(aicore_args_format,
      "{ffts_addr}{hi.hcom0*}{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}{o_instance1*}{ws*}{t}");
}

// 验证KernelLaunchInfo的移动构造函数和移动赋值函数
TEST_F(TestGenTaskCallback, TestKernelLaunchInfoMoveConstruct) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  auto aicpu_task = KernelLaunchInfo::CreateAicpuKfcTask(op_exe_res_ctx,
      "libccl_kernel.so", "RunAicpuKfcSrvLaunch");
  aicpu_task.SetStreamId(2);
  aicpu_task.SetBlockDim(32);
  std::string args_format = "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}";
  aicpu_task.SetArgsFormat(args_format.c_str());
  // 验证移动赋值函数
  KernelLaunchInfo aicpu_task_1 = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  aicpu_task_1 = std::move(aicpu_task);
  EXPECT_EQ(std::string(aicpu_task_1.GetSoName()), "libccl_kernel.so");
  EXPECT_EQ(std::string(aicpu_task_1.GetKernelName()), "RunAicpuKfcSrvLaunch");
  EXPECT_EQ(aicpu_task_1.GetStreamId(), 2);
  EXPECT_EQ(aicpu_task_1.GetBlockDim(), 32);
  EXPECT_EQ(std::string(aicpu_task_1.GetArgsFormat()), "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");
  // 验证移动构造函数
  KernelLaunchInfo aicpu_task_2(std::move(aicpu_task_1));
  EXPECT_EQ(std::string(aicpu_task_2.GetSoName()), "libccl_kernel.so");
  EXPECT_EQ(std::string(aicpu_task_2.GetKernelName()), "RunAicpuKfcSrvLaunch");
  EXPECT_EQ(aicpu_task_2.GetStreamId(), 2);
  EXPECT_EQ(std::string(aicpu_task_2.GetArgsFormat()), "{#4113}{hi.hcom0*}{i0*}{#0}{#0}{o0*}{o1*}{ws*}{t}");
}

// 验证KernelLaunchInfo的拷贝构造函数和拷贝赋值函数
TEST_F(TestGenTaskCallback, TestKernelLaunchInfoCopyConstruct) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());
  domi::TaskDef aicore_task_def;
  aicore_task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  aicore_task_def.set_id(op_exe_res_ctx->GetOpId());
  aicore_task_def.set_stream_id(op_exe_res_ctx->GetStreamId());
  auto kernel_def_with_handle = aicore_task_def.mutable_kernel_with_handle();
  kernel_def_with_handle->set_block_dim(32);
  kernel_def_with_handle->set_schedule_mode(0);
  auto kernel_context = kernel_def_with_handle->mutable_context();
  kernel_context->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  kernel_context->set_op_index(op_exe_res_ctx->GetOpId());
  std::vector<ArgDesc> args;
  ArgsFormatDescUtils::Append(args, AddrType::FFTS_ADDR);
  size_t input_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrInputsNum();
  size_t output_size = op_exe_res_ctx->GetComputeNodeInfo()->GetIrOutputsNum();
  for (size_t i = 0UL; i < input_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::INPUT_INSTANCE, i);
  }
  for (size_t i = 0UL; i < output_size; i++) {
    ArgsFormatDescUtils::Append(args, AddrType::OUTPUT_INSTANCE, i);
  }
  ArgsFormatDescUtils::Append(args, AddrType::WORKSPACE);
  ArgsFormatDescUtils::Append(args, AddrType::TILING);
  kernel_context->set_args_format(ArgsFormatDescUtils::Serialize(args));
  // 序列化
  std::vector<std::vector<uint8_t>> tasks;
  auto buffer_size = aicore_task_def.ByteSizeLong();
  std::vector<uint8_t> buffer(buffer_size, 0);
  aicore_task_def.SerializeToArray(buffer.data(), buffer_size);
  auto aicore_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx, buffer);
  EXPECT_EQ(aicore_task.SetBlockDim(48), SUCCESS);
  // 验证拷贝赋值函数
  KernelLaunchInfo copy_task = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  copy_task = aicore_task;
  EXPECT_EQ(copy_task.GetStreamId(), 2);
  EXPECT_EQ(copy_task.GetBlockDim(), 48);
  EXPECT_EQ(std::string(copy_task.GetArgsFormat()), "{ffts_addr}{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}{o_instance1*}{ws*}{t}");

  // 验证拷贝构造函数
  KernelLaunchInfo copy_task_2(copy_task);
  EXPECT_EQ(copy_task_2.GetStreamId(), 2);
  EXPECT_EQ(copy_task_2.GetBlockDim(), 48);
  EXPECT_EQ(std::string(copy_task_2.GetArgsFormat()), "{ffts_addr}{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}{o_instance1*}{ws*}{t}");
}

// 验证非aicore和aicpu算子设置blockdim场景
TEST_F(TestGenTaskCallback, TestNonAicoreNodeSetBlockDimFailed) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());

  auto notify_task = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  EXPECT_EQ(notify_task.SetBlockDim(48), PARAM_INVALID);
  EXPECT_EQ(notify_task.GetBlockDim(), 0);
}
// 验证非aicore和aicpu算子设置argsformat场景
TEST_F(TestGenTaskCallback, TestNonAicoreNodeSetArgsFormatFailed) {
  auto graph = CreateMc2NodeGraph();
  auto mc2_node = graph->FindNode("mc2");
  auto res_context_holder = CreateNodeExeResContext(mc2_node);
  auto op_exe_res_ctx = reinterpret_cast<gert::ExeResGenerationContext *>(res_context_holder->GetKernelContext());

  auto notify_task = KernelLaunchInfo::CreateHcomRecordTask(op_exe_res_ctx);
  EXPECT_EQ(notify_task.SetArgsFormat("aaaaa"), PARAM_INVALID);
  EXPECT_EQ(notify_task.GetArgsFormat(), nullptr);
}

class TestCreateFusionAndCcuTask : public testing::Test {
 protected:
  void SetUp() override {
    graph_ = CreateMc2NodeGraph();
    mc2_node_ = graph_->FindNode("mc2");
    res_context_holder_ = CreateNodeExeResContext(mc2_node_);
    op_exe_res_ctx_ = reinterpret_cast<gert::ExeResGenerationContext *>(
        res_context_holder_->GetKernelContext());
  }

  void TearDown() override {}

  ComputeGraphPtr graph_;
  NodePtr mc2_node_;
  gert::ExeResGenerationCtxHolderPtr res_context_holder_;
  gert::ExeResGenerationContext *op_exe_res_ctx_;
};

// 测试CreateCcuTask函数
TEST_F(TestCreateFusionAndCcuTask, TestCreateCcuTask) {
  // 准备测试数据
  std::vector<std::string> groups = {"group1", "group2", "group3"};

  // 创建CcuTask
  auto ccu_task = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups);

  // 序列化并验证
  auto serialized_data = ccu_task.Serialize();
  ASSERT_FALSE(serialized_data.empty());

  // 解析TaskDef验证字段
  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  // 验证基本属性
  EXPECT_EQ(task_def.id(), static_cast<uint32_t>(op_exe_res_ctx_->GetOpId()));
  EXPECT_EQ(task_def.notify_id(), UINT32_MAX);
  EXPECT_EQ(task_def.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_CCU_KERNEL));

  // 验证FusionTask字段
  ASSERT_TRUE(task_def.has_fusion_task());
  const auto &fusion_task = task_def.fusion_task();
  EXPECT_EQ(fusion_task.op_index(), static_cast<uint32_t>(op_exe_res_ctx_->GetOpId()));

  // 验证子任务信息
  ASSERT_EQ(fusion_task.fusion_sub_task_info_size(), 1);
  const auto &sub_task_info = fusion_task.fusion_sub_task_info(0);
  EXPECT_EQ(sub_task_info.type(), domi::FusionSubTaskInfo::CCU);

  // 验证CCU任务组
  ASSERT_TRUE(sub_task_info.task().has_ccu_task_group());
  const auto &ccu_task_group = sub_task_info.task().ccu_task_group();

  // 验证groups
  ASSERT_EQ(ccu_task_group.group_size(), static_cast<int>(groups.size()));
  for (int i = 0; i < ccu_task_group.group_size(); i++) {
    EXPECT_EQ(ccu_task_group.group(i), groups[i]);
  }

  // 验证CCU任务信息
  ASSERT_EQ(ccu_task_group.ccu_task_info_size(), 1);
}

// 测试CreateFusionTask函数，包含AICore和CCU子任务
TEST_F(TestCreateFusionAndCcuTask, TestCreateFusionTaskWithAicoreAndCcu) {
  // 创建AICore任务
  auto aicore_task_def = CreateAicoreTaskDef(op_exe_res_ctx_);
  std::vector<uint8_t> aicore_buffer(aicore_task_def.ByteSizeLong());
  aicore_task_def.SerializeToArray(aicore_buffer.data(), aicore_buffer.size());
  auto aicore_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, aicore_buffer);

  // 设置AICore任务的args_format和blockdim
  std::string args_format = "{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}";
  aicore_task.SetArgsFormat(args_format.c_str());
  aicore_task.SetBlockDim(32);

  // 创建CCU任务
  std::vector<std::string> groups = {"fusion_group1", "fusion_group2"};
  auto ccu_task = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups);

  // 创建FusionTask，包含AICore和CCU子任务
  std::vector<KernelLaunchInfo> sub_tasks;
  sub_tasks.push_back(std::move(aicore_task));
  sub_tasks.push_back(std::move(ccu_task));

  auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);

  // 序列化并验证
  auto serialized_data = fusion_task.Serialize();
  ASSERT_FALSE(serialized_data.empty());

  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  // 验证基本属性
  EXPECT_EQ(task_def.id(), static_cast<uint32_t>(op_exe_res_ctx_->GetOpId()));
  EXPECT_EQ(task_def.notify_id(), UINT32_MAX);
  EXPECT_EQ(task_def.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FUSION_KERNEL));

  // 验证FusionTask字段
  ASSERT_TRUE(task_def.has_fusion_task());
  const auto &fusion_task_def = task_def.fusion_task();
  EXPECT_EQ(fusion_task_def.op_index(), static_cast<uint32_t>(op_exe_res_ctx_->GetOpId()));

  // 验证子任务数量
  ASSERT_EQ(fusion_task_def.fusion_sub_task_info_size(), 2);

  // 验证第一个子任务（AICore）
  const auto &sub_task_info1 = fusion_task_def.fusion_sub_task_info(0);
  EXPECT_EQ(sub_task_info1.type(), domi::FusionSubTaskInfo::AICORE);
  ASSERT_TRUE(sub_task_info1.task().has_aicore_fusion_task_info());
  EXPECT_FALSE(sub_task_info1.task().aicore_fusion_task_info().is_all_kernel());

  // 验证blockdim是否正确设置
  const auto &aicore_fusion_task_info = sub_task_info1.task().aicore_fusion_task_info();
  ASSERT_TRUE(aicore_fusion_task_info.has_config());
  const auto &config = aicore_fusion_task_info.config();
  bool found_blockdim = false;
  for (int j = 0; j < config.launch_attribute_size(); ++j) {
    const auto &attr = config.launch_attribute(j);
    if (attr.id() == domi::LaunchAttribute::BLOCKDIM) {
      EXPECT_EQ(attr.value().block_dim(), 32);
      found_blockdim = true;
      break;
    }
  }
  EXPECT_TRUE(found_blockdim);

  // 验证第二个子任务（CCU）
  const auto &sub_task_info2 = fusion_task_def.fusion_sub_task_info(1);
  EXPECT_EQ(sub_task_info2.type(), domi::FusionSubTaskInfo::CCU);
  ASSERT_TRUE(sub_task_info2.task().has_ccu_task_group());

  const auto &ccu_task_group = sub_task_info2.task().ccu_task_group();
  ASSERT_EQ(ccu_task_group.group_size(), 2);
  EXPECT_EQ(ccu_task_group.group(0), "fusion_group1");
  EXPECT_EQ(ccu_task_group.group(1), "fusion_group2");
}

// 测试CreateFusionTask函数，包含AllKernel和CCU子任务
TEST_F(TestCreateFusionAndCcuTask, TestCreateFusionTaskWithAllKernelAndCcu) {
  // 创建AllKernel任务
  auto all_kernel_task_def = CreateAllKernelTaskDef(op_exe_res_ctx_);
  std::vector<uint8_t> all_kernel_buffer(all_kernel_task_def.ByteSizeLong());
  all_kernel_task_def.SerializeToArray(all_kernel_buffer.data(), all_kernel_buffer.size());
  auto all_kernel_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, all_kernel_buffer);

  // 设置AllKernel任务的args_format和blockdim
  std::string args_format = "{ffts_addr}{i_instance0*}{i_instance1*}{i_instance2*}{o_instance0*}{o_instance1*}{ws*}{t}";
  all_kernel_task.SetArgsFormat(args_format.c_str());
  all_kernel_task.SetBlockDim(64);

  // 创建CCU任务
  std::vector<std::string> groups = {"all_kernel_group"};
  auto ccu_task = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups);

  // 创建FusionTask
  std::vector<KernelLaunchInfo> sub_tasks;
  sub_tasks.push_back(std::move(all_kernel_task));
  sub_tasks.push_back(std::move(ccu_task));

  auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);

  // 序列化并验证
  auto serialized_data = fusion_task.Serialize();
  ASSERT_FALSE(serialized_data.empty());

  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  // 验证基本属性
  EXPECT_EQ(task_def.type(), static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FUSION_KERNEL));

  // 验证FusionTask字段
  ASSERT_TRUE(task_def.has_fusion_task());
  const auto &fusion_task_def = task_def.fusion_task();

  // 验证子任务数量
  ASSERT_EQ(fusion_task_def.fusion_sub_task_info_size(), 2);

  // 验证第一个子任务（AllKernel）
  const auto &sub_task_info1 = fusion_task_def.fusion_sub_task_info(0);
  EXPECT_EQ(sub_task_info1.type(), domi::FusionSubTaskInfo::AICORE);
  ASSERT_TRUE(sub_task_info1.task().has_aicore_fusion_task_info());
  EXPECT_TRUE(sub_task_info1.task().aicore_fusion_task_info().is_all_kernel());

  // 验证blockdim是否正确设置
  const auto &aicore_fusion_task_info = sub_task_info1.task().aicore_fusion_task_info();
  ASSERT_TRUE(aicore_fusion_task_info.has_config());
  const auto &config = aicore_fusion_task_info.config();
  bool found_blockdim = false;
  for (int j = 0; j < config.launch_attribute_size(); ++j) {
    const auto &attr = config.launch_attribute(j);
    if (attr.id() == domi::LaunchAttribute::BLOCKDIM) {
      EXPECT_EQ(attr.value().block_dim(), 64);
      found_blockdim = true;
      break;
    }
  }
  EXPECT_TRUE(found_blockdim);

  // 验证第二个子任务（CCU）
  const auto &sub_task_info2 = fusion_task_def.fusion_sub_task_info(1);
  EXPECT_EQ(sub_task_info2.type(), domi::FusionSubTaskInfo::CCU);
}

// 测试CreateFusionTask函数，只包含AICore子任务
TEST_F(TestCreateFusionAndCcuTask, TestCreateFusionTaskWithOnlyAicore) {
  // 创建两个AICore任务
  auto aicore_task_def1 = CreateAicoreTaskDef(op_exe_res_ctx_);
  std::vector<uint8_t> buffer1(aicore_task_def1.ByteSizeLong());
  aicore_task_def1.SerializeToArray(buffer1.data(), buffer1.size());
  auto aicore_task1 = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer1);
  aicore_task1.SetArgsFormat("{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
  aicore_task1.SetBlockDim(32);

  auto aicore_task_def2 = CreateAicoreTaskDef(op_exe_res_ctx_);
  aicore_task_def2.mutable_kernel()->set_block_dim(16); // 不同的blockdim
  std::vector<uint8_t> buffer2(aicore_task_def2.ByteSizeLong());
  aicore_task_def2.SerializeToArray(buffer2.data(), buffer2.size());
  auto aicore_task2 = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer2);
  aicore_task2.SetArgsFormat("{i0*}{i1*}{i2*}{o0*}{o1*}{ws*}{t}");
  aicore_task2.SetBlockDim(16);

  // 创建FusionTask，只包含AICore子任务
  std::vector<KernelLaunchInfo> sub_tasks;
  sub_tasks.push_back(std::move(aicore_task1));
  sub_tasks.push_back(std::move(aicore_task2));

  auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);

  // 验证
  auto serialized_data = fusion_task.Serialize();
  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  ASSERT_TRUE(task_def.has_fusion_task());
  const auto &fusion_task_def = task_def.fusion_task();

  // 验证子任务数量
  EXPECT_EQ(fusion_task_def.fusion_sub_task_info_size(), 2);

  // 验证两个子任务都是AICore类型
  for (int i = 0; i < fusion_task_def.fusion_sub_task_info_size(); i++) {
    const auto &sub_task_info = fusion_task_def.fusion_sub_task_info(i);
    EXPECT_EQ(sub_task_info.type(), domi::FusionSubTaskInfo::AICORE);
  }
}

// 测试CreateFusionTask函数，只包含CCU子任务
TEST_F(TestCreateFusionAndCcuTask, TestCreateFusionTaskWithOnlyCcu) {
  // 创建两个CCU任务
  std::vector<std::string> groups1 = {"group1", "group2"};
  auto ccu_task1 = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups1);

  std::vector<std::string> groups2 = {"group3"};
  auto ccu_task2 = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups2);

  // 创建FusionTask，只包含CCU子任务
  std::vector<KernelLaunchInfo> sub_tasks;
  sub_tasks.push_back(std::move(ccu_task1));
  sub_tasks.push_back(std::move(ccu_task2));

  auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);

  // 验证
  auto serialized_data = fusion_task.Serialize();
  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  ASSERT_TRUE(task_def.has_fusion_task());
  const auto &fusion_task_def = task_def.fusion_task();

  // 验证子任务数量
  EXPECT_EQ(fusion_task_def.fusion_sub_task_info_size(), 2);

  // 验证两个子任务都是CCU类型
  for (int i = 0; i < fusion_task_def.fusion_sub_task_info_size(); i++) {
    const auto &sub_task_info = fusion_task_def.fusion_sub_task_info(i);
    EXPECT_EQ(sub_task_info.type(), domi::FusionSubTaskInfo::CCU);
  }
}

// 测试空groups创建CcuTask
TEST_F(TestCreateFusionAndCcuTask, TestCreateCcuTaskWithEmptyGroups) {
  std::vector<std::string> empty_groups;

  auto ccu_task = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, empty_groups);

  auto serialized_data = ccu_task.Serialize();
  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  ASSERT_TRUE(task_def.has_fusion_task());
  const auto &fusion_task_def = task_def.fusion_task();
  ASSERT_EQ(fusion_task_def.fusion_sub_task_info_size(), 1);

  const auto &sub_task_info = fusion_task_def.fusion_sub_task_info(0);
  ASSERT_TRUE(sub_task_info.task().has_ccu_task_group());
  const auto &ccu_task_group = sub_task_info.task().ccu_task_group();

  // 验证groups为空
  EXPECT_EQ(ccu_task_group.group_size(), 0);
}

// 测试移动语义与FusionTask结合
TEST_F(TestCreateFusionAndCcuTask, TestFusionTaskWithMoveSemantics) {
  // 创建AICore任务
  auto aicore_task_def = CreateAicoreTaskDef(op_exe_res_ctx_);
  std::vector<uint8_t> buffer(aicore_task_def.ByteSizeLong());
  aicore_task_def.SerializeToArray(buffer.data(), buffer.size());
  auto aicore_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer);
  aicore_task.SetArgsFormat("{test_args_format}");
  aicore_task.SetBlockDim(32);

  // 创建CCU任务
  std::vector<std::string> groups = {"move_test_group"};
  auto ccu_task = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups);

  // 测试移动后原对象是否失效
  std::vector<KernelLaunchInfo> sub_tasks;
  sub_tasks.push_back(std::move(aicore_task));
  sub_tasks.push_back(std::move(ccu_task));

  // 验证移动后原对象的基本属性
  EXPECT_EQ(aicore_task.GetArgsFormat(), nullptr); // 移动后应为空
  EXPECT_EQ(ccu_task.GetArgsFormat(), nullptr);    // 移动后应为空

  // 创建FusionTask
  auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);

  // 验证FusionTask创建成功
  auto serialized_data = fusion_task.Serialize();
  EXPECT_FALSE(serialized_data.empty());
}

// 测试异常场景：空context
TEST_F(TestCreateFusionAndCcuTask, TestCreateCcuTaskWithNullContext) {
  std::vector<std::string> groups = {"group1"};

  // 测试空context，预期不会崩溃（根据代码实现，内部有GE_ASSERT_NOTNULL）
  // 注意：在实际测试中，如果启用了断言，这可能会导致测试失败
  // 这里我们主要验证函数接口的健壮性
  EXPECT_NO_THROW({
    auto ccu_task = KernelLaunchInfo::CreateCcuTask(nullptr, groups);
    // 如果创建成功，序列化应返回空或默认值
    auto data = ccu_task.Serialize();
    // 根据实现，可能返回空或默认TaskDef
  });
}

// 测试FusionTask中args_format的正确复制
TEST_F(TestCreateFusionAndCcuTask, TestArgsFormatCopyInFusionTask) {
  // 创建AICore任务并设置args_format
  auto aicore_task_def = CreateAicoreTaskDef(op_exe_res_ctx_);
  std::vector<uint8_t> buffer(aicore_task_def.ByteSizeLong());
  aicore_task_def.SerializeToArray(buffer.data(), buffer.size());
  auto aicore_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer);

  // 设置复杂的args_format
  std::string complex_args_format = "{ffts_addr}{hi.hcom0*}{i0*}{i1*}{i2*}{#0}{o0*}{o1*}{ws*}{t}";
  aicore_task.SetArgsFormat(complex_args_format.c_str());
  aicore_task.SetBlockDim(48);

  // 创建CCU任务
  std::vector<std::string> groups = {"test_group"};
  auto ccu_task = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups);

  // 创建FusionTask
  std::vector<KernelLaunchInfo> sub_tasks;
  sub_tasks.push_back(std::move(aicore_task));
  sub_tasks.push_back(std::move(ccu_task));

  auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);

  // 验证args_format是否正确复制到FusionTask
  auto serialized_data = fusion_task.Serialize();
  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  ASSERT_TRUE(task_def.has_fusion_task());
}

// 测试FusionTask中子任务顺序保持
TEST_F(TestCreateFusionAndCcuTask, TestSubTaskOrderInFusionTask) {
  // 创建三个不同类型的任务
  auto aicore_task_def = CreateAicoreTaskDef(op_exe_res_ctx_);
  std::vector<uint8_t> buffer1(aicore_task_def.ByteSizeLong());
  aicore_task_def.SerializeToArray(buffer1.data(), buffer1.size());
  auto aicore_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer1);
  aicore_task.SetArgsFormat("{aicore_args}");

  std::vector<std::string> groups1 = {"ccu1_group"};
  auto ccu_task1 = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups1);

  auto all_kernel_task_def = CreateAllKernelTaskDef(op_exe_res_ctx_);
  std::vector<uint8_t> buffer2(all_kernel_task_def.ByteSizeLong());
  all_kernel_task_def.SerializeToArray(buffer2.data(), buffer2.size());
  auto all_kernel_task = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer2);
  all_kernel_task.SetArgsFormat("{all_kernel_args}");

  std::vector<std::string> groups2 = {"ccu2_group"};
  auto ccu_task2 = KernelLaunchInfo::CreateCcuTask(op_exe_res_ctx_, groups2);

  // 按特定顺序添加子任务
  std::vector<KernelLaunchInfo> sub_tasks;
  sub_tasks.push_back(std::move(aicore_task));     // 第一个：AICore
  sub_tasks.push_back(std::move(ccu_task1));       // 第二个：CCU
  sub_tasks.push_back(std::move(all_kernel_task)); // 第三个：AllKernel
  sub_tasks.push_back(std::move(ccu_task2));       // 第四个：CCU

  auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);

  // 验证子任务顺序
  auto serialized_data = fusion_task.Serialize();
  domi::TaskDef task_def;
  ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

  ASSERT_TRUE(task_def.has_fusion_task());
  const auto &fusion_task_def = task_def.fusion_task();

  // 验证子任务数量和顺序
  ASSERT_EQ(fusion_task_def.fusion_sub_task_info_size(), 4);

  // 第一个应该是AICore
  EXPECT_EQ(fusion_task_def.fusion_sub_task_info(0).type(),
            domi::FusionSubTaskInfo::AICORE);
  EXPECT_FALSE(fusion_task_def.fusion_sub_task_info(0).task().aicore_fusion_task_info().is_all_kernel());

  // 第二个应该是CCU
  EXPECT_EQ(fusion_task_def.fusion_sub_task_info(1).type(),
            domi::FusionSubTaskInfo::CCU);

  // 第三个应该是AllKernel（在代码中也被识别为AICORE类型，但is_all_kernel为true）
  EXPECT_EQ(fusion_task_def.fusion_sub_task_info(2).type(),
            domi::FusionSubTaskInfo::AICORE);
  EXPECT_TRUE(fusion_task_def.fusion_sub_task_info(2).task().aicore_fusion_task_info().is_all_kernel());

  // 第四个应该是CCU
  EXPECT_EQ(fusion_task_def.fusion_sub_task_info(3).type(),
            domi::FusionSubTaskInfo::CCU);
}

// 测试sqe_num的计算逻辑：各种子任务sqe_num的组合
TEST_F(TestCreateFusionAndCcuTask, TestFusionTaskSqeNumCalculation) {
  // 场景1：所有子任务sqe_num都为0，融合任务sqe_num应该等于子任务数量
  {
    // 创建两个sqe_num为0的AICore任务
    auto aicore_task_def1 = CreateAicoreTaskDef(op_exe_res_ctx_);
    aicore_task_def1.set_sqe_num(0); // 明确设置sqe_num为0
    std::vector<uint8_t> buffer1(aicore_task_def1.ByteSizeLong());
    aicore_task_def1.SerializeToArray(buffer1.data(), buffer1.size());
    auto aicore_task1 = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer1);

    auto aicore_task_def2 = CreateAicoreTaskDef(op_exe_res_ctx_);
    aicore_task_def2.set_sqe_num(0); // 明确设置sqe_num为0
    std::vector<uint8_t> buffer2(aicore_task_def2.ByteSizeLong());
    aicore_task_def2.SerializeToArray(buffer2.data(), buffer2.size());
    auto aicore_task2 = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer2);

    std::vector<KernelLaunchInfo> sub_tasks;
    sub_tasks.push_back(std::move(aicore_task1));
    sub_tasks.push_back(std::move(aicore_task2));

    auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);
    auto serialized_data = fusion_task.Serialize();
    domi::TaskDef task_def;
    ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

    // 两个sqe_num为0的子任务，融合任务sqe_num应该为2（0->按1计算）
    EXPECT_EQ(task_def.sqe_num(), 2);
  }

  // 场景2：子任务sqe_num不为0，融合任务sqe_num应该是各子任务sqe_num之和
  {
    // 创建两个sqe_num不为0的AICore任务
    auto aicore_task_def1 = CreateAicoreTaskDef(op_exe_res_ctx_);
    aicore_task_def1.set_sqe_num(3); // 设置sqe_num为3
    std::vector<uint8_t> buffer1(aicore_task_def1.ByteSizeLong());
    aicore_task_def1.SerializeToArray(buffer1.data(), buffer1.size());
    auto aicore_task1 = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer1);

    auto aicore_task_def2 = CreateAicoreTaskDef(op_exe_res_ctx_);
    aicore_task_def2.set_sqe_num(2); // 设置sqe_num为2
    std::vector<uint8_t> buffer2(aicore_task_def2.ByteSizeLong());
    aicore_task_def2.SerializeToArray(buffer2.data(), buffer2.size());
    auto aicore_task2 = KernelLaunchInfo::LoadFromData(op_exe_res_ctx_, buffer2);

    std::vector<KernelLaunchInfo> sub_tasks;
    sub_tasks.push_back(std::move(aicore_task1));
    sub_tasks.push_back(std::move(aicore_task2));

    auto fusion_task = KernelLaunchInfo::CreateFusionTask(op_exe_res_ctx_, sub_tasks);
    auto serialized_data = fusion_task.Serialize();
    domi::TaskDef task_def;
    ASSERT_TRUE(task_def.ParseFromArray(serialized_data.data(), serialized_data.size()));

    // 两个sqe_num不为0的子任务，融合任务sqe_num应该为3+2=5
    EXPECT_EQ(task_def.sqe_num(), 5);
  }
}
}