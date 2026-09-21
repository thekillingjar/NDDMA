/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <vector>

#include "macro_utils/dt_public_scope.h"
#include "framework/common/taskdown_common.h"
#include "hybrid/executor/subgraph_context.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "api/gelib/gelib.h"
#include "depends/runtime/src/runtime_stub.h"
#include "macro_utils/dt_public_unscope.h"
#include "common/tbe_handle_store/bin_register_utils.h"
#include "register/op_impl_registry.h"
#include "common/share_graph.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "register/kernel_registry.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace testing;

namespace ge {
using namespace hybrid;
using namespace optiling;
namespace {
struct DummyTilingParams {
  int64_t x;
  int64_t y;
};
struct DummyCompileInfo {
  int64_t tiling_key;
  int64_t block_dim;
  bool is_need_atomic;
  int64_t tiling_cond;
  std::vector<int64_t> c;
};
}

class UtestAiCoreOpTask : public testing::Test {
 protected:
  void SetUp() {
    ResetNodeIndex();
  }
  void TearDown() {}
};

static domi::TaskDef CreateTaskDef() {
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  std::vector<uint8_t> args(100, 0);
  task_def.mutable_kernel()->set_args(args.data(), args.size());
  task_def.mutable_kernel()->set_args_size(100);
  task_def.mutable_kernel()->mutable_context()->set_kernel_type(static_cast<uint32_t>(ccKernelType::TE));
  uint16_t args_offset = 0;
  task_def.mutable_kernel()->mutable_context()->set_args_offset(&args_offset, sizeof(args_offset));
  return task_def;
}

TEST_F(UtestAiCoreOpTask, Init_failed) {
  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add", 2, 1, false);
  ge::AttrUtils::SetInt(*op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  auto node = graph->AddNode(op_desc);
  domi::TaskDef task_def = CreateTaskDef();
  EXPECT_EQ(task1->Init(node, task_def), SUCCESS);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

  const char tbe_bin[] = "tbe_bin";
  vector<char> buffer(tbe_bin, tbe_bin + strlen(tbe_bin));
  node = graph->AddNode(op_desc);
  OpKernelBinPtr tbe_kernel_ptr = std::make_shared<OpKernelBin>(node->GetName(), std::move(buffer));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel_ptr);
  void *stub_func = nullptr;
  AttrNameOfBinOnOp attr_names = {OP_EXTATTR_NAME_TBE_KERNEL, TVM_ATTR_NAME_METADATA, "_kernelname",
                                  TVM_ATTR_NAME_MAGIC};
  EXPECT_EQ(BinRegisterUtils::RegisterBin(*op_desc, task1->stub_name_, attr_names, stub_func), SUCCESS);
}

TEST_F(UtestAiCoreOpTask, Init_success) {
  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  OpDescPtr op_desc = CreateOpDesc("Add", "Add", 2, 1, false);
  ge::AttrUtils::SetInt(*op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto node = graph->AddNode(op_desc);
  domi::TaskDef task_def = CreateTaskDef();
  EXPECT_EQ(task1->Init(node, task_def), ge::SUCCESS);

  task1->need_tiling_ = true;
  EXPECT_EQ(task1->Init(node, task_def), ge::SUCCESS);
}

TEST_F(UtestAiCoreOpTask, UpdateOutputsShape_success) {
  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  OpDescPtr op_desc = CreateOpDesc("Add", "Add", 2, 1, false);
  ge::AttrUtils::SetInt(*op_desc, ge::ATTR_NAME_UNKNOWN_SHAPE_TYPE, DEPEND_SHAPE_RANGE);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto node = graph->AddNode(op_desc);
  domi::TaskDef task_def = CreateTaskDef();

  EXPECT_EQ(task1->Init(node, task_def), ge::SUCCESS);


  node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;

  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  graph_context.callback_manager = new (std::nothrow) RtCallbackManager();
  graph_context.own_callback_manager = true;

  auto node_state = subgraph_context.GetNodeState(node_item);
  ASSERT_NE(node_state, nullptr);
  auto outputs_shape = reinterpret_cast<uint32_t(*)[9]>(task1->shape_buffer_->GetData());
  outputs_shape[0][0] = 2;
  outputs_shape[0][1] = 1;
  outputs_shape[0][2] = 2;
  ASSERT_EQ(task1->UpdateOutputsShape(*node_state->GetTaskContext()), SUCCESS);
}

TEST_F(UtestAiCoreOpTask, RegisterKernelHandleTest) {
  auto op_desc = std::make_shared<ge::OpDesc>("data", DATA);
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  ge::AttrUtils::SetStr(op_desc, ge::TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF_AICUBE");
  std::unique_ptr<AiCoreOpTask> task(new AiCoreOpTask());
  void *handle = nullptr;
  AttrNameOfBinOnOp attr_names = {OP_EXTATTR_NAME_TBE_KERNEL, TVM_ATTR_NAME_METADATA, "_kernelname",
                                  TVM_ATTR_NAME_MAGIC};
  Status ret = BinRegisterUtils::RegisterBinWithHandle(*op_desc, attr_names, handle);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestAiCoreOpTask, TestKernelLaunchTiling) {
  std::unique_ptr<AiCoreOpTask> task(new AiCoreOpTask());
  task->need_tiling_ = true;
  ASSERT_EQ(task->LaunchKernel(nullptr), SUCCESS);
  char handle = 'a';
  task->handle_ = &handle;
  ASSERT_EQ(task->LaunchKernel(nullptr), SUCCESS);
}

TEST_F(UtestAiCoreOpTask, TestInitAtomicAddrCleanIndices) {
  std::unique_ptr<AtomicAddrCleanOpTask> task(new AtomicAddrCleanOpTask());
  OpDescPtr op_desc = CreateOpDesc("Atomic", "AtomicAddrClean", 0, 0, false);
  domi::TaskDef task_def = CreateTaskDef();

  std::vector<int64_t> atomic_output_indices = {};
  ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto node = graph->AddNode(op_desc);
  ASSERT_EQ(task->Init(node, task_def), INTERNAL_ERROR);

  std::map<std::string, std::map<int64_t, int64_t>> workspace_info;
  workspace_info["test1"] = {{2, 3}};
  workspace_info["test2"] = {{3, 4}};
  atomic_output_indices = {1, 1};
  ge::AttrUtils::SetListInt(op_desc, ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  op_desc->SetExtAttr(EXT_ATTR_ATOMIC_WORKSPACE_INFO, workspace_info);
  auto node2 = graph->AddNode(op_desc);
  task->need_tiling_ = true;
  ASSERT_EQ(task->Init(node2, task_def), SUCCESS);
  task->max_arg_count_ = 0U;
  ASSERT_EQ(task->InitAtomicAddrCleanIndices(*op_desc), INTERNAL_ERROR);
}

TEST_F(UtestAiCoreOpTask, TestAtomicCalcTilingInfo) {
  std::unique_ptr<AtomicAddrCleanOpTask> task(new AtomicAddrCleanOpTask());
  OpDescPtr op_desc = CreateOpDesc("Atomic", "AtomicAddrClean", 0, 0);
  OpDescPtr add_op_desc = CreateOpDesc("Add", "Add", 2, 1);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr add_node = graph->AddNode(add_op_desc);
  task->tiling_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(task->tiling_info_, nullptr);
  Operator op("add");
  ::optiling::OpTilingFuncInfo atomic_tiling_info;
  add_node->GetOpDesc()->SetAtomicTilingFuncInfo(&atomic_tiling_info);
  ASSERT_NE(task->CalcTilingInfo(add_node, op), SUCCESS);
}

TEST_F(UtestAiCoreOpTask, TestAtomicUpdateArgs) {
  std::unique_ptr<AtomicAddrCleanOpTask> task(new AtomicAddrCleanOpTask());

  task->atomic_output_indices_ = {0};
  task->atomic_workspace_indices_ = {0};

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr add_op_desc = CreateOpDesc("Add", "Add", 2, 1);
  NodePtr node = graph->AddNode(add_op_desc);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  auto &task_ctx = *node_state->GetTaskContext();
  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor_value = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  task_ctx.outputs_start_ = &tensor_value;

  auto work_space = allocator->Allocate(100);
  task_ctx.workspaces_.push_back(work_space);
  task_ctx.execution_context_->allocator = allocator;
  task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(uintptr_t) * 2]);
  task->arg_base_ = reinterpret_cast<uintptr_t *>(task->args_.get());
  task->need_tiling_ = false;
  node_state->GetTaskContext()->execution_context_->trace_enabled = true;
  ASSERT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
  ASSERT_NE(task->GetKeyForOpParamSize().size(), 0);
}

TEST_F(UtestAiCoreOpTask, TestUpdateTilingInfo) {
  std::unique_ptr<AiCoreOpTask> task(new AiCoreOpTask());
  auto graph = gert::ShareGraph::AicoreGraph();

  NodePtr add_node = graph->FindNode("add1");
  auto op_desc = add_node->GetOpDesc();
  ge::OpDescUtilsEx::SetType(op_desc, "MyAdd");
  auto add_op_desc = add_node->GetOpDesc();
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(add_node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  auto &task_ctx = *node_state->GetTaskContext();
  task->tiling_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  gert::OpImplKernelRegistry::TilingKernelFunc op_tiling_func = [](gert::TilingContext *tiling_context) -> graphStatus {
    // write tiling data
    DummyTilingParams *tiling_data_ptr = tiling_context->GetTilingData<DummyTilingParams>();
    EXPECT_NE(tiling_data_ptr, nullptr);
    tiling_data_ptr->x = 1;
    tiling_data_ptr->y = 2;
    return ge::GRAPH_SUCCESS;
  };
  graphStatus (*tiling_parse_func)(::gert::TilingParseContext *) =
      [](gert::TilingParseContext *parse_context) -> graphStatus { return GRAPH_SUCCESS; };

  typedef void* (*CreateCompileInfo)();
  typedef void (*DeleteCompileInfo)(void *obj);
  CreateCompileInfo create_compile_info = []() -> void *{
    return new int64_t();
  };
  DeleteCompileInfo delete_compile_info = [](void *obj) -> void {
    delete (int64_t*)obj;
  };
  auto op_impl_func = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("MyAdd");
  op_impl_func->tiling = op_tiling_func;
  op_impl_func->tiling_parse = reinterpret_cast<gert::KernelRegistry::KernelFunc>(tiling_parse_func);
  op_impl_func->compile_info_creator = create_compile_info;
  op_impl_func->compile_info_deleter = delete_compile_info;

  ge::AttrUtils::SetStr(add_op_desc, "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(add_op_desc, "compile_info_json", "op_compile_info_json");
  ge::AttrUtils::SetInt(add_op_desc, "op_para_size", 2 * sizeof(DummyTilingParams) + 18 * sizeof(int64_t));

  EXPECT_EQ(task->UpdateTilingInfo(task_ctx), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  task->args_size_ = sizeof(uintptr_t) * 3 + 160;
  task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[task->args_size_]);
  task->arg_base_ = reinterpret_cast<uintptr_t *>(task->args_.get());
  task->max_tiling_size_ = 160;
  EXPECT_EQ(task->UpdateTilingInfo(task_ctx), SUCCESS);
}

TEST_F(UtestAiCoreOpTask, TestTilingkey) {
  OpTilingFuncV2 op_tiling_func = [](const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &run_info) -> bool {
    run_info.SetTilingKey(0xFFFFFFFFFU);
    return true;
  };
  REGISTER_OP_TILING_UNIQ_V2(Cast, op_tiling_func, 1);
  OpTilingRegistryInterf_V2("Cast", op_tiling_func);

  OpDescPtr op_desc = CreateOpDesc("Cast", "Cast", 1, 1);
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "op_compile_info_json");
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  optiling::utils::OpRunInfo run_info(0, true, 0);
  ASSERT_EQ(optiling::OpParaCalculateV2(op, run_info), SUCCESS);

  std::unique_ptr<AiCoreOpTask> task(new AiCoreOpTask());
  task->tiling_key_ = run_info.GetTilingKey();
  ASSERT_EQ(run_info.GetTilingKey(), 0xFFFFFFFFFU);
  ASSERT_EQ(std::to_string(task->tiling_key_), std::to_string(run_info.GetTilingKey()));
}

TEST_F(UtestAiCoreOpTask, TestUpdateBinHandle) {
  OpDescPtr op_desc = CreateOpDesc("Cast", "Cast", 1, 1, false);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  // create task context
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  auto &task_ctx = *node_state->GetTaskContext();

  char handle = 'a';
  NodeCompileCacheItem cci;
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "op_compile_info_json");
  ge::AttrUtils::SetStr(op_desc, "_atomic_compile_info_key", "_atomic_compile_info_key");
  ge::AttrUtils::SetStr(op_desc, "_atomic_compile_info_json", "_atomic_compile_info_key");
  auto ret = NodeCompileCacheItem::Build(KernelLaunchBinType::kStubFunc, node, &handle, cci);
  EXPECT_EQ(ret, SUCCESS);

  std::unique_ptr<AiCoreOpTask> task(new AiCoreOpTask());
  task->handle_ = nullptr;
  ret = task->UpdateBinHandle(task_ctx, cci);
  EXPECT_EQ(ret, SUCCESS);  // update op task of stub_func success

  NodeCompileCacheItem cci_with_handle;
  ret = NodeCompileCacheItem::Build(KernelLaunchBinType::kWithHandle, node, &handle, cci_with_handle);
  EXPECT_EQ(ret, SUCCESS);
  ret = task->UpdateBinHandle(task_ctx, cci_with_handle);
  EXPECT_EQ(ret, SUCCESS);  // update op task of stub_func with handle success

  task->handle_ = &handle;  // op task of with handle
  ret = task->UpdateBinHandle(task_ctx, cci_with_handle);
  EXPECT_EQ(ret, SUCCESS);  // update op task of handle with handle success;

  ret = task->UpdateBinHandle(task_ctx, cci);
  EXPECT_EQ(ret, SUCCESS);  // update op task of handle with stub func success;

  NodeCompileCacheItem cci_with_handle_type_unsupported;
  NodeCompileCacheItem::Build(static_cast<KernelLaunchBinType>(100), node, &handle, cci_with_handle_type_unsupported);
  ret = task->UpdateBinHandle(task_ctx, cci_with_handle_type_unsupported);
  EXPECT_EQ(ret, FAILED);
}

// aicore with host mem input optimization
TEST_F(UtestAiCoreOpTask, TestUpdateArgsWithHostMemInput) {
  std::unique_ptr<AiCoreOpTask> task(new AiCoreOpTask());

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr add_op_desc = CreateOpDesc("Add", "Add", 2, 1, false);
  AttrUtils::SetInt(add_op_desc->MutableInputDesc(0), ATTR_NAME_PLACEMENT, 2);
  AttrUtils::SetInt(add_op_desc->MutableInputDesc(1), ATTR_NAME_PLACEMENT, 2);
  AttrUtils::SetInt(add_op_desc->MutableOutputDesc(0), ATTR_NAME_PLACEMENT, 2);
  AttrUtils::SetInt(add_op_desc->MutableOutputDesc(0), "globalworkspace_type", 0U);
  NodePtr node = graph->AddNode(add_op_desc);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  domi::TaskDef task_def = CreateTaskDef();

  // check, args are extended for host memory input data
  EXPECT_EQ(task->Init(node, task_def), SUCCESS);
  EXPECT_NE(task->host_mem_input_data_offset_, 0);
  EXPECT_EQ(task->args_size_ - task->host_mem_input_data_offset_, kMaxHostMemInputLen);

  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  uint8_t host_mem[64];
  TensorValue tensorValue(&host_mem, 64);
  subgraph_context.all_inputs_.emplace_back(tensorValue);
  subgraph_context.all_inputs_.emplace_back(tensorValue);
  EXPECT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  auto &task_ctx = *node_state->GetTaskContext();
  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor_value = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  task_ctx.outputs_start_ = &tensor_value;

  auto work_space = allocator->Allocate(100);
  task_ctx.workspaces_.push_back(work_space);
  task_ctx.execution_context_->allocator = allocator;
  task->need_tiling_ = false;
  task->SetNeedHostMemOpt(true);
  node_state->GetTaskContext()->execution_context_->trace_enabled = true;
  EXPECT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);

  // check host mem data
  ASSERT_NE(task->args_ex_.hostInputInfoPtr, nullptr);
  ASSERT_EQ(task->args_ex_.hostInputInfoNum, 2);

  uint64_t *addr = PtrToPtr<void, uint64_t>(ValueToPtr(PtrToValue(task->args_ex_.args) +
                                                       task->args_ex_.hostInputInfoPtr[0].addrOffset));
  uint64_t host_mem_data_addr = PtrToValue(task->args_ex_.args) +
                                task->args_ex_.hostInputInfoPtr[0].dataOffset;
  EXPECT_EQ(*addr, host_mem_data_addr);

  addr = PtrToPtr<void, uint64_t>(ValueToPtr(PtrToValue(task->args_ex_.args) +
                                             task->args_ex_.hostInputInfoPtr[1].addrOffset));
  host_mem_data_addr = PtrToValue(task->args_ex_.args) +
                       task->args_ex_.hostInputInfoPtr[1].dataOffset;
  EXPECT_EQ(*addr, host_mem_data_addr);
}
TEST_F(UtestAiCoreOpTask, TestUpdateArgsWithOverflow) {
  std::unique_ptr<AiCoreOpTask> task(new AiCoreOpTask());

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  OpDescPtr add_op_desc = CreateOpDesc("Add", "Add", 2, 1, false);
  AttrUtils::SetInt(add_op_desc->MutableInputDesc(0), ATTR_NAME_PLACEMENT, 2);
  AttrUtils::SetInt(add_op_desc->MutableInputDesc(1), ATTR_NAME_PLACEMENT, 2);
  AttrUtils::SetInt(add_op_desc->MutableOutputDesc(0), ATTR_NAME_PLACEMENT, 2);
  AttrUtils::SetInt(add_op_desc->MutableOutputDesc(0), "globalworkspace_type", 0U);
  NodePtr node = graph->AddNode(add_op_desc);
  std::unique_ptr<NodeItem> new_node;
  ASSERT_EQ(NodeItem::Create(node, new_node), SUCCESS);
  NodeItem *node_item = new_node.get();
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->is_dynamic = true;
  node_item->shape_inference_type = DEPEND_SHAPE_RANGE;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item);
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 1;

  domi::TaskDef task_def = CreateTaskDef();
  EXPECT_EQ(task->Init(node, task_def), SUCCESS);
  GraphExecutionContext graph_context;
  SubgraphContext subgraph_context(&graph_item, &graph_context);
  uint8_t host_mem[64];
  TensorValue tensorValue(&host_mem, 64);
  subgraph_context.all_inputs_.emplace_back(tensorValue);
  subgraph_context.all_inputs_.emplace_back(tensorValue);
  EXPECT_EQ(subgraph_context.Init(), SUCCESS);

  auto node_state = subgraph_context.GetNodeState(node_item);
  auto &task_ctx = *node_state->GetTaskContext();
  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor_value = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  task_ctx.outputs_start_ = &tensor_value;

  auto work_space = allocator->Allocate(100);
  task_ctx.workspaces_.push_back(work_space);
  task_ctx.execution_context_->allocator = allocator;
  task->need_tiling_ = false;
  node_state->GetTaskContext()->execution_context_->trace_enabled = true;
  uint64_t overflow_value = 111;
  task->overflow_addr_ = &overflow_value;
  task->need_overflow_addr_ = true;
  EXPECT_EQ(task->UpdateArgs(*node_state->GetTaskContext()), SUCCESS);
  EXPECT_EQ(ValueToPtr(task->arg_base_[4]), &overflow_value);
}
}  // namespace ge
