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
#include <vector>
#include "runtime/rt.h"

#include "macro_utils/dt_public_scope.h"
#include "graph/utils/node_utils.h"
#include "hybrid/model/hybrid_model_builder.h"
#include "hybrid/model/hybrid_model.h"
#include "hybrid/node_executor/node_executor.h"
#include "common/model/ge_model.h"
#include "common/model/ge_root_model.h"
#include "common/profiling/profiling_manager.h"
#include "framework/common/profiling_definitions.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "graph/ge_context.h"
#include "hybrid/executor/hybrid_execution_context.h"
#include "hybrid/hybrid_davinci_model.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "hybrid/node_executor/aicore/aicore_task_builder.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "hybrid/common/npu_memory_allocator.h"
#include "register/op_tiling_registry.h"
#include "graph/types.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/testcase/ge_graph/graph_builder_utils.h"
#include "single_op/task/build_task_utils.h"
#include "graph/normal_graph/op_desc_impl.h"
#include "framework/generator/ge_generator.h"
#include "depends/profiler/src/profiling_test_util.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"

#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace std;
using namespace ge;
using namespace optiling;
using namespace profiling;

namespace ge {
using namespace hybrid;

class UtestGeHybrid : public testing::Test {
 protected:
  void SetUp() {
    ResetNodeIndex();
  }

  void TearDown() {
    ProfilingTestUtil::Instance().Clear();
  }
};

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

struct FakeGraphItem : GraphItem {
  FakeGraphItem(NodePtr node) {
    NodeItem::Create(node, node_item);
    node_item->input_start = 0;
    node_item->output_start = 0;
    node_items_.emplace_back(node_item.get());
    total_inputs_ = node->GetAllInAnchors().size();
    total_outputs_ = node->GetAllOutAnchors().size();
  }

  NodeItem *GetNodeItem() {
    return node_item.get();
  }

 private:
  std::unique_ptr<NodeItem> node_item;
};

class DModelListener : public ModelListener {
 public:
  DModelListener(){};
  uint32_t OnComputeDone(uint32_t model_id, uint32_t data_index, uint32_t resultCode,
                         std::vector<gert::Tensor> &outputs) {
    return 0;
  }
};

}  // namespace

TEST_F(UtestGeHybrid, aicore_op_task_init_success) {
  // build aicore task
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_with_handle = task_def.mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add", 0, 0, false);
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
  auto graph = make_shared<ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  ASSERT_EQ(aicore_task->Init(node, task_def), SUCCESS);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
  char handle = '0';
  aicore_task->handle_ = &handle;
  aicore_task->tiling_key_ = 1;
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestGeHybrid, aicore_op_task_init_success2) {
  // build aicore task
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  aicore_task->is_single_op_ = true;
  domi::TaskDef task_def;
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_KERNEL));
  domi::KernelDef *kernel = task_def.mutable_kernel();
  kernel->set_block_dim(32);
  kernel->set_args_size(64);
  string args(64, '1');
  kernel->set_args(args.data(), 64);
  domi::KernelContext *context = kernel->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
  auto graph = make_shared<ComputeGraph>("graph");
  auto node = graph->AddNode(op_desc);
  ASSERT_EQ(aicore_task->InitWithTaskDef(node, task_def), SUCCESS);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
  char handle = '0';
  aicore_task->handle_ = &handle;
  aicore_task->tiling_key_ = 1;
  ASSERT_EQ(aicore_task->LaunchKernel(stream), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestGeHybrid, task_update_tiling_info) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "json");
  ge::AttrUtils::SetBool(op_desc, "support_dynamicshape", true);
  ge::AttrUtils::SetInt(op_desc, "op_para_size", 1);
  auto node = graph->AddNode(op_desc);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  //OpTilingFuncV2 op_tiling_func = [](const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &) -> bool {return true;};
  //REGISTER_OP_TILING_UNIQ_V2(Add, op_tiling_func, 1);
  //OpTilingRegistryInterf_V2("Add", op_tiling_func);
  gert::OpImplKernelRegistry::TilingKernelFunc op_tiling_func = [](gert::TilingContext *tiling_context) -> graphStatus {
    // simulate op write tiling info
    tiling_context->SetTilingKey(1);
    tiling_context->SetBlockDim(2);
    tiling_context->SetNeedAtomic(true);
    tiling_context->SetTilingCond(3);
    return ge::GRAPH_SUCCESS;
  };
  auto tiling_parse_func = [](gert::TilingParseContext *parse_context) -> graphStatus {return GRAPH_SUCCESS;};

  IMPL_OP(Add).TilingParse<DummyCompileInfo>(tiling_parse_func).Tiling(op_tiling_func);

  GraphExecutionContext execution_context;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item.get());
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  auto node_state = subgraph_context.GetNodeState(node_item.get());
  ASSERT_EQ(aicore_task->InitTilingInfo(*op_desc), SUCCESS);
  aicore_task->tiling_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(aicore_task->tiling_info_, nullptr);
  aicore_task->args_size_ = 64;
  ASSERT_EQ(aicore_task->UpdateTilingInfo(*node_state->GetTaskContext()), SUCCESS);
}

TEST_F(UtestGeHybrid, index_taskdefs_failed) {
  // build aicore task
  domi::ModelTaskDef model_task_def;

  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(1);
  context->set_kernel_type(2);    // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("kernel/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  ASSERT_EQ(hybrid_model_builder.Build(), INTERNAL_ERROR);
  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), INTERNAL_ERROR);
}

TEST_F(UtestGeHybrid, parse_force_infershape_nodes) {
  const char *const kForceInfershape = "_force_infershape_when_running";
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Conv2D", "Conv2D");
  ge::AttrUtils::SetBool(op_desc, kForceInfershape, true);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> new_node;
  NodeItem::Create(node, new_node);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  ASSERT_EQ(hybrid_model_builder.ParseForceInfershapeNodes(node, *new_node), SUCCESS);
}
static ComputeGraphPtr BuildDataDirectConnectGraph() {
  const char *kRefIndex = "_parent_node_index";
  ge::ut::GraphBuilder builder("subgraph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto netoutput = builder.AddNode("NetOutput", "NetOutput", 1, 1);
  (void)AttrUtils::SetInt(netoutput->GetOpDesc()->MutableInputDesc(0), kRefIndex, 0);

  builder.AddDataEdge(data, 0, netoutput, 0);
  return builder.GetGraph();
}
TEST_F(UtestGeHybrid, data_direct_connect) {
  std::unique_ptr<NodeItem> node_item;
  auto root_graph = make_shared<ComputeGraph>("root_graph");
  OpDescPtr op_desc = CreateOpDesc("PartitionedCall", "PartitionedCall");
  auto node = root_graph->AddNode(op_desc);
  node->SetOwnerComputeGraph(root_graph);
  auto sub_graph = BuildDataDirectConnectGraph();
  sub_graph->SetParentGraph(root_graph);
  sub_graph->SetParentNode(node);
  node->GetOpDesc()->AddSubgraphName("subgraph");
  node->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  root_graph->AddSubgraph("subgraph", sub_graph);
  std::unique_ptr<NodeItem> new_node;
  NodeItem::Create(node, new_node);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  auto ret = hybrid_model_builder.IdentifyVariableOutputs(*new_node.get(), sub_graph);
  ASSERT_EQ(ret, SUCCESS);
}

TEST_F(UtestGeHybrid, index_taskdefs_success) {
  // build aicore task
  domi::ModelTaskDef model_task_def;

  std::shared_ptr<domi::ModelTaskDef> model_task_def_ptr = make_shared<domi::ModelTaskDef>(model_task_def);
  domi::TaskDef *task_def = model_task_def_ptr->add_task();
  GeModelPtr ge_model = make_shared<GeModel>();
  ge_model->SetModelTaskDef(model_task_def_ptr);

  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new (std::nothrow) hybrid::AiCoreOpTask());
  task_def->set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_with_handle = task_def->mutable_kernel_with_handle();
  kernel_with_handle->set_original_kernel_key("");
  kernel_with_handle->set_node_info("");
  kernel_with_handle->set_block_dim(32);
  kernel_with_handle->set_args_size(64);
  string args(64, '1');
  kernel_with_handle->set_args(args.data(), 64);
  domi::KernelContext *context = kernel_with_handle->mutable_context();
  context->set_op_index(0);
  context->set_kernel_type(2);  // ccKernelType::TE
  uint16_t args_offset[9] = {0};
  context->set_args_offset(args_offset, 9 * sizeof(uint16_t));

  OpDescPtr op_desc = CreateOpDesc("name/Add", "Add");
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/Add", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  std::string kernel_name("name/Add");
  AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", kernel_name);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "MIX_AIC");
  AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, kernel_name);

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = graph->AddNode(op_desc);
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), SUCCESS);
  TBEKernelStore tbe_kernel_store;
  tbe_kernel_store.AddTBEKernel(tbe_kernel);
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), SUCCESS);
}

TEST_F(UtestGeHybrid, TestForTryMalloc) {
  auto allocator = hybrid::NpuMemoryAllocator::GetAllocator(nullptr);
  ASSERT_NE(allocator, nullptr);
  void *buffer = nullptr;
  ASSERT_EQ(allocator->TryFreeAndMalloc(1, &buffer), SUCCESS);
  ASSERT_NE(buffer, nullptr);
  buffer = nullptr;
  auto allocator2 = hybrid::NpuMemoryAllocator::GetAllocator();
  ASSERT_EQ(allocator2->TryFreeAndMalloc(1, &buffer), SUCCESS);
  ASSERT_NE(buffer, nullptr);
  hybrid::NpuMemoryAllocator::Finalize();
  rtFree(buffer);
}

TEST_F(UtestGeHybrid, init_weight_success) {
  // make graph with sub_graph
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("root_graph");
  OpDescPtr op_desc = CreateOpDesc("if", IF);
  NodePtr node = graph->AddNode(op_desc);
  // make sub graph
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("if_sub_graph");
  OpDescPtr const_op_desc = CreateOpDesc("const", CONSTANT);
  vector<int64_t> dims_vec_0 = {2, 1, 4, 1, 2};
  vector<int32_t> data_vec_0 = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  (void)TensorUtils::SetRealDimCnt(tensor_desc_0, dims_vec_0.size());
  ConstGeTensorPtr constTensor_0 =
    std::make_shared<GeTensor>(tensor_desc_0, (uint8_t *)&data_vec_0[0], data_vec_0.size() * sizeof(int32_t));
  AttrUtils::SetTensor(const_op_desc, ge::ATTR_NAME_WEIGHTS, constTensor_0);
  const_op_desc->AddOutputDesc(tensor_desc_0);
  NodePtr const_node = sub_graph->AddNode(const_op_desc);
  graph->AddSubgraph("sub", sub_graph);

  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  //Buffer weight_buffer = Buffer(128,0);
  //ge_sub_model->SetWeight(weight_buffer);
  ge_root_model->SetSubgraphInstanceNameToModel("sub",ge_sub_model);
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  auto ret = hybrid_model_builder.InitWeights();
  ASSERT_EQ(ret,SUCCESS);
}

TEST_F(UtestGeHybrid, hybrid_model_executor) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("abc");
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  HybridModel model(root_model);
  model.root_graph_item_.reset(new GraphItem);
  HybridModel *model_ptr = &model;

  uint32_t device_id = 0;
  rtStream_t stream = nullptr;
  HybridModelRtV1Executor executor(model_ptr, device_id, stream);
  executor.Init();
}

TEST_F(UtestGeHybrid, test_parse_parallel_group) {
  NodeExecutorManager::GetInstance().engine_mapping_.emplace("ops_kernel_info_hccl",
                                                             NodeExecutorManager::ExecutorType::HCCL);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test");
  OpDescPtr op_desc = CreateOpDesc("AllReduce", "AllReduce");
  op_desc->SetId(0);
  ge::AttrUtils::SetStr(op_desc, ATTR_NAME_PARALLEL_GROUP, "group_1");
  auto node = compute_graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->node_id = 0;

  op_desc->SetOpKernelLibName("ops_kernel_info_hccl");
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  HybridModel model(root_model);
  model.root_graph_ = compute_graph;

  HybridModelBuilder builder(model);
  ASSERT_EQ(builder.CollectParallelGroups(*node_item), SUCCESS);

  ASSERT_EQ(builder.node_to_parallel_groups_.size(), 1);
  ASSERT_EQ(builder.parallel_group_to_nodes_.size(), 1);

  OpDescPtr op_desc_1 = CreateOpDesc("subgraph", "PartitionedCall");
  op_desc_1->AddSubgraphName("subgraph");
  auto node_1 = compute_graph->AddNode(op_desc_1);

  ComputeGraphPtr subgraph = MakeShared<ComputeGraph>("subgraph");
  ASSERT_EQ(NodeUtils::SetSubgraph(*node_1, 0, subgraph), GRAPH_SUCCESS);

  std::unique_ptr<NodeItem> node_item_1;
  NodeItem::Create(node_1, node_item_1);
  node_item_1->node_id = 1;

  ASSERT_EQ(builder.CollectParallelGroups(*node_item_1), SUCCESS);
  ASSERT_EQ(builder.node_to_parallel_groups_.size(), 1);
  ASSERT_EQ(builder.parallel_group_to_nodes_.size(), 1);

  OpDescPtr op_desc_2 = CreateOpDesc("sub_node_1", "AllReduce");
  ge::AttrUtils::SetStr(op_desc_2, ATTR_NAME_PARALLEL_GROUP, "group_1");
  auto node_2 = subgraph->AddNode(op_desc_2);
  ASSERT_TRUE(node_2 != nullptr);

  OpDescPtr op_desc_3 = CreateOpDesc("sub_node_2", "AllReduce2");
  ge::AttrUtils::SetStr(op_desc_3, ATTR_NAME_PARALLEL_GROUP, "group_2");
  auto node_3 = subgraph->AddNode(op_desc_3);
  ASSERT_TRUE(node_3 != nullptr);

  ASSERT_EQ(builder.CollectParallelGroups(*node_item_1), SUCCESS);
  ASSERT_EQ(builder.node_to_parallel_groups_.size(), 2);
  ASSERT_EQ(builder.parallel_group_to_nodes_.size(), 2);
  ASSERT_EQ(builder.parallel_group_to_nodes_["group_1"].size(), 2);
  ASSERT_EQ(builder.parallel_group_to_nodes_["group_2"].size(), 1);

  builder.parallel_group_to_nodes_.clear();
  builder.node_ref_inputs_.clear();
  model.node_items_[node] = std::move(node_item);
  model.node_items_[node_1] = std::move(node_item_1);

  ASSERT_FALSE(model.node_items_[node]->has_observer);
  ASSERT_TRUE(model.node_items_[node_1]->dependents_for_execution.empty());
  ASSERT_EQ(builder.ParseDependentByParallelGroup(), SUCCESS);
  ASSERT_TRUE(model.node_items_[node]->has_observer);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution.size(), 1);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution[0], node);

  // repeat parse
  ASSERT_EQ(builder.ParseDependentByParallelGroup(), SUCCESS);
  ASSERT_TRUE(model.node_items_[node]->has_observer);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution.size(), 1);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution[0], node);
}

TEST_F(UtestGeHybrid, unfold_subgraphs_success) {
  ComputeGraphPtr root_graph = std::make_shared<ComputeGraph>("root_graph");
  auto partitioned_call_op_desc = CreateOpDesc("partitioned_call", PARTITIONEDCALL, 3, 1);
  auto partitioned_call_node = root_graph->AddNode(partitioned_call_op_desc);
  partitioned_call_op_desc->AddSubgraphName("f");
  partitioned_call_op_desc->SetSubgraphInstanceName(0, "sub_graph");

  ComputeGraphPtr sub_sub_graph1 = std::make_shared<ComputeGraph>("while_cond");
  {
    OpDescPtr sub_sub_graph_while_cond_data_op_desc = CreateOpDesc("cond_data", DATA);
    NodePtr sub_sub_graph_while_cond_data_node = sub_sub_graph1->AddNode(sub_sub_graph_while_cond_data_op_desc);
    sub_sub_graph1->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_sub_graph1);
  }

  ComputeGraphPtr sub_sub_graph2 = std::make_shared<ComputeGraph>("while_body");
  {
    OpDescPtr sub_sub_graph_while_body_data_op_desc = CreateOpDesc("body_data", DATA);
    NodePtr sub_sub_graph_while_body_data_node = sub_sub_graph2->AddNode(sub_sub_graph_while_body_data_op_desc);
    sub_sub_graph2->SetGraphUnknownFlag(true);
    sub_sub_graph2->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_sub_graph2);
  }

  // Will unfold to merged_graph.
  ComputeGraphPtr sub_graph = std::make_shared<ComputeGraph>("sub_graph");
  {
    OpDescPtr sub_graph_data1_op_desc = CreateOpDesc("data1", DATA, 1, 1);
    OpDescPtr sub_graph_data2_op_desc = CreateOpDesc("data2", DATA, 1, 1);
    OpDescPtr sub_graph_data3_op_desc = CreateOpDesc("data3", DATA, 1, 1);
    NodePtr sub_graph_data1_node = sub_graph->AddNode(sub_graph_data1_op_desc);
    NodePtr sub_graph_data2_node = sub_graph->AddNode(sub_graph_data2_op_desc);
    NodePtr sub_graph_data3_node = sub_graph->AddNode(sub_graph_data3_op_desc);

    AttrUtils::SetInt(sub_graph_data1_op_desc, ATTR_NAME_PARENT_NODE_INDEX, 0);
    AttrUtils::SetInt(sub_graph_data2_op_desc, ATTR_NAME_PARENT_NODE_INDEX, 1);
    AttrUtils::SetInt(sub_graph_data3_op_desc, ATTR_NAME_PARENT_NODE_INDEX, 2);

    OpDescPtr sub_graph_while_op_desc = CreateOpDesc("while", WHILE, 2, 2);
    NodePtr sub_graph_while_node = sub_graph->AddNode(sub_graph_while_op_desc);
    sub_sub_graph1->SetParentNode(sub_graph_while_node);
    sub_sub_graph2->SetParentNode(sub_graph_while_node);
    sub_graph_while_op_desc->AddSubgraphName("while_cond");
    sub_graph_while_op_desc->SetSubgraphInstanceName(0, "while_cond");
    sub_graph_while_op_desc->AddSubgraphName("while_body");
    sub_graph_while_op_desc->SetSubgraphInstanceName(1, "while_body");

    OpDescPtr sub_graph_matmul_op_desc = CreateOpDesc("matmul", MATMUL, 2, 1);
    NodePtr sub_graph_matmul_node = sub_graph->AddNode(sub_graph_matmul_op_desc);

    OpDescPtr sub_graph_output_op_desc = CreateOpDesc("output", NETOUTPUT, 1, 1);
    NodePtr sub_graph_output_node = sub_graph->AddNode(sub_graph_output_op_desc);

    GraphUtils::AddEdge(sub_graph_data1_node->GetOutDataAnchor(0), sub_graph_while_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sub_graph_data2_node->GetOutDataAnchor(0), sub_graph_while_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(sub_graph_data3_node->GetOutDataAnchor(0), sub_graph_matmul_node->GetInDataAnchor(0));
    GraphUtils::AddEdge(sub_graph_while_node->GetOutDataAnchor(0), sub_graph_matmul_node->GetInDataAnchor(1));
    GraphUtils::AddEdge(sub_graph_matmul_node->GetOutDataAnchor(0), sub_graph_output_node->GetInDataAnchor(0));

    sub_graph->SetGraphUnknownFlag(true);
    sub_graph->SetParentNode(partitioned_call_node);
    sub_graph->SetParentGraph(root_graph);
    root_graph->AddSubGraph(sub_graph);
  }

  OpDescPtr graph_data1_op_desc = CreateOpDesc("data1", DATA, 1, 1);
  OpDescPtr graph_data2_op_desc = CreateOpDesc("data2", DATA, 1, 1);
  OpDescPtr graph_data3_op_desc = CreateOpDesc("data3", DATA, 1, 1);
  NodePtr graph_data1_node = root_graph->AddNode(graph_data1_op_desc);
  NodePtr graph_data2_node = root_graph->AddNode(graph_data2_op_desc);
  NodePtr graph_data3_node = root_graph->AddNode(graph_data3_op_desc);
  AttrUtils::SetInt(graph_data1_op_desc, ATTR_NAME_INDEX, 0);
  AttrUtils::SetInt(graph_data2_op_desc, ATTR_NAME_INDEX, 1);
  AttrUtils::SetInt(graph_data3_op_desc, ATTR_NAME_INDEX, 2);
  GraphUtils::AddEdge(graph_data1_node->GetOutDataAnchor(0), partitioned_call_node->GetInDataAnchor(0));
  GraphUtils::AddEdge(graph_data2_node->GetOutDataAnchor(0), partitioned_call_node->GetInDataAnchor(1));
  GraphUtils::AddEdge(graph_data3_node->GetOutDataAnchor(0), partitioned_call_node->GetInDataAnchor(2));

  ComputeGraphPtr merged_graph = nullptr;
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(root_graph), SUCCESS);
  HybridModel hybrid_model(root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);
  EXPECT_EQ(hybrid_model_builder.UnfoldSubgraphs(root_graph, merged_graph), SUCCESS);
}

TEST_F(UtestGeHybrid, TestTaskContext) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;
  node_item->reuse_outputs[0] = 1;

  GraphExecutionContext execution_context;
  GraphItem graph_item;
  graph_item.node_items_.emplace_back(node_item.get());
  graph_item.total_inputs_ = 2;
  graph_item.total_outputs_ = 2;
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  float buffer[1];
  subgraph_context.all_outputs_[0] = TensorValue(buffer, sizeof(buffer));
  auto node_state = subgraph_context.GetNodeState(node_item.get());
  auto task_context = node_state->GetTaskContext();
  ASSERT_TRUE(task_context != nullptr);
  auto desc = task_context->MutableInputDesc(2);
  ASSERT_TRUE(desc == nullptr);
  desc = task_context->MutableOutputDesc(0);
  ASSERT_TRUE(desc != nullptr);
  ASSERT_EQ(desc->GetShape().GetDims(), shape.GetDims());

  desc = task_context->MutableInputDesc(0);
  ASSERT_TRUE(desc != nullptr);
  ASSERT_EQ(desc->GetShape().GetDims(), shape.GetDims());
  GeShape new_shape({8, 2});
  tensor_desc.SetShape(new_shape);
  task_context->UpdateInputDesc(1, tensor_desc);
  GeTensorDesc new_desc;
  ASSERT_EQ(task_context->GetInputDesc(1, new_desc), SUCCESS);
  ASSERT_EQ(new_desc.GetShape().GetDims(), new_shape.GetDims());
  TensorUtils::SetSize(tensor_desc, 128);
  ASSERT_EQ(task_context->AllocateOutput(0, tensor_desc, nullptr, nullptr), GRAPH_PARAM_INVALID);
  // 1 reuse 0; 0 allocated as netoutput, so 1 does
  ASSERT_EQ(task_context->AllocateOutput(1, tensor_desc, nullptr, nullptr), GRAPH_PARAM_INVALID);
}

TEST_F(UtestGeHybrid, hybrid_model_executor_update_args) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());

  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);

  GraphExecutionContext execution_context;
  FakeGraphItem graph_item(node);
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  auto node_state = subgraph_context.GetNodeState(graph_item.GetNodeItem());
  auto task_context = node_state->GetTaskContext();

  aicore_task->max_arg_count_ = 0;
  EXPECT_EQ(aicore_task->UpdateArgs(*task_context), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);
  aicore_task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(uintptr_t) * 2]);
  EXPECT_EQ(aicore_task->UpdateArgs(*task_context), SUCCESS);

  aicore_task->need_tiling_ = true;
  EXPECT_EQ(aicore_task->UpdateArgs(*task_context), ACL_ERROR_GE_MEMORY_OPERATE_FAILED);

  aicore_task->max_tiling_size_ = 64;
  ASSERT_EQ(aicore_task->UpdateArgs(*task_context), SUCCESS);

  hybrid::AtomicAddrCleanOpTask atomic_task;
  atomic_task.need_tiling_ = true;
  atomic_task.args_ = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(uintptr_t) * 2 + 64]);
  atomic_task.arg_base_ = reinterpret_cast<uintptr_t *>(atomic_task.args_.get());
  EXPECT_EQ(atomic_task.UpdateArgs(*task_context), SUCCESS);
  atomic_task.max_tiling_size_ = 64;
  atomic_task.args_size_ = sizeof(uintptr_t) * 2;
  EXPECT_EQ(atomic_task.UpdateArgs(*task_context), SUCCESS);
}

TEST_F(UtestGeHybrid, hybrid_model_executor_check_shape) {
  HybridModelExecutor::ExecuteArgs args;
  GeTensorDescPtr ge_tensor = make_shared<GeTensorDesc>(GeTensorDesc());
  vector<int64_t> dim = {2 , 3};
  ge_tensor->SetShape(GeShape(dim));
  args.input_desc.push_back(ge_tensor);

  // create node
  ge::ComputeGraphPtr graph = std::make_shared<ComputeGraph>("God");
  OpDescPtr op_desc = std::make_shared<OpDesc>("data", DATA);
  GeTensorDesc tensor_desc(GeShape({2, 3}));
  std::vector<std::pair<int64_t, int64_t>> shape_range({std::pair<int64_t, int64_t>(1, 3),
                                                       std::pair<int64_t, int64_t>(2, 4)});
  tensor_desc.SetShapeRange(shape_range);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);

  NodePtr node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> new_node;
  NodeItem::Create(node, new_node);
  new_node->is_dynamic = true;

  GraphItem graph_item;
  graph_item.input_nodes_.emplace_back(new_node.get());

  Status ret = HybridModelRtV1Executor::CheckInputShapeByShapeRange(&graph_item, args);
  ASSERT_EQ(ret, ge::SUCCESS);

  HybridModelExecutor::ExecuteArgs args1;
  ret = HybridModelRtV1Executor::CheckInputShapeByShapeRange(&graph_item, args1);
  ASSERT_EQ(ret, ge::INTERNAL_ERROR);

  HybridModelExecutor::ExecuteArgs args2;
  GeTensorDescPtr ge_tensor2 = make_shared<GeTensorDesc>(GeTensorDesc());
  vector<int64_t> dim2 = {-1 , 3};
  ge_tensor2->SetShape(GeShape(dim2));
  args2.input_desc.push_back(ge_tensor2);

  ret = HybridModelRtV1Executor::CheckInputShapeByShapeRange(&graph_item, args1);
  ASSERT_EQ(ret, ge::INTERNAL_ERROR);

  HybridModelExecutor::ExecuteArgs args3;
  ret = HybridModelRtV1Executor::CheckInputShapeByShapeRange(&graph_item, args3);
  ASSERT_EQ(ret, ge::INTERNAL_ERROR);
}

TEST_F(UtestGeHybrid, TestOptimizeDependenciesForConstInputs) {
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test");
  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  HybridModel model(root_model);
  model.root_graph_ = compute_graph;
  HybridModelBuilder builder(model);

  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  std::unique_ptr<NodeItem> const_node_item;
  {
    OpDescPtr const_op_desc = CreateOpDesc("Constant", "Const");
    const_op_desc->AddOutputDesc(tensor_desc);
    auto const_node = compute_graph->AddNode(const_op_desc);
    NodeItem::Create(const_node, const_node_item);
  }

  std::unique_ptr<NodeItem> non_const_node_item;
  {
    OpDescPtr op_desc = CreateOpDesc("Add", "Add");
    op_desc->AddOutputDesc(tensor_desc);
    auto const_node = compute_graph->AddNode(op_desc);
    NodeItem::Create(const_node, non_const_node_item);
  }

  std::unique_ptr<NodeItem> known_node_item;
  {
    OpDescPtr known_op_desc = CreateOpDesc("known", "PartitionedCall");
    known_op_desc->AddOutputDesc(tensor_desc);
    known_op_desc->AddOutputDesc(tensor_desc);
    auto known_node = compute_graph->AddNode(known_op_desc);
    NodeItem::Create(known_node, known_node_item);
  }

  std::unique_ptr<NodeItem> dst_node_item;
  {
    OpDescPtr known_op_desc = CreateOpDesc("SomeOp", "SomeOpType ");
    known_op_desc->AddOutputDesc(tensor_desc);
    known_op_desc->AddOutputDesc(tensor_desc);
    auto known_node = compute_graph->AddNode(known_op_desc);
    NodeItem::Create(known_node, dst_node_item);
  }

  float buffer[2 * 16];
  unique_ptr<TensorValue> tensor_value(new TensorValue(buffer, sizeof(buffer)));
  model.constant_tensors_[const_node_item->node] = std::move(tensor_value);

  // Case 1. connect to Const
  auto output_id = 1;
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(output_id, const_node_item.get());
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(0, non_const_node_item.get());
  dst_node_item->dependents_for_shape_inference.emplace_back(const_node_item->node);
  dst_node_item->dependents_for_shape_inference.emplace_back(non_const_node_item->node);

  ASSERT_EQ(builder.OptimizeDependenciesForConstantInputs(), SUCCESS);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference.size(), 1);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference[0], non_const_node_item->node);

  // Case 2. connect to known-subgraph, netoutput connect to Const
  builder.host_input_value_dependencies_.clear();
  dst_node_item->dependents_for_shape_inference.clear();

  builder.known_subgraph_constant_output_refs_[known_node_item.get()].emplace(output_id, const_node_item->node);
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(output_id, known_node_item.get());
  builder.host_input_value_dependencies_[dst_node_item.get()].emplace_back(0, non_const_node_item.get());

  dst_node_item->dependents_for_shape_inference.emplace_back(known_node_item->node);
  dst_node_item->dependents_for_shape_inference.emplace_back(non_const_node_item->node);

  ASSERT_EQ(builder.OptimizeDependenciesForConstantInputs(), SUCCESS);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference.size(), 1);
  ASSERT_EQ(dst_node_item->dependents_for_shape_inference[0], non_const_node_item->node);
}

TEST_F(UtestGeHybrid, test_key_for_kernel_bin) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  OpDesc op_desc("Sum", "Sum");
  EXPECT_EQ(aicore_task->GetKeyForTbeKernel(), OP_EXTATTR_NAME_TBE_KERNEL);
  EXPECT_EQ(aicore_task->GetKeyForTvmMagic(), TVM_ATTR_NAME_MAGIC);
  EXPECT_EQ(aicore_task->GetKeyForTvmMetaData(), TVM_ATTR_NAME_METADATA);
  EXPECT_EQ(aicore_task->GetKeyForKernelName(op_desc), "Sum_kernelname");

  auto atomic_task = std::unique_ptr<hybrid::AtomicAddrCleanOpTask>(new(std::nothrow)hybrid::AtomicAddrCleanOpTask());
  EXPECT_EQ(atomic_task->GetKeyForTbeKernel(), EXT_ATTR_ATOMIC_TBE_KERNEL);
  EXPECT_EQ(atomic_task->GetKeyForTvmMagic(), ATOMIC_ATTR_TVM_MAGIC);
  EXPECT_EQ(atomic_task->GetKeyForTvmMetaData(), ATOMIC_ATTR_TVM_METADATA);
  EXPECT_EQ(atomic_task->GetKeyForKernelName(op_desc), "Sum_atomic_kernelname");
}

TEST_F(UtestGeHybrid, test_op_type) {
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  aicore_task->op_type_ = "Add";
  EXPECT_EQ(aicore_task->GetOpType(), "Add");

  auto atomic_task = std::unique_ptr<hybrid::AtomicAddrCleanOpTask>(new(std::nothrow)hybrid::AtomicAddrCleanOpTask());
  EXPECT_EQ(atomic_task->GetOpType(), "DynamicAtomicAddrClean");
}

TEST_F(UtestGeHybrid, TestParseDependentInputNodesForHccl) {
  NodeExecutorManager::GetInstance().engine_mapping_.emplace("ops_kernel_info_hccl",
                                                             NodeExecutorManager::ExecutorType::HCCL);
  ComputeGraphPtr compute_graph = MakeShared<ComputeGraph>("test");

  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  auto node = compute_graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->node_id = 0;

  OpDescPtr op_desc_1 = CreateOpDesc("AllReduce", "AllReduce");
  op_desc_1->SetOpKernelLibName("ops_kernel_info_hccl");
  auto node_1 = compute_graph->AddNode(op_desc_1);
  std::unique_ptr<NodeItem> node_item_1;
  NodeItem::Create(node_1, node_item_1);
  node_item_1->node_id = 1;
  node->GetOutControlAnchor()->LinkTo(node_1->GetInControlAnchor());

  OpDescPtr op_desc_2 = CreateOpDesc("net_output", NETOUTPUT);
  auto node_2 = compute_graph->AddNode(op_desc_2);
  std::unique_ptr<NodeItem> node_item_2;
  NodeItem::Create(node_2, node_item_2);
  node_item_2->node_id = 2;
  node_1->GetOutControlAnchor()->LinkTo(node_2->GetInControlAnchor());

  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(compute_graph), SUCCESS);
  HybridModel model(root_model);
  model.root_graph_ = compute_graph;
  model.node_items_.emplace(node, std::move(node_item));
  model.node_items_.emplace(node_1, std::move(node_item_1));
  model.node_items_.emplace(node_2, std::move(node_item_2));

  HybridModelBuilder builder(model);
  ASSERT_EQ(builder.ParseDependentInputNodes(*model.node_items_[node_1]), SUCCESS);
  ASSERT_EQ(builder.ParseDependentInputNodes(*model.node_items_[node_2]), SUCCESS);
  ASSERT_FALSE(model.GetNodeItem(node)->has_observer);
  ASSERT_TRUE(model.GetNodeItem(node_1)->has_observer);
  ASSERT_EQ(model.node_items_[node_1]->dependents_for_execution.size(), 0);
  ASSERT_EQ(model.node_items_[node_2]->dependents_for_execution.size(), 1);
}

TEST_F(UtestGeHybrid, TestParseDependenciesSkipHostData) {
  // make graph
  ut::GraphBuilder graph_builder = ut::GraphBuilder("graph");
  auto data = graph_builder.AddNode("Data", "Data", 0, 1);
  auto netoutput = graph_builder.AddNode("Netoutput", "NetOutput", 1, 0);
  auto netoutput_opdesc = netoutput->GetOpDesc();
  netoutput_opdesc->UpdateInputName({{"Data", 0}});
  netoutput_opdesc->UpdateOutputName({{"y", 0}});
  netoutput_opdesc->AppendIrInput("Data", kIrInputRequired);
  netoutput_opdesc->AppendIrOutput("y", kIrOutputRequired);

  graph_builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = graph_builder.GetGraph();

  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(graph), SUCCESS);
  HybridModel model(root_model);
  HybridModelBuilder builder(model);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(netoutput, node_item);
  std::unique_ptr<NodeItem> node_item2;
  NodeItem::Create(data, node_item2);
  model.node_items_.emplace(data, std::move(node_item2));

  auto op_desc = netoutput->GetOpDesc();
  op_desc->SetOpInferDepends({"Data"});
  op_desc->impl_->input_name_idx_["Data"] = 0;
  auto data_desc = data->GetOpDesc();
  auto tensor = std::make_shared<GeTensor>();
  GeTensorDesc tensor_desc;
  AttrUtils::SetTensor(tensor_desc, "_value", tensor);
  AttrUtils::SetInt(tensor_desc, "_mem_type", 1); // skip d2h
  data_desc->AddInputDesc(tensor_desc);
  std::set<NodePtr> dependent_for_shape_inference;
  ASSERT_EQ(builder.ParseDependencies(*node_item, dependent_for_shape_inference), SUCCESS);
  EXPECT_EQ(dependent_for_shape_inference.empty(), true);
}

TEST_F(UtestGeHybrid, TestParseDependenciesSuccess) {
  // make graph
  ut::GraphBuilder graph_builder = ut::GraphBuilder("graph");
  auto data = graph_builder.AddNode("Data", "Data", 0, 1);
  auto netoutput = graph_builder.AddNode("Netoutput", "NetOutput", 1, 0);
  auto netoutput_opdesc = netoutput->GetOpDesc();
  netoutput_opdesc->UpdateInputName({{"Data", 0}});
  netoutput_opdesc->UpdateOutputName({{"y", 0}});
  netoutput_opdesc->AppendIrInput("Data", kIrInputRequired);
  netoutput_opdesc->AppendIrOutput("y", kIrOutputRequired);

  graph_builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = graph_builder.GetGraph();

  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(graph), SUCCESS);
  HybridModel model(root_model);
  HybridModelBuilder builder(model);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(netoutput, node_item);
  std::unique_ptr<NodeItem> node_item2;
  NodeItem::Create(data, node_item2);
  model.node_items_.emplace(data, std::move(node_item2));

  auto op_desc = netoutput->GetOpDesc();
  op_desc->SetOpInferDepends({"Data"});
  op_desc->impl_->input_name_idx_["Data"] = 0;
  auto data_desc = data->GetOpDesc();
  auto tensor = std::make_shared<GeTensor>();
  GeTensorDesc tensor_desc;
  AttrUtils::SetTensor(tensor_desc, "_value", tensor);
  AttrUtils::SetInt(tensor_desc, "_mem_type", 0); // need d2h
  data_desc->AddInputDesc(tensor_desc);
  std::set<NodePtr> dependent_for_shape_inference;
  ASSERT_EQ(builder.ParseDependencies(*node_item, dependent_for_shape_inference), SUCCESS);
  EXPECT_EQ(dependent_for_shape_inference.empty(), false);
}

TEST_F(UtestGeHybrid, TestParseDependenciesSuccessRT2) {
  // make graph
  ut::GraphBuilder graph_builder = ut::GraphBuilder("graph");
  auto data = graph_builder.AddNode("Data", "Data", 0, 1);
  auto relu = graph_builder.AddNode("relu", "Relu", 1, 1);
  auto netoutput = graph_builder.AddNode("Netoutput", "NetOutput", 1, 0);
  auto relu_opdesc = relu->GetOpDesc();
  relu_opdesc->UpdateInputName({{"Data", 0}});
  relu_opdesc->UpdateOutputName({{"y", 0}});
  relu_opdesc->AppendIrInput("Data", kIrInputRequired);
  relu_opdesc->AppendIrOutput("y", kIrOutputRequired);

  graph_builder.AddDataEdge(data, 0, relu, 0);
  graph_builder.AddDataEdge(relu, 0, netoutput, 0);
  auto graph = graph_builder.GetGraph();

  GeRootModelPtr root_model = MakeShared<ge::GeRootModel>();
  EXPECT_EQ(root_model->Initialize(graph), SUCCESS);
  HybridModel model(root_model);
  HybridModelBuilder builder(model);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(relu, node_item);
  std::unique_ptr<NodeItem> node_item2;
  NodeItem::Create(data, node_item2);
  model.node_items_.emplace(data, std::move(node_item2));

  IMPL_OP(Relu).InputsDataDependency({0}); // rt2 way
  auto data_desc = data->GetOpDesc();
  auto tensor = std::make_shared<GeTensor>();
  GeTensorDesc tensor_desc;
  AttrUtils::SetTensor(tensor_desc, "_value", tensor);
  AttrUtils::SetInt(tensor_desc, "_mem_type", 0); // need d2h
  data_desc->AddInputDesc(tensor_desc);
  std::set<NodePtr> dependent_for_shape_inference;
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistry(true);

  ASSERT_EQ(builder.ParseDependencies(*node_item, dependent_for_shape_inference), SUCCESS);
  EXPECT_EQ(dependent_for_shape_inference.empty(), false);
  gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry()->CreateOrGetOpImpl("Relu")->inputs_dependency = 0;
}

TEST_F(UtestGeHybrid, TestTaskExecuteAsync) {
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);

  GraphExecutionContext execution_context;
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  execution_context.model = &hybrid_model;

  FakeGraphItem fake_item(node);
  SubgraphContext subgraph_context(&fake_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  auto node_state = subgraph_context.GetNodeState(fake_item.GetNodeItem());
  auto task_context = *node_state->GetTaskContext();
  ASSERT_NE(BuildTaskUtils::GetTaskInfo(task_context), "");
  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  tasks.emplace_back(std::move(task1));
  AiCoreNodeTask node_task(std::move(tasks));
  ASSERT_EQ(node_task.ExecuteAsync(task_context, nullptr), SUCCESS);
}

TEST_F(UtestGeHybrid, TestHybridModelCheckHostMemOpt) {
  std::vector<NodePtr> node_with_hostmem;

  // create Node_item
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  // create HybridModel
  GraphExecutionContext execution_context;
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);

  // node_with_hostmem is empty, check result is false
  ASSERT_EQ(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem), false);

  node_with_hostmem.emplace_back(node);
  // node_item_ptr is null ,check result is false
  ASSERT_EQ(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem), false);

  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());
  constexpr char const *kAttrSupportDynamicShape = "support_dynamicshape";
  AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, true);
  task1->InitTilingInfo(*op_desc);
  task1->host_mem_input_data_offset_ = 1U;

  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  tasks.emplace_back(std::move(task1));
  node_item->kernel_task.reset(new (std::nothrow) AiCoreNodeTask(std::move(tasks)));
  hybrid_model.node_items_.emplace(node, std::move(node_item));
  // set kernel_task, check result is true
  ASSERT_EQ(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem), true);
  hybrid_model.SetNeedHostMemOpt(node_with_hostmem, true);

  // ai core task not support dynamic shape, check result is false
  std::unique_ptr<AiCoreOpTask> aicore_op_task(new AiCoreOpTask());
  AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, false);
  aicore_op_task->InitTilingInfo(*op_desc);

  std::vector<std::unique_ptr<AiCoreOpTask>> tasks_list;
  tasks_list.emplace_back(std::move(aicore_op_task));
  (hybrid_model.node_items_[node])->kernel_task.reset(new (std::nothrow) AiCoreNodeTask(std::move(tasks_list)));
  //hybrid_model.node_items_.emplace(node, std::move(node_item));
  ASSERT_EQ(hybrid_model.CheckHostMemInputOptimization(node_with_hostmem), false);
}

TEST_F(UtestGeHybrid, TestHybridModelSetHostMemInput) {
  std::vector<NodePtr> node_with_hostmem;

  // create Node_item
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  // create HybridModel
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);

  node_with_hostmem.emplace_back(node);

  std::unique_ptr<AiCoreOpTask> task1(new AiCoreOpTask());

  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  tasks.emplace_back(std::move(task1));
  node_item->kernel_task.reset(new (std::nothrow) AiCoreNodeTask(std::move(tasks)));
  hybrid_model.node_items_.emplace(node, std::move(node_item));
}

TEST_F(UtestGeHybrid, test_dynamic_singleop_CheckHostMemInputOptimization) {
  vector<pair<size_t, uint64_t>> inputs_size;
  inputs_size.emplace_back(0, 64);

  uintptr_t resource_id = 0;
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOpImpl dynamic_single_op(nullptr, resource_id, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = nullptr;
  data_buffer.length = 64;
  data_buffer.placement = 1U;
  input_buffers.emplace_back(data_buffer);

  // create Node_item
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = MakeShared<OpDesc>("name1", "type1");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);
  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);

  // create HybridModel
  GeRootModelPtr ge_root_model = std::make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  ge::hybrid::HybridModel *hybrid_model = new ge::hybrid::HybridModel(ge_root_model);
  dynamic_single_op.hybrid_model_.reset((ge::hybrid::HybridModel *)(hybrid_model));

  std::unique_ptr<ge::hybrid::AiCoreOpTask> aiCore_task1(new ge::hybrid::AiCoreOpTask());
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  tasks.emplace_back(std::move(aiCore_task1));
  tasks[0]->host_mem_input_data_offset_ = 1U;

  node_item->kernel_task.reset(new (std::nothrow) AiCoreNodeTask(std::move(tasks)));
  dynamic_single_op.hybrid_model_->node_items_.clear();
  dynamic_single_op.hybrid_model_->node_items_.emplace(node, std::move(node_item));

  dynamic_single_op.node_with_hostmem_ = {};

  ASSERT_EQ(dynamic_single_op.CheckHostMemInputOptimization(input_buffers), false);

  dynamic_single_op.node_with_hostmem_.emplace_back(node);
  ASSERT_EQ(dynamic_single_op.CheckHostMemInputOptimization(input_buffers), true);
  rtStreamDestroy(stream);
}

TEST_F(UtestGeHybrid, aicore_op_task_update_args_wit_host_input) {
  constexpr char const *kAttrSupportDynamicShape = "support_dynamicshape";
  constexpr char const *kAttrOpParamSize = "op_para_size";
  auto aicore_task = std::unique_ptr<hybrid::AiCoreOpTask>(new(std::nothrow)hybrid::AiCoreOpTask());
  auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = CreateOpDesc("Add", "Add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  AttrUtils::SetBool(op_desc, kAttrSupportDynamicShape, true);
  AttrUtils::SetInt(op_desc, kAttrOpParamSize, 1024);
  auto node = graph->AddNode(op_desc);

  size_t buffer_size = 64;
  void *buffer = new uint8_t[buffer_size];
  GraphExecutionContext execution_context;
  FakeGraphItem graph_item(node);
  SubgraphContext subgraph_context(&graph_item, &execution_context);
  ASSERT_EQ(subgraph_context.Init(), SUCCESS);
  subgraph_context.all_inputs_.resize(1);
  subgraph_context.all_outputs_.resize(1);
  auto node_state = subgraph_context.GetNodeState(graph_item.GetNodeItem());
  auto task_context = node_state->GetTaskContext();
  task_context->workspaces_.clear();
  //task_context->node_item_- = &node_item;

  TensorValue host_input(buffer, buffer_size);
  task_context->inputs_start_ = &host_input;
  aicore_task->output_indices_to_skip_.clear();
  aicore_task->shape_buffer_ = nullptr;
  aicore_task->max_tiling_size_ = 64;
  aicore_task->need_tiling_ = true;
  uint32_t args_size_without_tiling_ = 64;
  aicore_task->args_ex_.tilingAddrOffset = args_size_without_tiling_ - sizeof(void *);
  aicore_task->args_ex_.tilingDataOffset = args_size_without_tiling_;
  aicore_task->max_arg_count_ = 1;
  aicore_task->offset_ = 8;
  aicore_task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[sizeof(uintptr_t) * 2]);
  EXPECT_EQ(task_context->NumInputs(), 1);
  EXPECT_EQ(aicore_task->UpdateArgs(*task_context), SUCCESS);
  delete[] reinterpret_cast<uint8_t*>(buffer);
}

TEST_F(UtestGeHybrid, TestHybridDavinciModelMethods) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  auto model = HybridDavinciModel::Create(ge_root_model);
  ASSERT_FALSE(model->GetRunningFlag());
  auto listener_ptr = make_shared<DModelListener>();
  model->SetListener(listener_ptr);
  model->SetDeviceId(5);
  model->SetModelId(1);
  model->SetModelDescVersion(false);
  ASSERT_EQ(model->GetDeviceId(), 5);
  model->SetOmName("test_name");
  std::vector<std::vector<int64_t>> batch_info;
  int32_t dynamic_type = 0;
  ASSERT_EQ(model->GetDynamicBatchInfo(batch_info, dynamic_type), SUCCESS);
  RunAsyncCallbackV2 callback = nullptr;
  ASSERT_EQ(model->SetRunAsyncListenerCallback(callback), PARAM_INVALID);
  std::vector<std::string> user_input_shape_order{"test"};
  model->GetUserDesignateShapeOrder(user_input_shape_order);
  ASSERT_EQ(user_input_shape_order.size(), 0);
}

TEST_F(UtestGeHybrid, index_taskdefs_no_task) {
  GeModelPtr ge_model = make_shared<GeModel>();

  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(graph), SUCCESS);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  HybridModelBuilder hybrid_model_builder(hybrid_model);

  ASSERT_EQ(hybrid_model_builder.IndexTaskDefs(graph, ge_model), SUCCESS);
}
} // namespace ge
