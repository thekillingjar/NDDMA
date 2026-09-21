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
#include <vector>

#include "runtime/rt.h"

#include "macro_utils/dt_public_scope.h"
#include "hybrid/model/graph_item.h"
#include "hybrid/model/node_item.h"
#include "hybrid/node_executor/aicore/aicore_op_task.h"
#include "hybrid/node_executor/aicore/aicore_node_executor.h"
#include "single_op/single_op.h"
#include "single_op/single_op_impl.h"
#include "single_op/single_op_manager.h"
#include "single_op/task/build_task_utils.h"
#include "common/dump/dump_manager.h"
#include "runtime/subscriber/global_dumper.h"
#include "common/global_variables/diagnose_switch.h"
#include "framework/ge_runtime_stub/include/stub/gert_runtime_stub.h"
#include "framework/runtime/mem_allocator.h"
#include "macro_utils/dt_public_unscope.h"
#include "framework/ge_runtime_stub/include/faker/fake_allocator.h"
#include "depends/profiler/src/dump_stub.h"

using namespace std;
using namespace ge;
using namespace hybrid;
class UtestSingleOp : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
  ObjectPool<GeTensor> tensor_pool_;
};

TEST_F(UtestSingleOp, test_dynamic_singleop_execute_async) {
  uintptr_t resource_id = 0;
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOp dynamic_single_op(&tensor_pool_, resource_id, &stream_mu, stream);

  vector<int64_t> dims_vec_0 = {2};
  vector<GeTensorDesc> input_desc;
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  // input data from device
  AttrUtils::SetInt(tensor_desc_0, ATTR_NAME_PLACEMENT, 0);
  input_desc.emplace_back(tensor_desc_0);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  input_buffers.emplace_back(data_buffer);

  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers;

  // UpdateRunInfo failed
  EXPECT_EQ(dynamic_single_op.ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), ACL_ERROR_GE_PARAM_INVALID);
  rtStreamDestroy(stream);
  delete[] (char*)data_buffer.data;
}

TEST_F(UtestSingleOp, test_dynamic_singleop_execute_async1) {
  uintptr_t resource_id = 0;
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOpImpl dynamic_single_op(&tensor_pool_, resource_id, &stream_mu, stream);
  dynamic_single_op.num_inputs_ = 1;

  vector<int64_t> dims_vec_0 = {2};
  vector<GeTensorDesc> input_desc;
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  // input data from host
  AttrUtils::SetInt(tensor_desc_0, ATTR_NAME_PLACEMENT, 1);
  input_desc.emplace_back(tensor_desc_0);

  int64_t input_size = 0;
  EXPECT_EQ(TensorUtils::GetTensorMemorySizeInBytes(tensor_desc_0, input_size), SUCCESS);
  EXPECT_EQ(input_size, 64);
  EXPECT_NE(SingleOpManager::GetInstance().GetResource(resource_id, stream), nullptr);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  input_buffers.emplace_back(data_buffer);

  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers;

  auto *tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;

  dynamic_single_op.op_task_.reset((OpTask *)(tbe_task));

  OpDescPtr desc_ptr = MakeShared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  dynamic_single_op.op_task_->op_desc_ = desc_ptr;
  // UpdateRunInfo failed
  EXPECT_EQ(dynamic_single_op.ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), ACL_ERROR_GE_PARAM_INVALID);
  rtStreamDestroy(stream);
  delete[] (char*)data_buffer.data;
}

TEST_F(UtestSingleOp, test_stream_resource) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  uint8_t *ttt = new uint8_t[10];
  res->weight_list_.push_back(ttt);

  res->stream_ = ValueToPtr(11);
  EXPECT_EQ(res->GetStream(), res->stream_);

  EXPECT_NE(res->MallocWeight("test_purpose", 10), nullptr);
  EXPECT_EQ(res->MallocWeight("test_purpose", 123), nullptr);

  string model_data_str = "123456789";
  ModelData model_data;
  model_data.model_data = (void *)(const_cast<char *>(model_data_str.c_str()));
  model_data.model_len = model_data_str.size();
  DynamicSingleOp *single_op;

  ObjectPool<GeTensor> *tensor_pool = nullptr;
  uintptr_t resource_id = 2;
  std::mutex *stream_mutex = nullptr;
  rtStream_t stream = nullptr;
  res->dynamic_op_map_[10] = std::unique_ptr<DynamicSingleOp>(new DynamicSingleOp(tensor_pool, resource_id, stream_mutex, stream));

  EXPECT_EQ(res->BuildDynamicOperator(model_data, &single_op, 10), SUCCESS);
  EXPECT_NE(res->BuildDynamicOperator(model_data, &single_op, 9), SUCCESS);
  delete res;
}

TEST_F(UtestSingleOp, test_singleop_execute_async1) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  data_buffer.placement = kHostMemType;
  input_buffers.emplace_back(data_buffer);
  vector<DataBuffer> output_buffers;

  single_op.input_sizes_.emplace_back(4);
  SingleOpModelParam model_params;
  single_op.model_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  single_op.args_.resize(1);

  auto tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  EXPECT_EQ(op_desc->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_NE(BuildTaskUtils::GetTaskInfo(op_desc), "");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->args_ = MakeUnique<uint8_t[]>(8 + kMaxHostMemInputLen);
  tbe_task->arg_size_ = 8 + kMaxHostMemInputLen;
  tbe_task->node_ = node;
  tbe_task->op_desc_ = op_desc;
  tbe_task->args_ex_.args = tbe_task->args_.get();
  tbe_task->args_ex_.argsSize = tbe_task->arg_size_;
  tbe_task->args_item_offsets_.host_input_data_offset = 8;
  tbe_task->extend_args_for_host_input_ = true;
  tbe_task->op_ = std::make_unique<Operator>(ge::OpDescUtils::CreateOperatorFromNode(node));
  std::unique_ptr<OpTask> op_task;
  op_task.reset(tbe_task);
  single_op.tasks_.push_back(std::move(op_task));
  EXPECT_EQ(single_op.model_param_->runtime_param.mem_base, 0U);
  EXPECT_EQ(single_op.ExecuteAsync(input_buffers, output_buffers), SUCCESS);
  EXPECT_EQ(tbe_task->args_ex_.args, (void *)tbe_task->args_.get());
  EXPECT_TRUE(tbe_task->args_ex_.argsSize == tbe_task->arg_size_);
  EXPECT_EQ(tbe_task->args_ex_.hostInputInfoNum, 1);
  EXPECT_NE(tbe_task->args_ex_.hostInputInfoPtr, nullptr);
  if (tbe_task->args_ex_.hostInputInfoPtr != nullptr) {
    EXPECT_EQ(tbe_task->args_ex_.hostInputInfoPtr[0].addrOffset, 0);
    EXPECT_EQ(tbe_task->args_ex_.hostInputInfoPtr[0].dataOffset, 8);
  }
  delete[] (char*)data_buffer.data;
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOp, test_singleop_aicore_host_mem_input) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  uint64_t inputValue = 54110;
  data_buffer.data = &inputValue;
  data_buffer.length = sizeof(inputValue);
  data_buffer.placement = 1;
  input_buffers.emplace_back(data_buffer);
  vector<DataBuffer> output_buffers;

  single_op.input_sizes_.emplace_back(sizeof(inputValue));
  SingleOpModelParam model_params;
  single_op.model_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  single_op.args_.resize(1);

  auto tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  EXPECT_EQ(op_desc->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_NE(BuildTaskUtils::GetTaskInfo(op_desc), "");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;
  tbe_task->op_desc_ = op_desc;
  tbe_task->arg_size_ = sizeof(inputValue) + kMaxHostMemInputLen;
  tbe_task->args_.reset(new (std::nothrow) uint8_t[tbe_task->arg_size_]);
  tbe_task->args_ex_.args = tbe_task->args_.get();
  tbe_task->args_ex_.argsSize = tbe_task->arg_size_;
  tbe_task->arg_index_.emplace_back(0U);
  tbe_task->extend_args_for_host_input_ = true;
  tbe_task->args_item_offsets_.host_input_data_offset = sizeof(inputValue);
  tbe_task->op_ = std::make_unique<Operator>(ge::OpDescUtils::CreateOperatorFromNode(node));
  std::unique_ptr<OpTask> op_task;
  op_task.reset(tbe_task);
  single_op.tasks_.push_back(std::move(op_task));
  EXPECT_EQ(single_op.model_param_->runtime_param.mem_base, 0U);
  EXPECT_EQ(single_op.ExecuteAsync(input_buffers, output_buffers), SUCCESS);
  EXPECT_EQ(tbe_task->need_host_mem_opt_, true);

  if (tbe_task->args_ex_.hostInputInfoPtr != nullptr) {
    // check host memory input data in args
    void *hostValueAddr = ValueToPtr(PtrToValue(tbe_task->args_ex_.args) +
                                     tbe_task->args_ex_.hostInputInfoPtr->dataOffset);
    uint64_t hostValueInArgs = *(PtrToPtr<void, uint64_t>(hostValueAddr));
    EXPECT_EQ(hostValueInArgs, inputValue);
  }

  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOp, test_singleop_aicpu_tf_kernel_host_mem_input) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  uint64_t inputValue = 54110;
  data_buffer.data = &inputValue;
  data_buffer.length = sizeof(inputValue);
  data_buffer.placement = 1;
  input_buffers.emplace_back(data_buffer);
  vector<DataBuffer> output_buffers;

  single_op.input_sizes_.emplace_back(sizeof(inputValue));
  SingleOpModelParam model_params;
  single_op.model_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  single_op.args_.resize(1);

  auto task = new (std::nothrow) AiCpuTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Neg", NEG);
  EXPECT_EQ(op_desc->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_NE(BuildTaskUtils::GetTaskInfo(op_desc), "");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  task->op_desc_ = op_desc;
  task->arg_size_ = sizeof(inputValue) + 64U;
  task->io_addr_host_ = {data_buffer.data};
  task->host_mem_input_data_offset_ = task->io_addr_host_.size() * sizeof(void *);
  task->extend_args_for_host_input_ = true;
  for (size_t i = 0; i < kMaxHostMemInputLen / sizeof(void *); i++) {
    task->io_addr_host_.push_back(nullptr);
  }
  task->io_addr_size_ = task->io_addr_host_.size() * sizeof(void *);
  task->io_addr_ = new(std::nothrow) uint8_t[task->io_addr_size_];

  std::unique_ptr<OpTask> op_task;
  op_task.reset(task);
  single_op.tasks_.push_back(std::move(op_task));
  EXPECT_EQ(single_op.model_param_->runtime_param.mem_base, 0U);
  EXPECT_EQ(single_op.ExecuteAsync(input_buffers, output_buffers), SUCCESS);
  EXPECT_EQ(task->need_host_mem_opt_, true);

  // check host memory input data in args
  void *hostValueAddr = ValueToPtr(PtrToValue(task->io_addr_host_.data()) + sizeof(inputValue));
  uint64_t hostValueInArgs = *(reinterpret_cast<uint64_t *>(hostValueAddr));
  EXPECT_EQ(hostValueInArgs, inputValue);

  // check host memory input args
  hostValueAddr = ValueToPtr(PtrToValue(task->io_addr_) + sizeof(inputValue));
  EXPECT_EQ(task->io_addr_host_[0U], hostValueAddr);
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOp, test_singleop_aicpu_host_mem_input) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);

  // input
  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer0;
  data_buffer0.length = 24U;
  data_buffer0.placement = kHostMemType;
  ge::DataBuffer data_buffer1;
  data_buffer1.length = 8;
  data_buffer1.placement = kHostMemType;
  input_buffers.emplace_back(data_buffer0);
  input_buffers.emplace_back(data_buffer1);

  auto tf_aicpu_task = new (std::nothrow) AiCpuTask();
  tf_aicpu_task->extend_args_for_host_input_ = true;
  single_op.tasks_.emplace_back(tf_aicpu_task);
  auto aicpu_task = new (std::nothrow) AiCpuCCTask();
  single_op.tasks_.emplace_back(aicpu_task);
  aicpu_task->extend_args_for_host_input_ = true;
  auto tbe_task = new (std::nothrow) TbeOpTask();
  tbe_task->extend_args_for_host_input_ = true;
  single_op.tasks_.emplace_back(tbe_task);

  EXPECT_TRUE(single_op.CheckHostMemInputOptimization(input_buffers));
  EXPECT_TRUE(tf_aicpu_task->need_host_mem_opt_);
  EXPECT_TRUE(aicpu_task->need_host_mem_opt_);
  EXPECT_TRUE(tbe_task->need_host_mem_opt_);

  // host mem size is too big
  input_buffers[0].length = kMaxHostMemInputLen;
  EXPECT_FALSE(single_op.CheckHostMemInputOptimization(input_buffers));
  EXPECT_FALSE(tf_aicpu_task->need_host_mem_opt_);
  EXPECT_FALSE(aicpu_task->need_host_mem_opt_);
  EXPECT_FALSE(tbe_task->need_host_mem_opt_);

  // dynamic, test reentry, total host mem input size is less than 128
  input_buffers[0].length = 8;
  EXPECT_TRUE(single_op.CheckHostMemInputOptimization(input_buffers));
  EXPECT_TRUE(tf_aicpu_task->need_host_mem_opt_);
  EXPECT_TRUE(aicpu_task->need_host_mem_opt_);
  EXPECT_TRUE(tbe_task->need_host_mem_opt_);

  // AtomicAddrCleanOpTask not support host mem input optimize
  auto atomic_task = new (nothrow) ge::AtomicAddrCleanOpTask();
  single_op.tasks_.emplace_back(atomic_task);
  EXPECT_FALSE(single_op.CheckHostMemInputOptimization(input_buffers));

  // tasks_ is empty
  single_op.tasks_.clear();
  EXPECT_FALSE(single_op.CheckHostMemInputOptimization(input_buffers));

  // task is null
  single_op.tasks_.emplace_back(nullptr);
  EXPECT_FALSE(single_op.CheckHostMemInputOptimization(input_buffers));
  rtStreamDestroy(stream);
  delete res;
}

namespace {
GraphItem *MakeGraphItem() {
  static std::vector<std::unique_ptr<NodeItem>> vec_ptr;
  const auto graph = make_shared<ComputeGraph>("graph");
  OpDescPtr op_desc = MakeShared<OpDesc>("add", "add");
  GeShape shape({2, 16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddOutputDesc(tensor_desc);
  const auto node = graph->AddNode(op_desc);

  std::unique_ptr<NodeItem> node_item;
  NodeItem::Create(node, node_item);
  node_item->input_start = 0;
  node_item->output_start = 0;

  GraphItem *graph_item = new GraphItem;
  graph_item->node_items_.emplace_back(node_item.get());
  graph_item->total_inputs_ = node->GetAllInAnchors().size();
  graph_item->total_outputs_ = node->GetAllOutAnchors().size();
  vec_ptr.emplace_back(std::move(node_item));
  return graph_item;
}
}  // namespace

TEST_F(UtestSingleOp, test_hybrid_check_host_mem_input_opt) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOpImpl single_op(&tensor_pool_, 0, &stream_mu, stream);
  std::shared_ptr<ge::GeRootModel> root_model = ge::MakeShared<ge::GeRootModel>();
  single_op.hybrid_model_.reset(new (std::nothrow)hybrid::HybridModel(root_model));
  single_op.hybrid_model_->root_graph_item_.reset(MakeGraphItem());
  single_op.hybrid_model_executor_.
      reset(new (std::nothrow)hybrid::HybridModelRtV1Executor(single_op.hybrid_model_.get(), 0, stream));
  EXPECT_EQ(single_op.hybrid_model_executor_->Init(), SUCCESS);

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

  // input
  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer0;
  data_buffer0.length = 24U;
  data_buffer0.placement = kHostMemType;
  ge::DataBuffer data_buffer1;
  data_buffer1.length = 8;
  data_buffer1.placement = kHostMemType;
  input_buffers.emplace_back(data_buffer0);
  input_buffers.emplace_back(data_buffer1);

  // node_with_hostmem_ is empty
  ASSERT_FALSE(single_op.CheckHostMemInputOptimization(input_buffers));

  // node item is null
  single_op.node_with_hostmem_.emplace_back(node);
  ASSERT_FALSE(single_op.CheckHostMemInputOptimization(input_buffers));

  // return true
  auto aicore_task1 =  new (std::nothrow) ge::hybrid::AiCoreOpTask();
  aicore_task1->host_mem_input_data_offset_ = 1U;
  std::vector<std::unique_ptr<AiCoreOpTask>> tasks;
  tasks.emplace_back(aicore_task1);
  node_item->kernel_task.reset(new (std::nothrow) AiCoreNodeTask(std::move(tasks)));
  single_op.hybrid_model_->node_items_.clear();
  single_op.hybrid_model_->node_items_.emplace(node, std::move(node_item));
  ASSERT_TRUE(single_op.CheckHostMemInputOptimization(input_buffers));
  ASSERT_TRUE(aicore_task1->need_host_mem_opt_);
  rtStreamDestroy(stream);
  delete res;
}

TEST_F(UtestSingleOp, test_singleop_execute_async2) {
  ge::diagnoseSwitch::EnableProfiling({gert::ProfilingType::kMemory});
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOpImpl single_op(res, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  data_buffer.placement = 1;
  input_buffers.emplace_back(data_buffer);
  vector<DataBuffer> output_buffers;

  single_op.input_sizes_.emplace_back(4);
  SingleOpModelParam model_params;
  single_op.model_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  single_op.args_.resize(1);

  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NHWC, DT_UINT64);
  single_op.inputs_desc_.emplace_back(tensor_desc);
  std::shared_ptr<ge::GeRootModel> root_model = ge::MakeShared<ge::GeRootModel>();
  EXPECT_EQ(single_op.model_param_->runtime_param.mem_base, 0U);
  EXPECT_EQ(single_op.tasks_.size(), 0);
  single_op.model_param_->runtime_param.mem_size = 1024;
  FakeAllocator fake_allocator{};
  single_op.stream_resource_->SetAllocator(&fake_allocator);
  GeTensorDesc tensor;
  int64_t storage_format_val = static_cast<Format>(FORMAT_NCHW);
  AttrUtils::SetInt(tensor, "storage_format", storage_format_val);
  std::vector<int64_t> storage_shape{1, 1, 1, 1};
  AttrUtils::SetListInt(tensor, "storage_shape", storage_shape);
  single_op.inputs_desc_.emplace_back(tensor);
  EXPECT_EQ(single_op.ExecuteAsync(input_buffers, output_buffers), SUCCESS);
  ge::diagnoseSwitch::DisableProfiling();
  rtStreamDestroy(stream);
  delete[] (char*)data_buffer.data;
  delete res;
}

TEST_F(UtestSingleOp, test_set_host_mem) {
  std::mutex stream_mu_;
  DynamicSingleOpImpl single_op(&tensor_pool_, 0, &stream_mu_, nullptr);
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  std::shared_ptr<ge::GeRootModel> root_model = ge::MakeShared<ge::GeRootModel>();
  single_op.hybrid_model_.reset(new (std::nothrow)hybrid::HybridModel(root_model));
  single_op.hybrid_model_->root_graph_item_.reset(MakeGraphItem());
  single_op.hybrid_model_executor_.
      reset(new (std::nothrow)hybrid::HybridModelRtV1Executor(single_op.hybrid_model_.get(), 0, stream));
  EXPECT_EQ(single_op.hybrid_model_executor_->Init(), SUCCESS);

  vector<DataBuffer> input_buffers;
  DataBuffer data_buffer;
  input_buffers.emplace_back(data_buffer);

  vector<GeTensorDesc> input_descs;
  GeTensorDesc tensor_desc1;
  input_descs.emplace_back(tensor_desc1);

  single_op.hostmem_node_id_map_.emplace(0, 0);
  EXPECT_EQ(single_op.SetHostTensorValue(input_descs, input_buffers), SUCCESS);
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOp, dynamic_single_CheckHostMemInputOptimization) {
  uintptr_t resource_id = 0;
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOpImpl dynamic_single_op(&tensor_pool_, resource_id, &stream_mu, stream);
  dynamic_single_op.num_inputs_ = 1;

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[64];
  data_buffer.length = 64;
  data_buffer.placement = 1;

  input_buffers.emplace_back(data_buffer);

  auto tbe_task = new (nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;
  tbe_task->need_tiling_ = true;
  tbe_task->need_host_mem_opt_ = false;
  tbe_task->extend_args_for_host_input_ = true;

  dynamic_single_op.hybrid_model_ = nullptr;
  // op_task_ is null
  bool flag = dynamic_single_op.CheckHostMemInputOptimization(input_buffers);
  EXPECT_FALSE(flag);

  dynamic_single_op.op_task_.reset(tbe_task);
  flag = dynamic_single_op.CheckHostMemInputOptimization(input_buffers);
  EXPECT_TRUE(flag);
  EXPECT_TRUE(tbe_task->need_host_mem_opt_);

  // input_buffers is too large
  input_buffers[0U].length = kMaxHostMemInputLen + 1U;
  flag = dynamic_single_op.CheckHostMemInputOptimization(input_buffers);
  EXPECT_FALSE(flag);

  auto aicpu_task = new (nothrow) AiCpuCCTask();
  aicpu_task->extend_args_for_host_input_ = true;
  dynamic_single_op.op_task_.reset(aicpu_task);
  input_buffers[0U].length = kMaxHostMemInputLen;
  // aicpu task, true
  flag = dynamic_single_op.CheckHostMemInputOptimization(input_buffers);
  EXPECT_TRUE(flag);
  EXPECT_TRUE(aicpu_task->need_host_mem_opt_);

  // input_buffers host mem length is zero
  input_buffers[0U].placement = 0U;
  flag = dynamic_single_op.CheckHostMemInputOptimization(input_buffers);
  EXPECT_FALSE(flag);

  auto atomic_task = new (nothrow) ge::AtomicAddrCleanOpTask();
  dynamic_single_op.op_task_.reset(atomic_task);
  input_buffers[0U].length = 41U;
  input_buffers[0U].placement = 1U;
  // op type does not support
  flag = dynamic_single_op.CheckHostMemInputOptimization(input_buffers);
  EXPECT_FALSE(flag);
  EXPECT_FALSE(atomic_task->need_host_mem_opt_);
  delete[] (char *)data_buffer.data;
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOp, test_dynamic_singleop_execute_async_with_host_input) {
  uintptr_t resource_id = 0;
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOpImpl dynamic_single_op(&tensor_pool_, resource_id, &stream_mu, stream);
  dynamic_single_op.num_inputs_ = 1;

  vector<int64_t> dims_vec_0 = {2};
  vector<GeTensorDesc> input_desc;
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  // input data from host
  AttrUtils::SetInt(tensor_desc_0, ATTR_NAME_PLACEMENT, 1);
  input_desc.emplace_back(tensor_desc_0);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[64];
  data_buffer.length = 64;
  data_buffer.placement = 1;

  input_buffers.emplace_back(data_buffer);

  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers;

  auto *tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;
  tbe_task->need_tiling_ = true;
  tbe_task->need_host_mem_opt_ = false;
  tbe_task->extend_args_for_host_input_ = true;

  dynamic_single_op.hybrid_model_ = nullptr;
  dynamic_single_op.op_task_.reset((OpTask *)(tbe_task));

  OpDescPtr desc_ptr = MakeShared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  dynamic_single_op.op_task_->op_desc_ = desc_ptr;
  dynamic_single_op.ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers);
  EXPECT_EQ(tbe_task->need_host_mem_opt_, true);
  ge::DataBuffer data_buffer_long;

  delete [] static_cast<char *>(data_buffer.data);
  data_buffer.length = kMaxHostMemInputLen + 1;
  data_buffer.data = new char[kMaxHostMemInputLen + 1];
  data_buffer.placement = 1;
  input_buffers[0] = data_buffer;
  tbe_task->need_host_mem_opt_ = false;
  dynamic_single_op.ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers);
  EXPECT_EQ(tbe_task->need_host_mem_opt_, false);
  delete[] (char *)data_buffer.data;
  rtStreamDestroy(stream);
}

TEST_F(UtestSingleOp, test_OpenDump) {
  auto *tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;
  tbe_task->op_desc_ = op_desc;

  DumpManager::GetInstance().dump_properties_map_[kInferSessionId].SetDumpOpSwitch("on");

  EXPECT_NE(tbe_task->OpenDump(0), SUCCESS);

  tbe_task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[24]());
  tbe_task->arg_size_ = 24;
  tbe_task->max_tiling_size_ = 0;
  tbe_task->need_tiling_ = false;
  tbe_task->arg_num_ = 3;

  EXPECT_EQ(tbe_task->OpenDump(0), SUCCESS);
  delete tbe_task;
}

TEST_F(UtestSingleOp, test_SetStubFunc) {
  auto *tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;
  tbe_task->op_desc_ = op_desc;

  tbe_task->SetStubFunc("test", nullptr);

  EXPECT_EQ(tbe_task->stub_name_, "test");
  EXPECT_EQ(tbe_task->stub_func_, nullptr);
  EXPECT_EQ(tbe_task->task_name_, "test");
  delete tbe_task;
}

TEST_F(UtestSingleOp, test_OpTask_related) {
  auto *task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);

  task->node_ = node;
  task->op_desc_ = op_desc;

  EXPECT_EQ(task->OpTask::UpdateRunInfo(), UNSUPPORTED);
  std::vector<GeTensorDesc> output_desc;
  std::vector<DataBuffer> output_buffers;
  rtStream_t stream = nullptr;
  EXPECT_EQ(task->OpTask::LaunchKernel(
      std::vector<GeTensorDesc>(), std::vector<DataBuffer>(), output_desc, output_buffers, stream), UNSUPPORTED);
  EXPECT_EQ(task->OpTask::GetTaskType(), kTaskTypeInvalid);
  delete task;
}

TEST_F(UtestSingleOp, test_UpdateArgTable) {
  auto *tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;
  tbe_task->op_desc_ = op_desc;

  SingleOpModelParam param;
  param.runtime_param.mem_size = 1000;

  tbe_task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[24]());
  tbe_task->arg_size_ = 24;
  tbe_task->max_tiling_size_ = 0;
  tbe_task->need_tiling_ = false;

  tbe_task->sm_desc_ = malloc(10);

  EXPECT_EQ(tbe_task->UpdateArgTable(param), SUCCESS);

  tbe_task->arg_size_ = 16;
  EXPECT_NE(tbe_task->UpdateArgTable(param), ACL_ERROR_GE_INTERNAL_ERROR);   // ??

  delete tbe_task;
}

// TEST_F(UtestSingleOp, test_OpTask_SetRuntimeContext) {
//   ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
//   ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
//   ge::NodePtr node = graph->AddNode(op_desc);
//   auto *task = new (std::nothrow) TbeOpTask(node);

//   OpTask *base = dynamic_cast<OpTask *>(task);

//   RuntimeInferenceContext *context = nullptr;
//   task->SetRuntimeContext(context);
//   //EXPECT_EQ(base->op_->operator_impl_.runtime_context_, nullptr);
//   delete task;
// }

TEST_F(UtestSingleOp, test_TbeOpTask_LaunchKernelFail) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) TbeOpTask(node);

  task->block_dim_ = 99;
  task->args_ex_.args = task->args_.get();
  task->args_ex_.argsSize = task->arg_size_;
  task->SetOpDesc(op_desc);

  rtStream_t stream = nullptr;
  EXPECT_EQ(task->LaunchKernel(stream), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_TbeOpTask_AllocateWorkspaces) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  EXPECT_NO_THROW(
    auto *task = new (std::nothrow) TbeOpTask(node);
    task->stream_resource_ = new StreamResource(1);
    task->stream_resource_->allocator_ = &task->stream_resource_->internal_allocator_;
    delete task->stream_resource_;
    delete task;
  );
}

TEST_F(UtestSingleOp, test_TbeOpTask_CheckAndExecuteAtomic) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) TbeOpTask(node);
  task->node_ = node;
  task->op_desc_ = op_desc;

  std::vector<GeTensorDesc> input_desc;
  std::vector<DataBuffer> input_buffers;
  std::vector<GeTensorDesc> output_desc;
  std::vector<DataBuffer> output_buffers;
  rtStream_t stream = nullptr;

  task->clear_atomic_ = true;
  task->atomic_task_ = std::unique_ptr<ge::AtomicAddrCleanOpTask>(new ge::AtomicAddrCleanOpTask());
  ASSERT_NE(task->atomic_task_, nullptr);
  task->atomic_task_->node_ = node;
  task->run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(task->run_info_, nullptr);
  task->atomic_task_->run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(task->atomic_task_->run_info_, nullptr);

  EXPECT_NE(task->CheckAndExecuteAtomic(input_desc, input_buffers, output_desc, output_buffers, stream), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_AtomicAddrCleanOpTask_UpdateNodeByShape) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) ge::AtomicAddrCleanOpTask(node);

  EXPECT_EQ(task->UpdateNodeByShape(std::vector<GeTensorDesc>(), std::vector<GeTensorDesc>()), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuBaseTask_SetInputConst) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();
  task->op_desc_ = op_desc;

  EXPECT_EQ(task->SetInputConst(), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuBaseTask_UpdateExtInfo) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();
  task->op_desc_ = op_desc;
  task->aicpu_ext_handle_ = std::unique_ptr<ge::hybrid::AicpuExtInfoHandler>(
      new ge::hybrid::AicpuExtInfoHandler("Mul", 2, 1, DEPEND_IN_SHAPE));
  std::unique_ptr<AicpuShapeAndType> input_shape_and_type =
    std::unique_ptr<AicpuShapeAndType>(new AicpuShapeAndType());
  input_shape_and_type->type = 3;
  input_shape_and_type->dims[0] = 4;
  task->aicpu_ext_handle_->input_shape_and_type_.push_back(input_shape_and_type.get());
  task->aicpu_ext_handle_->input_shape_and_type_.push_back(input_shape_and_type.get());
  task->aicpu_ext_handle_->output_shape_and_type_.push_back(input_shape_and_type.get());

  std::vector<GeTensorDesc> input_desc = {GeTensorDesc(), GeTensorDesc()};
  std::vector<GeTensorDesc> output_desc = {GeTensorDesc()};
  rtStream_t stream = nullptr;
  EXPECT_EQ(task->UpdateExtInfo(input_desc, output_desc, stream), SUCCESS);

  task->num_inputs_ = 2;
  task->num_outputs_ = 1;
  task->input_is_const_ = std::vector<int8_t>({1});
  EXPECT_EQ(task->UpdateExtInfo(input_desc, output_desc, stream), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuBaseTask_UpdateOutputShape) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();
  task->op_desc_ = op_desc;
  task->aicpu_ext_handle_ = std::unique_ptr<ge::hybrid::AicpuExtInfoHandler>(
      new ge::hybrid::AicpuExtInfoHandler("Mul", 2, 1, DEPEND_IN_SHAPE));
  std::unique_ptr<AicpuShapeAndType> output_shape_and_type =
    std::unique_ptr<AicpuShapeAndType>(new AicpuShapeAndType());
  output_shape_and_type->type = 3;
  output_shape_and_type->dims[0] = 4;
  task->aicpu_ext_handle_->output_shape_and_type_.push_back(output_shape_and_type.get());

  std::vector<GeTensorDesc> input_desc = {GeTensorDesc(), GeTensorDesc()};
  std::vector<GeTensorDesc> output_desc = {GeTensorDesc()};
  EXPECT_EQ(task->UpdateOutputShape(output_desc), SUCCESS);

  task->num_inputs_ = 2;
  task->num_outputs_ = 1;
  task->input_is_const_ = std::vector<int8_t>({1});
  EXPECT_EQ(task->UpdateOutputShape(output_desc), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuBaseTask_UpdateShapeToOutputDesc) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();
  task->op_desc_ = op_desc;
  task->aicpu_ext_handle_ = std::unique_ptr<ge::hybrid::AicpuExtInfoHandler>(
      new ge::hybrid::AicpuExtInfoHandler("Mul", 2, 1, DEPEND_IN_SHAPE));

  GeTensorDesc output_desc = GeTensorDesc();
  GeShape shape_new;

  output_desc.SetShape(GeShape(std::vector<int64_t>{4}));
  output_desc.SetFormat(FORMAT_NHWC);
  output_desc.SetDataType(DT_INT32);
  EXPECT_NE(task->UpdateShapeToOutputDesc(shape_new, output_desc), SUCCESS);
  delete task;
}


TEST_F(UtestSingleOp, test_AiCpuBaseTask_ReadResultSummaryAndPrepareMemory) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();
  task->op_desc_ = op_desc;
  task->num_outputs_ = 1;
  void *ptr = new uint8_t[16];

  task->output_summary_host_.push_back(aicpu::FWKAdapter::ResultSummary({PtrToValue(ptr), 16,
    PtrToValue(ptr), 16}));
  void *test = new uint8_t[sizeof(aicpu::FWKAdapter::ResultSummary)];
  aicpu::FWKAdapter::ResultSummary *p = PtrToPtr<void, aicpu::FWKAdapter::ResultSummary>(test);
  p->shape_data_ptr = PtrToValue(ptr);
  p->shape_data_size = 16;
  p->raw_data_ptr = PtrToValue(ptr);
  p->raw_data_size = 16;;
  task->output_summary_.push_back(p);

  EXPECT_EQ(task->ReadResultSummaryAndPrepareMemory(), SUCCESS);
  task->output_summary_[0] = nullptr;
  delete[] (uint8_t *)test;
  delete[] (uint8_t *)ptr;
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuTask_CopyDataToHbm) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();
  task->op_desc_ = op_desc;
  task->num_outputs_ = 1;
  void *ptr = new uint8_t[16];
  task->output_summary_host_.push_back(aicpu::FWKAdapter::ResultSummary({PtrToValue(ptr), 16,
    PtrToValue(ptr), 16}));
  task->output_summary_.push_back(new uint8_t[sizeof(aicpu::FWKAdapter::ResultSummary)]);

  void* ptr1 = new uint8_t[16];
  task->copy_input_release_flag_dev_ = ptr1;
  void* ptr2 = new uint8_t[16];
  task->copy_input_data_size_dev_ = ptr2;
  void* ptr3 = new uint8_t[16];
  task->copy_input_src_dev_ = ptr3;
  void* ptr4 = new uint8_t[16];
  task->copy_input_dst_dev_ = ptr4;

  void *ptr5 = new uint8_t[sizeof(STR_FWK_OP_KERNEL)];
  task->copy_task_args_buf_ = ptr5;

  rtStream_t stream = nullptr;
  std::vector<DataBuffer> output;
  DataBuffer buffer;
  buffer.data = new uint8_t[16];
  buffer.length = 16;
  output.push_back(buffer);
  task->out_shape_hbm_.push_back(buffer.data);
  EXPECT_EQ(task->CopyDataToHbm(output, stream), SUCCESS);
  delete[] (uint8_t *)ptr;
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuTask_UpdateShapeAndDataByResultSummary) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();

  task->op_desc_ = op_desc;
  task->num_outputs_ = 1;
  void *ptr = new uint8_t[16];
  task->output_summary_host_.push_back(aicpu::FWKAdapter::ResultSummary({PtrToValue(ptr), 16,
    PtrToValue(ptr), 16}));
  void *test = new uint8_t[sizeof(aicpu::FWKAdapter::ResultSummary)];
  aicpu::FWKAdapter::ResultSummary *p = PtrToPtr<void, aicpu::FWKAdapter::ResultSummary>(test);
  p->shape_data_ptr = PtrToValue(ptr);
  p->shape_data_size = 16;
  p->raw_data_ptr = PtrToValue(ptr);
  p->raw_data_size = 16;;
  task->output_summary_.push_back(p);
  void* ptr1 = new uint8_t[16];
  task->copy_input_release_flag_dev_ = ptr1;
  void* ptr2 = new uint8_t[16];
  task->copy_input_data_size_dev_ = ptr2;
  void* ptr3 = new uint8_t[16];
  task->copy_input_src_dev_ = ptr3;
  void* ptr4 = new uint8_t[16];
  task->copy_input_dst_dev_ = ptr4;

  void *ptr5 = new uint8_t[sizeof(STR_FWK_OP_KERNEL)];
  task->copy_task_args_buf_ = ptr5;

  rtStream_t stream = nullptr;
  std::vector<GeTensorDesc> output_desc;
  output_desc.push_back(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  std::vector<DataBuffer> output;
  DataBuffer buffer;
  buffer.data = new uint8_t[16];
  buffer.length = 16;
  output.push_back(buffer);
  EXPECT_NE(task->UpdateShapeAndDataByResultSummary(output_desc, output, stream), SUCCESS);
  delete[] (uint8_t *)buffer.data;
  delete[] (uint8_t *)ptr;
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuTask_InitForSummaryAndCopy) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();

  task->op_desc_ = op_desc;
  task->unknown_type_ = DEPEND_COMPUTE;
  task->num_outputs_ = 1;
  void *ptr = new uint8_t[16];
  task->output_summary_host_.push_back(aicpu::FWKAdapter::ResultSummary({PtrToValue(ptr), 16,
    PtrToValue(ptr), 16}));
  void *test = new uint8_t[sizeof(aicpu::FWKAdapter::ResultSummary)];
  aicpu::FWKAdapter::ResultSummary *p = PtrToPtr<void, aicpu::FWKAdapter::ResultSummary>(test);
  p->shape_data_ptr = PtrToValue(ptr);
  p->shape_data_size = 16;
  p->raw_data_ptr = PtrToValue(ptr);
  p->raw_data_size = 16;;
  task->output_summary_.push_back(p);
  void* ptr1 = new uint8_t[16];
  task->copy_input_release_flag_dev_ = ptr1;
  void* ptr2 = new uint8_t[16];
  task->copy_input_data_size_dev_ = ptr2;
  void* ptr3 = new uint8_t[16];
  task->copy_input_src_dev_ = ptr3;
  void* ptr4 = new uint8_t[16];
  task->copy_input_dst_dev_ = ptr4;

  void *ptr5 = new uint8_t[sizeof(STR_FWK_OP_KERNEL)];
  task->copy_task_args_buf_ = ptr5;

  EXPECT_EQ(task->InitForSummaryAndCopy(), SUCCESS);
  delete[] (uint8_t *)ptr5;
  delete[] (uint8_t *)ptr4;
  delete[] (uint8_t *)ptr3;
  delete[] (uint8_t *)ptr2;
  delete[] (uint8_t *)ptr1;
  delete[] (uint8_t *)ptr;
  delete[] (uint8_t *)test;
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuTask_SetMemCopyTask) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();

  task->op_desc_ = op_desc;
  task->unknown_type_ = DEPEND_COMPUTE;
  task->num_outputs_ = 1;
  void *ptr = new uint8_t[16];
  task->output_summary_host_.push_back(aicpu::FWKAdapter::ResultSummary({PtrToValue(ptr), 16,
    PtrToValue(ptr), 16}));
  void *test = new uint8_t[sizeof(aicpu::FWKAdapter::ResultSummary)];
  aicpu::FWKAdapter::ResultSummary *p = PtrToPtr<void, aicpu::FWKAdapter::ResultSummary>(test);
  p->shape_data_ptr = PtrToValue(ptr);
  p->shape_data_size = 16;
  p->raw_data_ptr = PtrToValue(ptr);
  p->raw_data_size = 16;;
  task->output_summary_.push_back(p);
  void* ptr1 = new uint8_t[16];
  task->copy_input_release_flag_dev_ = ptr1;
  void* ptr2 = new uint8_t[16];
  task->copy_input_data_size_dev_ = ptr2;
  void* ptr3 = new uint8_t[16];
  task->copy_input_src_dev_ = ptr3;
  void* ptr4 = new uint8_t[16];
  task->copy_input_dst_dev_ = ptr4;

  void *ptr5 = new uint8_t[sizeof(STR_FWK_OP_KERNEL)];
  task->copy_task_args_buf_ = ptr5;

  domi::KernelExDef kernel_def;
  std::vector<uint8_t> task_info(16, 0);
  kernel_def.set_task_info(task_info.data(), task_info.size());
  kernel_def.set_task_info_size(task_info.size());
  std::vector<uint8_t> args_info(sizeof(STR_FWK_OP_KERNEL), 0);
  kernel_def.set_args(args_info.data(), args_info.size());
  kernel_def.set_args_size(args_info.size());
  EXPECT_EQ(task->SetMemCopyTask(kernel_def), SUCCESS);

  kernel_def.set_args_size(sizeof(STR_FWK_OP_KERNEL) + 1);
  EXPECT_EQ(task->SetMemCopyTask(kernel_def), ACL_ERROR_GE_PARAM_INVALID);
  delete[] (uint8_t *)ptr;
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuTask_LaunchKernel) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();

  task->op_desc_ = op_desc;
  task->num_outputs_ = 0;

  GeTensorDesc tensor = GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32);
  std::vector<GeTensorDesc> input_desc = {tensor, tensor};
  std::vector<DataBuffer> input_buffers = {};
  std::vector<GeTensorDesc> output_desc;
  std::vector<DataBuffer> output_buffers;
  rtStream_t stream = nullptr;

  task->unknown_type_ = DEPEND_COMPUTE;
  EXPECT_NE(task->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream), SUCCESS);
  task->unknown_type_ = DEPEND_SHAPE_RANGE;
  EXPECT_NE(task->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuCCTask_LaunchKernel) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuCCTask();

  task->op_desc_ = op_desc;
  task->num_outputs_ = 0;

  GeTensorDesc tensor = GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32);
  std::vector<GeTensorDesc> input_desc = {tensor, tensor};
  std::vector<DataBuffer> input_buffers = {};
  std::vector<GeTensorDesc> output_desc;
  std::vector<DataBuffer> output_buffers;
  rtStream_t stream = nullptr;

  task->unknown_type_ = DEPEND_COMPUTE;
  EXPECT_NE(task->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream), SUCCESS);
  task->unknown_type_ = DEPEND_SHAPE_RANGE;
  EXPECT_NE(task->LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream), SUCCESS);
  delete task;
}

TEST_F(UtestSingleOp, test_AiCpuBaseTask_UpdateArgTable) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();

  task->op_desc_ = op_desc;
  task->num_outputs_ = 1;

  SingleOpModelParam param;
  EXPECT_EQ(task->UpdateArgTable(param), SUCCESS);
  EXPECT_EQ(task->GetTaskType(), kTaskTypeAicpu);

  void *ptr = new uint8_t[16];
  task->io_addr_host_.push_back(ptr);

  uintptr_t *arg_base;
  size_t arg_count;
  task->GetIoAddr(arg_base, arg_count);
  EXPECT_EQ(arg_count, 1);

  delete[] (uint8_t *)ptr;
  delete task;
}

TEST_F(UtestSingleOp, TbeTask_SaveExceptionDumpInfo_EnableExceptionDump) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) TbeOpTask();
  task->node_ = node;
  task->op_desc_ = op_desc;
  task->args_ = std::unique_ptr<uint8_t[]>(new uint8_t[24]());
  task->arg_size_ = 24;
  task->max_tiling_size_ = 0;
  task->need_tiling_ = false;
  task->arg_num_ = 3;
  SingleOpModelParam param;
  param.runtime_param.mem_size = 1000;
  task->UpdateArgTable(param);
  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  ge::DumpStub::GetInstance().ClearOpInfos();
  ge::DumpStub::GetInstance().SetFuncRet("AdumpAddExceptionOperatorInfoV2", -1);
  ge::diagnoseSwitch::EnableExceptionDump();
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(task->SaveExceptionDumpInfo(), SUCCESS);

  // dynamic op, no op_desc saved
  auto op_desc_info = gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->op_desc_info_;
  EXPECT_EQ(op_desc_info.size(), 0U);

  // args check
  const auto &op_info = DumpStub::GetInstance().GetOpInfos()[0];
  EXPECT_EQ(AdxGetAdditionalInfo(op_info, "is_host_args"), "true");

  // io addr check
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0UL;
  task->GetIoAddr(arg_base, arg_num);
  EXPECT_EQ(op_info.tensorInfos[0].tensorAddr, reinterpret_cast<void *>(*arg_base));
  ++arg_base;
  EXPECT_EQ(op_info.tensorInfos[1].tensorAddr, reinterpret_cast<void *>(*arg_base));
  ++arg_base;
  EXPECT_EQ(op_info.tensorInfos[2].tensorAddr, reinterpret_cast<void *>(*arg_base));

  delete task;
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  DumpStub::GetInstance().ClearOpInfos();
  ge::DumpStub::GetInstance().ClearFuncRet();
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(UtestSingleOp, AiCpuTask_SaveExceptionDumpInfo_EnableExceptionDump) {
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddInputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->AddOutputDesc(GeTensorDesc(GeShape(std::vector<int64_t>{4}), FORMAT_NCHW, DT_INT32));
  op_desc->SetWorkspaceBytes({0});
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  auto *task = new (std::nothrow) AiCpuTask();

  task->op_desc_ = op_desc;
  task->num_outputs_ = 1;

  SingleOpModelParam param;
  EXPECT_EQ(task->UpdateArgTable(param), SUCCESS);
  EXPECT_EQ(task->GetTaskType(), kTaskTypeAicpu);

  auto *ptr1 = new uint8_t[8];
  auto *ptr2 = new uint8_t[8];
  auto *ptr3 = new uint8_t[8];
  auto *ptr4 = new uint8_t[8];
  task->io_addr_host_.push_back(ptr1);
  task->io_addr_host_.push_back(ptr2);
  task->io_addr_host_.push_back(ptr3);
  task->io_addr_host_.push_back(ptr4);

  ge::DumpStub::GetInstance().ClearOpInfos();
  gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->Clear();
  ge::diagnoseSwitch::EnableExceptionDump();
  dlog_setlevel(GE_MODULE_NAME, DLOG_INFO, 0);
  EXPECT_EQ(task->SaveExceptionDumpInfo(), SUCCESS);

  // dynamic op, no op_desc saved
  auto op_desc_info = gert::GlobalDumper::GetInstance()->MutableExceptionDumper()->op_desc_info_;
  EXPECT_EQ(op_desc_info.size(), 0U);

  // args check
  const auto &op_info = DumpStub::GetInstance().GetOpInfos()[0];
  EXPECT_EQ(AdxGetAdditionalInfo(op_info, "is_host_args"), "true");

  // io addr check
  uintptr_t *arg_base = nullptr;
  size_t arg_num = 0UL;
  task->GetIoAddr(arg_base, arg_num);
  EXPECT_EQ(op_info.tensorInfos[0].tensorAddr, reinterpret_cast<void *>(*arg_base));
  ++arg_base;
  EXPECT_EQ(op_info.tensorInfos[1].tensorAddr, reinterpret_cast<void *>(*arg_base));

  delete[] ptr1;
  delete[] ptr2;
  delete[] ptr3;
  delete[] ptr4;
  delete task;

  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  DumpStub::GetInstance().ClearOpInfos();
  ge::diagnoseSwitch::DisableDumper();
}

TEST_F(UtestSingleOp, SingleOp_MallocMemAndFreeOk_WithExternalAllocator) {
  std::mutex stream_mutex;
  void *stream = (void *)0x01;
  FakeAllocator fake_allocator{};
  auto stream_resource = MakeUnique<StreamResource>(1);
  stream_resource->SetAllocator(&fake_allocator);
  auto new_op = MakeUnique<SingleOp>(stream_resource.get(), &stream_mutex, stream);
  EXPECT_NE(new_op, nullptr);
  EXPECT_NE(new_op->impl_, nullptr);
  auto op_task = make_unique<TbeOpTask>();
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  op_task->op_desc_ = op_desc;
  new_op->impl_->tasks_.emplace_back(op_task.release());
  SingleOpModelParam model_params;
  model_params.runtime_param.mem_size = 20;
  model_params.runtime_param.zero_copy_size = 10;
  new_op->impl_->model_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  EXPECT_EQ(new_op->impl_->MallocOnExecute(), SUCCESS);
  EXPECT_EQ(new_op->impl_->model_param_->runtime_param.mem_base, 0);
  new_op->impl_->FreeAllocatedMem();
  EXPECT_TRUE(new_op->impl_->allocated_mem_ == nullptr);
}

TEST_F(UtestSingleOp, SingleOp_GetMemBase_WithExternalAllocator) {
  std::mutex stream_mutex;
  void *stream = (void *)0x01;
  auto stream_resource = MakeUnique<StreamResource>(1);
  auto new_op = MakeUnique<SingleOp>(stream_resource.get(), &stream_mutex, stream);
  EXPECT_NE(new_op, nullptr);
  EXPECT_NE(new_op->impl_, nullptr);
  FakeAllocator allocator;
  MemBlock block{allocator, (void *)0x04, 1};
  new_op->impl_->allocated_mem_ = &block;
  EXPECT_EQ(new_op->impl_->GetMemoryBase(), (uint8_t *)0x04);
}

TEST_F(UtestSingleOp, SingleOp_GetMemBase_WithoutExternalAllocator) {
  std::mutex stream_mutex;
  void *stream = (void *)0x01;
  auto stream_resource = MakeUnique<StreamResource>(1);
  auto new_op = MakeUnique<SingleOp>(stream_resource.get(), &stream_mutex, stream);
  EXPECT_NE(new_op, nullptr);
  EXPECT_NE(new_op->impl_, nullptr);
  InternalAllocator allocator;
  MemBlock block{allocator, (void *)0x04, 1};
  new_op->impl_->allocated_mem_ = &block;
  EXPECT_EQ(new_op->impl_->GetMemoryBase(), (uint8_t *)0x04);
}
