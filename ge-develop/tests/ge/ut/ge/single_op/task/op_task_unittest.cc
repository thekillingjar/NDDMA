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
#include <gmock/gmock.h>

#include "macro_utils/dt_public_scope.h"
#include "single_op/task/op_task.h"
#include "depends/profiler/src/dump_stub.h"
#include "graph/utils/graph_utils.h"
#include "aicpu_task_struct.h"
#include "depends/runtime/src/runtime_stub.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "runtime/subscriber/built_in_subscriber_definitions.h"
#include "runtime/subscriber/global_profiler.h"
#include "depends/profiler/src/profiling_test_util.h"

using namespace std;
using namespace testing;
using namespace ge;
using namespace optiling;

using namespace hybrid;
class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD5(rtKernelGetAddrAndPrefCntV2, int32_t(void *handle, const uint64_t tilingKey, const void *const stubFunc,
                                                    const uint32_t flag, rtKernelDetailInfo_t *kernelInfo));
};

class UtestOpTask : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {
    MockRuntime::Reset();
  }
};

TEST_F(UtestOpTask, test_tbe_launch_kernel) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);

  TbeOpTask task;
  ge::DataBuffer data_buffer;
  vector<GeTensorDesc> input_desc;
  vector<DataBuffer> input_buffers = { data_buffer };
  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers = { data_buffer };
  task.op_desc_ = op_desc;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  task.op_ = std::move(std::unique_ptr<Operator>(new(std::nothrow) Operator(op)));
  task.node_ = node;
  OpTilingFuncV2 op_tiling_func = [](const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &) -> bool {return true;};
  REGISTER_OP_TILING_UNIQ_V2(Add, op_tiling_func, 1);
  OpTilingRegistryInterf_V2("Add", op_tiling_func);
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "op_compile_info_json");
  task.max_tiling_size_ = 64;
  task.need_tiling_ = true;
  task.arg_index_ = {0};
  task.input_num_ = 1;
  task.output_num_ = 1;
  task.arg_size_ = sizeof(void *) * 3 + 64;
  task.args_.reset(new (std::nothrow) uint8_t[sizeof(void *) * 3 + 64]);
  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = task.arg_size_;
  task.run_info_= MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(task.run_info_, nullptr);

  rtStream_t stream_ = nullptr;
  ASSERT_EQ(task.LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream_), SUCCESS);
  char handle = '0';
  task.SetHandle(&handle);
  ASSERT_EQ(task.LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream_), SUCCESS);
}

TEST_F(UtestOpTask, test_update_args) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  auto node = graph->AddNode(op_desc);

  TbeOpTask task;
  task.EnableDynamicSupport(node, 64);
  task.max_tiling_size_ = 64;
  task.input_num_ = 1;
  task.output_num_ = 1;
  task.arg_size_ = sizeof(void *) * 3 + 64 + kMaxHostMemInputLen;
  task.args_.reset(new (std::nothrow) uint8_t[task.arg_size_]);
  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = task.arg_size_;
  task.args_item_offsets_.tiling_addr_offset = sizeof(void *) * 2;
  task.args_item_offsets_.tiling_data_offset = sizeof(void *) * 3;
  task.arg_num_ = 2;
  task.run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(task.run_info_, nullptr);

  EXPECT_EQ(task.UpdateTilingArgs(), SUCCESS);
  uint64_t *new_tiling_addr =
      PtrToPtr<void, uint64_t>(ValueToPtr(PtrToValue(task.args_.get()) + task.args_item_offsets_.tiling_addr_offset));
  char_t *new_tiling_data =
      PtrToPtr<void, char_t>(ValueToPtr(PtrToValue(task.args_.get()) + task.args_item_offsets_.tiling_data_offset));
  EXPECT_EQ(*new_tiling_addr, PtrToValue(new_tiling_data));
  size_t arg_count = 0;
  uintptr_t *arg_base = nullptr;
  task.GetIoAddr(arg_base, arg_count);
  EXPECT_EQ(arg_base, reinterpret_cast<uintptr_t *>(task.args_.get()));
  ASSERT_NE(arg_count, 2);

  AtomicAddrCleanOpTask atomic_task;
  atomic_task.node_ = node;
  atomic_task.need_tiling_ = true;
  atomic_task.arg_size_ = sizeof(void *) * 3 + 64;
  atomic_task.args_.reset(new (std::nothrow) uint8_t[sizeof(void *) * 3 + 64]);
  atomic_task.args_ex_.args = atomic_task.args_.get();
  atomic_task.args_ex_.argsSize = atomic_task.arg_size_;
  atomic_task.run_info_ = MakeUnique<optiling::utils::OpRunInfo>();
  ASSERT_NE(atomic_task.run_info_, nullptr);

  atomic_task.max_tiling_size_ = 64;
  atomic_task.UpdateOverflowAddr();
  EXPECT_EQ(atomic_task.UpdateTilingArgs(), SUCCESS);
  EXPECT_EQ(atomic_task.args_ex_.tilingAddrOffset, 0);
}

// for dynamic single op
TEST_F(UtestOpTask, test_tbe_update_args) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc tensor_desc;
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  AttrUtils::SetInt(op_desc, "globalworkspace_type", 0);
  auto node = graph->AddNode(op_desc);
  size_t total_size = 0U;

  TbeOpTask task;
  task.node_ = node;
  task.input_num_ = 2;
  task.output_num_ = 1;
  task.arg_size_ = 64;
  task.args_ = MakeUnique<uint8_t []>(task.arg_size_);
  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = task.arg_size_;
  task.arg_num_ = 3;

  // input + output
  total_size += (task.arg_num_ * sizeof(void *));
  task.op_desc_ = op_desc;

  op_desc->SetIsInputConst({false, false});

  // workspace
  task.workspaces_.emplace_back((void *)222);
  task.workspaces_.emplace_back((void *)333);
  total_size += (task.workspaces_.size() * sizeof(void *));

  // tiling
  task.max_tiling_size_ = 64;
  task.need_tiling_ = true;
  total_size += task.max_tiling_size_; // tiling data
  total_size += sizeof(void *); // tiling addr

  // overflow
  task.has_overflow_attr_ = true;
  uint64_t overflow_value = 111;
  task.overflow_addr_ = &overflow_value;
  total_size += sizeof(void *);

  // host mem input
  task.need_host_mem_opt_ = true;
  task.extend_args_for_host_input_ = true;
  task.args_item_offsets_.host_input_data_offset =
      (task.arg_num_ + 1 + task.workspaces_.size() + 1) * sizeof(uintptr_t) + task.max_tiling_size_;
  task.arg_index_.emplace_back(0);
  task.arg_index_.emplace_back(1);
  total_size += kMaxHostMemInputLen;

  // input
  uint64_t host_value0 = 333;
  DataBuffer buffer0;
  buffer0.placement = kHostMemType;
  buffer0.length = 8;
  buffer0.data = &host_value0;

  uint64_t host_value1[4] = {444, 555, 666, 777};
  DataBuffer buffer1;
  buffer1.placement = kHostMemType;
  buffer1.length = 32;
  buffer1.data = &host_value1[0];

  vector<DataBuffer> inputs;
  inputs.emplace_back(buffer1);
  inputs.emplace_back(buffer0);

  EXPECT_EQ(task.UpdateArgsItem(inputs, {}), SUCCESS);
  // chck args size
  EXPECT_EQ(task.arg_size_, total_size);

  // check overflow offset
  size_t overflow_addr_offset = (task.arg_num_ + 1 + task.workspaces_.size()) * sizeof(uintptr_t);
  EXPECT_EQ(task.args_item_offsets_.overflow_addr_offset, overflow_addr_offset);
  // check overflow addr
  uint64_t *args_base = reinterpret_cast<uint64_t *>(task.args_.get());
  EXPECT_EQ(args_base[overflow_addr_offset / sizeof(uintptr_t)], PtrToValue(&overflow_value));

  // check workspace offset
  size_t ws_offset = task.arg_num_ * sizeof(uintptr_t);
  EXPECT_EQ(task.args_item_offsets_.workspace_addr_offset, ws_offset);
  // check workspace addr
  EXPECT_EQ(args_base[ws_offset / sizeof(uintptr_t)], PtrToValue(task.workspaces_.at(0U)));

  // check tiling addr offset、tiling data offset
  size_t tiling_addr_offset = (task.arg_num_ + task.workspaces_.size()) * sizeof(uintptr_t);
  EXPECT_EQ(task.args_item_offsets_.tiling_addr_offset, tiling_addr_offset);
  size_t tiling_data_offset = (task.arg_num_+ 1 + task.workspaces_.size() + 1) * sizeof(uintptr_t);
  EXPECT_EQ(task.args_item_offsets_.tiling_data_offset, tiling_data_offset);
  uint64_t tiling_data_addr_in_args = PtrToValue(args_base + tiling_data_offset / sizeof(uintptr_t));
  EXPECT_EQ(args_base[tiling_addr_offset / sizeof(uintptr_t)], tiling_data_addr_in_args);

  EXPECT_EQ(task.args_ex_.tilingAddrOffset, tiling_addr_offset);
  EXPECT_EQ(task.args_ex_.tilingDataOffset, tiling_data_offset);
  EXPECT_TRUE(task.args_ex_.hasTiling);

  // check host mem offset
  size_t host_mem0_offset = tiling_data_offset + task.max_tiling_size_;
  EXPECT_EQ(task.args_item_offsets_.host_input_data_offset, host_mem0_offset);
  uint64_t host_mem0_data_in_args = PtrToValue(args_base + host_mem0_offset / sizeof(uintptr_t));
  EXPECT_EQ(args_base[0U], host_mem0_data_in_args);
  uint64_t host_mem1_data_in_args = host_mem0_data_in_args + inputs.at(0).length;
  EXPECT_EQ(args_base[1U], host_mem1_data_in_args);

  // check host mem data
  EXPECT_EQ(args_base[host_mem0_offset / sizeof(uintptr_t)], host_value1[0U]);
  auto host_mem1_offset = host_mem0_offset + inputs.at(0).length;
  EXPECT_EQ(args_base[host_mem1_offset / sizeof(uintptr_t)], host_value0);

  // check host mem args
  ASSERT_NE(task.args_ex_.hostInputInfoPtr, nullptr);
  EXPECT_EQ(task.args_ex_.hostInputInfoNum, inputs.size());
  EXPECT_EQ(task.args_ex_.hostInputInfoPtr[0].dataOffset, host_mem0_offset);
  EXPECT_EQ(task.args_ex_.hostInputInfoPtr[1].dataOffset, host_mem1_offset);
  EXPECT_EQ(task.args_ex_.hostInputInfoPtr[0].addrOffset, 0);
  EXPECT_EQ(task.args_ex_.hostInputInfoPtr[1].addrOffset, sizeof(uintptr_t));

}

TEST_F(UtestOpTask, test_tbe_update_args_notiling) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc tensor_desc;
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  uint64_t globalworkspace_type = 0;
  AttrUtils::SetInt(op_desc, "globalworkspace_type", globalworkspace_type);
  auto node = graph->AddNode(op_desc);
  size_t total_size = 0U;

  TbeOpTask task;
  task.node_ = node;
  task.input_num_ = 2;
  task.output_num_ = 1;
  task.arg_size_ = 64;
  task.args_ = MakeUnique<uint8_t []>(task.arg_size_);
  task.args_ex_.args = task.args_.get();
  task.args_ex_.argsSize = task.arg_size_;
  task.arg_num_ = 3;

  // input + output
  total_size += (task.arg_num_ * sizeof(void *));
  task.op_desc_ = op_desc;

  op_desc->SetIsInputConst({false, false});

  // workspace
  task.workspaces_.emplace_back((void *)222);
  task.workspaces_.emplace_back((void *)333);
  total_size += (task.workspaces_.size() * sizeof(void *));

  // tiling
  task.max_tiling_size_ = 64;
  task.need_tiling_ = false;
  total_size += task.max_tiling_size_; // tiling data
  total_size += sizeof(void *); // tiling addr

  // overflow
  task.has_overflow_attr_ = true;
  uint64_t overflow_value = 111;
  task.overflow_addr_ = &overflow_value;
  total_size += sizeof(void *);

  // host mem input
  task.need_host_mem_opt_ = true;
  task.extend_args_for_host_input_ = true;
  task.args_item_offsets_.host_input_data_offset =
      (task.arg_num_ + 1 + task.workspaces_.size() + 1) * sizeof(uintptr_t) + task.max_tiling_size_;
  task.arg_index_.emplace_back(0);
  task.arg_index_.emplace_back(1);
  total_size += kMaxHostMemInputLen;

  // input
  uint64_t host_value0 = 333;
  DataBuffer buffer0;
  buffer0.placement = kHostMemType;
  buffer0.length = 8;
  buffer0.data = &host_value0;

  uint64_t host_value1[4] = {444, 555, 666, 777};
  DataBuffer buffer1;
  buffer1.placement = kHostMemType;
  buffer1.length = 32;
  buffer1.data = &host_value1[0];

  vector<DataBuffer> inputs;
  inputs.emplace_back(buffer1);
  inputs.emplace_back(buffer0);

  EXPECT_EQ(task.UpdateArgsItem(inputs, {}), SUCCESS);
}

TEST_F(UtestOpTask, test_tf_aicpu_update_host_mem_input_args) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc tensor_desc;
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);

  AiCpuTask task;
  size_t io_num = 3U;
  // host args
  task.arg_size_ = io_num * sizeof(void *) + kMaxHostMemInputLen;
  // tf cpu input address must be 64 bytes aligned
  task.arg_size_ = (task.arg_size_ + 64 - 1) / 64 * 64;
  uint8_t args[task.arg_size_];
  task.args_ = args;
  for (size_t i = 0; i < task.arg_size_ / sizeof(void *); i++) {
    task.io_addr_host_.emplace_back(nullptr);
  }

  // device args
  task.io_addr_size_ = task.arg_size_;
  uint8_t device_args[task.io_addr_size_];
  task.io_addr_ = device_args;

  task.op_desc_ = op_desc;
  op_desc->SetIsInputConst({false, false});

  // host mem input
  task.need_host_mem_opt_ = true;
  task.host_mem_input_data_offset_ = task.arg_size_ - kMaxHostMemInputLen;

  // input
  uint64_t host_value0 = 333;
  DataBuffer buffer0;
  buffer0.placement = kHostMemType;
  buffer0.length = 8;
  buffer0.data = &host_value0;

  uint64_t host_value1[4] = {444, 555, 666, 777};
  DataBuffer buffer1;
  buffer1.placement = kHostMemType;
  buffer1.length = 32;
  buffer1.data = &host_value1[0];

  vector<DataBuffer> inputs;
  inputs.emplace_back(buffer1);
  inputs.emplace_back(buffer0);

  EXPECT_EQ(task.UpdateHostMemInputArgs(inputs, {}), SUCCESS);

  // check host mem data
  uint64_t *args_base = reinterpret_cast<uint64_t *>(task.io_addr_host_.data());
  size_t host_mem0_data_offset = task.host_mem_input_data_offset_;
  EXPECT_EQ(args_base[host_mem0_data_offset / sizeof(void *)], host_value1[0]);
  size_t host_mem1_data_offset = host_mem0_data_offset + buffer1.length;
  // tf cpu input address must be 64 bytes aligned
  host_mem1_data_offset = (host_mem1_data_offset + 64 - 1) / 64 * 64;
  EXPECT_EQ(args_base[host_mem1_data_offset / sizeof(void *)], host_value0);

  // check host mem addr
  size_t addr0_offset = 0U;
  size_t addr1_offset = sizeof(void *);
  uint64_t device_io_addr = PtrToValue(task.io_addr_);
  EXPECT_EQ(args_base[addr0_offset / sizeof(void *)], device_io_addr + host_mem0_data_offset);
  EXPECT_EQ(args_base[addr1_offset / sizeof(void *)], device_io_addr + host_mem1_data_offset);
  task.args_ = nullptr;
  task.io_addr_ = nullptr;
}

TEST_F(UtestOpTask, test_aicpu_update_host_mem_input_args) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc tensor_desc;
  op_desc->AddInputDesc(tensor_desc);
  op_desc->AddInputDesc(tensor_desc);
  auto node = graph->AddNode(op_desc);

  AiCpuCCTask task;
  size_t io_num = 3U;
  // host args
  task.arg_size_ = sizeof(aicpu::AicpuParamHead) + io_num * sizeof(void *) + kMaxHostMemInputLen;
  task.args_ = MakeUnique<uint8_t[]>(task.arg_size_);
  task.io_addr_ = PtrToPtr<void, uintptr_t>(ValueToPtr(PtrToValue(task.args_.get()) +
                                                       sizeof(aicpu::AicpuParamHead)));
  task.io_addr_num_ = io_num;

  task.op_desc_ = op_desc;
  op_desc->SetIsInputConst({false, false});

  // host mem input
  task.need_host_mem_opt_ = true;
  task.host_mem_input_data_offset_ = io_num * sizeof(void *);

  // input
  uint64_t host_value0 = 333;
  DataBuffer buffer0;
  buffer0.placement = kHostMemType;
  buffer0.length = 8;
  buffer0.data = &host_value0;

  uint64_t host_value1[4] = {444, 555, 666, 777};
  DataBuffer buffer1;
  buffer1.placement = kHostMemType;
  buffer1.length = 32;
  buffer1.data = &host_value1[0];

  vector<DataBuffer> inputs;
  inputs.emplace_back(buffer1);
  inputs.emplace_back(buffer0);

  EXPECT_EQ(task.UpdateHostMemInputArgs(inputs, {}), SUCCESS);

  // check host mem data
  uint64_t *args_base = reinterpret_cast<uint64_t *>(task.io_addr_);
  size_t host_mem0_data_offset = task.host_mem_input_data_offset_;
  EXPECT_EQ(args_base[host_mem0_data_offset / sizeof(void *)], host_value1[0]);
  size_t host_mem1_data_offset = host_mem0_data_offset + buffer1.length;
  EXPECT_EQ(args_base[host_mem1_data_offset / sizeof(void *)], host_value0);

  // check host mem addr
  size_t addr0_offset = 0U;
  size_t addr1_offset = sizeof(void *);
  uint64_t io_addr = PtrToValue(task.io_addr_);
  EXPECT_EQ(args_base[addr0_offset / sizeof(void *)], io_addr + host_mem0_data_offset);
  EXPECT_EQ(args_base[addr1_offset / sizeof(void *)], io_addr + host_mem1_data_offset);

  // check task.args_ex_
  ASSERT_FALSE(task.args_ex_.hasTiling);
  ASSERT_NE(task.args_ex_.hostInputInfoPtr, nullptr);
  EXPECT_EQ(task.args_ex_.hostInputInfoNum, 2);
  EXPECT_EQ(task.args_ex_.hostInputInfoPtr[0].dataOffset, host_mem0_data_offset + sizeof(aicpu::AicpuParamHead));
  EXPECT_EQ(task.args_ex_.hostInputInfoPtr[1].dataOffset, host_mem1_data_offset + sizeof(aicpu::AicpuParamHead));
  task.args_ = nullptr;
  task.io_addr_ = nullptr;
}

TEST_F(UtestOpTask, test_mixl2_kernel_success) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);

  MixL2OpTask task(node);
  ge::DataBuffer data_buffer;
  vector<GeTensorDesc> input_desc;
  vector<DataBuffer> input_buffers = { data_buffer };
  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers = { data_buffer };
  task.op_desc_ = op_desc;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  task.op_ = std::move(std::unique_ptr<Operator>(new(std::nothrow) Operator(op)));
  task.node_ = node;
  OpTilingFuncV2 op_tiling_func = [](const ge::Operator &, const OpCompileInfoV2 &, OpRunInfoV2 &) -> bool {return true;};
  REGISTER_OP_TILING_UNIQ_V2(Add, op_tiling_func, 1);
  OpTilingRegistryInterf_V2("Add", op_tiling_func);
  ge::AttrUtils::SetStr(op_desc, "compile_info_key", "op_compile_info_key");
  ge::AttrUtils::SetStr(op_desc, "compile_info_json", "op_compile_info_json");
  task.max_tiling_size_ = 64;
  task.args_addr_base_idx_ = 1;
  task.need_tiling_ = true;
  task.arg_index_ = {0};
  task.input_num_ = 1;
  task.output_num_ = 1;
  task.arg_size_ = sizeof(void *) * 3 + 64;
  task.host_args_.resize(4);
  task.args_addr_cnt_ = 3;
  task.host_mem_base_idx_ = 1;

  {
     auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc,
                    const uint32_t flag, rtKernelDetailInfo_t *kernelInfo) -> int {
     kernelInfo->functionInfoNum = 1;
     kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
     kernelInfo->functionInfo[0].prefetchCnt = 1;
     return RT_ERROR_NONE;
     };
     auto runtime_stub = std::make_shared<MockRuntime>();
     RuntimeStub::SetInstance(runtime_stub);
     EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
     (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aic_kernel");
     TBEHandleStore::GetInstance().StoreTBEHandle("mix_aic_kernel", nullptr, nullptr);
     (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aiv_kernel");
     TBEHandleStore::GetInstance().StoreTBEHandle("mix_aiv_kernel", nullptr, nullptr);
     std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
     task.names_prefix_ = name_prefix;
     ASSERT_EQ(task.UpdateIoAddr(input_buffers, output_buffers), SUCCESS);
  }

  {
    auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc,
                   const uint32_t flag, rtKernelDetailInfo_t *kernelInfo) -> int {
      kernelInfo->functionInfoNum = 2;
      kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
      kernelInfo->functionInfo[0].prefetchCnt = 1;
      kernelInfo->functionInfo[1].pcAddr = (void *)(0x1234);
      kernelInfo->functionInfo[1].prefetchCnt = 2;
      return RT_ERROR_NONE;
    };
    auto runtime_stub = std::make_shared<MockRuntime>();
    RuntimeStub::SetInstance(runtime_stub);
    EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
    (void)AttrUtils::SetStr(op_desc, "_mix_enhanced" + ATTR_NAME_TBE_KERNEL_NAME, "mix_enhanced_kernel");
    TBEHandleStore::GetInstance().StoreTBEHandle("mix_enhanced_kernel", nullptr, nullptr);
    std::vector<std::string> name_prefix = {"_mix_enhanced"};
    task.names_prefix_ = name_prefix;
    ASSERT_EQ(task.UpdateIoAddr(input_buffers, output_buffers), SUCCESS);
  }

  ASSERT_EQ(task.UpdateIoAddr(input_buffers, output_buffers), SUCCESS);

  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));
  uint64_t launch_begin_time;
  ASSERT_EQ(task.PreProcess(launch_begin_time), SUCCESS);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);

  uintptr_t *base = nullptr;
  size_t cnt = 0;
  task.GetIoAddr(base, cnt);
  ASSERT_EQ(base, &task.host_args_[1]);
  ASSERT_EQ(cnt, 2);
  rtStream_t stream = nullptr;
  ASSERT_EQ(task.DoLaunchKernel(stream), SUCCESS);
}

TEST_F(UtestOpTask, test_dynamic_mixl2_launch_failed) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);
  MixL2OpTask task(node);
  ge::DataBuffer data_buffer;
  vector<GeTensorDesc> input_desc;
  vector<DataBuffer> input_buffers = {data_buffer};
  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers = {data_buffer};
  task.op_desc_ = op_desc;
  auto op = OpDescUtils::CreateOperatorFromNode(node);
  task.op_ = std::move(std::unique_ptr<Operator>(new (std::nothrow) Operator(op)));
  task.node_ = node;
  rtStream_t stream = nullptr;
  EXPECT_NE(task.LaunchKernel(input_desc, input_buffers, output_desc, output_buffers, stream), SUCCESS);
}

TEST_F(UtestOpTask, test_mixl2_kernel_host_mem) {
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);

  MixL2OpTask task(node);
  task.arg_index_.push_back(0);
  vector<DataBuffer> input_buffers;
  vector<DataBuffer> output_buffers;
  ASSERT_NE(task.UpdateIoAddr(input_buffers, output_buffers), SUCCESS);
  uint64_t host_value1[4] = {444, 555, 666, 777};
  ge::DataBuffer input_buffer;
  input_buffer.placement = kHostMemType;
  input_buffer.length = 32;
  input_buffer.data = &host_value1[0];
  task.need_host_mem_opt_ = true;
  input_buffers.push_back(input_buffer);
  vector<uint8_t> tmp(20 * sizeof(uintptr_t));
  task.host_args_.resize(20);
  ASSERT_EQ(task.UpdateHostMemInputArgs(input_buffers, output_buffers), SUCCESS);
}

TEST_F(UtestOpTask, test_OpTask_ReportProfAdditionalInfo) {
  thread_local const int32_t tid = mmGetTid();
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);

  MixL2OpTask task(node);
  task.op_desc_ = op_desc;
  task.context_ids_.push_back(0);
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kDevice}));
  const uint64_t end_time = MsprofSysCycleTime();
  const uint64_t op_name_hash = MsprofGetHashId(op_desc->GetName().c_str(), op_desc->GetName().length());
  ASSERT_EQ(task.ReportProfAdditionalInfo(end_time, op_name_hash, tid), SUCCESS);
}

TEST_F(UtestOpTask, test_TbeOpTask_FFTSTASK_ReportProfAdditionalInfo) {
  thread_local const int32_t tid = mmGetTid();
  auto graph = make_shared<ComputeGraph>("graph");
  auto op_desc = make_shared<OpDesc>("Add", "Add");
  GeTensorDesc desc;
  op_desc->AddInputDesc(desc);
  op_desc->AddOutputDesc(desc);
  auto node = graph->AddNode(op_desc);

  uint32_t task_ratio = 10U;
  uint32_t block_dim = 16U;
  ge::AttrUtils::SetBool(op_desc, "_is_fftsplus_task", true);
  ge::AttrUtils::SetBool(op_desc, "_mix_is_aiv", true);
  (void)AttrUtils::SetInt(op_desc, "_task_ratio", task_ratio);

  bool has_context_id = false;
  auto check_func = [&task_ratio, &block_dim, &has_context_id](uint32_t moduleId, uint32_t type, void *data,
                                                               uint32_t len) -> int32_t {
    if ((moduleId == 0) && (type == InfoType::kInfo)) {
      const auto *context_info = reinterpret_cast<MsprofAdditionalInfo *>(data);
      if (context_info->type == MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE) {
        const auto context_data = reinterpret_cast<const MsprofContextIdInfo *>(context_info->data);
        EXPECT_EQ(context_data->ctxIdNum, 1U);
        EXPECT_EQ(context_data->ctxIds[0], 0U);
        has_context_id = true;
      }
      return 0;
    }

    if ((moduleId == 0) && (type == InfoType::kCompactInfo)) {
      const auto &node_basic_info = (reinterpret_cast<MsprofCompactInfo *>(data))->data.nodeBasicInfo;
      EXPECT_EQ(node_basic_info.taskType, MSPROF_GE_TASK_TYPE_MIX_AIV);
      EXPECT_EQ(node_basic_info.blockDim, ((block_dim & 0xFFFFU) | (task_ratio << 16U)));
      return 0;
    }
    return 0;
  };
  ge::ProfilingTestUtil::Instance().SetProfFunc(check_func);

  TbeOpTask task(node);
  task.op_desc_ = op_desc;
  task.block_dim_ = block_dim;
  gert::GlobalProfilingWrapper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::ProfilingType>({gert::ProfilingType::kDevice}));
  const uint64_t end_time = MsprofSysCycleTime();
  const uint64_t op_name_hash = MsprofGetHashId(op_desc->GetName().c_str(), op_desc->GetName().length());
  ASSERT_EQ(task.ReportProfAdditionalInfo(end_time, op_name_hash, tid), SUCCESS);
  EXPECT_TRUE(has_context_id == true);
  ge::ProfilingTestUtil::Instance().Clear();
}

TEST_F(UtestOpTask, test_tbe_get_tasktype) {
  auto op_desc = make_shared<OpDesc>("Add", "Add");

  TbeOpTask task;
  task.op_desc_ = op_desc;
  EXPECT_EQ(task.GetTaskType(), "AI_CORE");

  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV");
  EXPECT_EQ(task.GetTaskType(), "AIV");
}

TEST_F(UtestOpTask, test_exception_dump) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));
  auto op_desc = make_shared<OpDesc>("Add", "Add");

  TbeOpTask task;
  task.op_desc_ = op_desc;
  GeTensorDesc desc(ge::GeShape(std::vector<int64_t>({1, 2})));
  GeTensorDesc invalid_desc(ge::GeShape(std::vector<int64_t>({-1, 2})));
  invalid_desc.SetDataType(DT_UNDEFINED);
  GeTensorDesc out_desc(ge::GeShape(std::vector<int64_t>({2, 2})));
  task.op_desc_->AddInputDesc(desc);
  task.op_desc_->AddInputDesc(invalid_desc);
  task.op_desc_->AddOutputDesc(out_desc);
  const std::vector<int64_t> workspace_bytes{16, 18, 16};
  task.op_desc_->SetWorkspaceBytes(workspace_bytes);
  EXPECT_EQ(task.GetTaskType(), "AI_CORE");
  task.SaveForL0ExceptionDump();
  uint64_t launch_time = 0U;
  task.PreProcess(launch_time);
  EXPECT_EQ(ReportL0ExceptionDumpInfo(op_desc, task.l0_dump_list_), SUCCESS);

  EXPECT_TRUE(!ge::DumpStub::GetInstance().units_.empty());
  std::vector<uint64_t> size_list = *ge::DumpStub::GetInstance().units_.rbegin();
  EXPECT_TRUE(size_list.size() >= 8UL);
  EXPECT_EQ(size_list[1], 6);
  EXPECT_EQ(size_list[3], 0);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}
