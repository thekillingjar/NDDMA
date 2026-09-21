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

#include "macro_utils/dt_public_scope.h"
#include "ge_graph_dsl/graph_dsl.h"

#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_task_info.h"
#include "graph/load/model_manager/task_info/ffts_plus/ffts_plus_args_helper.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "graph/load/model_manager/tbe_kernel_handle.h"
#include "depends/runtime/src/runtime_stub.h"
#include "graph/attr_value.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/model_manager.h"
#include "graph/execute/model_executor.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "ge/ut/ge/ffts_plus_proto_tools.h"
#include "ge/ut/ge/test_tools_task_info.h"
#include "graph/operator_factory_impl.h"
#include "depends/runtime/src/runtime_stub.h"
#include "register/op_impl_registry.h"
#include "faker/space_registry_faker.h"
#include "graph/utils/op_desc_utils.h"
#include "engine/aicore/fe_rt2_common.h"
#include "graph/args_format_desc.h"

#include "base/registry/op_impl_space_registry_v2.h"
#include "common/opskernel/ops_kernel_info_types.h"

using namespace ge;
namespace ge {
class MockRuntime : public RuntimeStub {
 public:
  MOCK_METHOD5(rtKernelGetAddrAndPrefCntV2, int32_t(void *handle, const uint64_t tilingKey, const void *const stubFunc,
                                                    const uint32_t flag, rtKernelDetailInfo_t *kernelInfo));
};
namespace {
void *g_dev = nullptr;
void *g_host = nullptr;
int64_t g_arg_len = 0;

}  // namespace
class UtestFftsPlusTaskInfo : public testing::Test {
 protected:
  void SetUp() {
    ResetNodeIndex();
  }

  void TearDown() {
    MockRuntime::Reset();
    if (g_dev != nullptr) {
      rtFree(g_dev);
      g_dev = nullptr;
    }
    if (g_host != nullptr) {
      free(g_host);
      g_host = nullptr;
    }
  }
};
// test FftsPlusTaskInfo Init software ctx
TEST_F(UtestFftsPlusTaskInfo, invalid_ffts_plus_task_info_model_param) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);
  FftsPlusTaskInfo ffts_plus_task_info;
  // init failed when model without op_desc
  EXPECT_EQ(ffts_plus_task_info.Init(task_def, &davinci_model), PARAM_INVALID);
}

// test FftsPlusTaskInfo Init software ctx
TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_start) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  FftsPlusTaskInfo ffts_plus_task_info;

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  InitTaskSQEInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *fftsplusstartctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusstartctx->set_op_index(0);
  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_START));
  domi::FftsPlusAtStartCtxDef *startctxdef = fftsplusstartctx->mutable_at_start_ctx();
  InitAtStartCtx(startctxdef);
  startctxdef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  int64_t op_index = ffts_plus_task_info.ParseOpIndex(task_def);
  EXPECT_EQ(op_index, 0);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  ffts_plus_task_info.ffts_flus_args_helper_->io_addrs_ = {0x12000000, 0x12001000};
  EXPECT_NE(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_software_start) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  FftsPlusTaskInfo ffts_plus_task_info;

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  davinci_model.runtime_param_.mem_size = 0x4000000;
  davinci_model.runtime_param_.logic_mem_base = 0x10000000;
  davinci_model.runtime_param_.mem_base = 0x0;

  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *fftsplusstartctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusstartctx->set_op_index(0);
  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_START));
  domi::FftsPlusAtStartCtxDef *startctxdef = fftsplusstartctx->mutable_at_start_ctx();
  InitAtStartCtx(startctxdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}

// test FftsPlusTaskInfo Init software ctx
TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_software_end) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  FftsPlusTaskInfo ffts_plus_task_info;

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  InitTaskSQEInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *fftsplusendctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusendctx->set_op_index(0);
  fftsplusendctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_END));
  domi::FftsPlusAtEndCtxDef *endctxdef = fftsplusendctx->mutable_at_end_ctx();
  InitAtEndCtx(endctxdef);
  endctxdef->add_succ_at_start_slot(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}
// test FftsPlusTaskInfo Init software ctx
TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_software_end_label) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  FftsPlusTaskInfo ffts_plus_task_info;

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *fftsplusendctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusendctx->set_op_index(0);
  fftsplusendctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_END));
  domi::FftsPlusAtEndCtxDef *endctxdef = fftsplusendctx->mutable_at_end_ctx();
  InitAtEndCtx(endctxdef);
  endctxdef->add_succ_out_label_slot(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}

// test FftsPlusTaskInfo Init software ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_software_end) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  FftsPlusTaskInfo ffts_plus_task_info;

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  davinci_model.runtime_param_.mem_size = 0x4000000;
  davinci_model.runtime_param_.logic_mem_base = 0x10000000;
  davinci_model.runtime_param_.mem_base = 0x0;

  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *fftsplusendctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusendctx->set_op_index(0);
  fftsplusendctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_END));
  domi::FftsPlusAtEndCtxDef *endctxdef = fftsplusendctx->mutable_at_end_ctx();
  InitAtEndCtx(endctxdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_software_ctx_lable) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  FftsPlusTaskInfo ffts_plus_task_info;

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *fftspluslabelctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftspluslabelctx->set_op_index(0);
  fftspluslabelctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_LABEL));
  domi::FftsPlusLabelCtxDef *labelctxdef = fftspluslabelctx->mutable_label_ctx();
  InitLabelCtx(labelctxdef);
  labelctxdef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}
// test FftsPlusTaskInfo Init software ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_software_ctx) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  FftsPlusTaskInfo ffts_plus_task_info;

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  davinci_model.runtime_param_.mem_size = 0x4000000;
  davinci_model.runtime_param_.logic_mem_base = 0x10000000;
  davinci_model.runtime_param_.mem_base = 0x0;

  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *fftsplusstartctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusstartctx->set_op_index(0);
  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_START));
  domi::FftsPlusAtStartCtxDef *startctxdef = fftsplusstartctx->mutable_at_start_ctx();
  InitAtStartCtx(startctxdef);

  domi::FftsPlusCtxDef *fftsplusendctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusendctx->set_op_index(0);
  fftsplusendctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_END));
  domi::FftsPlusAtEndCtxDef *endctxdef = fftsplusendctx->mutable_at_end_ctx();
  InitAtEndCtx(endctxdef);

  domi::FftsPlusCtxDef *fftspluslabelctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftspluslabelctx->set_op_index(0);
  fftspluslabelctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_LABEL));
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_hardware_notify_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *notifyctx = ffts_plus_task_def->add_ffts_plus_ctx();
  notifyctx->set_op_index(0);
  notifyctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_NOTIFY_WAIT));

  domi::FftsPlusNotifyCtxDef *notifydef = notifyctx->mutable_notify_ctx();
  InitNotifyCtx(notifydef);
  notifydef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_hardware_notify_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *notifyctx = ffts_plus_task_def->add_ffts_plus_ctx();
  notifyctx->set_op_index(0);
  notifyctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_NOTIFY_RECORD));
  domi::FftsPlusNotifyCtxDef *notifydef = notifyctx->mutable_notify_ctx();
  InitNotifyCtx(notifydef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_sdma_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *sdmactx = ffts_plus_task_def->add_ffts_plus_ctx();
  sdmactx->set_op_index(0);
  sdmactx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_SDMA));
  domi::FftsPlusSdmaCtxDef *smdadef = sdmactx->mutable_sdma_ctx();
  InitSdmaCtx(smdadef);
  smdadef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_hardware_sdma_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  davinci_model.runtime_param_.logic_mem_base = 0x1;
  MemAllocation fm_mem_allocation = {0U,
                                     davinci_model.runtime_param_.mem_base,
                                     static_cast<uint64_t>(davinci_model.runtime_param_.mem_size),
                                     ge::MemAllocation::Type::FEATURE_MAP,
                                     0U,
                                     kFmMemType,
                                     0UL};
  davinci_model.logical_mem_allocations_.push_back(fm_mem_allocation);
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *sdmactx = ffts_plus_task_def->add_ffts_plus_ctx();
  sdmactx->set_op_index(0);
  sdmactx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_SDMA));
  domi::FftsPlusSdmaCtxDef *smdadef = sdmactx->mutable_sdma_ctx();
  InitSdmaCtx(smdadef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_writeval_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *writevalctx = ffts_plus_task_def->add_ffts_plus_ctx();
  writevalctx->set_op_index(0);
  writevalctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_WRITE_VALUE));
  domi::FftsPlusWriteValueCtxDef *writedef = writevalctx->mutable_write_value_ctx();
  InitWriteValueCtx(writedef);
  writedef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_hardware_writeval_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *writevalctx = ffts_plus_task_def->add_ffts_plus_ctx();
  writevalctx->set_op_index(0);
  writevalctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_WRITE_VALUE));
  domi::FftsPlusWriteValueCtxDef *writedef = writevalctx->mutable_write_value_ctx();
  InitWriteValueCtx(writedef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_aicpu_ctx_id) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuFwkCtxCtx(aicpudef);
  aicpudef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_aicpu_fwk_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.logic_mem_base = 0x1;
  davinci_model.runtime_param_.mem_size = 0x800;
  MemAllocation fm_mem_allocation = {0U,
                                     davinci_model.runtime_param_.mem_base,
                                     static_cast<uint64_t>(davinci_model.runtime_param_.mem_size),
                                     ge::MemAllocation::Type::FEATURE_MAP,
                                     0U,
                                     kFmMemType,
                                     0UL};
  davinci_model.logical_mem_allocations_.push_back(fm_mem_allocation);
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  OpDescPtr aicpu_node = CreateOpDesc("test", PARTITIONEDCALL);

  GeTensorDesc tensor(GeShape(), FORMAT_ND, DT_INT64);
  TensorUtils::SetSize(tensor, 64);
  aicpu_node->AddInputDesc(tensor);
  std::vector<int64_t> input_offset{100};
  aicpu_node->SetInputOffset(input_offset);
  davinci_model.op_list_[0] = aicpu_node;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuFwkCtxCtx(aicpudef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_aicpu_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 512;

  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  InitTaskSQEInfo(ffts_plus_task_def);

  const auto cpu_op_dsec = CreateOpDesc("Gather", FRAMEWORKOP, 1, 1);
  davinci_model.op_list_[cpu_op_dsec->GetId()] = cpu_op_dsec;
  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(cpu_op_dsec->GetId());
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuCtxCtx(cpu_op_dsec, aicpudef);
  ffts_plus_task_def->set_addr_size(1000);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_data_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *datactx = ffts_plus_task_def->add_ffts_plus_ctx();
  datactx->set_op_index(0);
  datactx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_FLUSH_DATA));
  domi::FftsPlusDataCtxDef *datadef = datactx->mutable_data_ctx();
  InitDataCtx(datadef);
  datadef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_hardware_data_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *datactx = ffts_plus_task_def->add_ffts_plus_ctx();
  datactx->set_op_index(0);
  datactx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_WRITEBACK_DATA));
  domi::FftsPlusDataCtxDef *datadef = datactx->mutable_data_ctx();
  InitDataCtx(datadef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_caseswitch_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *caseswitchctx = ffts_plus_task_def->add_ffts_plus_ctx();
  caseswitchctx->set_op_index(0);
  caseswitchctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_CASE_SWITCH));
  domi::FftsPlusCaseSwitchCtxDef *caseswitchdef = caseswitchctx->mutable_case_switch_ctx();
  InitCaseSwitchCtx(caseswitchdef);
  caseswitchdef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_hardware_caseswitch_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *caseswitchctx = ffts_plus_task_def->add_ffts_plus_ctx();
  caseswitchctx->set_op_index(0);
  caseswitchctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_CASE_SWITCH));
  domi::FftsPlusCaseSwitchCtxDef *caseswitchdef = caseswitchctx->mutable_case_switch_ctx();
  InitCaseSwitchCtx(caseswitchdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, ffts_plus_task_info_hardware_candswitch_ctx_param) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *candswitchctx = ffts_plus_task_def->add_ffts_plus_ctx();
  candswitchctx->set_op_index(0);
  candswitchctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_COND_SWITCH));
  domi::FftsPlusCondSwitchCtxDef *candswitchdef = candswitchctx->mutable_cond_switch_ctx();
  InitCondSwitchCtx(candswitchdef);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}
// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_candswitch_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *candswitchctx = ffts_plus_task_def->add_ffts_plus_ctx();

  candswitchctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_COND_SWITCH));
  domi::FftsPlusCondSwitchCtxDef *candswitchdef = candswitchctx->mutable_cond_switch_ctx();
  InitCondSwitchCtx(candswitchdef);

  candswitchdef->add_true_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, sucess_ffts_plus_task_info_hardware_candswitch_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *candswitchctx = ffts_plus_task_def->add_ffts_plus_ctx();
  candswitchctx->set_op_index(0);
  candswitchctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_COND_SWITCH));
  domi::FftsPlusCondSwitchCtxDef *candswitchdef = candswitchctx->mutable_cond_switch_ctx();
  InitCondSwitchCtx(candswitchdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_hardware_aicaiv_ctx_1) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICORE));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_hardware_aicaiv_ctx_2) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  aicaivdef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}
// test FftsPlusTaskInfo Init cache persistent ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_cache_persistent_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);

  GeTensorDesc invalid_tensor(GeShape(), FORMAT_RESERVED, DT_UNDEFINED);
  TensorUtils::SetSize(invalid_tensor, 512);

  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  op_desc->AddInputDesc(tensor);
  op_desc->AddInputDesc(invalid_tensor);
  op_desc->AddInputDesc(tensor);
  op_desc->SetInputOffset({1024, 2048, 3072});
  AttrUtils::SetInt(op_desc->MutableInputDesc(0U), "_cache_persist", 1);
  AttrUtils::SetInt(op_desc->MutableInputDesc(1U), "_cache_persist", 2);
  AttrUtils::SetInt(op_desc->MutableInputDesc(2U), "_cache_persist", 3);
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *cachepersistentctx = ffts_plus_task_def->add_ffts_plus_ctx();
  cachepersistentctx->set_op_index(0);
  cachepersistentctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_PERSISTENT_CACHE));
  domi::FftsPlusCachePersistCtxDef *cachepersistentdef = cachepersistentctx->mutable_cache_persist_ctx();
  InitCachePersistentCtx(cachepersistentdef);

  cachepersistentdef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_aicaiv_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  auto desc_temp_ptr = std::make_shared<ge::GeTensorDesc>();
  auto desc_temp = *desc_temp_ptr;
  davinci_model.op_list_[0]->AddInputDesc(desc_temp);
  davinci_model.op_list_[0]->AddOutputDesc(desc_temp);
  std::vector<int64_t> workspace;
  workspace.push_back(1024);
  davinci_model.op_list_[0]->SetWorkspace(workspace);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  aicaivdef->add_successor_list(1);
  aicaivdef->add_kernel_name("aivtest");
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, sucess_ffts_plus_task_info_hardware_aicaiv_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(6);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  aicaivdef->add_successor_list(1);
  aicaivdef->add_kernel_name("aivtest");
  aicaivdef->add_src_slot(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_hardware_aicaiv_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(6);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  aicaivdef->add_successor_list(1);
  aicaivdef->add_kernel_name("aivtest");
  aicaivdef->add_src_slot(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, sucess_ffts_plus_task_info_hardware_aicaiv_ctx_with_tiling_data) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  OpDescPtr op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  auto atomic_node = graph->AddNode(op_desc);
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info;
  tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  tiling_info->AddTilingData("666");
  op_desc->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  op_desc->SetExtAttr("atomic_op_run_info", tiling_info);
  // op_desc和atomic_node循环引用导致"Indirect leaks"
  // op_desc->SetExtAttr("memset_node_ptr", atomic_node);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  (void)AttrUtils::SetStr(op_desc, ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string kernel_handle_name = davinci_model.GetBinHandleKey(*op_desc, "", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, nullptr, nullptr);
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(6);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);
  aicaivdef->set_atm(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));

  aicaivdef->add_successor_list(1);
  aicaivdef->add_src_slot(1);

  domi::FftsPlusCtxDef *aicaivctx1 = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx1->set_op_index(0);
  aicaivctx1->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
  aicaivctx1->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef1 = aicaivctx1->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef1);

  aicaivdef1->add_successor_list(1);
  aicaivdef1->add_src_slot(1);
  aicaivdef1->add_kernel_name("aictest");

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);

  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_hardware_mixaicaiv_ctx_1) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaic_a"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaiv_b"].emplace_back(std::make_pair((void *)(0x12435), 2));
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, sucess_ffts_plus_task_info_hardware_mixaicaiv_ctx_memcheck) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1);
  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  AttrUtils::SetBool(op_desc, "_memcheck", true);
  AttrUtils::SetInt(op_desc, "op_para_size", 128);
  AttrUtils::SetInt(op_desc, "ori_op_para_size", 24);
  op_desc->AppendIrInput("x1", ge::kIrInputDynamic);
  op_desc->MutableAllInputName() = {{"x1", 0}};
  auto run_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  run_info->AddTilingData("6");
  run_info->SetWorkspaces({1, 2, 3});
  op_desc->SetExtAttr(ATTR_NAME_OP_RUN_INFO, run_info);
  AttrUtils::SetInt(op_desc, "ori_op_para_size", 24);
  davinci_model.op_list_[0] = op_desc;

  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aic");
  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aiv");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string mix_aic_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aic", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aic_handle_name, nullptr, nullptr);
  std::string mix_aiv_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aiv", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aiv_handle_name, nullptr, nullptr);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);
  ge::ArgsFormatDesc args_format;
  args_format.Append(ge::AddrType::INPUT_DESC, 0);
  auto args_format_str = args_format.ToString();
  mixctxdef->set_args_format(args_format_str);
  mixctxdef->add_task_addr(std::numeric_limits<uint64_t>::max());
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_hardware_mixaicaiv_ctx_2) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaic_a"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaic_b"].emplace_back(std::make_pair((void *)(0x1235), 2));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaiv_a"].emplace_back(std::make_pair((void *)(0x12345), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaiv_b"].emplace_back(std::make_pair((void *)(0x12435), 2));

  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(6);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);

  mixctxdef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}
// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_hardware_mixaicaiv_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaic_a"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaic_b"].emplace_back(std::make_pair((void *)(0x1235), 2));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaiv_a"].emplace_back(std::make_pair((void *)(0x12345), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaiv_b"].emplace_back(std::make_pair((void *)(0x12435), 2));
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1234);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, sucess_ffts_plus_task_info_hardware_mixaicaiv_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  davinci_model.op_list_[0] = op_desc;
  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aic");
  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aiv");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string mix_aic_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aic", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aic_handle_name, nullptr, nullptr);
  std::string mix_aiv_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aiv", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aiv_handle_name, nullptr, nullptr);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);
  mixctxdef->add_task_addr(std::numeric_limits<uint64_t>::max());
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, sucess_ffts_plus_task_info_hardware_auto_mixaicaiv_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  //  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaic_a"].emplace_back(std::make_pair((void *)(0x1245), 1));
  //  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaic_b"].emplace_back(std::make_pair((void *)(0x1235), 2));
  //  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaiv_a"].emplace_back(std::make_pair((void *)(0x12345),
  //  1)); davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mixaiv_b"].emplace_back(std::make_pair((void
  //  *)(0x12435), 2));
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aic");
  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aiv");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string mix_aic_handle_name = davinci_model.GetBinHandleKey(*op_desc, "", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aic_handle_name, nullptr, nullptr);
  std::string mix_aiv_handle_name = davinci_model.GetBinHandleKey(*op_desc, "", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aiv_handle_name, nullptr, nullptr);
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef, true, true);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, sucess_ffts_plus_task_info_hardware_mixaicaiv_ctx_with_tiling_data) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aic");
  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aiv");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string mix_aic_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aic", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aic_handle_name, nullptr, nullptr);
  std::string mix_aiv_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aiv", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aiv_handle_name, nullptr, nullptr);
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info;
  tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  tiling_info->AddTilingData("666");
  op_desc->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  op_desc->SetExtAttr("atomic_op_run_info", tiling_info);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_hardware_ctx_ex) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *casesdefaultctx = ffts_plus_task_def->add_ffts_plus_ctx();
  casesdefaultctx->set_op_index(0);
  casesdefaultctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_CASE_SWITCH));
  domi::FftsPlusCaseDefaultCtxDef *casesdefaultdef = casesdefaultctx->mutable_case_default_ctx();
  InitCaseDefaultCtx(casesdefaultdef);
  casesdefaultdef->add_successor_list(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}
// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_hardware_ctx_ex) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *casesdefaultctx = ffts_plus_task_def->add_ffts_plus_ctx();

  casesdefaultctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_CASE_SWITCH));
  domi::FftsPlusCaseDefaultCtxDef *casesdefaultdef = casesdefaultctx->mutable_case_default_ctx();
  InitCaseDefaultCtx(casesdefaultdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model), SUCCESS);
  task_info.Release();
}
// test FftsPlusTaskInfo UpdateArgs
TEST_F(UtestFftsPlusTaskInfo, failed_ffts_plus_task_info_update_args) {
  DavinciModel davinci_model(0, nullptr);
  FftsPlusTaskInfo task_info;
  task_info.ffts_plus_task_info_.fftsPlusSqe = nullptr;
  task_info.ffts_plus_task_info_.descBuf = nullptr;
  task_info.davinci_model_ = &davinci_model;
  RuntimeParam runtime_param;
  FftsPlusArgsHelper ffts_plus_args_helper(runtime_param);
  std::vector<void *> ext_args;
  task_info.ffts_proto_transfer_ =
      std::make_unique<FftsPlusProtoTransfer>(0, &ffts_plus_args_helper, runtime_param, ext_args);
  domi::TaskDef task_def;
  TaskRunParam task_run_param = {};
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  EXPECT_NE(task_info.UpdateHostArgs({}, nullptr, 0UL), SUCCESS);  // memcpy_s failed.
  task_info.Release();
}

// test FftsPlusTaskInfo ParseTaskRunParam
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_calculate_args) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  task_info.davinci_model_ = &(davinci_model);
  task_info.ffts_plus_task_info_.fftsPlusSqe = nullptr;
  task_info.ffts_plus_task_info_.descBuf = nullptr;
  TaskRunParam task_run_param = {};
  EXPECT_NE(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
}

// test FftsPlusTaskInfo Distribute
TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_distribute) {
  dlog_setlevel(GE_MODULE_NAME, DLOG_DEBUG, 0);
  DavinciModel davinci_model(0, nullptr);
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  task_info.op_desc_ = std::make_shared<OpDesc>();
  task_info.stream_ = stream;
  task_info.davinci_model_ = &davinci_model;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_FALSE(task_info.IsSupportReDistribute());
  dlog_setlevel(GE_MODULE_NAME, DLOG_ERROR, 0);
  rtStreamDestroy(stream);
}

// test FftsPlusTaskInfo FindDumpArgs
TEST_F(UtestFftsPlusTaskInfo, test_get_dump_args) {
  FftsPlusTaskInfo task_info;
  EXPECT_EQ(task_info.FindDumpArgs(std::string("test_op")), 0);
  EXPECT_EQ(task_info.FindDumpArgs(std::string("test")), 0);
}

// test FftsPlusTaskInfo InitDumpArgs
TEST_F(UtestFftsPlusTaskInfo, test_init_dump_args) {
  DavinciModel davinci_model(0, nullptr);
  OpDescPtr op_desc = CreateOpDesc("test", "optp");
  OpDescPtr op_aicpu = CreateOpDesc("test_cput", "optp");
  davinci_model.is_op_debug_reg_ = true;

  FftsPlusTaskInfo task_info;
  task_info.davinci_model_ = &davinci_model;
  task_info.InitDumpArgs(op_desc, 0, 0, {});
  EXPECT_EQ(task_info.dump_args_offset_.size(), 1);
  task_info.InitDumpArgs(op_aicpu, 0x12000000, 0, {});
  EXPECT_EQ(task_info.dump_op_args_.size(), 1);
  task_info.InitDumpArgs(op_aicpu, 0, 0, {0x1});
  EXPECT_EQ(task_info.dump_op_2_first_level_args_.size(), 1);
}

// test FftsPlusTaskInfo Init aicpu ctx
TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_aicpu_fwk_ctx_copy_workspace) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.runtime_param_.mem_size = 10240;
  davinci_model.runtime_param_.mem_base =
      reinterpret_cast<uintptr_t>(new uint8_t[davinci_model.runtime_param_.mem_size]);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuFwkCtxAndExtInfo(aicpudef);

  // Workspace is empty, SUCCESS for dynamic.
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();

  delete[] reinterpret_cast<uint8_t *>(davinci_model.runtime_param_.mem_base);
  davinci_model.runtime_param_.mem_base = 0U;
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_aicpu_fwk_ctx) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.runtime_param_.mem_size = 10240;
  davinci_model.runtime_param_.mem_base =
      reinterpret_cast<uintptr_t>(new uint8_t[davinci_model.runtime_param_.mem_size]);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  davinci_model.op_list_[0]->SetWorkspace({1308});
  davinci_model.op_list_[0]->SetWorkspaceBytes({150});
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuFwkCtxAndExtInfo(aicpudef);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();

  delete[] reinterpret_cast<uint8_t *>(davinci_model.runtime_param_.mem_base);
  davinci_model.runtime_param_.mem_base = 0;
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_dsa_ctx) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.SetFeatureBaseRefreshable(true);
  davinci_model.runtime_param_.mem_size = 10240;
  davinci_model.runtime_param_.mem_base =
      reinterpret_cast<uintptr_t>(new uint8_t[davinci_model.runtime_param_.mem_size]);
  davinci_model.is_op_debug_reg_ = true;  // open dump
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(davinci_model.runtime_param_.mem_base),
                                     davinci_model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  davinci_model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  domi::ModelTaskDef model_task_def;
  auto &task_def = *model_task_def.add_task();
  std::shared_ptr<FftsPlusTaskInfo> ffts_task_info = std::make_shared<FftsPlusTaskInfo>();
  FftsPlusTaskInfo &task_info = *ffts_task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  davinci_model.op_list_[0] = CreateOpDesc("test", DSAGENBITMASK);
  GeTensorDesc ge_tensor_desc(GeShape({10}));

  // dsa has 4 inputs. 1 output. 2 workspace
  std::vector<int64_t> input_offsets;
  for (size_t i = 0; i < kDSAArgsInputAddrSize; ++i) {
    davinci_model.op_list_[0]->AddInputDesc(ge_tensor_desc);
    input_offsets.push_back(i);
  }
  davinci_model.op_list_[0]->SetInputOffset(input_offsets);
  std::vector<int64_t> output_offsets;
  for (size_t i = 0; i < kDSAOutputAddrSize; ++i) {
    davinci_model.op_list_[0]->AddOutputDesc(ge_tensor_desc);
    output_offsets.push_back(i);
  }
  davinci_model.op_list_[0]->SetOutputOffset(output_offsets);
  AttrUtils::SetListInt(davinci_model.op_list_[0], gert::kContextIdList, {0});
  std::vector<int64_t> workspace_offsets;
  std::vector<int64_t> workspace_size;
  for (size_t i = 0; i < kDSAWorkspaceAddrSize; ++i) {
    workspace_offsets.push_back(i);
    workspace_size.push_back(256);
  }
  davinci_model.op_list_[0]->SetWorkspace(workspace_offsets);
  davinci_model.op_list_[0]->SetWorkspaceBytes(workspace_size);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);
  ffts_plus_task_def->_impl_.addr_size_ = 2U;
  domi::FftsPlusCtxDef *dsa_ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  dsa_ctx->set_op_index(0);
  dsa_ctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_DSA));
  domi::FftsPlusDsaCtxDef *dsa_def = dsa_ctx->mutable_dsa_ctx();
  InitDsaCtx(dsa_def, true);
  // init dsa ctx in ffts transfor
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len + 256 + 256;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  // for dump
  task_info.PostProcess(task_def);
  // 这里地址依赖UpdateCtxLevel1Addrs刷新后传递给dump，所以这里是empty
  EXPECT_TRUE(davinci_model.data_dumper_.op_list_.empty());
  const auto &helper = task_info.ffts_flus_args_helper_;
  EXPECT_EQ(task_info.ffts_proto_transfer_->UpdateCtxLevel1Addrs(helper->level1_ctx_addr_infos_, helper->io_addrs_,
                                                                 helper->aicaiv_addr_size_, helper->level1_addr_size_),
            SUCCESS);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_FALSE(infos.empty());
  // for dump
  task_info.PostProcess(task_def);
  task_info.Release();
  // dump可以拿到地址
  EXPECT_FALSE(davinci_model.data_dumper_.op_list_.empty());
  delete[] reinterpret_cast<uint8_t *>(davinci_model.runtime_param_.mem_base);
  davinci_model.runtime_param_.mem_base = 0;
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_dsa_ctx_2) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.SetFeatureBaseRefreshable(true);
  davinci_model.runtime_param_.mem_size = 10240;
  davinci_model.runtime_param_.mem_base =
      reinterpret_cast<uintptr_t>(new uint8_t[davinci_model.runtime_param_.mem_size]);
  davinci_model.is_op_debug_reg_ = true;  // open dump
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(davinci_model.runtime_param_.mem_base),
                                     davinci_model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  davinci_model.logical_mem_allocations_.emplace_back(fm_mem_allocation);
  domi::ModelTaskDef model_task_def;
  auto &task_def = *model_task_def.add_task();
  std::shared_ptr<FftsPlusTaskInfo> ffts_task_info = std::make_shared<FftsPlusTaskInfo>();
  FftsPlusTaskInfo &task_info = *ffts_task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  davinci_model.op_list_[0] = CreateOpDesc("test", DSAGENBITMASK);
  GeTensorDesc ge_tensor_desc(GeShape({10}));

  // dsa 4 inputs. 1 output. 1 workspace
  std::vector<int64_t> input_offsets;
  for (size_t i = 0U; i < kDSAArgsInputAddrSize; ++i) {
    davinci_model.op_list_[0]->AddInputDesc(ge_tensor_desc);
    input_offsets.push_back(i);
  }
  davinci_model.op_list_[0]->SetInputOffset(input_offsets);
  std::vector<int64_t> output_offsets;
  for (size_t i = 0U; i < kDSAOutputAddrSize; ++i) {
    davinci_model.op_list_[0]->AddOutputDesc(ge_tensor_desc);
    output_offsets.push_back(i);
  }
  davinci_model.op_list_[0]->SetOutputOffset(output_offsets);
  AttrUtils::SetListInt(davinci_model.op_list_[0], gert::kContextIdList, {0});
  std::vector<int64_t> workspace_offsets;
  std::vector<int64_t> workspace_size;
  for (size_t i = 0U; i < 1U; ++i) {
    workspace_offsets.push_back(i);
    workspace_size.push_back(256);
  }
  davinci_model.op_list_[0]->SetWorkspace(workspace_offsets);
  davinci_model.op_list_[0]->SetWorkspaceBytes(workspace_size);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);
  ffts_plus_task_def->_impl_.addr_size_ = 2U;
  domi::FftsPlusCtxDef *dsa_ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  dsa_ctx->set_op_index(0);
  dsa_ctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_DSA));
  domi::FftsPlusDsaCtxDef *dsa_def = dsa_ctx->mutable_dsa_ctx();
  InitDsaCtx(dsa_def, true);
  // init dsa ctx in ffts transfor
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len + 256;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);

  task_info.Release();
  delete[] reinterpret_cast<uint8_t *>(davinci_model.runtime_param_.mem_base);
  davinci_model.runtime_param_.mem_base = 0;
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_stateless_dsa_ctx) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.runtime_param_.mem_size = 10240;
  davinci_model.runtime_param_.mem_base =
      reinterpret_cast<uintptr_t>(new uint8_t[davinci_model.runtime_param_.mem_size]);
  domi::ModelTaskDef model_task_def;
  auto &task_def = *model_task_def.add_task();
  std::shared_ptr<FftsPlusTaskInfo> ffts_task_info = std::make_shared<FftsPlusTaskInfo>();
  FftsPlusTaskInfo &task_info = *ffts_task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  task_def.set_type(static_cast<uint32_t>(ModelTaskType::MODEL_TASK_FFTS_PLUS));
  davinci_model.op_list_[0] = CreateOpDesc("test", DSAGENBITMASK);
  GeTensorDesc ge_tensor_desc(GeShape({10}));

  // dsa has 5 inputs. 1 output. 1 workspace
  std::vector<int64_t> input_offsets;
  for (size_t i = 0; i < kDSAStateInputAddrSize; ++i) {
    davinci_model.op_list_[0]->AddInputDesc(ge_tensor_desc);
    input_offsets.push_back(i);
  }
  davinci_model.op_list_[0]->SetInputOffset(input_offsets);
  std::vector<int64_t> output_offsets;
  for (size_t i = 0; i < kDSAOutputAddrSize; ++i) {
    davinci_model.op_list_[0]->AddOutputDesc(ge_tensor_desc);
    output_offsets.push_back(i);
  }
  davinci_model.op_list_[0]->SetOutputOffset(output_offsets);
  std::vector<int64_t> workspace_offsets;
  std::vector<int64_t> workspace_size;
  workspace_offsets.push_back(0);
  workspace_size.push_back(256);

  davinci_model.op_list_[0]->SetWorkspace(workspace_offsets);
  davinci_model.op_list_[0]->SetWorkspaceBytes(workspace_size);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *dsa_ctx = ffts_plus_task_def->add_ffts_plus_ctx();
  dsa_ctx->set_op_index(0);
  dsa_ctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_DSA));
  domi::FftsPlusDsaCtxDef *dsa_def = dsa_ctx->mutable_dsa_ctx();
  InitDsaCtx(dsa_def, true);
  dsa_def->set_input1_value_or_ptr(0);

  // init dsa ctx in ffts transfor
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();

  delete[] reinterpret_cast<uint8_t *>(davinci_model.runtime_param_.mem_base);
  davinci_model.runtime_param_.mem_base = 0;
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_aicpu_fwk_ctx_with_io) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.runtime_param_.mem_size = 10240;
  davinci_model.runtime_param_.mem_base =
      reinterpret_cast<uintptr_t>(new uint8_t[davinci_model.runtime_param_.mem_size]);
  davinci_model.runtime_param_.logic_mem_base = 0x10000000;
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  op_desc->SetInputOffset({1});
  op_desc->SetOutputOffset({100});
  GeTensorDesc input_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(input_desc, 4);
  op_desc->AddInputDesc(input_desc);
  GeTensorDesc output_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  TensorUtils::SetSize(output_desc, 32);
  op_desc->AddOutputDesc(output_desc);
  op_desc->SetId(0);
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
  davinci_model.op_list_[0] = op_desc;
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuFwkCtxAndExtInfo(aicpudef);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
  delete[] reinterpret_cast<uint8_t *>(davinci_model.runtime_param_.mem_base);
  davinci_model.runtime_param_.mem_base = 0;
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_aicpu_op_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  task_def.set_stream_id(0);
  davinci_model.stream_list_ = {stream};
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetBool(op_desc, "_AllShape", true);
  op_desc->SetId(0);
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(16);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuCtxAndExtInfo(aicpudef);
  ffts_plus_task_def->set_addr_size(1000);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_aicpu_block_op_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  task_def.set_stream_id(0);
  davinci_model.stream_list_ = {stream};
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetBool(op_desc, "_AllShape", true);
  AttrUtils::SetBool(op_desc, "_is_blocking_op", true);
  op_desc->SetId(0);
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(16);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuCtxAndExtInfo(aicpudef);
  ffts_plus_task_def->set_addr_size(1000);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_aicpu_block_op_ctx_rts) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  task_def.set_stream_id(0);
  davinci_model.stream_list_ = {stream};
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetBool(op_desc, "_AllShape", true);
  AttrUtils::SetBool(op_desc, "_is_blocking_op", true);
  op_desc->SetId(0);
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(16);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_MODE_NO_FFTS);
  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuCtxAndExtInfo(aicpudef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_NE(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, fail_ffts_plus_task_info_aicpu_block_op_ctx_not_support) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  task_def.set_stream_id(0);
  davinci_model.stream_list_ = {stream};
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetBool(op_desc, "_AllShape", true);
  AttrUtils::SetBool(op_desc, "_is_blocking_op", true);
  op_desc->SetId(0);
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(16);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);
  RTS_STUB_OUTBOUND_VALUE(rtGetDeviceCapability, int32_t, value, RT_AICPU_BLOCKING_OP_NOT_SUPPORT);
  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuCtxAndExtInfo(aicpudef);
  ffts_plus_task_def->set_addr_size(1000);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_custom_aicpu_op_ctx) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.ge_model_ = MakeShared<GeModel>();
  const auto model_def = MakeShared<domi::ModelTaskDef>();
  davinci_model.ge_model_->SetModelTaskDef(model_def);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  task_def.set_stream_id(0);
  davinci_model.stream_list_ = {stream};
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetBool(op_desc, "_AllShape", true);
  op_desc->SetId(0);
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(16);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitCustomAicpuCtxAndExtInfo(aicpudef);
  ffts_plus_task_def->set_addr_size(1000);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_task_info_aicpu_op_ctx_with_io) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x4000000;
  davinci_model.runtime_param_.logic_mem_base = 0x10000000;
  davinci_model.runtime_param_.mem_base = 0x0;
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  auto op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  op_desc->SetInputOffset({1});
  op_desc->SetOutputOffset({100});

  GeTensorDesc input_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(input_desc, 4);
  op_desc->AddInputDesc(input_desc);
  GeTensorDesc output_desc(GeShape({1, 1, 1, 1}), FORMAT_NCHW, DT_FLOAT16);
  TensorUtils::SetSize(output_desc, 32);
  op_desc->AddOutputDesc(output_desc);
  op_desc->SetId(0);
  op_desc->SetWorkspace({1308});
  op_desc->SetWorkspaceBytes({150});
  davinci_model.op_list_[0] = op_desc;

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(16);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(0);
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuCtxAndExtInfo(aicpudef);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.Release();
}

// test FftsPlusTaskInfo Init hardware ctx
TEST_F(UtestFftsPlusTaskInfo, update_ctx_level1_addr_success) {
  DavinciModel davinci_model(0, nullptr);
  davinci_model.SetFeatureBaseRefreshable(true);
  domi::ModelTaskDef model_task_def;
  auto &task_def = *model_task_def.add_task();
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  davinci_model.runtime_param_.mem_base = 0x56789;
  MemAllocation fm_mem_allocation = {0U,
                                     davinci_model.runtime_param_.mem_base,
                                     static_cast<uint64_t>(davinci_model.runtime_param_.mem_size),
                                     ge::MemAllocation::Type::FEATURE_MAP,
                                     0U,
                                     kFmMemType,
                                     0UL};
  davinci_model.logical_mem_allocations_.push_back(fm_mem_allocation);
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetInt(davinci_model.op_list_[0], "_parallel_group_id", 1);

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);
  InitTaskSQEInfo(ffts_plus_task_def);
  // sdma
  domi::FftsPlusCtxDef *sdmactx = ffts_plus_task_def->add_ffts_plus_ctx();
  sdmactx->set_op_index(0);
  sdmactx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_SDMA));
  domi::FftsPlusSdmaCtxDef *smdadef = sdmactx->mutable_sdma_ctx();
  InitSdmaCtx(smdadef);
  smdadef->set_src_addr_base(0x100);
  smdadef->set_dst_addr_base(0x200);

  // cmo
  domi::FftsPlusCtxDef *datactx = ffts_plus_task_def->add_ffts_plus_ctx();
  datactx->set_op_index(0);
  datactx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_FLUSH_DATA));
  domi::FftsPlusDataCtxDef *datadef = datactx->mutable_data_ctx();
  InitDataCtx(datadef);
  datadef->set_addr_base(0x300);

  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_FALSE(infos.empty());
  task_info.Release();

  EXPECT_EQ(task_info.UpdateHostArgs({0x12345}, nullptr, 0UL), SUCCESS);
  rtFftsPlusSdmaCtx_t *const ctx = PtrToPtr<rtFftsPlusComCtx_t, rtFftsPlusSdmaCtx_t>(
      task_info.ffts_flus_args_helper_->level1_ctx_addr_infos_[0U].rts_ctx);
  const uint32_t addr_high = static_cast<uint32_t>((uint64_t)0x12445 >> 32U);
  const uint32_t addr_low = static_cast<uint32_t>(0x12445 & 0xFFFFFFFFU);
  EXPECT_EQ(ctx->sourceAddressBaseH, addr_high);
  EXPECT_EQ(ctx->sourceAddressBaseL, addr_low);
}

static void BuildFftsPlusGraphWithSlice2(ComputeGraphPtr &root_graph, ComputeGraphPtr &ffts_subgraph) {
  DEF_GRAPH(g0) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
  };
  root_graph = ToComputeGraph(g0);
  root_graph->SetGraphUnknownFlag(false);
  const auto root_call_0 = root_graph->FindNode("PartitionedCall_0");
  EXPECT_NE(root_call_0, nullptr);

  DEF_GRAPH(g1) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto conv_0 = OP_CFG(CONV2D)
                      .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto cast_slice0 = OP_CFG(CAST)
                           .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                           .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                           .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto cast_slice1 = OP_CFG(CAST)
                           .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIV")
                           .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                           .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
    auto pc = OP_CFG("PhonyConcat");

    CHAIN(NODE("sgt_graph/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/cast_slice0", cast_slice0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/pc", pc)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT));
    CHAIN(NODE("sgt_graph/Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/cast_slice1", cast_slice1)
              ->EDGE(0, 1)
              ->NODE("sgt_graph/pc", pc));
  };
  ffts_subgraph = ToComputeGraph(g1);
  ffts_subgraph->SetGraphUnknownFlag(false);

  const auto conv2d = ffts_subgraph->FindNode("sgt_graph/Conv2D");
  GeTensorDesc tensor(GeShape({2, 2, 2, 2}), FORMAT_NCHW, DT_FLOAT);
  TensorUtils::SetSize(tensor, 64);
  conv2d->GetOpDesc()->AddInputDesc(tensor);
  conv2d->GetOpDesc()->SetInputOffset({0});
  conv2d->GetOpDesc()->AddOutputDesc(tensor);
  conv2d->GetOpDesc()->SetOutputOffset({1024});

  const auto cast_slice0 = ffts_subgraph->FindNode("sgt_graph/cast_slice0");
  EXPECT_NE(cast_slice0, nullptr);
  GeTensorDesc tensor_slice0(GeShape({1, 2, 2, 2}), FORMAT_NCHW, DT_FLOAT);
  AttrUtils::SetInt(tensor_slice0, ATTR_NAME_INNER_OFFSET, 0);
  TensorUtils::SetSize(tensor_slice0, 32);
  cast_slice0->GetOpDesc()->AddInputDesc(tensor_slice0);
  cast_slice0->GetOpDesc()->SetInputOffset({1024});
  cast_slice0->GetOpDesc()->AddOutputDesc(tensor_slice0);
  cast_slice0->GetOpDesc()->SetOutputOffset({2048});

  const auto cast_slice1 = ffts_subgraph->FindNode("sgt_graph/cast_slice1");
  EXPECT_NE(cast_slice1, nullptr);
  GeTensorDesc tensor_slice1(GeShape({1, 2, 2, 2}), FORMAT_NCHW, DT_FLOAT);
  AttrUtils::SetInt(tensor_slice1, ATTR_NAME_INNER_OFFSET, 32);
  TensorUtils::SetSize(tensor_slice1, 32);
  cast_slice1->GetOpDesc()->AddInputDesc(tensor_slice1);
  cast_slice1->GetOpDesc()->SetInputOffset({1056});
  cast_slice1->GetOpDesc()->AddOutputDesc(tensor_slice1);
  cast_slice1->GetOpDesc()->SetOutputOffset({2080});

  const auto pc = ffts_subgraph->FindNode("sgt_graph/pc");
  EXPECT_NE(pc, nullptr);
  GeTensorDesc tensor_pc(GeShape({1, 2, 2, 2}), FORMAT_NCHW, DT_FLOAT);

  AttrUtils::SetBool(pc->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, true);
  TensorUtils::SetSize(tensor_pc, 32);
  pc->GetOpDesc()->AddInputDesc(tensor_pc);
  pc->GetOpDesc()->AddInputDesc(tensor_pc);
  pc->GetOpDesc()->SetInputOffset({1024, 1056});
}

UINT32 StubTilingFFTS(gert::TilingContext *context) {
  context->SetNeedAtomic(false);
  context->SetTilingKey(666U);
  context->SetBlockDim(666U);
  auto tiling_data = context->GetTilingData<uint64_t>();
  *tiling_data = 100;
  return ge::GRAPH_SUCCESS;
}

UINT32 StubTilingParseFFTS(gert::KernelContext *context) {
  (void)context;
  return ge::GRAPH_SUCCESS;
}

void *CompileInfoCreatorFFTS() {
  auto tmp = ge::MakeUnique<char>();
  return tmp.get();
}

// test ffts_manual ffts subgraph with slice2
TEST_F(UtestFftsPlusTaskInfo, manual_ffts_subgraph_args_init) {
  gert::SpaceRegistryFaker::CreateDefaultSpaceRegistryImpl2();
  auto space_registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry();
  ASSERT_NE(space_registry, nullptr);
  auto funcs = space_registry->CreateOrGetOpImpl(CAST);
  ASSERT_NE(funcs, nullptr);

  funcs->tiling = StubTilingFFTS;
  funcs->tiling_parse = StubTilingParseFFTS;
  funcs->compile_info_creator = CompileInfoCreatorFFTS;
  funcs->compile_info_deleter = nullptr;

  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_subgraph;
  BuildFftsPlusGraphWithSlice2(root_graph, ffts_subgraph);
  const auto cast_slice1 = ffts_subgraph->FindNode("sgt_graph/cast_slice1");
  const auto op_desc = cast_slice1->GetOpDesc();

  AttrUtils::SetBool(op_desc, ATTR_NAME_STATIC_TO_DYNAMIC_SOFT_SYNC_OP, true);
  AttrUtils::SetStr(op_desc, ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AiCore");
  std::string json_str = R"({"_sgt_cube_vector_core_type": "AiCore"})";
  AttrUtils::SetStr(op_desc, "compile_info_json", json_str);
  AttrUtils::SetInt(op_desc, "op_para_size", 512);
  std::vector<char> kernelBin;
  TBEKernelPtr tbe_kernel = std::make_shared<ge::OpKernelBin>("name/data", std::move(kernelBin));
  op_desc->SetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, tbe_kernel);
  AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");

  auto root_call_0 = root_graph->FindNode("PartitionedCall_0");
  AttrUtils::SetBool(root_call_0->GetOpDesc(), ATTR_NAME_FFTS_PLUS_SUB_GRAPH, true);
  root_call_0->GetOpDesc()->AddSubgraphName(ffts_subgraph->GetName());
  root_call_0->GetOpDesc()->SetSubgraphInstanceName(0, ffts_subgraph->GetName());
  ffts_subgraph->SetParentNode(root_call_0);
  ffts_subgraph->SetParentGraph(root_graph);
  root_graph->AddSubgraph(ffts_subgraph);

  DavinciModel davinci_model(0, nullptr);
  GeModelPtr ge_model = MakeShared<GeModel>();
  ge_model->SetGraph(root_graph);
  davinci_model.ge_model_ = ge_model;

  auto space_registry_array = ge::MakeShared<gert::OpImplSpaceRegistryV2Array>();
  ASSERT_NE(space_registry_array, nullptr);
  space_registry_array->at(static_cast<size_t>(OppImplVersion::kOpp)) = space_registry;
  davinci_model.SetSpaceRegistries(space_registry_array);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(root_call_0->GetOpDesc()->GetId());
  std::shared_ptr<optiling::utils::OpRunInfo> tiling_info;
  tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  root_call_0->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  AttrUtils::SetBool(root_call_0->GetOpDesc(), "_kernel_list_first_name", true);

  davinci_model.op_list_[root_call_0->GetOpDesc()->GetId()] = root_call_0->GetOpDesc();

  ffts_plus_task_def->set_addr_size(16);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(cast_slice1->GetOpDesc()->GetId());
  davinci_model.op_list_[cast_slice1->GetOpDesc()->GetId()] = cast_slice1->GetOpDesc();
  davinci_model.operator_list_[cast_slice1->GetOpDesc()->GetId()] =
      MakeShared<Operator>(OpDescUtils::CreateOperatorFromNode(cast_slice1));
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);
  aicaivdef->set_atm(0);

  domi::FftsPlusCtxDef *aicaivctx1 = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx1->set_op_type(domi::FftsPlusCtxDef::ATOMIC);
  aicaivctx1->set_op_index(cast_slice1->GetOpDesc()->GetId());
  aicaivctx1->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef1 = aicaivctx1->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef1);
  aicaivdef1->set_atm(0);

  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  (void)AttrUtils::SetStr(cast_slice1->GetOpDesc(), "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aic_kernel");
  (void)AttrUtils::SetStr(cast_slice1->GetOpDesc(), "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aiv_kernel");
  TBEHandleStore::GetInstance().StoreTBEHandle("mix_aic_kernel", nullptr, nullptr);
  TBEHandleStore::GetInstance().StoreTBEHandle("mix_aiv_kernel", nullptr, nullptr);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);

  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  PisToPersistentWorkspace persistant_workspace;
  int64_t persist_dev[512] = {0};
  persistant_workspace[0].dev_addr = reinterpret_cast<uint64_t>(persist_dev);
  persistant_workspace[0].len = 512;
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace), SUCCESS);
  task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, ffts_plus_task_info_get_block_dim) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *fftsplusstartctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusstartctx->set_op_index(0);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICORE));
  fftsplusstartctx->mutable_aic_aiv_ctx()->set_non_tail_block_dim(1);
  fftsplusstartctx->mutable_aic_aiv_ctx()->set_tail_block_dim(2);
  EXPECT_EQ(davinci_model.GetBlockDim(*fftsplusstartctx), 1);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  fftsplusstartctx->mutable_mix_aic_aiv_ctx()->set_non_tail_block_dim(5);
  fftsplusstartctx->mutable_mix_aic_aiv_ctx()->set_tail_block_dim(6);
  fftsplusstartctx->mutable_mix_aic_aiv_ctx()->set_non_tail_block_ratio_n(9);
  EXPECT_EQ(davinci_model.GetBlockDim(*fftsplusstartctx), 0x90005);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  fftsplusstartctx->mutable_aicpu_ctx()->set_non_tail_block_dim(9);
  fftsplusstartctx->mutable_aicpu_ctx()->set_tail_block_dim(10);
  EXPECT_EQ(davinci_model.GetBlockDim(*fftsplusstartctx), 9);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_START));
  EXPECT_EQ(davinci_model.GetBlockDim(*fftsplusstartctx), 0);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_WRITEBACK_DATA));
  EXPECT_EQ(davinci_model.GetBlockDim(*fftsplusstartctx), 0);
}

TEST_F(UtestFftsPlusTaskInfo, ffts_plus_task_info_get_thread) {
  DavinciModel davinci_model(0, nullptr);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  domi::TaskDef task_def;
  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(2);

  InitTaskSQEInfo(ffts_plus_task_def);
  domi::FftsPlusCtxDef *fftsplusstartctx = ffts_plus_task_def->add_ffts_plus_ctx();
  fftsplusstartctx->set_op_index(0);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICORE));
  fftsplusstartctx->mutable_aic_aiv_ctx()->set_thread_id(3);
  fftsplusstartctx->mutable_aic_aiv_ctx()->set_thread_dim(4);
  EXPECT_EQ(davinci_model.GetThreadId(*fftsplusstartctx), 3);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  fftsplusstartctx->mutable_mix_aic_aiv_ctx()->set_thread_id(7);
  fftsplusstartctx->mutable_mix_aic_aiv_ctx()->set_thread_dim(8);
  EXPECT_EQ(davinci_model.GetThreadId(*fftsplusstartctx), 7);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  fftsplusstartctx->mutable_aicpu_ctx()->set_thread_id(11);
  fftsplusstartctx->mutable_aicpu_ctx()->set_thread_dim(12);
  EXPECT_EQ(davinci_model.GetThreadId(*fftsplusstartctx), 11);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AT_START));
  EXPECT_EQ(davinci_model.GetThreadId(*fftsplusstartctx), 0);

  fftsplusstartctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_WRITEBACK_DATA));
  EXPECT_EQ(davinci_model.GetThreadId(*fftsplusstartctx), 0);
}

TEST_F(UtestFftsPlusTaskInfo, FftsPlusTaskInfo_MixAicAiv_MixEnhanced_Init_Success) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo ffts_plus_task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.device_id_ = 0U;
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);
  MemAllocation fm_mem_allocation = {0, 0U, 0x45678U, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  davinci_model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mix_a"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mix_a"].emplace_back(std::make_pair((void *)(0x1235), 2));
  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
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
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_enhanced"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  davinci_model.op_list_[0] = op_desc;
  (void)AttrUtils::SetStr(op_desc, "_mix" + ATTR_NAME_TBE_KERNEL_NAME, "mix_kernel");
  TBEHandleStore::GetInstance().StoreTBEHandle("mix_kernel", nullptr, nullptr);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtxForSingleKernel(mixctxdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = 1024;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  std::vector<TaskArgsRefreshInfo> infos;
  EXPECT_EQ(ffts_plus_task_info.GetTaskArgsRefreshInfos(infos), SUCCESS);
  EXPECT_FALSE(infos.empty());
  ffts_plus_task_info.Release();
}

// mix_enhanced 1:0 or 0:1场景
TEST_F(UtestFftsPlusTaskInfo, FftsPlusTaskInfo_MixAicAiv_MixEnhanced_Init_Success_v2) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo ffts_plus_task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.device_id_ = 0U;
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);
  // 只有一组pc
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["mix_a"].emplace_back(std::make_pair((void *)(0x1245), 1));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_enhanced"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  davinci_model.op_list_[0] = op_desc;
  (void)AttrUtils::SetStr(op_desc, "_mix" + ATTR_NAME_TBE_KERNEL_NAME, "mix_kernel");
  TBEHandleStore::GetInstance().StoreTBEHandle("mix_kernel", nullptr, nullptr);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtxForSingleKernel(mixctxdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(ffts_plus_task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = 1024;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(ffts_plus_task_info.Init(task_def, &davinci_model, args), SUCCESS);
  ffts_plus_task_info.Release();
}

TEST_F(UtestFftsPlusTaskInfo, FftsPlusTaskInfo_append_aicpu_value) {
  uint8_t src[80] = {0};
  RuntimeParam param;
  FftsPlusArgsHelper args_helper(param);
  args_helper.bin_args_host_ = src;
  args_helper.bin_args_size_ = 80;
  args_helper.args_relevent_offsets_ = {10, 18};
  EXPECT_NE(args_helper.UpdateAicpuAddrs({0x100, 0x200, 0x300}, 2UL), SUCCESS);
  EXPECT_EQ(args_helper.UpdateAicpuAddrs({0x100, 0x200, 0x300}, 1UL), SUCCESS);
  uint64_t *addr1 = reinterpret_cast<uint64_t *>(&src[10]);
  uint64_t *addr2 = reinterpret_cast<uint64_t *>(&src[18]);
  EXPECT_EQ(*addr1, 0x200);
  EXPECT_EQ(*addr2, 0x300);

  //
  args_helper.used_bin_args_size_ = 100;
  uint8_t kernel[100] = {0};
  EXPECT_NE(args_helper.AppendBinArgs(kernel, 100), SUCCESS);
}

TEST_F(UtestFftsPlusTaskInfo, success_dump_ffts_plus_aicaiv_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetListInt(davinci_model.op_list_[0], gert::kContextIdList, {0});
  davinci_model.is_op_debug_reg_ = true;  // open dump

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(6);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AIV));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  aicaivdef->add_successor_list(1);
  aicaivdef->add_kernel_name("aivtest");
  aicaivdef->add_src_slot(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.PostProcess(task_def);
  task_info.Release();
  EXPECT_FALSE(davinci_model.data_dumper_.op_list_.empty());
}

TEST_F(UtestFftsPlusTaskInfo, success_dump_ffts_plus_cmo_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aictest"].emplace_back(std::make_pair((void *)(0x1245), 1));
  davinci_model.bin_kernel_handle_.addr_and_pref_cnt_["aivtest"].emplace_back(std::make_pair((void *)(0x1235), 2));
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  davinci_model.op_list_[0] = CreateOpDesc("test", PARTITIONEDCALL);
  AttrUtils::SetListInt(davinci_model.op_list_[0], gert::kContextIdList, {0});
  davinci_model.is_op_debug_reg_ = true;  // open dump

  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(6);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *aicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicaivctx->set_op_index(0);
  aicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_WRITEBACK_DATA));
  domi::FftsPlusAicAivCtxDef *aicaivdef = aicaivctx->mutable_aic_aiv_ctx();
  InitAicAivCtx(aicaivdef);

  aicaivdef->add_successor_list(1);
  aicaivdef->add_kernel_name("aivtest");
  aicaivdef->add_src_slot(1);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.PostProcess(task_def);
  task_info.Release();
  EXPECT_TRUE(davinci_model.data_dumper_.op_list_.empty());
}

// aclmdlCreateAndGetOpDesc will call GetOpDescInfo on exception occurs, even the L1 exception dump is not enabled.
TEST_F(UtestFftsPlusTaskInfo, success_dump_ffts_plus_mixaiv_ctx) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);  // disable l1 exception
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  davinci_model.op_list_[0] = op_desc;
  AttrUtils::SetListInt(davinci_model.op_list_[0], gert::kContextIdList, {0});
  davinci_model.is_op_debug_reg_ = true;  // open dump
  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aic");
  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aiv");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string mix_aic_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aic", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aic_handle_name, nullptr, nullptr);
  std::string mix_aiv_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aiv", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aiv_handle_name, nullptr, nullptr);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);

  task_info.PostProcess(task_def);
  EXPECT_EQ(davinci_model.exception_dumper_.op_desc_info_.size(), 1UL);
  EXPECT_EQ(davinci_model.exception_dumper_.op_desc_info_[0UL].args_size, 48UL);
  task_info.Release();
  EXPECT_FALSE(davinci_model.data_dumper_.op_list_.empty());
}

TEST_F(UtestFftsPlusTaskInfo, success_dump_ffts_plus_mixaic_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  const auto &op_desc = CreateOpDesc("test", PARTITIONEDCALL);
  std::vector<std::string> name_prefix = {"_mix_aic", "_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  davinci_model.op_list_[0] = op_desc;
  AttrUtils::SetListInt(davinci_model.op_list_[0], gert::kContextIdList, {0});
  davinci_model.is_op_debug_reg_ = true;  // open dump
  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aic");
  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aiv");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string mix_aic_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aic", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aic_handle_name, nullptr, nullptr);
  std::string mix_aiv_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aiv", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aiv_handle_name, nullptr, nullptr);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(10);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIC));
  domi::FftsPlusMixAicAivCtxDef *mixctxdef = mixaicaivctx->mutable_mix_aic_aiv_ctx();
  InitMixAicAivCtx(mixctxdef);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.PostProcess(task_def);
  task_info.Release();
  EXPECT_FALSE(davinci_model.data_dumper_.op_list_.empty());
}

TEST_F(UtestFftsPlusTaskInfo, success_dump_ffts_plus_aicpu_ctx) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 512;

  task_def.set_stream_id(0);

  const auto sgt_op_desc = CreateOpDesc("test", PARTITIONEDCALL, 1, 1);
  davinci_model.op_list_[sgt_op_desc->GetId()] = sgt_op_desc;

  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();
  ffts_plus_task_def->set_op_index(sgt_op_desc->GetId());
  ffts_plus_task_def->set_addr_size(sgt_op_desc->GetInputsSize() + sgt_op_desc->GetOutputsSize());
  InitTaskSQEInfo(ffts_plus_task_def);

  const auto cpu_op_dsec = CreateOpDesc("Gather", FRAMEWORKOP, 1, 1);
  davinci_model.op_list_[cpu_op_dsec->GetId()] = cpu_op_dsec;
  davinci_model.is_op_debug_reg_ = true;  // open dump
  domi::FftsPlusCtxDef *aicpuctx = ffts_plus_task_def->add_ffts_plus_ctx();
  aicpuctx->set_op_index(cpu_op_dsec->GetId());
  aicpuctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_AICPU));
  domi::FftsPlusAicpuCtxDef *aicpudef = aicpuctx->mutable_aicpu_ctx();
  InitAicpuCtxCtx(cpu_op_dsec, aicpudef);
  ffts_plus_task_def->set_addr_size(1000);
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);
  task_info.PostProcess(task_def);
  task_info.Release();
  EXPECT_FALSE(davinci_model.data_dumper_.op_list_.empty());
}

static void SetAicAivOpKernel(const ComputeGraphPtr &graph, const std::string name,
                              TBEKernelStore *kernel_store = nullptr) {
  const auto &node = graph->FindNode(name);
  EXPECT_NE(node, nullptr);
  const auto &op_desc = node->GetOpDesc();
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM));
  std::vector<char> aic_kernel_bin(64, '\0');
  std::vector<char> aiv_kernel_bin(64, '\0');
  std::vector<std::string> thread_kernel_names = {"aictest", "aivtest"};
  (void)AttrUtils::SetListStr(op_desc, "_thread_kernelname", thread_kernel_names);

  std::vector<TBEKernelPtr> tbe_kernel_vec{
      std::make_shared<ge::OpKernelBin>(thread_kernel_names[0], std::move(aic_kernel_bin)),
      std::make_shared<ge::OpKernelBin>(thread_kernel_names[1], std::move(aiv_kernel_bin))};
  if (kernel_store == nullptr) {
    op_desc->SetExtAttr(OP_EXTATTR_NAME_THREAD_TBE_KERNEL, tbe_kernel_vec);
  } else {
    kernel_store->AddTBEKernel(tbe_kernel_vec[0]);
    kernel_store->AddTBEKernel(tbe_kernel_vec[1]);
  }

  std::vector<string> bin_file_keys{op_desc->GetName() + "_aic", op_desc->GetName() + "_aiv"};
  (void)AttrUtils::SetListStr(op_desc, "_register_stub_func", bin_file_keys);
  (void)AttrUtils::SetStr(op_desc, op_desc->GetName() + "_kernelname", op_desc->GetName());
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_MODE, 1);
  (void)AttrUtils::SetInt(op_desc, ATTR_NAME_THREAD_SCOPE_ID, 1);
  // Init Binary Magic
  std::vector<std::string> json_list{"RT_DEV_BINARY_MAGIC_ELF_AIVEC", "RT_DEV_BINARY_MAGIC_ELF_AICUBE"};
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_magic", json_list);
  // Init meta data
  std::vector<std::string> meta_data_list{"AIVEC_META_DATA", "AICUBE_META_DATA"};
  (void)AttrUtils::SetListStr(op_desc, "_thread_tvm_metadata", meta_data_list);
}

static void BuildFftsPlusGraph(ComputeGraphPtr &root_graph, ComputeGraphPtr &ffts_plus_graph,
                               TBEKernelStore *kernel_store = nullptr) {
  uint32_t mem_offset = 0U;
  DEF_GRAPH(g1) {
    CHAIN(NODE("_arg_0", DATA)->NODE("PartitionedCall_0", PARTITIONEDCALL)->NODE("Node_Output", NETOUTPUT));
    CHAIN(NODE("_arg_1", DATA)->NODE("PartitionedCall_0"));
    CHAIN(NODE("_arg_2", DATA)->NODE("PartitionedCall_0"));
  };
  root_graph = ToComputeGraph(g1);
  SetUnknownOpKernel(root_graph, mem_offset, true);

  DEF_GRAPH(g2) {
    auto data_0 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 0);
    auto data_1 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 1);
    auto data_2 = OP_CFG(DATA).Attr(ATTR_NAME_PARENT_NODE_INDEX, 2);
    auto conv_0 = OP_CFG(CONV2D)
                      .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::TVM))
                      .Attr(ATTR_NAME_CUBE_VECTOR_CORE_TYPE, "AIC")
                      .Attr(TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF")
                      .Attr("_mix_with_enhanced_kernel", true);
    auto relu_0 = OP_CFG(RELU).Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU));
    auto add_0 = OP_CFG(ADD)
                     .Attr(ATTR_NAME_IMPLY_TYPE, static_cast<int64_t>(domi::ImplyType::AI_CPU))
                     .Attr("_is_blocking_op", true);
    CHAIN(NODE("sgt_graph/_arg_0", data_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Conv2D", conv_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Add", add_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Relu", relu_0)
              ->EDGE(0, 0)
              ->NODE("sgt_graph/Node_Output", NETOUTPUT));
    auto const_0 = OP_CFG(CONSTANT);
    CHAIN(NODE("sgt_graph/_arg_1", data_1)->EDGE(0, 1)->NODE("sgt_graph/Conv2D", conv_0));
    CHAIN(NODE("sgt_graph/_const_0", const_0)->EDGE(0, 2)->NODE("sgt_graph/Conv2D", conv_0));
    CHAIN(NODE("sgt_graph/_arg_2", data_2)->EDGE(0, 1)->NODE("sgt_graph/Add", add_0));
  };
  ffts_plus_graph = ToComputeGraph(g2);
  auto const_node = ffts_plus_graph->FindFirstNodeMatchType(CONSTANT);
  GeTensor tensor(const_node->GetOpDesc()->GetOutputDesc(0));
  std::vector<int64_t> tensor_data({1, 2, 2});
  tensor.SetData(reinterpret_cast<uint8_t *>(tensor_data.data()), tensor_data.size() * sizeof(int64_t));
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", tensor);

  SetUnknownOpKernel(ffts_plus_graph, mem_offset);
  AddPartitionedCall(root_graph, "PartitionedCall_0", ffts_plus_graph);
  (void)AttrUtils::SetListInt(ffts_plus_graph->FindNode("sgt_graph/Conv2D")->GetOpDesc(), gert::kContextIdList,
                              {0, 1, 2, 3});
  SetAicAivOpKernel(ffts_plus_graph, "sgt_graph/Conv2D", kernel_store);
}

TEST_F(UtestFftsPlusTaskInfo, success_ffts_plus_profiling) {
  ComputeGraphPtr root_graph;
  ComputeGraphPtr ffts_plus_graph;
  TBEKernelStore tbe_kernel_store;
  BuildFftsPlusGraph(root_graph, ffts_plus_graph, &tbe_kernel_store);

  // Build FftsTaskDef.
  std::shared_ptr<domi::ModelTaskDef> model_task_def = MakeShared<domi::ModelTaskDef>();
  auto &task_def = *model_task_def->add_task();
  InitFftsplusTaskDef(ffts_plus_graph, task_def);
  auto &ffts_plus_task_def = *task_def.mutable_ffts_plus_task();
  auto &aic_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicCtxDef(ffts_plus_graph, aic_ctx_def, "sgt_graph/Conv2D");
  aic_ctx_def.set_uniq_ctx_name("testtesttest1");
  auto &aicpu_ctx_def = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuCtxDef(ffts_plus_graph, aicpu_ctx_def, "sgt_graph/Add");
  aicpu_ctx_def.set_uniq_ctx_name("testtesttest2");
  auto &aicpu_ctx_def_relu = *ffts_plus_task_def.add_ffts_plus_ctx();
  InitFftsPlusAicpuFwkCtxDef(ffts_plus_graph, aicpu_ctx_def_relu, "sgt_graph/Relu");
  aicpu_ctx_def_relu.set_uniq_ctx_name("testtesttest2");
  ffts_plus_task_def.set_addr_size(1000);

  const auto aic_node = ffts_plus_graph->FindNode("sgt_graph/Conv2D");
  EXPECT_NE(aic_node, nullptr);
  (void)ge::AttrUtils::SetListInt(aic_node->GetOpDescBarePtr()->MutableInputDesc(0), gert::kTensorCtxId, {1});
  (void)ge::AttrUtils::SetListInt(aic_node->GetOpDescBarePtr()->MutableOutputDesc(0), gert::kTensorCtxId, {2});

  // Build GeModel.
  GeModelPtr ge_model = std::make_shared<GeModel>();
  ge_model->SetName(root_graph->GetName());
  ge_model->SetModelTaskDef(model_task_def);
  ge_model->SetGraph(root_graph);
  ge_model->SetTBEKernelStore(tbe_kernel_store);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_MEMORY_SIZE, 10240);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_STREAM_NUM, 1);
  AttrUtils::SetInt(ge_model, ATTR_MODEL_VAR_SIZE, 5120);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);
  RTS_STUB_RETURN_VALUE(rtQueryFunctionRegistered, rtError_t, 0x78000001);

  GeRootModelPtr ge_root_model = MakeShared<GeRootModel>();
  EXPECT_EQ(ge_root_model->Initialize(root_graph), SUCCESS);
  ge_root_model->SetSubgraphInstanceNameToModel(root_graph->GetName(), ge_model);

  GraphId graph_id = 1001;
  GraphNodePtr graph_node = MakeShared<GraphNode>(graph_id);
  graph_node->SetGeRootModel(ge_root_model);
  graph_node->SetLoadFlag(true);
  graph_node->SetAsync(true);

  // Test for Load.
  ModelExecutor model_executor;
  ASSERT_EQ(model_executor.Initialize({}, 0), SUCCESS);
  model_executor.StartRunThread();
  EXPECT_EQ(model_executor.LoadGraph(ge_root_model, graph_node), SUCCESS);
  const auto davinci_model = ModelManager::GetInstance().GetModel(ge_root_model->GetModelId());
  ASSERT_NE(davinci_model, nullptr);
  EXPECT_EQ(davinci_model->node_basic_infos_.size(), 3);
  EXPECT_EQ(model_executor.UnloadGraph(ge_root_model, graph_id), SUCCESS);
  ASSERT_EQ(model_executor.Finalize(), SUCCESS);
}

TEST_F(UtestFftsPlusTaskInfo, load_mixl2_with_args_format_successfully) {
  auto hcom_hidden_funcs = [](const ge::OpDescPtr &op_desc, std::vector<void *> &addrs) {
    addrs.push_back(reinterpret_cast<void *>(0xf1));
    return ge::GRAPH_SUCCESS;
  };
  REG_HIDDEN_INPUTS_FUNC(HiddenInputsType::HCOM, hcom_hidden_funcs);

  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  task_def.set_stream_id(0);

  auto func = [](void *handle, const uint64_t tilingKey, const void *const stubFunc, const uint32_t flag,
                 rtKernelDetailInfo_t *kernelInfo) -> int {
    kernelInfo->functionInfoNum = 1;
    kernelInfo->functionInfo[0].pcAddr = (void *)(0x1245);
    kernelInfo->functionInfo[0].prefetchCnt = 1;
    return RT_ERROR_NONE;
  };
  auto runtime_stub = std::make_shared<MockRuntime>();
  RuntimeStub::SetInstance(runtime_stub);
  EXPECT_CALL(*runtime_stub, rtKernelGetAddrAndPrefCntV2).WillRepeatedly(testing::Invoke(func));
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  const auto &op_desc = CreateOpDesc("ifa", "IFAFake");

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->AddInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("k", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->UpdateInputDesc(3, desc);
  op_desc->UpdateInputDesc(4, scalar_desc);
  op_desc->AddInputDesc(5, desc);
  op_desc->AddDynamicInputDescByIndex("addition_in", 1, 6);
  op_desc->UpdateInputDesc(6, desc);

  op_desc->AddOutputDesc("fake", desc);
  op_desc->AddDynamicOutputDesc("attention_out", 2, true);
  op_desc->UpdateOutputDesc("attention_out0", desc);
  op_desc->UpdateOutputDesc("attention_out1", scalar_desc);
  op_desc->AddOutputDesc("addition_out", desc);

  auto ifa_node = root_graph->AddNode(op_desc);
  auto tiling_info = std::make_shared<optiling::utils::OpRunInfo>(0, false, 0);
  tiling_info->AddTilingData("666");
  ifa_node->GetOpDesc()->SetExtAttr(ge::ATTR_NAME_OP_RUN_INFO, tiling_info);
  std::vector<std::string> name_prefix = {"_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  davinci_model.op_list_[0] = op_desc;
  davinci_model.operator_list_[0] = MakeShared<Operator>(OpDescUtils::CreateOperatorFromNode(ifa_node));
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  rtStream_t stream1 = nullptr;
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  davinci_model.stream_list_ = {stream1, stream2};

  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "00_0_kernel_mix_aiv");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::string mix_aiv_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aiv", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(mix_aiv_handle_name, nullptr, nullptr);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(47);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  InitMixL2DefForIFA(root_graph, *mixaicaivctx, "ifa");
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);

  auto arg_holder = task_info.ffts_flus_args_helper_->op_to_format_holder_;
  EXPECT_TRUE(!arg_holder.empty());

  // level2_addr
  EXPECT_EQ(task_info.ffts_flus_args_helper_->io_addrs_[4],
            PtrToValue(task_info.ffts_flus_args_helper_->args_dev_) + 14 * sizeof(uintptr_t));
  EXPECT_EQ(task_info.ffts_flus_args_helper_->io_addrs_[5],
            PtrToValue(task_info.ffts_flus_args_helper_->args_dev_) + 27 * sizeof(uintptr_t));
  EXPECT_EQ(task_info.ffts_flus_args_helper_->io_addrs_[9],
            PtrToValue(task_info.ffts_flus_args_helper_->args_dev_) + 37 * sizeof(uintptr_t));

  // offset_val
  EXPECT_EQ(task_info.ffts_flus_args_helper_->io_addrs_[14], 88);
  EXPECT_EQ(task_info.ffts_flus_args_helper_->io_addrs_[27], 64);
  EXPECT_EQ(task_info.ffts_flus_args_helper_->io_addrs_[37], 64);

  auto cust_to_relevant = task_info.ffts_flus_args_helper_->cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = {{0, 2}, {1, 24}, {2, 25}, {3, 34}, {4, 35}, {5, 5}, {6, 6}, {7, 7}, {8, 44}, {9, 45}, {10, 9}};
  EXPECT_EQ(golden, cust_to_relevant);
  HiddenInputsFuncRegistry::GetInstance().type_to_funcs_.clear();
  task_info.Release();
  rtStreamDestroy(stream);
}

TEST_F(UtestFftsPlusTaskInfo, load_ifa_with_tiling_dependence_successfully) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);

  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  davinci_model.runtime_param_.logic_mem_base = 0x0;
  davinci_model.runtime_param_.mem_base = 0x10000000;

  MemAllocation fm_mem_allocation = {0U,
                                     davinci_model.runtime_param_.mem_base,
                                     static_cast<uint64_t>(davinci_model.runtime_param_.mem_size),
                                     ge::MemAllocation::Type::FEATURE_MAP,
                                     0U,
                                     kFmMemType,
                                     0UL};
  davinci_model.logical_mem_allocations_.push_back(fm_mem_allocation);
  task_def.set_stream_id(0);

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
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  const auto &op_desc = CreateOpDesc("ifa", "IFAFake");

  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->AddInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->AddInputDesc(3, desc);
  op_desc->AddInputDesc(4, scalar_desc);
  op_desc->AddInputDesc("atten_mask", desc);

  op_desc->AddOutputDesc("fake", desc);
  op_desc->AddDynamicOutputDesc("attention_out", 2, true);
  op_desc->UpdateOutputDesc("attention_out0", desc);
  op_desc->UpdateOutputDesc("attention_out1", scalar_desc);

  std::shared_ptr<TilingContextAddr> tiling_context_addr = MakeShared<TilingContextAddr>();
  tiling_context_addr->tiling_context_addr = 0U;
  tiling_context_addr->op_type_addr = 0U;
  tiling_context_addr->tiling_key_addr = 0U;
  tiling_context_addr->block_dim_addr = 0U;
  tiling_context_addr->tiling_data_addr = 0U;
  EXPECT_TRUE(op_desc->SetExtAttr(kTilingContextAddrs, tiling_context_addr));

  auto ifa_node = root_graph->AddNode(op_desc);
  std::vector<std::string> name_prefix = {"_mix_aiv"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  davinci_model.op_list_[0] = op_desc;
  davinci_model.operator_list_[0] =
      MakeShared<Operator>(OpDescUtils::CreateOperatorFromNode(ifa_node));
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  rtStream_t stream1 = nullptr;
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  davinci_model.stream_list_ = {stream1, stream2};

  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aiv_kernel");
  TBEHandleStore::GetInstance().StoreTBEHandle("_mix_aivtbeKernel_test", nullptr, nullptr);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(44);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  InitMixL2DefForIFATilingSink(root_graph, *mixaicaivctx, "ifa");
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);

  auto arg_holder = task_info.ffts_flus_args_helper_->op_to_format_holder_;
  EXPECT_TRUE(!arg_holder.empty());

  auto cust_to_relevant = task_info.ffts_flus_args_helper_->cust_to_relevant_offset_;
  std::map<uint64_t, uint64_t> golden = {{0, 1}, {1, 21}, {2, 22}, {3, 31}, {4, 32}, {6, 5}, {7, 41}, {8, 42}};
  EXPECT_EQ(golden, cust_to_relevant);

  // distribute
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  std::shared_ptr<TilingSinkTaskInfo> default_ptr = nullptr;
  std::shared_ptr<TilingSinkTaskInfo> sink_task_info = op_desc->TryGetExtAttr(kTilingSinkTaskInfo, default_ptr);
  EXPECT_TRUE(sink_task_info != nullptr);

  task_info.Release();
  rtStreamDestroy(stream);
}

TEST_F(UtestFftsPlusTaskInfo, load_mixl2_with_args_format_with_l0_exception_dump_successfully) {
  gert::GlobalDumper::GetInstance()->SetEnableFlags(
      gert::BuiltInSubscriberUtil::BuildEnableFlags<gert::DumpType>({gert::DumpType::kLiteExceptionDump}));

  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  FftsPlusTaskInfo task_info;
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);

  davinci_model.stream_list_ = {stream};
  davinci_model.runtime_param_.mem_size = 0x45678;
  davinci_model.runtime_param_.logic_mem_base = 0x0;
  davinci_model.runtime_param_.mem_base = 0x10000000;

  MemAllocation fm_mem_allocation = {0U,
                                     davinci_model.runtime_param_.mem_base,
                                     static_cast<uint64_t>(davinci_model.runtime_param_.mem_size),
                                     ge::MemAllocation::Type::FEATURE_MAP,
                                     0U,
                                     kFmMemType,
                                     0UL};
  davinci_model.logical_mem_allocations_.push_back(fm_mem_allocation);
  task_def.set_stream_id(0);

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
  domi::FftsPlusTaskDef *ffts_plus_task_def = task_def.mutable_ffts_plus_task();

  auto root_graph = std::make_shared<ComputeGraph>("root_graph");
  const auto &op_desc = CreateOpDesc("ifa", "IFAFake");
  ASSERT_NE(op_desc, nullptr);
  GeShape shape({1, 40, 1024, 256});
  GeTensorDesc desc(shape);
  GeShape scalar_shape;
  GeTensorDesc scalar_desc(scalar_shape);
  op_desc->AddInputDesc(0, desc);
  op_desc->AddDynamicInputDescByIndex("key", 2, 1);
  op_desc->UpdateInputDesc(1, desc);
  op_desc->UpdateInputDesc(2, desc);
  op_desc->AddDynamicInputDescByIndex("value", 2, 3);
  op_desc->AddInputDesc(3, desc);
  op_desc->AddInputDesc(4, scalar_desc);
  op_desc->AddInputDesc("atten_mask", desc);

  op_desc->AddOutputDesc("fake", desc);
  op_desc->AddDynamicOutputDesc("attention_out", 2, true);
  op_desc->UpdateOutputDesc("attention_out0", desc);
  op_desc->UpdateOutputDesc("attention_out1", scalar_desc);

  std::shared_ptr<TilingContextAddr> tiling_context_addr = MakeShared<TilingContextAddr>();
  tiling_context_addr->tiling_context_addr = 0U;
  tiling_context_addr->op_type_addr = 0U;
  tiling_context_addr->tiling_key_addr = 0U;
  tiling_context_addr->block_dim_addr = 0U;
  tiling_context_addr->tiling_data_addr = 0U;
  EXPECT_TRUE(op_desc->SetExtAttr(kTilingContextAddrs, tiling_context_addr));

  auto ifa_node = root_graph->AddNode(op_desc);
  std::vector<std::string> name_prefix = {"_mix_aiv"};
  AttrUtils::SetStr(op_desc, ATTR_NAME_ALIAS_ENGINE_NAME, "mix_l2");
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  AttrUtils::SetBool(op_desc, "_kernel_list_first_name", true);
  davinci_model.op_list_[0] = op_desc;
  davinci_model.operator_list_[0] =
      MakeShared<Operator>(OpDescUtils::CreateOperatorFromNode(ifa_node));
  AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_STREAM_ID, 0);
  rtStream_t stream1 = nullptr;
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream1, 0, 0, 0);
  rtStream_t stream2 = nullptr;
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream2, 0, 0, 0);
  davinci_model.stream_list_ = {stream1, stream2};

  (void)AttrUtils::SetStr(op_desc, "_mix_aiv" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aiv_kernel");
  TBEHandleStore::GetInstance().StoreTBEHandle("_mix_aivtbeKernel_test", nullptr, nullptr);
  ffts_plus_task_def->set_op_index(0);
  ffts_plus_task_def->set_addr_size(44);
  InitTaskSQEInfo(ffts_plus_task_def);
  InitTaskAdditionalDataInfo(ffts_plus_task_def);

  domi::FftsPlusCtxDef *mixaicaivctx = ffts_plus_task_def->add_ffts_plus_ctx();
  mixaicaivctx->set_op_index(0);
  mixaicaivctx->set_context_id(0);
  mixaicaivctx->set_context_type(static_cast<uint32_t>(RT_CTX_TYPE_MIX_AIV));
  InitMixL2DefForIFATilingSink(root_graph, *mixaicaivctx, "ifa");
  TaskRunParam task_run_param = {};
  EXPECT_EQ(task_info.ParseTaskRunParam(task_def, &davinci_model, task_run_param), SUCCESS);
  PisToArgs args;
  g_arg_len = task_run_param.args_descs[0U].args_len;
  rtMalloc(&g_dev, g_arg_len, RT_MEMORY_HBM, GE_MODULE_NAME_U16);
  g_host = malloc(g_arg_len);
  args[0U].host_addr = g_host;
  args[0U].dev_addr = PtrToValue(g_dev);
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args), SUCCESS);

  // distribute
  EXPECT_EQ(task_info.Distribute(), SUCCESS);

  std::shared_ptr<TilingSinkTaskInfo> default_ptr = nullptr;
  std::shared_ptr<TilingSinkTaskInfo> sink_task_info = op_desc->TryGetExtAttr(kTilingSinkTaskInfo, default_ptr);
  EXPECT_TRUE(sink_task_info != nullptr);

  task_info.Release();
  rtStreamDestroy(stream);
  gert::GlobalDumper::GetInstance()->SetEnableFlags(0UL);
}
}  // namespace ge
