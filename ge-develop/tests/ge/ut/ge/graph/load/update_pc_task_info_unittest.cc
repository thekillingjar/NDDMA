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
#include "graph/compute_graph.h"
#include "graph/load/model_manager/task_info/fe/update_pc_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "proto/task.pb.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/load/model_manager/tbe_kernel_handle.h"

namespace ge {
class UtestUpdatePcTaskInfo : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestUpdatePcTaskInfo, init_update_pc_mxil2_success) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  auto op_desc = MakeShared<OpDesc>("ifa", "IFA");

  uint8_t handle[100];
  (void)AttrUtils::SetStr(op_desc, "_mix_aic" + ATTR_NAME_TBE_KERNEL_NAME, "mix_aic_kernel");
  (void)AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle);

  std::vector<std::string> name_prefix = {"_mix_aic"};
  AttrUtils::SetListStr(op_desc, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);

  std::string kernel_handle_name = davinci_model.GetBinHandleKey(*op_desc, "_mix_aic", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, handle, nullptr);

  davinci_model.op_list_[op_desc->GetId()] = op_desc;
  task_def.mutable_update_pc_task()->set_op_index(op_desc->GetId());
  task_def.set_stream_id(0U);
  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  rtFftsPlusTaskInfo_t tmp_task_info;
  std::shared_ptr<TilingSinkTaskInfo> default_ptr = make_shared<TilingSinkTaskInfo>();
  default_ptr->task_id = 0U;
  default_ptr->ffts_task_handle = &tmp_task_info;
  default_ptr->stream = stream;
  op_desc->SetExtAttr(kTilingSinkTaskInfo, default_ptr);

  std::shared_ptr<TilingContextAddr> default_ctx_ptr = make_shared<TilingContextAddr>();
  default_ctx_ptr->tiling_context_addr = 0x100U;
  default_ctx_ptr->op_type_addr = 0x200U;
  default_ctx_ptr->tiling_key_addr = 0x300U;
  default_ctx_ptr->block_dim_addr = 0x400U;
  op_desc->SetExtAttr(kTilingContextAddrs, default_ctx_ptr);

  davinci_model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(davinci_model.runtime_param_.mem_size);
  davinci_model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(davinci_model.runtime_param_.mem_base),
                                     davinci_model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  davinci_model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  UpdatePCTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace, iow_addrs), SUCCESS);

  // distribute
  domi::GetContext().is_online_model = true;
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  EXPECT_TRUE(task_info.IsSupportReDistribute());
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
  domi::GetContext().is_online_model = false;
}

TEST_F(UtestUpdatePcTaskInfo, init_update_pc_custom_success) {
  DavinciModel davinci_model(0, nullptr);
  domi::TaskDef task_def;
  auto op_desc = MakeShared<OpDesc>("ifa", "IFA");
  davinci_model.op_list_[op_desc->GetId()] = op_desc;
  task_def.mutable_update_pc_task()->set_op_index(op_desc->GetId());
  task_def.set_stream_id(0U);

  uint8_t handle[100];
  AttrUtils::SetStr(op_desc, ATTR_NAME_KERNEL_BIN_ID, "00_0_kernel");
  (void)AttrUtils::SetStr(op_desc, TVM_ATTR_NAME_MAGIC, "RT_DEV_BINARY_MAGIC_ELF");
  std::vector<char> kernel_bin(64, '\0');
  TBEKernelPtr kernel_handle = MakeShared<OpKernelBin>(op_desc->GetName(), std::move(kernel_bin));
  op_desc->SetExtAttr(OP_EXTATTR_NAME_TBE_KERNEL, kernel_handle);
  std::string kernel_handle_name = davinci_model.GetBinHandleKey(*op_desc, "", false);
  TBEHandleStore::GetInstance().StoreTBEHandle(kernel_handle_name, handle, nullptr);

  rtStream_t stream = nullptr;
  davinci_model.reusable_stream_allocator_ = ReusableStreamAllocator::Create();
  davinci_model.reusable_stream_allocator_->GetOrCreateRtStream(stream, 0, 0, 0);
  davinci_model.stream_list_ = {stream};

  std::shared_ptr<TilingSinkTaskInfo> default_ptr = make_shared<TilingSinkTaskInfo>();
  default_ptr->task_id = 0U;
  default_ptr->ffts_task_handle = reinterpret_cast<void *>(0x112U);
  default_ptr->stream = stream;
  op_desc->SetExtAttr(kTilingSinkTaskInfo, default_ptr);

  std::shared_ptr<TilingContextAddr> default_ctx_ptr = make_shared<TilingContextAddr>();
  default_ctx_ptr->tiling_context_addr = 0x100U;
  default_ctx_ptr->op_type_addr = 0x200U;
  default_ctx_ptr->tiling_key_addr = 0x300U;
  default_ctx_ptr->block_dim_addr = 0x400U;
  op_desc->SetExtAttr(kTilingContextAddrs, default_ctx_ptr);

  davinci_model.runtime_param_.mem_size = 10000UL;
  std::vector<uint8_t> memory_holder(davinci_model.runtime_param_.mem_size);
  davinci_model.runtime_param_.mem_base = reinterpret_cast<uintptr_t>(memory_holder.data());
  MemAllocation fm_mem_allocation = {0, static_cast<uint64_t>(davinci_model.runtime_param_.mem_base),
                                     davinci_model.runtime_param_.mem_size, ge::MemAllocation::Type::FEATURE_MAP, 0U};
  davinci_model.logical_mem_allocations_.emplace_back(fm_mem_allocation);

  PisToArgs args;
  const PisToPersistentWorkspace persistant_workspace = {};
  IowAddrs iow_addrs;
  UpdatePCTaskInfo task_info;
  EXPECT_EQ(task_info.Init(task_def, &davinci_model, args, persistant_workspace, iow_addrs), SUCCESS);
  // distribute
  EXPECT_EQ(task_info.Distribute(), SUCCESS);
}
}  // namespace ge
