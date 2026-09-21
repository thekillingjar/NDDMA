/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/load/model_manager/task_info/fe/update_pc_task_info.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/tbe_handle_store/tbe_handle_store.h"
#include "graph/load/model_manager/tbe_kernel_handle.h"
#include "graph/load/model_manager/task_info/args_format/args_format_utils.h"
#include "ge/ge_api_types.h"
#include "graph/load/model_manager/kernel/kernel_register_info_builder.h"

namespace ge {
Status UpdatePCTaskInfo::Init(const domi::TaskDef &task_def, DavinciModel *const davinci_model, const PisToArgs &args,
                              const PisToPersistentWorkspace &persistent_workspace, const IowAddrs &iow_addrs) {
  GELOGI("UpdatePCTaskInfo Init Start.");
  (void)args;
  (void)persistent_workspace;
  (void)iow_addrs;
  GE_CHECK_NOTNULL(davinci_model);
  davinci_model_ = davinci_model;
  op_desc_ = davinci_model->GetOpByIndex(task_def.update_pc_task().op_index());
  GE_ASSERT_NOTNULL(op_desc_);
  GE_CHK_STATUS_RET_NOLOG(SetStream(task_def.stream_id(), davinci_model->GetStreamList()));
  GELOGI("UpdatePCTaskInfo Init Success, logic stream id: %u, stream: %p.", task_def.stream_id(), stream_);
  return SUCCESS;
}

Status UpdatePCTaskInfo::Distribute() {
  GE_ASSERT_NOTNULL(op_desc_);
  GELOGI("UpdatePCTaskInfo %s Distribute Start.", op_desc_->GetNamePtr());
  std::shared_ptr<TilingSinkTaskInfo> default_ptr = nullptr;
  std::shared_ptr<TilingSinkTaskInfo> sink_task_info = op_desc_->TryGetExtAttr(kTilingSinkTaskInfo, default_ptr);
  GE_ASSERT_NOTNULL(sink_task_info, "Sink task info is nullptr, please check if target task has been launched.");

  std::shared_ptr<TilingContextAddr> default_ctx_ptr = nullptr;
  std::shared_ptr<TilingContextAddr> tiling_context_addr =
      op_desc_->TryGetExtAttr(kTilingContextAddrs, default_ctx_ptr);
  GE_ASSERT_NOTNULL(tiling_context_addr, "Tiling info is nullptr, please check if tiling task has been launched.");
  
  void *handle{nullptr};
  GE_ASSERT_SUCCESS(GetKernelHandle(handle));
  update_info_.hdl = handle;
  update_info_.fftsPlusTaskInfo = reinterpret_cast<rtFftsPlusTaskInfo_t *>(sink_task_info->ffts_task_handle);
  update_info_.blockDimAddr = reinterpret_cast<uint64_t *>(tiling_context_addr->block_dim_addr);
  update_info_.tilingKeyAddr = reinterpret_cast<uint64_t *>(tiling_context_addr->tiling_key_addr);

  GE_CHK_RT_RET(rtModelTaskUpdate(sink_task_info->stream, sink_task_info->task_id, stream_, &update_info_));

  is_support_redistribute_ = true;
  GELOGI("UpdatePCTaskInfo %s Distribute Success, stream: %p.", op_desc_->GetNamePtr(), stream_);

  return SUCCESS;
}

Status UpdatePCTaskInfo::GetKernelHandle(void *&handle) {
  auto kernel_handles_manager = davinci_model_->GetKernelHandlesManager(KernelHandleType::kAicore);
  GE_ASSERT_NOTNULL(kernel_handles_manager);
  KernelRegisterInfo register_info;
  GE_ASSERT_SUCCESS(KernelRegisterInfoBuilder::ConstructAicoreRegisterInfo(op_desc_, false, davinci_model_->GetModelId(), register_info));
  const auto bin_name = kernel_handles_manager->GenerateKey(register_info);
  handle = kernel_handles_manager->FindKernel(bin_name);
  if (handle != nullptr) {
    GELOGI("[%s][%s] Get tiling device kernel handle from kernel manager.",
        op_desc_->GetNamePtr(), op_desc_->GetTypePtr());
    return SUCCESS;
  }
  std::vector<std::string> name_prefix;
  (void)AttrUtils::GetListStr(op_desc_, ATTR_NAME_KERNEL_NAMES_PREFIX, name_prefix);
  std::string prefix;
  if (!name_prefix.empty()) {
    GE_ASSERT(name_prefix.size() == 1UL, "Rts does not support multi handles.");
    prefix = name_prefix[0UL];
  }
  std::string kernel_handle_name = davinci_model_->GetBinHandleKey(*op_desc_, prefix, false);
  GE_ASSERT_TRUE(TBEHandleStore::GetInstance().FindTBEHandle(kernel_handle_name, handle),
                 "Kernel bin is not found for op :[%s] with kernel name:[%s]", op_desc_->GetNamePtr(),
                 kernel_handle_name.c_str());

  return SUCCESS;
}

REGISTER_TASK_INFO(MODEL_TASK_UPDATE, UpdatePCTaskInfo);
}  // namespace ge
