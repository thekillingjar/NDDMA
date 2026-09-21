/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "adapter/tbe_adapter/kernel_launch/tbe_kernel_launch.h"
#include "common/fe_log.h"
#include "common/aicore_util_attr_define.h"
#include "common/aicore_util_constants.h"
#include "common/fe_inner_error_codes.h"
#include "rt_error_codes.h"
#include "runtime/mem.h"
#include "common/platform_utils.h"

namespace fe {
namespace {
const uint32_t MIN_ARG_SIZE = 4;
const uint32_t ARG_ENTRY_SIZE = 2624;              // 1536 + 1024(tiling) + 64(host mem)
}

uint32_t g_args_count = 1;
uint16_t g_args_offset[ARG_ENTRY_SIZE / MIN_ARG_SIZE];

Status TbeKernelLaunch::DealKernelLaunch(const ge::Node &node, const void *args,
                                         const uint32_t &args_size, const std::string &stub_func,
                                         const uint32_t &core_dim, domi::TaskDef &task_def) {
  string op_name = node.GetName();
  string op_type = node.GetType();
  auto op_desc = node.GetOpDesc();
  // malloc  args_size + append_args_size
  size_t append_args_size = GetAppendArgsSizeOf() * GetAppendArgsNum();
  uint32_t total_args_size = args_size + append_args_size;

  std::vector<uint8_t> args_buff(total_args_size, 0);
  if (memcpy_s(args_buff.data(), total_args_size, args, args_size) != EOK) {
    FE_LOGE("[GenTask][KernelLaunch] Copy args data failed.");
    return TASK_BUILDER_STATUS_INTERNAL_ERROR;
  }
  // memcpy form append_args
  if (AddAppendArgs(node, args_buff.data(), args_size) != SUCCESS) {
    return TASK_BUILDER_STATUS_INTERNAL_ERROR;
  }
  // 5. call KernelLaunch
  FE_LOGD("Op[name=%s,type=%s]: args_size:%u bytes, append_args_size:%zu bytes, total_args_size:%u bytes.",
          op_name.c_str(), op_type.c_str(), args_size, append_args_size, total_args_size);
  if (append_args_size > 0) {
    PrintAllArgs(op_name, op_type, args_buff.data(), args_size);
  }

  bool ret = false;
  std::string first_kernel_name;
  if (ge::AttrUtils::GetStr(op_desc, ATTR_NAME_KERNEL_LIST_FIRST_NAME, first_kernel_name)) {
    FE_LOGD("Node name is[%s], first kernel name is[%s].", op_name.c_str(), first_kernel_name.c_str());
    ret = KernelLaunchWithHandle(core_dim, args_buff.data(), total_args_size, nullptr, task_def);
  } else {
    ret = KernelLaunch(stub_func, core_dim, args_buff.data(), total_args_size, nullptr, task_def);
  }

  if (!ret) {
    return TASK_BUILDER_STATUS_RUNTIME_ERROR;
  }
  return SUCCESS;
}

void TbeKernelLaunch::PrintAllArgs(const string &op_name, const string &op_type, const void *all_args_buff,
                                   uint32_t args_size) {
  for (size_t i = 0; i != args_size / sizeof(uint64_t); ++i) {
    uint64_t value = *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(all_args_buff) + i * sizeof(uint64_t)));
    FE_LOGD("Op[name=%s,type=%s]: args[%zu]=[%lu].", op_name.c_str(), op_type.c_str(), i, value);
  }

  for (size_t i = 0; i != GetAppendArgsNum(); ++i) {
    uint64_t value = *(reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(all_args_buff) + args_size +
                                                    i * GetAppendArgsSizeOf()));
    FE_LOGD("Op[name=%s,type=%s]: append_args[%zu]=[%lu].", op_name.c_str(), op_type.c_str(), i, value);
  }
}

size_t TbeKernelLaunch::GetAppendArgsSizeOf() {
  return 0;
}
size_t TbeKernelLaunch::GetAppendArgsNum() {
  return 0;
}
Status TbeKernelLaunch::AddAppendArgs(const ge::Node &node, void *all_args_buff, const uint32_t &args_size) {
  (void)node;
  (void)all_args_buff;
  (void)args_size;
  return SUCCESS;
}

bool TbeKernelLaunch::KernelLaunch(const std::string &stub_func, const uint32_t block_dim, const void *args,
                                   uint32_t args_size, const rtSmDesc_t *sm_desc, domi::TaskDef &task_def) {
  task_def.set_type(static_cast<uint32_t>(RT_MODEL_TASK_KERNEL));
  domi::KernelDef *kernel_def = task_def.mutable_kernel();
  if (kernel_def == nullptr) {
    FE_LOGE("[GenTask][KernelLaunch] kernel_def is nullptr.");
    return false;
  }

  FE_LOGD("[GenTask][KernelLaunch] stub_func_name is [%s].", stub_func.c_str());
  kernel_def->set_stub_func(stub_func);
  if (sm_desc != nullptr) {
    uintptr_t sm_desc_data = reinterpret_cast<uintptr_t>(sm_desc);
    uint8_t *sm_desc_ptr = reinterpret_cast<uint8_t *>(sm_desc_data);
    kernel_def->set_sm_desc(sm_desc_ptr, sizeof(rtSmDesc_t));
  }

  kernel_def->set_block_dim(block_dim);
  kernel_def->set_args_size(args_size);
  kernel_def->set_args(args, args_size);

  domi::KernelContext *kernel_context = kernel_def->mutable_context();
  if (kernel_context == nullptr) {
    REPORT_FE_ERROR("[GenTask][KernelLaunch] kernel_context is nullptr.");
    return false;
  }
  g_args_offset[0] = 0;
  kernel_context->set_args_count(g_args_count);
  kernel_context->set_args_offset(g_args_offset, g_args_count * sizeof(uint16_t));
  return true;
}

bool TbeKernelLaunch::KernelLaunchWithHandle(const uint32_t block_dim, const void *args, uint32_t args_size,
                                             const rtSmDesc_t *sm_desc, domi::TaskDef &task_def) {
  task_def.set_type(static_cast<uint32_t>(RT_MODEL_TASK_ALL_KERNEL));
  domi::KernelDefWithHandle *kernel_def_with_handle = task_def.mutable_kernel_with_handle();
  if (kernel_def_with_handle == nullptr) {
    FE_LOGE("[GenTask][KernelLaunchWithHandle] kernel_def_with_handle is nullptr.");
    return false;
  }

  if (sm_desc != nullptr) {
    uintptr_t sm_desc_data = reinterpret_cast<uintptr_t>(sm_desc);
    uint8_t *sm_desc_ptr = reinterpret_cast<uint8_t *>(sm_desc_data);
    kernel_def_with_handle->set_sm_desc(sm_desc_ptr, sizeof(rtSmDesc_t));
  }
  kernel_def_with_handle->set_block_dim(block_dim);
  kernel_def_with_handle->set_args_size(args_size);
  kernel_def_with_handle->set_args(args, args_size);

  domi::KernelContext *kernel_context = kernel_def_with_handle->mutable_context();
  if (kernel_context == nullptr) {
    FE_LOGE("[GenTask][KernelLaunch] kernel_context is nullptr.");
    return false;
  }
  g_args_offset[0] = 0;
  kernel_context->set_args_count(g_args_count);
  kernel_context->set_args_offset(g_args_offset, g_args_count * sizeof(uint16_t));
  return true;
}
}  // namespace fe
