/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_op/task/aicpu_c_c_task_builder.h"
#include "common/checker.h"
#include "framework/common/taskdown_common.h"
#include "graph/load/model_manager/model_manager.h"
#include "single_op/task/build_task_utils.h"

namespace ge {
namespace {
  std::mutex cust_op_mutex;
} // namespace

AiCpuCCTaskBuilder::AiCpuCCTaskBuilder(const OpDescPtr &op_desc, const domi::KernelDef &kernel_def)
    : op_desc_(op_desc), kernel_def_(kernel_def) {}

Status AiCpuCCTaskBuilder::SetKernelArgs(AiCpuCCTask &task, const SingleOpModelParam &param) const {
  size_t aicpu_arg_size = kernel_def_.args_size();
  // Old model will not take this value, its default value is 0,need to convert to the real default value 1.
  task.block_dim_ = (kernel_def_.block_dim() == 0U) ? 1U : kernel_def_.block_dim();
  if (aicpu_arg_size <= sizeof(aicpu::AicpuParamHead)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Size]aicpu_arg_size is invalid, value = %zu", aicpu_arg_size);
    REPORT_INNER_ERR_MSG("E19999", "aicpu_arg_size is invalid, value = %zu", aicpu_arg_size);
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  task.io_addr_num_ = op_desc_->GetInputsSize() + op_desc_->GetOutputsSize();
  GE_CHECK_GE((aicpu_arg_size - sizeof(aicpu::AicpuParamHead)), (task.io_addr_num_ * sizeof(void *)));

  if (task.extend_args_for_host_input_) {
    task.host_mem_input_data_offset_ = aicpu_arg_size - sizeof(aicpu::AicpuParamHead);
    GE_ASSERT_TRUE(!ge::AddOverflow(aicpu_arg_size, kMaxHostMemInputLen, aicpu_arg_size));
    GELOGD("%s has host memory input, args size is extened %zu, args_size = %zu, host_mem_input_data_offset = %zu.",
           op_desc_->GetName().c_str(), kMaxHostMemInputLen, aicpu_arg_size, task.host_mem_input_data_offset_);
  }
  std::unique_ptr<uint8_t[]> aicpu_args = MakeUnique<uint8_t[]>(aicpu_arg_size);
  if (aicpu_args == nullptr) {
    GELOGE(ACL_ERROR_GE_MEMORY_ALLOCATION, "[New][Memory] failed, size = %zu", aicpu_arg_size);
    REPORT_INNER_ERR_MSG("E19999", "new Memory failed, size = %zu", aicpu_arg_size);
    return ACL_ERROR_GE_MEMORY_ALLOCATION;
  }

  GE_CHECK_LE(kernel_def_.args().size(), aicpu_arg_size);
  const auto err = memcpy_s(aicpu_args.get(), aicpu_arg_size, kernel_def_.args().data(), kernel_def_.args().size());
  if (err != EOK) {
    GELOGE(ACL_ERROR_GE_INTERNAL_ERROR, "[Memcpy_s][Args] failed, size = %zu, err = %d", aicpu_arg_size, err);
    REPORT_INNER_ERR_MSG("E19999", "memcpy_s aicpu_args failed, size = %zu, err = %d", aicpu_arg_size, err);
    return ACL_ERROR_GE_INTERNAL_ERROR;
  }

  task.SetIoAddr(PtrToPtr<void, uintptr_t>(ValueToPtr(PtrToValue(aicpu_args.get()) +
                                                      sizeof(aicpu::AicpuParamHead))));
  task.SetKernelArgs(std::move(aicpu_args), aicpu_arg_size);

  const auto addresses = BuildTaskUtils::GetKernelArgs(op_desc_, param);
  GE_CHECK_GE(addresses.size(), task.io_addr_num_);
  auto io_addr = task.io_addr_;
  for (size_t i = 0U; i < task.io_addr_num_; ++i) {
    *io_addr = static_cast<uintptr_t>(PtrToValue(addresses[i]));
    ++io_addr;
  }
  return SUCCESS;
}

Status AiCpuCCTaskBuilder::BuildTask(AiCpuCCTask &task, const uint64_t kernel_id,
                                     const SingleOpModelParam &param) const {
  auto ret = SetKernelArgs(task, param);
  if (ret != SUCCESS) {
    return ret;
  }
  const std::string &so_name = kernel_def_.so_name();
  const std::string &kernel_name = kernel_def_.kernel_name();
  task.SetSoName(so_name);
  task.SetkernelName(kernel_name);
  GE_CHECK_NOTNULL(op_desc_);
  task.op_desc_ = op_desc_;

  const auto &context = kernel_def_.context();
  const auto kernel_type = static_cast<ccKernelType>(context.kernel_type());
  if (kernel_type == ccKernelType::CUST_AI_CPU) {
    const std::lock_guard<std::mutex> lk(cust_op_mutex);
    task.is_custom_ = true;
    task.dump_flag_ |= RT_KERNEL_CUSTOM_AICPU;
    bool loaded = false;
    GE_CHK_STATUS_RET(ModelManager::GetInstance().LoadCustAicpuSo(op_desc_, so_name, loaded),
                      "[Load][CustAicpuSo] failed.");
    if (!loaded) {
      GE_CHK_STATUS_RET(ModelManager::GetInstance().LaunchCustAicpuSo(), "[Launch][CustAicpuSo] failed.");
    }
  }

  task.num_inputs_ = op_desc_->GetInputsSize();
  task.num_outputs_ = op_desc_->GetOutputsSize();

  // get kernel_ext_info
  auto &kernel_ext_info = kernel_def_.kernel_ext_info();
  const auto kernel_ext_info_size = kernel_def_.kernel_ext_info_size();
  GE_CHK_BOOL_RET_STATUS(kernel_ext_info.size() == kernel_ext_info_size, FAILED,
                         "[Check][Size]task def kernel_ext_info.size=%zu, but kernel_ext_info_size=%u.",
                         kernel_ext_info.size(), kernel_ext_info_size);

  ret = task.SetExtInfoAndType(kernel_ext_info, kernel_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][ExtInfoAndType]failed, kernel_id=%" PRIu64 ".", kernel_id);
    REPORT_INNER_ERR_MSG("E19999", "SetExtInfoAndType failed, kernel_id=%" PRIu64 ".", kernel_id);
    return ret;
  }
  GE_CHK_STATUS_RET(task.SetInputConst(), "[Set][InputConst] failed.");
  GE_CHK_STATUS_RET(task.InitForSummaryAndCopy(), "[Init][SummaryAndCopy] failed.");

  const auto aicpu_param_head = PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task.args_.get());
  if (task.ext_info_addr_dev_ != nullptr) {
    aicpu_param_head->extInfoLength = static_cast<uint32_t>(kernel_ext_info.size());
    aicpu_param_head->extInfoAddr = PtrToValue(task.ext_info_addr_dev_);
  }

  task.op_type_ = op_desc_->GetName();
  task.kernel_id_ = kernel_id;
  const auto debug_info = BuildTaskUtils::GetTaskInfo(op_desc_);
  GELOGI("[TASK_INFO] %" PRIu64 "/%s %s", kernel_id, task.op_type_.c_str(), debug_info.c_str());
  return SUCCESS;
}
}  // namespace ge
