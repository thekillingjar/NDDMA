/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dsa_task_builder.h"

#include "framework/common/ge_types.h"
#include "single_op/task/build_task_utils.h"

namespace ge {
namespace {
constexpr uint32_t kMask32Bits = 0xFFFFFFFFU;  // 32 bits, 1111,1111,1111,1111,1111,1111,1111,1111
}
Status DsaTaskBuilder::BuildDsaTask(const GetOpDescFunc &get_op_desc_func, const SingleOpModelParam &param,
                                    const domi::TaskDef &task_def, OpTask *&op_task, const StreamResource &resource) {
  GELOGI("DSATaskInfo Build Start.");
  (void)resource;
  const domi::DSATaskDef &dsa_task = task_def.dsa_task();
  OpDescPtr op_desc = nullptr;
  GE_CHK_STATUS_RET(get_op_desc_func(dsa_task.op_index(), op_desc), "get op desc failed");

  std::unique_ptr<DsaTask> task = MakeUnique<DsaTask>();
  GE_CHECK_NOTNULL(task);

  task->SetOpDesc(op_desc);
  task->dsa_sqe_.sqeHeader.type = static_cast<uint8_t>(dsa_task.sqe_type());
  task->dsa_sqe_.start = dsa_task.start();
  task->dsa_sqe_.functionType = dsa_task.distribution_type();
  task->dsa_sqe_.dataType = dsa_task.data_type();
  task->dsa_sqe_.algoType = dsa_task.alg_type();
  task->dsa_sqe_.paramVldBitmap = dsa_task.input_vld();
  task->dsa_sqe_.paramAddrValBitmap = dsa_task.input_value_addr_flag();
  task->dsa_sqe_.kernelCredit = 100U;

  const auto addresses = BuildTaskUtils::GetAddresses(op_desc, param, true);
  task->io_addr_ = BuildTaskUtils::JoinAddresses(addresses);

  const auto output_data_addrs = ModelUtils::GetOutputDataAddrsValue(param.runtime_param, op_desc);
  if (output_data_addrs.size() != kDSAOutputAddrSize) {
    GELOGE(INTERNAL_ERROR, "Node %s output addr size %zu is wrong", op_desc->GetName().c_str(),
           output_data_addrs.size());
    return INTERNAL_ERROR;
  }

  const auto workspace_data_addrs = ModelUtils::GetWorkspaceDataAddrsValue(param.runtime_param, op_desc);
  if (workspace_data_addrs.empty()) {
    GELOGE(INTERNAL_ERROR, "Node %s workspace addr size %zu is wrong", op_desc->GetName().c_str(),
           workspace_data_addrs.size());
    return INTERNAL_ERROR;
  }

  const auto input_data_addrs = ModelUtils::GetInputDataAddrsValue(param.runtime_param, op_desc);
  if (input_data_addrs.size() < kDSAInputAddrSize) {
    GELOGE(INTERNAL_ERROR, "Node %s input addr size %zu is wrong", op_desc->GetName().c_str(), input_data_addrs.size());
    return INTERNAL_ERROR;
  }
  task->input_size_ = input_data_addrs.size();
  task->output_size_ = output_data_addrs.size();
  task->workspace_size_ = workspace_data_addrs.size();
  task->input1_value_or_ptr_ = dsa_task.input1_value_or_ptr();
  task->seed_value_or_ptr_ = dsa_task.seed_value_or_ptr();
  task->random_count_value_or_ptr_ = dsa_task.random_count_value_or_ptr();

  if (dsa_task.input1_value_or_ptr() != kDSASetInputAddr) {
    const std::string &input1 = dsa_task.args().input1_value_or_addr();
    auto mem_ret = memcpy_s(&task->input_data_[0], sizeof(uint64_t), input1.c_str(), input1.size());
    if (mem_ret != EOK) {
      GELOGE(INTERNAL_ERROR, "dsa input data memcpy failed.");
      return INTERNAL_ERROR;
    }
    if ((input_data_addrs.size() == kDSAStateInputAddrSize) ||
        ((input_data_addrs.size() == kDSAArgsInputAddrSize) &&
         (workspace_data_addrs.size() == kDSAWorkspaceAddrSize))) {
      const std::string &input2 = dsa_task.args().input2_value_or_addr();
      mem_ret = memcpy_s(&task->input_data_[1], sizeof(uint64_t), input2.c_str(), input2.size());
      if (mem_ret != EOK) {
        GELOGE(INTERNAL_ERROR, "dsa input data memcpy failed.");
        return INTERNAL_ERROR;
      }
    }
  }

  if (dsa_task.seed_value_or_ptr() != kDSASetInputAddr) {
    const uint64_t seed_value_or_addr = *(PtrToPtr<char_t, uint64_t>(dsa_task.args().seed_value_or_addr().c_str()));
    task->dsa_sqe_.dsaCfgSeedLow = static_cast<uint32_t>(seed_value_or_addr & kMask32Bits);
    task->dsa_sqe_.dsaCfgSeedHigh = static_cast<uint32_t>(seed_value_or_addr >> k32Bits);
  }

  if (dsa_task.random_count_value_or_ptr() != kDSASetInputAddr) {
    const uint64_t random_count_value_or_addr =
        *(PtrToPtr<char_t, uint64_t>(dsa_task.args().random_count_value_or_addr().c_str()));
    task->dsa_sqe_.dsaCfgNumberLow = static_cast<uint32_t>(random_count_value_or_addr & kMask32Bits);
    task->dsa_sqe_.dsaCfgNumberHigh = static_cast<uint32_t>(random_count_value_or_addr >> k32Bits);
  }

  op_task = task.release();
  GELOGI("DSATaskInfo Build Success");
  return SUCCESS;
}
}  // namespace ge
