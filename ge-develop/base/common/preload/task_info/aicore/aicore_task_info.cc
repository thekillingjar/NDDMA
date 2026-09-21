/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/aicore/aicore_task_info.h"
#include "common/preload/model/pre_model_utils.h"
#include "common/preload/model/pre_model_types.h"
#include "common/preload/task_info/pre_task_status.h"

namespace ge {
PreTaskResult GenerateAiCoreTask(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                 const PreTaskInput &pre_task_input) {
  PreTaskResult result;
  if (op_desc == nullptr) {
    result.status = PreTaskStatus::ErrorStatus("op_desc is nullptr.");
    return result;
  }
  GELOGD("Init generate aicore task %s.", op_desc->GetName().c_str());
  PreTaskDescInfo pre_task_desc_info;
  const domi::KernelDef &kernel_def = task_def.kernel();
  // step1. set kernel args info
  auto &args_param = pre_task_desc_info.kernel_args_desc_info.kernel_args_desc_data;
  std::vector<uint64_t> args_offset_values;
  const auto input_data_addr_offset =
      PreModelUtils::GetInputDataAddrOffset(pre_task_input.rts_param, op_desc, args_param, args_offset_values);
  const auto output_data_addr_offset =
      PreModelUtils::GetOutputDataAddrOffset(pre_task_input.rts_param, op_desc, args_param, args_offset_values);
  const auto workspace_data_addr_offset =
      PreModelUtils::GetWorkspaceDataAddrOffset(pre_task_input.rts_param, op_desc, args_param, args_offset_values);

  // step2. set kernel args
  const uint32_t args_size = static_cast<uint32_t>(kernel_def.args().size());
  pre_task_desc_info.kernel_args_info.kernel_args_data_size = args_size;
  pre_task_desc_info.kernel_args_info.kernel_args_data.reset(new (std::nothrow) uint8_t[args_size],
                                                             std::default_delete<uint8_t[]>());
  errno_t sec_ret = memcpy_s(pre_task_desc_info.kernel_args_info.kernel_args_data.get(), static_cast<size_t>(args_size),
                             kernel_def.args().data(), static_cast<size_t>(args_size));
  if (sec_ret != EOK) {
    result.status = PreTaskStatus::ErrorStatus("memcpy_s failed, args_size:%u, ret:%d", args_size, sec_ret);
    return result;
  }

  const size_t args_offset_value_size = args_offset_values.size() * sizeof(uint64_t);
  sec_ret = memcpy_s(pre_task_desc_info.kernel_args_info.kernel_args_data.get(), static_cast<size_t>(args_size),
                     args_offset_values.data(), args_offset_value_size);
  if (sec_ret != EOK) {
    result.status = PreTaskStatus::ErrorStatus("memcpy_s failed, args_offset_value_size:%zu, ret:%d",
                                               args_offset_value_size, sec_ret);
    return result;
  }

  // step3. set seq info
  const auto itr = pre_task_input.names_to_bin_offset.find(kernel_def.kernel_name());
  if (itr != pre_task_input.names_to_bin_offset.end()) {
    pre_task_desc_info.seq_info.u.aicoreTask.kernelBinOffset = itr->second;
  } else {
    result.status = PreTaskStatus::ErrorStatus("can't find bin offset from op_name[%s]. kernel_name[%s]",
                                               op_desc->GetName().c_str(), kernel_def.kernel_name().c_str());
    return result;
  }
  pre_task_desc_info.seq_info.taskType = RT_TASK_TYPE_KERNEL_AICORE;
  pre_task_desc_info.seq_info.streamId = static_cast<uint16_t>(task_def.stream_id());
  pre_task_desc_info.seq_info.u.aicoreTask.argsOffset = 0U;
  pre_task_desc_info.seq_info.u.aicoreTask.blockDim =
          (kernel_def.block_dim() == 0U) ? 1U : static_cast<uint16_t>(kernel_def.block_dim());
  (void)result.pre_task_desc_infos.emplace_back(pre_task_desc_info);

  result.status = PreTaskStatus::Success();
  GELOGD("Init generate aicore task success");
  return result;
}
REFISTER_PRE_GENERATE_TASK(kPreEngineAiCore, GenerateAiCoreTask);
}  // namespace ge
