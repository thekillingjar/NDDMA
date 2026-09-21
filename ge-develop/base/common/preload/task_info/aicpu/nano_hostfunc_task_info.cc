/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/aicpu/nano_hostfunc_task_info.h"
#include "common/preload/model/pre_model_utils.h"
#include "common/preload/model/pre_model_types.h"
#include "common/preload/task_info/pre_task_status.h"
#include "common/preload/task_info/aicpu/nano_hostfunc_param.h"
#include "common/preload/model/pre_model_partition_utils.h"

namespace ge {
PreTaskResult GenerateNanoHostFuncTask(const domi::TaskDef &task_def, const OpDescPtr &op_desc,
                                       const PreTaskInput &pre_task_input) {
  GELOGD("Init generate hostFunc task");
  PreTaskResult result;
  if (op_desc == nullptr) {
    result.status = PreTaskStatus::ErrorStatus("op_desc is nullptr.");
    return result;
  }
  GELOGD("Init generate nano hostfunc task %s.", op_desc->GetName().c_str());
  PreTaskDescInfo pre_task_desc_info;
  const domi::KernelDef &kernel_def = task_def.kernel();
  // step1. static task desc
  pre_task_desc_info.seq_info.taskType = RT_TASK_TYPE_KERNEL_NANO_AICPU_HOSTFUNC;
  pre_task_desc_info.seq_info.streamId = static_cast<uint16_t>(task_def.stream_id());
  pre_task_desc_info.seq_info.u.nanoHostFuncTask.type = HWTS_STATIC_TASK_DESC;
  pre_task_desc_info.seq_info.bufType = HWTS_STATIC_TASK_DESC;
  rtHwtsStaticTaskDesc_t hwts_static_task_desc = {};
  hwts_static_task_desc.type = static_cast<uint16_t>(NanoTaskDescType::NANO_PLACE_HOLDER);
  hwts_static_task_desc.preP = static_cast<uint16_t>(NanoTaskPreStatus::NANO_PRE_ENABLE);
  hwts_static_task_desc.softUser = static_cast<uint16_t>(NanoTaskSoftUserStatus::NANO_SOFTUSER_HOSTFUNC);
  hwts_static_task_desc.swapOut = static_cast<uint8_t>(DEFAULT_INFO_VALUE_ONE);
  pre_task_desc_info.seq_info.u.nanoHostFuncTask.u.hwtsTaskDesc = hwts_static_task_desc;
  result.pre_task_desc_infos.push_back(pre_task_desc_info);

  // step2. dynamic task desc
  pre_task_desc_info.seq_info.taskType = RT_TASK_TYPE_KERNEL_NANO_AICPU_HOSTFUNC;
  pre_task_desc_info.seq_info.streamId = static_cast<uint16_t>(task_def.stream_id());
  pre_task_desc_info.seq_info.u.nanoHostFuncTask.type = HWTS_DYNAMIC_TASK_DESC;
  pre_task_desc_info.seq_info.bufType = HWTS_DYNAMIC_TASK_DESC;
  rtHwtsDynamicTaskDesc_t hwts_dynamic_task_desc = {};
  hwts_dynamic_task_desc.vld = static_cast<uint8_t>(DEFAULT_INFO_VALUE_ONE);
  hwts_dynamic_task_desc.codeSize = DEFAULT_INFO_VALUE_EIGHT;
  hwts_dynamic_task_desc.dynTaskDescSize = DEFAULT_INFO_VALUE_ONE;
  hwts_dynamic_task_desc.blockDim = (kernel_def.block_dim() == 0U) ? 1U : kernel_def.block_dim();

  pre_task_desc_info.seq_info.u.nanoHostFuncTask.u.hwtsDynamicTaskDesc = hwts_dynamic_task_desc;
  result.pre_task_desc_infos.push_back(pre_task_desc_info);

  // step3. task param info desc
  pre_task_desc_info.seq_info.taskType = RT_TASK_TYPE_KERNEL_NANO_AICPU_HOSTFUNC;
  pre_task_desc_info.seq_info.streamId = static_cast<uint16_t>(task_def.stream_id());
  pre_task_desc_info.seq_info.u.nanoHostFuncTask.type = PARAM_TASK_INFO_DESC;
  pre_task_desc_info.seq_info.bufType = PARAM_TASK_INFO_DESC;
  rtParamBufDesc_t &param_buf_desc = pre_task_desc_info.seq_info.u.nanoHostFuncTask.u.paramBufDesc;
  param_buf_desc.prefetchBufSize = 0U;
  param_buf_desc.paramBufSize = 0U;
  param_buf_desc.bufSize = 0U;

  const std::shared_ptr<NanoHostfuncParam> hostfunc =
      ge::MakeShared<NanoHostfuncParam>(task_def, op_desc, pre_task_input);
  if (hostfunc == nullptr) {
    result.status = PreTaskStatus::ErrorStatus("HostFunc makeshared fail.");
    return result;
  }
  const Status ret = hostfunc->GenHostFuncParamBufDesc(param_buf_desc);
  if (ret != SUCCESS) {
    result.status = PreTaskStatus::ErrorStatus("HostFunc param tlv fail.");
    return result;
  }
  const std::shared_ptr<uint8_t> buff = hostfunc->Data();
  // save share_ptr
  PreModelPartitionUtils::GetInstance().AddNanoHostFuncParamData(buff);
  param_buf_desc.bufInfo = hostfunc->Data().get();
  param_buf_desc.bufSize = hostfunc->DataSize();
  result.pre_task_desc_infos.push_back(pre_task_desc_info);

  result.status = PreTaskStatus::Success();
  GELOGD("success generate hostFunc task");
  return result;
}

REFISTER_PRE_GENERATE_TASK(kPreEngineNanoAiCpu, GenerateNanoHostFuncTask);
}  // namespace ge
