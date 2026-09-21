/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "executor/npu_sched_model_configurator.h"
#include "acl/acl.h"
#include "runtime/rt_external.h"
#include "aicpu_task_struct.h"
#include "graph/def_types.h"
#include "framework/common/debug/log.h"
#include "graph_metadef/common/ge_common/util.h"
#include "common/df_chk.h"

namespace ge {
namespace {
constexpr int32_t kDefaultPriority = 0;
constexpr uint32_t kKernelBlockDim = 1U;
constexpr const char *kKernelNameModelConfig = "AicpuModelConfig";
}

Status NpuSchedModelConfigurator::SetModelConfig(const AicpuModelConfig &config) {
  GELOGD("Start to configure model:%u.", config.model_id);
  std::vector<uint8_t> task_args;
  GE_CHK_STATUS_RET(BuildModelConfigTask(config, task_args), "Fail to init task args.");
  GE_CHK_STATUS_RET(ExecuteKernel(kKernelNameModelConfig, task_args),
                    "Fail to execute kernel.");
  GELOGD("Success to configure model:%u.", config.model_id);
  return SUCCESS;
}

Status NpuSchedModelConfigurator::BuildModelConfigTask(const AicpuModelConfig &config,
                                                       std::vector<uint8_t> &task_args) {
  const size_t args_size = sizeof(aicpu::AicpuParamHead) + sizeof(config);
  task_args.resize(args_size);
  auto &param_head = *(PtrToPtr<uint8_t, aicpu::AicpuParamHead>(task_args.data()));
  param_head.length = args_size;
  param_head.ioAddrNum = 0U;  // no input
  *(PtrToPtr<uint8_t, AicpuModelConfig>(&task_args[sizeof(aicpu::AicpuParamHead)])) = config;
  GELOGD("Success to configure %s task args, size = %zu", kKernelNameModelConfig, args_size);
  return SUCCESS;
}

Status NpuSchedModelConfigurator::ExecuteKernel(const std::string &kernel_name,
                                                const std::vector<uint8_t> &task_args) {
  aclrtStream stream = nullptr;
  DF_CHK_ACL_RET(aclrtCreateStream(&stream));
  DF_MAKE_GUARD_ACLSTREAM(stream);
  rtArgsEx_t args_info = {};
  args_info.args = const_cast<void *>(static_cast<const void *>(task_args.data()));
  args_info.argsSize = static_cast<uint32_t>(task_args.size());
  GE_CHK_RT_RET(rtCpuKernelLaunchWithFlag(nullptr, kernel_name.c_str(), kKernelBlockDim,
      &args_info, nullptr, stream, RT_KERNEL_DEFAULT));
  GELOGD("Success to launch kernel, kernel name = %s", kernel_name.c_str());
  DF_CHK_ACL_RET(aclrtSynchronizeStream(stream));
  GELOGD("Success to sync stream, kernel name = %s", kernel_name.c_str());
  return SUCCESS;
}
}
