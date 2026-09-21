/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "subscriber_utils.h"
#include "aprof_pub.h"
#include "core/builder/node_types.h"
#include "runtime/subscriber/global_profiler.h"
#include "utils/extern_math_util.h"
namespace gert {
bool SubscriberUtils::IsNodeNeedProf(const char *kernel_type) {
  // todo ge profiling register all kernel type
  if (GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kGeHost)) {
    return true;
  }
  return ProfilerRegistry::GetInstance().IsProfLaunchType(kernel_type) ||
         (kNamesToProfTypes.find(kernel_type) != kNamesToProfTypes.cend());
}

// 不上报additional info时，通过cann_host_profiler统一上报包括launch在内的api，否则就把launch api的上报和additional
// info的上报放在一起
uint32_t SubscriberUtils::GetProfKernelType(const char *kernel_type, bool is_for_additional_info) {
  const auto iter = kNamesToProfTypes.find(kernel_type);
  const auto is_prof_launch_type = ProfilerRegistry::GetInstance().IsProfLaunchType(kernel_type);
  const bool is_extend_type = (iter == kNamesToProfTypes.cend() && !is_prof_launch_type);
  const bool is_for_host_l1 =
      (!is_for_additional_info && GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kCannHostL1));

  if (is_extend_type) {
    if (!is_for_host_l1) {
      return kInvalidProfKernelType;
    }
    GELOGI("Kernel type %s is not builtin.", kernel_type);
    uint32_t type_id;
    if (ge::AddOverflow(static_cast<uint32_t>(GlobalProfilingWrapper::GetInstance()->RegisterString(kernel_type)),
                        static_cast<uint32_t>(GeProfInfoType::kNodeLevelEnd), type_id) == ge::SUCCESS) {
      return type_id;
    } else {
      return kInvalidProfKernelType;
    }
  }

  const uint32_t prof_type_id =
      iter == kNamesToProfTypes.cend() ? MSPROF_REPORT_NODE_LAUNCH_TYPE : static_cast<uint32_t>(iter->second);
  if (prof_type_id != MSPROF_REPORT_NODE_LAUNCH_TYPE) {
    return prof_type_id;
  }
  if (!is_for_additional_info && GlobalProfilingWrapper::GetInstance()->IsEnabled(ProfilingType::kTaskTime)) {
    return kInvalidProfKernelType;
  }
  return prof_type_id;
}
}  // namespace gert
