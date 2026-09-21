/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_COMMON_UTILS_BIND_CPU_UTILS_H_
#define AIR_RUNTIME_COMMON_UTILS_BIND_CPU_UTILS_H_
#include "framework/common/debug/log.h"
#include "pthread.h"
#include "sched.h"
#include "ge/ge_api_error_codes.h"

namespace ge {
class BindCpuUtils {
 public:
  static Status BindCore(uint32_t cpu_id) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu_id, &mask);
    int32_t ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
    if (ret != 0) {
      GELOGW("[Set][Affinity] set affinity with cpu[%u] failed, ret=%d", cpu_id, ret);
    }

    if (CPU_ISSET(cpu_id, &mask) == 0) {
      GELOGW("[Check][Bind] check process bind to cpu[%u] failed.", cpu_id);
    }
    return SUCCESS;
  }
};
}  // namespace ge
#endif  // AIR_RUNTIME_COMMON_UTILS_BIND_CPU_UTILS_H_
