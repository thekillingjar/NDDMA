/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef COMMON_BIND_CPU_UTILS_H
#define COMMON_BIND_CPU_UTILS_H

#include "udf_log.h"
#include "pthread.h"
#include "sched.h"

namespace FlowFunc {
class BindCpuUtils {
public:
    static void BindCore(uint32_t cpu_id) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu_id, &mask);
        int32_t ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &mask);
        if (ret != 0) {
            UDF_LOG_WARN("set affinity with cpu[%u] failed, ret=%d", cpu_id, ret);
        }

        if (CPU_ISSET(cpu_id, &mask) == 0) {
            UDF_LOG_WARN("check process bind to cpu[%u] failed.", cpu_id);
        }
    }
};
}  // namespace FlowFunc
#endif  // COMMON_BIND_CPU_UTILS_H
