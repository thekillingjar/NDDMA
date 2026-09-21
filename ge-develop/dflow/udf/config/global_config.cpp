/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "global_config.h"

namespace FlowFunc {
thread_local uint32_t GlobalConfig::current_sched_thread_idx_{0};
thread_local uint32_t GlobalConfig::current_sched_group_id_{0};

#ifdef RUN_ON_DEVICE
bool GlobalConfig::on_device_ = true;
#else
bool GlobalConfig::on_device_ = false;
#endif

GlobalConfig &GlobalConfig::Instance() {
    static GlobalConfig inst;
    return inst;
}
}
