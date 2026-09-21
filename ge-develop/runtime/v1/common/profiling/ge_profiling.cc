/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "framework/common/profiling/ge_profiling.h"
#include "common/profiling/profiling_manager.h"

#include "runtime/rt.h"
#include "framework/common/debug/log.h"
#include "graph/load/graph_loader.h"
#include "graph/ge_context.h"
#include "framework/common/ge_types.h"

ge::Status ProfGetDeviceFormGraphId(const uint32_t graph_id, uint32_t &device_id) {
  return ge::ProfilingManager::Instance().GetDeviceIdFromGraph(graph_id, device_id);
}