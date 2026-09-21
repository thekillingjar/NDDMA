/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_BLOCK_CHECKER_H_
#define AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_BLOCK_CHECKER_H_
#include <runtime/rt.h>
#include "common/debug/ge_log.h"
#include "common/checker.h"
#include "aicpu_args_handler.h"

namespace gert {
ge::Status CheckDeviceSupportBlockingAicpuOpProcess(bool &is_support);
ge::Status DistributeWaitTaskForAicpuBlockingOp(rtStream_t stream, const AicpuArgsHandler *arg_handler, const char *op_name);
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_GRAPH_BUILDER_AICPU_BLOCK_CHECKER_H_
