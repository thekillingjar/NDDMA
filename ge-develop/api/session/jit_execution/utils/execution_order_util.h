/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTION_ORDER_UTIL_H
#define EXECUTION_ORDER_UTIL_H

#include <mutex>
#include "api/session/jit_execution/exe_points/execution_order.h"
#include "api/session/jit_execution/exe_points/guarded_execution_point.h"
#include "execution_point_util.h"
#include "compiled_model_cache_util.h"

namespace ge {
class ExecutionOrderUtil {
  public:
    ExecutionOrderUtil() = default;
    ~ExecutionOrderUtil() = default;

    Status GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key);

    Status CreateKeyOptionForGuardedExecutionPoint(const std::string root_dir, const std::string user_graph_key,
                         const GuardedExecutionPoint *gep, std::map<std::string, std::string> &options);

    Status RestoreExecutionOrder(const std::string root_dir, const std::string user_graph_key, ExecutionOrder &order);

    Status SaveExecutionOrder(const std::string root_dir, const std::string user_graph_key, uint32_t user_graph_id, ExecutionOrder &order);

  private:
    ExecutionPointUtil ep_util_;
  };
}
#endif // EXECUTION_ORDER_UTIL_H