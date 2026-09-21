/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXECUTION_POINT_UTIL_H
#define EXECUTION_POINT_UTIL_H

#include <mutex>
#include "api/session/jit_execution/exe_points/execution_point.h"
#include "api/session/jit_execution/exe_points/guarded_execution_point.h"
#include "guarded_execution_point_util.h"
#include "compiled_model_cache_util.h"

namespace ge {
class ExecutionPointUtil {
 public:
  ExecutionPointUtil() = default;
  ~ExecutionPointUtil() = default;

  Status GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key);

  Status CreateKeyOptionForGuardedExecutionPoint(const std::string root_dir, const std::string user_graph_key,
		               const GuardedExecutionPoint *gep, std::map<std::string, std::string> &options);

  // Restore GEPs in EP
  Status RestoreExecutionPoint(const std::string root_dir, const std::string user_graph_key,
		               const SliceGraphInfo &slice_graph_info, std::unique_ptr<ExecutionPoint> &exec_point_ptr);

  // Save pb files and gep_list.json
  Status SaveExecutionPoint(const std::string root_dir, const std::string user_graph_key,
		            const std::unique_ptr<ExecutionPoint> &exec_point_ptr);

  std::mutex ep_util_mutex_;

 private:
  // used to save the eps that have newly added geps.
  std::set<const ExecutionPoint*> updated_eps_;

  GuardedExecutionPointUtil gep_util_;
};
}
#endif // EXECUTION_POINT_UTIL_H