/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GUARDED_EXECUTION_POINT_UTIL_H
#define GUARDED_EXECUTION_POINT_UTIL_H

#include <mutex>
#include "api/session/jit_execution/exe_points/execution_point.h"
#include "api/session/jit_execution/exe_points/guarded_execution_point.h"
#include "compiled_model_cache_util.h"

namespace ge {
class GuardedExecutionPointUtil {
 public:
  GuardedExecutionPointUtil() = default;
  ~GuardedExecutionPointUtil() = default;

  Status AddGuardedExecutionPointGraphKey(std::string root_dir, std::string user_graph_key, const GuardedExecutionPoint *gep);

  Status EmplaceGuardedExecutionPointOption(std::string root_dir, std::string user_graph_key, const GuardedExecutionPoint *gep,
                                            std::map<std::string, std::string> &options);

  Status GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key);

  /* check whether the gep has been registered in cmc */
  Status HasGuardedExecutionPointRegistered(const GuardedExecutionPoint *gep, bool &has_registered);

  // Restore GEP by gep_graph_key
  Status RestoreGuardedExecutionPoint(const std::string root_dir, const GuardedExecutionPointInfo &info,
		                      ExecutionPoint &exec_point, GuardedExecutionPoint *gep);

  // Save gep_list.json by gep_graph_graph_key, which is found by gep
  Status SaveGuardedExecutionPoint(std::string user_graph_root_dir, const ExecutionPoint &exec_point,
		                          const std::vector<std::unique_ptr<GuardedExecutionPoint>> &geps);

 private:
  std::mutex gep_util_mutex_;

  std::map<const GuardedExecutionPoint*, std::string> gep_to_gep_graph_key_;
};
}
#endif // GUARDED_EXECUTION_POINT_UTIL_H