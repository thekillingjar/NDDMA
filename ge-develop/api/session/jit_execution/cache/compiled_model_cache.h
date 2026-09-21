/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILED_MODEL_CACHE_H
#define COMPILED_MODEL_CACHE_H
#include <mutex>
#include <fstream>
#include "api/session/jit_execution/exe_points/execution_order.h"
#include "api/session/jit_execution/compile_context.h"
#include "api/session/jit_execution/utils/compiled_model_cache_util.h"
#include "api/session/jit_execution/utils/execution_order_util.h"

namespace ge {

/**
 * 该类管理一张UserGraph中的编译缓存结果和断图结果
 * 包括 ExecutionOrder(EO)/ExecutionPoint(EP)/GuardExecutionPoint(GEP)
 */
class CompiledModelCache {
 public:
  CompiledModelCache() = delete;

  CompiledModelCache(uint32_t user_graph_id, CompileContext &context, GraphManager &graph_manager);

  CompiledModelCache(const CompiledModelCache &) = delete;

  CompiledModelCache &operator=(const CompiledModelCache &) = delete;

  ~CompiledModelCache() = default;

  Status GetGuardedExecutionPointGraphKey(const GuardedExecutionPoint *gep, std::string &gep_graph_key);

  // Create or emplace gep_graph_key for option
  Status CreateKeyOptionForGuardedExecutionPoint(const GuardedExecutionPoint *gep, std::map<std::string, std::string> &options);

  // Load all cache.
  Status RestoreCache(ExecutionOrder &order);

  // Save all cache.
  Status SaveCache(ExecutionOrder &order);

 private:
  uint32_t user_graph_id_;

  CompileContext &compile_context_;

  GraphManager &graph_manager_;

  std::string user_graph_key_;

  std::string root_dir_;  // cache_dir when jit is opened

  ExecutionOrderUtil eo_util_;
};
}  // namespace ge

#endif  // COMPILED_MODEL_CACHE_H