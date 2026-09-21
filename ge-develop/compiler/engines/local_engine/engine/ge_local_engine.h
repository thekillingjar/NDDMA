/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_ENGINE_GE_LOCAL_ENGINE_H_
#define GE_GE_LOCAL_ENGINE_ENGINE_GE_LOCAL_ENGINE_H_

#if defined(_MSC_VER)
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY _declspec(dllexport)
#else
#define GE_FUNC_VISIBILITY
#endif
#else
#ifdef FUNC_VISIBILITY
#define GE_FUNC_VISIBILITY __attribute__((visibility("default")))
#else
#define GE_FUNC_VISIBILITY
#endif
#endif

#include <map>
#include <memory>
#include <string>
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"
#include "ge_local_graph_optimizer.h"

using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;
using GeLocalGraphOptimizerPtr = std::shared_ptr<ge::GeLocalGraphOptimizer>;

namespace ge {
namespace ge_local {
/**
 * ge local engine.
 * Used for the ops not belong to any engine. eg:netoutput
 */
class GE_FUNC_VISIBILITY GeLocalEngine {
 public:
  /**
   * get GeLocalEngine instance.
   * @return  GeLocalEngine instance.
   */
  static GeLocalEngine &Instance();

  virtual ~GeLocalEngine() = default;

  /**
   * When Ge start, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  Status Initialize(const std::map<std::string, std::string> &options);

  /**
   * After the initialize, GE will invoke this interface
   * to get the Ops kernel Store.
   * @param ops_kernel_map The ge local's ops kernel info
   */
  void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map);

  /**
   * After the initialize, GE will invoke this interface
   * to get the Graph Optimizer.
   * @param graph_optimizers The ge local's Graph Optimizer objs
   */
  void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers) const;

  /**
   * When the graph finished, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  Status Finalize();

  // Copy prohibited
  GeLocalEngine(const GeLocalEngine &geLocalEngine) = delete;

  // Move prohibited
  GeLocalEngine(const GeLocalEngine &&geLocalEngine) = delete;

  // Copy prohibited
  GeLocalEngine &operator=(const GeLocalEngine &geLocalEngine) = delete;

  // Move prohibited
  GeLocalEngine &operator=(GeLocalEngine &&geLocalEngine) = delete;

 private:
  GeLocalEngine() = default;

  OpsKernelInfoStorePtr ops_kernel_store_ = nullptr;

  GeLocalGraphOptimizerPtr ge_local_graph_optimizer_ = nullptr;
};
}  // namespace ge_local
}  // namespace ge

extern "C" {
/**
 * When Ge start, GE will invoke this interface
 * @return The status whether initialize successfully
 */
GE_FUNC_VISIBILITY ge::Status Initialize(const std::map<std::string, std::string> &options);

/**
 * After the initialize, GE will invoke this interface to get the Ops kernel Store
 * @param ops_kernel_map The ge local's ops kernel info
 */
GE_FUNC_VISIBILITY void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_map);

/**
 * After the initialize, GE will invoke this interface to get the Graph Optimizer
 * @param graph_optimizers The ge local's Graph Optimizer objs
 */
GE_FUNC_VISIBILITY void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graph_optimizers);

/**
 * When the graph finished, GE will invoke this interface
 * @return The status whether initialize successfully
 */
GE_FUNC_VISIBILITY ge::Status Finalize();
}

#endif  // GE_GE_LOCAL_ENGINE_ENGINE_GE_LOCAL_ENGINE_H_
