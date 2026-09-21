/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RTS_ENGINE_ENGINE_RTS_ENGINE_H
#define RTS_ENGINE_ENGINE_RTS_ENGINE_H

#include <map>
#include <memory>
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/optimizer/graph_optimizer.h"

using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;

namespace cce {
namespace runtime {
/**
 * rts engine.
 * Used for the ops not belong to any engine. eg:netoutput
 */
class RtsEngine {
 public:
  /**
   * get RtsEngine instance.
   * @return  RtsEngine instance.
   */
  static RtsEngine &Instance();

  virtual ~RtsEngine() = default;

  /**
   * When Ge start, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  ge::Status Initialize(const std::map<string, string> &options);

  /**
   * After the initialize, GE will invoke this interface to get the Ops kernel Store
   * @param opsKernelMap The rts's ops kernel info
   */
  void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &opsKernelMap);

  /**
   * After the initialize, GE will invoke this interface to get the Graph Optimizer
   * @param graphOptimizers The rts's Graph Optimizer objs
   */
  void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graphOptimizers);

  /**
   * When the graph finished, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  ge::Status Finalize();

  // Copy prohibited
  RtsEngine(const RtsEngine &rtsEngine) = delete;

  // Move prohibited
  RtsEngine(const RtsEngine &&rtsEngine) = delete;

  // Copy prohibited
  RtsEngine &operator=(const RtsEngine &rtsEngine) = delete;

  // Move prohibited
  RtsEngine &operator=(RtsEngine &&rtsEngine) = delete;

 private:
  RtsEngine() = default;

 private:
  OpsKernelInfoStorePtr ops_kernel_store_ = nullptr;
  GraphOptimizerPtr graph_optimizer_ptr_ = nullptr;
  OpsKernelInfoStorePtr ffts_plus_ops_kernel_store_ = nullptr;
  GraphOptimizerPtr ffts_plus_graph_optimizer_ptr_ = nullptr;
};
}  // namespace runtime
}  // namespace cce

extern "C" {
/**
 * When Ge start, GE will invoke this interface
 * @return The status whether initialize successfully
 */
ge::Status Initialize(const map<string, string> &options);

/**
 * After the initialize, GE will invoke this interface to get the Ops kernel Store
 * @param opsKernelMap The rts's ops kernel info
 */
void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr> &opsKernelMap);

/**
 * After the initialize, GE will invoke this interface to get the Graph Optimizer
 * @param graphOptimizers The rts's Graph Optimizer objs
 */
void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr> &graphOptimizers);

/**
 * When the graph finished, GE will invoke this interface
 * @return The status whether initialize successfully
 */
ge::Status Finalize();
}

#endif  // RTS_ENGINE_ENGINE_RTS_ENGINE_H