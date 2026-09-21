/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DVPP_ENGINE_ADAPTER_DVPP_ENGINE_H_
#define DVPP_ENGINE_ADAPTER_DVPP_ENGINE_H_

#include "adapter/dvpp_graph_optimizer.h"
#include "adapter/dvpp_ops_kernel_builder.h"
#include "adapter/dvpp_ops_kernel_info_store.h"

using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;
using OpsKernelBuilderPtr = std::shared_ptr<ge::OpsKernelBuilder>;
using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;

namespace dvpp {
class DvppEngine {
 public:
  /**
   * @brief Get DvppEngine instance
   * @return DvppEngine instance
   */
  static DvppEngine& Instance();

  /**
   * @brief Destructor
   */
  virtual ~DvppEngine() = default;

  // Copy constructor prohibited
  DvppEngine(const DvppEngine& dvpp_engine) = delete;

  // Move constructor prohibited
  DvppEngine(const DvppEngine&& dvpp_engine) = delete;

  // Copy assignment prohibited
  DvppEngine& operator=(const DvppEngine& dvpp_engine) = delete;

  // Move assignment prohibited
  DvppEngine& operator=(DvppEngine&& dvpp_engine) = delete;

  /**
   * @brief When GE start, GE will invoke dvpp Initialize
   * @param options initial options
   * @return status whether success
   */
  ge::Status Initialize(const std::map<std::string, std::string>& options);

  /**
   * @brief after initialize, GE will get dvpp graph optimizer
   * @param graph_optimizers all engine graph optimizer objs
   */
  void GetGraphOptimizerObjs(
      std::map<std::string, GraphOptimizerPtr>& graph_optimizers);

  /**
   * @brief after initialize, GE will get dvpp ops kernel builder
   * @param graph_optimizers all engine ops kernel builder objs
   */
  void GetOpsKernelBuilderObjs(
      std::map<std::string, OpsKernelBuilderPtr>& ops_kernel_builders);

  /**
   * @brief after initialize, GE will get dvpp ops kernel info store
   * @param graph_optimizers all engine ops kernel info store objs
   */
  void GetOpsKernelInfoStores(
      std::map<std::string, OpsKernelInfoStorePtr>& ops_kernel_info_stores);

  /**
   * @brief When Ge finsh, GE will invoke dvpp Finalize
   * @return status whether success
   */
  ge::Status Finalize() const;

 private:
  /**
   * @brief Constructor
   */
  DvppEngine() = default;

 private:
  GraphOptimizerPtr dvpp_graph_optimizer_{nullptr};
  OpsKernelBuilderPtr dvpp_ops_kernel_builder_{nullptr};
  OpsKernelInfoStorePtr dvpp_ops_kernel_info_store_{nullptr};
}; // class DvppEngine
} // namespace dvpp

extern "C" {
/**
 * @brief When GE start, GE will invoke dvpp Initialize
 * @param options initial options
 * @return status whether success
 */
ge::Status Initialize(const std::map<std::string, std::string>& options);

/**
 * @brief after initialize, GE will get dvpp graph optimizer
 * @param graph_optimizers all engine graph optimizer objs
 */
void GetGraphOptimizerObjs(
    std::map<std::string, GraphOptimizerPtr>& graph_optimizers);

/**
 * @brief after initialize, GE will get dvpp ops kernel builder
 * @param graph_optimizers all engine ops kernel builder objs
 */
void GetOpsKernelBuilderObjs(
    std::map<std::string, OpsKernelBuilderPtr>& ops_kernel_builders);

/**
 * @brief after initialize, GE will get dvpp ops kernel info store
 * @param graph_optimizers all engine ops kernel info store objs
 */
void GetOpsKernelInfoStores(
    std::map<std::string, OpsKernelInfoStorePtr>& ops_kernel_info_stores);

/**
 * @brief When Ge finsh, GE will invoke dvpp Finalize
 * @return status whether success
 */
ge::Status Finalize();
} // extern "C"

#endif // DVPP_ENGINE_ADAPTER_DVPP_ENGINE_H_