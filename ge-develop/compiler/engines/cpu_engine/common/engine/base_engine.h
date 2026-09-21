/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_BASE_ENGINE_H_
#define AICPU_BASE_ENGINE_H_

#include <map>
#include <string>

#include "aicpu_graph_optimizer/aicpu_graph_optimizer.h"
#include "aicpu_ops_kernel_builder/aicpu_ops_kernel_builder.h"
#include "aicpu_ops_kernel_info_store/aicpu_ops_kernel_info_store.h"
#include "factory/factory.h"

using OpsKernelInfoStorePtr = std::shared_ptr<ge::OpsKernelInfoStore>;
using GraphOptimizerPtr = std::shared_ptr<ge::GraphOptimizer>;
using OpsKernelBuilderPtr = std::shared_ptr<ge::OpsKernelBuilder>;
namespace aicpu {
using AicpuOpsKernelInfoStorePtr = std::shared_ptr<AicpuOpsKernelInfoStore>;
using AicpuGraphOptimizerPtr = std::shared_ptr<AicpuGraphOptimizer>;
using AicpuOpsKernelBuilderPtr = std::shared_ptr<AicpuOpsKernelBuilder>;
class BaseEngine {
 public:
  explicit BaseEngine(const std::string &engine_name)
      : engine_name_(engine_name),
        ops_kernel_info_store_(nullptr),
        graph_optimizer_(nullptr),
        ops_kernel_builder_(nullptr) {}
  virtual ~BaseEngine() = default;
  virtual ge::Status Initialize(
      const std::map<std::string, std::string> &options) = 0;
  virtual ge::Status Finalize() = 0;
  virtual void GetOpsKernelInfoStores(
      std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_info_stores)
      const = 0;
  virtual void GetGraphOptimizerObjs(
      std::map<std::string, GraphOptimizerPtr> &graph_optimizers) const = 0;
  virtual void GetOpsKernelBuilderObjs(
      std::map<std::string, OpsKernelBuilderPtr> &ops_kernel_builders) const = 0;
  const AicpuOpsKernelInfoStorePtr GetAicpuOpsKernelInfoStore() const {
    return ops_kernel_info_store_;
  }
  const AicpuGraphOptimizerPtr GetAicpuGraphOptimizer() const {
    return graph_optimizer_;
  }
  const AicpuOpsKernelBuilderPtr GetAicpuOpsKernelBuilder() const {
    return ops_kernel_builder_;
  }

 protected:
  /**
   * When load config file
   * @param instance, config file object
   * @return The status whether initialize successfully
   */
  ge::Status LoadConfigFile(const void *instance) const;

 protected:
  const std::string engine_name_;
  AicpuOpsKernelInfoStorePtr ops_kernel_info_store_;
  AicpuGraphOptimizerPtr graph_optimizer_;
  AicpuOpsKernelBuilderPtr ops_kernel_builder_;
};

#define FACTORY_ENGINE Factory<BaseEngine>

#define FACTORY_ENGINE_CLASS_KEY(CLASS, KEY) \
  FACTORY_ENGINE::Register<CLASS> __##CLASS(KEY);
}  // namespace aicpu

extern "C" {
/**
 * When Ge start, GE will invoke this interface
 * @return The status whether initialize successfully
 */
ge::Status Initialize(const std::map<std::string, std::string> &options);

/**
 * After the initialize, GE will invoke this interface to get the Ops kernel
 * Store
 * @param ops_kernel_info_stores The ops kernel info
 */
void GetOpsKernelInfoStores(
    std::map<std::string, OpsKernelInfoStorePtr> &ops_kernel_info_stores);

/**
 * After the initialize, GE will invoke this interface to get the Graph
 * Optimizer
 * @param graph_optimizers The graph optimizer objs
 */
void GetGraphOptimizerObjs(
    std::map<std::string, GraphOptimizerPtr> &graph_optimizers);

/**
 * After the initialize, GE will invoke this interface to get the kernel builder
 * util
 * @param ops_kernel_builders The ops kernel builder objs
 */
void GetOpsKernelBuilderObjs(
    std::map<std::string, OpsKernelBuilderPtr> &ops_kernel_builders);

/**
 * When the graph finished, GE will invoke this interface
 * @return The status whether initialize successfully
 */
ge::Status Finalize();
}

#endif