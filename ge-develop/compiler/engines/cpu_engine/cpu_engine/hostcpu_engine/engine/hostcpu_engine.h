/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_HOST_CPU_ENGINE_H_
#define AICPU_HOST_CPU_ENGINE_H_

#include "common/engine/base_engine.h"
#include "common/util/constant.h"

namespace aicpu {
using BaseEnginePtr = std::shared_ptr<BaseEngine>;

class HostCpuEngine : public BaseEngine {
 public:
  /**
   * get HostCpuEngine instance.
   * @return  HostCpuEngine instance.
   */
  static BaseEnginePtr Instance();

  /**
   * Destructor
   */
  virtual ~HostCpuEngine() = default;

  /**
   * When Ge start, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  ge::Status Initialize(
      const std::map<std::string, std::string> &options) override;

  /**
   * After the initialize, GE will invoke this interface
   * to get the Ops kernel Store.
   * @param opsKernels The tf ops kernel info
   */
  void GetOpsKernelInfoStores(std::map<std::string, OpsKernelInfoStorePtr>
                              &ops_kernel_info_stores) const override;

  /**
   * After the initialize, GE will invoke this interface
   * to get the Graph Optimizer.
   * @param graph_optimizers The tf Graph Optimizer objs
   */
  void GetGraphOptimizerObjs(std::map<std::string, GraphOptimizerPtr>
                             &graph_optimizers) const override;

  /**
   * After the initialize, GE will invoke this interface
   * to get the Graph Optimizer.
   * @param opsKernelBuilders The tf ops kernel builder objs
   */
  void GetOpsKernelBuilderObjs(std::map<std::string, OpsKernelBuilderPtr>
                               &ops_kernel_builders) const override;

  /**
   * When the graph finished, GE will invoke this interface
   * @return The status whether initialize successfully
   */
  ge::Status Finalize() override;

  // Copy prohibited
  HostCpuEngine(const HostCpuEngine &hostcpu_engine) = delete;

  // Move prohibited
  HostCpuEngine(const HostCpuEngine &&hostcpu_engine) = delete;

  // Copy prohibited
  HostCpuEngine &operator=(const HostCpuEngine &hostcpu_engine) = delete;

  // Move prohibited
  HostCpuEngine &operator=(HostCpuEngine &&hostcpu_engine) = delete;

 private:
  /**
   * Contructor
   */
  HostCpuEngine() : BaseEngine(kHostCpuEngine) {}

 private:
  static BaseEnginePtr instance_;
};
}  // namespace aicpu
#endif  // AICPU_HOSTCPU_ENGINE_H_
