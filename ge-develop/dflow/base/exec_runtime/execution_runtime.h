/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DFLOW_BASE_RUNTIME_EXECUTION_RUNTIME_H_
#define DFLOW_BASE_RUNTIME_EXECUTION_RUNTIME_H_

#include <memory>
#include <map>
#include <string>
#include "ge/ge_api_error_codes.h"
#include "dflow/base/deploy/model_deployer.h"
#include "dflow/base/deploy/exchange_service.h"

namespace ge {
class ExecutionRuntime {
 public:
  ExecutionRuntime() = default;
  GE_DELETE_ASSIGN_AND_COPY(ExecutionRuntime);
  virtual ~ExecutionRuntime() = default;

  static ExecutionRuntime *GetInstance();

  static Status InitHeterogeneousRuntime(const std::map<std::string, std::string> &options);

  static void FinalizeExecutionRuntime();

  static void SetExecutionRuntime(const std::shared_ptr<ExecutionRuntime> &instance);

  /// Initialize ExecutionRuntime
  /// @param execution_runtime    instance of execution runtime
  /// @param options              options for initialization
  /// @return                     SUCCESS if initialization is successful, otherwise returns appropriate error code
  virtual Status Initialize(const std::map<std::string, std::string> &options) = 0;

  /// Finalize ExecutionRuntime
  virtual Status Finalize() = 0;

  /// Get ModelDeployer
  /// @return                       pointer to the instance of ModelDeployer
  virtual ModelDeployer &GetModelDeployer() = 0;

  /// Get ExchangeService
  /// @return                       pointer to the instance of ExchangeService
  virtual ExchangeService &GetExchangeService() = 0;

  virtual const std::string &GetCompileHostResourceType() const;

  virtual const std::map<std::string, std::string> &GetCompileDeviceInfo() const;
 private:
  static Status LoadHeterogeneousLib();
  static Status SetupHeterogeneousRuntime(const std::map<std::string, std::string> &options);
  static std::mutex mu_;
  static void *handle_;
  static std::shared_ptr<ExecutionRuntime> instance_;
};
}  // namespace ge

#endif  // DFLOW_BASE_RUNTIME_EXECUTION_RUNTIME_H_
