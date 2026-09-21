/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/heterogeneous_execution_runtime.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "deploy/flowrm/network_manager.h"
#include "deploy/flowrm/tsd_client.h"
#include "common/mem_grp/memory_group_manager.h"
#include "common/subprocess/subprocess_manager.h"
#include "common/utils/rts_api_utils.h"
#include "deploy/deployer/master_model_deployer.h"
#include "dflow/base/exec_runtime/execution_runtime.h"
#include "common/config/numa_config_manager.h"
#include "deploy/resource/resource_manager.h"

namespace ge {
class HeterogeneousExecutionRuntime : public ExecutionRuntime {
 public:
  static HeterogeneousExecutionRuntime& GetInstance() {
    static HeterogeneousExecutionRuntime instance;
    return instance;
  }
  Status Initialize(const std::map<std::string, std::string> &options) override;
  Status Finalize() override;
  ModelDeployer &GetModelDeployer() override;
  ExchangeService &GetExchangeService() override;
  const std::string &GetCompileHostResourceType() const override;
  const std::map<std::string, std::string> &GetCompileDeviceInfo() const override;
 private:
  void RollbackInit() const;
  Status InitializeInner(const std::map<std::string, std::string> &options);

  MasterModelDeployer model_deployer_;
};

Status HeterogeneousExecutionRuntime::Initialize(const std::map<std::string, std::string> &options) {
  GE_DISMISSABLE_GUARD(rollback, ([this]() { RollbackInit(); }));
  GE_CHK_STATUS_RET(InitializeInner(options), "Failed to initialize.");
  GE_DISMISS_GUARD(rollback);
  return ge::SUCCESS;
}

Status HeterogeneousExecutionRuntime::InitializeInner(const std::map<std::string, std::string> &options) {
  GE_CHK_STATUS_RET_NOLOG(Configurations::GetInstance().InitInformation());
  GE_CHK_STATUS_RET_NOLOG(SubprocessManager::GetInstance().Initialize());
  GE_CHK_STATUS_RET_NOLOG(RtsApiUtils::MbufInit());
  (void) MemoryGroupManager::GetInstance().Initialize(Configurations::GetInstance().GetLocalNode());
  GE_CHK_STATUS_RET(HeterogeneousExchangeService::GetInstance().Initialize(0), "Failed to init model deployer");
  GE_CHK_STATUS_RET(model_deployer_.Initialize(options), "Failed to init model deployer");
  GE_CHK_STATUS_RET(NumaConfigManager::InitNumaConfig(), "Failed to init numa config");
  return SUCCESS;
}

Status HeterogeneousExecutionRuntime::Finalize() {
  GELOGD("Heterogeneous execution runtime finalize begin.");
  GE_CHK_STATUS(model_deployer_.Finalize());
  GE_CHK_STATUS(HeterogeneousExchangeService::GetInstance().Finalize());
  GE_CHK_STATUS(NetworkManager::GetInstance().Finalize());
  SubprocessManager::GetInstance().Finalize();
  Configurations::GetInstance().Finalize();
  TsdClient::GetInstance().Finalize();
  MemoryGroupManager::GetInstance().Finalize();
  return SUCCESS;
}

ModelDeployer &HeterogeneousExecutionRuntime::GetModelDeployer() {
  return model_deployer_;
}

ExchangeService &HeterogeneousExecutionRuntime::GetExchangeService() {
  return HeterogeneousExchangeService::GetInstance();
}

void HeterogeneousExecutionRuntime::RollbackInit() const {
  SubprocessManager::GetInstance().Finalize();
  GE_CHK_STATUS(NetworkManager::GetInstance().Finalize());
}

const std::string &HeterogeneousExecutionRuntime::GetCompileHostResourceType() const {
  return ResourceManager::GetInstance().GetCompileResource().host_resource_type;
}

const std::map<std::string, std::string> &HeterogeneousExecutionRuntime::GetCompileDeviceInfo() const {
  return ResourceManager::GetInstance().GetCompileResource().logic_dev_id_to_res_type;
}
}  // namespace ge

ge::Status InitializeHeterogeneousRuntime(const std::map<std::string, std::string> &options) {
  auto heterogeneous_runtime = ge::MakeShared<ge::HeterogeneousExecutionRuntime>();
  GE_CHECK_NOTNULL(heterogeneous_runtime);
  GE_CHK_STATUS_RET_NOLOG(heterogeneous_runtime->Initialize(options));
  ge::ExecutionRuntime::SetExecutionRuntime(heterogeneous_runtime);
  return ge::SUCCESS;
}
