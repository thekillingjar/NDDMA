/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "heterogeneous_stub_env.h"
#include "common/env_path.h"

#include "deploy/flowrm/flow_route_planner.h"
#include "common/data_flow/route/rank_table_builder.h"
#include "deploy/resource/resource_manager.h"
#include "deploy/flowrm/network_manager.h"

namespace ge {
void HeterogeneousStubEnv::SetupAIServerEnv() {
  SubprocessManager::GetInstance().executable_paths_.emplace("queue_schedule", "/var/queue_schedule");
  LoadAIServerHostConfig("valid/server/numa_config2.json");

  auto &deployer_proxy = DeployerProxy::GetInstance();
  deployer_proxy.deployers_.emplace_back(MakeUnique<LocalDeployer>());
  deployer_proxy.deployers_[0]->Initialize();

  for (auto &node : Configurations::GetInstance().GetAllNodeConfigs()) {
    if (!node.is_local) {
      auto remote_deployer = MakeUnique<stub::MockRemoteDeployer>(node);
      remote_deployer->Initialize();
      deployer_proxy.deployers_.emplace_back(std::move(remote_deployer));
    }
  }

  ResourceManager::GetInstance().Initialize();
  ExecutionRuntime::SetExecutionRuntime(std::make_shared<stub::MockExecutionRuntime>());

  MasterModelDeployer master_model_deployer;
  HcomRankTable rank_table{};
  master_model_deployer.CreateRankTable(rank_table);
  DeployContext::LocalContext().GetRankTableBuilder().SetRankTable(rank_table);
  std::string rank_table_str;
  DeployContext::LocalContext().GetRankTableBuilder().GetRankTable(rank_table_str);
  PneExecutorClientCreatorRegistrar<stub::MockPneExecutorClient> registrar(PNE_ID_NPU);
}

void HeterogeneousStubEnv::LoadAIServerHostConfig(const string &path) {
  Configurations::GetInstance().information_ = DeployerConfig{};
  std::string config_path = PathUtils::Join({EnvPath().GetAirBasePath(), "tests/dflow/runner/ut/ge/runtime/data", path});
  setenv("RESOURCE_CONFIG_PATH", config_path.c_str(), 1);
  EXPECT_EQ(Configurations::GetInstance().InitInformation(), SUCCESS);
}

void HeterogeneousStubEnv::ClearEnv() {
  DeployerProxy::GetInstance().Finalize();
  ResourceManager::GetInstance().Finalize();
  DeviceDebugConfig::global_configs_ = nlohmann::json{};
  DeployContext::LocalContext().GetRankTableBuilder().ip_rank_map_.clear();
  Configurations::GetInstance().information_ = DeployerConfig{};
  NetworkManager::GetInstance().main_port_ = 0;
  unsetenv("RESOURCE_CONFIG_PATH");
}
}  // namespace ge
