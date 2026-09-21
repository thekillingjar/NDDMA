/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_DEPLOY_MODEL_RECV_FLOW_MODEL_RECEIVER_H_
#define AIR_RUNTIME_HETEROGENEOUS_DEPLOY_MODEL_RECV_FLOW_MODEL_RECEIVER_H_

#include <cstdint>
#include <fstream>
#include <map>
#include <mutex>
#include "deploy/deployer/deploy_state.h"

namespace ge {
class FlowModelReceiver {
 public:
  void DestroyDeployState(uint32_t model_id);
  void DestroyAllDeployStates();

  Status UpdateDeployPlan(const deployer::UpdateDeployPlanRequest &request);

  Status AddFlowRoutePlan(const deployer::AddFlowRoutePlanRequest &request);

  Status AppendToFile(const std::string &path, const char_t *file_content, size_t size, bool is_eof);

  Status GetDeployState(uint32_t root_model_id, DeployState *&deploy_state);

  Status AddDataGwSchedInfos(const deployer::DataGwSchedInfos &request);

 private:
  std::string GetDirectory(const std::string &file_path) const;

  std::mutex mu_;
  std::map<uint32_t, DeployState> deploy_states_;
  std::map<std::string, std::unique_ptr<std::ofstream>> receiving_files_;
};
}  // namespace ge

#endif  // AIR_RUNTIME_HETEROGENEOUS_DEPLOY_MODEL_RECV_FLOW_MODEL_RECEIVER_H_
