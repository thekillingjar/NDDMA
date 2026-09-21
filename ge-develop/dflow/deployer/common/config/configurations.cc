/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config/configurations.h"
#include "common/config/json_parser.h"
#include "common/debug/log.h"
#include "common/config/config_parser.h"
#include "dflow/base/utils/process_utils.h"
#include "common/checker.h"
#include "framework/common/ge_types.h"
#include "graph/ge_context.h"
#include "graph_metadef/graph/utils/file_utils.h"

namespace ge {
namespace {
const char_t *const kConfigFileName = "/resource.json";
const char_t *const kHelperResFilePath = "HELPER_RES_FILE_PATH";
const char_t *const kResourceConfigPath = "RESOURCE_CONFIG_PATH";
const char_t *const kHomeEnvName = "HOME";
}  // namespace

Configurations &Configurations::GetInstance() {
  static Configurations instance;
  return instance;
}

void Configurations::Finalize() {
  information_ = {};
  GELOGI("Finalize success, remote node size = %zu", information_.remote_node_config_list.size());
}

Status Configurations::GetConfigDir(std::string &config_dir) {
  const char_t *file_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HELPER_RES_FILE_PATH, file_path);
  if (file_path != nullptr) {
    const std::string real_path = RealPath(file_path);
    if (real_path.empty()) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "The path[%s] of env[%s] is invalid", file_path, kHelperResFilePath);
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    if (ProcessUtils::IsValidPath(real_path) != SUCCESS) {
      GELOGE(ACL_ERROR_GE_PARAM_INVALID, "env %s config value[%s] real path[%s] is invalid", kHelperResFilePath,
             file_path, real_path.c_str());
      return ACL_ERROR_GE_PARAM_INVALID;
    }
    config_dir = real_path;
    GEEVENT("Get config dir[%s] success from env[%s]", config_dir.c_str(), kHelperResFilePath);
    return SUCCESS;
  }
  GELOGE(ACL_ERROR_GE_PARAM_INVALID, "Env HELPER_RES_FILE_PATH  don't exist.");
  return ACL_ERROR_GE_PARAM_INVALID;
}

Status Configurations::GetWorkingDir(std::string &working_dir) const {
  if (!information_.node_config.deploy_res_path.empty()) {
    working_dir = information_.node_config.deploy_res_path;
    GELOGI("Get working dir success, path = %s", working_dir.c_str());
    return SUCCESS;
  }

  std::string ascend_work_path;
  GE_ASSERT_SUCCESS(GetAscendWorkPath(ascend_work_path));
  if (!ascend_work_path.empty()) {
    working_dir = ascend_work_path;
  } else {
    working_dir = GetHostDirByEnv();
  }
  GE_CHK_BOOL_RET_STATUS(!working_dir.empty(), ACL_ERROR_GE_PARAM_INVALID, "Env HOME don't exist.");
  GELOGI("Get working dir success, path = %s", working_dir.c_str());
  return SUCCESS;
}

std::string Configurations::GetDeployResDir() {
  return information_.working_dir + "/runtime/deploy_res/";
}

std::string Configurations::GetHostDirByEnv() {
  const char_t *res_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_HOME, res_path);
  if (res_path == nullptr) {
    return "";
  }
  const std::string real_path = RealPath(res_path);
  if (real_path.empty()) {
    GELOGW("The path[%s] of env[%s] is invalid", res_path, kHomeEnvName);
    return "";
  }
  if (ProcessUtils::IsValidPath(real_path) != SUCCESS) {
    GELOGW("env %s config value[%s] real path[%s] is invalid", kHomeEnvName, res_path, real_path.c_str());
    return "";
  }
  GELOGI("Get host dir success from env[%s], path = %s", kHomeEnvName, real_path.c_str());
  return real_path;
}

std::vector<NodeConfig> Configurations::GetAllNodeConfigs() const {
  std::vector<NodeConfig> configs {information_.node_config};
  configs.insert(configs.cend(),
                 information_.remote_node_config_list.cbegin(),
                 information_.remote_node_config_list.cend());
  std::sort(configs.begin(), configs.end(),
            [](const NodeConfig &lhs, const NodeConfig &rhs) -> bool {
              return (lhs.node_id < rhs.node_id);
            });
  return configs;
}

const std::vector<NodeConfig> &Configurations::GetRemoteNodeConfigs() const {
  return information_.remote_node_config_list;
}

Status Configurations::GetResourceConfigPath(std::string &config_dir) {
  std::string resource_path;
  const char_t *file_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_RESOURCE_CONFIG_PATH, file_path);
  if (file_path == nullptr) {
    (void)ge::GetContext().GetOption(RESOURCE_CONFIG_PATH, resource_path);
    GE_CHK_BOOL_RET_STATUS(!resource_path.empty(), ACL_ERROR_GE_PARAM_INVALID,
                           "Env RESOURCE_CONFIG_PATH or option[%s] does not exist.", RESOURCE_CONFIG_PATH);
    GEEVENT("Get config dir[%s] success from option[%s]", resource_path.c_str(), RESOURCE_CONFIG_PATH);
  } else {
    resource_path = file_path;
    GEEVENT("Get config dir[%s] success from env[%s]", resource_path.c_str(), kResourceConfigPath);
  }
  const std::string real_path = RealPath(resource_path.c_str());
  GE_CHK_BOOL_RET_STATUS(!real_path.empty(),
                         ACL_ERROR_GE_PARAM_INVALID,
                         "The path[%s] is invalid", resource_path.c_str());
  config_dir = resource_path;
  return SUCCESS;
}

Status Configurations::InitInformation() {
  information_ = {};
  std::string file_path;
  GE_CHK_STATUS_RET_NOLOG(GetResourceConfigPath(file_path));
  GE_CHK_STATUS_RET_NOLOG(ConfigParser::ParseServerInfo(file_path, information_));
  GE_CHK_STATUS_RET_NOLOG(GetWorkingDir(information_.working_dir));
  return SUCCESS;
}

const NodeConfig &Configurations::GetLocalNode() const {
  return information_.node_config;
}

std::vector<std::string> Configurations::GetHeterogeneousEnvs() {
  static std::vector<std::string> envs = {kResourceConfigPath};
  return envs;
}
}  // namespace ge
