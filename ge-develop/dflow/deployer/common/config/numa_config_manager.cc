/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/config/numa_config_manager.h"
#include "mmpa/mmpa_api.h"
#include "common/config/configurations.h"
#include "common/config/json_parser.h"
#include "common/config/config_parser.h"
#include "graph/ge_global_options.h"
#include "graph/ge_context.h"
#include "framework/common/ge_types.h"
#include "deploy/deployer/deployer_proxy.h"

namespace ge {
namespace {
using Json = nlohmann::json;
}  // namespace

static void to_json(nlohmann::json &j, const ItemDeviceInfo &item_device_info) {
  j = Json();
  j["device_id"] = item_device_info.device_id;
}

static void to_json(nlohmann::json &j, const ItemDef &item_def) {
  j = Json();
  j["item_type"] = item_def.item_type;
  if (!item_def.resource_type.empty()) {
    j["resource_type"] = item_def.resource_type;
  }
  j["memory"] = item_def.memory;
  j["aic_type"] = item_def.aic_type;
  if (!item_def.links_mode.empty()) {
    j["links_mode"] = item_def.links_mode;
  }
  if (!item_def.device_list.empty()) {
    j["device_list"] = item_def.device_list;
  }
}

static void to_json(nlohmann::json &j, const LinkPair &link_pair) {
  j = std::vector<int32_t>{link_pair.id, link_pair.pair_id};
}

static void to_json(nlohmann::json &j, const ItemTopology &item_topology) {
  j = Json();
  j["links_mode"] = item_topology.links_mode;
  j["links"] = item_topology.links;
}

static void to_json(nlohmann::json &j, const NodeDef &node_def) {
  j = Json();
  j["node_type"] = node_def.node_type;
  if (!node_def.resource_type.empty()) {
    j["resource_type"] = node_def.resource_type;
  }
  j["support_links"] = node_def.support_links;
  j["item_type"] = node_def.item_type;
  if (!node_def.h2d_bw.empty()) {
    j["h2d_bw"] = node_def.h2d_bw;
  }
  if (!node_def.item_topology.empty()) {
    j["item_topology"] = node_def.item_topology;
  }
}

static void to_json(nlohmann::json &j, const Plane &plane) {
  j = Json();
  j["plane_id"] = plane.plane_id;
  j["devices"] = plane.devices;
}

static void to_json(nlohmann::json &j, const NodesTopology &nodes_topology) {
  j = Json();
  j["type"] = nodes_topology.type;
  if (!nodes_topology.protocol.empty()) {
    j["protocol"] = nodes_topology.protocol;
  }
  j["topos"] = nodes_topology.topos;
}

static void to_json(nlohmann::json &j, const ItemInfo &item_info) {
  j = Json();
  j["item_id"] = item_info.item_id;
  if (item_info.device_id >= 0) {
    j["device_id"] = item_info.device_id;
  }
}

static void to_json(nlohmann::json &j, const ClusterNode &cluster_node) {
  j = Json();
  j["node_id"] = cluster_node.node_id;
  j["node_type"] = cluster_node.node_type;
  if (cluster_node.memory >= 0) {
    j["memory"] = cluster_node.memory;
  }
  j["is_local"] = cluster_node.is_local;
  j["item_list"] = cluster_node.item_list;
}

static void to_json(nlohmann::json &j, const ClusterInfo &cluster_info) {
  j = Json();
  j["cluster_nodes"] = cluster_info.cluster_nodes;
  if (cluster_info.has_nodes_topology) {
    j["nodes_topology"] = cluster_info.nodes_topology;
  }
}

static void to_json(nlohmann::json &j, const NumaConfig &numa_config) {
  j = Json();
  j["cluster"] = numa_config.cluster;
  j["node_def"] = numa_config.node_def;
  j["item_def"] = numa_config.item_def;
}

std::string NumaConfigManager::ToJsonString(const NumaConfig &numa_config) {
  try {
    const Json j = numa_config;
    return j.dump();
  } catch (const nlohmann::json::exception &e) {
    GELOGE(FAILED, "Failed to dump object, err = %s", e.what());
    return "";
  }
}

Status NumaConfigManager::InitServerNumaConfig(NumaConfig &numa_config) {
  std::string file_path;
  GE_CHK_STATUS_RET_NOLOG(Configurations::GetInstance().GetResourceConfigPath(file_path));
  GE_CHK_STATUS_RET_NOLOG(ConfigParser::InitNumaConfig(file_path, numa_config));
  return SUCCESS;
}

bool NumaConfigManager::ExportOptionSupported() {
  std::string resource_path;
  (void)ge::GetContext().GetOption(RESOURCE_CONFIG_PATH, resource_path);

  const char_t *env_resource_config_path = nullptr;
  MM_SYS_GET_ENV(MM_ENV_RESOURCE_CONFIG_PATH, env_resource_config_path);
  bool enable_env_resource_config_path = (env_resource_config_path != nullptr);
  return enable_env_resource_config_path || !resource_path.empty();
}

Status NumaConfigManager::InitNumaConfig() {
  if (!ExportOptionSupported()) {
    return SUCCESS;
  }
  NumaConfig numa_config;
  GE_CHK_STATUS_RET(InitServerNumaConfig(numa_config), "Failed to init numa config");
  std::string numa_config_json_string = ToJsonString(numa_config);
  GE_CHK_BOOL_RET_STATUS(!numa_config_json_string.empty(), FAILED, "Invalid json string");
  auto &global_options_mutex = GetGlobalOptionsMutex();
  const std::lock_guard<std::mutex> lock(global_options_mutex);
  auto &global_options = GetMutableGlobalOptions();
  global_options[OPTION_NUMA_CONFIG] = numa_config_json_string;
  GELOGI("Set %s = %s", OPTION_NUMA_CONFIG, numa_config_json_string.c_str());
  GELOGI("Success to init numa config");
  return SUCCESS;
}
}  // namespace ge
