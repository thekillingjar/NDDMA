/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_RUNTIME_HETEROGENEOUS_COMMON_CONFIG_CONFIG_PARSER_H_
#define AIR_RUNTIME_HETEROGENEOUS_COMMON_CONFIG_CONFIG_PARSER_H_

#include <string>
#include "ge/ge_api_error_codes.h"
#include "common/config/configurations.h"
#include "common/config/numa_config_manager.h"

namespace ge {
struct ClusterConfig;
struct NodeDefConfig;
struct ItemDefConfig;
class ConfigParser {
 public:
  /*
   *  @ingroup ge
   *  @brief   parse json file
   */
  static Status ParseServerInfo(const std::string &file_path, DeployerConfig &deployer_config);

  static Status InitNumaConfig(const std::string &file_path, NumaConfig &numa_config);

 private:
  static void InitNetWorkInfo(DeployerConfig &deployer_config);

  static Status InitDeployerConfig(const std::vector<ClusterConfig> &clusters,
                                   const std::vector<NodeDefConfig> &node_defs,
                                   const std::vector<ItemDefConfig> &item_defs,
                                   DeployerConfig &deployer_config);

  static Status InitAllNodeConfig(const std::vector<ClusterConfig> &clusters,
                                  const std::vector<NodeDefConfig> &node_defs,
                                  const std::vector<ItemDefConfig> &item_defs,
                                  std::vector<NodeConfig> &node_configs);

  static Status ParseTopologyLinks(const nlohmann::json &json_link, LinkPair &link_pair);

  static Status ParseClusterNode(const nlohmann::json &json_cluster_node,
                                 const int32_t local_node_id,
                                 ClusterNode &cluster_node);

  static Status ParseNodesTopology(const nlohmann::json &json_topology, NodesTopology &nodes_topology);

  static Status ParseTopologyPlanes(const nlohmann::json &json_topo, Plane &plane);

  static Status ParseClusterInfo(const nlohmann::json &json_config, ClusterInfo &cluster_info);

  static Status ParseNodeDef(const nlohmann::json &json_config, std::vector<NodeDef> &nodes_def);

  static Status ParseItemDef(const nlohmann::json &json_config, std::vector<ItemDef> &items_def);

  static Status GetResourceType(const std::vector<NodeDefConfig> &node_defs,
                                const std::vector<ItemDefConfig> &item_defs,
                                std::map<std::string, std::string> &node_type_to_node_resource_type,
                                std::map<std::string, std::string> &node_type_to_item_resource_type);

  static Status InitNodeResourceType(const std::map<std::string, std::string> &node_type_to_node_resource_type,
                                     const std::map<std::string, std::string> &node_type_to_item_resource_type,
                                     NodeConfig &node_config);

  static Status CheckProtocolType(const std::string &protocol);
};
}  // namespace ge
#endif  // AIR_RUNTIME_HETEROGENEOUS_COMMON_CONFIG_CONFIG_PARSER_H_
