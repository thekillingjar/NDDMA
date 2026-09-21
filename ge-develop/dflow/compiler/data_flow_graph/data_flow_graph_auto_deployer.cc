/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dflow/compiler/data_flow_graph/data_flow_graph_auto_deployer.h"
#include <regex>
#include "framework/common/util.h"
#include "framework/common/types.h"
#include "common/checker.h"
#include "common/string_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_context.h"
#include "dflow/flow_graph/data_flow_attr_define.h"
#include "mmpa/mmpa_api.h"
#include "graph/ge_local_context.h"

namespace ge {
namespace {
constexpr const char *ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE = "_dflow_runnable_resource";
constexpr const char *ATTR_NAME_DATA_FLOW_FINAL_LOCATION = "_dflow_final_location";
constexpr const char *ATTR_NAME_DATA_FLOW_PROCESS_POINT_RELEASE_PKG = "_dflow_process_point_release_pkg";
constexpr const char *ATTR_NAME_DATA_FLOW_COMPILER_RESULT = "_dflow_compiler_result";
constexpr const char *ATTR_NAME_DATA_FLOW_HEAVY_LOAD = "_dflow_heavy_load";
constexpr const char *ATTR_NAME_FLOW_ATTR_FLOW_NODE_ALIAS = "_flow_attr_flow_node_alias";
constexpr const char *ATTR_NAME_DATA_FLOW_DEVICE_MEM_CFG = "_dflow_logic_device_memory_config";
constexpr const char *ATTR_NAME_DATA_FLOW_DYNAMIC_SCHEDULE_CFG = "dynamic_schedule_enable";
constexpr const char *ATTR_NAME_DATA_FLOW_INVOKE_DEPLOY_INFOS = "_invoke_deploy_infos";
constexpr const char *ATTR_NAME_DATA_FLOW_SUB_DATA_FLOW_DEPLOY_INFOS = "_sub_data_flow_deploy_infos";
constexpr const char *kDeployInfoFile = "deploy_info_file";
}  // namespace

Status DataFlowGraphAutoDeployer::AutoDeployDataFlowGraph(const DataFlowGraph &data_flow_graph,
                                                          const std::string &deploy_info_path) {
  const auto &root_graph = data_flow_graph.GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  std::map<std::string, std::pair<std::string, std::string>> deploy_logic_device_map;
  // key:flow_node_name; value:vector of <flow node invoke name, invoke name> pairs
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> invoke_deploy_map;
  // key device_id; value: std_mem && shared_mem
  std::map<std::string, std::pair<uint32_t, uint32_t>> device_id_to_mem_cfg;
  bool dynamic_schedule_enable = false;
  std::string logic_device_id;
  std::string redundant_logic_device_id;
  GE_CHK_STATUS_RET(GetConfigDeployInfo(deploy_logic_device_map, device_id_to_mem_cfg, dynamic_schedule_enable,
                                        invoke_deploy_map, deploy_info_path),
                                        "Get data flow config deploy info failed.");
  (void)root_graph->SetExtAttr(ATTR_NAME_DATA_FLOW_DEVICE_MEM_CFG, device_id_to_mem_cfg);
  if (data_flow_graph.IsRootDataFlow()) {
    (void)AttrUtils::SetBool(root_graph, ATTR_NAME_DATA_FLOW_DYNAMIC_SCHEDULE_CFG, dynamic_schedule_enable);
  }
  (void)AttrUtils::GetStr(root_graph, ATTR_NAME_LOGIC_DEV_ID, logic_device_id);
  (void)AttrUtils::GetStr(root_graph, ATTR_NAME_REDUNDANT_LOGIC_DEV_ID, redundant_logic_device_id);
  for (const NodePtr &node : root_graph->GetDirectNode()) {
    if (node->GetType() != FLOWNODE) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    bool is_heavy_load = false;
    (void)AttrUtils::GetBool(op_desc, ATTR_NAME_DATA_FLOW_HEAVY_LOAD, is_heavy_load);
    std::string node_deploy_name;
    (void)GetNodeDeployName(data_flow_graph, op_desc, node_deploy_name, 0);
    if (!deploy_logic_device_map.empty()) {
      auto find_ret = deploy_logic_device_map.find(node_deploy_name);
      if (find_ret == deploy_logic_device_map.end()) {
        GELOGE(FAILED, "node[%s]'s deploy info is not configured, deploy name %s", node->GetName().c_str(),
               node_deploy_name.c_str());
        return FAILED;
      }
      logic_device_id = find_ret->second.first;
      redundant_logic_device_id = find_ret->second.second;
    }

    if (!logic_device_id.empty() || !redundant_logic_device_id.empty()) {
      GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(op_desc, ATTR_NAME_LOGIC_DEV_ID, logic_device_id), FAILED,
                             "set attr[%s] to node[%s] failed", ATTR_NAME_LOGIC_DEV_ID.c_str(),
                             node->GetName().c_str());
      GELOGI("set attr[%s] value[%s] for node[%s] success.", ATTR_NAME_LOGIC_DEV_ID.c_str(), logic_device_id.c_str(),
             node->GetName().c_str());
      GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(op_desc, ATTR_NAME_REDUNDANT_LOGIC_DEV_ID,
                                               redundant_logic_device_id),
                             FAILED,
                             "set attr[%s] to node[%s] failed", ATTR_NAME_REDUNDANT_LOGIC_DEV_ID.c_str(),
                             node->GetName().c_str());
      GELOGI("set attr[%s] value[%s] for node[%s] success.", ATTR_NAME_REDUNDANT_LOGIC_DEV_ID.c_str(),
             redundant_logic_device_id.c_str(), node->GetName().c_str());
    }

    ge::NamedAttrs compile_results;
    if (!ge::AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_DATA_FLOW_COMPILER_RESULT, compile_results)) {
      GELOGI("Node[%s] doesn't have attr[%s], no need to select deploy resource", node->GetName().c_str(),
             ATTR_NAME_DATA_FLOW_COMPILER_RESULT);
      continue;
    }
    auto attrs = ge::AttrUtils::GetAllAttrs(compile_results);
    for (const auto &attr : attrs) {
      ge::NamedAttrs runnable_resources_info;
      ge::NamedAttrs current_compile_result;
      GE_CHK_STATUS_RET(compile_results.GetItem(attr.first).GetValue<ge::NamedAttrs>(current_compile_result),
                        "Get pp[%s]'s compile result failed.", attr.first.c_str());
      GE_CHK_BOOL_RET_STATUS(AttrUtils::GetNamedAttrs(current_compile_result, ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE,
                                                      runnable_resources_info),
                             FAILED, "Get pp[%s]'s attr[%s] from node[%s] failed.", attr.first.c_str(),
                             ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE, node->GetName().c_str());
      auto runnable_infos = ge::AttrUtils::GetAllAttrs(runnable_resources_info);
      std::vector<std::string> runnable_resources_type;
      for (const auto &runnable_info : runnable_infos) {
        runnable_resources_type.emplace_back(runnable_info.first);
      }
      GELOGI("Get pp[%s]'s runnable resource info[%s] from node[%s]", attr.first.c_str(),
             ToString(runnable_resources_type).c_str(), node->GetName().c_str());

      std::string select_resource_type;
      GE_CHK_STATUS_RET(
          SelectResourceType(runnable_resources_type, logic_device_id, select_resource_type, is_heavy_load),
          "select resource type for pp[%s] in node[%s] from runnable resource info[%s] failed.", attr.first.c_str(),
          node->GetName().c_str(), ToString(runnable_resources_type).c_str());

      GE_CHK_BOOL_RET_STATUS(
          AttrUtils::SetStr(current_compile_result, ATTR_NAME_DATA_FLOW_FINAL_LOCATION, select_resource_type), FAILED,
          "Set pp[%s]'s attr[%s] to node[%s] failed.", attr.first.c_str(), ATTR_NAME_DATA_FLOW_FINAL_LOCATION,
          node->GetName().c_str());
      GELOGI("Set pp[%s]'s attr[%s], value[%s] to node[%s] success.", attr.first.c_str(),
             ATTR_NAME_DATA_FLOW_FINAL_LOCATION, select_resource_type.c_str(), node->GetName().c_str());
      GE_CHK_BOOL_RET_STATUS(AttrUtils::SetNamedAttrs(compile_results, attr.first, current_compile_result), FAILED,
                             "Set pp[%s]'s compile result to node[%s] failed.", attr.first.c_str(),
                             node->GetName().c_str());
    }

    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetNamedAttrs(op_desc, ATTR_NAME_DATA_FLOW_COMPILER_RESULT, compile_results),
                           FAILED, "Set node[%s]'s attr[%s] failed.", node->GetName().c_str(),
                           ATTR_NAME_DATA_FLOW_COMPILER_RESULT);
    auto iter = invoke_deploy_map.find(node_deploy_name);
    if (iter != invoke_deploy_map.end()) {
      std::vector<std::string> invoke_deploy_infos;
      const auto &invoke_names = iter->second;
      (void)GetInvokeDeployInfos(invoke_names, deploy_logic_device_map, invoke_deploy_infos);
      GE_CHK_BOOL_RET_STATUS(AttrUtils::SetListStr(op_desc, ATTR_NAME_DATA_FLOW_INVOKE_DEPLOY_INFOS,
                             invoke_deploy_infos), FAILED,
                             "set attr[%s] to node[%s] failed", ATTR_NAME_DATA_FLOW_INVOKE_DEPLOY_INFOS,
                             node->GetName().c_str());
      GELOGI("set attr[%s] size[%zu] for node[%s] success.", ATTR_NAME_DATA_FLOW_INVOKE_DEPLOY_INFOS,
             invoke_deploy_infos.size(), node->GetName().c_str());
    }
  }

  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::GetNodeDeployName(const DataFlowGraph &data_flow_graph,
                                                    const OpDescPtr &op_desc,
                                                    std::string &node_deploy_name,
                                                    int32_t depth) {
  constexpr int32_t kMaxDepth = 16;
  if (depth >= kMaxDepth) {
    GELOGE(FAILED, "Depth limit (%d) reached.", depth);
    return FAILED;
  }
  constexpr const char *kExtAttrDeployAffinityNode = "_data_flow_deploy_affinity_node";
  std::string node_affinity_name = op_desc->TryGetExtAttr(kExtAttrDeployAffinityNode, std::string(""));
  if (!node_affinity_name.empty()) {
    const auto &root_graph = data_flow_graph.GetRootGraph();
    GE_CHECK_NOTNULL(root_graph);
    const auto affinity_node = root_graph->FindNode(node_affinity_name);
    GE_CHECK_NOTNULL(affinity_node);
    return GetNodeDeployName(data_flow_graph, affinity_node->GetOpDesc(), node_deploy_name, depth + 1);
  }

  (void)AttrUtils::GetStr(op_desc, ATTR_NAME_FLOW_ATTR_FLOW_NODE_ALIAS, node_deploy_name);
  if (!node_deploy_name.empty()) {
    GELOGI("Get node[%s] deploy name[%s] success", op_desc->GetName().c_str(), node_deploy_name.c_str());
    return SUCCESS;
  }

  node_deploy_name = op_desc->GetName();
  GELOGI("Get deploy node deploy name[%s] success", node_deploy_name.c_str());
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::GetInvokeDeployInfos(
           const std::vector<std::pair<std::string, std::string>> &invoke_names,
           const std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
           std::vector<std::string> &invoke_deploy_infos) {
  for (const auto &invoke_name : invoke_names) {
    auto find_ret = deploy_logic_device_map.find(invoke_name.first);
    if (find_ret != deploy_logic_device_map.end()) {
      std::string deploy_info = invoke_name.second + "@" + find_ret->second.first + ";" + find_ret->second.second;
      invoke_deploy_infos.emplace_back(deploy_info);
    }
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::SelectResourceType(const std::vector<std::string> &runnable_resources_type,
                                                     const std::string &logic_device_id, std::string &resources_type,
                                                     bool is_heavy_load) {
  if (runnable_resources_type.empty()) {
    GELOGE(FAILED, "runnable resource info is empty, can not select.");
    return FAILED;
  }
  // if no logic device id is specified, ascend is preferred.
  if (logic_device_id.empty()) {
    // server heavy load must assign device id.
    if (is_heavy_load) {
      GELOGE(FAILED, "heavy load must assign logic device id.");
      return FAILED;
    }
    if (find(runnable_resources_type.cbegin(), runnable_resources_type.cend(), kResourceTypeAscend) !=
        runnable_resources_type.cend()) {
      resources_type = kResourceTypeAscend;
    } else {
      resources_type = runnable_resources_type[0];
    }
    GELOGD("select resource type is [%s].", resources_type.c_str());
    return SUCCESS;
  }

  for (const auto &type : runnable_resources_type) {
    if (is_heavy_load) {
      if (type != kResourceTypeAscend) {
        resources_type = type;
        break;
      }
    } else {
      if (type == kResourceTypeAscend) {
        resources_type = type;
        break;
      }
    }
  }
  if (resources_type.empty()) {
    GELOGE(FAILED, "no suitable resource type found, logic_device_id=%s, runnable_resources_type=%s.",
           logic_device_id.c_str(), ToString(runnable_resources_type).c_str());
    return FAILED;
  }

  GELOGD("select resource type[%s], logic_device_id=%s.", resources_type.c_str(), logic_device_id.c_str());
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::UpdateFlowFuncDeployInfo(const DataFlowGraph &data_flow_graph) {
  const auto &root_graph = data_flow_graph.GetRootGraph();
  GE_CHECK_NOTNULL(root_graph);
  const auto &subgraphs = data_flow_graph.GetAllSubgraphs();
  for (const NodePtr &node : root_graph->GetDirectNode()) {
    if (node->GetType() != FLOWNODE) {
      continue;
    }
    auto op_desc = node->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    GE_CHK_STATUS_RET(UpdateFlowNodeSubGraphDeployInfo(op_desc, data_flow_graph),
                      "update flow node[%s] subgraph deploy info failed", node->GetName().c_str());
    ge::NamedAttrs compile_results;
    if (!ge::AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_DATA_FLOW_COMPILER_RESULT, compile_results)) {
      GELOGI("Node[%s] doesn't have attr[%s], no need to update deploy info", node->GetName().c_str(),
             ATTR_NAME_DATA_FLOW_COMPILER_RESULT);
      continue;
    }
    auto attrs = ge::AttrUtils::GetAllAttrs(compile_results);
    for (const auto &attr : attrs) {
      std::map<std::string, ComputeGraphPtr>::const_iterator iter = subgraphs.find(attr.first);
      GE_CHK_BOOL_RET_STATUS(iter != subgraphs.cend(), FAILED, "Cannot find pp[%s]'s subgraph.", attr.first.c_str());
      // for each subgraph, only has one FlowFunc Node
      const auto flow_func_node = (*iter).second->FindFirstNodeMatchType(FLOWFUNC);
      GE_CHECK_NOTNULL(flow_func_node);
      ge::NamedAttrs current_compile_result;
      GE_CHK_STATUS_RET(compile_results.GetItem(attr.first).GetValue<ge::NamedAttrs>(current_compile_result),
                        "Get pp[%s]'s compile result failed.", attr.first.c_str());

      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::SetNamedAttrs(iter->second,
                                                          ATTR_NAME_DATA_FLOW_COMPILER_RESULT,
                                                          current_compile_result),
                             FAILED, "Set pp[%s]'s attr[%s] to graph failed.",
                             attr.first.c_str(), ATTR_NAME_DATA_FLOW_COMPILER_RESULT);
      GELOGI("Set pp[%s]'s attr[%s] to graph success.", attr.first.c_str(), ATTR_NAME_DATA_FLOW_COMPILER_RESULT);

      ge::NamedAttrs runnable_resources_info;
      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetNamedAttrs(current_compile_result, ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE,
                                                          runnable_resources_info),
                             FAILED, "Get pp[%s]'s attr[%s] from node[%s] failed.", attr.first.c_str(),
                             ATTR_NAME_DATA_FLOW_RUNNABLE_RESOURCE, node->GetName().c_str());
      auto runnable_infos = ge::AttrUtils::GetAllAttrs(runnable_resources_info);
      std::string final_location;
      GE_CHK_BOOL_RET_STATUS(
          ge::AttrUtils::GetStr(current_compile_result, ATTR_NAME_DATA_FLOW_FINAL_LOCATION, final_location), FAILED,
          "Get pp[%s]'s attr[%s] from node[%s] failed.", attr.first.c_str(), ATTR_NAME_DATA_FLOW_FINAL_LOCATION,
          node->GetName().c_str());
      std::string release_dir;
      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetStr(runnable_resources_info, final_location, release_dir), FAILED,
                             "Get pp[%s]'s attr[%s] from node[%s] failed.", attr.first.c_str(), final_location.c_str(),
                             node->GetName().c_str());

      auto flow_func_desc = flow_func_node->GetOpDesc();
      GE_CHECK_NOTNULL(flow_func_desc);
      GE_CHK_BOOL_RET_STATUS(
          ge::AttrUtils::SetStr(flow_func_desc, ATTR_NAME_DATA_FLOW_PROCESS_POINT_RELEASE_PKG, release_dir), FAILED,
          "Set pp[%s]'s attr[%s], value[%s] to node[%s] failed.", attr.first.c_str(),
          ATTR_NAME_DATA_FLOW_PROCESS_POINT_RELEASE_PKG, release_dir.c_str(), flow_func_desc->GetName().c_str());
      GELOGI("Set pp[%s]'s attr[%s], value[%s] to node[%s] success.", attr.first.c_str(),
             ATTR_NAME_DATA_FLOW_PROCESS_POINT_RELEASE_PKG, release_dir.c_str(), flow_func_desc->GetName().c_str());
      GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::SetStr(flow_func_desc, ATTR_NAME_DATA_FLOW_FINAL_LOCATION, final_location),
                             FAILED, "set pp[%s]'s attr[%s], value[%s] to node[%s] failed.", attr.first.c_str(),
                             ATTR_NAME_DATA_FLOW_FINAL_LOCATION, final_location.c_str(),
                             flow_func_desc->GetName().c_str());
      GELOGI("Set pp[%s]'s attr[%s], value[%s] to node[%s] success.", attr.first.c_str(),
             ATTR_NAME_DATA_FLOW_FINAL_LOCATION, final_location.c_str(), flow_func_desc->GetName().c_str());
    }
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::ExpandToSingleLogicDevice(const std::string &logic_device_id_range,
                                                            std::vector<std::string> &logic_device_ids) {
  std::vector<std::string> segment_list = StringUtils::Split(logic_device_id_range, ':');
  // init to an empty string
  std::vector<std::string> tmp_expand_list = {""};
  for (const auto &segment : segment_list) {
    auto find_ret = segment.find('~');
    if (find_ret == std::string::npos) {
      for (auto &elem : tmp_expand_list) {
        elem.append(elem.empty() ? "" : ":").append(segment);
      }
      continue;
    }
    auto start_str = segment.substr(0, find_ret);
    auto end_str = segment.substr(find_ret + 1);
    int32_t start = 0;
    int32_t end = 0;
    try {
      start = std::stoi(start_str);
      end = std::stoi(end_str);
    } catch (const std::exception &e) {
      GELOGE(FAILED, "parse start and end for range[%s] exception, error=%s", segment.c_str(), e.what());
      return FAILED;
    }
    if (start > end) {
      GELOGE(FAILED, "config[%s] is invalid, start[%d] must be <= end[%d]", logic_device_id_range.c_str(), start, end);
      return FAILED;
    }
    if (start < 0 || end > INT16_MAX) {
      GELOGE(FAILED, "config[%s] is invalid, start[%d] or end[%d] is out of range[0, %d]",
             logic_device_id_range.c_str(), start, end, INT16_MAX);
      return FAILED;
    }
    size_t expand_size = tmp_expand_list.size() * static_cast<size_t>(end + 1 - start);
    if (expand_size > UINT16_MAX) {
      GELOGE(FAILED, "range[%s] config too many device, over %u", logic_device_id_range.c_str(), UINT16_MAX);
      return FAILED;
    }
    std::vector<std::string> tmp_list;
    tmp_list.reserve(expand_size);
    for (const auto &elem : tmp_expand_list) {
      std::string merge_prefix(elem);
      if (!merge_prefix.empty()) {
        merge_prefix.append(":");
      }
      for (int32_t i = start; i <= end; ++i) {
        tmp_list.emplace_back(merge_prefix + std::to_string(i));
      }
    }
    tmp_expand_list = std::move(tmp_list);
  }
  logic_device_ids = std::move(tmp_expand_list);
  GELOGI("config[%s] expand list size is %zu.", logic_device_id_range.c_str(), logic_device_ids.size());
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::CheckAndExpandLogicDeviceIds(const std::string &logic_device_id_list,
                                                               std::vector<std::string> &expand_logic_device_id_list) {
  std::regex no_range_config_regex(R"(((\d{1,5}:){2,3}\d{1,5},)*(\d{1,5}:){2,3}\d{1,5})");
  if (logic_device_id_list.find("~") == std::string::npos) {
    if (!std::regex_match(logic_device_id_list, no_range_config_regex)) {
      GELOGE(FAILED, "config logic_device_list[%s] is invalid", logic_device_id_list.c_str());
      return FAILED;
    }
    expand_logic_device_id_list.emplace_back(logic_device_id_list);
    return SUCCESS;
  }
  std::regex range_config_regex(R"((\d{1,5}(~\d{1,5})?:){2,3}\d{1,5}(~\d{1,5})?)");
  std::vector<std::string> split_list = StringUtils::Split(logic_device_id_list, ',');
  for (const auto &logic_device_id_conf : split_list) {
    if (!std::regex_match(logic_device_id_conf, range_config_regex)) {
      GELOGE(FAILED, "config logic_device_list[%s] is invalid, logic_device_id_conf=%s",
             logic_device_id_list.c_str(), logic_device_id_conf.c_str());
      return FAILED;
    }
    if (logic_device_id_conf.find("~") == std::string::npos) {
      expand_logic_device_id_list.emplace_back(logic_device_id_conf);
    } else {
      std::vector<std::string> logic_device_ids;
      GE_CHK_STATUS_RET(ExpandToSingleLogicDevice(logic_device_id_conf, logic_device_ids),
                        "expand [%s] to single logic device list failed.", logic_device_id_conf.c_str());
      expand_logic_device_id_list.insert(expand_logic_device_id_list.cend(), logic_device_ids.begin(),
                                         logic_device_ids.end());
    }
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::ExpandRangeConfig(
    std::vector<CompileConfigJson::FlowNodeBatchDeployInfo> &batch_deploy_info_list) {
  if (batch_deploy_info_list.empty()) {
    return SUCCESS;
  }

  for (auto &batch_deploy_info : batch_deploy_info_list) {
    std::string resolved_logic_device_id_list;
    std::string resolved_redundant_logic_device_id_list;
    GE_CHK_STATUS_RET(GetExpandLogicDeviceIds(batch_deploy_info.logic_device_list,
                                              resolved_logic_device_id_list),
                      "GetExpandLogicDeviceIds failed, flow_node_list[%s], logic_device_list[%s]",
                      ToString(batch_deploy_info.flow_node_list).c_str(),
                      batch_deploy_info.logic_device_list.c_str());
    if (!batch_deploy_info.redundant_logic_device_list.empty()) {
      GE_CHK_STATUS_RET(GetExpandLogicDeviceIds(batch_deploy_info.redundant_logic_device_list,
                                                resolved_redundant_logic_device_id_list),
                        "GetExpandLogicDeviceIds failed, flow_node_list[%s], redundant_logic_device_list[%s]",
                        ToString(batch_deploy_info.flow_node_list).c_str(),
                        batch_deploy_info.redundant_logic_device_list.c_str());
    }
    GELOGI("expand nodes %s logic devices list[%s] to [%s], redundant logic devices list[%s] to [%s]",
           ToString(batch_deploy_info.flow_node_list).c_str(),
           batch_deploy_info.logic_device_list.c_str(), resolved_logic_device_id_list.c_str(),
           batch_deploy_info.redundant_logic_device_list.c_str(),
           resolved_redundant_logic_device_id_list.c_str());
    batch_deploy_info.logic_device_list = resolved_logic_device_id_list;
    batch_deploy_info.redundant_logic_device_list = resolved_redundant_logic_device_id_list;
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::GetExpandLogicDeviceIds(const std::string &origin_logic_device_id_list,
                                                          std::string &resolved_logic_device_id_list) {
  std::vector<std::string> expand_logic_device_id_list;
  GE_CHK_STATUS_RET(CheckAndExpandLogicDeviceIds(origin_logic_device_id_list, expand_logic_device_id_list),
                    "CheckAndExpandLogicDeviceIds logic_device_list[%s] is invalid",
                    origin_logic_device_id_list.c_str());
  for (const auto &logic_device_id : expand_logic_device_id_list) {
    if (!resolved_logic_device_id_list.empty()) {
      resolved_logic_device_id_list.append(",");
    }
    resolved_logic_device_id_list.append(logic_device_id);
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::GetSortedLogicDeviceIds(const std::string &origin_logic_device_id_list,
                                                          std::string &resolved_logic_device_id_list) {
  std::vector<std::vector<std::string>> device_id_list;
  const auto logic_dev_id_list = StringUtils::Split(origin_logic_device_id_list, ',');
  device_id_list.reserve(logic_dev_id_list.size());
  for (const auto &logic_device_id : logic_dev_id_list) {
    const auto device_id_vector = StringUtils::Split(logic_device_id, ':');
    device_id_list.emplace_back(device_id_vector);
  }
  std::sort(device_id_list.begin(), device_id_list.end(),
            [](std::vector<std::string> &a, std::vector<std::string> &b) {
              for (size_t i = 0; i < a.size() && i < b.size(); i++) {
                if (a[i].length() != b[i].length()) {
                  return (a[i].length() < b[i].length());
                } else if (a[i] != b[i]) {
                  return (a[i] < b[i]);
                }
              }
              return (a.size() < b.size());
            });

  for (const auto &device_id : device_id_list) {
    std::string logic_device_id;
    for (const auto &id : device_id) {
      if (!logic_device_id.empty()) {
        logic_device_id.append(":");
      }
      logic_device_id.append(id);
    }
    if (!resolved_logic_device_id_list.empty()) {
      resolved_logic_device_id_list.append(",");
    }
    resolved_logic_device_id_list.append(logic_device_id);
  }
  GELOGI("logic device list [%s] to [%s]", origin_logic_device_id_list.c_str(), resolved_logic_device_id_list.c_str());
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::GetDeployLogicDeviceForInvoke(
    std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> &invoke_deploy_map,
    const std::vector<CompileConfigJson::InvokeDeployInfo> &invoke_deploy_infos,
    const std::string &flow_node_name, const bool dynamic_schedule_enable) {
  for (const auto &invoke_node : invoke_deploy_infos) {
    std::vector<std::pair<std::string, std::string>> flow_node_invoke = invoke_deploy_map[flow_node_name];
    const auto node_invoke_name = flow_node_name + "/" + invoke_node.invoke_name;
    auto iter = std::find_if(flow_node_invoke.begin(), flow_node_invoke.end(),
                             [&node_invoke_name](const std::pair<std::string, std::string> &pair) {
                              return pair.first == node_invoke_name;
                             });
    if (iter != flow_node_invoke.end()) {
      GELOGE(FAILED, "invoke[%s] repeat config on node [%s]", invoke_node.invoke_name.c_str(), flow_node_name.c_str());
      return FAILED;
    }
    if (invoke_node.logic_device_list.empty() && !invoke_node.redundant_logic_device_list.empty()) {
      GELOGE(FAILED, "invoke[%s] config error for setting redundant dev without logic device list on node %s",
             invoke_node.invoke_name.c_str(), flow_node_name.c_str());
      return FAILED;
    }
    if (!invoke_node.deploy_info_file.empty() && !invoke_node.logic_device_list.empty()) {
      GELOGE(FAILED, "invoke[%s] config error for setting logic device list and deploy path simultaneously on node %s",
             invoke_node.invoke_name.c_str(), flow_node_name.c_str());
      return FAILED;
    }
    auto &invoke_deploy_device = deploy_logic_device_map[node_invoke_name];
    if (!invoke_node.deploy_info_file.empty()) {
      invoke_deploy_device.first = kDeployInfoFile;
      invoke_deploy_device.second = invoke_node.deploy_info_file;
    } else {
      invoke_deploy_device.first = invoke_node.logic_device_list.empty() ? "" : invoke_node.logic_device_list;
      if (dynamic_schedule_enable) {
        invoke_deploy_device.second = invoke_node.redundant_logic_device_list.empty() ? "" :
                                      invoke_node.redundant_logic_device_list;
        if (!invoke_node.redundant_logic_device_list.empty()) {
          invoke_deploy_device.first.append(",");
          invoke_deploy_device.first.append(invoke_deploy_device.second);
        }
      }
    }
    invoke_deploy_map[flow_node_name].emplace_back(std::make_pair(node_invoke_name, invoke_node.invoke_name));
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::GetDeployLogicDeviceFromBatchInfo(
    std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
    std::set<std::string> &logic_device_ids,
    std::map<std::string, std::vector<std::pair<std::string, std::string>>> &invoke_deploy_map,
    const std::vector<CompileConfigJson::FlowNodeBatchDeployInfo> &batch_deploy_info_list,
    const bool dynamic_schedule_enable) {
  for (const auto &batch_deploy_info : batch_deploy_info_list) {
    for (const auto &flow_node : batch_deploy_info.flow_node_list) {
      if (deploy_logic_device_map.count(flow_node) > 0U) {
        GELOGE(FAILED, "flow_node_name[%s] repeat config.", flow_node.c_str());
        return FAILED;
      }
      auto &deploy_logic_device = deploy_logic_device_map[flow_node];
      deploy_logic_device.first = batch_deploy_info.logic_device_list;
      (void)logic_device_ids.insert(batch_deploy_info.logic_device_list);
      if (dynamic_schedule_enable) {
        deploy_logic_device.second = batch_deploy_info.redundant_logic_device_list;
        (void)logic_device_ids.insert(batch_deploy_info.redundant_logic_device_list);
        // redundant instance will be deployed
        if (!deploy_logic_device.second.empty()) {
          deploy_logic_device.first.append(",");
          deploy_logic_device.first.append(deploy_logic_device.second);
        }
      }
      GE_CHK_STATUS_RET(GetDeployLogicDeviceForInvoke(deploy_logic_device_map, invoke_deploy_map,
                        batch_deploy_info.invoke_deploy_infos, flow_node, dynamic_schedule_enable),
                        "GetDeployLogicDeviceForInvoke failed for node[%s].", flow_node.c_str());
    }
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::GetConfigDeployInfo(
  std::map<std::string, std::pair<std::string, std::string>> &deploy_logic_device_map,
  std::map<std::string, std::pair<uint32_t, uint32_t>> &device_id_to_mem_cfg, bool &dynamic_schedule_enable,
  std::map<std::string, std::vector<std::pair<std::string, std::string>>> &invoke_deploy_map,
  const std::string &deploy_info_str) {
  // deploy_info_str: 1. kDeployInfoFile;file.json 2. logic_device_list;redundant_device_list
  const auto pos = deploy_info_str.find_first_of(';');
  if (deploy_info_str.empty() || (pos == std::string::npos)) {
    GELOGD("Data flow deploy info path does not exist.");
    return SUCCESS;
  }
  std::string deploy_info_file;
  if ((deploy_info_str.substr(0U, pos) != kDeployInfoFile) || (pos == (deploy_info_str.length() - 1))) {
    GELOGD("Data flow deploy info path is not valid");
    return SUCCESS;
  }
  deploy_info_file = deploy_info_str.substr(pos + 1U);

  CompileConfigJson::DeployConfigInfo deploy_config;

  GE_CHK_STATUS_RET(
      CompileConfigJson::ReadDeployInfoFromJsonFile(deploy_info_file, deploy_config),
      "read deploy info json file failed, path=%s", deploy_info_file.c_str());
  GELOGI("read deploy info from file[%s] success.", deploy_info_file.c_str());
  dynamic_schedule_enable = deploy_config.dynamic_schedule_enable;

  std::set<std::string> logic_device_ids;
  // expand and merge config.
  GE_CHK_STATUS_RET(ExpandRangeConfig(deploy_config.batch_deploy_info_list), "expand range config failed.");
  for (const auto &deploy_info : deploy_config.deploy_info_list) {
    if (deploy_logic_device_map.count(deploy_info.flow_node_name) > 0U) {
      GELOGE(FAILED, "flow_node_name[%s] repeat config in file[%s].", deploy_info.flow_node_name.c_str(),
             deploy_info_str.c_str());
      return FAILED;
    }
    deploy_logic_device_map[deploy_info.flow_node_name].first = deploy_info.logic_device_id;
    (void)logic_device_ids.insert(deploy_info.logic_device_id);
  }
  
  GE_CHK_STATUS_RET(GetDeployLogicDeviceFromBatchInfo(deploy_logic_device_map, logic_device_ids, invoke_deploy_map,
                                                      deploy_config.batch_deploy_info_list,
                                                      deploy_config.dynamic_schedule_enable),
                    "Get deploy logic device from batch info failed in file[%s].", deploy_info_str.c_str());
  if (!deploy_config.keep_logic_device_order) {
    for (auto &deploy_logic_device : deploy_logic_device_map) {
      std::string resolved_logic_device_id_list;
      GE_CHK_STATUS_RET(GetSortedLogicDeviceIds(deploy_logic_device.second.first,
                        resolved_logic_device_id_list),
                        "GetSortedLogicDeviceIds failed, logic_device_list[%s]",
                        deploy_logic_device.second.first.c_str());
      deploy_logic_device.second.first = resolved_logic_device_id_list;
    }
  }
  GE_CHK_STATUS_RET(CheckAndProcessMemCfg(deploy_config.mem_size_cfg, logic_device_ids, device_id_to_mem_cfg),
                    "Check and process memory config failed.");
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::CheckAndProcessMemCfg(
    const std::vector<CompileConfigJson::FlowNodeBatchMemCfg> &mem_size_cfg,
    const std::set<std::string> &logic_dev_ids,
    std::map<std::string, std::pair<uint32_t, uint32_t>> &device_id_to_mem_cfg) {
  if (mem_size_cfg.empty()) {
    GELOGD("Memory limit is not set by user config json.");
    return SUCCESS;
  }
  int32_t std_mem_size = 0U;
  int32_t shared_mem_size = 0U;
  for (const auto &mem_cfg : mem_size_cfg) {
    GE_CHK_STATUS_RET(ConvertToInt32(mem_cfg.std_mem_size, std_mem_size),
                      "Convert std memory size %s failed.", mem_cfg.std_mem_size.c_str());
    GE_ASSERT_TRUE(std_mem_size > 0, "std memory[%d] should be set greater than zero", std_mem_size);
    GE_CHK_STATUS_RET(ConvertToInt32(mem_cfg.shared_mem_size, shared_mem_size),
                      "Convert std memory size %s failed.", mem_cfg.shared_mem_size.c_str());
    GE_ASSERT_TRUE(shared_mem_size > 0, "std memory[%d] should be set greater than zero", shared_mem_size);
    std::vector<std::string> expand_logic_device_ids;
    GE_CHK_STATUS_RET(CheckAndExpandLogicDeviceIds(mem_cfg.logic_device_list, expand_logic_device_ids),
                      "Check and expand logic device ids %s failed.", mem_cfg.logic_device_list.c_str());
    GE_CHK_STATUS_RET(SetMemCfgRecord(static_cast<uint32_t>(std_mem_size), static_cast<uint32_t>(shared_mem_size),
                                      expand_logic_device_ids, device_id_to_mem_cfg),
                      "Set memory config failed.");
  }
  // check all logic device have memory config
  for (const auto &orig_logic_device_id : logic_dev_ids) {
    const auto logic_dev_id_list = StringUtils::Split(orig_logic_device_id, ',');
    for (const auto &logic_device_id : logic_dev_id_list) {
      if (device_id_to_mem_cfg.count(logic_device_id) == 0UL) {
        GELOGE(FAILED, "Logic device id %s need memory config.", logic_device_id.c_str());
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::SetMemCfgRecord(const uint32_t &std_mem_size, const uint32_t &shared_mem_size,
    const std::vector<std::string> &expand_logic_device_ids,
    std::map<std::string, std::pair<uint32_t, uint32_t>> &device_id_to_mem_cfg) {
  for (const auto &logic_device_str : expand_logic_device_ids) {
    const auto logic_dev_ids = StringUtils::Split(logic_device_str, ',');
    for (const auto &logic_device : logic_dev_ids) {
      const auto iter = device_id_to_mem_cfg.find(logic_device);
      if ((iter != device_id_to_mem_cfg.cend()) &&
          ((iter->second.first != std_mem_size) || (iter->second.second != shared_mem_size))) {
        GELOGE(FAILED, "Conflict memory value set bo device %s. Please check config json.", logic_device.c_str());
        return FAILED;
      }
      device_id_to_mem_cfg[logic_device] = std::make_pair(std_mem_size, shared_mem_size);
    }
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::ExpandDeployInfoStr(const std::string &deploy_info_str,
                                                      bool is_sub_dataflow,
                                                      std::string &resolved_logic_device_id_list,
                                                      std::string &resolved_redundant_logic_device_id_list,
                                                      bool &is_expand) {
  if (deploy_info_str.empty()) {
    return SUCCESS;
  }

  const auto deploy_infos = StringUtils::Split(deploy_info_str, ';');
  std::string device_id_list;
  std::string redundant_device_id_list;
  if (!is_sub_dataflow && (deploy_infos[0] == kDeployInfoFile)) {
    GELOGE(FAILED, "Get invalid config for setting deploy path[%s] for invoked nn.", deploy_info_str.c_str());
    return FAILED;
  }
  if (deploy_infos[0] != kDeployInfoFile) {
    device_id_list = deploy_infos[0];
    redundant_device_id_list = deploy_infos[1];
  }
  if (!device_id_list.empty()) {
    GE_CHK_STATUS_RET(GetExpandLogicDeviceIds(device_id_list,
                                              resolved_logic_device_id_list),
                      "GetExpandLogicDeviceIds failed, logic_device_list[%s]",
                      device_id_list.c_str());
    is_expand = true;
  }
  if (!redundant_device_id_list.empty()) {
    GE_CHK_STATUS_RET(GetExpandLogicDeviceIds(redundant_device_id_list,
                                              resolved_redundant_logic_device_id_list),
                      "GetExpandLogicDeviceIds failed, redundant_logic_device_list[%s]",
                      redundant_device_id_list.c_str());
    is_expand = true;
  }
  GELOGI("Deploy info str %s logic devices list[%s] to [%s], redundant logic devices list[%s] to [%s]",
         deploy_info_str.c_str(),
         device_id_list.c_str(), resolved_logic_device_id_list.c_str(),
         redundant_device_id_list.c_str(), resolved_redundant_logic_device_id_list.c_str());

  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::UpdateFlowNodeSubGraphDeployInfo(const OpDescPtr &flow_node_op_desc,
                                                                   const DataFlowGraph &graph) {
  GE_CHK_STATUS_RET_NOLOG(UpdateFlowNodeSubGraphDeployInfo(flow_node_op_desc, graph, false));
  GE_CHK_STATUS_RET_NOLOG(UpdateFlowNodeSubGraphDeployInfo(flow_node_op_desc, graph, true));
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::UpdateFlowNodeSubGraphDeployInfo(const OpDescPtr &flow_node_op_desc,
                                                                   const DataFlowGraph &graph,
                                                                   bool is_redundant) {
  std::string logic_device_list;
  std::string redundant_logic_device_list;
  (void)AttrUtils::GetStr(flow_node_op_desc, ATTR_NAME_LOGIC_DEV_ID, logic_device_list);
  (void)AttrUtils::GetStr(flow_node_op_desc, ATTR_NAME_REDUNDANT_LOGIC_DEV_ID, redundant_logic_device_list);

  std::string attr_name = ATTR_NAME_LOGIC_DEV_ID;
  std::string logic_device_id = logic_device_list;
  if (is_redundant) {
    attr_name = ATTR_NAME_REDUNDANT_LOGIC_DEV_ID;
    logic_device_id = redundant_logic_device_list;
  }

  std::map<std::string, std::string> invoke_key_to_subgraph_name;
  std::vector<std::string> subgraph_deploy_infos;
  if (logic_device_id.empty()) {
    GELOGD("flow node[%s] has not set logic device id, no need update, redundant flag[%d].",
      flow_node_op_desc->GetName().c_str(), static_cast<int32_t>(is_redundant));
    return SUCCESS;
  }

  (void)AttrUtils::GetListStr(flow_node_op_desc, ATTR_NAME_DATA_FLOW_INVOKE_DEPLOY_INFOS, subgraph_deploy_infos);
  for (const auto &it : subgraph_deploy_infos) {
    // invoke_name@deploy_info
    const auto pos = it.find_first_of('@');
    if (pos != std::string::npos) {
      invoke_key_to_subgraph_name[it.substr(0U, pos)] = it.substr(pos + 1U);
    }
  }

  const auto &node_loaded_models = graph.GetNodeLoadedModels();
  const auto &find_models = node_loaded_models.find(flow_node_op_desc->GetName());
  if (find_models != node_loaded_models.cend()) {
    for (const auto &flow_model : find_models->second) {
      (void)flow_model->SetLogicDeviceId(logic_device_id);
    }
  }

  const auto &node_subgraphs = graph.GetNodeSubgraphs();
  const auto &find_ret = node_subgraphs.find(flow_node_op_desc->GetName());
  if (find_ret == node_subgraphs.cend()) {
    GELOGE(FAILED, "can not find subgraph of node[%s]", flow_node_op_desc->GetName().c_str());
    return FAILED;
  }
  bool is_host = (logic_device_id.find("-1") != std::string::npos);
  for (const auto &subgraph : find_ret->second) {
    std::string assign_logic_device_id = logic_device_id;
    if (is_host && graph.IsInvokedGraph(subgraph->GetName())) {
      // udf call nn at host, assign nn to first device.
      assign_logic_device_id = "0:0:0:0";
      GELOGI("udf logic device id[%s] is at host, so change subgraph[%s] logic device id to first device[%s], " \
             "redundant flag[%d].",
             logic_device_id.c_str(), subgraph->GetName().c_str(), assign_logic_device_id.c_str(),
             static_cast<int32_t>(is_redundant));
    }
    if (!is_host && graph.IsInvokedGraph(subgraph->GetName())) {
      const auto logic_dev_ids = StringUtils::Split(logic_device_list, ',');
      const auto redundant_logic_dev_ids = StringUtils::Split(redundant_logic_device_list, ',');
      const auto logic_dev_num = redundant_logic_device_list.empty() ? logic_dev_ids.size()
                                 : logic_dev_ids.size() - redundant_logic_dev_ids.size();
      std::string subgraph_deploy_info = "";
      const auto &invoke_key = graph.GetInvokedGraphKey(subgraph->GetName());
      const auto &usr_invoke_key = graph.GetInvokedKeyOriginName(invoke_key);
      auto iter = invoke_key_to_subgraph_name.find(usr_invoke_key);
      if (iter != invoke_key_to_subgraph_name.end()) {
        subgraph_deploy_info = iter->second;
      }
      GE_CHK_STATUS_RET(HandleInvokedSubgraph(subgraph, logic_dev_num, subgraph_deploy_info,
                        is_redundant, assign_logic_device_id), "handle subgraph [%s] deploy info failed for node[%s]",
                        subgraph->GetName().c_str(), flow_node_op_desc->GetName().c_str());
    }
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(subgraph, attr_name, assign_logic_device_id), FAILED,
                           "set attr[%s] to subgraph[%s] failed", attr_name.c_str(),
                           subgraph->GetName().c_str());
    GELOGI("set subgraph[%s] logic device id[%s], redundant flag[%d].",
           subgraph->GetName().c_str(), assign_logic_device_id.c_str(), static_cast<int32_t>(is_redundant));
  }
  return SUCCESS;
}

Status DataFlowGraphAutoDeployer::HandleInvokedSubgraph(const ComputeGraphPtr &subgraph, const size_t logic_dev_num,
                                                        const std::string &subgraph_deploy_info, bool is_redundant,
                                                        std::string &assign_logic_device_id) {
  bool is_data_flow_graph = false;
  (void)AttrUtils::GetBool(subgraph, dflow::ATTR_NAME_IS_DATA_FLOW_GRAPH, is_data_flow_graph);
  if (is_data_flow_graph && (logic_dev_num > 1)) {
    GELOGE(FAILED, "Subgraph[%s]'s parent node could not config multi instances", subgraph->GetName().c_str());
    return FAILED;
  }

  std::string resolved_logic_device_id_list;
  std::string resolved_redundant_logic_device_id_list;
  bool is_expand = false;
  GE_CHK_STATUS_RET(ExpandDeployInfoStr(subgraph_deploy_info, is_data_flow_graph, resolved_logic_device_id_list,
                    resolved_redundant_logic_device_id_list, is_expand),
                    "Get subgraph[%s] config deploy info failed.", subgraph->GetName().c_str());
  if (is_data_flow_graph) {
    GE_CHK_BOOL_RET_STATUS(AttrUtils::SetStr(subgraph, ATTR_NAME_DATA_FLOW_SUB_DATA_FLOW_DEPLOY_INFOS,
                           subgraph_deploy_info), FAILED,
                           "set attr[%s] to subgraph[%s] failed", ATTR_NAME_DATA_FLOW_SUB_DATA_FLOW_DEPLOY_INFOS,
                           subgraph->GetName().c_str());
    GELOGI("set subgraph[%s] attr[%s] value[%s] success.", subgraph->GetName().c_str(),
           ATTR_NAME_DATA_FLOW_SUB_DATA_FLOW_DEPLOY_INFOS, subgraph_deploy_info.c_str());
  }
  
  if (is_expand && !is_redundant) {
    assign_logic_device_id = resolved_logic_device_id_list;
    const auto nn_logic_dev_ids = StringUtils::Split(resolved_logic_device_id_list, ',');
    const auto nn_redundant_logic_dev_ids = StringUtils::Split(resolved_redundant_logic_device_id_list, ',');
    const auto nn_logic_dev_num = resolved_redundant_logic_device_id_list.empty() ? nn_logic_dev_ids.size()
                                  : nn_logic_dev_ids.size() - nn_redundant_logic_dev_ids.size();
    if (!is_data_flow_graph && (logic_dev_num != nn_logic_dev_num)) {
      GELOGE(FAILED, "Config nn[%s]'s instance num is not equal to parent flow node's instance num",
             subgraph->GetName().c_str());
      return FAILED;
    }
  } else if (is_expand && is_redundant) {
    assign_logic_device_id = resolved_redundant_logic_device_id_list;
  }
  return SUCCESS;
}
}  // namespace ge
