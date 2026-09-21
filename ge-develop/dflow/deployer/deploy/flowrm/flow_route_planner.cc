/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/flowrm/flow_route_planner.h"
#include "common/checker.h"
#include "graph/ge_context.h"
#include "deploy/flowrm/network_manager.h"
#include "deploy/resource/resource_manager.h"
#include "dflow/flow_graph/data_flow_attr_define.h"

namespace ge {
namespace {
bool HasNMappingNode(const FlowModelPtr &flow_model) {
  GE_ASSERT_NOTNULL(flow_model);
  bool contains_n_mapping_node = false;
  constexpr const char *kAttrNameDataFlowContainNMappingNode = "_contains_n-mapping_node";
  (void)AttrUtils::GetBool(flow_model->GetRootGraph(), kAttrNameDataFlowContainNMappingNode, contains_n_mapping_node);
  GELOGI("flow model[%s] attr[%s] value is %d.", flow_model->GetModelName().c_str(),
         kAttrNameDataFlowContainNMappingNode, static_cast<int32_t>(contains_n_mapping_node));
  return contains_n_mapping_node;
}
}  // namespace

Status FlowRoutePlanner::ResolveFlowRoutePlans(DeployState &deploy_state) {
  if (deploy_state.GetLocalFlowRoutePlan().first > 0) {
    GELOGI("flow route plan already resolved");
    return SUCCESS;
  }
  const auto &deploy_plan = deploy_state.GetDeployPlan();
  std::set<int32_t> unique_node_ids;
  // add head device for modelIO
  DeployPlan::DeviceInfo head_device = deploy_plan.GetRootModelQueueDeviceInfo();
  unique_node_ids.emplace(head_device.GetNodeId());
  for (const auto &it : deploy_plan.GetSubmodels()) {
    const auto &target_device = it.second.queue_device_info;
    (void) unique_node_ids.emplace(target_device.GetNodeId());
  }
  PlanAttrs plan_attrs;
  plan_attrs.root_model_id = deploy_state.GetRootModelId();
  plan_attrs.is_dynamic_sched = deploy_state.GetIsDynamicSched();
  // when has n-mapping or enable exception catch flow gw no need keep order.
  plan_attrs.keep_out_of_order = HasNMappingNode(deploy_state.GetFlowModel()) || deploy_state.IsEnableExceptionCatch();
  for (const auto &node_id : unique_node_ids) {
    deployer::FlowRoutePlan route_plan;
    GE_CHK_STATUS_RET_NOLOG(ResolveFlowRoutePlan(deploy_plan, node_id, route_plan, plan_attrs));
    deploy_state.AddFlowRoutePlanToDeploy(node_id, route_plan);
  }
  return SUCCESS;
}

Status FlowRoutePlanner::ResolveFlowRoutePlan(const DeployPlan &deploy_plan,
                                              int32_t node_id,
                                              deployer::FlowRoutePlan &flow_route_plan,
                                              const PlanAttrs &plan_attrs) {
  GELOGD("ResolveFlowRoutePlan start, target node id = %d", node_id);
  GE_CHK_STATUS_RET_NOLOG(ResolveEndpoints(deploy_plan, node_id, flow_route_plan, plan_attrs));
  GE_CHK_STATUS_RET_NOLOG(
      ResolveBindings(deploy_plan, node_id, flow_route_plan, plan_attrs.root_model_id));
  PrintFlowRoutePlan(flow_route_plan);
  return SUCCESS;
}

void FlowRoutePlanner::SetEndpointType(const DeployPlan::QueueInfo &endpoint_info,
                                       deployer::EndpointDesc &endpoint_desc) {
  if (endpoint_info.ref_index >= 0) {
    endpoint_desc.set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeRefQueue));
    endpoint_desc.mutable_queue_desc()->set_ref_index(endpoint_info.ref_index);
  } else if (endpoint_info.is_dummy) {
    endpoint_desc.set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeDummyQueue));
  } else {
    endpoint_desc.set_type(endpoint_info.owned ?
                           static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue) :
                           static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeExternalQueue));
  }
}

Status FlowRoutePlanner::ResolveEndpoints(const DeployPlan &deploy_plan,
                                          int32_t node_id,
                                          deployer::FlowRoutePlan &flow_route_plan,
                                          const PlanAttrs &plan_attrs) {
  for (size_t i = 0U; i < deploy_plan.GetQueueInfoList().size(); ++i) {
    auto endpoint_index = static_cast<int32_t>(i);
    auto &endpoint_info = deploy_plan.GetQueueInfoList()[i];
    const auto &device_info = endpoint_info.device_info;
    auto endpoint_desc = flow_route_plan.add_endpoints();
    endpoint_desc->set_name(endpoint_info.name);
    endpoint_desc->set_node_id(device_info.GetNodeId());
    endpoint_desc->set_device_id(device_info.GetDeviceId());
    endpoint_desc->set_device_type(device_info.GetType());
    endpoint_desc->set_model_id(endpoint_info.model_id);
    endpoint_desc->set_is_dynamic_sched(plan_attrs.is_dynamic_sched);
    endpoint_desc->set_root_model_id(plan_attrs.root_model_id);
    if (device_info.GetNodeId() == node_id) {  // endpoint(queue/group) in target node
      if (deploy_plan.IsGroupEndpoint(endpoint_index)) {
        GE_CHK_STATUS_RET_NOLOG(
            FillGroupEndpoint(deploy_plan, endpoint_index, plan_attrs.keep_out_of_order, *endpoint_desc));
        const auto &group_desc = endpoint_desc->group_desc();
        GELOGD("[Group] added ,index = %d, name = %s, elements = %s",
               endpoint_index,
               endpoint_desc->name().c_str(),
               ToString(std::vector<int32_t>(group_desc.endpoint_indices().begin(),
                                             group_desc.endpoint_indices().end())).c_str());
      } else {
        SetEndpointType(endpoint_info, *endpoint_desc);
        endpoint_desc->mutable_queue_desc()->set_depth(endpoint_info.depth);
        endpoint_desc->mutable_queue_desc()->set_enqueue_policy(endpoint_info.enqueue_policy);
        endpoint_desc->mutable_queue_desc()->set_name(endpoint_info.name);
        endpoint_desc->mutable_queue_desc()->set_fusion_offset(endpoint_info.fusion_offset);
        GELOGD("[Queue] added, index = %d, name = %s, device = %s, type = %d, ref index = %d, fusion offset = %d",
               endpoint_index, endpoint_desc->name().c_str(), device_info.GetDesc().c_str(),
               endpoint_desc->type(), endpoint_info.ref_index, endpoint_info.fusion_offset);
      }
    } else {
      endpoint_desc->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeNone));
    }
  }
  return SUCCESS;
}

Status FlowRoutePlanner::FillGroupEndpoint(const DeployPlan &deploy_plan,
                                           int32_t group_index, bool keep_out_of_order,
                                           deployer::EndpointDesc &endpoint_desc) {
  endpoint_desc.set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeGroup));
  const DeployPlan::QueueInfo *group_queue_info = nullptr;
  GE_CHK_STATUS_RET_NOLOG(deploy_plan.GetQueueInfo(group_index, group_queue_info));
  endpoint_desc.set_instance_num(group_queue_info->instance_num);
  endpoint_desc.set_instance_idx(group_queue_info->instance_idx);
  auto queue_list_iter = deploy_plan.GetGroups().find(group_index);
  GE_CHK_BOOL_RET_STATUS(queue_list_iter != deploy_plan.GetGroups().end(),
                         FAILED, "Get group[%d] info failed.", group_index);
  auto group_entries_index_start = static_cast<int32_t>(deploy_plan.GetQueueInfoList().size());
  for (auto entry_index : queue_list_iter->second) {
    auto &entry_info = deploy_plan.GetGroupEntryInfoList()[entry_index];
    if (entry_info.ref_index >= 0) {
      endpoint_desc.mutable_group_desc()->add_endpoint_indices(entry_info.ref_index);
    } else {
      endpoint_desc.mutable_group_desc()->add_endpoint_indices(group_entries_index_start + entry_index);
    }
  }
  endpoint_desc.mutable_group_desc()->set_keep_out_of_order(keep_out_of_order);
  GELOGD("group[%s] keep out of order value=%d", endpoint_desc.name().c_str(), static_cast<int32_t>(keep_out_of_order));
  return SUCCESS;
}

Status FlowRoutePlanner::ResolveBindings(const DeployPlan &deploy_plan,
                                         int32_t node_id,
                                         deployer::FlowRoutePlan &flow_route_plan,
                                         const uint32_t root_model_id) {
  std::map<std::string, std::set<size_t>> device_2_tag_indices;
  std::vector<DeployPlan::DeviceInfo> devices;
  GE_CHK_STATUS_RET(AddFlowRoutePlanBindings(deploy_plan, node_id, devices, flow_route_plan, device_2_tag_indices),
                    "Failed to add flow route bindings.");
  for (size_t i = 0U; i < deploy_plan.GetGroupEntryInfoList().size(); ++i) {
    const auto &entry_info = deploy_plan.GetGroupEntryInfoList()[i];
    const auto &device_info = entry_info.device_info;
    auto endpoint_desc = flow_route_plan.add_endpoints();
    bool is_valid_tag = false;
    for (const auto &target_device_info : devices) {
      // tag index is recount from 0, need to identify accoring to ref_index, target device is group device
      const auto &relative_tag_indices = device_2_tag_indices[target_device_info.GetKey()];
      if ((entry_info.ref_index < 0) && (device_info.GetKey() != target_device_info.GetKey()) &&
          (relative_tag_indices.count(i) > 0U)) {
        uint32_t target_device_rank_id = UINT32_MAX;
        GE_CHK_STATUS_RET(GetDeviceRankId(target_device_info, target_device_rank_id),
                          "Failed to get target device rank id, entry_index[%zu], name[%s].",
                          i, entry_info.name.c_str());
        endpoint_desc->set_name(entry_info.name);
        endpoint_desc->set_node_id(target_device_info.GetNodeId());
        endpoint_desc->set_device_id(target_device_info.GetHcomDeviceId());
        endpoint_desc->set_device_type(target_device_info.GetType());
        endpoint_desc->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag));
        endpoint_desc->set_root_model_id(root_model_id);
        auto tag_desc = endpoint_desc->mutable_tag_desc();
        GE_CHK_STATUS_RET(AssignTagInfo(device_info, *tag_desc, entry_info.name, i));

        tag_desc->set_rank_id(target_device_rank_id);
        tag_desc->set_name(entry_info.name);
        tag_desc->set_depth(entry_info.depth);
        tag_desc->set_peer_node_id(device_info.GetNodeId());
        tag_desc->set_peer_device_id(device_info.GetHcomDeviceId());
        tag_desc->set_peer_device_type(device_info.GetType());
        GELOGI("[Tag] added, entry_index[%zu], name[%s], device[%s], rank_id[%u], peer_rank_id[%u], "
               "tag_id[%u], hcom_device_id[%d], target device_type[%d]",
               i, tag_desc->name().c_str(), device_info.GetDesc().c_str(), tag_desc->rank_id(),
               tag_desc->peer_rank_id(), tag_desc->tag_id(),
               target_device_info.GetHcomDeviceId(), target_device_info.GetType());
        is_valid_tag = true;
      }
    }
    if (!is_valid_tag) {
      endpoint_desc->set_type(static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeNone));
    }
  }
  return SUCCESS;
}

Status FlowRoutePlanner::AddFlowRoutePlanBindings(const DeployPlan &deploy_plan,
                                                  int32_t node_id,
                                                  std::vector<DeployPlan::DeviceInfo> &devices,
                                                  deployer::FlowRoutePlan &flow_route_plan,
                                                  std::map<std::string, std::set<size_t>> &relative_tag_indices) {
  for (auto &binding : deploy_plan.GetQueueBindings()) {
    auto src_q_idx = binding.first;
    auto dst_q_idx = binding.second;
    const DeployPlan::QueueInfo *src_queue_info = nullptr;
    const DeployPlan::QueueInfo *dst_queue_info = nullptr;
    GE_CHK_STATUS_RET_NOLOG(deploy_plan.GetQueueInfo(src_q_idx, src_queue_info));
    GE_CHK_STATUS_RET_NOLOG(deploy_plan.GetQueueInfo(dst_q_idx, dst_queue_info));
    // collect bindings that related to target device
    if (src_queue_info->device_info.GetNodeId() == node_id || dst_queue_info->device_info.GetNodeId() == node_id) {
      GE_CHK_STATUS_RET(AddGroupEntries(deploy_plan,
                                        src_q_idx,
                                        devices,
                                        relative_tag_indices),
                        "Failed to add group entries.");
      GE_CHK_STATUS_RET(AddGroupEntries(deploy_plan,
                                        dst_q_idx,
                                        devices,
                                        relative_tag_indices),
                        "Failed to add group entries.");
      GELOGD("src index = %d, dst index = %d, src = %s@%s, dst = %s@%s, src is group = %d, dst is group = %d",
             src_q_idx,
             dst_q_idx,
             src_queue_info->name.c_str(),
             src_queue_info->device_info.GetDesc().c_str(),
             dst_queue_info->name.c_str(),
             dst_queue_info->device_info.GetDesc().c_str(),
             deploy_plan.IsGroupEndpoint(src_q_idx),
             deploy_plan.IsGroupEndpoint(dst_q_idx));
      auto new_binding = src_queue_info->owned ?
                         flow_route_plan.add_bindings_before_load() :
                         flow_route_plan.add_bindings();
      GE_CHECK_NOTNULL(new_binding);
      new_binding->set_src_index(src_q_idx);
      new_binding->set_dst_index(dst_q_idx);
    }
  }
  return SUCCESS;
}

Status FlowRoutePlanner::AddGroupEntries(const DeployPlan &deploy_plan,
                                         int32_t index,
                                         std::vector<DeployPlan::DeviceInfo> &devices,
                                         std::map<std::string, std::set<size_t>> &relative_tag_indices) {
  if (deploy_plan.IsGroupEndpoint(index)) {
    const DeployPlan::QueueInfo *queue_info = nullptr;
    GE_CHK_STATUS_RET_NOLOG(deploy_plan.GetQueueInfo(index, queue_info));
    const auto &device_info = queue_info->device_info;
    const auto &entries = deploy_plan.GetGroups().at(index);
    GELOGI("Add group entries, group index = %d, entries = %s", index, ToString(entries).c_str());
    const auto &it = relative_tag_indices.find(device_info.GetKey());
    if (it == relative_tag_indices.cend()) {
      devices.emplace_back(device_info);
    }
    relative_tag_indices[device_info.GetKey()].insert(entries.begin(), entries.end());
  }
  return SUCCESS;
}

Status FlowRoutePlanner::GetDeviceHcomInfo(const DeployPlan::DeviceInfo &device_info, uint32_t &rank_id) {
  auto dev_info = ResourceManager::GetInstance().GetDeviceInfo(device_info.GetNodeId(),
                                                               device_info.GetDeviceId(), device_info.GetType());
  GE_CHECK_NOTNULL(dev_info);
  const auto &ip = dev_info->GetDeviceIp();
  auto device_id = dev_info->GetHcomDeviceId();
  GELOGI("Get remote device info success, ip = %s, device_id = %d", ip.c_str(), device_id);
  int32_t tmp_rank_id = 0;
  auto ret = DeployContext::LocalContext().GetRankTableBuilder().GetRankIdByDeviceId(ip, device_id, tmp_rank_id);
  GE_CHK_BOOL_RET_STATUS(ret, FAILED, "Get rank id info failed, ip is %s, devid = %d.",
                         ip.c_str(), device_info.GetHcomDeviceId());
  rank_id = static_cast<uint32_t>(tmp_rank_id);
  return SUCCESS;
}

Status FlowRoutePlanner::GetDeviceRankId(const DeployPlan::DeviceInfo &device_info, uint32_t &rank_id) {
  return GetDeviceHcomInfo(device_info, rank_id);
}

uint32_t FlowRoutePlanner::GetTagIdByName(const std::string &tag_name) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  static uint32_t index = 0;
  static std::unordered_map<std::string, uint32_t> tag_name_map;
  if (tag_name_map.find(tag_name) != tag_name_map.end()) {
    return tag_name_map[tag_name];
  }
  index++;
  tag_name_map[tag_name] = index;
  GELOGI("GetTagIdByName tag_name is %s, tag_id is %d.", tag_name.c_str(), index);
  return index;
}

void FlowRoutePlanner::PrintFlowRoutePlan(const deployer::FlowRoutePlan &flow_route_plan) {
  if (!IsLogEnable(GE_MODULE_NAME, DLOG_DEBUG)) {
    return;
  }
  std::set<int32_t> printed;
  for (const auto &binding : flow_route_plan.bindings()) {
    printed.emplace(binding.src_index());
    printed.emplace(binding.dst_index());
    auto &src_endpoint = flow_route_plan.endpoints(binding.src_index());
    auto &dst_endpoint = flow_route_plan.endpoints(binding.dst_index());
    GELOGD("[Binding] src endpoint index = %d, type = %d, dst endpoint index = %d, type = %d",
           binding.src_index(), src_endpoint.type(),
           binding.dst_index(), dst_endpoint.type());
    GELOGD("Src endpoint: ");
    PrintEndpointDesc(flow_route_plan, src_endpoint, printed);
    GELOGD("Dst endpoint: ");
    PrintEndpointDesc(flow_route_plan, dst_endpoint, printed);
  }
  GELOGD("Unbound endpoints: ");
  for (int32_t i = 0; i < flow_route_plan.endpoints_size(); ++i) {
    if (printed.count(i) > 0) {
      continue;
    }
    PrintEndpointDesc(flow_route_plan, flow_route_plan.endpoints(i), printed);
  }
}

void FlowRoutePlanner::PrintEndpointDesc(const deployer::FlowRoutePlan &flow_route_plan,
                                         const deployer::EndpointDesc &endpoint_desc,
                                         std::set<int32_t> &printed) {
  if (endpoint_desc.type() == static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeExternalQueue)) {
    GELOGD("  [ExternalQueue] name = %s", endpoint_desc.name().c_str());
  } else if (endpoint_desc.type() == static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue)) {
    GELOGD("  [Queue] name = %s", endpoint_desc.name().c_str());
  } else if (endpoint_desc.type() == static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeRefQueue)) {
    GELOGD("  [RefQueue] name = %s, ref index = %d",
           endpoint_desc.name().c_str(), endpoint_desc.queue_desc().ref_index());
  } else if (endpoint_desc.type() == static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeGroup)) {
    GELOGD("  [Group] name = %s, element number = %d",
           endpoint_desc.name().c_str(),
           endpoint_desc.group_desc().endpoint_indices_size());
    for (auto idx : endpoint_desc.group_desc().endpoint_indices()) {
      printed.emplace(idx);
      auto &group_entry = flow_route_plan.endpoints(idx);
      if (group_entry.type() == static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeTag)) {
        auto &tag_desc = group_entry.tag_desc();
        GELOGD("    [GroupEntry] index = %d, tag name = %s, tag_id = %u, rank_id = %u, peer_rank_id = %u",
               idx,
               group_entry.name().c_str(),
               tag_desc.tag_id(),
               tag_desc.rank_id(),
               tag_desc.peer_rank_id());
      } else if (group_entry.type() == static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeQueue)) {
        GELOGD("    [GroupEntry] index = %d, queue name = %s", idx, endpoint_desc.name().c_str());
      } else {
        GELOGW("    [GroupEntry] Unexpected group entry, index = %d, name = %s, type = %d",
               idx,
               endpoint_desc.name().c_str(),
               group_entry.type());
      }
    }
  }
}

Status FlowRoutePlanner::AssignTagInfo(const DeployPlan::DeviceInfo &device_info, deployer::TagDesc &tag_desc,
                                       const std::string &tag_name, uint32_t index) {
  uint32_t rank_id = UINT32_MAX;
  GE_CHK_STATUS_RET(GetDeviceRankId(device_info, rank_id),
                    "Failed to get device rank id, entry_index[%zu], name[%s].",
                    index, tag_name.c_str());
  tag_desc.set_tag_id(GetTagIdByName(tag_name));
  tag_desc.set_peer_rank_id(rank_id);
  return SUCCESS;
}
}  // namespace ge
