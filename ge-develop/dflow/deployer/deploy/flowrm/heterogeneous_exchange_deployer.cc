/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "deploy/flowrm/heterogeneous_exchange_deployer.h"
#include "common/data_flow/queue/heterogeneous_exchange_service.h"
#include "common/utils/rts_api_utils.h"
#include "framework/common/util.h"
#include "securec.h"
#include "dflow/base/deploy/exchange_service.h"

namespace ge {
Status ExchangeRoute::GetQueueId(int32_t queue_index, uint32_t &queue_id) const {
  auto endpoint = MutableEndpoint(queue_index);
  GE_CHECK_NOTNULL(endpoint);
  GE_CHK_BOOL_RET_STATUS((endpoint->type == ExchangeEndpointType::kEndpointTypeQueue ||
                          endpoint->type == ExchangeEndpointType::kEndpointTypeRefQueue ||
                          endpoint->type == ExchangeEndpointType::kEndpointTypeDummyQueue),
                         FAILED,
                         "The type[%d] of endpoint index[%d] is not queue.",
                         static_cast<int32_t>(endpoint->type), queue_index);
  queue_id = endpoint->id;
  return SUCCESS;
}

Status ExchangeRoute::GetFusionOffset(int32_t queue_index, int32_t &fusion_offset) const {
  auto endpoint = MutableEndpoint(queue_index);
  GE_CHECK_NOTNULL(endpoint);
  fusion_offset = endpoint->fusion_offset;
  return SUCCESS;
}

Status ExchangeRoute::GetQueueIds(const std::vector<int32_t> &queue_indices, std::vector<uint32_t> &queue_ids) const {
  for (auto queue_index : queue_indices) {
    uint32_t queue_id = UINT32_MAX;
    GE_CHK_STATUS_RET_NOLOG(GetQueueId(queue_index, queue_id));
    queue_ids.emplace_back(queue_id);
  }
  return SUCCESS;
}

void ExchangeRoute::GetQueueIds(std::vector<uint32_t> &queue_ids) const {
  for (const auto &endpoint_it : endpoints_) {
    if (endpoint_it.second->type == ExchangeEndpointType::kEndpointTypeQueue ||
        endpoint_it.second->type == ExchangeEndpointType::kEndpointTypeRefQueue) {
      queue_ids.emplace_back(endpoint_it.second->id);
    }
  }
}

bool ExchangeRoute::IsProxyQueue(int32_t queue_index) const {
  auto endpoint = MutableEndpoint(queue_index);
  if (endpoint != nullptr) {
    return (endpoint->type == ExchangeEndpointType::kEndpointTypeQueue) &&
           (endpoint->device_type != static_cast<int32_t>(CPU));
  }
  return false;
}

Status ExchangeRoute::GetQueueAttr(int32_t queue_index, DeployQueueAttr &queue_attr) const {
  auto endpoint = MutableEndpoint(queue_index);
  GE_CHECK_NOTNULL(endpoint);
  GE_CHK_BOOL_RET_STATUS((endpoint->type == ExchangeEndpointType::kEndpointTypeQueue ||
                          endpoint->type == ExchangeEndpointType::kEndpointTypeRefQueue ||
                          endpoint->type == ExchangeEndpointType::kEndpointTypeDummyQueue),
                         FAILED,
                         "The type[%d] of endpoint index[%d] is not queue.",
                         static_cast<int32_t>(endpoint->type), queue_index);
  queue_attr.queue_id = endpoint->id;
  queue_attr.device_id = endpoint->device_id;
  queue_attr.device_type = endpoint->device_type;
  queue_attr.global_logic_id = queue_index;
  return SUCCESS;
}

Status ExchangeRoute::GetQueueAttrs(const std::vector<int32_t> &queue_indices,
                                    std::vector<DeployQueueAttr> &queue_attrs) const {
  for (auto queue_index : queue_indices) {
    DeployQueueAttr queue_attr = {};
    GE_CHK_STATUS_RET(GetQueueAttr(queue_index, queue_attr), "Failed to get queue_attr.");
    queue_attrs.emplace_back(queue_attr);
  }
  return SUCCESS;
}

void ExchangeRoute::GetQueueAttrs(std::vector<DeployQueueAttr> &queue_attrs) const {
  for (const auto &endpoint_it : endpoints_) {
    if (endpoint_it.second->type == ExchangeEndpointType::kEndpointTypeQueue ||
        endpoint_it.second->type == ExchangeEndpointType::kEndpointTypeRefQueue) {
      DeployQueueAttr queue_attr = {};
      queue_attr.queue_id = endpoint_it.second->id;
      queue_attr.device_id = endpoint_it.second->device_id;
      queue_attr.device_type = endpoint_it.second->device_type;
      queue_attrs.emplace_back(queue_attr);
    }
  }
}

Status ExchangeRoute::GetFusionOffsets(const std::vector<int32_t> &queue_indices,
                                       std::vector<int32_t> &fusion_offsets) const {
  for (auto queue_index : queue_indices) {
    int32_t fusion_offset = 0;
    GE_CHK_STATUS_RET_NOLOG(GetFusionOffset(queue_index, fusion_offset));
    fusion_offsets.emplace_back(fusion_offset);
  }
  return SUCCESS;
}

ExchangeEndpoint *ExchangeRoute::MutableEndpoint(int32_t index, int32_t expect_type) const {
  return MutableEndpoint(index, static_cast<ExchangeEndpointType>(expect_type));
}

ExchangeEndpoint *ExchangeRoute::MutableEndpoint(int32_t index, ExchangeEndpointType expect_type) const {
  auto it = endpoints_.find(index);
  if (it == endpoints_.end()) {
    GELOGE(PARAM_INVALID, "Failed to get endpoint by index: %d", index);
    return nullptr;
  }

  if (it->second->type != expect_type) {
    GELOGE(PARAM_INVALID,
           "Endpoint type mismatches, index = %d, expect = %d, but = %d",
           index,
           static_cast<int32_t>(expect_type),
           static_cast<int32_t>(it->second->type));
    return nullptr;
  }
  return it->second.get();
}

ExchangeEndpoint *ExchangeRoute::MutableEndpoint(int32_t index) const {
  auto it = endpoints_.find(index);
  if (it == endpoints_.end()) {
    return nullptr;
  }
  return it->second.get();
}

HeterogeneousExchangeDeployer::HeterogeneousExchangeDeployer(ExchangeService &exchange_service,
                                                             deployer::FlowRoutePlan route_plan,
                                                             FlowGwClientManager &client_manager)
    : exchange_service_(exchange_service),
      route_plan_(std::move(route_plan)),
      client_manager_(client_manager) {
}

HeterogeneousExchangeDeployer::~HeterogeneousExchangeDeployer() {
  (void) Undeploy(exchange_service_, deploying_, client_manager_);
}

Status HeterogeneousExchangeDeployer::PreDeploy() {
  if (!pre_deployed_) {
    GE_CHK_STATUS_RET(CreateHcomHandles(), "Failed to create hcom handles");
    GE_CHK_STATUS_RET(CreateExchangeEndpoints(), "Failed to create endpoints");
    GE_CHK_STATUS_RET(BindEndpoints(GetBindingsBeforeLoad()), "Failed to create endpoints");
    pre_deployed_ = true;
  }
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::Deploy(ExchangeRoute &deployed, bool wait_config_effect) {
  GE_CHK_STATUS_RET_NOLOG(PreDeploy());
  GE_CHK_STATUS_RET_NOLOG(BindEndpoints(GetBindingsAfterLoad()));
  if (wait_config_effect) {
    GE_CHK_STATUS_RET(client_manager_.WaitAllClientConfigEffect(), "Failed to wait config effect");
  }
  deployed = std::move(deploying_);
  GELOGI("Build exchange route successfully.");
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::GetSrcEndpointIndices(std::set<int32_t> &src_endpoint_indices) {
  for (const auto &binding : GetAllBindings()) {
    auto src_index = binding.src_index();
    auto endpoint_desc = GetEndpointDesc(src_index);
    GE_CHECK_NOTNULL(endpoint_desc);
    if (endpoint_desc->type() == static_cast<int32_t>(ExchangeEndpointType::kEndpointTypeGroup)) {
      auto &group_desc = endpoint_desc->group_desc();
      for (int32_t j = 0; j < group_desc.endpoint_indices_size(); ++j) {
        const auto endpoint_index = group_desc.endpoint_indices(j);
        src_endpoint_indices.emplace(endpoint_index);
        GELOGI("Add group element to src indices, group index = %d, element index = %d.", src_index, endpoint_index);
      }
    }
    src_endpoint_indices.emplace(src_index);
  }
  return SUCCESS;
}

void HeterogeneousExchangeDeployer::UpdateCommonInfo(ExchangeEndpoint &endpoint,
                                                     const deployer::EndpointDesc &endpoint_desc) {
  endpoint.type = static_cast<ExchangeEndpointType>(endpoint_desc.type());
  endpoint.name = endpoint_desc.name();
  endpoint.node_id = endpoint_desc.node_id();
  endpoint.device_id = endpoint_desc.device_id();
  endpoint.device_type = endpoint_desc.device_type();
  endpoint.model_id = endpoint_desc.model_id();
  endpoint.is_dynamic_sched = endpoint_desc.is_dynamic_sched();
  endpoint.root_model_id = endpoint_desc.root_model_id();
}

Status HeterogeneousExchangeDeployer::CreateExchangeEndpoints() {
  std::set<int32_t> src_endpoint_indices;
  GE_CHK_STATUS_RET(GetSrcEndpointIndices(src_endpoint_indices), "Failed to get src indices.");
  std::vector<int32_t> group_indices;
  for (int32_t i = 0; i < route_plan_.endpoints_size(); ++i) {
    const auto &endpoint_desc = route_plan_.endpoints(i);
    auto endpoint_ptr = MakeShared<ExchangeEndpoint>();
    GE_CHECK_NOTNULL(endpoint_ptr);
    auto &endpoint = *endpoint_ptr;
    endpoint.index = i;
    UpdateCommonInfo(endpoint, endpoint_desc);
    if (endpoint.type == ExchangeEndpointType::kEndpointTypeQueue) {
      uint32_t work_mode = RT_MQ_MODE_PULL;
      if (src_endpoint_indices.find(i) != src_endpoint_indices.end()) {
        GELOGD("Queue[%s] used as a binding source, set work mode to PUSH", endpoint.name.c_str());
        work_mode = RT_MQ_MODE_PUSH;
      }
      GE_CHK_STATUS_RET(DoCreateQueue(endpoint_desc.queue_desc(), work_mode, endpoint));
    } else if (endpoint.type == ExchangeEndpointType::kEndpointTypeExternalQueue) {
      GE_CHK_STATUS_RET(RtsApiUtils::MemQueryGetQidByName(endpoint_desc.device_id(), endpoint.name, endpoint.id),
                        "Failed to get queue_id by name: %s", endpoint.name.c_str());
      GELOGD("External queue, queue_name = %s, queue_id = %u", endpoint.name.c_str(), endpoint.id);
    }  else if (endpoint.type == ExchangeEndpointType::kEndpointTypeRefQueue) {
      auto ref_index = endpoint_desc.queue_desc().ref_index();
      GE_CHK_BOOL_RET_STATUS((ref_index >= 0 && ref_index < static_cast<int32_t>(deploying_.endpoints_.size())),
                             FAILED,
                             "RefQueue index[%d] is invalid, must in range[0, %zu)",
                             ref_index, deploying_.endpoints_.size());
      endpoint.id = deploying_.endpoints_[ref_index]->id;
      endpoint.fusion_offset = endpoint_desc.queue_desc().fusion_offset();
      GELOGD("Ref queue, queue_name = %s, queue_id = %u", endpoint.name.c_str(), endpoint.id);
    } else if (endpoint.type == ExchangeEndpointType::kEndpointTypeTag) {
      GE_CHK_STATUS_RET(DoCreateTag(endpoint_desc.tag_desc(), endpoint));
    } else if (endpoint.type == ExchangeEndpointType::kEndpointTypeGroup) {
      group_indices.emplace_back(i);
      endpoint.instance_num = endpoint_desc.instance_num();
      endpoint.instance_idx = endpoint_desc.instance_idx();
      // only src group need check keep_out_of_order
      if ((!endpoint.is_dynamic_sched) &&
          (src_endpoint_indices.find(i) != src_endpoint_indices.end())) {
        // src group is_dynamic_sched means whether keep out of order.
        endpoint.is_dynamic_sched = endpoint_desc.group_desc().keep_out_of_order();
        GELOGD("update src group is_dynamic_sched to keep out of order, name=%s, value=%d",
               endpoint.name.c_str(), static_cast<int32_t>(endpoint.is_dynamic_sched));
      }
    } else if (endpoint.type == ExchangeEndpointType::kEndpointTypeDummyQueue) {
      endpoint.id = UINT32_MAX;
    } else {
      // skip
    }
    deploying_.endpoints_.emplace(i, endpoint_ptr);
    GELOGI("Create endpoint success, device[%d], name[%s], index[%d], endpoint_id[%u], type[%d], fusion_offset[%d]",
           endpoint.device_id, endpoint.name.c_str(), i, endpoint.id, endpoint.type, endpoint.fusion_offset);
  }
  GE_CHK_STATUS_RET(CreateGroups(group_indices), "Failed to create groups.");
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::CreateGroups(const std::vector<int32_t> &group_indices) {
  for (auto index : group_indices) {
    auto endpoint_desc = GetEndpointDesc(index);
    GE_CHECK_NOTNULL(endpoint_desc);
    auto endpoint = deploying_.MutableEndpoint(index, ExchangeEndpointType::kEndpointTypeGroup);
    GE_CHECK_NOTNULL(endpoint);
    GE_CHK_STATUS_RET(CreateGroupTags(index, *endpoint_desc, *endpoint), "Create group failed.");
  }
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::DoCreateQueue(const deployer::QueueDesc &queue_desc,
                                                    uint32_t work_mode,
                                                    ExchangeEndpoint &endpoint) const {
  MemQueueAttr mem_queue_attr{};
  mem_queue_attr.work_mode = work_mode;
  mem_queue_attr.depth = queue_desc.depth();
  mem_queue_attr.overwrite = queue_desc.enqueue_policy() == "OVERWRITE";
  mem_queue_attr.is_client = endpoint.device_type != static_cast<int32_t>(CPU);
  endpoint.fusion_offset = queue_desc.fusion_offset();
  auto ret = exchange_service_.CreateQueue(endpoint.device_id, queue_desc.name(), mem_queue_attr, endpoint.id);
  GE_CHK_STATUS_RET(ret, "Failed to create queue, queue_name = %s.", queue_desc.name().c_str());
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::DoCreateTag(const deployer::TagDesc &tag_desc,
                                                  ExchangeEndpoint &endpoint) const {
  endpoint.id = tag_desc.tag_id();
  endpoint.tag_id = tag_desc.tag_id();
  endpoint.peer_tag_id = tag_desc.tag_id();
  endpoint.rank_id = tag_desc.rank_id();
  endpoint.peer_rank_id = tag_desc.peer_rank_id();
  endpoint.depth = tag_desc.depth();
  endpoint.peer_node_id = tag_desc.peer_node_id();
  endpoint.peer_device_id = tag_desc.peer_device_id();
  endpoint.peer_device_type = tag_desc.peer_device_type();
  uint64_t hcom_handle = UINT64_MAX;
  GE_CHK_STATUS_RET(client_manager_.GetHcomHandle(endpoint.device_id, endpoint.device_type, hcom_handle),
                    "get hcom handle failed");
  endpoint.hcom_handle = hcom_handle;
  GELOGI("Create tag success, hcom_handle = %lu, device_id = %d, tag_id = %u, peer_tag_id = %u, "
         "rank_id = %u, peer_rank_id = %u, depth = %u, peer_node_id = %d, peer_device_id = %d, peer_device_type = %d",
         endpoint.hcom_handle, endpoint.device_id, endpoint.tag_id,
         endpoint.peer_tag_id, endpoint.rank_id, endpoint.peer_rank_id, endpoint.depth, endpoint.peer_node_id,
         endpoint.peer_device_id, endpoint.peer_device_type);
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::CreateGroupTags(int32_t group_index,
                                                      const deployer::EndpointDesc &endpoint_desc,
                                                      ExchangeEndpoint &endpoint) {
  auto &group_desc = endpoint_desc.group_desc();
  for (int32_t i = 0; i < group_desc.endpoint_indices_size(); ++i) {
    const auto endpoint_index = group_desc.endpoint_indices(i);
    endpoint.endpoint_indices.emplace_back(endpoint_index);
    GELOGI("Add group element, group index = %d, element index = %d.", group_index, endpoint_index);
  }

  if (group_desc.endpoint_indices_size() > 1) {
    GE_CHK_STATUS_RET_NOLOG(CreateGroup(group_index, endpoint));
  }
  GELOGI("Create group success, group index = %d, element size = %d.",
         group_index, group_desc.endpoint_indices_size());
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::BindEndpoints(const std::vector<deployer::EndpointBinding> &bindings) {
  std::vector<std::pair<const ExchangeEndpoint *,
                        const ExchangeEndpoint *>> queue_routes;
  for (const auto &binding : bindings) {
    // pointers were checked in CreateTags
    auto src_queue_desc = GetEndpointDesc(binding.src_index());
    auto dst_queue_desc = GetEndpointDesc(binding.dst_index());
    GE_CHECK_NOTNULL(src_queue_desc);
    GE_CHECK_NOTNULL(dst_queue_desc);
    auto src_endpoint = deploying_.MutableEndpoint(binding.src_index(), src_queue_desc->type());
    auto dst_endpoint = deploying_.MutableEndpoint(binding.dst_index(), dst_queue_desc->type());
    GE_CHECK_NOTNULL(src_endpoint);
    GE_CHECK_NOTNULL(dst_endpoint);

    std::pair<const ExchangeEndpoint *, const ExchangeEndpoint *> queue_route;
    if ((src_endpoint->type == ExchangeEndpointType::kEndpointTypeGroup) &&
        (src_endpoint->endpoint_indices.size() == 1U)) {
      auto queue_desc = GetEndpointDesc(src_endpoint->endpoint_indices[0]);
      GE_CHECK_NOTNULL(queue_desc);
      auto endpoint = deploying_.MutableEndpoint(src_endpoint->endpoint_indices[0], queue_desc->type());
      GE_CHECK_NOTNULL(endpoint);
      GELOGI("Bind src endpoint is group, element size = 1, change src endpoint, "
             "index[%d->%d], type[%d->%d], id[%u->%u]",
             binding.src_index(), src_endpoint->endpoint_indices[0],
             src_queue_desc->type(), queue_desc->type(), src_endpoint->id, endpoint->id);
      src_endpoint = endpoint;
    }
    queue_route.first = src_endpoint;

    if ((dst_endpoint->type == ExchangeEndpointType::kEndpointTypeGroup) &&
        (dst_endpoint->endpoint_indices.size() == 1U)) {
      auto queue_desc = GetEndpointDesc(dst_endpoint->endpoint_indices[0]);
      GE_CHECK_NOTNULL(queue_desc);
      auto endpoint = deploying_.MutableEndpoint(dst_endpoint->endpoint_indices[0], queue_desc->type());
      GE_CHECK_NOTNULL(endpoint);
      GELOGD("Bind dst endpoint is group, element size = 1, change dst index[%d -> %d], type[%d], id[%u]",
             binding.dst_index(), dst_endpoint->endpoint_indices[0], queue_desc->type(), endpoint->id);
      dst_endpoint = endpoint;
    }
    GEEVENT("[IO info] Add bind relation, relation info = [%s]->[%s], src info = [%s], dst info = [%s]",
            src_endpoint->GetKey().c_str(), dst_endpoint->GetKey().c_str(),
            src_endpoint->DebugString().c_str(), dst_endpoint->DebugString().c_str());
    queue_route.second = dst_endpoint;
    queue_routes.emplace_back(queue_route);
    deploying_.queue_routes_.emplace_back(queue_route);
  }
  GE_CHK_STATUS_RET(BindRoute(queue_routes), "Failed to bind routes");
  return SUCCESS;
}

const deployer::EndpointDesc *HeterogeneousExchangeDeployer::GetEndpointDesc(int32_t index) {
  if (index < 0 || index >= route_plan_.endpoints_size()) {
    GELOGE(PARAM_INVALID, "index out of range, queue size = %d, index = %d", route_plan_.endpoints_size(), index);
    return nullptr;
  }
  return &route_plan_.endpoints(index);
}

Status HeterogeneousExchangeDeployer::CreateGroup(int32_t group_index,
                                                  ExchangeEndpoint &group_endpoint) {
  GELOGD("Start to create group, element count = %zu", group_endpoint.endpoint_indices.size());
  int32_t group_id = 0;
  std::vector<const ExchangeEndpoint *> endpoint_list;
  for (const auto index : group_endpoint.endpoint_indices) {
    auto endpoint_desc = GetEndpointDesc(index);
    GE_CHECK_NOTNULL(endpoint_desc);
    auto endpoint = deploying_.MutableEndpoint(index, endpoint_desc->type());
    GE_CHECK_NOTNULL(endpoint);
    if (endpoint->type == ExchangeEndpointType::kEndpointTypeGroup) {
      GELOGE(INTERNAL_ERROR, "Type of endpoint of group can't be group");
      return INTERNAL_ERROR;
    }
    endpoint_list.emplace_back(endpoint);
  }
  GE_CHK_STATUS_RET(client_manager_.CreateFlowGwGroup(static_cast<uint32_t>(group_endpoint.device_id),
                                                      group_endpoint.device_type,
                                                      endpoint_list,
                                                      group_id),
                    "Failed to create group, name = %s", group_endpoint.name.c_str());
  group_endpoint.id = static_cast<uint32_t>(group_id);
  deploying_.groups_[group_index] = endpoint_list;
  GELOGD("DataGW group created, name = %s, group id = %d", group_endpoint.name.c_str(), group_id);
  for (const auto index : group_endpoint.endpoint_indices) {
    auto endpoint_desc = GetEndpointDesc(index);
    GE_CHECK_NOTNULL(endpoint_desc);
    auto endpoint = deploying_.MutableEndpoint(index, endpoint_desc->type());
    GE_CHECK_NOTNULL(endpoint);
    GEEVENT("[IO info] Group element added, group = [%s], element = [%s], group info=[%s], element info=[%s]",
            group_endpoint.GetKey().c_str(), endpoint->GetKey().c_str(),
            group_endpoint.DebugString().c_str(), endpoint->DebugString().c_str());
  }
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::BindRoute(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes) {
  GE_CHK_STATUS_RET(client_manager_.BindQueues(queue_routes), "Failed to bind routes.");
  return SUCCESS;
}

Status HeterogeneousExchangeDeployer::UnbindRoute(
    const std::vector<std::pair<const ExchangeEndpoint *,
                                const ExchangeEndpoint *>> &queue_routes,
    FlowGwClientManager &client_manager) {
  return client_manager.UnbindQueues(queue_routes);
}

Status HeterogeneousExchangeDeployer::Undeploy(ExchangeService &exchange_service,
                                               const ExchangeRoute &deployed,
                                               FlowGwClientManager &client_manager) {
  (void)UnbindRoute(deployed.queue_routes_, client_manager);
  for (const auto &it : deployed.groups_) {
    const auto &endpoint = deployed.endpoints_[it.first];
    GELOGI("[Destroy][Group] name = %s, id = %u", endpoint->name.c_str(), endpoint->id);
    (void) client_manager.DestroyFlowGwGroup(endpoint->device_id,
                                             endpoint->device_type,
                                             static_cast<int32_t>(endpoint->id));
  }
  for (const auto &it : deployed.endpoints_) {
    const auto &endpoint = it.second;
    if (endpoint->type == ExchangeEndpointType::kEndpointTypeQueue) {
      exchange_service.DestroyQueue(endpoint->device_id, endpoint->id);
    } else if (endpoint->type == ExchangeEndpointType::kEndpointTypeTag) {
      GELOGD("[Destroy][Tag] name = %s, handle = %lu, id = %u",
             endpoint->name.c_str(),
             endpoint->hcom_handle,
             endpoint->id);
      // destroying tags are not supported with MPI
    } else {
      // do nothing
    }
  }
  return SUCCESS;
}

bool HeterogeneousExchangeDeployer::CheckExceptionEndpoint(const ExchangeEndpoint *endpoint,
    const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices) {
  std::vector<FlowGwClient::ExceptionDeviceInfo> endpoint_devices;
  FlowGwClient::ExceptionDeviceInfo exception_device_info;
  exception_device_info.node_id = endpoint->node_id;
  exception_device_info.device_id = endpoint->device_id;
  exception_device_info.device_type = endpoint->device_type;
  endpoint_devices.emplace_back(exception_device_info);
  // tag需要考虑本device和对端device
  if (endpoint->type == ExchangeEndpointType::kEndpointTypeTag) {
    FlowGwClient::ExceptionDeviceInfo peer_endpoint_device;
    peer_endpoint_device.node_id = endpoint->peer_node_id;
    peer_endpoint_device.device_id = endpoint->peer_device_id;
    peer_endpoint_device.device_type = endpoint->peer_device_type;
    endpoint_devices.emplace_back(peer_endpoint_device);
  }
  for (auto &endpoint_device : endpoint_devices) {
    for (auto &device : devices) {
      if ((endpoint_device.node_id == device.node_id) && (endpoint_device.device_id == device.device_id) &&
          (endpoint_device.device_type == device.device_type)) {
        return true;
      }
    }
  }
  return false;
}

bool HeterogeneousExchangeDeployer::CheckExceptionBind(ExchangeEndpoint *endpoint,
                                                       const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices,
                                                       std::vector<const ExchangeEndpoint *> &del_endpoints,
                                                       ExchangeRoute &deployed) {
  if (endpoint->type != ExchangeEndpointType::kEndpointTypeGroup) {
    auto is_excepton = CheckExceptionEndpoint(endpoint, devices);
    if (is_excepton) {
      endpoint->is_del = true;
      del_endpoints.emplace_back(endpoint);
      GELOGI("[UpdateExceptionRoutes] find exception route, endpoint = %s.", endpoint->DebugString().c_str());
      return true;
    }
  } else {
    bool group_is_exception = true;
    bool indice_is_exception = false;
    for (auto indice : endpoint->endpoint_indices) {
      auto indice_endpoint = deployed.MutableEndpoint(indice);
      GE_RT_FALSE_CHECK_NOTNULL(indice_endpoint);
      auto is_excepton = CheckExceptionEndpoint(indice_endpoint, devices);
      if (is_excepton) {
        indice_endpoint->is_del = true;
        del_endpoints.emplace_back(indice_endpoint);
        indice_is_exception = true;
        GELOGI("[UpdateExceptionRoutes] find exception route from group indice = %d, elements = %s.", indice,
               indice_endpoint->DebugString().c_str());
      } else {
        group_is_exception = false;
      }
    }
    if (group_is_exception) {
      endpoint->is_del = true;
      del_endpoints.emplace_back(endpoint);
      GELOGI("[UpdateExceptionRoutes] find full group exception, group = %s.", endpoint->DebugString().c_str());
    }
    return indice_is_exception;
  }
  return false;
}

void HeterogeneousExchangeDeployer::DeleteExceptionGroup(
    ExchangeRoute &deployed, const std::vector<FlowGwClient::ExceptionDeviceInfo> &devices) {
  for (auto it = deployed.groups_.begin(); it != deployed.groups_.end();) {
    const auto &endpoint = deployed.endpoints_[it->first];
    bool need_del = false;
    for (const auto &device : devices) {
      if ((endpoint->node_id == device.node_id) && (endpoint->device_id == device.device_id) &&
          (endpoint->device_type == device.device_type)) {
        need_del = true;
        break;
      }
    }
    GELOGD("[UpdateExceptionRoutes] group id = %d is need delete (%d)", it->first, need_del);
    if (need_del) {
      it = deployed.groups_.erase(it);
    } else {
      ++it;
    }
  }
}

Status HeterogeneousExchangeDeployer::UpdateExceptionRoutes(ExchangeRoute &deployed,
    FlowGwClientManager &client_manager,
    const std::vector<FlowGwClient::ExceptionDeviceInfo> &exception_devices) {
  std::vector<const ExchangeEndpoint *> del_endpoints;
  std::vector<std::pair<const ExchangeEndpoint *, const ExchangeEndpoint *>> exception_routes;
  for (auto it = deployed.queue_routes_.begin(); it != deployed.queue_routes_.end();) {
    auto src = deployed.MutableEndpoint(it->first->index);
    GE_CHECK_NOTNULL(src);
    auto dst =  deployed.MutableEndpoint(it->second->index);
    GE_CHECK_NOTNULL(dst);
    auto src_is_exception = CheckExceptionBind(src, exception_devices, del_endpoints, deployed);
    auto dst_is_exception = CheckExceptionBind(dst, exception_devices, del_endpoints, deployed);
    GELOGI("[UpdateExceptionRoutes] check exception route end, src endpoint = %s, dst endpoint = %s,"
           " src is excepion = %d, dst is exception = %d, src is del = %d, dst is del = %d.",
           src->DebugString().c_str(), dst->DebugString().c_str(), src_is_exception, dst_is_exception,
           src->is_del, dst->is_del);
    if (src->is_del || dst->is_del) {
      // 这个异常路由要解除绑定
      if (!(src->is_del && dst->is_del)) {
        exception_routes.emplace_back(*it);
      }
      it = deployed.queue_routes_.erase(it); // 删除无效路由
      GELOGI("[UpdateExceptionRoutes] route info delete from deploy record: src endpoint = %s, dst endpoint = %s.",
             src->DebugString().c_str(), dst->DebugString().c_str());
    } else {
      // 如果绑定关系在异常device上就不要去处理了，异常设备可能消息都发不去出，undeploy的时候会尝试释放
      if (src_is_exception || dst_is_exception) {
        exception_routes.emplace_back(*it);
      }
      ++it;
    }
  }
  DeleteExceptionGroup(deployed, exception_devices);
  if (exception_routes.size() != 0U) {
    client_manager.UpdateExceptionRoutes(exception_routes, deployed.endpoints_);
  }
  for (const auto &endpoint : del_endpoints) {
    // group和queue本身可以不删除，信息不会变多，在undeploy时同意删除，提前删除会有坏设备上进程范围该资源导致异常日志
    GELOGI("[UpdateExceptionRoutes] invalid endpoint = %s.", endpoint->DebugString().c_str());
    auto mutable_endpoint = deployed.MutableEndpoint(endpoint->index);
    GE_CHECK_NOTNULL(mutable_endpoint);
    mutable_endpoint->is_del = false; // 更新后的部署信息均为normal状态，绑定关系已经刷新
  }
  return SUCCESS;
}

const ExchangeRoute *HeterogeneousExchangeDeployer::GetRoute() const {
  return &deploying_;
}
ExchangeRoute *HeterogeneousExchangeDeployer::MutableRoute() {
  return &deploying_;
}

std::vector<deployer::EndpointBinding> HeterogeneousExchangeDeployer::GetBindingsAfterLoad() const {
  std::vector<deployer::EndpointBinding> bindings(route_plan_.bindings().cbegin(), route_plan_.bindings().cend());
  return bindings;
}

std::vector<deployer::EndpointBinding> HeterogeneousExchangeDeployer::GetBindingsBeforeLoad() const {
  std::vector<deployer::EndpointBinding>
      bindings(route_plan_.bindings_before_load().cbegin(), route_plan_.bindings_before_load().cend());
  return bindings;
}

std::vector<deployer::EndpointBinding> HeterogeneousExchangeDeployer::GetAllBindings() const {
  auto all_bindings = GetBindingsBeforeLoad();
  const auto &bindings = GetBindingsAfterLoad();
  all_bindings.insert(all_bindings.cend(), bindings.cbegin(), bindings.cend());
  return all_bindings;
}

Status HeterogeneousExchangeDeployer::CreateHcomHandles() {
  std::map<int32_t, int32_t> device_id_to_device_type;
  for (int32_t i = 0; i < route_plan_.endpoints_size(); ++i) {
    const auto &endpoint_desc = route_plan_.endpoints(i);
    if (static_cast<ExchangeEndpointType>(endpoint_desc.type()) == ExchangeEndpointType::kEndpointTypeTag) {
      (void) device_id_to_device_type.emplace(endpoint_desc.device_id(), endpoint_desc.device_type());
    }
  }
  if (!device_id_to_device_type.empty()) {
    GELOGI("create hcom handles in %zu devices", device_id_to_device_type.size());
    ThreadPool pool("ge_dpl_crth", device_id_to_device_type.size(), false);
    std::vector<std::future<Status>> futs;
    for (const auto &device_id_and_type : device_id_to_device_type) {
      auto fut = pool.commit([this, device_id_and_type]() -> ge::Status {
        uint64_t unused_handle;
        GE_CHK_STATUS_RET(client_manager_.GetHcomHandle(device_id_and_type.first,
                                                        device_id_and_type.second,
                                                        unused_handle),
                          "get hcom handle failed");
        return ge::SUCCESS;
      });
      futs.emplace_back(std::move(fut));
    }
    for (auto &fut : futs) {
      GE_CHK_STATUS_RET(fut.get(), "Failed to create hcom handle");
    }
    GELOGI("create hcom handles success");
  }
  return ge::SUCCESS;
}
}  // namespace ge
