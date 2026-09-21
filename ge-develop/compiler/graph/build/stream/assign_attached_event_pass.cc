/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assign_attached_event_pass.h"

#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "stream_utils.h"
#include "utils/extern_math_util.h"

namespace {
const std::string DEFAULT_EVENT_INFO_GROUP_NAME = "_default_event_info_group";
const std::string RES_TYPE_EVENT = "event";
const std::string ATTACHED_SYNC_RES_TYPE = "_attached_sync_res_type";
const std::string ATTACHED_SYNC_RES_KEY = "_attached_sync_res_key";
const std::string ATTR_NAME_ATTACHED_SYNC_RES_INFO = "_attached_sync_res_info";
}
namespace ge {
Status AssignAttachedEventPass::Run(const ComputeGraphPtr &graph, uint32_t &event_num) {
  GE_ASSERT_NOTNULL(graph);
  // 动态子图不做处理
  if (StreamUtils::IsDynamicSubGraph(graph)) {
    GELOGI("Skip assign attached event for [%s]", graph->GetName().c_str());
    return SUCCESS;
  }

  GE_ASSERT_SUCCESS(AttachedResourceAssignHelper::ClassifyNodesByGroup(
      graph, CheckAndGetAttachedEventInfo, CheckAndGetAttachedEventInfoV2, groups_2_nodes_));
  if (groups_2_nodes_.empty()) {
    return SUCCESS;
  }
  int64_t event_num_new = static_cast<int64_t>(event_num);
  GELOGI("Start assign attached event for graph %s with group num: %zu", graph->GetName().c_str(),
         groups_2_nodes_.size());
  for (const auto &group_2_nodes : groups_2_nodes_) {
    const auto &group_name = group_2_nodes.first;
    const auto &nodes_in_this_group = group_2_nodes.second;
    GE_ASSERT_SUCCESS(AttachedResourceAssignHelper::AssignAttachedResource(nodes_in_this_group, SetAttachedEvent,
                                                                           SetAttachedEventV2, event_num_new));
    GELOGI(
        "Assign attached event successfully for nodes with same group name {%s} in graph %s, total event number "
        "old is:%d, now is:%ld",
        group_name.c_str(), graph->GetName().c_str(), event_num, event_num_new);
  }
  GE_ASSERT_TRUE(ge::IntegerChecker<uint32_t>::Compat(event_num_new));
  event_num = static_cast<uint32_t>(event_num_new);
  return SUCCESS;
}

Status AssignAttachedEventPass::CheckAndGetAttachedEventInfo(const OpDescPtr &op_desc,
                                                             std::vector<AttachedEventInfo> &attached_event_info) {
  std::vector<NamedAttrs> res_info_attrs;
  if (!(AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO, res_info_attrs))) {
    return SUCCESS;
  }
  size_t attr_cnt = 0UL;
  for (const auto &attr : res_info_attrs) {
    std::string res_type;
    (void)AttrUtils::GetStr(attr, ATTACHED_SYNC_RES_TYPE, res_type);
    if (res_type != RES_TYPE_EVENT) {
      continue;
    }
    AttachedEventInfo cur_attached_event_info;
    cur_attached_event_info.attached_policy = GROUP_POLICY;
    cur_attached_event_info.attached_group_name = DEFAULT_EVENT_INFO_GROUP_NAME;
    cur_attached_event_info.attached_resource_type = 0U; // 默认类型
    cur_attached_event_info.attached_resource_num = 1U; // 默认数
    std::string res_key;
    res_key.append(op_desc->GetName())
           .append("_attr_")
           .append(std::to_string(attr_cnt++));
    (void)AttrUtils::GetStr(attr, ATTACHED_SYNC_RES_KEY, res_key); // 不设属性默认不复用
    cur_attached_event_info.attached_reuse_key = res_key;
    GELOGD("Op [%s %s] get [%s] successfully.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
            cur_attached_event_info.ToString(RES_TYPE_EVENT).c_str());

    attached_event_info.emplace_back(cur_attached_event_info);
  }
  // ATTR_NAME_ATTACHED_SYNC_RES_INFO属性的生命周期到这里应该就结束了，为了减少OM体积，删除他
  (void)op_desc->DelAttr(ATTR_NAME_ATTACHED_SYNC_RES_INFO);
  return SUCCESS;
}

Status AssignAttachedEventPass::SetAttachedEvent(const OpDescPtr &op_desc, const uint32_t resource_num,
                                                 int64_t &event_id) {
  vector<int64_t> event_ids;
  (void)AttrUtils::GetListInt(op_desc, RECV_ATTR_EVENT_ID, event_ids);
  for (uint32_t i = 0UL; i < resource_num; ++i) {
    event_ids.emplace_back(event_id + i);
  }
  GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, RECV_ATTR_EVENT_ID, event_ids));
  GELOGI("Assign attached event id [start:%ld, num:%u] for node {%s %s} successfully", event_id, resource_num,
         op_desc->GetNamePtr(), op_desc->GetTypePtr());
  return SUCCESS;
}

Status AssignAttachedEventPass::CheckAndGetAttachedEventInfoV2(const ge::OpDescPtr &op_desc,
                                                               std::vector<AttachedEventInfoV2> &attached_event_info) {
  std::vector<NamedAttrs> res_info_attrs;
  if (!(AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, res_info_attrs))) {
    return SUCCESS;
  }
  for (auto &attr : res_info_attrs) {
    auto res_type = static_cast<int64_t>(SyncResType::kSyncResInvalid);
    (void) AttrUtils::GetInt(attr, ATTR_NAME_ATTACHED_RESOURCE_TYPE, res_type);
    if (res_type != static_cast<int64_t>(SyncResType::kSyncResEvent)) {
      GELOGI("Resource type [%d] is not equal to event type[%d], will not assign attached event", res_type,
             static_cast<int64_t>(SyncResType::kSyncResEvent));
      continue;
    }
    AttachedEventInfoV2 cur_attached_event_info;
    (void) AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_NAME, cur_attached_event_info.name);
    (void) AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, cur_attached_event_info.reuse_key);
    cur_attached_event_info.reuse_key =
        CalcuSyncResourceReuseKey(cur_attached_event_info.name, cur_attached_event_info.reuse_key, op_desc);
    // 更新计算后的reuse_key
    (void) AttrUtils::SetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, cur_attached_event_info.reuse_key);
    (void) AttrUtils::GetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, cur_attached_event_info.required);

    GELOGD("Op [%s %s] get [%s] successfully.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           cur_attached_event_info.ToString(RES_TYPE_EVENT).c_str());

    attached_event_info.emplace_back(cur_attached_event_info);
  }
  (void) AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, res_info_attrs);
  return SUCCESS;
}

Status AssignAttachedEventPass::SetAttachedEventV2(const OpDescPtr &op_desc, const std::string &reuse_key,
                                                 int64_t &event_id) {
  GE_ASSERT_NOTNULL(op_desc);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  GELOGD("Try set attached event id %d with reuse_key %s, op %s [%s]", event_id, reuse_key.c_str(),
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (!AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                   attached_sync_res_info_list_from_attr)) {
    return ge::SUCCESS;
  }

  for (auto &attr : attached_sync_res_info_list_from_attr) {
    std::string reuse_key_from_attr;
    (void) AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, reuse_key_from_attr);
    GELOGD("Try to set event id, reuse key is %s, reuse_key_from_attr is %s", reuse_key.c_str(),
           reuse_key_from_attr.c_str());
    if (reuse_key != reuse_key_from_attr) {
      GELOGD("Calculated reuse key [%s] is not equal to current reuse key[%s]", reuse_key_from_attr.c_str(),
             reuse_key.c_str());
      continue;
    }
    GE_ASSERT_TRUE(AttrUtils::SetInt(attr, ATTR_NAME_ATTACHED_RESOURCE_ID, event_id));
    GE_ASSERT_TRUE(AttrUtils::SetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, true));
    GELOGI("Assign attached event [%ld] for node {%s %s} with main stream [%ld] and reuse_key [%s] successfully",
           event_id, op_desc->GetNamePtr(), op_desc->GetTypePtr(), op_desc->GetStreamId(), reuse_key.c_str());
  }
  GE_ASSERT_TRUE(AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                              attached_sync_res_info_list_from_attr));
  return SUCCESS;
}
}  // namespace ge
