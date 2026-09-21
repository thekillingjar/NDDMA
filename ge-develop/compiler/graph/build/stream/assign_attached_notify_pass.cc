/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assign_attached_notify_pass.h"

#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "stream_utils.h"
#include "utils/extern_math_util.h"
namespace {
const std::string RES_TYPE_NOTIFY = "notify";
}
namespace ge {
const char *const kAttachedNotifyNum = "_attached_notify_num";
Status AssignAttachedNotifyPass::Run(const ComputeGraphPtr &graph, uint32_t &notify_num,
                                     std::vector<uint32_t> &notify_types) {
  GE_ASSERT_NOTNULL(graph);
  // 动态子图不做处理
  GE_ASSERT_SUCCESS(AttachedResourceAssignHelper::ClassifyNodesByGroup(
      graph, CheckAndGetAttachedNotifyInfo, CheckAndGetAttachedNotifyInfoV2, groups_2_nodes_));
  if (groups_2_nodes_.empty()) {
    return SUCCESS;
  }
  GE_ASSERT_EQ(notify_num, static_cast<uint32_t>(notify_types.size()));
  int64_t notify_num_new = static_cast<int64_t>(notify_num);
  GELOGI("Start assign attached notify for graph %s with group num: %zu ", graph->GetName().c_str(),
         groups_2_nodes_.size());
  for (const auto &group_2_nodes : groups_2_nodes_) {
    const auto &group_name = group_2_nodes.first;
    const auto &nodes_in_this_group = group_2_nodes.second;
    std::vector<uint32_t> notify_types_of_this_group;
    GE_ASSERT_SUCCESS(CheckAndGetNotifyType(nodes_in_this_group, notify_types_of_this_group));
    GE_ASSERT_SUCCESS(AttachedResourceAssignHelper::AssignAttachedResource(nodes_in_this_group, SetAttachedNotify,
                                                                           SetAttachedNotifyV2, notify_num_new));
    notify_types.insert(notify_types.cend(), notify_types_of_this_group.begin(), notify_types_of_this_group.end());
    GE_ASSERT_EQ(notify_num_new, static_cast<int64_t>(notify_types.size()));
    GELOGI(
        "Assign attached notify successfully for nodes with same group name {%s} in graph %s, total notify number "
        "old is:%ld, now is:%ld",
        group_name.c_str(), graph->GetName().c_str(), notify_num, notify_num_new);
  }
  GE_ASSERT_TRUE(ge::IntegerChecker<uint32_t>::Compat(notify_num_new));
  notify_num = static_cast<uint32_t>(notify_num_new);
  return SUCCESS;
}

Status AssignAttachedNotifyPass::CheckAndGetAttachedNotifyInfo(const OpDescPtr &op_desc,
                                                               std::vector<AttachedNotifyInfo> &attached_notify_info) {
  NamedAttrs attached_notify_info_from_attr;
  if (!(AttrUtils::GetNamedAttrs(op_desc, ATTR_NAME_ATTACHED_NOTIFY_INFO, attached_notify_info_from_attr))) {
    return SUCCESS;
  }
  AttachedNotifyInfo cur_attached_notify_info;
  // 校验是否有设置策略
  GE_ASSERT_TRUE(AttrUtils::GetStr(attached_notify_info_from_attr, ATTR_NAME_ATTACHED_NOTIFY_POLICY,
                                   cur_attached_notify_info.attached_policy));
  // 当前仅仅支持group级别的分配策略
  GE_ASSERT_EQ(cur_attached_notify_info.attached_policy, GROUP_POLICY);
  // 校验是否有设置group名称
  GE_ASSERT_TRUE(AttrUtils::GetStr(op_desc, GROUP_POLICY, cur_attached_notify_info.attached_group_name));
  GE_ASSERT_TRUE(AttrUtils::GetStr(attached_notify_info_from_attr, ATTR_NAME_ATTACHED_NOTIFY_KEY,
                                   cur_attached_notify_info.attached_reuse_key));

  cur_attached_notify_info.attached_resource_type = 0U;  // 默认类型
  (void)AttrUtils::GetInt(attached_notify_info_from_attr, ATTR_NAME_ATTACHED_NOTIFY_TYPE,
                          cur_attached_notify_info.attached_resource_type);
  cur_attached_notify_info.attached_resource_num = 1U;  // 默认数
  (void)AttrUtils::GetInt(attached_notify_info_from_attr, kAttachedNotifyNum,
                          cur_attached_notify_info.attached_resource_num);
  // 为了便于处理，notify_type复制一份到op_desc上保存
  GE_ASSERT_TRUE(
      AttrUtils::SetInt(op_desc, ATTR_NAME_ATTACHED_NOTIFY_TYPE, cur_attached_notify_info.attached_resource_type));

  attached_notify_info.emplace_back(cur_attached_notify_info);
  GELOGD("Op [%s %s] get [%s] successfully.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
         cur_attached_notify_info.ToString("notify").c_str());
  // ATTR_NAME_ATTACHED_NOTIFY_INFO属性的生命周期到这里应该就结束了，为了减少OM体积，删除他
  (void)op_desc->DelAttr(ATTR_NAME_ATTACHED_NOTIFY_INFO);
  return SUCCESS;
}

Status AssignAttachedNotifyPass::CheckAndGetNotifyType(const AttachedNotifys2Nodes &attached_notifys_2_nodes,
                                                       std::vector<uint32_t> &notify_types) {
  for (const auto &attach_notify_2_nodes : attached_notifys_2_nodes) {
    std::set<uint32_t> notify_type_of_this_attach_notify;
    for (const auto &node : attach_notify_2_nodes.second) {
      GE_ASSERT_NOTNULL(node);
      GE_ASSERT_NOTNULL(node->GetOpDesc());
      uint32_t notify_type_of_this_node = 0U;
      // 新的流程下，不再使用这个属性，因此默认用0U即可，不必校验报错
      (void) AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_ATTACHED_NOTIFY_TYPE, notify_type_of_this_node);
      notify_type_of_this_attach_notify.insert(notify_type_of_this_node);
    }
    // 复用一个notify id的算子, 其notify type应该要一样
    GE_ASSERT_EQ(notify_type_of_this_attach_notify.size(), 1U);
    const size_t new_notify_size = notify_types.size() + attach_notify_2_nodes.first.second;
    notify_types.resize(new_notify_size, *notify_type_of_this_attach_notify.begin());
  }
  return SUCCESS;
}

Status AssignAttachedNotifyPass::SetAttachedNotify(const OpDescPtr &op_desc, const uint32_t resource_num,
                                                   int64_t &notify_id) {
  vector<int64_t> notify_ids(resource_num);
  for (uint32_t i = 0UL; i < resource_num; ++i) {
    notify_ids[i] = notify_id + i;
  }
  GE_ASSERT_TRUE(AttrUtils::SetListInt(op_desc, RECV_ATTR_NOTIFY_ID, notify_ids));
  GELOGI("Assign attached notify id [%ld] for node {%s %s} successfully", notify_id, op_desc->GetNamePtr(),
         op_desc->GetTypePtr());
  return SUCCESS;
}

Status AssignAttachedNotifyPass::CheckAndGetAttachedNotifyInfoV2(
    const ge::OpDescPtr &op_desc, std::vector<AttachedNotifyInfoV2> &attached_notify_info) {
  std::vector<ge::NamedAttrs> res_info_attrs;
  if (!(AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, res_info_attrs))) {
    return SUCCESS;
  }
  for (auto &attr : res_info_attrs) {
    auto res_type = static_cast<int64_t>(SyncResType::kSyncResInvalid);
    (void)AttrUtils::GetInt(attr, ATTR_NAME_ATTACHED_RESOURCE_TYPE, res_type);
    if (res_type != static_cast<int64_t>(SyncResType::kSyncResNotify)) {
      GELOGI("Resource type [%d] is not equal to notify type[%d], will not assign attached notify", res_type,
             static_cast<int64_t>(SyncResType::kSyncResNotify));
      continue;
    }
    AttachedNotifyInfoV2 cur_attached_notify_info;
    (void)AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_NAME, cur_attached_notify_info.name);
    (void)AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, cur_attached_notify_info.reuse_key);
    cur_attached_notify_info.reuse_key =
        CalcuSyncResourceReuseKey(cur_attached_notify_info.name, cur_attached_notify_info.reuse_key, op_desc);
    // 更新计算后的reuse_key
    (void)AttrUtils::SetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, cur_attached_notify_info.reuse_key);
    (void)AttrUtils::GetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_REQUIRED_FLAG, cur_attached_notify_info.required);
    attached_notify_info.emplace_back(cur_attached_notify_info);

    GELOGD("Op [%s %s] get [%s] successfully.", op_desc->GetNamePtr(), op_desc->GetTypePtr(),
           cur_attached_notify_info.ToString(RES_TYPE_NOTIFY).c_str());
  }
  (void)AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST, res_info_attrs);
  return SUCCESS;
}

Status AssignAttachedNotifyPass::SetAttachedNotifyV2(const OpDescPtr &op_desc, const std::string &reuse_key,
                                                     int64_t &notify_id) {
  GE_ASSERT_NOTNULL(op_desc);
  std::vector<NamedAttrs> attached_sync_res_info_list_from_attr;
  GELOGD("Try set attached notify id %d with reuse_key %s, op %s [%s]", notify_id, reuse_key.c_str(),
         op_desc->GetName().c_str(), op_desc->GetType().c_str());
  if (!AttrUtils::GetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                              attached_sync_res_info_list_from_attr)) {
    return SUCCESS;
  }

  for (auto &attr : attached_sync_res_info_list_from_attr) {
    std::string reuse_key_from_attr;
    GE_ASSERT_TRUE(AttrUtils::GetStr(attr, ATTR_NAME_ATTACHED_RESOURCE_REUSE_KEY, reuse_key_from_attr));
    GELOGD("Try to set notify id, reuse key is %s, reuse_key_from_attr is %s", reuse_key.c_str(),
           reuse_key_from_attr.c_str());
    if (reuse_key != reuse_key_from_attr) {
      continue;
    }
    GE_ASSERT_TRUE(AttrUtils::SetInt(attr, ATTR_NAME_ATTACHED_RESOURCE_ID, notify_id));
    GE_ASSERT_TRUE(AttrUtils::SetBool(attr, ATTR_NAME_ATTACHED_RESOURCE_IS_VALID, true));
    GELOGI("Assign attached notify [%ld] for node {%s %s} with main stream [%ld] and reuse_key [%s] successfully",
           notify_id, op_desc->GetNamePtr(), op_desc->GetTypePtr(), op_desc->GetStreamId(), reuse_key.c_str());
  }
  (void)AttrUtils::SetListNamedAttrs(op_desc, ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST,
                                     attached_sync_res_info_list_from_attr);
  return SUCCESS;
}
}  // namespace ge
