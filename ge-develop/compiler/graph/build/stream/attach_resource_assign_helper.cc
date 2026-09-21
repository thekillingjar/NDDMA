/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attach_resource_assign_helper.h"

#include "compute_graph.h"
#include "common/checker.h"
#include "graph/debug/ge_attr_define.h"

namespace ge {
constexpr char_t const *kDisableAttachedResource = "_disable_attached_resource";

std::string CalcuSyncResourceReuseKey(const std::string &usage_name, const std::string &reuse_key,
                                      const ge::OpDescPtr &op_desc) {
  std::string calc_reuse_key = usage_name;
  if (reuse_key.empty()) {  // 为空表示不复用
    static std::atomic<size_t> event_cnt(0U);
    calc_reuse_key.append(op_desc->GetName()).append("_attr_").append(std::to_string(event_cnt.fetch_add(1U)));
  } else {
    calc_reuse_key.append(reuse_key);
  }
  return calc_reuse_key;
}

Status AttachedResourceAssignHelper::ClassifyNodesByGroup(const ComputeGraphPtr &graph,
                                                          const GetAttachedResourceInfoFunc &get_resource_func,
                                                          const GetAttachedResourceInfoFuncV2 &get_resource_func_v2,
                                                          Groups2Nodes &groups_2_nodes) {
  for (const auto &node : graph->GetDirectNode()) {
    GE_ASSERT_NOTNULL(node);
    const auto &op_desc = node->GetOpDesc();
    GE_ASSERT_NOTNULL(op_desc);
    if (op_desc->HasAttr(kDisableAttachedResource)) {
      continue;
    }
    if (ge::AttrUtils::HasAttr(op_desc, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST) ||
        ge::AttrUtils::HasAttr(op_desc, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST)) {
      std::vector<AttachedResourceInfoV2> attached_resource_info_v2;
      GE_ASSERT_SUCCESS(get_resource_func_v2(op_desc, attached_resource_info_v2),
                        "op %s get attached stream info v2 failed.", op_desc->GetName().c_str());       //
      const static std::string attached_group_name = DEFAULT_STREAM_INFO_GROUP;  // 使用默认，后续待算子完整整改后删除
      const static uint32_t attached_resource_num = 1U;  // 使用默认，后续待算子完整整改后删除
      for (auto &res_info : attached_resource_info_v2) {
        if (!res_info.required) {
          continue;
        }
        groups_2_nodes[attached_group_name][std::make_pair(res_info.reuse_key, attached_resource_num)].push_back(
            node);
      }
    } else {
      std::vector<AttachedResourceInfo> attached_resource_info;
      GE_ASSERT_SUCCESS(get_resource_func(op_desc, attached_resource_info), "op %s get attached stream info failed.",
                        op_desc->GetName().c_str());
      for (auto &res_info : attached_resource_info) {
        if (!res_info.NeedAssignAttachedResource()) {
          continue;
        }
        groups_2_nodes[res_info.attached_group_name]
                      [std::make_pair(res_info.attached_reuse_key, res_info.attached_resource_num)]
                          .push_back(node);
      }
    }
  }
  return SUCCESS;
}

Status AttachedResourceAssignHelper::AssignAttachedResource(
    const AttachedReuseKeys2Nodes &attached_reuse_keys_2_nodes, const SetAttachedResourceFunc &set_resource_func,
    const SetAttachedResourceFuncV2 &set_resource_func_v2, int64_t &resource_cnt) {
  for (const auto &attach_reuse_key_2_nodes : attached_reuse_keys_2_nodes) {
    GELOGI(
        "Begin to assign same attached resource for nodes size{%zu} with same attach reuse key{%s}, resource num{%u} "
        "resource_cnt before "
        "is %ld",
        attach_reuse_key_2_nodes.second.size(), attach_reuse_key_2_nodes.first.first.c_str(),
        attach_reuse_key_2_nodes.first.second, resource_cnt);
    for (const auto &node : attach_reuse_key_2_nodes.second) {
      const auto &op_desc = node->GetOpDesc();
      GE_ASSERT_NOTNULL(op_desc);
      if (ge::AttrUtils::HasAttr(op_desc, ge::ATTR_NAME_ATTACHED_STREAM_INFO_LIST) ||
          ge::AttrUtils::HasAttr(op_desc, ge::ATTR_NAME_ATTACHED_SYNC_RES_INFO_LIST)) {
        GE_ASSERT_SUCCESS(set_resource_func_v2(op_desc, attach_reuse_key_2_nodes.first.first, resource_cnt));
      } else {
        GE_ASSERT_SUCCESS(set_resource_func(op_desc, attach_reuse_key_2_nodes.first.second, resource_cnt));
      }
    }
    resource_cnt += attach_reuse_key_2_nodes.first.second;
  }
  return SUCCESS;
}
}  // namespace ge
