/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ATTACH_RESOURCE_ASSIGN_HELPER_H_
#define AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ATTACH_RESOURCE_ASSIGN_HELPER_H_

#include <string>
#include <sstream>
#include <unordered_map>

#include "ge/ge_api_types.h"
#include "ge_common/ge_api_error_codes.h"
#include "graph/any_value.h"
#include "graph/node.h"

namespace ge {
constexpr const char_t *GROUP_POLICY = "group";
const std::string DEFAULT_STREAM_INFO_GROUP = "_default_stream_info_group";
using AttachedReuseKeys2Nodes = std::map<std::pair<std::string, uint32_t>, std::vector<NodePtr>>;
using Groups2Nodes = std::unordered_map<std::string, AttachedReuseKeys2Nodes>;
struct AttachedResourceInfo {
  std::string attached_policy;      // 从属策略, 当前仅支持group
  std::string attached_group_name;  // 当从属策略为group时， 标识属于哪个group;
  std::string attached_reuse_key;   // group内按照key值进行复用分配资源， key一样的分配一样的stream、notify等
  uint32_t attached_resource_type;  // 用于给到rts接口申请时，区分不同类型的资源, 可选字段
  uint32_t attached_resource_num = 1U;
  bool NeedAssignAttachedResource() const {
    return (!attached_policy.empty() && !attached_reuse_key.empty() && !attached_group_name.empty() &&
            (attached_resource_num > 0U));
  };
  std::string ToString(const std::string &tag) const {
    std::stringstream ss;
    ss << "attached resource " << tag << " info: { policy: " << attached_policy;
    ss << ", group name: " << attached_group_name;
    ss << ", attached reuse key: " << attached_reuse_key;
    ss << ", attached resource type: " << attached_resource_type;
    ss << ", attached resource num: " << attached_resource_num;
    ss << " }";
    return ss.str();
  }
};

using GetAttachedResourceInfoFunc =
    std::function<Status(const OpDescPtr &op_desc, std::vector<AttachedResourceInfo> &attached_resource_info)>;
using SetAttachedResourceFunc =
    std::function<Status(const OpDescPtr &op_desc, const uint32_t resource_num, int64_t &resource_id)>;

struct AttachedResourceInfoV2 {
  std::string name;       // 用于指定用途,对于stream是必选字段，拼接到reuse_key中，拼接后的reuse_key相同则资源复用；
                          // 对于event/notify是可选字段，拼接到reuse_key中，如果用户未填写，框架需要生成唯一name;
                          // 后续算子按此整改，attached_policy与attached_group_name需要废弃
  std::string reuse_key;  // 可选，与name拼接后组成复用关系最终的key
                          // 即：不填时则按照默认name复用
  std::vector<int64_t> depend_value_input_indices; // 可选，用于表达值依赖，默认为空
  bool required = false; // 可选，true表示必须申请，false表示可以申请不到，默认为false
  bool force_reuse = false; // 可选，当设置为true时，GE在拼接新的reuse_key时不拼上主流stream id，默认为false

  int64_t resource_id; // GE分配完成后的stream/event/notify id
  bool is_valid = false; // GE分配后，成功返回true，失败返回false

  std::string ToString(const std::string &tag) const {
    std::stringstream ss;
    ss << "attached resource " << tag << " info: { name: " << name;
    ss << ", reuse key: " << reuse_key;
    if (!depend_value_input_indices.empty()) {
      ss << ", depend value input indices:  ";
    }
    for (const auto &item : depend_value_input_indices) {
      ss << std::to_string(item) << ", ";
    }
    ss << ", required flag: " << required;
    ss << ", resource_id: " << resource_id;
    ss << ", is_valid: " << is_valid;
    ss << " }";
    return ss.str();
  }
};

std::string CalcuSyncResourceReuseKey(const std::string &usage_name, const std::string &reuse_key,
                                       const ge::OpDescPtr &op_desc);

// 后续全部整改成V2方式
using GetAttachedResourceInfoFuncV2 =
    std::function<Status(const OpDescPtr &op_desc, std::vector<AttachedResourceInfoV2> &attached_resource_info)>;
using SetAttachedResourceFuncV2 =
    std::function<Status(const OpDescPtr &op_desc, const std::string &reuse_key, int64_t &resource_id)>;

enum class SyncResType { kSyncResEvent, kSyncResNotify, kSyncResInvalid };

class AttachedResourceAssignHelper {
 public:
  AttachedResourceAssignHelper() = default;
  ~AttachedResourceAssignHelper() = default;

  /**
   *
   * @param graph 入口图
   * @param get_resource_func 外部传入的用于从节点上获取附属信息的函数
   * @param groups_2_nodes 出参，graph内的节点按照group不同进行了分类， 同一个group内的节点按照复用信息进行了分类
   * @return
   */
  static Status ClassifyNodesByGroup(const ComputeGraphPtr &graph, const GetAttachedResourceInfoFunc &get_resource_func,
                                     const GetAttachedResourceInfoFuncV2 &get_resource_func_v2,
                                     Groups2Nodes &groups_2_nodes);

  /**
   *
   * @param attached_reuse_keys_2_nodes 一组复用作用域内的节点， 作用域当前支持是`group`级别
   * @param set_resource_func 外部传入的设置分配的resource资源到节点上的函数
   * @param resource_cnt resource资源的计数
   * @return
   */
  static Status AssignAttachedResource(const AttachedReuseKeys2Nodes &attached_reuse_keys_2_nodes,
                                       const SetAttachedResourceFunc &set_resource_func,
                                       const SetAttachedResourceFuncV2 &set_resource_func_v2, int64_t &resource_cnt);
};
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ATTACH_RESOURCE_ASSIGN_HELPER_H_
