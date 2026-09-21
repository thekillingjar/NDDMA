/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_NOTIFY_PASS_H_
#define AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_NOTIFY_PASS_H_
#include "attach_resource_assign_helper.h"

namespace ge {
using AttachedNotifys2Nodes = AttachedReuseKeys2Nodes;
using AttachedNotifyInfo = AttachedResourceInfo;
using AttachedNotifyInfoV2 = AttachedResourceInfoV2;
class AssignAttachedNotifyPass {
 public:
  AssignAttachedNotifyPass() = default;
  ~AssignAttachedNotifyPass() = default;
  Status Run(const ComputeGraphPtr &graph, uint32_t &notify_num, std::vector<uint32_t> &notify_types);

 private:
  static Status CheckAndGetAttachedNotifyInfo(const OpDescPtr &op_desc,
                                              std::vector<AttachedNotifyInfo> &attached_notify_info);
  static Status CheckAndGetNotifyType(const AttachedNotifys2Nodes &attached_notifys_2_nodes,
                                      std::vector<uint32_t> &notify_types);
  static Status SetAttachedNotify(const OpDescPtr &op_desc, const uint32_t resource_num, int64_t &notify_id);

  static Status CheckAndGetAttachedNotifyInfoV2(const OpDescPtr &op_desc,
                                                std::vector<AttachedNotifyInfoV2> &attached_notify_info);
  static Status SetAttachedNotifyV2(const OpDescPtr &op_desc, const std::string &reuse_key, int64_t &notify_id);

  Groups2Nodes groups_2_nodes_;
};
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_NOTIFY_PASS_H_
