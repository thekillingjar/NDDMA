/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_EVENT_PASS_H
#define AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_EVENT_PASS_H
#include "attach_resource_assign_helper.h"

namespace ge {
using AttachedEvents2Nodes = AttachedReuseKeys2Nodes;
using AttachedEventInfo = AttachedResourceInfo;
using AttachedEventInfoV2 = AttachedResourceInfoV2;
class AssignAttachedEventPass {
 public:
  AssignAttachedEventPass() = default;
  ~AssignAttachedEventPass() = default;
  Status Run(const ComputeGraphPtr &graph, uint32_t &event_num);

 private:
  static Status CheckAndGetAttachedEventInfo(const OpDescPtr &op_desc,
                                             std::vector<AttachedEventInfo> &attached_event_info);
  static Status SetAttachedEvent(const OpDescPtr &op_desc, const uint32_t resource_num, int64_t &event_id);

  static Status CheckAndGetAttachedEventInfoV2(const OpDescPtr &op_desc,
                                               std::vector<AttachedEventInfoV2> &attached_event_info);
  static Status SetAttachedEventV2(const OpDescPtr &op_desc, const std::string &reuse_key, int64_t &event_id);
  Groups2Nodes groups_2_nodes_;
};
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_EVENT_PASS_H_
