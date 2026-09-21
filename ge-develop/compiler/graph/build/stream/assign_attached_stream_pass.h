/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_STREAM_PASS_H_
#define AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_STREAM_PASS_H_
#include "logical_stream_allocator.h"

#include "attach_resource_assign_helper.h"
namespace ge {
class AssignAttachedStreamPass : public LogicalStreamPass {
 public:
  using AttachedStreams2Nodes = AttachedReuseKeys2Nodes;
  using AttachedStreamInfo = AttachedResourceInfo;
  using AttachedStreamInfoV2 = AttachedResourceInfoV2;
  STREAM_PASS_DEFAULT_FUNC(AssignAttachedStreamPass);
  Status Run(ComputeGraphPtr graph, const std::vector<SubgraphPtr> &subgraphs, Context &context) override;
  Status Run(ComputeGraphPtr graph, int64_t &stream_num);

 private:
  static Status CheckAndGetAttachedStreamInfo(const OpDescPtr &op_desc,
                                              std::vector<AttachedStreamInfo> &attached_stream_info);
  static Status SetAttachedStream(const OpDescPtr &op_desc, const uint32_t stream_num, int64_t &stream_id);

  static Status CheckAndGetAttachedStreamInfoV2(const OpDescPtr &op_desc,
                                                std::vector<AttachedStreamInfoV2> &attached_stream_info);

  static Status SetAttachedStreamV2(const OpDescPtr &op_desc, const std::string &reuse_key, int64_t &stream_id);

  static Status GetAttachedStreamInfoByNamedAttrs(const OpDescPtr &op_desc,
                                                  std::vector<AttachedStreamInfo> &attached_stream_info);
  static Status GetAttachedStreamInfoByListNamedAttrs(const OpDescPtr &op_desc,
                                                      std::vector<AttachedStreamInfo> &attached_stream_info);
  Groups2Nodes groups_2_nodes_;
};
}  // namespace ge
#endif  // AIR_CXX_COMPILER_GRAPH_BUILD_STREAM_ASSIGN_ATTACHED_STREAM_PASS_H_
