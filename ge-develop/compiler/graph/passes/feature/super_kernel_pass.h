/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_SUPER_KERNEL_PASS_H_
#define GE_GRAPH_PASSES_SUPER_KERNEL_PASS_H_

#include "graph/passes/graph_pass.h"

namespace ge {
struct InputNodeInfo {
  std::string node_name;
  NodePtr cur_node;
  size_t out_data_anchor_id; // 输入节点的输出anchor id
  size_t in_data_anchor_id; // 本阶段的输入anchor_id
};
  
struct OutputNodeInfo {
  std::string node_name;
  NodePtr cur_node;
  size_t in_data_anchor_id; // 输出节点输入anchor_id
  size_t out_data_anchor_id; // 本节点的输出anchor id
};

struct EventNodeInfo {
  NodePtr src_node;
  std::string src_node_name;
  NodePtr send_node;
  std::string send_node_name;
  NodePtr dst_node;
  std::string dst_node_name;
  NodePtr rcv_node;
  std::string rcv_node_name;
  uint32_t event_id;
  int64_t send_stream_id;
  int64_t rcv_stream_id;
};

struct NodeIoInfo {
  std::string node_name;
  NodePtr cur_node;
  std::vector<InputNodeInfo> input_nodes_info;
  std::vector<std::vector<OutputNodeInfo>> output_nodes_info;
  std::vector<EventNodeInfo> send_nodes_info;
  std::vector<EventNodeInfo> rcv_nodes_info;
};

class SuperKernelPass : public GraphPass {
 public:
  SuperKernelPass() {}

  virtual ~SuperKernelPass() {}

  Status Run(ge::ComputeGraphPtr graph) override;

 private:
  Status SelectFusionScope();
  Status AutomaticSplitScope(const std::set<std::string> &no_fusion_scope,
                             std::map<std::string, std::vector<int64_t>> &scope_cut_id);

  Status RefreshAllNodesTopoId(ge::ComputeGraphPtr root_graph) const;
  Status ParseSuperKernelOptions(const std::string &super_kernel_scope, const std::string &super_kernel_options,
                                 std::string &check_val);

  std::map<std::string, std::vector<NodePtr>> ori_super_nodes_;
  std::map<std::string, std::map<int64_t, std::vector<size_t>>> ori_super_nodes_id_;
  std::map<std::string, std::map<int64_t, std::vector<int64_t>>> ori_super_nodes_delete_id_;
  std::map<int64_t, std::vector<NodePtr>> ori_stream_ordered_nodes_;
  std::map<std::string, std::string> super_kernel_scope_options_;
};

class SuperKernelScope {
public:
  SuperKernelScope() {}
  virtual ~SuperKernelScope() {}
  
  Status Init(const std::string &name, const std::vector<NodePtr> &sk_nodes, uint32_t event_begin_id);
  
  Status MergeSuperKernelsToSubgraph();

  size_t GetScopeEventSize() const;
  
private:
  Status RecordSendInfo(const NodePtr &send_node);

  Status RecordRcvInfo(const NodePtr &rcv_node);

  Status UpdateWholeSendRcvMap(const std::vector<NodePtr> &send_rcv_nodes);

  Status GetSuperNodesIoInfo();

  Status GetSkNodeLinkInfo();

  Status GetSuperNodesEventInfo(const NodePtr &cur_node, NodeIoInfo &cur_node_info);

  Status GetSkBoundaryEventInfo();

  NodePtr ConstructSkNode();
  
  Status ConstructSubgraph();

  Status LinkSkInputNode();

  Status LinkSkOutputNode(NodePtr &inner_netoutput_node);

  Status LinkInnerNodes(NodePtr &inner_netoutput_node);

  NodePtr ConstructInnerOutputNode();

  Status LinkEventNode();

  Status InsertRecvMem2DstNode(const NodePtr &dst_node, const uint32_t event_id, const int64_t rcv_stream_id,
                               const bool is_mem_event);

  Status InsertSrcNode2SendMem(const NodePtr &src_node, const uint32_t event_id, const int64_t send_stream_id,
                               const bool is_mem_event);

  Status UnlinkSkNodes();

  Status UnlinkSrcSendLink(const NodePtr send_node) const;

  Status UnlinkRcvDstLink(const NodePtr send_node) const;

  Status UnlinkSyncLink(EventNodeInfo &event_node_info);

  Status RefreshSendList(const NodePtr src_node, const uint32_t event_id, const bool just_refresh);

  Status RefreshRcvList(const NodePtr dst_node, const uint32_t event_id, const bool just_refresh);

  void SelectSkStreamId(const std::map<int64_t, std::vector<NodePtr>> &stream_super_nodes);

  std::vector<NodePtr> super_nodes_;
  std::set<std::string> super_nodes_set_;
  std::map<std::string, NodeIoInfo> super_nodes_info_;
  std::string super_scope_name_;
  NodePtr super_node_;
  std::vector<std::pair<NodePtr, size_t>> merge_node_input_vec_;
  std::vector<std::vector<std::pair<NodePtr, size_t>>> merge_node_output_vec_;
  std::map<std::string, std::vector<std::pair<size_t, size_t>>> out_nodes_order_;
  std::map<std::string, NodePtr> newNodeMap_;
  std::map<NodePtr, std::vector<std::pair<NodePtr, uint32_t>>> send_nodes_map_;
  std::map<NodePtr, std::vector<std::pair<NodePtr, uint32_t>>> rcv_nodes_map_;
  std::map<uint32_t, EventNodeInfo> event_nodes_list_;
  std::set<uint32_t> delete_event_id_set_;
  std::set<uint32_t> reuse_event_id_set_;
  std::map<int64_t, std::vector<size_t>> super_nodes_id_;
  std::map<int64_t, std::vector<NodePtr>> stream_ordered_nodes_;

  std::set<std::string> first_other_stm_sub_nodes_;

  ComputeGraphPtr origin_graph_;
  ComputeGraphPtr sub_graph_;
  int64_t scope_stream_id_;
  uint32_t event_begin_id_;
  uint32_t event_num_ = 0;
};

}  // namespace ge
#endif  // GE_GRAPH_PASSES_SUPER_KERNEL_PASS_H_
