/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PARTITION_BASE_CLUSTER_H_
#define GE_GRAPH_PARTITION_BASE_CLUSTER_H_

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>
#include "framework/common/ge_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "common/checker.h"

namespace ge {
class BasePartitioner;
using SetAttrFn = bool (*)(AttrUtils::AttrHolderAdapter &&obj, const std::string &attr_name, const AnyValue &value);
using GetAttrFn = bool (*)(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &attr_name, AnyValue &val);
class PartitionNodeAttrName {
 public:
  PartitionNodeAttrName(const std::string &attr_name, const bool is_support_tensor, const bool is_need_copy,
                        const SetAttrFn set_fn, const GetAttrFn get_fn)
      : attr_name_(attr_name),
        is_support_tensor_(is_support_tensor),
        is_need_copy_(is_need_copy),
        set_fn_(set_fn),
        get_fn_(get_fn) {}
  ~PartitionNodeAttrName() = default;
  bool IsSupportTensor() const { return is_support_tensor_; }
  bool IsNeedCopy() const { return is_need_copy_; }
  const std::string &GetAttrName() const { return attr_name_; }
  SetAttrFn SetAttrFunction() const { return set_fn_; }
  GetAttrFn GetAttrFunction() const { return get_fn_; }
 private:
  std::string attr_name_;
  bool is_support_tensor_;
  bool is_need_copy_;
  SetAttrFn set_fn_;
  GetAttrFn get_fn_;
};

class PartitionNodeAttrNameManager {
 public:
  PartitionNodeAttrNameManager(const PartitionNodeAttrNameManager &attr_name) = delete;
  PartitionNodeAttrNameManager &operator=(const PartitionNodeAttrNameManager &attr_name) = delete;
  static PartitionNodeAttrNameManager &GetInstance();
  void RegisterNodeAttrName(const PartitionNodeAttrName &attr_name);
  const std::vector<PartitionNodeAttrName> &GetAllPartitionNodeAttrNames() const { return attr_array_; }

 private:
  PartitionNodeAttrNameManager() = default;
  ~PartitionNodeAttrNameManager() = default;

  std::vector<PartitionNodeAttrName> attr_array_;
  mutable std::mutex mutex_;
};

class PartitionNodeAttrRegister {
 public:
  PartitionNodeAttrRegister(const std::string &attr_name, bool is_support_tensor,
                            const bool is_need_copy, const SetAttrFn set_fn,
                            const GetAttrFn get_fn) noexcept;
  ~PartitionNodeAttrRegister() = default;
  PartitionNodeAttrRegister(const PartitionNodeAttrRegister &other) = delete;
  PartitionNodeAttrRegister &operator=(const PartitionNodeAttrRegister &other) = delete;
};

class BaseCluster : public std::enable_shared_from_this<BaseCluster> {
 public:
  BaseCluster(size_t rank, int32_t type_index, NodePtr node, BasePartitioner *partitioner)
      : type_index_(type_index),
        id_(rank),
        min_(rank),
        max_(rank),
        partitioner_(partitioner) {
    nodes_.push_back(node);
  }
  virtual ~BaseCluster() = default;
  std::string DebugString() const;
  // Basic bean functions
  size_t Id() const;
  size_t MinId() const;
  size_t UniqueId() const;
  void UpdateRank(size_t rank);
  bool IsData() const;
  bool IsIndependent() const;
  bool IsNetOutput() const;
  std::vector<BaseCluster *> Inputs() const;
  std::vector<BaseCluster *> Outputs() const;
  bool IsInputNode() const;
  const std::vector<NodePtr> &Nodes() const;
  bool IsRefVariable() const;
  // BaseCluster modify functions
  void AddInput(std::shared_ptr<BaseCluster> in);
  void RemoveInput(std::shared_ptr<BaseCluster> in);
  void AddOutput(std::shared_ptr<BaseCluster> out);
  void RemoveOutput(std::shared_ptr<BaseCluster> out);
  // Merge other cluster to this cluster, Whether it leads to a ring or not
  // Merge src to dst means: All links to src will break and link to dst instead,
  // All nodes of src will change its owner to dst
  // Update max and min rank of dst
  virtual void Merge(std::shared_ptr<BaseCluster> other);
  // Try merge other cluster to this cluster, ONLY if will not leads to a ring
  bool TryMerge(std::shared_ptr<BaseCluster> other);
  // Merge all clusters on path(s) from other to this
  std::vector<std::shared_ptr<BaseCluster>> MergeAllPathFrom(const std::shared_ptr<BaseCluster> &other);
  // Convert cluster to functioned call functions
  bool AddFrameInput(InDataAnchorPtr anchor);
  void AddFrameOutput(OutDataAnchorPtr anchor);
  InDataAnchorPtr GetFrameInDataAnchor(InDataAnchorPtr anchor);
  OutDataAnchorPtr GetFrameOutDataAnchor(OutDataAnchorPtr anchor);
  InControlAnchorPtr GetFrameInControlAnchor();
  OutControlAnchorPtr GetFrameOutControlAnchor();
  Status BuildFrame();
  Status AddPartitionedCallNodeOutput(const NodePtr &node, const OpDescPtr &partition_op);
  Status BuildPartitionNodes(const OpDescPtr &partition_op);
  Status RemoveNodeFromRoot(const ge::ComputeGraphPtr &graph);
  Status CombinePartitionFrame();
  virtual Status BuildPartitionFrame();
  virtual Status BuildPartitionSubgraph();
  virtual Status SetAttrToNetoutput(const OpDescPtr &op) {
    (void)op;
    return SUCCESS;
  }
  static Status CopyOpAttr(const OpDescPtr &from, OpDescPtr &to);
  static Status UpdateFrameTensor(const OpDescPtr &op_desc, GeTensorDescPtr &tensor);
  static Status CopyAttr(AttrUtils::ConstAttrHolderAdapter &&from, AttrUtils::AttrHolderAdapter &&to,
                         const PartitionNodeAttrName &item);
  void MergeAllPath(const std::vector<std::shared_ptr<BaseCluster>> &path_clusters);
  // Clear resource and break circular dependency
  void Clear();
  int32_t GetTypeIndex() const {
    return type_index_;
  }
  void SetTypeIndex(int32_t type_index) {
    type_index_ = type_index;
  }
  void SetMergeInputs(bool merge_inputs);
  std::string GetPartitionedCallName() {
    if (partition_node_ != nullptr) {
      return partition_node_->GetName();
    }
    return "";
  }
  Status SetGraphId(const uint32_t graph_id) const;

 protected:
  std::vector<NodePtr> &GetMutableDirectNode();
  size_t GetSubgraphNodesSize() const { return nodes_.size(); }
  const std::unordered_map<InDataAnchorPtr, size_t> &GetInputIndexs() const { return inputs_index_; }
  const std::unordered_map<OutDataAnchorPtr, size_t> &GetOutputIndexs() const { return outputs_index_; }
  ComputeGraphPtr &GetMutableSubgraph() { return subgraph_; }
  const ComputeGraphPtr &GetSubgraph() const { return subgraph_; }
  const NodePtr &GetPartitionNode() const { return partition_node_; }
  const BasePartitioner *GetPartitioner() const { return partitioner_; };
  void UpdateInOutClusters(const std::unordered_set<BaseCluster *> &path_cluster_set);
  void MergeInOutClusters(const std::unordered_set<BaseCluster *> &path_cluster_set,
                          const std::vector<BaseCluster *> &ordered_path_clusters);

 private:
  static thread_local size_t unique_id_;
  int32_t type_index_;
  size_t id_;
  // Each Cluster records the maximum and minimum topological order of its node
  size_t min_;  // maximum topological order
  size_t max_;  // minimum topological order
  std::vector<BaseCluster *> in_clusters_;
  std::vector<BaseCluster *> out_clusters_;
  std::vector<InDataAnchorPtr> inputs_;
  std::map<size_t, std::vector<InDataAnchorPtr>> data_to_node_inputs_;
  std::map<std::string, size_t> src_key_to_frame_input_;
  std::vector<OutDataAnchorPtr> outputs_;
  std::unordered_set<std::shared_ptr<BaseCluster>> control_inputs_;
  std::unordered_set<OutControlAnchorPtr> control_outputs_;
  std::vector<NodePtr> nodes_;
  std::unordered_map<InDataAnchorPtr, size_t> inputs_index_;
  std::unordered_map<OutDataAnchorPtr, size_t> outputs_index_;
  // Fields for build partitioned call and subgraph
  BasePartitioner *partitioner_{nullptr};  // Not owned, the partitioner this cluster belongs to
  ComputeGraphPtr subgraph_{nullptr};  // corresponding subgraph
  NodePtr partition_node_{nullptr};  // corresponding partitioned call node
  bool merge_inputs_{false};
};
}  // namespace ge
#define REGISTER_PARTITION_ATTR_NAME(attr_name, is_support_tensor, is_need_copy, set_attr_fn, get_attr_fn)  \
  static ge::PartitionNodeAttrRegister g_##attr_name##_register(attr_name, is_support_tensor, is_need_copy, \
                                                                set_attr_fn, get_attr_fn)

#define SET_ATTR_UTIL_IMP(ArgType, AttrUtilsFun)                                                            \
  static bool Set##AttrUtilsFun##Attr(ge::AttrUtils::AttrHolderAdapter &&obj, const std::string &attr_name, \
                                      const ge::AnyValue &value) {                                          \
    ArgType val;                                                                                            \
    (void)value.GetValue(val);                                                                              \
    return ge::AttrUtils::Set##AttrUtilsFun(obj.get(), attr_name, val);                                     \
  }

#define GET_ATTR_UTIL_IMP(ArgType, AttrUtilsFun)                                                        \
  static bool Get##AttrUtilsFun##Attr(AttrUtils::ConstAttrHolderAdapter &&obj, const std::string &attr, \
                                      AnyValue &value) {                                                \
    ArgType val;                                                                                        \
    const auto res = AttrUtils::Get##AttrUtilsFun(obj.get(), attr, val);                                \
    value.SetValue(val);                                                                                \
    return res;                                                                                         \
  }
#endif  // GE_GRAPH_PARTITION_BASE_CLUSTER_H_
