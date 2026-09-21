/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CLUSTER_H
#define AUTOFUSE_CLUSTER_H

#include "graph/node.h"
#include "ge_common/ge_api_error_codes.h"
#include "ascendc_ir.h"

namespace optimize {
struct Cluster {
  struct MetaData {
    bool enable_vf{false};
    uint32_t ins_num{0U};
    int64_t loop_axis{-1};
  };
  struct ClusterCmp {
    bool operator()(const Cluster *a, const Cluster *b) const {
      // AddInput & AddOutput has ensure a and b are not nullptr.
      return a->Id() < b->Id();
    }
  };
  Cluster(const ge::AscNodePtr &node, const size_t id) : nodes_{node}, node_set_{node}, id_(id) {}
  virtual ~Cluster() = default;

  static std::unordered_set<ge::AscNodePtr> FindConnectedNodes(const Cluster &pre, const Cluster &post);

  static std::unordered_set<ge::AscNodePtr> CalculateMergedInNodes(
      const Cluster &pre, const Cluster &post, const std::unordered_set<ge::AscNodePtr> &connected_nodes);

  static std::unordered_set<ge::AscNodePtr> CalculateMergedOutNodes(const Cluster &pre, const Cluster &post);

  void MergeFrom(Cluster &from);
  void AddInput(Cluster &input);
  void AddOutput(Cluster &output);
  void RemoveInput(Cluster &input);
  void RemoveOutput(Cluster &output);
  const std::list<ge::AscNodePtr> &Nodes() const;

  bool ContainsNode(const ge::AscNodePtr &node) const {
    return node_set_.count(node) > 0UL;
  }

  size_t Id() const {
    return id_;
  }
  const std::set<Cluster *, ClusterCmp> &Inputs() const {
    return inputs_;
  }
  const std::set<Cluster *, ClusterCmp> &Outputs() const {
    return outputs_;
  }
  std::string DebugString() const;

  MetaData meta_data_;
  std::list<ge::AscNodePtr> nodes_;
  std::unordered_set<ge::AscNodePtr> node_set_;  // 用于 O(1) 节点查找
  std::unordered_set<ge::AscNodePtr> in_nodes_;
  std::unordered_set<ge::AscNodePtr> out_nodes_;
  std::set<Cluster *, ClusterCmp> inputs_;
  std::set<Cluster *, ClusterCmp> outputs_;
  size_t id_;
};
using ClusterPtr = std::shared_ptr<Cluster>;
}  // namespace optimize
#endif  // AUTOFUSE_CLUSTER_H
