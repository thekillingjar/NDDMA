/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_FUSION_STRATEGY_SOLVER_H_
#define AUTOFUSE_CAN_FUSE_BACKEND_FUSION_STRATEGY_SOLVER_H_
#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "exe_graph/runtime/symbolic_tensor.h"
#include "can_fuse/backend/fusion_decider.h"
#include "graph/utils/cycle_detector.h"
#include "autofuse_frame/autofuse_frames.h"

namespace ge {
class FusingNode;
using FusingNodePtr = std::shared_ptr<FusingNode>;

struct FusionStatistics {
  size_t min_scale;          // AscBackend节点最小规模（asc子图内节点数)
  size_t max_scale;          // AscBackend节点最大规模（asc子图内节点数)
  float_t average_scale;     // AscBackend节点平均规模（asc子图内节点数)
  size_t nodes_size;         // 整图上节点数
  Expression rw_memory_size; // 整图读写内存大小
};

// 保存可以融合的节点对，并缓存融合后节省的内存大小和邻近性信息，减少重复运算提升性能
class NodePair {
 public:
  NodePair() = delete;
  NodePair(const NodePair &other) = default;
  NodePair &operator=(const NodePair &other) = default;
  NodePair(NodePair &&other) = default;
  NodePair &operator=(NodePair &&other) = default;

  NodePair(const FusingNodePtr &f, const FusingNodePtr &s);

  // 根据内存大小和邻近性进行排序
  bool operator<(const NodePair &other) const;

  // 节点对相同就认为相同
  bool operator==(const NodePair &other) const {
    return ((this == &other) || ((first == other.first) && (second == other.second)));
  }

  FusingNodePtr first;
  FusingNodePtr second;
  Expression memory_score;
  int64_t proximity_score;
};

using NodePairList = std::vector<NodePair>;
using FusingNodes = std::map<Node *, FusingNodePtr>;

class MemoryBuffer {
 public:
  MemoryBuffer() = delete;
  MemoryBuffer(const MemoryBuffer &other) = default;
  MemoryBuffer &operator=(const MemoryBuffer &other) = default;
  MemoryBuffer(MemoryBuffer &&other) = default;
  MemoryBuffer &operator=(MemoryBuffer &&other) = default;

  MemoryBuffer(const Anchor *const b, const Expression &s)
      : buffer(b), size(s), node(nullptr), hash_name(0U), index(0) {
    if (b != nullptr) {
      index = b->GetIdx();
      node = b->GetOwnerNodeBarePtr();
      if (node != nullptr) {
        std::hash<std::string> hash_fn;
        hash_name = hash_fn(node->GetName());
      }
    }
  }

  bool operator<(const MemoryBuffer &other) const;

  const Anchor *Buffer() const {
    return buffer;
  }

  const Anchor *buffer;
  const Expression size;
  Node *node;
  size_t hash_name;
  int32_t index;
};

// 融合处理中间过程的临时node主要用于存储读写内存大小，原始节点数，最大最小排序等信息，用于融合决策判断
class FusingNode {
 public:
  explicit FusingNode(const NodePtr &node);
  FusingNode() = delete;
  FusingNode(const FusingNode &other) = default;
  FusingNode &operator=(const FusingNode &other) = default;
  FusingNode(FusingNode &&other) = default;
  FusingNode &operator=(FusingNode &&other) = default;

  Status Init();
  Status UpdateReadsAndWrites();
  Status Fuse(const FusingNodePtr &node1, const FusingNodePtr &node2, const FusingNodes &fusing_nodes);
  bool IsAncestor(const FusingNodePtr &node) const;
  void UpdateOrder();

  int64_t GetId() const {
    return id_;
  }

  void SetId(int64_t id) {
    id_ = id;
  }

  const NodePtr &GetOrgNode() const {
    return node_;
  }

  const std::vector<FusingNodePtr> &GetNodes() const {
    return nodes_;
  }

  // fusion_nodes_size_只记录原始图上Node用于控制融合规模，融合过程中创建出来的FusingNode计数不累加
  // lowering阶段已经融合的node要看子图里的node个数
  void AddFusionNodeSize(size_t node_size = 1U) {
    fusion_nodes_size_ += node_size;
  }

  const char_t *GetNamePtr() const {
    return node_->GetNamePtr();
  }

  int64_t GetMinOrder() const {
    return min_order_;
  }

  int64_t GetMaxOrder() const {
    return max_order_;
  }

  const std::set<NodePtr> &GetAncestors() const {
    return ancestors_;
  }

  std::set<NodePtr> &MutableAncestors() {
    return ancestors_;
  }

  size_t GetFusionNodesSize() const {
    // 这里返回的size和nodes_的size是有区别的，这里返回的是被融合掉的原始图上节点个数，nodes_里包含了中间节点FusingNode
    return fusion_nodes_size_;
  }

  const std::set<MemoryBuffer> &GetReadWrites() const {
    return read_writes_;
  }

  // 打印被融合的原始节点名
  void PrintOriginalNodes();

 private:
  void MergeNode(const FusingNodePtr &node);

 private:
  std::vector<FusingNodePtr> nodes_;
  std::set<MemoryBuffer> read_writes_;
  std::set<NodePtr> ancestors_;
  int64_t min_order_;
  int64_t max_order_;
  int64_t id_;
  const NodePtr node_;
  size_t fusion_nodes_size_;
};

// 融合后根据节省的内存搬运大小进行打分
class ScoreFusion {
 public:
  static bool Compare(const NodePair &pair1, const NodePair &pair2);
  static Expression ScoreFusionMemory(const FusingNode &node1, const FusingNode &node2);
};

/**
 * 策略融合器，基于各类策略综合求解，得到最优融合方案，并完成融合
 */
class FusionStrategySolver {
 public:
  explicit FusionStrategySolver(CounterPtr counter = nullptr) : counter_(counter) {}
  ~FusionStrategySolver() = default;
  Status Fuse(const ComputeGraphPtr &graph) const;

 private:
  CounterPtr counter_ = nullptr;
  Status FuseGraph(const ComputeGraphPtr &graph, const uint32_t max_fuse_rounds, const std::string &suffix) const;
  Status FuseNodesOnce(const uint32_t round, const ComputeGraphPtr &graph, const CycleDetectorSharedPtr &cycle_detector,
                       std::vector<FusingNodePtr> &nodes,
                       std::set<std::pair<FusingNode *, FusingNode *>> &can_not_fuse_nodes) const;

  NodePairList GetPossibleFusions(const uint32_t round, const ComputeGraphPtr &graph, std::vector<FusingNodePtr> &nodes,
                                  const FusingNodes &fusing_nodes,
                                  std::set<std::pair<FusingNode*, FusingNode*>> &can_not_fuse_nodes) const;
  NodePairList GetPossibleFusionsWithPrioritySort(const uint32_t round, const ComputeGraphPtr &graph,
                                                     const NodePairList &possible_fusions) const;
  bool CanFuse(const ComputeGraphPtr &graph, const FusingNodePtr &node1, const FusingNodePtr &node2,
               std::set<std::pair<FusingNode*, FusingNode*>> &can_not_fuse_nodes) const;
  bool WillFusionCreateCycle(const CycleDetectorSharedPtr &cycle_detector, const FusingNodePtr &node1,
                             const FusingNodePtr &node2) const;
  FusingNodePtr FuseNode(const ComputeGraphPtr &graph, const FusingNodePtr &node1, const FusingNodePtr &node2,
                         FusingNodes &fusing_nodes) const;
  bool CanFusionIncreasePeakMemory(const FusingNodePtr &node1, const FusingNodePtr &node2) const;
  const std::unique_ptr<FusionDecider> &GetBackEnd(const ComputeGraphPtr &graph) const;
  Status GetNodes(const ComputeGraphPtr &graph, std::vector<FusingNodePtr> &nodes) const;
  FusingNodePtr CreateFusingNode(const NodePtr &node) const;
  Status ComputeAncestors(std::vector<FusingNodePtr> &snodes) const;
  Status UpdateNodesAndTopoId(const ComputeGraphPtr &graph, const FusingNodes &fusing_nodes,
                              std::vector<FusingNodePtr> &nodes) const;
  static inline Status SortAndLogPossibleFusions(NodePairList &possible_fusions) {
    std::sort(possible_fusions.begin(), possible_fusions.end());
    GELOGI("found %d possible fusions.", possible_fusions.size());
    if (IsLogEnable(GE_MODULE_NAME, DLOG_INFO)) {
      for (const auto &pair : possible_fusions) {
        GELOGI("[%s,%s] memory_score:%s proximity_score:%ld", pair.first->GetNamePtr(),
               pair.second->GetNamePtr(), pair.memory_score.Str().get(), pair.proximity_score);
      }
    }
    return SUCCESS;
  }
  /*
   * 遍历graph中所有节点，打印并设置其中AscendBackend和FusedAscendBackend节点的输出与对应GE节点的输出的对应关系，并将这个对应关系设置到对应的
   * 属性中
   *
   * @param graph CanFuse融合后的计算图
   * @return 如果设置成功，返回SUCCESS；否则返回相应的错误码
   */
  Status PrintAndSetOriginInputAndOutput(const ComputeGraphPtr &graph) const;
  /*
   * 检查node1和node2融合后的输出内存是否超过阈值
   *
   * @param node1 节点1的指针
   * @param node2 节点2的指针
   * @return 如果node1和node2融合后的输出内存超过阈值，返回true；否则返回false
   */
  bool CheckWriteMemoryAfterFusion(const FusingNodePtr &node1, const FusingNodePtr &node2) const;
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_BACKEND_FUSION_STRATEGY_SOLVER_H_
