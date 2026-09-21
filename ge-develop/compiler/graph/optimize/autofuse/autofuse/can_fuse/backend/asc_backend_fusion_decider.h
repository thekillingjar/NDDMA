/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_ASC_BACKEND_FUSION_DECIDER_H_
#define AUTOFUSE_CAN_FUSE_BACKEND_ASC_BACKEND_FUSION_DECIDER_H_
#include "ge_common/ge_api_types.h"
#include "graph/compute_graph.h"
#include "fusion_decider.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "common/checker.h"
#include "backend_utils.h"
#include "asc_graph_axis_mapping.h"

namespace ge {
class AscBackendFusionDecider : public FusionDecider {
 public:
  AscBackendFusionDecider() = default;
  ~AscBackendFusionDecider() override = default;

  AscBackendFusionDecider(const AscBackendFusionDecider &) = delete;
  AscBackendFusionDecider &operator=(const AscBackendFusionDecider &) = delete;

  // 检查两个节点是否可以垂直融合
  bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) override {
    return CanFuse(node1, node2);
  }

  // 检查两个节点是否可以水平融合
  bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) override {
    return CanFuse(node1, node2);
  }

  /**
   * 该函数尝试将两个节点融合成一个新节点，如果两个节点都是 AscBackend 类型，则尝试进行循环合并
   * 如果无法进行循环合并，则创建一个FusedAscBackend子图来保存 AscBackend 节点的图结构
   *
   * @param node1 第一个节点指针
   * @param node2 第二个节点指针
   * @param counter 融合节点名命名计数指针
   * @return 融合后的节点指针，如果融合失败，则返回 nullptr
   */
  NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) override;

  // 获取融合对的优先级
  FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) override;

 protected:
  /**
   * 该函数检查两个子图是否满足合并条件,
   * 它检查子图的输入节点和输出节点的数量是否匹配，并且检查子图之间的节点是否具有相同的调度轴
   *
   * @param subgraph1 第一个ascgraph子图指针
   * @param subgraph2 第二个ascgraph子图指针
   * @param node1 第一个节点指针
   * @param node2 第二个节点指针
   * @return 如果可以合并，返回 SUCCESS；否则返回相应的错误码
   */
  Status CanMergeAscGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2, const NodePtr &node1,
                          const NodePtr &node2) const;

  /**
   * 该函数将两个节点（node1 和 node2）融合为一个新节点，并更新图中的数据和控制边
   *
   * @param node1 第一个节点指针
   * @param node2 第二个节点指针
   * @param merged_graph 子图融合后的图
   * @param counter 融合节点名命名计数指针
   * @return 如果融合成功，返回新节点的指针；否则返回 nullptr
   */
  NodePtr FuseNode(NodePtr node1, NodePtr node2, const ComputeGraphPtr merged_graph,
                   const NodeFuseInfo &node_fuse_info, const CounterPtr &counter) const;

  /**
   * 该函数将两个子图（subgraph1 和 subgraph2）根据循环进行合并, 合并后的子图将包含两个子图的所有节点和边
   *
   * @param subgraph1 第一个ascgraph子图指针
   * @param subgraph2 第二个ascgraph子图指针
   * @param node1 第一个节点指针
   * @param node2 第二个节点指针
   * @return 如果合并成功，返回合并后的子图指针；否则返回 nullptr
   */
  ComputeGraphPtr MergeAscGraphByLoop(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                                      const NodePtr &node1, const NodePtr &node2, NodeFuseInfo &node_fuse_info) const;

  /**
   * 该函数通过分析两个子图（subgraph1 和 subgraph2）的输入节点和输出节点，找出所有可能进行再次循环合并的节点对
   *
   * @param subgraph1 第一个子图指针
   * @param subgraph2 第二个子图指针
   * @param fuse_possibility_map 用于存储可能融合的节点对的集合
   * @return 如果成功获取所有可能融合的节点对，返回 SUCCESS；否则返回相应的错误状态
   */
  Status GetAllFusePossibilityNodes(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                                    std::set<std::pair<NodePtr, NodePtr>> &fuse_possibility_map,
                                    const NodeFuseInfo &node_fuse_info) const;

  /**
   * 该函数用于统一两个子图节点的轴信息，并判断它们是否可以融合。
   * 主要逻辑包括：
   * 1. 检查两个节点是否能够对齐轴信息。如果不能，则返回 FAILED。
   * 2. 刷新子图节点的轴信息，确保它们映射到统一的轴集合。
   * 3. 如果需要刷新轴信息（即 `need_flash` 为 true），则直接返回 SUCCESS。
   * 4. 如果不需要刷新轴信息，则进一步校验轴的大小是否一致：
   *    4.1 获取两个节点的融合子图。
   *    4.2 遍历子图的轴信息，检查映射后的轴 ID 是否一致。
   *    4.3 如果轴 ID 一致但大小不一致，则返回 FAILED，表示不能融合。
   * 5. 如果所有校验通过，则返回 SUCCESS。
   *
   * @param node1 第一个子图节点指针
   * @param node2 第二个子图节点指针
   * @param need_flash 是否需要刷新轴信息
   * @return 返回 SUCCESS 表示轴信息统一成功且可以融合，否则返回 FAILED
   */
  Status UnifySubgraphAxis(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info,
                           AscGraphAxisMapping &graph_axis_map, bool need_flash = true) const;

  /**
   * 该函数用于更新融合后的子图节点的轴属性。
   *
   * @param new_node 融合后的新子图节点指针
   * @param node1 第一个原始子图节点指针
   * @param node2 第二个原始子图节点指针
   * @return 返回 SUCCESS 表示轴属性更新成功，否则返回 FAILED
   */
  Status UpdateSubgraphAxisAttr(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2) const;

  /**
   * 该函数用于将融合后的轴组信息更新到融合节点属性中，以便在后续融合操作中直接使用该轴组信息。
   * 主要逻辑包括：
   * 1. 获取两个原始节点的轴组信息（axes_group1 和 axes_group2）。
   * 2. 检查两个轴组是否可以合并，并将合并后的轴组信息存储到 merged_axes_group 中。
   * 3. 将合并后的轴组信息更新到新融合节点的属性中。
   *
   * @param node1 第一个原始节点指针
   * @param node2 第二个原始节点指针
   * @param new_node 融合后的新节点指针
   * @return 返回 SUCCESS 表示轴组信息更新成功，否则返回 FAILED
   */
  Status UpdateAxisGroupInfo(const NodePtr &node1, const NodePtr &node2, const NodePtr &new_node,
                             const AscGraphAxisMapping &graph_axis_map) const;

  /**
   * 该函数将两个AscBackendNode子图（subgraph1 和
   * subgraph2）合并为一个融合的子图，合并后，函数还会尝试进一步融合可能循环合并
   *
   * @param subgraph1 第一个子图指针
   * @param subgraph2 第二个子图指针
   * @param node1 第一个节点指针（通常是 AscBackend 类型节点）
   * @param node2 第二个节点指针（通常是 AscBackend 类型节点）
   * @return 返回合并后的子图指针，如果合并失败则返回 nullptr
   */
  ComputeGraphPtr MergeGraphToFusedAscBackendNode(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                                                  const NodePtr &node1, const NodePtr &node2, const NodePtr &fused_node,
                                                  const NodeFuseInfo &node_fuse_info) const;

  Status UpdateFusedAscGraphAxisGroup(const NodePtr &node, const AxisPairSet &axis_map) const;

  /**
   * 设置can_fuse融合后的节点的输出与对应原图上节点的输出映射关系
   * 1.遍历融合后节点的所有输出
   * 2.通过node1/2_output_map找到对应AscNode的输出，比如:node1的node1_output_map是[1]，node2的node2_output_map是[-1, 0, 1]，这样
   * 表示融合后的节点的第0个输出对应node1的第1个输出；融合后的节点的第1个输出对应node2的第0个输出；融合后的节点的第2个输出对应node2的第1个输出
   * 3.通过AscNode的输出以及lowering阶段记录的AscNode的输出与原图节点的映射关系origin_output_names拿到对应原图节点的名字以及输出id
   * 4.设置can_fuse中的融合节点输出与原图中对应节点的输出对应关系
   *
   * @param new_node can_fuse融合后的节点
   * @param node1 需要合并的AscNode
   * @param node2 需要合并的AscNode
   * @param node_fuse_info 节点融合信息
   * @return 如果设置成功，返回SUCCESS；否则返回相应的错误码
   */
  static Status UpdateNewNodeAndGENodeOutputMappingRelation(const NodePtr &new_node, const NodePtr &node1,
                                                            const NodePtr &node2, const NodeFuseInfo &node_fuse_info);

  /**
   * 设置can_fuse融合后的节点的输入与对应原图上节点的输入映射关系
   * 1.遍历融合后节点的所有输入
   * 2.通过node1/2_input_map找到对应AscNode的输入，比如:node1的node1_input_map是[1]，node2的node2_input_map是[-1, 0, 1]，这样
   * 表示融合后的节点的第0个输入对应node1的第1个输入；融合后的节点的第1个输入对应node2的第0个输入；融合后的节点的第2个输入对应node2的第1个输入
   * 3.通过AscNode的输入以及lowering阶段记录的AscNode的输入与原图节点的映射关系origin_input_names拿到对应原图节点的名字以及输入id
   * 4.设置can_fuse中的融合节点输入与原图中对应节点的输入对应关系
   *
   * @param new_node can_fuse融合后的节点
   * @param node1 需要合并的AscNode
   * @param node2 需要合并的AscNode
   * @param node_fuse_info 节点融合信息
   * @return 如果设置成功，返回SUCCESS；否则返回相应的错误码
   */
  static Status UpdateNewNodeAndGENodeInputMappingRelation(const NodePtr &new_node, const NodePtr &node1,
                                                           const NodePtr &node2, const NodeFuseInfo &node_fuse_info);

  static Status GetNodeOriginInfo(const NodePtr &node, const uint32_t &new_node_in_or_out_data_anchor_size,
                                  const vector<int32_t> &node_input_or_output_map,
                                  std::vector<std::pair<std::string, int32_t>> &new_output_names, bool is_output);

  static Status SetReduceFusedElementwiseNodeNum(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2);

  static Status SetSplitNodeGlobalId(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2);

  static Status SetSplitNodeConcreteEdges(const NodePtr &new_node, const NodePtr &node1, const NodePtr &node2);

 private:
  /**
   * 判断两个节点是否可以融合
   *
   * @param node1 第一个节点指针
   * @param node2 第二个节点指针
   * @return 如果可以融合返回 true，否则返回 false
   */
  bool CanFuse(const NodePtr &node1, const NodePtr &node2) const override;

  /**
   * 该函数创建一个新的AscBackendNode子图，并将指定的 AscBackend 节点移动到该子图中,
   * 同时它会为输入和输出创建相应的数据节点和网络输出节点，并将它们与 AscBackend 节点连接起来
   *
   * @param node 要移动到子图中的 AscBackend 节点指针
   * @param in_nums 输入数据节点的数量
   * @param out_nums 输出数据节点的数量
   * @return 返回创建的子图指针，如果创建失败则返回 nullptr
   */
  ComputeGraphPtr CreateAscBackendNodeSubGraph(const NodePtr &node, uint32_t in_nums, uint32_t out_nums,
                                               const std::vector<uint32_t> &node_output_index,
                                               const std::vector<std::pair<ge::NodePtr, int32_t>> &pre_nodes) const;

  /**
   * 合并两个子图并生成一个新的合并后的子图, 把subgraph2 merge到subgraph1中
   *
   * @param subgraph1 第一个子图
   * @param subgraph2 第二个子图
   * @return 如果合并成功，返回SUCCESS；否则返回相应的错误码
   */
  Status MergeSubGraph(const ComputeGraphPtr &subgraph1, const ComputeGraphPtr &subgraph2,
                       const NodeFuseInfo &node_fuse_info) const;

  /**
   * 将子图的netoutput节点与输入节点进行连接，并记录需要删除的data节点
   *
   * @param subgraph_netoutput 子图的netoutput节点
   * @param inputs 子图的输入节点集合
   * @param subgraph_link_map 子图连接映射表，包含输入索引和netoutput索引的对应关系
   * @param del_data_nodes 存储需要删除的data节点的列表
   * @return 如果连接成功，返回SUCCESS；否则返回相应的错误码
   */
  Status LinkSubGraphNode(const NodePtr &subgraph_netoutput, const ComputeGraph::Vistor<NodePtr> &inputs,
                          const std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map,
                          std::vector<NodePtr> &del_data_nodes) const;

  /**
   * 该函数根据子图链接映射关系，将子图的输出节点（store_node）与输入节点（data_node）进行链接，并记录需要删除的节点
   *
   * @param node1 ascbackend
   * @param node2 ascbackend
   * @param outputs 子图的输出节点集合
   * @param inputs 子图的输入节点集合
   * @param subgraph_link_map 子图节点链接映射关系，包含输出节点和输入节点的索引对
   * @param del_data_nodes 存储需要删除的data节点的向量
   * @param del_output_and_store_nodes 存储需要删除的store、output节点的向量
   * @return 如果成功链接子图节点，返回SUCCESS；否则返回相应的错误码
   */
  Status LinkAscSubGraphNode(const NodePtr &node1, const NodePtr &node2, std::vector<NodePtr> &outputs,
                             const ComputeGraph::Vistor<NodePtr> &inputs, const NodeFuseInfo &node_fuse_info,
                             std::vector<NodePtr> &del_data_nodes, std::vector<NodePtr> &del_output_and_store_nodes,
                             bool is_reduction) const;

  /**
   * 该函数用于将节点输出链接到data节点的输出节点上
   *
   * @param in_anchor 目标输入锚点
   * @param data_node 要链接的数据节点
   * @return 如果链接成功，返回SUCCESS；否则返回相应的错误码
   */
  Status LinkDataNode(InDataAnchorPtr &in_anchor, const NodePtr &data_node, bool need_del_edge = true) const;

  /**
   * 该函数用于检查子图输出节点与输入节点之间的链接可能性，并构建链接映射关系
   *
   * @param subgraph_netoutput 子图输出节点
   * @param inputs 子图输入节点集合
   * @param link_map 链接映射对，指示子图输出节点与输入节点之间的链接关系
   * @param link_ascnode_map 链接映射集合，存储所有可能的链接节点对
   * @return 如果成功获取链接可能性，返回SUCCESS；否则返回相应的错误码
   */
  Status GetFusePossibilityLinkNodes(const NodePtr &subgraph_netoutput, const ComputeGraph::Vistor<NodePtr> &inputs,
                                     const std::pair<int32_t, int32_t> &link_map,
                                     std::set<std::pair<NodePtr, NodePtr>> &link_ascnode_map) const;

  /**
   * 该函数用于更新融合后的子图（subgraph）的执行顺序（exec_order）属性。
   * 主要逻辑包括：
   * 1. 根据融合类型（垂直融合或水平融合），更新子图节点的执行顺序：
   *    1.1 对于垂直融合，使用前序子图的最大执行顺序更新后序子图的执行顺序。
   *    1.2 对于水平融合，根据拓扑排序结果更新节点的执行顺序。
   *
   * @param subgraph 融合后的子图指针
   * @param subgraph1_nodes 第一个原始子图的节点集合
   * @param subgraph2_nodes 第二个原始子图的节点集合
   * @return 返回 SUCCESS 表示执行顺序更新成功，否则返回 FAILED
   */
  Status UpdateAscGraph(const ComputeGraphPtr &subgraph, const ComputeGraph::Vistor<NodePtr> &subgraph1_nodes,
                        const ComputeGraph::Vistor<NodePtr> &subgraph2_nodes, const NodeFuseInfo &node_fuse_info) const;

  /**
   * 该函数用于更新融合后的新节点的属性，将两个原始节点的属性信息合并到新节点的属性中。
   *
   * @param op 新节点的 OpDesc 指针，用于获取或创建新节点的属性
   * @param node1 第一个原始节点的指针
   * @param node2 第二个原始节点的指针
   * @return 返回 SUCCESS 表示属性更新成功，否则返回 FAILED
   */
  Status UpdateNewNodeAttr(const OpDescPtr op, const NodePtr &node1, const NodePtr &node2,
                           const NodeFuseInfo &node_fuse_info) const;
  /**
   * 该函数用于更新FusedAscBackend点中输入节点的index值
   *
   * @param merged_subgraph 融合后的子图指针
   * @return 返回 SUCCESS 表示属性更新成功，否则返回 FAILED
   */
  Status AddIndexForFusedAscBackendNode(const ComputeGraphPtr &merged_subgraph) const;

  static Status GetPossibleFuseNodePairFromSubgraph(std::set<std::pair<NodePtr, NodePtr>> &fuse_possibility_map,
                                                    const ComputeGraphPtr &subgraph);
};

// fusedAscBackendNode下的子图融合策略，canfuse条件是是否能循环合并
class AscBackendSubGraphFusionDecider : public AscBackendFusionDecider {
 public:
  AscBackendSubGraphFusionDecider() = default;

  ~AscBackendSubGraphFusionDecider() override = default;

  AscBackendSubGraphFusionDecider(const AscBackendSubGraphFusionDecider &) = delete;
  AscBackendSubGraphFusionDecider &operator=(const AscBackendSubGraphFusionDecider &) = delete;

  // 检查两个节点是否可以垂直融合
  bool CanFuseVertical(const NodePtr &node1, const NodePtr &node2) override {
    // 检查两个节点是否满足融合条件
    return CanFuse(node1, node2);
  }

  // 检查两个节点是否可以水平融合
  bool CanFuseHorizontal(const NodePtr &node1, const NodePtr &node2) override {
    return CanFuse(node1, node2);
  }

  // 融合两个节点
  NodePtr Fuse(const NodePtr &node1, const NodePtr &node2, const CounterPtr &counter) override;

  // 获取融合对的优先级
  FusionPriority GetFusionPairPriority(const NodePtr &node1, const NodePtr &node2) override {
    (void)node1;
    (void)node2;
    return FusionPriority::HIGH;
  }

 private:
  bool CanFuse(const NodePtr &node1, const NodePtr &node2) const override;
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_BACKEND_ASC_BACKEND_FUSION_DECIDER_H_
