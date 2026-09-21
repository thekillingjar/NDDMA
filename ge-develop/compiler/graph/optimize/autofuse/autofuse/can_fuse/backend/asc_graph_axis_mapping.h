/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AUTOFUSE_CAN_FUSE_BACKEND_ASC_GRAPH_AXIS_MAPPING_H_
#define AUTOFUSE_CAN_FUSE_BACKEND_ASC_GRAPH_AXIS_MAPPING_H_
#include "backend_utils.h"
#include "graph/utils/type_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/utils/graph_utils.h"

#define GELOGD_IF(condition, fmt, ...)                                                                             \
  do {                                                                                                             \
    if (condition) {                                                                                               \
      dlog_debug(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__);             \
    }                                                                                                              \
  } while (false)

#define GELOGI_IF(condition, fmt, ...)                                                                             \
  do {                                                                                                             \
    if (condition) {                                                                                               \
      dlog_info(GE_MODULE_NAME, "%" PRIu64 " %s:" fmt, GeLog::GetTid(), __FUNCTION__, ##__VA_ARGS__);              \
    }                                                                                                              \
  } while (false)

namespace ge {
struct NodeFuseInfo {
  explicit NodeFuseInfo(bool open_log = true)
      : node1_in_data_size_(0U),
        node2_in_data_size_(0U),
        node1_out_data_size_(0U),
        node2_out_data_size_(0U),
        node1_out_node_size_(0U),
        node2_out_node_size_(0U),
        open_log_(open_log) {}

  Status UpdateNodeFuseInfo(const NodePtr &node1, const NodePtr &node2);
  const std::vector<int32_t> &GetNode1InputMap() const {
    return node1_input_map_;
  }
  const std::vector<int32_t> &GetNode2InputMap() const {
    return node2_input_map_;
  }
  const std::vector<int32_t> &GetNode1OutputMap() const {
    return node1_output_map_;
  }
  const std::vector<int32_t> &GetNode2OutputMap() const {
    return node2_output_map_;
  }
  const std::vector<std::pair<int32_t, int32_t>> &GetSameInputMap() const {
    return same_input_map_;
  }
  const std::vector<std::pair<int32_t, int32_t>> &GetNode1ToNode2LinkMap() const {
    return node1_to_node2_link_map_;
  }
  const std::vector<uint32_t> &GetNode1OutputIndex() const {
    return node1_output_index_;
  }
  const std::vector<uint32_t> &GetNode2OutputIndex() const {
    return node2_output_index_;
  }
  uint32_t GetNode1InDataSize() const {
    return node1_in_data_size_;
  }
  uint32_t GetNode2InDataSize() const {
    return node2_in_data_size_;
  }
  uint32_t GetNode1OutNodeSize() const {
    return node1_out_node_size_;
  }
  uint32_t GetNode2OutNodeSize() const {
    return node2_out_node_size_;
  }
  bool GetHasSliceVertical() const {
    return has_slice_vertical_;
  }
  bool CanDoHorizontalMapping() const {
    return can_do_horizontal_mapping_ && (!has_slice_vertical_);
  }
  bool IsSingleReference(int32_t index) const {
    auto it = std::find_if(is_single_reference_.begin(), is_single_reference_.end(),
                           [&index](const std::pair<int32_t, bool> &p) { return p.first == index; });
    if (it != is_single_reference_.end()) {
      return it->second;
    }
    return true;
  }

  bool HasMulReference() const {
    auto it = std::find_if(is_single_reference_.begin(), is_single_reference_.end(),
                           [](const std::pair<int32_t, bool> &p) { return !p.second; });
    return it != is_single_reference_.end();
  }

  /**
   * 该函数用于回滚节点的输入映射信息。
   * 主要逻辑包括：
   * 1. 通过二分查找找到需要回滚的输入数据在映射向量中的插入位置。
   * 2. 将输入数据插入到映射向量中，恢复原始的输入映射关系。
   *
   * @param data_input 需要回滚的输入数据索引
   * @return 无返回值
   */
  void RollbackNode2InputMap(int32_t data_input);

  std::vector<std::pair<ge::NodePtr, int32_t>> GetNode1PreNodes() const {
    return node1_pre_nodes_;
  }

  std::vector<std::pair<ge::NodePtr, int32_t>> GetNode2PreNodes() const {
    return node2_pre_nodes_;
  }

 private:
  void Clear() {
    node1_input_map_.clear();
    node2_input_map_.clear();
    node1_output_map_.clear();
    node2_output_map_.clear();
    same_input_map_.clear();
    node1_to_node2_link_map_.clear();
    node1_in_data_size_ = 0U;
    node2_in_data_size_ = 0U;
    node1_out_data_size_ = 0U;
    node2_out_data_size_ = 0U;
    node1_out_node_size_ = 0U;
    node2_out_node_size_ = 0U;
    is_single_reference_.clear();
    node1_output_index_.clear();
    node2_output_index_.clear();
    node1_pre_nodes_.clear();
    node2_pre_nodes_.clear();
    has_slice_vertical_ = false;
    can_do_horizontal_mapping_ = true;
  }

  /**
   * 获取两个节点中具有相同输入锚点的索引对
   *
   * @param node1 第一个节点
   * @param node2 第二个节点
   * @param same_input_map 存储具有相同输入锚点的索引对的向量
   * @return 如果成功获取相同输入索引对，返回SUCCESS；否则返回相应的错误码
   */
  Status GetSubgraphSameInputIndex(const NodePtr &node1, const NodePtr &node2,
                                   std::vector<std::pair<int32_t, int32_t>> &same_input_map) const;

  /**
   * 获取两个节点之间的连接索引对
   *
   * @param node1 第一个节点
   * @param node2 第二个节点
   * @param subgraph_link_map 存储连接索引对的向量
   * @return 如果成功获取连接索引对，返回SUCCESS；否则返回相应的错误码
   */
  Status GetSubgraphLinkIndex(const NodePtr &node1, const NodePtr &node2,
                              std::vector<std::pair<int32_t, int32_t>> &subgraph_link_map);

  /**
   * 获取两个节点的输入映射关系，并生成两个输入映射向量，具体map含义参考ReplaceNodeDataAnchors接口
   * 排列方式是node1在左，node2在右
   *
   * @param node1 第一个节点
   * @param node2 第二个节点
   * @param node1_input_map 存储node1的有效输入索引的向量
   * @param node2_input_map 存储node2的有效输入索引的向量
   * @return 如果成功生成输入映射关系，返回SUCCESS；否则返回相应的错误码
   */
  Status GetFuseNodeInput(const NodePtr &node1, const NodePtr &node2, std::vector<int32_t> &node1_input_map,
                          std::vector<int32_t> &node2_input_map) const;

  /**
   * 获取两个节点的输出映射关系，并生成两个输出映射向量，具体map含义参考ReplaceNodeDataAnchors接口
   * 排列方式是node1在左，node2在右
   *
   * @param node1 第一个节点
   * @param node2 第二个节点
   * @param node1_output_map 存储node1的有效输出索引的向量
   * @param node2_output_map 存储node2的有效输出索引的向量
   * @return 如果成功生成输出映射关系，返回SUCCESS；否则返回相应的错误码
   */
  Status GetFuseNodeOutput(const NodePtr &node1, const NodePtr &node2, std::vector<int32_t> &node1_output_map,
                           std::vector<int32_t> &node2_output_map) const;

  /**
   * 该函数用于获取两个节点之间的输出映射关系。
   * 主要逻辑包括：
   * 1. 遍历第一个节点的所有输出锚点。
   * 2. 对于每个输出锚点，检查其对应的下游节点是否为第二个节点。
   * 3. 如果第一个节点的某个输出只连接到第二个节点，则将该输出的映射值设置为 -1，表示该输出被完全融合。
   * 4. 否则，将映射值设置为 0，表示该输出未被融合。
   *
   * @param node1 第一个节点的指针
   * @param node2 第二个节点的指针
   * @param node_output_map 输出映射关系的向量，用于存储结果
   * @return 返回 SUCCESS 表示操作成功，否则返回 FAILED
   */
  Status GetNodeOutputMap(const NodePtr &node1, const NodePtr &node2, std::vector<int32_t> &node_output_map) const;

  /**
   * 该函数用于获取节点的输出索引信息。
   * 主要逻辑包括：
   * 1. 遍历节点的所有输出锚点。
   * 2. 对于每个输出锚点，检查其连接的下游节点的数量。
   * 3. 将每个输出锚点的索引值存储到结果向量中，索引值的数量与下游节点的数量一致。
   *
   * @param node 目标节点的指针
   * @param node_out_node_size 输出节点的总数量
   * @param node_out_data_size 输出数据的总数量
   * @param node_output_index 输出索引信息的向量，用于存储结果
   * @return 返回 SUCCESS 表示操作成功，否则返回 FAILED
   */
  Status GetNodeOutputIndex(const NodePtr &node, uint32_t &node_out_node_size, const uint32_t node_out_data_size,
                            std::vector<uint32_t> &node_output_index) const;

  /**
   * 该函数用于获取node的前序节点信息
   * 主要逻辑包括：
   * 1. 遍历node的输入anchor，获取连接它的上游输出anchor，然后获取输出anchor对应的节点
   * 2. 将对应节点以及输出anchor的index添加到pre_nodes里
   *
   * @param node 数据节点指针
   * @param in_nums node输入anchor的大小
   * @param pre_nodes node的前序节点信息
   * @return 返回 SUCCESS表示操作成功，否则返回 FAILED
   */
  Status GetNodePreInfo(const NodePtr &node, const uint32_t &in_nums,
                        std::vector<std::pair<ge::NodePtr, int32_t>> &pre_nodes) const;

 private:
  std::vector<int32_t> node1_input_map_;
  std::vector<int32_t> node2_input_map_;
  std::vector<int32_t> node1_output_map_;
  std::vector<int32_t> node2_output_map_;
  std::vector<std::pair<int32_t, int32_t>> same_input_map_;
  std::vector<std::pair<int32_t, int32_t>> node1_to_node2_link_map_;
  std::vector<uint32_t> node1_output_index_;
  std::vector<uint32_t> node2_output_index_;
  uint32_t node1_in_data_size_;
  uint32_t node2_in_data_size_;
  uint32_t node1_out_data_size_;
  uint32_t node2_out_data_size_;
  uint32_t node1_out_node_size_;
  uint32_t node2_out_node_size_;
  std::vector<std::pair<int32_t, bool>> is_single_reference_;
  std::vector<std::pair<ge::NodePtr, int32_t>> node1_pre_nodes_;
  std::vector<std::pair<ge::NodePtr, int32_t>> node2_pre_nodes_;
  bool has_slice_vertical_{false};  // slice节点需要判断初始状态是否同时可以水平融合和垂直融合
  bool can_do_horizontal_mapping_{true}; // matmul节点同时有水平融合和垂直融合时不做水平融合轴映射为false，其他节点默认true
  bool open_log_;  // 是否打印轴映射日志，true表示打印，false表示不打印
};

class AscGraphAxisMapping {
 public:
  struct AscNodeAxisInfo {
    std::vector<int64_t> node_axis;
    std::vector<ge::Expression> node_repeats;
    std::vector<int64_t> graph_axis;
    std::vector<ge::Expression> graph_size;
  };

  explicit AscGraphAxisMapping(bool open_log = true): open_log_(open_log) {}

  /**
   * 该函数用于创建两个节点之间的子图轴映射信息。它首先更新节点的融合信息，
   * 然后分别进行水平融合轴映射和垂直融合轴映射，确保两个节点的轴信息可以正确映射。
   * 最后，它会将轴 ID 刷新为连续的 ID，并记录最终的轴映射信息。
   *
   * @param node1 第一个节点指针（输入参数），表示需要进行轴映射的第一个节点。
   * @param node2 第二个节点指针（输入参数），表示需要进行轴映射的第二个节点。
   * @param fuse_info node1和node2的链接信息。
   * @return 如果轴映射信息创建成功，则返回 SUCCESS；否则返回 FAILED。
   */
  Status CreateSubGraphAxisMapInfo(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info);

  const AxisPairSet &GetNode1AxisMap() const {
    return node1_map_;
  }

  const AxisPairSet &GetNode2AxisMap() const {
    return node2_map_;
  }

  /**
   * 该函数用于刷新指定节点的子图轴信息。它会根据节点的类型（kAscBackendType或kFusedAscBackendType）
   * 调用相应的刷新函数来更新子图中的轴信息。
   *
   * @param node 需要刷新子图轴信息的节点指针
   * @param node_map 轴映射关系，用于将旧的轴信息映射到新的轴信息
   * @param need_flash 是否需要刷新轴信息，如果为true，则更新轴信息；如果为false，则不更新
   * @return 返回SUCCESS表示刷新成功，返回FAILED表示刷新失败
   */
  Status FlushSubGraphAxisInfo(const NodePtr &node, const AxisPairSet &node_map, bool need_flash);

  /**
   * 该函数用于将两个节点的轴映射信息调整为连续的轴ID，以便后端处理。
   *
   * @return 返回 SUCCESS 表示成功调整轴映射信息
   */
  Status FlashContinueAxisId();

  bool IsOpenLog() const {
    return open_log_;
  }

 private:
  /**
   * 该函数用于获取指定节点的前置节点的属性信息，包括维度、轴和大小。。
   *
   * @param node 当前节点的指针
   * @param index 当前节点的输入锚点的索引
   * @param dims 输出参数，用于存储前置节点的维度信息
   * @param axis 输出参数，用于存储前置节点的轴信息
   * @param repeats 输出参数，用于存储前置节点的重复次数信息
   * @return 返回状态，成功时返回 SUCCESS，失败时返回相应的错误状态
   */
  Status GetPreNodeAttrs(const NodePtr &node, const int32_t index, std::vector<ge::Expression> &dims,
                         std::vector<int64_t> &axis, std::vector<ge::Expression> &repeats) const;

  /**
   * 该函数用于获取当前节点的属性信息，包括当前节点的轴信息和大小。
   *
   * @param node 需要获取属性的节点指针
   * @param index 输入的索引
   * @param node_cur_axis 存储当前节点的轴信息
   * @param node_cur_repeats 存储当前节点的重复次数
   * @return 返回SUCCESS表示获取成功，返回FAILED表示获取失败
   */
  Status GetCurNodeAttrs(const NodePtr &node, const int32_t index, std::vector<int64_t> &axis,
                         std::vector<ge::Expression> &repeats) const;

  /**
   * 该函数用于找到节点（node_repeats）在基准信息（base_repeats）中的索引。
   * 它会遍历node_repeats中的每个元素，并在base_repeats中查找匹配的元素，记录其索引。
   * 如果所有元素都能找到匹配的索引，则返回SUCCESS；否则返回FAILED。
   *
   * @param node_repeats 要查找索引的节点重复信息
   * @param base_repeats 基准重复信息
   * @param axis_index 用于存储匹配索引的向量
   * @return 如果成功找到所有索引，则返回SUCCESS；否则返回FAILED
   */
  Status FindAxisIndex(std::vector<ge::Expression> &node_repeats, std::vector<ge::Expression> &base_repeats,
                       std::vector<uint32_t> &axis_index) const;

  /**
   * 该函数用于获取节点及其前驱节点的轴映射信息，并检查是否可以进行轴映射。
   * 轴映射的目的是确保两个图（或节点）的调度轴信息能够匹配，从而支持后续的图融合操作。
   *
   * @param node 当前节点的指针，用于获取节点的属性信息。
   * @param index 当前节点的索引，用于区分不同的输入或输出。
   * @param node1_map 存储第一个图的轴映射信息，格式为 {原始轴ID, 映射后的轴ID}。
   * @param node2_map 存储第二个图的轴映射信息，格式为 {原始轴ID, 映射后的轴ID}。
   * @param asc_node1 实际链接的两个ascback node的前序node。
   * @param asc_node2 实际链接的两个ascback node的后序node。
   * @return 如果轴映射成功，返回 SUCCESS；否则返回 FAILED。
   */
  Status GetVerticalAxisMapInfo(const NodePtr &node, const int32_t index, AxisPairSet &node1_map,
                                AxisPairSet &node2_map, NodePtr &asc_node1, NodePtr &asc_node2);

  /**
   * 该函数用于刷新子图中的轴信息。它会遍历子图中的所有节点，并根据给定的轴映射关系（node_map）
   * 更新节点的轴信息。如果找不到映射关系，则会根据已有最大轴ID递增的方式补充一个映射关系。
   *
   * @param graph 需要刷新轴信息的子图指针
   * @param node_map 轴映射关系，用于将旧的轴信息映射到新的轴信息
   * @param need_flash 是否需要刷新轴信息，如果为true，则更新轴信息；如果为false，则不更新
   * @return 返回SUCCESS表示刷新成功，返回FAILED表示刷新失败
   */
  Status FlushAscSubGraphAxisInfo(const ComputeGraphPtr &graph, const AxisPairSet &node_map, bool need_flash) const;

  /**
   * 该函数用于获取当前 Asc 节点的属性信息。
   * 主要逻辑包括：
   * 1. 获取父节点所属的融合子图。
   * 2. 从子图的输入节点列表中获取指定索引的 Data 节点。
   * 3. 从 Data 节点的 OpDesc 中获取节点属性组。
   * 4. 从 Data 节点的输出描述中获取张量属性组。
   * 5. 提取并返回 axis 和 repeats 属性。
   *
   * @param parent_node 父节点的指针
   * @param index 目标 Data 节点的索引
   * @param axis 存储 axis 属性的向量（引用传递）
   * @param repeats 存储 repeats 属性的向量（引用传递）
   * @return 返回 SUCCESS 表示操作成功，否则返回 FAILED
   */
  Status GetCurAscNodeAttrs(const NodePtr &parent_node, const int32_t index, std::vector<int64_t> &axis,
                            std::vector<ge::Expression> &repeats) const;

  /**
   * 该函数用于获取当前节点的AscGraph属性，包括轴（axis）和大小（size）信息。
   * 如果节点的类型是kAscBackendType或kFusedAscBackendType，则从其子图中获取相应的属性信息，并填充到传入的参数中。
   *
   * @param node 要获取属性的节点指针
   * @param index 输入锚点的索引
   * @param axis 用于存储轴信息的向量
   * @param size 用于存储大小信息的向量
   * @param ascback_node 链接的ascback节点
   * @return 如果成功获取属性，则返回SUCCESS；否则返回FAILED
   */
  Status GetCurNodeAscGraphAttrs(const NodePtr &node, const int32_t index, std::vector<int64_t> &axis,
                                 std::vector<ge::Expression> &size, NodePtr &ascback_node);
  /**
   * 该函数用于判断两个轴映射关系是否相同。
   *
   * @param node1_map 第一个轴映射关系
   * @param node2_map 第二个轴映射关系
   * @return 返回true表示两个映射关系相同，返回false表示不同
   */
  bool IsSameMapAxis(AxisPairSet &map1, AxisPairSet &map2) const;

  /**
   * 该函数用于判断两个节点的轴信息是否可以进行映射。
   * 它会根据两个节点的信息（repeats）来确定轴的映射关系，并将映射结果存储到临时映射集合中。
   * 如果两个节点的重复信息可以完全匹配，则返回true；否则返回false。
   *
   * @param node1_axis 第一个节点的轴信息
   * @param node1_repeats 第一个节点的重复信息
   * @param node2_axis 第二个节点的轴信息
   * @param node2_repeats 第二个节点的重复信息
   * @param node1_map 第一个节点的已有轴映射集合
   * @param node2_map 第二个节点的已有轴映射集合
   * @param temp_node1_map 用于存储第一个节点临时轴映射的集合
   * @param temp_node2_map 用于存储第二个节点临时轴映射的集合
   * @return 如果两个节点的轴信息可以进行映射，则返回true；否则返回false
   */
  bool CanAxisMap(std::vector<int64_t> &node1_axis, std::vector<ge::Expression> &node1_repeats,
                  std::vector<int64_t> &node2_axis, std::vector<ge::Expression> &node2_repeats, AxisPairSet &node1_map,
                  AxisPairSet &node2_map, AxisPairSet &temp_node1_map, AxisPairSet &temp_node2_map) const;

  /**
   * 该函数用于获取指定节点的前置节点的子图属性信息，包括轴和尺寸信息。
   *
   * @param node 当前节点的指针
   * @param index 当前节点的输入锚点的索引
   * @param axis 输出参数，用于存储前置节点的轴信息
   * @param size 输出参数，用于存储前置节点的尺寸信息
   * @param ascback_node 链接的ascback node
   * @return 返回状态，成功时返回 SUCCESS，失败时返回相应的错误状态
   */
  Status GetPreNodeAscGraphAttrs(const NodePtr &node, const int32_t index, std::vector<int64_t> &axis,
                                 std::vector<ge::Expression> &size, NodePtr &ascback_node) const;

  /**
   * 该函数用于判断两个节点是否可以进行循环合并。
   *
   * @param node1 第一个节点指针（输入参数）
   * @param node2 第二个节点指针（输入参数）
   * @return 如果两个节点可以进行循环合并，则返回 true；否则返回 false。
   */
  bool CanLoopMerge(const NodePtr &node1, const NodePtr &node2);

  /**
   * 该函数用于获取两个子图的水平映射信息，并判断它们是否可以进行轴映射。
   * 它会分别获取两个节点的当前轴信息和图形轴信息，并调用 CanAxisMap 函数判断轴信息是否可以映射。
   * 如果轴映射成功，则进一步判断是否可以进行循环合并。
   *
   * @param node1 第一个节点指针（输入参数）
   * @param node2 第二个节点指针（输入参数）
   * @param fuse_info node1和node2的链接信息。
   * @return 如果轴映射成功且可以进行循环合并，则返回 SUCCESS；否则返回 FAILED。
   */
  Status ProcessSubGraphHorizontalMapInfo(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info);

  /**
   * 该函数用于获取子图的垂直轴映射信息。它会根据给定的节点和链接映射关系，
   * 生成两个节点之间的轴映射信息，并检查是否可以进行循环合并操作。
   * 如果节点之间存在归约操作，还会进一步判断是否满足融合条件。
   *
   * @param node1 第一个节点的指针（输入参数），表示需要进行轴映射的第一个节点。
   * @param node2 第二个节点的指针（输入参数），表示需要进行轴映射的第二个节点。
   * @param fuse_info node1和node2的链接信息。
   * @return 如果轴映射信息获取成功且满足融合条件，则返回 SUCCESS；否则返回 FAILED。
   */
  Status ProcessSubGraphVerticalMapInfo(const NodePtr &node1, const NodePtr &node2, const NodeFuseInfo &fuse_info,
                                        AxisPairSet &node1_map, AxisPairSet &node2_map);

  /**
   * 该函数用于检查两个子图的水平轴映射关系，确保它们的节点轴和调度轴能够正确映射。
   * 主要逻辑包括：
   * 1. 检查两个节点的数据轴是否能够直接映射。
   * 2. 检查两个节点的调度轴是否能够映射。
   * 3. 确保映射关系一致性，避免冲突。
   *
   * @param node1_cur_info 第一个节点的轴信息，包含数据轴和调度轴等信息。
   * @param node2_cur_info 第二个节点的轴信息，包含数据轴和调度轴等信息。
   * @return 返回 SUCCESS 表示映射检查成功，返回 FAILED 表示映射失败，无法进行后续融合操作。
   */
  Status CheckSubGraphHorizontalAxisMapping(const NodePtr &node1, const NodePtr &node2, AscNodeAxisInfo &node1_cur_info,
                                            AscNodeAxisInfo &node2_cur_info);

  /**
   * 该函数用于检查两个子图的垂直轴映射关系，确保前序节点的存储轴与当前节点的数据轴或调度轴能够正确映射。
   * 主要逻辑包括：
   * 1. 检查前序节点的存储轴与当前节点的调度轴是否能够直接映射。
   * 2. 如果第一步失败，检查前序节点的存储轴与当前节点的数据轴是否能够映射。
   * 3. 确保所有映射关系一致性，避免冲突。
   *
   * @param pre_axis_info 前序节点的轴信息，包含存储轴和调度轴等信息。
   * @param cur_axis_info 当前节点的轴信息，包含数据轴和调度轴等信息。
   * @return 返回 SUCCESS 表示映射检查成功，返回 FAILED 表示映射失败，无法进行后续融合操作。
   */
  Status CheckSubGraphtVerticalAxisMapping(const NodePtr &node, AscNodeAxisInfo &pre_axis_info,
                                           AscNodeAxisInfo &cur_axis_info, AxisPairSet &node1_map,
                                           AxisPairSet &node2_map);

  /**
   * 该函数用于检查或填充轴映射信息
   * @param node1_map 第一个节点的已有轴映射集合
   * @param node2_map 第二个节点的已有轴映射集合
   * @param temp_node1_map 临时变量，用于检查或者填充node1_map
   * @param temp_node2_map 临时变量，用于检查或者填充node2_map
   * @return 表示映射检查成功，返回 FAILED 表示映射失败，无法进行后续融合操作。
   */
  Status CheckAndFillAxisMap(AxisPairSet &node1_map, AxisPairSet &node2_map, AxisPairSet &temp_node1_map,
                             AxisPairSet &temp_node2_map) const;

 private:
  AxisPairSet node1_map_;
  AxisPairSet node2_map_;
  bool pre_node_is_reduction_{false};
  bool open_log_;  // 是否打印轴映射日志，true表示打印，false表示不打印
};
}  // namespace ge

#endif  // AUTOFUSE_CAN_FUSE_BACKEND_ASC_GRAPH_AXIS_MAPPING_H_
