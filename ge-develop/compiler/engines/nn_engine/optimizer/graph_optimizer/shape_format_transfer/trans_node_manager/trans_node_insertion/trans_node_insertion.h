/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_INSERTION_TRANS_NODE_INSERTION_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_INSERTION_TRANS_NODE_INSERTION_H_

#include <map>
#include <memory>
#include <vector>
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"

namespace fe {
enum class ConcecutivePrinciple {
  kConcecutive = 0,
  kOriFormatUnConcecutive,
  kOriShapeUnConcecutive,
  kShapeUnConcecutive,
  kRelaFormatShapeUnConcecutive,
  kReShapeTypeUnConcecutive,
  kDumpableUnConcecutive,
  kConflictFormat
};

/** @brief class of inserting all necessary trans-nodes into graph
* by specific strategy according to the format and data type
* @version 1.0 */
class TransNodeInsertion {
 public:
  explicit TransNodeInsertion(FEOpsKernelInfoStorePtr fe_ops_store_ptr);

  ~TransNodeInsertion();

  TransNodeInsertion(const TransNodeInsertion &) = delete;

  TransNodeInsertion &operator=(const TransNodeInsertion &) = delete;

  Status initialize();
  /* Insert all trans-nodes by strategy ID */
  Status InsertTransNodes(ge::ComputeGraph &fused_graph, ge::NodePtr dst_node);

 private:
  void InitReshapeType();

  void GetSrcReshapeType(const OpKernelInfoPtr &op_kernel);
  void GetDstReshapeType(const OpKernelInfoPtr &op_kernel);

  Status GetReshapeType();

  Status AddTransNodeType(TransNodeBaseGenerator *trans_node_type);

  Status CombineAllStrategy(TransInfoPtr trans_info_ptr, uint64_t global_strategy_id,
                            std::vector<std::vector<uint32_t>> &strategy_vector_combination);

  Status GenerateStrategy(std::vector<std::vector<uint32_t>> &strategy_vector_combination);

  Status GenerateStrategyByOrignalFormat(std::vector<std::vector<uint32_t>> &strategy_vector_combination);
  /* position_flag = 0 means from father node to current node
   * or from original format to current format (That means
   * insert trans node before current node.);
   * position_flag = 1 means from current format to
   * original format (That means
   * insert trans node after current node.)
   */
  Status InsertTransOpByConcecutiveStrategy(ge::ComputeGraph &fused_graph);

  /* For node A, insert trans-nodes in front of A, form A's input's original
   * format to A's current format and at the end of A, from A's output's
   * original format to it's */
  Status InsertTransOpByOriginalFormat(ge::ComputeGraph &fused_graph);

  FEOpsKernelInfoStorePtr fe_ops_store_info_ptr_;

  /* Key is Strategy Id, it's a 32 bits integer defines as :
   * bit0 ~ bit7 new format;
   * bit8 ~ bit15 original format;
   * bit16 ~ bit23 dst dtype
   * bit24 ~ bit31 source dtype
   *
   * StrategyIdMap is a map of strategy_id and a list of trans nodes which
   * will be inserted into graph.
   * The outside vector represents multiple insertion mode of enum
   * InsertionMode */
  strategy_id_map trans_nodes_insertion_strategy_;

  std::vector<TransNodeBaseGenerator *> whole_trans_nodes_vector_ = {};
  void SetTransInfoForInsertionModeFront();

  void SetTransInfoForInsertionModeEnd();

  Status FillTransInfo(const ge::InDataAnchorPtr &dst_anchor, const ge::OutDataAnchorPtr &src_anchor,
                     const ge::NodePtr &src_node, const ge::NodePtr &dst_node,
                     ConcecutivePrinciple &use_concecutive_principle);

  static Status InsertCastGeneralCase(const TransInfoPtr trans_info_ptr, const uint32_t front_strategy_vector_index,
                                      const uint32_t end_strategy_vector_index,
                                      std::vector<std::vector<uint32_t>> &strategy_vector_combination);

  static bool IsInsertCastFirst(const TransInfoPtr trans_info_ptr);

  TransInfoPtr global_trans_info_ptr_;
  ConcecutivePrinciple IsAbleToUseConcecutivePrinciple();
  /* This function is used when we successfully inserted from src node's
   * current format to its original format. We need to update the second
   * element of trans_info_front_and_end_ because the src-node would be
   * different */
  Status UpdateNextTransInfo(uint32_t end_strategy_index);
  /* These two pointers below is for insertion mode
   * INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_FRONT and
   * INSERT_TRANS_NODE_BY_ORIGINAL_FORMAT_END*/
  std::vector<TransInfoPtr> trans_info_front_and_end_;
};

}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_INSERTION_TRANS_NODE_INSERTION_H_
