/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_MANAGER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_MANAGER_H_

#include "common/fe_inner_error_codes.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/trans_node_insertion.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_merging/trans_node_merging.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
namespace fe {
using TransNodeMergingPtr = std::shared_ptr<TransNodeMerging>;
using TransNodeInsertionPtr = std::shared_ptr<TransNodeInsertion>;

/** @brief This class manage all operations of trans-nodes including insertion
* and merging.
* @version 1.0 */
class TransNodeManager {
 public:
  explicit TransNodeManager(FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr);

  ~TransNodeManager();

  TransNodeManager(const TransNodeManager &) = delete;

  TransNodeManager &operator=(const TransNodeManager &) = delete;

  Status Initialize();

  /* This is interface for graph optimizer. In this function we  */
  Status InsertAndMergeTransNodes(ge::ComputeGraph &graph);

  const std::vector<ge::NodePtr> &GetOptimizableCast();

  Status InsertTransNodes(ge::ComputeGraph &graph, ge::NodePtr node);

  Status MergeAllTransOps(ge::ComputeGraph &graph);

 private:
  TransNodeMergingPtr trans_node_merging_ptr_;

  TransNodeInsertionPtr trans_node_insertion_ptr_;

  FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr_;

  std::vector<ge::NodePtr> origin_cast_list_;
};
}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_MANAGER_TRANS_NODE_MANAGER_H_
