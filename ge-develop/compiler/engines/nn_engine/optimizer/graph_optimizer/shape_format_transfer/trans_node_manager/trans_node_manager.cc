/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_manager.h"
#include <memory>
#include <string>
#include "common/fe_utils.h"
#include "common/graph/fe_graph_utils.h"

namespace fe {

TransNodeManager::TransNodeManager(FEOpsKernelInfoStorePtr fe_ops_kernel_info_store_ptr)
    : trans_node_merging_ptr_(nullptr),
      trans_node_insertion_ptr_(nullptr),
      fe_ops_kernel_info_store_ptr_(fe_ops_kernel_info_store_ptr) {}

TransNodeManager::~TransNodeManager() {}

Status TransNodeManager::Initialize() {
  FE_MAKE_SHARED(trans_node_insertion_ptr_ = std::make_shared<TransNodeInsertion>(fe_ops_kernel_info_store_ptr_),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_CHECK(trans_node_insertion_ptr_ == nullptr,
           REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][TransNdManager] spaceSizeCalculatorPtr_ is null."),
           return FAILED);

  if (trans_node_insertion_ptr_->initialize() != SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][TransNdManager] TransNode Insert init failed.");
    return FAILED;
  }

  FE_MAKE_SHARED(trans_node_merging_ptr_ = std::make_shared<TransNodeMerging>(),
                 return GRAPH_OPTIMIZER_MAKE_SHARED_FAILED);
  FE_CHECK(trans_node_merging_ptr_ == nullptr,
           REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][TransNdManager] opJudgePtr_ is null."), return FAILED);
  return SUCCESS;
}

Status TransNodeManager::InsertTransNodes(ge::ComputeGraph &graph, ge::NodePtr node) {
  if (trans_node_insertion_ptr_ == nullptr) {
    FE_LOGD("trans_node_insertion_ptr_ is nullptr");
    return FAILED;
  }
  return trans_node_insertion_ptr_->InsertTransNodes(graph, node);
}

Status TransNodeManager::MergeAllTransOps(ge::ComputeGraph &graph) {
  if (trans_node_merging_ptr_ == nullptr) {
    FE_LOGD("trans_node_merging_ptr_ nullptr");
    return FAILED;
  }
  return trans_node_merging_ptr_->MergeAllTransOps(graph, origin_cast_list_, false);
}

Status TransNodeManager::InsertAndMergeTransNodes(ge::ComputeGraph &graph) {
  FE_TIMECOST_START(InsertAndMergeTransNodes);
  FE_LOGD("Function InsertTransNode begins.");
  if (graph.TopologicalSorting() != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][TopoSort] Failed to perform topological sorting for graph %s.",
                    graph.GetName().c_str());

    return fe::SHAPE_FORMAT_TRANSFER_SORTING_FAILED;
  }

  for (ge::NodePtr &dst_node : graph.GetDirectNode()) {
    if (dst_node == nullptr) {
      continue;
    }

    if (dst_node->GetOpDesc() == nullptr) {
      continue;
    }

    /* Store Cast in original grpah. */
    if (dst_node->GetType() == CAST) {
      origin_cast_list_.emplace_back(dst_node);
    }

    /* Loop all input data anchor of dst_node's and insert transop
     * if the input's dtype or format is not the same as it's father
     * node's output dtype and format */
    Status ret = trans_node_insertion_ptr_->InsertTransNodes(graph, dst_node);
    if (ret != SUCCESS) {
      return ret;
    }
  }

  // re-order nodes in graph
  if (graph.TopologicalSorting() != ge::GRAPH_SUCCESS) {
    REPORT_FE_ERROR("[GraphOptJdgInst][ShapeTrans][TopoSort] TopologicalSorting failed!");
    return fe::SHAPE_FORMAT_TRANSFER_SORTING_FAILED;
  }

  // dump
  FE_LOGD("After processing, the graph [%s] appears as follows:", graph.GetName().c_str());
  if (IsDebugLogLevel) {
    for (auto &node : graph.GetDirectNode()) {
      FE_LOGD("Node named [%s]", node->GetName().c_str());
    }
  }

  /* passing the cast nodes list in origin graph and whether need to check it */
  if (trans_node_merging_ptr_->MergeAllTransOps(graph, origin_cast_list_, true) == FAILED) {
    return FAILED;
  }
  FE_TIMECOST_END(InsertAndMergeTransNodes, "InsertAndMergeTransNodes during FEGraphOptimizer::OptimizeOriginalGraph");
  return SUCCESS;
}

const std::vector<ge::NodePtr> &TransNodeManager::GetOptimizableCast() { return origin_cast_list_; }
}  // namespace fe
