/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_TF_OPTIMIZER_UTILS_H_
#define AICPU_TF_OPTIMIZER_UTILS_H_

#include "graph/compute_graph.h"
#include "ge/ge_api_error_codes.h"
namespace aicpu {
class TfVariableGraph {
 public:
  /**
   * Generate the graph after inserte CacheUpdate node
   * @param graph Compute graph
   * @param have_insert_variable whether have insert variable op
   * @return status whether this operation success
   */
  static ge::Status GenerateTfVariableGraph(ge::ComputeGraph &graph, bool &have_insert_variable);

 private:
  /**
   * Create assign node
   * @param first_src_tensor_desc first src Tensor Desc
   * @param second_src_tensor_desc second src Tensor Desc
   * @param dst_tensor_desc dst Tensor Desc
   * @param graph Compute graph
   * @param assign_node Assign node ptr
   * @return status whether this operation success
   */
  static ge::Status CreateAssign(ge::GeTensorDesc &first_src_tensor_desc,
                                 ge::GeTensorDesc &second_src_tensor_desc,
                                 ge::GeTensorDesc &dst_tensor_desc,
                                 ge::ComputeGraph &graph,
                                 ge::NodePtr &assign_node);

  /**
   * Create identity node
   * @param src_tensor_desc src Tensor Desc
   * @param dst_tensor_desc dst Tensor Desc
   * @param graph Compute graph
   * @param identity_node Indentity node ptr
   * @return status whether this operation success
   */
  static ge::Status CreateIdentity(ge::GeTensorDesc &src_tensor_desc,
                                   ge::GeTensorDesc &dst_tensor_desc,
                                   ge::ComputeGraph &graph,
                                   ge::NodePtr &identity_node);

  /**
   * Create Variable node
   * @param dst_tensor_desc dst Tensor Desc
   * @param graph Compute graph
   * @param variable_node Variable node ptr
   * @return status whether this operation success
   */
  static ge::Status CreateVariable(ge::GeTensorDesc &dst_tensor_desc,
                                   ge::ComputeGraph &graph,
                                   ge::NodePtr &variable_node);

  /**
   * Insert Variable node
   * @param src_anchor Src anchor ptr
   * @param dst_anchor Dst anchor ptr
   * @param variable_node Variable node ptr
   * @return status whether this operation success
   */
  static ge::Status InsertVariable(ge::OutDataAnchorPtr &src_anchor,
                                   const ge::InDataAnchorPtr &dst_anchor,
                                   ge::NodePtr &variable_node);

  /**
   * Insert assign node
   * @param src_anchor Src anchor ptr
   * @param dst_anchor Dst anchor ptr
   * @param node_ptr Node ptr
   * @param assign_node Assign node ptr
   * @return status whether this operation success
   */
  static ge::Status InsertAssign(ge::OutDataAnchorPtr &src_anchor,
                                 const ge::InDataAnchorPtr &dst_anchor,
                                 const ge::NodePtr &node_ptr,
                                 ge::NodePtr &assign_node);

  /**
   * Insert identity node
   * @param src_anchor Src anchor ptr
   * @param dst_anchor Dst anchor ptr
   * @param identity_node Identity node ptr
   * @return status whether this operation success
   */
  static ge::Status InsertIdentity(ge::OutDataAnchorPtr &src_anchor,
                                   const ge::InDataAnchorPtr &dst_anchor,
                                   ge::NodePtr &identity_node);

  /**
   * Create and insert the variable node in graph
   * @param src_anchor Src anchor ptr, need update after insert
   * @param dst_anchor Dst anchor ptr
   * @param variable_node Variable node ptr
   * @param graph Compute graph
   * @return status whether this operation success
   */
  static ge::Status CreateAndInsertVariable(ge::OutDataAnchorPtr &src_anchor,
                                            const ge::InDataAnchorPtr &dst_anchor,
                                            ge::NodePtr &variable_node,
                                            ge::ComputeGraph &graph);

  /**
   * Create and insert the identity node in graph
   * @param src_anchor Src anchor ptr, need update after insert
   * @param dst_anchor Dst anchor ptr
   * @param graph Compute graph
   * @return status whether this operation success
   */
  static ge::Status CreateAndInsertIdentity(ge::OutDataAnchorPtr &src_anchor,
                                            const ge::InDataAnchorPtr &dst_anchor,
                                            ge::ComputeGraph &graph);

  /**
   * Create and insert the Assign node in graph
   * @param src_anchor Src anchor ptr
   * @param dst_anchor Dst anchor ptr
   * @param node_ptr Assign target Op Node(eg. Variable Op)
   * @param graph Compute graph
   * @return status whether this operation success
   */
  static ge::Status CreateAndInsertAssign(ge::OutDataAnchorPtr &src_anchor,
                                          const ge::InDataAnchorPtr &dst_anchor,
                                          const ge::NodePtr &node_ptr,
                                          ge::ComputeGraph &graph);

  /**
   * Check Tensor is ref or not
   * @param tensor_desc tensor desc
   * @return bool if is ref tensor
   */
  static bool IsRefTensorDesc(const ge::GeTensorDesc &tensor_desc);
};

class TfTransposeGraph {
 public:
  static ge::Status GenerateTfTransposeGraph(ge::ComputeGraph &graph);

 private:
  static ge::NodePtr CreateTransposeNode(ge::ComputeGraph &graph,
                                         const std::string &name,
                                         const ge::GeTensorDesc tensor_desc);

  static ge::OpDescPtr CreateConstNode();

  static ge::Status CreateAndInsertTransposeNode(ge::ComputeGraph &graph,
                                                 const ge::NodePtr &node);
};

class TfBatchMatMulV2Graph {
 public:
  static ge::Status GenerateBatchMatMulV2Graph(ge::ComputeGraph &graph);

 private:
  static ge::Status CreateAndInsertBiasConstNode(const ge::NodePtr &dst_node, ge::ComputeGraph &graph);
  static ge::Status CreateAndInsertOffsetWConstNode(const ge::NodePtr &dst_node, ge::ComputeGraph &graph);
};

class TfFixInvalidNodeName {
 public:
  static ge::Status TfGraphFixInvalidNodeName(ge::ComputeGraph &graph);
};
} // namespace aicpu
#endif // AICPU_TF_OPTIMIZER_UTILS_H_