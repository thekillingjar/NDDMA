/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_BASE_GENERATOR_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_BASE_GENERATOR_H_
#include "common/fe_inner_attr_define.h"
#include "common/fe_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "ops_kernel_store/fe_ops_kernel_info_store.h"
#include "graph_optimizer/shape_format_transfer/trans_node_manager/trans_node_insertion/insert_trans_node_strategy.h"

namespace fe {
/** @brief Trans-node base class. Provide function of setting basic op_desc and
* attributes and adding nodes nad edges. It will inherited by specific
* trans-nodes like Cast, Reshape, TransData, Transpose. */
class TransNodeBaseGenerator {
 public:
  TransNodeBaseGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr, TransInfoPtr trans_info_ptr);

  virtual ~TransNodeBaseGenerator();

  TransNodeBaseGenerator(const TransNodeBaseGenerator &) = delete;

  TransNodeBaseGenerator &operator=(const TransNodeBaseGenerator &) = delete;

  /* Create a Opdesc Pointer for All trans-ops. It includes all common info
   * like of input, output, fe imply type, ge imply type e.g. of trans-nodes
   * such as cast, reshape, transpose, transdata.
   * */
  ge::OpDescPtr CreateBasicOpDescForTransNode(const string &op_type) const;

  Status AddEdgesAndFreshTransInfo(ge::ComputeGraph &fused_graph, const ge::OpDescPtr &op_desc_ptr);

  /* Add specific op info into OpDescPtr. It's implemented by different sub
   * classes because different trans-nodes have different attributes. */
  virtual Status AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) = 0;

  Status TransformDimsWithFormat(bool increasing_flag) const;

  virtual Status SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const;

  Status SetTensorRealDimCountAndNewShape(ge::OpDescPtr &op_desc_ptr, std::vector<ge::GeShape> inputs_shape,
                                          ge::GeShape output_shape) const;

  Status SetNewShapeRange(const ge::OpDescPtr &op_desc_ptr, vector<std::pair<int64_t, int64_t>> &inputs_range,
                          vector<std::pair<int64_t, int64_t>> &output_range) const;
  
  void SetNodeParallelGroupId(ge::OpDescPtr &op_desc_ptr, ge::OpDescPtr &src_op_desc,
                              ge::OpDescPtr &dst_op_desc);
  void GetGroupIdFromSrcOrDst(ge::OpDescPtr &op_desc_ptr, uint32_t &parallel_group_id,
              uint32_t &heavy_op_parallel_group_id);

  TransInfoPtr trans_info_ptr_;

  bool TransNodeCheckAccuracySupported(const ge::OpDescPtr &op_desc_ptr, bool real_query) const;

  static uint64_t GetTransAtomicId();
  
  static bool Is3DFormat(const ge::Format &format);

  bool TransNodeCheckSupportedByFormatTune(ge::ComputeGraph &fused_graph, const ge::OpDescPtr &op_desc_ptr);
 private:
  /* Add necessory peer nodes */
  virtual Status AddNecessaryPeerNodes(ge::ComputeGraph &fused_graph, ge::NodePtr new_node) const;

  void RefreshSourceTransInfo(ge::NodePtr src_node) const;

  bool InsertNodeBeforeJudge() const;

  FEOpsKernelInfoStorePtr fe_ops_store_info_ptr_;
};

}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_BASE_GENERATOR_H_
