/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONV_WEIGHT_COMPRESS_FUSION_PASS_H_
#define FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONV_WEIGHT_COMPRESS_FUSION_PASS_H_

#include <vector>
#include "graph_optimizer/fusion_common/pattern_fusion_base_pass.h"

namespace fe {
class ConvWeightCompressFusionPass : public PatternFusionBasePass {
 protected:
  vector<FusionPattern *> DefinePatterns() override;
  Status Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusion_nodes) override;

 private:
  /**
   * whether convert the conv node to compress conv node
   * @param node_ptr node pointer
   * @return result, 1 compress , 2 sparse_matrix, 0 nothing
   */
  uint32_t IsCompressWeight(ge::NodePtr node_ptr) const;
  /**
   * create and fill in parameters for conv compress op desc
   * @param conv_op_desc op of conv2d/fc
   * @param conv_compress_op_desc target op
   * @return SUCCESS / FAILED
   */
  Status CreateConvCompressOpDesc(ge::OpDescPtr conv_op_desc, ge::OpDescPtr &conv_compress_op_desc,
                                  const uint8_t &compress_flag) const;

  ge::NodePtr CreateHostNode(const ge::OpDescPtr &conv_op_desc, ge::ComputeGraph &graph,
                             const uint8_t &compress_flag) const;

  ge::NodePtr CreateSwitchNode(const ge::OpDescPtr &conv_op_desc, ge::ComputeGraph &graph) const;

  ge::NodePtr CreateMergeNode(const ge::OpDescPtr &conv_op_desc, ge::ComputeGraph &graph) const;

  bool CheckWeightConstFoldNode(ge::NodePtr conv_node_ptr) const;

  bool CheckConstFoldNode(ge::NodePtr node_ptr) const;

  Status RelinkNodeEdges(ge::NodePtr conv_node, ge::NodePtr conv_compress_node, ge::NodePtr host_node,
                         ge::NodePtr switch_node, ge::NodePtr merge_node) const;
};
}  // namespace fe
#endif  // FUSION_ENGINE_FUSION_GRAPH_OPTIMIZER_GRAPH_FUSION_FUSION_PASS_MANAGER_BUILTIN_PASS_CONV_WEIGHT_COMPRESS_FUSION_PASS_H_
