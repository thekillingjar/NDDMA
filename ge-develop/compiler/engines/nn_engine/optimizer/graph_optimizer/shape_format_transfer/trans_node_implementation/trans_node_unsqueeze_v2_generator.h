/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_UNSQUEEZE_V2_GENERATOR_H_
#define COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_UNSQUEEZE_V2_GENERATOR_H_

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"

#include "common/fe_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"

namespace fe {

/** @brief class of generating Trans-node unsqueezeV2. Provide function of setting
* unique attrs and tensors for unsqueezeV2 op.
* @version 1.0 */
class TransNodeUnsqueezeV2Generator : public TransNodeBaseGenerator {
 public:
  TransNodeUnsqueezeV2Generator(FEOpsKernelInfoStorePtr fe_ops_store_ptr, TransInfoPtr trans_info_ptr);

  ~TransNodeUnsqueezeV2Generator() override;

  TransNodeUnsqueezeV2Generator(const TransNodeUnsqueezeV2Generator &) = delete;

  TransNodeUnsqueezeV2Generator &operator=(const TransNodeUnsqueezeV2Generator &) = delete;

  Status AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) override;

 private:
  Status SetAttr(const ge::OpDescPtr &op_desc_ptr) const;

  Status SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const override;

  Status AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape, const ge::Format &primary_format,
                      const int32_t &sub_format, const ge::DataType &dtype);

  void GetNewShapeRange(vector<std::pair<int64_t, int64_t>> &new_range) const;
};

}  // namespace fe
#endif  // COMPILER_GRAPHCOMPILER_ENGINES_NNENG_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_UNSQUEEZE_V2_GENERATOR_H_
