/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_REFORMAT_GENERATOR_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_REFORMAT_GENERATOR_H_

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"

#include "common/fe_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace fe {
/** @brief class of generating Trans-node reformat. Provide function of setting
* unique attrs and tensors for reformat op.
* Reformat op is used when we need to transfer NCHW/NHWC/HWCN to Nz or
* Nz to NCHW/NHWC/HWCN. First we need to use reformat to transform NCHW to ND
* and then use transdata to Nz. Reversely,  we need to transform Nz to ND first
* and ND to NCHW then.
* @version 1.0 */
class TransNodeReformatGenerator : public TransNodeBaseGenerator {
 public:
  TransNodeReformatGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr, TransInfoPtr trans_info_ptr);

  ~TransNodeReformatGenerator() override;

  TransNodeReformatGenerator(const TransNodeReformatGenerator &) = delete;

  TransNodeReformatGenerator &operator=(const TransNodeReformatGenerator &) = delete;

  Status AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) override;

 private:
  Status AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::GeShape &shape, const ge::Format &primary_format,
                      const int32_t &sub_format, const ge::DataType &dtype);
};
}

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_REFORMAT_GENERATOR_H_
