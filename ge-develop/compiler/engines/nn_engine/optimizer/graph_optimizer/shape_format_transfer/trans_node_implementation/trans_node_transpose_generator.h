/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_TRANSPOSE_GENERATOR_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_TRANSPOSE_GENERATOR_H_

#include "graph_optimizer/shape_format_transfer/trans_node_implementation/trans_node_base_generator.h"

#include "common/fe_inner_error_codes.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"

namespace fe {
using trans_pose_map = std::unordered_map<uint64_t, ge::Format>;

const std::map<ge::Format, uint32_t> FORMAT_INDEX_MAP = {
    {ge::FORMAT_NCHW, 0}, {ge::FORMAT_NHWC, 1}, {ge::FORMAT_HWCN, 2}, {ge::FORMAT_CHWN, 3}};
/* First row:  NCHW->NCHW, NCHW->NHWC, NCHW->HWCN, NCHW->CHWN;
 * Second row: NHWC->NCHW, NHCW->NHWC, NHWC->HWCN, NHWC->CHWN;
 * Third row:  HWCN->NCHW, HWCN->NHWC, HWNC->HWCN, HWNC->CHWN;
 * Fourth row: CHWN->NCHW, CHWN->NHWC, CHWN->HWCN, CHWN->CHWN;
 */
const std::vector<std::vector<std::vector<int64_t>>> PERM_VALUE_VECTOR = {
    {{}, {0, 2, 3, 1}, {2, 3, 1, 0}, {1, 2, 3, 0}},
    {{0, 3, 1, 2}, {}, {1, 2, 3, 0}, {3, 1, 2, 0}},
    {{3, 2, 0, 1}, {3, 0, 1, 2}, {}, {3, 0, 1, 2}},
    {{3, 0, 1, 2}, {3, 1, 2, 0}, {1, 2, 0, 3}, {}}};

const std::map<ge::Format, uint32_t> FORMAT_3D_INDEX_MAP = {
    {ge::FORMAT_NCDHW, 0}, {ge::FORMAT_NDHWC, 1}, {ge::FORMAT_DHWCN, 2}, {ge::FORMAT_DHWNC, 3}};
/* First row:  NCDHW->NCDHW, NCDHW->NDHWC, NCDHW->DHWCN, NCDHW->DHWNC;
 * Second row: NDHWC->NCDHW, NHCW->NDHWC, NDHWC->DHWCN, NDHWC->DHWNC;
 * Third row:  DHWCN->NCDHW, DHWCN->NDHWC, DHWCN->DHWCN, DHWCN->DHWNC;
 * Fourth row: DHWNC->NCDHW, DHWNC->NDHWC, DHWNC->DHWCN, DHWNC->DHWNC;
 */
const std::vector<std::vector<std::vector<int64_t>>> PERM_VALUE_3D_VECTOR = {
    {{}, {0, 2, 3, 4, 1}, {2, 3, 4, 1, 0}, {2, 3, 4, 0, 1}},
    {{0, 4, 1, 2, 3}, {}, {1, 2, 3, 4, 0}, {1, 2, 3, 0, 4}},
    {{4, 3, 0, 1, 2}, {4, 0, 1, 2, 3}, {}, {0, 1, 2, 4, 3}},
    {{3, 4, 0, 1, 2}, {3, 0, 1, 2, 4}, {0, 1, 2, 4, 3}, {}}};

/** @brief class of generating Trans-node Transpose. Provide function of setting
* unique attrs and tensors for Transpose op.
* @version 1.0 */
class TransNodeTransposeGenerator : public TransNodeBaseGenerator {
 public:
  TransNodeTransposeGenerator(FEOpsKernelInfoStorePtr fe_ops_store_ptr, TransInfoPtr trans_info_ptr);

  ~TransNodeTransposeGenerator() override;

  TransNodeTransposeGenerator(const TransNodeTransposeGenerator &) = delete;

  TransNodeTransposeGenerator &operator=(const TransNodeTransposeGenerator &) = delete;

  Status AddTransNode(ge::ComputeGraph &fused_graph, TransInfoPtr trans_info_ptr) override;

 private:
  void GetNewShape(ge::OpDescPtr &op_desc_ptr, ge::Format format, ge::DataType dtype, ge::GeShape &newshape);

  Status AddOpAndNode(ge::ComputeGraph &fused_graph, const ge::Format &primary_format, const int32_t &sub_format,
                      const ge::DataType &dtype);

  Status AddNecessaryPeerNodes(ge::ComputeGraph &fused_graph, ge::NodePtr new_node) const override;

  Status SetTensorDescInfo(ge::OpDescPtr &op_desc_ptr) const override;
};

}  // namespace fe
#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_SHAPE_FORMAT_TRANSFER_TRANS_NODE_IMPLEMENTATION_TRANS_NODE_TRANSPOSE_GENERATOR_H_
