/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/graph_optimizer/graph_fusion/fusion_quant_util.h"
#include "register/graph_optimizer/graph_fusion/fusion_quant_util_impl.h"

namespace fe {
Status QuantUtil::BiasOptimizeByEdge(ge::NodePtr &quant_node, BiasOptimizeEdges &param,
                                     std::vector<ge::NodePtr> &fusion_nodes) {
  return QuantUtilImpl::BiasOptimizeByEdge(quant_node, param, fusion_nodes);
}

Status QuantUtil::BiasOptimizeByEdge(BiasOptimizeEdges &param, std::vector<ge::NodePtr> &fusion_nodes) {
  return QuantUtilImpl::BiasOptimizeByEdge(param, fusion_nodes);
}

Status QuantUtil::BiasOptimizeByEdge(QuantParam &quant_param, BiasOptimizeEdges &param,
                                     std::vector<ge::NodePtr> &fusion_nodes,
                                     WeightMode cube_type) {
  return QuantUtilImpl::BiasOptimizeByEdge(quant_param, param, fusion_nodes, cube_type);
}

Status QuantUtil::InsertFixpipeDequantScaleConvert(ge::InDataAnchorPtr deq_scale,
    std::vector<ge::NodePtr> &fusion_nodes) {
  return QuantUtilImpl::InsertFixpipeDequantScaleConvert(deq_scale, fusion_nodes);
}

Status QuantUtil::InsertFixpipeDequantScaleConvert(ge::InDataAnchorPtr &deq_scale,
    ge::InDataAnchorPtr &quant_offset, std::vector<ge::NodePtr> &fusion_nodes) {
  return QuantUtilImpl::InsertFixpipeDequantScaleConvert(deq_scale, quant_offset, fusion_nodes);
}

Status QuantUtil::InsertQuantScaleConvert(ge::InDataAnchorPtr &quant_scale, ge::InDataAnchorPtr &quant_offset,
    std::vector<ge::NodePtr> &fusion_nodes) {
  return QuantUtilImpl::InsertQuantScaleConvert(quant_scale, quant_offset, fusion_nodes);
}

Status QuantUtil::InsertRequantScaleConvert(ge::InDataAnchorPtr &req_scale, ge::InDataAnchorPtr &quant_offset,
    ge::InDataAnchorPtr &cube_bias, std::vector<ge::NodePtr> &fusion_nodes) {
  return QuantUtilImpl::InsertRequantScaleConvert(req_scale, quant_offset, cube_bias, fusion_nodes);
}
}  // namespace fe
