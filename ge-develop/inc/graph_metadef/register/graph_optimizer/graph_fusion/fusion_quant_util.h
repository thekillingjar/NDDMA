/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FUSION_QUANT_UTIL_H_
#define INC_FUSION_QUANT_UTIL_H_
#include "graph/node.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"
#include <vector>

struct BiasOptimizeEdges {
  ge::InDataAnchorPtr quant_scale;
  ge::InDataAnchorPtr quant_offset;
  ge::InDataAnchorPtr cube_weight;
  ge::InDataAnchorPtr cube_bias;
  ge::InDataAnchorPtr deq_scale;
  bool isValid() {
    return !(cube_weight == nullptr || cube_bias == nullptr);
  }
};

namespace fe {
struct QuantParam {
  float quant_scale;
  float quant_offset;
};

enum class WeightMode {
    WEIGHTWITH2D = 0,
    WEIGHTWITH5D = 1,
    RESERVED
};

class QuantUtil {
 public:
  static Status BiasOptimizeByEdge(BiasOptimizeEdges &param, std::vector<ge::NodePtr> &fusion_nodes);
  static Status BiasOptimizeByEdge(ge::NodePtr &quant_node, BiasOptimizeEdges &param,
                                   std::vector<ge::NodePtr> &fusion_nodes);
  static Status BiasOptimizeByEdge(QuantParam &quant_param, BiasOptimizeEdges &param,
                                   std::vector<ge::NodePtr> &fusion_nodes,
                                   WeightMode cube_type = WeightMode::RESERVED);
  static Status InsertFixpipeDequantScaleConvert(ge::InDataAnchorPtr deq_scale, std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertFixpipeDequantScaleConvert(ge::InDataAnchorPtr &deq_scale, ge::InDataAnchorPtr &quant_offset,
                                                 std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertQuantScaleConvert(ge::InDataAnchorPtr &quant_scale, ge::InDataAnchorPtr &quant_offset,
                                        std::vector<ge::NodePtr> &fusion_nodes);
  static Status InsertRequantScaleConvert(ge::InDataAnchorPtr &req_scale, ge::InDataAnchorPtr &quant_offset,
                                          ge::InDataAnchorPtr &cuba_bias, std::vector<ge::NodePtr> &fusion_nodes);
};
}  // namespace fe
#endif
