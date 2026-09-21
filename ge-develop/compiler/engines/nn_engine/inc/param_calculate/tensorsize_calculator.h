/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_TENSORSIZE_CALCULATOR_H
#define FUSION_ENGINE_INC_COMMON_TENSORSIZE_CALCULATOR_H

#include "graph_optimizer/graph_optimize_register_error_codes.h"

#include <map>
#include <string>
#include "graph/compute_graph.h"
#include "graph/op_desc.h"

namespace fe {
class TensorSizeCalculator {
 public:
  /**
   * Calculate the tensor size of input and output of each opdesc
   * @param op_desc opdesc object
   * @param op_impl_type op impl type
   * @return status SUCCESS or FAILED
   */
  static Status CalculateOpTensorSize(ge::NodePtr node);

 private:
  static Status CalcSingleTensorSize(const ge::OpDesc &op_desc, const ge::GeTensorDescPtr &tensor_desc_ptr,
      const string &direction, size_t i, bool output_real_calc_flag, int64_t &tensor_size);

  static Status CalcInputOpTensorSize(const ge::NodePtr &node, const int32_t &output_real_calc_flag);
  
  static bool IsEndNoteOuputNode(const ge::OutDataAnchorPtr &anchor);

  static Status CalcOutputOpTensorSize(const ge::OpDesc &op_desc, const int32_t &output_real_calc_flag,
                                       const ge::Node::Vistor<ge::OutDataAnchorPtr> &output_data_anchors);
};
}  // namespace fe

#endif  // FUSION_ENGINE_INC_COMMON_TENSORSIZE_CALCULATOR_H
