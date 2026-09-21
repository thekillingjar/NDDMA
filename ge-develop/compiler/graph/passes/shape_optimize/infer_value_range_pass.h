/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GRAPH_PASSES_INFER_VALUE_RANGE_PASS_H_
#define GE_GRAPH_PASSES_INFER_VALUE_RANGE_PASS_H_

#include "graph/passes/shape_optimize/infer_base_pass.h"

namespace ge {
class InferValueRangePass : public InferBasePass {
 public:
  graphStatus Infer(NodePtr &node) override;

 private:
  std::string SerialTensorInfo(const GeTensorDescPtr &tensor_desc) const override;
  graphStatus UpdateTensorDesc(const GeTensorDescPtr &src, GeTensorDescPtr &dst, bool &changed) override;
  graphStatus UpdateOutputFromSubgraphs(const std::vector<GeTensorDescPtr> &src, const GeTensorDescPtr &dst) override;
  graphStatus UpdateOutputFromSubgraphsForMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                    const GeTensorDescPtr &dst) override;
  graphStatus UpdateOutputFromSubgraphsForSubgraphMultiDims(const std::vector<GeTensorDescPtr> &src,
                                                            const GeTensorDescPtr &dst) override;
  bool NeedInfer(const NodePtr &node) const override;

  bool InputIsDynamic(const NodePtr &node) const;
  bool InputIsConstOrHasValueRange(const NodePtr &node) const;
  void CheckInputValueRange(const NodePtr &node, bool &has_unknown_value_range, bool &has_zero_in_value_range) const;
  graphStatus GenerateWorstValueRange(const NodePtr &node) const;
  template <typename T>
  graphStatus ConstructData(const GeTensorDesc &tensor_desc, bool use_floor_value,
                            const GeTensorPtr &output_ptr) const;
  graphStatus ConstructDataByType(const GeTensorDesc &tensor_desc, bool use_floor_value,
                                  const GeTensorPtr &output_ptr) const;
  std::vector<ConstGeTensorPtr> ConstructInputTensors(const NodePtr &node, bool use_floor_value) const;
  template <typename T>
  void ConstructValueRange(const GeTensorPtr &left_tensor, const GeTensorPtr &right_tensor,
                           std::vector<std::pair<int64_t, int64_t>> &value_range) const;
  graphStatus ConstructInputAndInferValueRange(const NodePtr &node) const;
};
}  // namespace ge
#endif  // GE_GRAPH_PASSES_INFER_VALUE_RANGE_PASS_H_
