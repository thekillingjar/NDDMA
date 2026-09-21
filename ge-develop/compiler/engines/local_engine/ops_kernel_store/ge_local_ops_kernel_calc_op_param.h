/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_GE_LOCAL_ENGINE_OPS_KERNEL_CALC_OP_PARAM_H_
#define GE_GE_LOCAL_ENGINE_OPS_KERNEL_CALC_OP_PARAM_H_

#include "ge/ge_api_error_codes.h"
#include "graph/node.h"
#include "external/ge_common/ge_api_types.h"

namespace ge {
namespace ge_local {
class GE_FUNC_VISIBILITY GeLocalOpsKernelBuilderCalcOpParam {
 public:
  static graphStatus CalcPhonyConcatNodeOffset(const Node &node);
  static graphStatus CalcPhonySplitNodeOffset(const Node &node);
  static graphStatus CalcNodeOffsetByReuseInput(const Node &node);

 private:
  static Status CheckDimAttrValid(const std::vector<int64_t> &dim_list, const std::vector<int64_t> &num_list,
                                  const uint32_t anchors_size);
  static Status CheckDimAttrMatchTensor(std::vector<int64_t> &dim_list, const GeTensorDescPtr &tensor_desc);

  static Status CheckPhonyConcatInputShapeSame(const Node &node);
  static Status CheckPhonySplitOutputShapeSame(const Node &node);

  static Status CheckPhonyConcatInputIs32Align(const ConstOpDescPtr &op_desc);

  static int64_t GetSplitOffset(const GeTensorDescPtr &tensor_desc, uint32_t node_idx,
      const std::vector<int64_t> &split_dim_list, const std::vector<int64_t> &split_num_list);
  static int64_t GetConcatOffset(const GeTensorDescPtr &tensor_desc, uint32_t node_idx,
      const std::vector<int64_t> &concat_dim_list, const std::vector<int64_t> &concat_num_list);
  static graphStatus HasBufferFusionOffset(const Node &node, bool &has_fusion_offset);
};
}  // namespace ge_local
}  // namespace ge

#endif  // GE_GE_LOCAL_ENGINE_OPS_KERNEL_CALC_OP_PARAM_H_
