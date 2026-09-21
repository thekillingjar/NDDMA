/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_STRIDEDWRITE_OPTIMIZER_H_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_STRIDEDWRITE_OPTIMIZER_H_

#include <map>
#include <memory>
#include <string>
#include "common/fe_inner_error_codes.h"
#include "common/fe_log.h"
#include "common/fe_utils.h"
#include "common/util/op_info_util.h"
#include "graph/compute_graph.h"
#include "graph/types.h"

namespace fe {
/** @brief provide one interface: 1. optimize sub graph */
class StridedWriteOptimizer {
 public:
  void DoOptimizeForConcat(const ge::ComputeGraph &graph) const;

 private:
  int32_t GetRealConcatDim(const ge::OpDescPtr &op_desc) const;
  void CalSliceOffset(const std::vector<int64_t> &output_shape, ge::DataType data_type,
                      int64_t &output_offset_buff, const int32_t &concat_dim) const;
  void SetAttrForConcat(const ge::NodePtr &node, const ge::OpDescPtr &op_desc) const;
  Status SetStrideWriteInfoPhonyConcat(const ge::NodePtr &concat_node,
                                       const std::vector<int64_t> &concat_out_shape) const;
  Status SetStrideWriteInfoForInputs(const ge::NodePtr &concat_node, const int32_t &concat_dim) const;
  Status FeedToOpStructInfo(ge::NodePtr &node, const size_t &idx,
                            const std::vector<int64_t> &concat_out_shape,
                            const bool &is_last_input, const int32_t &concat_dim,
                            int64_t &phony_concat_offset,
                            const bool &set_offset = false) const;
  Status FeedToOpStructInfo(ge::OpDescPtr& op_desc, const size_t &idx, const std::vector<int64_t> &concat_out_shape,
                            const bool &is_last_input, const int32_t &concat_dim,
                            int64_t &phony_concat_offset, const bool &set_offset = false) const;
};
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_FUSION_GRAPH_OPTIMIZER_NODE_OPTIMIZER_STRIDEDWRITE_OPTIMIZER_H_
