/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_FORMAT_OP_FORMAT_MATCHER_H_
#define FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_FORMAT_OP_FORMAT_MATCHER_H_

#include "common/fe_inner_error_codes.h"
#include "common/fe_op_info_common.h"
#include "graph/compute_graph.h"

namespace fe {
class OpFormatMatcher {
 public:
  OpFormatMatcher();
  virtual ~OpFormatMatcher();
  Status Match(const vector<ge::Format> &op_kernel_format_vec, const ge::Format &expected_format,
               const ge::Format &cur_origin_format, vector<uint32_t> &matched_index_vec);

  void FilterFormatIndex(const ge::NodePtr &node,
                         const OpKernelInfoPtr &op_kernel_info_ptr,
                         const vector<ge::Format> &kernel_format_vec,
                         vector<uint32_t>& matched_index_vec);

 private:
  Status FindSuitableFormat(const vector<ge::Format> &op_kernel_format_vec, const ge::Format &expected_format,
                            const ge::Format &cur_origin_format, vector<uint32_t> &matched_index_vec);
};
}  // namespace fe

#endif  // FUSION_ENGINE_OPTIMIZER_GRAPH_OPTIMIZER_OP_JUDGE_FORMAT_AND_DTYPE_STRATEGY_MATCHER_FORMAT_OP_FORMAT_MATCHER_H_
