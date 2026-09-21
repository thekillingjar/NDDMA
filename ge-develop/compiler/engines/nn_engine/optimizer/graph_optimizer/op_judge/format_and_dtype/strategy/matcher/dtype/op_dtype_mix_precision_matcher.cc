/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_mix_precision_matcher.h"
#include "graph/utils/type_utils.h"
#include "common/fe_log.h"
#include "common/util/op_info_util.h"
#include "common/math_util.h"

namespace fe {
OpDtypeMixPrecisionMatcher::OpDtypeMixPrecisionMatcher() : OpDtypeMatcherBase() {}
OpDtypeMixPrecisionMatcher::~OpDtypeMixPrecisionMatcher() {}

Status OpDtypeMixPrecisionMatcher::FindSuitableDtype(const vector<ge::DataType> &op_kernel_dtype_vec,
    const ge::DataType &expected_dtype, vector<uint32_t> &matched_index_vec, const ge::DataType &forbidden_dtype) {
  (void) expected_dtype;
  uint32_t op_kernel_dtype_vec_size = op_kernel_dtype_vec.size();
  vector<uint32_t> temp_index_vec;
  for (auto iter = matched_index_vec.begin(); iter != matched_index_vec.end();) {
    uint32_t index = *iter;
    if (index >= op_kernel_dtype_vec_size) {
      iter = matched_index_vec.erase(iter);
      /* Delete the illegal iter and continue. */
      continue;
    }
    auto op_kernel_dtype = op_kernel_dtype_vec[index];
    if (op_kernel_dtype == forbidden_dtype) {
      iter = matched_index_vec.erase(iter);
      FE_LOGD("The dtype %u in the op kernel is forbidden, can not use it", op_kernel_dtype);
      continue;
    }
    if (op_kernel_dtype == ge::DT_FLOAT16 || op_kernel_dtype == ge::DT_BF16) {
      temp_index_vec.emplace_back(index);
    }
    iter++;
  }

  if (!temp_index_vec.empty()) {
    matched_index_vec.swap(temp_index_vec);
    return SUCCESS;
  }
  return FAILED;
}
}  // namespace fe
