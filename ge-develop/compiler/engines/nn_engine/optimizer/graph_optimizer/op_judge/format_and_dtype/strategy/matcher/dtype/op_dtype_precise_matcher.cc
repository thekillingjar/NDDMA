/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_precise_matcher.h"
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/util/op_info_util.h"

namespace fe {
OpDtypePreciseMatcher::OpDtypePreciseMatcher() : OpDtypeMatcherBase() {}
OpDtypePreciseMatcher::~OpDtypePreciseMatcher() {}

Status OpDtypePreciseMatcher::FindSuitableDtype(const vector<ge::DataType> &op_kernel_dtype_vec,
                                                const ge::DataType &expected_dtype, vector<uint32_t> &matched_index_vec,
                                                const ge::DataType &dtype_forbid) {
  (void) dtype_forbid;
  auto op_kernel_dtype_vec_size = op_kernel_dtype_vec.size();
  vector<uint32_t> priority_index_vec;
  for (auto iter = matched_index_vec.begin(); iter != matched_index_vec.end();
       /* iter will not increase automatically */) {
    uint32_t index = *iter;
    if (index >= op_kernel_dtype_vec_size) {
      iter = matched_index_vec.erase(iter);
      /* Delete the illegal iter and continue. */
      continue;
    }

    auto op_kernel_dtype = op_kernel_dtype_vec[index];
    if (op_kernel_dtype == expected_dtype) {
      priority_index_vec.push_back(index);
      iter++;
      continue;
    }
    iter++;
  }

  if (!priority_index_vec.empty()) {
    // key is default in ascending order;
    matched_index_vec.swap(priority_index_vec);
    return SUCCESS;
  } else {
    return FAILED;
  }
}

}  // namespace fe
