/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/dtype/op_dtype_reduce_matcher.h"
#include <utility>
#include "common/fe_inner_attr_define.h"
#include "common/fe_utils.h"
#include "common/util/op_info_util.h"
#include "common/math_util.h"
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"

namespace fe {
OpDtypeReduceMatcher::OpDtypeReduceMatcher() : OpDtypeMatcherBase() {}
OpDtypeReduceMatcher::~OpDtypeReduceMatcher() {}

Status OpDtypeReduceMatcher::FindSuitableDtype(const vector<ge::DataType> &op_kernel_dtype_vec,
    const ge::DataType &expected_dtype, vector<uint32_t> &matched_index_vec,
    const ge::DataType &forbidden_dtype) {
  auto expected_dtype_iter = DATATYPE_PRIORITY_MAP_AMPLIFIED.find(expected_dtype);
  if (expected_dtype_iter == DATATYPE_PRIORITY_MAP_AMPLIFIED.end()) {
    FE_LOGD("the dtype %s is not found in DATATYPE_PRIORITY_MAP_AMPLIFIED.",
            ge::TypeUtils::DataTypeToSerialString(expected_dtype).c_str());
    return FindAccuracyDtype(op_kernel_dtype_vec, expected_dtype, matched_index_vec, forbidden_dtype);
  }

  map<int32_t, uint32_t> priority_index_map;
  int32_t prio_gap_increment_value = 1;
  // The flag below shows whether we have found the expected_dtype exactly
  // instead its higher precision version. For example, fp32 is higher precision
  // version of fp16.
  uint32_t op_kernel_dtype_vec_size = op_kernel_dtype_vec.size();
  for (auto iter = matched_index_vec.begin(); iter != matched_index_vec.end(); ++iter) {
    uint32_t index_reduce = *iter;
    if (index_reduce >= op_kernel_dtype_vec_size) {
      continue;
    }
    auto op_kernel_dtype = op_kernel_dtype_vec[index_reduce];
    auto op_kernel_dtype_iter = DATATYPE_PRIORITY_MAP_AMPLIFIED.find(op_kernel_dtype);
    if (op_kernel_dtype_iter == DATATYPE_PRIORITY_MAP_AMPLIFIED.end()) {
      FE_LOGD("The dtype %u in the op kernel is invalid, index [%u].", op_kernel_dtype, index_reduce);
      continue;
    }

    if (op_kernel_dtype == forbidden_dtype) {
      FE_LOGD("The dtype %u in the op kernel is forbidden, can not use it.", op_kernel_dtype);
      continue;
    }

    // 1. the priority gap between the op_kernel data type
    // and the exptected data type
    int32_t prio_gap_reduce = op_kernel_dtype_iter->second - expected_dtype_iter->second;
    // 2. the exptected data type is equal to the op_kernel data type
    if (prio_gap_reduce > LOW_GAP_AMPLIFIED && prio_gap_reduce < HIGH_GAP_AMPLIFIED) {
      if (priority_index_map.find(prio_gap_reduce) != priority_index_map.end()) {
        FE_ADD_OVERFLOW(prio_gap_reduce, prio_gap_increment_value, prio_gap_reduce);
        FE_ADD_OVERFLOW(prio_gap_increment_value, 1, prio_gap_increment_value);
      }
      priority_index_map.emplace(std::make_pair(prio_gap_reduce, index_reduce));
    }
  }

  // 4. if the priority_index_map is not empty, the matched_index_vec is replaced by
  // the priority_index_map
  if (!priority_index_map.empty()) {
    vector<uint32_t> empty_vec;
    matched_index_vec.swap(empty_vec);
    // key is default in ascending order;
    for (const auto &item : priority_index_map) {
      matched_index_vec.emplace_back(item.second);
    }
    return SUCCESS;
  } else {
    // Here if priority_index_map is empty, that means dtype is not found for
    // input_x, we can randomly pick a dtype which op kernel store supports. So
    // just return Success and do nothing with matched_index_vec. For precisely
    // matching subsequent input, We go through input_x+1 and check whether there
    // would be any format and dtype match.
    return FAILED;
  }
}
}  // namespace fe
