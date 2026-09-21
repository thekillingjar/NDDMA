/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph_optimizer/op_judge/format_and_dtype/strategy/matcher/format/op_format_matcher.h"
#include "common/fe_utils.h"
#include "ffts_type.h"
#include "framework/common/ge_types.h"

namespace fe {
OpFormatMatcher::OpFormatMatcher() {}
OpFormatMatcher::~OpFormatMatcher() {}

void OpFormatMatcher::FilterFormatIndex(const ge::NodePtr &node, const OpKernelInfoPtr &op_kernel_info_ptr,
                                        const vector<ge::Format> &kernel_format_vec,
                                        vector<uint32_t> &matched_index_vec) {
  if (!op_kernel_info_ptr->IsHeavyOp()) {
    return;
  }

  bool ffts_flag = false;
  (void)ge::AttrUtils::GetBool(node->GetOwnerComputeGraph(), ffts::kFftsSwitch, ffts_flag);
  if (ffts_flag) {
    return;
  }

  if (KFeFormatModeFilterOp.count(node->GetType()) != 0) {
    FE_LOGD("node[%s:%s] has filtered format index.", node->GetName().c_str(), node->GetType().c_str());
    return;
  }

  vector<uint32_t> non_nd_index;
  for (size_t i = 0; i < matched_index_vec.size(); i++) {
    if (matched_index_vec[i] >= kernel_format_vec.size()) {
      FE_LOGW("Index[%u] over format vec size[%zu].", matched_index_vec[i], kernel_format_vec.size());
      continue;
    }
    if (FE_ORIGIN_FORMAT_SET.count(kernel_format_vec[matched_index_vec[i]]) == 0) {
      non_nd_index.emplace_back(matched_index_vec[i]);
    }
  }

  FE_LOGD("After filtering node %s, non-nd indices are %s.", node->GetName().c_str(),
          StringUtils::IntegerVecToString(non_nd_index).c_str());
  if (!non_nd_index.empty()) {
    matched_index_vec = non_nd_index;
  }
}

Status OpFormatMatcher::Match(const vector<ge::Format> &op_kernel_format_vec, const ge::Format &expected_format,
                              const ge::Format &cur_origin_format, vector<uint32_t> &matched_index_vec) {
  vector<uint32_t> matched_index_vec_temp = matched_index_vec;
  FindSuitableFormat(op_kernel_format_vec, expected_format, cur_origin_format, matched_index_vec_temp);
  if (!matched_index_vec_temp.empty()) {
    matched_index_vec = matched_index_vec_temp;
    return SUCCESS;
  } else {
    if (IsNd(cur_origin_format)) {
      uint32_t op_kernel_format_vec_size = op_kernel_format_vec.size();
      for (auto iter = matched_index_vec.begin(); iter < matched_index_vec.end();
           /* The iter will increase in loop body. */) {
        uint32_t index = *iter;
        if (index < op_kernel_format_vec_size) {
          ge::Format op_kernel_format = op_kernel_format_vec[index];
          if (op_kernel_format == ge::FORMAT_NC1HWC0) {
            iter = matched_index_vec.erase(iter);
          } else {
            iter++;
          }
        } else {
          FE_LOGD("the matched index %u is larger than or equal to the opKernelFormatVecSize %u.",
                  index, op_kernel_format_vec_size);
          iter = matched_index_vec.erase(iter);
        }
      }
    }
    return FAILED;
  }
}

Status OpFormatMatcher::FindSuitableFormat(const vector<ge::Format> &op_kernel_format_vec,
                                           const ge::Format &expected_format, const ge::Format &cur_origin_format,
                                           vector<uint32_t> &matched_index_vec) {
  uint32_t op_kernel_format_vec_size = op_kernel_format_vec.size();
  for (auto iter = matched_index_vec.begin(); iter < matched_index_vec.end();
       /* The iter will increase in loop body. */) {
    uint32_t index = *iter;
    if (index < op_kernel_format_vec_size) {
      // 1. the op_kernel_format is equal to the expected_format
      // 2. the op_kernel_format is in {ND,MD}, the origin format is equal to
      // expected_format
      ge::Format op_kernel_format = op_kernel_format_vec[index];
      if (op_kernel_format == expected_format || (IsNd(op_kernel_format) && cur_origin_format == expected_format)) {
        iter++;
        continue;
      } else {
        iter = matched_index_vec.erase(iter);
      }
    } else {
      FE_LOGD(
          "the matched index %u is larger than or equal to the "
          "opKernelFormatVecSize %u.",
          index, op_kernel_format_vec_size);
      iter = matched_index_vec.erase(iter);
    }
  }
  return SUCCESS;
}
}  // namespace fe
