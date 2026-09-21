/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func/meta_flow_func.h"
#include "flow_func/flow_func_log.h"

namespace FlowFunc {
class FlowFuncExchangeQueues : public MetaFlowFunc {
 public:
  FlowFuncExchangeQueues() = default;

  ~FlowFuncExchangeQueues() override = default;

  int32_t Init() override {
    auto get_ret = context_->GetAttr("in_to_out_idx_list", input_to_output_idx_list_);
    if (get_ret != FLOW_FUNC_SUCCESS) {
      return get_ret;
    }
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) override {
    size_t queue_num = input_tensors.size();
    if (input_to_output_idx_list_.size() != queue_num) {
      FLOW_FUNC_LOG_ERROR("input_to_output_idx_list_ size=%zu is not same as input tensor size=%zu.",
                          input_to_output_idx_list_.size(), queue_num);
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    for (size_t i = 0; i < queue_num; ++i) {
      auto ret = context_->SetOutput(input_to_output_idx_list_[i], input_tensors[i]);
      if (ret != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("set output failed, input idx=%zu, output idx=%ld.", i, input_to_output_idx_list_[i]);
        return ret;
      }
    }
    return FLOW_FUNC_SUCCESS;
  }

 private:
  std::vector<int64_t> input_to_output_idx_list_;
};

REGISTER_FLOW_FUNC("UdfExchangeQueues", FlowFuncExchangeQueues);
}  // namespace FlowFunc
