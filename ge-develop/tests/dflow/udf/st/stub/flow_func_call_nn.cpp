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
class FlowFuncCallNn : public MetaFlowFunc {
 public:
  FlowFuncCallNn() = default;

  ~FlowFuncCallNn() override = default;

  int32_t Init() override {
    int64_t timeout = 1000;
    if (context_->GetAttr("_TEST_ATTR_TIMEOUT", timeout) == FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_RUN_LOG_INFO("_TEST_ATTR_TIMEOUT value=%ld", timeout);
      timeout_ = static_cast<int32_t>(timeout);
    }
    FLOW_FUNC_RUN_LOG_INFO("FlowFuncCallNn Init");
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_tensors) override {
    FLOW_FUNC_LOG_INFO("FlowFuncCallNn proc begin");
    if (input_tensors.size() != 2) {
      FLOW_FUNC_LOG_ERROR("input num size=%zu must be 2.", input_tensors.size());
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    auto tensor = input_tensors[0]->GetTensor();
    if (tensor == nullptr) {
      FLOW_FUNC_LOG_ERROR("input[0]'s tensor is null.");
      return FLOW_FUNC_ERR_PARAM_INVALID;
    }
    int32_t ret = FLOW_FUNC_SUCCESS;
    std::vector<std::shared_ptr<FlowMsg>> output_msgs;

    if (tensor->GetDataType() == FlowFunc::TensorDataType::DT_FLOAT) {
      FLOW_FUNC_RUN_LOG_INFO("run flowmodel begin, timeout=%d", timeout_);
      ret = context_->RunFlowModel("float_model_key", input_tensors, output_msgs, timeout_);
    } else {
      FLOW_FUNC_RUN_LOG_INFO("run other flowmodel begin, timeout=%d", timeout_);
      ret = context_->RunFlowModel("other_model_key", input_tensors, output_msgs, timeout_);
    }

    FLOW_FUNC_RUN_LOG_INFO("run flowmodel end, timeout=%d", timeout_);

    if (ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("run flow model failed, ret=%d.", ret);
      // must return error, test case depend this.
      return ret;
    }

    for (size_t i = 0; i < output_msgs.size(); ++i) {
      ret = context_->SetOutput(i, output_msgs[i]);
      if (ret != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("set output failed, output idx=%zu.", i);
        return ret;
      }
      FLOW_FUNC_LOG_DEBUG("set output:%zu success.", i);
    }
    FLOW_FUNC_LOG_INFO("FlowFuncCallNn proc end");
    return FLOW_FUNC_SUCCESS;
  }

 private:
  int32_t timeout_ = 1000;
};

REGISTER_FLOW_FUNC("ST_FlowFuncCallNn", FlowFuncCallNn);
}  // namespace FlowFunc