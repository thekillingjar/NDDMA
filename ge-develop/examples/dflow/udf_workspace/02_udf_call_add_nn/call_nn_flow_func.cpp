/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <iostream>
#include "flow_func/meta_flow_func.h"
#include "flow_func/flow_func_log.h"

namespace FlowFunc {
class CallNnFlowFunc : public MetaFlowFunc {
 public:
  int32_t Init() override {
    (void)context_->GetAttr("enableExceptionCatch", enable_catch_exception_);
    return FLOW_FUNC_SUCCESS;
  }

  int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) override {
    int32_t exp_code = -1;
    uint64_t user_context_id = 0;
    if (context_->GetException(exp_code, user_context_id)) {
      // the following process is a sample. UDF can stop or do other things while got an exception.
      FLOW_FUNC_LOG_ERROR("Get exception raised by normal UDF. exp_code[%d] user_context_id[%lu].", exp_code,
                          user_context_id);
      return FLOW_FUNC_SUCCESS;
    }
    if (input_msgs.size() == 0) {
      FLOW_FUNC_LOG_ERROR("Input msg is empty.");
      return FLOW_FUNC_FAILED;
    }
    const uint32_t raise_exception_trans_id = 3;
    const int32_t raise_exp_code = -100;
    if (enable_catch_exception_ && input_msgs[0]->GetTransactionId() == raise_exception_trans_id) {
      // udf will print error and stop if call RaiseException interface while dataflow disable exception catch
      context_->RaiseException(raise_exp_code, raise_exception_trans_id);
      FLOW_FUNC_LOG_ERROR("Raise exception in nn for test.");
    }
    std::vector<std::shared_ptr<FlowMsg>> output_msgs;
    auto ret = context_->RunFlowModel(depend_key_.c_str(), input_msgs, output_msgs, 100000);
    if (ret != FLOW_FUNC_SUCCESS) {
      FLOW_FUNC_LOG_ERROR("Run flow model failed.");
      return ret;
    }
    for (size_t i = 0UL; i < output_msgs.size(); ++i) {
      ret = context_->SetOutput(i, output_msgs[i]);
      if (ret != FLOW_FUNC_SUCCESS) {
        FLOW_FUNC_LOG_ERROR("Set output msgs failed.");
        return ret;
      }
    }
    FLOW_FUNC_LOG_INFO("Run flow func success.");
    return FLOW_FUNC_SUCCESS;
  }

 private:
  std::string depend_key_{"invoke_graph"};
  bool enable_catch_exception_ = false;
};

REGISTER_FLOW_FUNC("call_nn", CallNnFlowFunc);
}  // namespace FlowFunc