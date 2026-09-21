/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "single_func_wrapper.h"
#include "common/inner_error_codes.h"
#include "common/udf_log.h"
#include "flow_func_context.h"

namespace FlowFunc {
int32_t SingleFuncWrapper::Init(
    const std::shared_ptr<MetaParams> &meta_param, const std::shared_ptr<MetaRunContext> &run_context) {
    if (meta_param == nullptr) {
        UDF_LOG_ERROR("SingleFuncWrapper init failed for MetaParams is null.");
        return FLOW_FUNC_FAILED;
    }
    if (run_context == nullptr) {
        UDF_LOG_ERROR("SingleFuncWrapper init failed for MetaRunContext is null.");
        return FLOW_FUNC_FAILED;
    }
    if (context_ == nullptr) {
        try {
            context_ = std::make_shared<FlowFuncContext>(meta_param, run_context);
        } catch (const std::exception &e) {
            UDF_LOG_ERROR("make FlowFuncContext failed.");
            return FLOW_FUNC_FAILED;
        }
        flow_func_->SetContext(context_.get());
    }
    auto ret = flow_func_->Init();
    if ((ret != FLOW_FUNC_SUCCESS) && (ret != FLOW_FUNC_ERR_INIT_AGAIN)) {
        UDF_LOG_ERROR("init failed, ret=%d.", ret);
    }
    return ret;
}

int32_t SingleFuncWrapper::Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) {
    return flow_func_->Proc(input_msgs);
}

int32_t SingleFuncWrapper::ResetFlowFuncState() {
    return flow_func_->ResetFlowFuncState();
}

int32_t SingleFuncWrapper::Proc(const std::vector<std::shared_ptr<FlowMsgQueue>> &input_queues) {
    (void) input_queues;
    UDF_LOG_ERROR("Stream input is not supported for single func.");
    return FLOW_FUNC_ERR_NOT_SUPPORT;
}
}  // namespace FlowFunc
