/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flow_func_context.h"
#include "common/udf_log.h"

namespace FlowFunc {

std::shared_ptr<FlowMsg> MetaContext::AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
    TensorDataType dataType, uint32_t align) {
    (void)shape;
    (void)dataType;
    (void)align;
    UDF_LOG_ERROR("Not supported, please implement the AllocTensorMsgWithAlign method.");
    return nullptr;
}

void MetaContext::RaiseException(int32_t expCode, uint64_t userContextId) {
    (void)expCode;
    (void)userContextId;
    UDF_LOG_ERROR("Not supported, please implement the RaiseException method.");
}

bool MetaContext::GetException(int32_t &expCode, uint64_t &userContextId) {
    (void)expCode;
    (void)userContextId;
    return false;
}

FlowFuncContext::FlowFuncContext(
    const std::shared_ptr<MetaParams> &meta_param, const std::shared_ptr<MetaRunContext> &run_context)
    : MetaContext(), params_(meta_param), run_context_(run_context){
    }

FlowFuncContext::~FlowFuncContext() = default;

size_t FlowFuncContext::GetInputNum() const {
    return params_->GetInputNum();
}

size_t FlowFuncContext::GetOutputNum() const {
    // now all output must with queues, pls change here if support part output with queues.
    return params_->GetOutputNum();
}

const char *FlowFuncContext::GetWorkPath() const {
    return params_->GetWorkPath();
}

std::shared_ptr<const AttrValue> FlowFuncContext::GetAttr(const char *attr_name) const {
    return params_->GetAttr(attr_name);
}

int32_t FlowFuncContext::SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) {
    return run_context_->SetOutput(out_idx, out_msg);
}

std::shared_ptr<FlowMsg> FlowFuncContext::AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type) {
    return run_context_->AllocTensorMsg(shape, data_type);
}

std::shared_ptr<FlowMsg> FlowFuncContext::AllocEmptyDataMsg(MsgType msg_type) {
    return run_context_->AllocEmptyDataMsg(msg_type);
}

int32_t FlowFuncContext::RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
    std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) {
    return run_context_->RunFlowModel(model_key, input_msgs, output_msgs, timeout);
}

int32_t FlowFuncContext::SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg, const OutOptions &options) {
    return run_context_->SetOutput(out_idx, out_msg, options);
}

int32_t FlowFuncContext::SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
    const OutOptions &options) {
    return run_context_->SetMultiOutputs(out_idx, out_msgs, options);
}

std::shared_ptr<FlowMsg> FlowFuncContext::AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
    TensorDataType data_type, uint32_t align) {
    return run_context_->AllocTensorMsgWithAlign(shape, data_type, align);
}

bool FlowFuncContext::GetException(int32_t &exp_code, uint64_t &user_context_id) {
    return run_context_->GetException(exp_code, user_context_id);
}

void FlowFuncContext::RaiseException(int32_t exp_code, uint64_t user_context_id) {
    return run_context_->RaiseException(exp_code, user_context_id);
}
}
