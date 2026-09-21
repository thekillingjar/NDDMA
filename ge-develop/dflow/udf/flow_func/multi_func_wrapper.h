/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_MULTI_FUNC_WRAPPER_H
#define FLOW_FUNC_MULTI_FUNC_WRAPPER_H

#include "func_wrapper.h"
#include "flow_func/meta_multi_func.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MultiFuncWrapper : public FuncWrapper {
public:
    explicit MultiFuncWrapper(const std::shared_ptr<MetaMultiFunc> &multi_func,
        const PROC_FUNC_WITH_CONTEXT &proc_func = nullptr,
        const PROC_FUNC_WITH_CONTEXT_Q &proc_func_with_q = nullptr)
        : multi_func_(multi_func), proc_func_(proc_func), proc_func_with_q_(proc_func_with_q) {}

    ~MultiFuncWrapper() override;

    int32_t Init(
        const std::shared_ptr<MetaParams> &meta_param, const std::shared_ptr<MetaRunContext> &run_context) override;

    int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &input_msgs) override;

    int32_t ResetFlowFuncState() override;

    int32_t Proc(const std::vector<std::shared_ptr<FlowMsgQueue>> &input_queues) override;
private:
    std::shared_ptr<MetaMultiFunc> multi_func_;
    PROC_FUNC_WITH_CONTEXT proc_func_;
    PROC_FUNC_WITH_CONTEXT_Q proc_func_with_q_;
    std::shared_ptr<MetaParams> meta_param_;
    std::shared_ptr<MetaRunContext> run_context_;
};
}  // namespace FlowFunc
#endif  // FLOW_FUNC_MULTI_FUNC_WRAPPER_H
