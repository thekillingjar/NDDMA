/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_CONTEXT_H
#define FLOW_FUNC_CONTEXT_H

#include <map>
#include "flow_func/meta_context.h"
#include "flow_func/meta_params.h"
#include "flow_func/meta_run_context.h"
#include "flow_func/attr_value.h"
#include "mbuf_flow_msg.h"
#include "flow_model.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY FlowFuncContext : public MetaContext {
public:
    FlowFuncContext(const std::shared_ptr<MetaParams> &meta_param, const std::shared_ptr<MetaRunContext> &run_context);

    ~FlowFuncContext() override;

    int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) override;

    /*
     * get attr.
     * @return AttrValue *: not null->success, null->failed
     */
    std::shared_ptr<const AttrValue> GetAttr(const char *attr_name) const override;

    std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type) override;

    size_t GetInputNum() const override;

    size_t GetOutputNum() const override;

    int32_t RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
        std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) override;

    std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msg_type) override;

    const char *GetWorkPath() const override;

    int32_t GetRunningDeviceId() const override {
        return params_->GetRunningDeviceId();
    }

    int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const override {
        return run_context_->GetUserData(data, size, offset);
    }

    int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg, const OutOptions &options) override;

    int32_t SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
        const OutOptions &options) override;

    std::shared_ptr<FlowMsg> AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
        TensorDataType data_type, uint32_t align) override;
    
    void RaiseException(int32_t exp_code, uint64_t user_context_id) override;

    bool GetException(int32_t &exp_code, uint64_t &user_context_id) override;
private:
    std::shared_ptr<MetaParams> params_;
    std::shared_ptr<MetaRunContext> run_context_;
};
}  // namespace FlowFunc

#endif // FLOW_FUNC_CONTEXT_H
