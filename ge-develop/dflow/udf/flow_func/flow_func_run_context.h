/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_RUN_CONTEXT_H
#define FLOW_FUNC_RUN_CONTEXT_H

#include <map>
#include "flow_func/meta_run_context.h"
#include "flow_func/flow_func_params.h"
#include "mbuf_flow_msg.h"

namespace FlowFunc {
using WriterCallback = std::function<int32_t(uint32_t, const std::shared_ptr<FlowMsg>&)>;
#pragma pack(push, 1)
struct UdfExceptionInfo {
    uint64_t trans_id;
    uint64_t user_context_id;
    int32_t exp_code;
    uint32_t exp_context_size;
    uint8_t exp_context[kMaxMbufHeadLen];
};
struct UdfExceptionAbstract {
    int32_t exp_code;
    uint64_t user_context_id;
};
#pragma pack(pop)
class FLOW_FUNC_VISIBILITY FlowFuncRunContext : public MetaRunContext {
public:
    FlowFuncRunContext(uint32_t device_id, std::shared_ptr<FlowFuncParams> params,
                       WriterCallback writer_call_back, uint32_t processor_id = 0UL);

    ~FlowFuncRunContext() override = default;

    int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg) override;

    std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType data_type) override;

    std::shared_ptr<FlowMsg> AllocRawDataMsg(int64_t size, uint32_t align = 512U) override;

    std::shared_ptr<FlowMsg> AllocTensorListMsg(const std::vector<std::vector<int64_t>> &shapes,
                                                const std::vector<TensorDataType> &data_types, uint32_t align = 512U) override;

    std::shared_ptr<FlowMsg> ToFlowMsg(std::shared_ptr<Tensor> tensor) override;

    int32_t RunFlowModel(const char *model_key, const std::vector<std::shared_ptr<FlowMsg>> &input_msgs,
        std::vector<std::shared_ptr<FlowMsg>> &output_msgs, int32_t timeout) override;

    std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msg_type) override;

    int32_t SetInputMbufHead(const MbufInfo &mbuf_info);

    int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const override;

    int32_t SetOutput(uint32_t out_idx, std::shared_ptr<FlowMsg> out_msg, const OutOptions &options) override;

    int32_t SetMultiOutputs(uint32_t out_idx, const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
        const OutOptions &options) override;

    std::shared_ptr<MbufFlowMsg> AllocMbufEmptyDataMsg(MsgType msg_type) const;

    std::shared_ptr<FlowMsg> AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
        TensorDataType data_type, uint32_t align) override;

    void RaiseException(int32_t exp_code, uint64_t user_context_id) override;
    bool GetException(int32_t &exp_code, uint64_t &user_context_id) override;

    int32_t GetExceptionByTransId(uint64_t trans_id, UdfExceptionInfo &exception_info);

    bool SelfRaiseChkAndDel(uint64_t trans_id);

    void RecordException(const UdfExceptionInfo &exp_info);

private:
    int32_t CheckParamsForUserData(const void *data, size_t size, size_t offset) const;
    int32_t BalanceOptionFilter(const OutOptions &options,
        const std::vector<std::shared_ptr<FlowMsg>> &out_msgs,
        std::vector<std::shared_ptr<FlowMsg>> &after_filter_out_msgs) const;

    int32_t CheckAffinityPolicy(AffinityPolicy policy) const;

    MbufHead input_mbuf_head_;
    std::shared_ptr<FlowFuncParams> params_;
    // raised by self
    std::mutex raise_mutex_;
    std::map<uint64_t, UdfExceptionInfo> trans_id_to_exception_raised_;

    bool exception_existed_ = false;
    UdfExceptionAbstract exception_cache_ = {};

    WriterCallback writer_call_back_;
    uint32_t device_id_;
    uint32_t processor_idx_ = 0;
};
}  // namespace FlowFunc

#endif // FLOW_FUNC_RUN_CONTEXT_H
