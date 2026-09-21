/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLOW_FUNC_META_RUN_CONTEXT_H
#define FLOW_FUNC_META_RUN_CONTEXT_H

#include <vector>
#include <memory>
#include "flow_func_defines.h"
#include "attr_value.h"
#include "flow_msg.h"
#include "out_options.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaRunContext {
public:
    MetaRunContext() = default;

    virtual ~MetaRunContext() = default;

    /**
     * @brief alloc tensor msg by shape and data type.
     * @param shape tensor shape
     * @param dataType data type
     * @return tensor
     */
    virtual std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType dataType) = 0;

    /**
     * @brief alloc empty data msg.
     * @param msgType msg type which msg will be alloc
     * @return empty data FlowMsg
     */
    virtual std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msgType) = 0;
    /**
     * @brief set output tensor.
     * @param outIdx output index, start from 0.
     * @param outMsg output msg.
     * @return 0:success, other failed.
     */
    virtual int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowMsg> outMsg) = 0;

    /**
     * @brief run flow model.
     * @param modelKey invoked flow model key.
     * @param inputMsgs flow model input message.
     * @param outputMsgs flow model output message.
     * @param timeout timeout(ms), -1 means never timeout.
     * @return 0:success, other failed.
     */
    virtual int32_t RunFlowModel(const char *modelKey, const std::vector<std::shared_ptr<FlowMsg>> &inputMsgs,
        std::vector<std::shared_ptr<FlowMsg>> &outputMsgs, int32_t timeout) = 0;

    /**
     * @brief get user data, max data size is 64.
     * @param data user data point, output.
     * @param size user data size, need in (0, 64].
     * @param offset user data offset, need in [0, 64), size + offset <= 64.
     * @return success:FLOW_FUNC_SUCCESS, failed:OTHERS.
     */
    virtual int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const = 0;

    /**
    * @brief set output tensor.
     * @param outIdx output index, start from 0.
     * @param outMsg output msg.
     * @param options output options.
     * @return 0:success, other failed.
     */
    virtual int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowMsg> outMsg, const OutOptions &options) = 0;

    /**
     * @brief set output tensor.
     * @param outIdx output index, start from 0.
     * @param outMsgs output msgs.
     * @param options output options.
     * @return 0:success, other failed.
     */
    virtual int32_t SetMultiOutputs(uint32_t outIdx, const std::vector<std::shared_ptr<FlowMsg>> &outMsgs,
        const OutOptions &options) = 0;

    /**
     * @brief alloc tensor msg by shape and data type.
     * @param shape tensor shape
     * @param dataType data type
     * @param align: align of tensor, must be[32, 1024] and must can be divisible by 1024.
     * @return tensor
     */
    virtual std::shared_ptr<FlowMsg> AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
        TensorDataType dataType, uint32_t align);

    /**
     * @brief alloc tensor msg by shape and data type.
     * @param shapes tensor shape list
     * @param dataTypes data type list
     * @param align align of tensor, default value is 512.
     * @return tensor
     */
    virtual std::shared_ptr<FlowMsg> AllocTensorListMsg(const std::vector<std::vector<int64_t>> &shapes,
        const std::vector<TensorDataType> &dataTypes, uint32_t align = 512U);

    /**
     * @brief raise exception with exception code and context id.
     * @param expCode exception code defined by user
     * @param userContextId user define context id raise with exception
     * @return void
     */
    virtual void RaiseException(int32_t expCode, uint64_t userContextId);

    /**
     * @brief get exception while exception is existed.
     * @param expCode exception code recorded during dataflow running procedure 
     * @param userContextId user define context id raise with exception
     * @return true:there are some exception can be got. false: there is nothing can be got
     */
    virtual bool GetException(int32_t &expCode, uint64_t &userContextId);

    /**
     * @brief alloc raw data msg.
     * @param size msg size
     * @return raw data FlowMsg
     */
    virtual std::shared_ptr<FlowMsg> AllocRawDataMsg(int64_t size, uint32_t align = 512U);

    /**
     * @brief transfer tensor to flow msg.
     * @param tensor tensor
     * @return tensor FlowMsg
     */
    virtual std::shared_ptr<FlowMsg> ToFlowMsg(std::shared_ptr<Tensor> tensor);
};
}

#endif // FLOW_FUNC_META_RUN_CONTEXT_H
