/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLOW_FUNC_META_CONTEXT_H
#define FLOW_FUNC_META_CONTEXT_H

#include <vector>
#include <memory>
#include "flow_func_defines.h"
#include "attr_value.h"
#include "flow_msg.h"
#include "out_options.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaContext {
public:
    MetaContext() = default;

    virtual ~MetaContext() = default;

    /**
     * @brief alloc tensor msg by shape and data type.
     * @param shape tensor shape
     * @param dataType data type
     * @return tensor
     */
    virtual std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType dataType) = 0;

    /**
     * @brief set output tensor.
     * @param outIdx output index, start from 0.
     * @param outMsg output msg.
     * @return 0:success, other failed.
     */
    virtual int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowMsg> outMsg) = 0;

    /**
     * @brief get attr.
     * @param attrName attr name, cannot be null, must end with '\0'.
     * @return AttrValue *: not null->success, null->failed
     */
    virtual std::shared_ptr<const AttrValue> GetAttr(const char *attrName) const = 0;

    template<class T>
    int32_t GetAttr(const char *attrName, T &value) const
    {
        auto attrValue = GetAttr(attrName);
        if (attrValue == nullptr) {
            return FLOW_FUNC_ERR_ATTR_NOT_EXITS;
        }
        return attrValue->GetVal(value);
    }

    /**
     * @brief get flow func input num.
     * used for check whether the number of inputs is consistent.
     * @return input num.
     */
    virtual size_t GetInputNum() const = 0;

    /**
     * @brief get flow func output num.
     * used for check whether the number of outputs is consistent.
     * @return output num.
     */
    virtual size_t GetOutputNum() const = 0;

    /**
     * @brief alloc empty data msg.
     * @param msgType msg type which msg will be alloc
     * @return empty data FlowMsg
     */
    virtual std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msgType) = 0;

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
     * @brief get flow func work path.
     * @return flow func work path.
     */
    virtual const char *GetWorkPath() const = 0;

    /**
     * @brief get running device id.
     * @return device id.
     */
    virtual int32_t GetRunningDeviceId() const = 0;

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
    virtual int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowMsg> outMsg, const OutOptions &options)
    {
        (void)outIdx;
        (void)outMsg;
        (void)options;
        // default not support, compatible old python interface
        return FLOW_FUNC_ERR_NOT_SUPPORT;
    }

    /**
     * @brief set output tensor.
     * @param outIdx output index, start from 0.
     * @param outMsgs output msgs.
     * @param options output options.
     * @return 0:success, other failed.
     */
    virtual int32_t SetMultiOutputs(uint32_t outIdx, const std::vector<std::shared_ptr<FlowMsg>> &outMsgs,
        const OutOptions &options)
    {
        (void)outIdx;
        (void)outMsgs;
        (void)options;
        // default not support, compatible old python interface
        return FLOW_FUNC_ERR_NOT_SUPPORT;
    }

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
};
}

#endif // FLOW_FUNC_META_CONTEXT_H
