/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLOW_FUNC_META_PARAMS_H
#define FLOW_FUNC_META_PARAMS_H

#include <vector>
#include <memory>
#include "flow_func_defines.h"
#include "attr_value.h"

namespace FlowFunc {
class FLOW_FUNC_VISIBILITY MetaParams {
public:
    MetaParams() = default;

    virtual ~MetaParams() = default;

    virtual const char *GetName() const = 0;

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
     * @brief get flow func work path.
     * used for check whether the number of outputs is consistent.
     * @return output num.
     */
    virtual const char *GetWorkPath() const = 0;

    /**
     * @brief get running device id.
     * @return device id.
     */
    virtual int32_t GetRunningDeviceId() const = 0;

    /**
     * @brief get running instance id.
     * @return instance id.
     */
    virtual int32_t GetRunningInstanceId() const;

    /**
     * @brief get running instance num.
     * @return instance num.
     */
    virtual int32_t GetRunningInstanceNum() const;
};
}

#endif // FLOW_FUNC_META_PARAMS_H
