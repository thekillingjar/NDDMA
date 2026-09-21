/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLOW_FUNC_ATTR_VALUE_IMPL_H
#define FLOW_FUNC_ATTR_VALUE_IMPL_H

#include "flow_func/attr_value.h"
#include "ff_udf_attr.pb.h"

namespace FlowFunc {
class AttrValueImpl : public AttrValue {
public:
    explicit AttrValueImpl(ff::udf::AttrValue proto_attr) : AttrValue(), proto_attr_(std::move(proto_attr)) {}

    ~AttrValueImpl() override = default;

    /*
     * get string value of attr.
     * @param value: string value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(AscendString &value) const override;

    /*
     * get string list value of attr.
     * @param value: string list value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(std::vector<AscendString> &value) const override;

    /*
     * get int value of attr.
     * @param value: int value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(int64_t &value) const override;

    /*
     * get int list value of attr.
     * @param value: int list value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(std::vector<int64_t> &value) const override;

    /*
     * get int list list value of attr.
     * @param value: int list list value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(std::vector<std::vector<int64_t>> &value) const override;

    /*
     * get float value of attr.
     * @param value: float value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(float &value) const override;

    /*
     * get float list value of attr.
     * @param value: float list value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(std::vector<float> &value) const override;

    /*
     * get bool value of attr.
     * @param value: bool value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(bool &value) const override;

    /*
     * get bool list value of attr.
     * @param value: bool list value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(std::vector<bool> &value) const override;

    /*
     * get data type value of attr.
     * @param value: data type value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(TensorDataType &value) const override;

    /*
     * get data type list value of attr.
     * @param value: data type list value of attr
     * @return 0:SUCCESS, other:failed
     */
    int32_t GetVal(std::vector<TensorDataType> &value) const override;

private:
    // proto attr
    const ff::udf::AttrValue proto_attr_;
};
}

#endif // FLOW_FUNC_ATTR_VALUE_IMPL_H
