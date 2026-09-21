/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_PARAM_H_
#define ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_PARAM_H_

#include <string>
#include <vector>
#include "tensor_engine/fusion_types.h"
#include "tensor_engine/tbe_op_tensor.h"

namespace te {
class TbeOpParam {
public:
    TbeOpParam()
    {
    }

    ~TbeOpParam()
    {
    }

    TbeOpParam(const TensorType type, const std::vector<TbeOpTensor> &tensors): type_(type), tensors_(tensors)
    {
    }

    void GetType(TensorType& type) const
    {
        type = type_;
    }

    TensorType GetType() const
    {
        return type_;
    }

    void SetType(const TensorType type)
    {
        type_ = type;
    }

    void GetTensors(std::vector<TbeOpTensor>& tensors) const
    {
        tensors = tensors_;
    }

    const std::vector<TbeOpTensor>& GetTensors() const
    {
        return tensors_;
    }

    std::vector<TbeOpTensor>& MutableTensors()
    {
        return tensors_;
    }

    void SetTensors(const std::vector<TbeOpTensor> &tensors)
    {
        tensors_.assign(tensors.begin(), tensors.end());
    }

    void GetName(std::string& name) const
    {
        name = name_;
    }

    const std::string& GetName() const
    {
        return name_;
    }

    void SetName(const std::string &name)
    {
        name_ = name;
    }

    void Clear()
    {
        tensors_.clear();
    }

    void SetTensor(const TbeOpTensor& tensor)
    {
        tensors_.emplace_back(tensor);
    }

    void SetValueDepend(const OP_VALUE_DEPEND valueDepend)
    {
        value_depend_ = valueDepend;
    }

    OP_VALUE_DEPEND GetValueDepend() const
    {
        return value_depend_;
    }

    bool operator==(TbeOpParam& rObject);

private:
    // need to adapt operator== func while adding new variable
    TensorType type_{TT_REQ};
    std::vector<TbeOpTensor> tensors_;
    OP_VALUE_DEPEND value_depend_{VALUE_DEPEND_IGNORE};
    std::string name_;
};
}
#endif  // ATC_OPCOMPILER_INC_TENSOR_ENGINE_TBE_OP_PARAM_H_