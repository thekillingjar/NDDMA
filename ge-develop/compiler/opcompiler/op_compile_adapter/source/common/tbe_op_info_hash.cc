/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/tbe_op_info_hash.h"
#include <memory>
#include <nlohmann/json.hpp>
#include "assemble_json/te_attr_utils.h"

namespace te {
namespace fusion {
using std::hash;
namespace {
constexpr size_t HASH_INITIAL_VALUE = 17;
constexpr size_t HASH_MULTIPER = 31;

// Hash function for TbeAttrValue
struct HashTbeAttrValue {
    size_t operator()(const TbeAttrValue &cattr) const {
        nlohmann::json jsonData;
        GetAttrValueToJson(cattr, jsonData);
        return hash<string>()(jsonData.dump());
    }
};

// Hash function for TbeOpTensor
struct HashTbeOpTensor {
    size_t operator()(const TbeOpTensor &ctensor) const {
        size_t key = HASH_INITIAL_VALUE;
        key = key * HASH_MULTIPER + hash<string>()(ctensor.GetFormat());
        key = key * HASH_MULTIPER + hash<int32_t>()(ctensor.GetSubFormat());
        key = key * HASH_MULTIPER + hash<string>()(ctensor.GetOriginFormat());
        std::string dType = ctensor.GetType() == "bool" ? "int8" : ctensor.GetType();
        key = key * HASH_MULTIPER + hash<string>()(dType);
        key = key * HASH_MULTIPER + static_cast<size_t>(ctensor.GetShapeType());
        key = key * HASH_MULTIPER + hash<bool>()(ctensor.HasConstValue());
        for (const int64_t &shapeDim : ctensor.GetShape()) {
            key = key * HASH_MULTIPER + hash<int64_t>()(shapeDim);
        }

        for (const int64_t &oriShapeDim : ctensor.GetOriginShape()) {
            key = key * HASH_MULTIPER + hash<int64_t>()(oriShapeDim);
        }
        // L1 fusion begin
        key = key * HASH_MULTIPER + hash<size_t>()(ctensor.GetAddrType());

        for (const int64_t validShapeDim : ctensor.GetValidShape()) {
            key = key * HASH_MULTIPER + hash<int64_t>()(validShapeDim);
        }

        for (const int64_t sgtShapeDim : ctensor.GetSgtSliceShape()) {
            key = key * HASH_MULTIPER + hash<int64_t>()(sgtShapeDim);
        }

        for (const int64_t offsetItem : ctensor.GetSliceOffset()) {
            key = key * HASH_MULTIPER + hash<int64_t>()(offsetItem);
        }
        // L1 fusion end
        key = key * HASH_MULTIPER + hash<int32_t>()(ctensor.GetL1FusionType());
        key = key * HASH_MULTIPER + hash<int64_t>()(ctensor.GetL1AddrFlag());
        key = key * HASH_MULTIPER + hash<int64_t>()(ctensor.GetAddrOffset());
        key = key * HASH_MULTIPER + hash<int64_t>()(ctensor.GetL1ValidSize());
        key = key * HASH_MULTIPER + HashTbeAttrValue()(ctensor.GetConstValue());

        for (const int64_t totalShapeDim : ctensor.GetTotalShape()) {
            key = key * HASH_MULTIPER + hash<uint32_t>()(totalShapeDim);
        }

        key = key * HASH_MULTIPER + hash<uint32_t>()(ctensor.GetSplitIndex());

        std::vector<std::pair<int64_t, int64_t>> range;
        if (ctensor.GetShapeRange(range)) {
            for (auto &t : range) {
                key = key * HASH_MULTIPER + hash<int64_t>()(t.first);
                key = key * HASH_MULTIPER + hash<int64_t>()(t.second);
            }
        }

        if (ctensor.GetOriginShapeRange(range)) {
            for (auto &t : range) {
                key = key * HASH_MULTIPER + hash<int64_t>()(t.first);
                key = key * HASH_MULTIPER + hash<int64_t>()(t.second);
            }
        }
        return key;
    }
};

// Hash function for TbeOpParam
struct HashTbeOpParam {
    size_t operator()(const TbeOpParam &copparam) const {
        size_t key = HASH_INITIAL_VALUE;
        te::TensorType tt = copparam.GetType();
        key = key * HASH_MULTIPER + static_cast<size_t>(tt);

        const std::vector<TbeOpTensor> &tensors = copparam.GetTensors();
        for (auto &t : tensors) {
            key = key * HASH_MULTIPER + HashTbeOpTensor()(t);
        }

        return key;
    }
};

template<typename OP, typename Vec, void(OP::*get_vector)(Vec&) const>
bool TbeElemCompare(const OP *op1, const OP *op2)
{
    Vec elems1, elems2;
    (op1->*get_vector)(elems1);
    (op2->*get_vector)(elems2);

    if (elems1.size() != elems2.size()) {
        return false;
    }

    for (size_t i = 0; i < elems1.size(); ++i) {
        if (!(elems1[i] == elems2[i])) {
            return false;
        }
    }
    return true;
}
}

size_t HashTbeOpInfo::operator()(const TbeOpInfo &tbeOpInfo) const
{
    size_t key = HASH_INITIAL_VALUE;
    string tmp;

    std::string opType;
    (void)tbeOpInfo.GetOpType(opType);
    key = key * HASH_MULTIPER + hash<string>()(opType);

    // L1 fusion begin
    int64_t opL1Space = 0;
    (void)tbeOpInfo.GetL1Space(opL1Space);
    key = key * HASH_MULTIPER + hash<int64_t>()(opL1Space);
    // L1 fusion end
    const vector<TbeOpParam> &inputs = tbeOpInfo.GetInputs();
    for (auto &item : inputs) {
        key = key * HASH_MULTIPER + HashTbeOpParam()(item);
    }

    const vector<TbeOpParam> &outputs = tbeOpInfo.GetOutputs();
    for (auto &item : outputs) {
        key = key * HASH_MULTIPER + HashTbeOpParam()(item);
    }

    vector<TbeAttrValue> attrs = {};
    tbeOpInfo.GetAttrValues(attrs);
    for (auto &item : attrs) {
        key = key * HASH_MULTIPER + HashTbeAttrValue()(item);
    }

    std::string opImplMode = tbeOpInfo.GetOpImplMode();
    key = key * HASH_MULTIPER + hash<string>()(opImplMode);

    return key;
}

size_t HashTbeOpInfo::operator()(const std::shared_ptr<TbeOpInfo> &tbeOpInfo) const
{
    return HashTbeOpInfo()(*tbeOpInfo);
}

bool EqualToTbeOpInfo::operator()(const TbeOpInfo &a, const TbeOpInfo &b) const
{
    string tmp1, tmp2;
    (void)a.GetModuleName(tmp1);
    (void)b.GetModuleName(tmp2);

    if (tmp1 != tmp2) {
        return false;
    }

    if (!TbeElemCompare<TbeOpInfo, vector<TbeOpParam>, &TbeOpInfo::GetInputs>(&a, &b) ||
        !TbeElemCompare<TbeOpInfo, vector<TbeOpParam>, &TbeOpInfo::GetOutputs>(&a, &b) ||
        !TbeElemCompare<TbeOpInfo, vector<TbeAttrValue>, &TbeOpInfo::GetAttrValues>(&a, &b)) {
        return false;
    }

    return true;
}

bool EqualToTbeOpInfo::operator()(const std::shared_ptr<TbeOpInfo> &a, const std::shared_ptr<TbeOpInfo> &b) const
{
    return EqualToTbeOpInfo()(*a, *b);
}
}
}