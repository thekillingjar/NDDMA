/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shape_range_utils.h"

namespace acl {
void ShapeRangeUtils::SetShapeStatus(const int32_t tensorNum,
                                     const aclTensorDesc *const tensorDesc[],
                                     std::vector<ShapeRangeInfo> &inputVec,
                                     std::vector<ShapeRangeInfo> &storageVec)
{
    for (size_t tensorIndex = 0U; tensorIndex < static_cast<size_t>(tensorNum); ++tensorIndex) {
        ShapeRangeInfo rangeInfo;
        if ((tensorDesc[tensorIndex] -> dims.size() > 0U) && (tensorDesc[tensorIndex]->dims[0U] == UNKNOW_RANK)) {
            rangeInfo.isUnknowRank = true;
            inputVec.emplace_back(rangeInfo);
            continue;
        }
        rangeInfo.isUnknowRank = false;
        for (size_t dimIndex = 0U; dimIndex < tensorDesc[tensorIndex]->dims.size(); ++dimIndex) {
            DimRangeInfo dimInfo;
            if (tensorDesc[tensorIndex]->dims[dimIndex] == -1) {
                dimInfo.isRange = false;
            } else {
                dimInfo.isRange = true;
            }
            rangeInfo.dims.emplace_back(dimInfo);
        }
        ShapeRangeInfo storageInfo;
        for (size_t dimIndex = 0U; dimIndex < tensorDesc[tensorIndex]->storageDims.size(); ++dimIndex) {
            DimRangeInfo dimInfo;
            if (tensorDesc[tensorIndex]->storageDims[dimIndex] == -1) {
                dimInfo.isRange = false;
            } else {
                dimInfo.isRange = true;
            }
            storageInfo.dims.emplace_back(dimInfo);
        }
        inputVec.emplace_back(rangeInfo);
        storageVec.emplace_back(storageInfo);
    }
}

void ShapeRangeUtils::SetShapeRange(const int32_t tensorNum,
                                    const aclTensorDesc *const tensorDesc[],
                                    std::vector<ShapeRangeInfo> &shapeRanges)
{
    for (size_t tensorIndex = 0U; tensorIndex < static_cast<size_t>(tensorNum); ++tensorIndex) {
        // Complete the shape range for static tensor
        if (tensorDesc[tensorIndex]->shapeRange.empty()) {
            if ((tensorDesc[tensorIndex]->dims.size() > 0U) &&
                (tensorDesc[tensorIndex]->dims[0U] == UNKNOW_RANK)) {
                ACL_LOG_INFO("the %zu tensor dim is unknowrank", tensorIndex);
            } else {
                std::vector<std::pair<int64_t, int64_t>> range;
                for (size_t dimIndex = 0U; dimIndex < tensorDesc[tensorIndex]->dims.size(); ++dimIndex) {
                    range.emplace_back(std::make_pair(tensorDesc[tensorIndex]->dims[dimIndex],
                        tensorDesc[tensorIndex]->dims[dimIndex]));
                }
                const_cast<aclTensorDesc *>(tensorDesc[tensorIndex])->shapeRange = range;
            }
        }
        // Save shape range for tensors
        for (size_t rangeIndex = 0U; rangeIndex < tensorDesc[tensorIndex]->shapeRange.size(); ++rangeIndex) {
            if (rangeIndex >= shapeRanges[tensorIndex].dims.size()) {
                // This case is triggered when shape is infered by GE to -2 and shaperange is specificed by
                // user.In this case, the shaperange will not be used by the ACL match,
                // so it can return directly
                break;
            }
            shapeRanges[tensorIndex].dims[rangeIndex].range = tensorDesc[tensorIndex]->shapeRange[rangeIndex];
        }
    }
}

void ShapeRangeUtils::SetTensorShapeInfo(const AclOp &op, uint64_t &seq)
{
    // Save shape status and shape range for tensors
    OpRangeInfo opInfo;
    SetShapeStatus(op.numInputs, op.inputDesc, opInfo.inputs, opInfo.inputsStorage);
    SetShapeRange(op.numInputs, op.inputDesc, opInfo.inputs);
    SetShapeStatus(op.numOutputs, op.outputDesc, opInfo.outputs, opInfo.outputsStorage);
    SetShapeRange(op.numOutputs, op.outputDesc, opInfo.outputs);
    const MmWRLockGuard lk(&shapeInfoMutex_);
    std::vector<OpRangeInfo> &opRangeVec = opModelRanges_[op.opType];
    for (const OpRangeInfo &info : opRangeVec) {
        if (opInfo == info) {
            ACL_LOG_INFO("range is already existed for op %s", op.opType.c_str());
            seq = info.seq;
            return;
        }
    }
    ++rangeSeq;
    seq = rangeSeq;
    opInfo.seq = rangeSeq;
    opRangeVec.emplace_back(opInfo);
}

std::vector<OpRangeInfo> *ShapeRangeUtils::GetTensorShapeInfo(const AclOp &op)
{
    const auto it = opModelRanges_.find(op.opType);
    if (it == opModelRanges_.end()) {
        return nullptr;
    }
    return &(it->second);
}

bool ShapeRangeUtils::CheckDimRange(const std::pair<int64_t, int64_t> &rangeLeft,
                                    const std::pair<int64_t, int64_t> &rangeRight)
{
    const int64_t minRange = rangeLeft.first;
    const int64_t maxRange = rangeLeft.second;
    // the range Max in model is -1
    if (rangeRight.second == UNKNOW_DIM) {
        if (minRange < rangeRight.first) {
            return false;
        }
    } else if (maxRange == UNKNOW_DIM) { // the input range max is -1
        return false;
    } else if ((minRange < rangeRight.first) || // input range or model range is static
        (maxRange > rangeRight.second)) {
        return false;
    } else {
        return true;
    }
    return true;
}

bool ShapeRangeUtils::CheckRange(const int32_t tensorNum,
                                 const aclTensorDesc *const tensorDesc[],
                                 const std::vector<ShapeRangeInfo> &inputs)
{
    if (static_cast<size_t>(tensorNum) != inputs.size()) {
        return false;
    }
    for (size_t tensorIndex = 0U; tensorIndex < static_cast<size_t>(tensorNum); ++tensorIndex) {
        if (inputs[tensorIndex].isUnknowRank) {
            continue;
        }
        if (inputs[tensorIndex].dims.size() != tensorDesc[tensorIndex]->dims.size()) {
            return false;
        }
        for (size_t dimIndex = 0U; dimIndex < tensorDesc[tensorIndex]->dims.size(); ++dimIndex) {
            if (tensorDesc[tensorIndex]->dims[dimIndex] == UNKNOW_DIM) {
                if (tensorDesc[tensorIndex]->shapeRange.size() > dimIndex) {
                    if (!CheckDimRange(tensorDesc[tensorIndex]->shapeRange[dimIndex],
                        inputs[tensorIndex].dims[dimIndex].range)) {
                        return false;
                    }
                }
            } else if (tensorDesc[tensorIndex]->dims[dimIndex] == UNKNOW_RANK) {
                return false;
            } else {
                if (tensorDesc[tensorIndex]->dims[dimIndex] < inputs[tensorIndex].dims[dimIndex].range.first) {
                    return false;
                } else if (inputs[tensorIndex].dims[dimIndex].range.second == UNKNOW_DIM) {
                    continue;
                } else if (tensorDesc[tensorIndex]->dims[dimIndex] > inputs[tensorIndex].dims[dimIndex].range.second) {
                    return false;
                } else {
                    continue;
                }
            }
        }
    }
    return true;
}

bool ShapeRangeUtils::CheckShapeRange(const AclOp &op, const OpRangeInfo &rangeInfo)
{
    bool ret = CheckRange(op.numInputs, op.inputDesc, rangeInfo.inputs);
    if (!ret) {
        return false;
    }
    ret = CheckRange(op.numOutputs, op.outputDesc, rangeInfo.outputs);
    if (!ret) {
        return false;
    }
    return true;
}
}
