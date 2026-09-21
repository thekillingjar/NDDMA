/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_OP_SHAPE_RANGE_UTILS_H_
#define ACL_OP_SHAPE_RANGE_UTILS_H_

#include <string>
#include <vector>
#include <unordered_map>
#include "types/acl_op_inner.h"

namespace acl {
class MmRDLockGuard {
public:
    explicit MmRDLockGuard(mmRWLock_t *const lk) : rwLock_(lk)
    {
        (void)mmRWLockRDLock(rwLock_);
    }
    ~MmRDLockGuard() noexcept
    {
        (void)mmRDLockUnLock(rwLock_);
    }
private:
    mmRWLock_t *rwLock_;
};

class MmWRLockGuard {
public:
    explicit MmWRLockGuard(mmRWLock_t *const lk) : rwLock_(lk)
    {
        (void)mmRWLockWRLock(rwLock_);
    }
    ~MmWRLockGuard() noexcept
    {
        (void)mmWRLockUnLock(rwLock_);
    }
private:
    mmRWLock_t *rwLock_;
};

struct DimRangeInfo {
    bool isRange = false;
    std::pair<int64_t, int64_t> range;
    friend bool operator==(const DimRangeInfo &left, const DimRangeInfo &other) noexcept
    {
        if (left.isRange != other.isRange) {
            return false;
        }
        if (left.range != other.range) {
            return false;
        }
        return true;
    }
};

struct ShapeRangeInfo {
    bool isUnknowRank = false;
    std::vector<DimRangeInfo> dims;
    friend bool operator==(const ShapeRangeInfo &left, const ShapeRangeInfo &other) noexcept
    {
        if (left.isUnknowRank != other.isUnknowRank) {
            return false;
        }
        if (left.dims != other.dims) {
            return false;
        }
        return true;
    }
};

struct OpRangeInfo {
    uint64_t seq;
    std::vector<ShapeRangeInfo> inputs;
    std::vector<ShapeRangeInfo> outputs;
    std::vector<ShapeRangeInfo> inputsStorage;
    std::vector<ShapeRangeInfo> outputsStorage;
    friend bool operator==(const OpRangeInfo &left, const OpRangeInfo &other) noexcept
    {
        if (left.inputs != other.inputs) {
            return false;
        }
        if (left.outputs != other.outputs) {
            return false;
        }
        if (left.inputsStorage != other.inputsStorage) {
            return false;
        }
        if (left.outputsStorage != other.outputsStorage) {
            return false;
        }
        return true;
    }
};

class ShapeRangeUtils {
public:
    ShapeRangeUtils()
    {
        (void) mmRWLockInit(&shapeInfoMutex_);
    }
    ~ShapeRangeUtils() noexcept
    {
        (void) mmRWLockDestroy(&shapeInfoMutex_);
    }

    static ShapeRangeUtils &GetInstance()
    {
        static ShapeRangeUtils instance;
        return instance;
    }

    void SetTensorShapeInfo(const AclOp &op, uint64_t &seq);
    static bool CheckShapeRange(const AclOp &op, const OpRangeInfo &rangeInfo);
    std::vector<OpRangeInfo>* GetTensorShapeInfo(const AclOp &op);

public:
    mmRWLock_t shapeInfoMutex_;

private:
    static void SetShapeStatus(const int32_t tensorNum,
                               const aclTensorDesc *const tensorDesc[],
                               std::vector<ShapeRangeInfo> &inputVec,
                               std::vector<ShapeRangeInfo> &storageVec);
    static void SetShapeRange(const int32_t tensorNum,
                              const aclTensorDesc *const tensorDesc[],
                              std::vector<ShapeRangeInfo> &shapeRanges);
    static bool CheckDimRange(const std::pair<int64_t, int64_t> &rangeLeft,
                              const std::pair<int64_t, int64_t> &rangeRight);

    static bool CheckRange(const int32_t tensorNum,
                           const aclTensorDesc *const tensorDesc[],
                           const std::vector<ShapeRangeInfo> &inputs);

private:
    std::unordered_map<std::string, std::vector<OpRangeInfo>> opModelRanges_;
    uint64_t rangeSeq = 0U;
};
}

#endif // ACL_OP_SHAPE_RANGE_UTILS_H_

