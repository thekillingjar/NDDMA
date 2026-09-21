/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACL_TYPES_TENSOR_DESC_INTERNAL_H
#define ACL_TYPES_TENSOR_DESC_INTERNAL_H

#include <vector>
#include <string>
#include <memory>

#include "graph/ge_attr_value.h"
#include "graph/small_vector.h"
#include "graph/ascend_limits.h"
#include "acl/acl_base.h"

namespace acl {
    constexpr int64_t UNKNOW_DIM = -1;
    constexpr int64_t UNKNOW_RANK = -2;
    enum class AttrRangeType : std::uint8_t {
        RANGE_TYPE,
        VALUE_TYPE
    };

    void ACL_FUNC_VISIBILITY ConvertSvecToVec(const ge::SmallVector<int64_t,
        static_cast<size_t>(ge::kDefaultMaxInputNum)> &svec, std::vector<int64_t> &vec);
    // use only for aclop
    void ACL_FUNC_VISIBILITY ConvertVecToSvec(const std::vector<int64_t> &vec, ge::SmallVector<int64_t,
        static_cast<size_t>(ge::kDefaultMaxInputNum)> &svec);
}

struct ACL_FUNC_VISIBILITY aclTensorDesc {
    aclTensorDesc(const aclDataType aclTensorDataType, const std::initializer_list<int64_t> shape,
        const aclFormat aclTensorFormat);
    aclTensorDesc(const aclDataType aclTensorDataType, const size_t numDims, const int64_t *const aclTensorDims,
        const aclFormat aclTensorFormat);
    aclTensorDesc(const aclTensorDesc &tensorDesc);
    aclTensorDesc &operator=(const aclTensorDesc &tensorDesc);
    aclTensorDesc() = default;
    ~aclTensorDesc() = default;
    aclDataType dataType;
    aclFormat storageFormat = ACL_FORMAT_UNDEFINED;
    aclFormat format;
    ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> dims;
    ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> dimsBackup;
    ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> storageDims;
    ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> storageDimsBackup;
    std::string name;
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    std::vector<std::pair<int64_t, int64_t>> shapeRangeBackup;
    void *address = nullptr;
    std::string dynamicInputName;
    bool isConst = false;
    std::shared_ptr<void> constDataBuf;
    size_t constDataLen = 0U;
    bool isConstBackup = false;
    std::shared_ptr<void> constDataBufBackup;
    size_t constDataLenBackup = 0U;
    aclMemType memtype = ACL_MEMTYPE_DEVICE;
    // valRange is set from aclSetTensorValueRange
    std::vector<std::pair<int64_t, int64_t>> valRange;
    // for windows compile,use map ignore dvpp.so find the implementation GeAttrValue
    std::map<acl::AttrRangeType, ge::GeAttrValue> valueRange;
    std::string DebugString() const;
    bool IsSameTensor(const aclTensorDesc *const other) const;
    bool IsDynamicTensor() const;
    bool CheckShapeRange() const;
    bool IsConstTensor() const
    {
        return isConst;
    }
    bool IsHostMemTensor() const
    {
        return (memtype == ACL_MEMTYPE_HOST) || (memtype == ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT);
    }
    inline bool IsOptinalTensor() const
    {
        return (dataType == ACL_DT_UNDEFINED) && (format == ACL_FORMAT_UNDEFINED) && (dims.empty());
    }
    void Init(const aclTensorDesc &tensorDesc);
    void UpdateTensorShape(const std::vector<int64_t> &shape);
    void UpdateTensorShapeRange(const std::vector<std::pair<int64_t, int64_t>> &ranges);
    inline bool CheckConstTensor(const bool needCheckHostMem) const
    {
        return isConst || (needCheckHostMem && (memtype == ACL_MEMTYPE_HOST));
    }

    bool operator==(const aclTensorDesc *const other) const;
    void BackupDimsAndShapeRanges();
    void RecoverDimsAndShapeRanges();
    void BackupConst();
    void RecoverConst();

private:
    mutable std::string cachedKey;
    mutable std::string cachedShapeKey;
};

#endif // ACL_TYPES_TENSOR_DESC_INTERNAL_H
