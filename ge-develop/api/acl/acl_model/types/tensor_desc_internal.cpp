/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "tensor_desc_internal.h"
#include <sstream>
#include "framework/common/ge_format_util.h"
#include "utils/string_utils.h"
#include "utils/math_utils.h"
#include "model/acl_resource_manager.h"
#include "common/prof_api_reg.h"
#include "common/error_codes_inner.h"
#include "model/acl_model_impl.h"

namespace acl {
    void ConvertSvecToVec(const ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> &svec,
        std::vector<int64_t> &vec)
    {
        vec.resize(svec.size());
        for (size_t i = 0U; i < vec.size(); ++i) {
            vec[i] = svec[i];
        }
    }

    void ConvertVecToSvec(const std::vector<int64_t> &vec,
        ge::SmallVector<int64_t, static_cast<size_t>(ge::kDefaultMaxInputNum)> &svec)
    {
        svec.resize(vec.size());
        for (size_t i = 0U; i < vec.size(); ++i) {
            svec[i] = vec[i];
        }
    }
}

aclTensorDesc::aclTensorDesc(const aclDataType aclTensorDataType,
    const std::initializer_list<int64_t> shape, const aclFormat aclTensorFormat) :
    dataType(aclTensorDataType),
    format(aclTensorFormat),
    dims(shape)
{
    if (acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(false)) { // how to deal with it for model?
        this->storageDims = dims;
    }
}

aclTensorDesc::aclTensorDesc(const aclDataType aclTensorDataType,
    const size_t numDims, const int64_t *const aclTensorDims, const aclFormat aclTensorFormat) :
    dataType(aclTensorDataType),
    format(aclTensorFormat)
{
    for (size_t i = 0U; i < numDims; ++i) {
        this->dims.push_back(*(aclTensorDims + i));
    }
    if (acl::AclResourceManager::GetInstance().IsRuntimeV2Enable(false)) { // how to deal with it for model?
        this->storageDims = dims;
    }
}

void aclTensorDesc::Init(const aclTensorDesc &tensorDesc)
{
    this->dataType = tensorDesc.dataType;
    this->storageFormat = tensorDesc.storageFormat;
    this->format = tensorDesc.format;
    this->dims = tensorDesc.dims;
    this->dimsBackup = tensorDesc.dimsBackup;
    this->storageDims = tensorDesc.storageDims;
    this->storageDimsBackup = tensorDesc.storageDimsBackup;
    this->name = tensorDesc.name;
    this->shapeRange = tensorDesc.shapeRange;
    this->shapeRangeBackup = tensorDesc.shapeRangeBackup;
    this->address = tensorDesc.address;
    this->isConst = tensorDesc.isConst;
    this->isConstBackup = tensorDesc.isConstBackup;
    this->constDataLen = tensorDesc.constDataLen;
    this->constDataLenBackup = tensorDesc.constDataLenBackup;
    this->constDataBuf = tensorDesc.constDataBuf;
    this->constDataBufBackup=tensorDesc.constDataBufBackup;
    this->cachedKey = tensorDesc.cachedKey;
    this->cachedShapeKey = tensorDesc.cachedShapeKey;
    this->memtype = tensorDesc.memtype;
    for (const auto &it : tensorDesc.valueRange) {
        this->valueRange[it.first] = it.second.Copy();
    }
}

aclTensorDesc::aclTensorDesc(const aclTensorDesc &tensorDesc)
{
    Init(tensorDesc);
}

aclTensorDesc &aclTensorDesc::operator=(const aclTensorDesc &tensorDesc)
{
    Init(tensorDesc);
    return *this;
}

std::string aclTensorDesc::DebugString() const
{
    std::stringstream ss;
    ss << "[TensorDesc] ";
    ss << "DataType = " << dataType;
    ss << ", Format = " << format;
    ss << ", StorageFormat = " << storageFormat;
    ss << ", Shape = " << acl::StringUtils::VectorToString(dims);
    ss << ", StorageShape = " << acl::StringUtils::VectorToString(storageDims);
    ss << ", shapeRange = " << acl::StringUtils::VectorToString(shapeRange);
    ss << ", memtype = " << memtype;
    ss << ", isConst = " << isConst;
    if (isConst) {
        ss << " , isConst = true, Const Len = " << (constDataLen / sizeof(int32_t)) << " , Const data = ";
        for (size_t i = 0U; i < (constDataLen / sizeof(int32_t)); ++i) {
            ss <<  *(static_cast<const int32_t*>(constDataBuf.get()) + i);
            ss << ",";
        }
    }
    return ss.str();
}

bool aclTensorDesc::IsDynamicTensor() const
{
    for (size_t i = 0U; i < dims.size(); ++i) {
        if ((dims[i] == acl::UNKNOW_DIM) || (dims[i] == acl::UNKNOW_RANK)) {
            return true;
        }
    }
    if (!valueRange.empty()) {
        return true;
    }
    return false;
}

void aclTensorDesc::UpdateTensorShape(const std::vector<int64_t> &shape)
{
    dims.clear();
    for (size_t i = 0U; i < shape.size(); ++i) {
        dims.emplace_back(shape[i]);
    }
}

void aclTensorDesc::UpdateTensorShapeRange(const std::vector<std::pair<int64_t, int64_t>> &ranges)
{
    shapeRange.clear();
    for (size_t i = 0U; i < ranges.size(); ++i) {
        shapeRange.emplace_back(ranges[i]);
    }
}

bool aclTensorDesc::CheckShapeRange() const
{
    if ((dims.size() > 0U) && (dims.at(0U) == acl::UNKNOW_RANK)) {
        return shapeRange.empty();
    }
    bool isUnkownDim = false;
    for (size_t i = 0U; i < dims.size(); ++i) {
        if (dims[i] == acl::UNKNOW_DIM) {
            isUnkownDim = true;
            break;
        }
    }
    if (isUnkownDim) {
        if (dims.size() != shapeRange.size()) {
            return false;
        }
    }
    return true;
}

bool aclTensorDesc::operator==(const aclTensorDesc *const other) const
{
    // when check model matched failed, we should report WARNING log not ERROR
    if (other == nullptr) {
        ACL_LOG_DEBUG("aclTensorDesc must not be null.");
        return false;
    }

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->dataType), static_cast<int32_t>(other->dataType));

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->format), static_cast<int32_t>(other->format));

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->storageFormat), static_cast<int32_t>(other->storageFormat));

    if (this->dims != other->dims) {
        return false;
    }
    if (this->shapeRange != other->shapeRange) {
        return false;
    }

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->isConst), static_cast<int32_t>(other->isConst));

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->memtype), static_cast<int32_t>(other->memtype));

    return true;
}

bool aclTensorDesc::IsSameTensor(const aclTensorDesc *const other) const
{
    // when check model matched failed, we should report WARNING log not ERROR
    if (other == nullptr) {
        ACL_LOG_DEBUG("aclTensorDesc must not be null.");
        return false;
    }

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->dataType), static_cast<int32_t>(other->dataType));

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->format), static_cast<int32_t>(other->format));

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->storageFormat), static_cast<int32_t>(other->storageFormat));

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->isConst), static_cast<int32_t>(other->isConst));

    ACL_CHECK_INT32_EQUAL(static_cast<int32_t>(this->memtype), static_cast<int32_t>(other->memtype));

    return true;
}

aclTensorDesc *aclCreateTensorDescImpl(aclDataType dataType,
                                   int numDims,
                                   const int64_t *dims,
                                   aclFormat format)
{
    ACL_PROFILING_REG(acl::AclProfType::AclCreateTensorDesc);
    if (numDims < 0) {
        ACL_LOG_ERROR("[Check][NumDims]numDims[%d] is smaller than 0", numDims);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"numDims", std::to_string(numDims).c_str(),
            "can't smaller than 0"}));
        return nullptr;
    }
    if ((numDims > 0) && (dims == nullptr)) {
        ACL_LOG_ERROR("[Check][Dims]dims is null while numDims[%d] > 0", numDims);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"numDims", std::to_string(numDims).c_str(),
            "dims is null while numDims > 0"}));
        return nullptr;
    }

    // 1 respresents that creates one tensor
    return new(std::nothrow) aclTensorDesc[1]{{dataType, static_cast<size_t>(numDims), dims, format}};
}

void aclDestroyTensorDescImpl(const aclTensorDesc *desc)
{
    ACL_PROFILING_REG(acl::AclProfType::AclDestroyTensorDesc);
    ACL_DELETE_ARRAY_AND_SET_NULL(desc);
}

aclError aclSetTensorShapeRangeImpl(aclTensorDesc* desc, size_t dimsCount, int64_t dimsRange[][ACL_TENSOR_SHAPE_RANGE_NUM])
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    // dimsCount should be equal to length of array dimsRange
    desc->shapeRange.clear();
    for (size_t i = 0U; i < dimsCount; ++i) {
        desc->shapeRange.emplace_back(std::make_pair(dimsRange[i][0], dimsRange[i][1]));
    }
    return ACL_SUCCESS;
}

aclError aclSetTensorValueRangeImpl(aclTensorDesc *desc, size_t valueCount,
    int64_t valueRange[][ACL_TENSOR_VALUE_RANGE_NUM])
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    desc->valRange.clear();
    for (size_t i = 0U; i < valueCount; ++i) {
        desc->valRange.emplace_back(std::make_pair(valueRange[i][0], valueRange[i][1]));
    }
    return ACL_SUCCESS;
}

aclDataType aclGetTensorDescTypeImpl(const aclTensorDesc *desc)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"desc"}));
        return ACL_DT_UNDEFINED;
    }

    return desc->dataType;
}

aclFormat aclGetTensorDescFormatImpl(const aclTensorDesc *desc)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"desc"}));
        return ACL_FORMAT_UNDEFINED;
    }

    return desc->format;
}

size_t aclGetTensorDescSizeImpl(const aclTensorDesc *desc)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"desc"}));
        return 0U;
    }
    size_t size = 0U;
    const size_t descCount = aclGetTensorDescElementCountImpl(desc);
    const size_t typeSize = aclDataTypeSize(desc->dataType);
    (void)acl::CheckSizeTMultiOverflow(descCount, typeSize, size);
    return size;
}

size_t aclGetTensorDescElementCountImpl(const aclTensorDesc *desc)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"desc"}));
        return 0U;
    }

    if (desc->dims.empty()) {
        return static_cast<size_t>(1);
    }

    size_t elementCount = 1U;
    for (const int64_t dim : desc->dims) {
        if (dim < 0) { // dim cannot be less than 0
            ACL_LOG_INNER_ERROR("[Check][Dim]invalid dim value %ld", dim);
            return 0U;
        }
        const aclError ret = acl::CheckSizeTMultiOverflow(elementCount, static_cast<size_t>(dim), elementCount);
        if (ret != ACL_SUCCESS) {
            return 0U;
        }
    }

    return elementCount;
}

size_t aclGetTensorDescNumDimsImpl(const aclTensorDesc *desc)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"desc"}));
        return 0U;
    }
    if ((desc->dims.size() > 0U) && (desc->dims[0U] == acl::UNKNOW_RANK)) {
        return static_cast<size_t>(ACL_UNKNOWN_RANK);
    }
    return desc->dims.size();
}

aclError aclGetTensorDescDimRangeImpl(const aclTensorDesc* desc,
                                  size_t index,
                                  size_t dimRangeNum,
                                  int64_t *dimRange)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dimRange);
    if (index >= desc->shapeRange.size()) {
        ACL_LOG_ERROR("[Check][Index]index out of range. index = %zu, numDims = %zu",
            index, desc->shapeRange.size());
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("index out of range. numDims = %zu",
            desc->shapeRange.size());
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"index", std::to_string(index).c_str(), errMsg.c_str()}));
        return ACL_ERROR_INVALID_PARAM;
    }
    if (dimRangeNum < static_cast<size_t>(ACL_TENSOR_SHAPE_RANGE_NUM)) {
        ACL_LOG_ERROR("[Check][DimRangeNum]dimRangeNum cannot be less than 2. dimRangeNum = %zu",
            dimRangeNum);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"dimRangeNum", std::to_string(dimRangeNum).c_str(),
            "cannot be less than 2"}));
        return ACL_ERROR_INVALID_PARAM;
    }
    *dimRange = desc->shapeRange[index].first;
    *(dimRange + 1) = desc->shapeRange[index].second;

    return ACL_SUCCESS;
}

int64_t aclGetTensorDescDimImpl(const aclTensorDesc *desc, size_t index)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"desc"}));
        return -1;
    }

    if (index >= desc->dims.size()) {
        ACL_LOG_ERROR("[Check][Index]index out of range. index = %zu, numDims = %zu",
            index, desc->dims.size());
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("index out of range. "
            "numDims = %zu", desc->dims.size());
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"index", std::to_string(index).c_str(), errMsg.c_str()}));
        return -1;
    }

    return desc->dims[index];
}

aclError aclGetTensorDescDimV2Impl(const aclTensorDesc *desc, size_t index, int64_t *dimSize)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}),
            std::vector<const char *>({"desc"}));
        return ACL_ERROR_INVALID_PARAM;
    }
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dimSize);
    if (index >= desc->dims.size()) {
        ACL_LOG_ERROR("[Check][Index]index out of range. index = %zu, numDims = %zu",
            index, desc->dims.size());
        const std::string errMsg = acl::AclErrorLogManager::FormatStr("index out of range. numDims = %zu",
            desc->dims.size());
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
            std::vector<const char *>({"param", "value", "reason"}),
            std::vector<const char *>({"index", std::to_string(index).c_str(), errMsg.c_str()}));
        return ACL_ERROR_INVALID_PARAM;
    }
    *dimSize = desc->dims[index];

    return ACL_SUCCESS;
}

void aclSetTensorDescNameImpl(aclTensorDesc *desc, const char *name)
{
    if (desc == nullptr) {
        ACL_LOG_ERROR("[Check][Desc]desc is null");
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}), std::vector<const char *>({"desc"}));
        return;
    }

    if (name == nullptr) {
        desc->name = "";
        return;
    }

    desc->name = std::string(name);
}

const char *aclGetTensorDescNameImpl(aclTensorDesc *desc)
{
    if (desc == nullptr) {
        return "";
    }

    return desc->name.c_str();
}

aclError aclTransTensorDescFormatImpl(const aclTensorDesc *srcDesc, aclFormat dstFormat, aclTensorDesc **dstDesc)
{
    ACL_PROFILING_REG(acl::AclProfType::AclTransTensorDescFormat);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(srcDesc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dstDesc);

    std::vector<int64_t> dims;
    acl::ConvertSvecToVec(srcDesc->dims, dims);
    const ge::Shape shape(dims);
    const auto srcFormat = static_cast<ge::Format>(srcDesc->format);
    const auto dataType = static_cast<ge::DataType >(srcDesc->dataType);
    const ge::TensorDesc desc(shape, srcFormat, dataType);

    std::vector<int64_t> dstShape;
    const auto geRet = ge::GeFormatUtil::TransShape(desc, static_cast<ge::Format>(dstFormat), dstShape);
    if (geRet != ge::SUCCESS) {
        ACL_LOG_CALL_ERROR("[Call][TransShape]invoke TransShape failed. ge result = %u",
                           geRet);
        return ACL_GET_ERRCODE_GE(geRet);
    }

    *dstDesc = aclCreateTensorDescImpl(srcDesc->dataType, static_cast<int32_t>(dstShape.size()),
                                   dstShape.data(), srcDesc->format);
    if (*dstDesc == nullptr) {
        ACL_LOG_INNER_ERROR("[Create][Desc]aclCreateTensorDesc failed.");
        return ACL_ERROR_BAD_ALLOC;
    }

    return ACL_SUCCESS;
}

aclError aclSetTensorStorageFormatImpl(aclTensorDesc *desc, aclFormat format)
{
    if (desc != nullptr) {
        desc->storageFormat = format;
    }

    return ACL_SUCCESS;
}

aclError aclSetTensorStorageShapeImpl(aclTensorDesc *desc, int numDims, const int64_t *dims)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);

    desc->storageDims.clear();
    for (int32_t i = 0; i < static_cast<int32_t>(numDims); ++i) {
        desc->storageDims.push_back(*(dims + i));
    }

    return ACL_SUCCESS;
}

aclError aclSetTensorFormatImpl(aclTensorDesc *desc, aclFormat format)
{
    if (desc != nullptr) {
        desc->storageFormat = format;
    }

    return ACL_SUCCESS;
}

aclError aclSetTensorShapeImpl(aclTensorDesc *desc, int numDims, const int64_t *dims)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);

    desc->storageDims.clear();
    for (int32_t i = 0; i < static_cast<int32_t>(numDims); ++i) {
        desc->storageDims.push_back(*(dims + i));
    }

    return ACL_SUCCESS;
}

aclError aclSetTensorOriginFormatImpl(aclTensorDesc *desc, aclFormat format)
{
    if (desc != nullptr) {
        desc->format = format;
    }

    return ACL_SUCCESS;
}

aclError aclSetTensorOriginShapeImpl(aclTensorDesc *desc, int numDims, const int64_t *dims)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dims);

    desc->dims.clear();
    for (int32_t i = 0; i < numDims; ++i) {
        desc->dims.push_back(*(dims + i));
    }

    return ACL_SUCCESS;
}

aclTensorDesc *aclGetTensorDescByIndexImpl(aclTensorDesc *desc, size_t index)
{
    ACL_REQUIRES_NOT_NULL_RET_NULL_INPUT_REPORT(desc);
    return (desc + index);
}

void *aclGetTensorDescAddressImpl(const aclTensorDesc *desc)
{
    ACL_REQUIRES_NOT_NULL_RET_NULL_INPUT_REPORT(desc);
    return desc->address;
}

aclError aclSetTensorDynamicInputImpl(aclTensorDesc *desc, const char *dynamicInputName)
{
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dynamicInputName);

    desc->dynamicInputName = std::string(dynamicInputName);
    return ACL_SUCCESS;
}

aclError aclSetTensorConstImpl(aclTensorDesc *desc, void *dataBuffer, size_t length)
{
    ACL_LOG_INFO("start to execute aclSetTensorConst");
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(desc);
    ACL_REQUIRES_NOT_NULL_WITH_INPUT_REPORT(dataBuffer);
    if (length == 0U) {
        ACL_LOG_ERROR("[Check][Length]The length of const dataBuffer is invalid. size = %zu", length);
        acl::AclErrorLogManager::ReportInputError(acl::INVALID_NULL_POINTER_MSG,
            std::vector<const char *>({"param"}),
            std::vector<const char *>({"desc"}));
        return ACL_ERROR_INVALID_PARAM;
    }
    desc->isConst = true;

    auto *constData = new (std::nothrow) char[length];
    ACL_REQUIRES_NOT_NULL(constData);
    if (memcpy_s(constData, length, dataBuffer, length) != EOK) {
        ACL_LOG_INNER_ERROR("[Copy][Data]Copy const data failed. size = %zu", length);
        ACL_DELETE_ARRAY_AND_SET_NULL(constData);
        return ACL_ERROR_FAILURE;
    }

    desc->constDataBuf.reset(constData, [](const char_t* const ptr)
        { delete[]ptr; ACL_LOG_DEBUG("delete const data in aclSetTensorConst"); });
    desc->constDataLen = length;
    return ACL_SUCCESS;
}

aclError aclSetTensorPlaceMentImpl(aclTensorDesc *desc, aclMemType memType)
{
    ACL_LOG_INFO("start to execute aclSetTensorPlaceMent, memType is %d", static_cast<int32_t>(memType));
    ACL_REQUIRES_NOT_NULL(desc);
    desc->memtype = memType;
    return ACL_SUCCESS;
}

void aclTensorDesc::BackupDimsAndShapeRanges()
{
    dimsBackup = dims;
    storageDimsBackup = storageDims;
    shapeRangeBackup = shapeRange;
}

void aclTensorDesc::RecoverDimsAndShapeRanges()
{
    dims = dimsBackup;
    storageDims = storageDimsBackup;
    shapeRange = shapeRangeBackup;
}

void aclTensorDesc::BackupConst()
{
    isConstBackup = isConst;
    constDataBufBackup = constDataBuf;
    constDataLenBackup = constDataLen;
}

void aclTensorDesc::RecoverConst()
{
    isConst = isConstBackup;
    constDataBuf = constDataBufBackup;
    constDataLen = constDataLenBackup;
}

