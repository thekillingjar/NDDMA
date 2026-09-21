/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "op_compiler.h"
#include <string>
#include <map>
#include <chrono>
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils_inner.h"
#include "common/ge_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "utils/array_utils.h"

namespace {
    const std::map<ge::GeAttrValue::ValueType,  std::string> ATTR_TYPES_MAP = {
        {ge::GeAttrValue::VT_NONE, "string"},
        {ge::GeAttrValue::VT_STRING, "string"},
        {ge::GeAttrValue::VT_FLOAT, "float"},
        {ge::GeAttrValue::VT_BOOL, "bool"},
        {ge::GeAttrValue::VT_INT, "int"},
        {ge::GeAttrValue::VT_DATA_TYPE, "datatype"},
        {ge::GeAttrValue::VT_LIST_STRING, "liststring"},
        {ge::GeAttrValue::VT_LIST_FLOAT, "listfloat"},
        {ge::GeAttrValue::VT_LIST_BOOL, "listbool"},
        {ge::GeAttrValue::VT_LIST_INT, "listint"},
        {ge::GeAttrValue::VT_LIST_DATA_TYPE, "listdatatype"},
    };
    int32_t opCompileJitCompile = 1;
    int32_t globalCompileFlag = 0;
}

namespace acl {
void SetGlobalCompileFlag(const int32_t flag)
{
    globalCompileFlag = flag;
}

int32_t GetGlobalCompileFlag()
{
    return globalCompileFlag;
}

void SetGlobalJitCompileFlag(const int32_t flag)
{
    opCompileJitCompile = flag;
}

int32_t GetGlobalJitCompileFlag()
{
    return opCompileJitCompile;
}

static void MakeHostMemTensor(const aclTensorDesc *const desc, const aclDataBuffer *const dataBuffer,
                              const int32_t compileFlag, ge::GeTensorDesc &geTensorDesc)
{
    if (((desc->memtype == ACL_MEMTYPE_HOST) || (desc->memtype == ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)) &&
        (!desc->isConst)) {
        if ((compileFlag != 0) || (GetGlobalJitCompileFlag() != 1) ||
            (desc->memtype == ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT)) {
            // During fuzzy compilation or ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT, change hostMem to data input.
            ACL_LOG_INFO("compleFlag is ACL_OP_COMPILE_FUZZ or memtype is ACL_MEMTYPE_HOST_COMPILE_INDEPENDENT, "
                "change hostMem to data.");
            ge::ConstGeTensorPtr dataTensor = nullptr;
            ACL_MAKE_SHARED(dataTensor = std::make_shared<ge::GeTensor>(geTensorDesc,
                static_cast<uint8_t *>(dataBuffer->data), dataBuffer->length), ;);
            (void)ge::AttrUtils::SetTensor(geTensorDesc, ge::ATTR_NAME_VALUE, dataTensor);
        } else {
            // During static compilation, change hostMem to const input.
            ACL_LOG_INFO("compleFlag is ACL_OP_COMPILE_DEFAULT and memtype is ACL_MEMTYPE_HOST, "
                "change hostMem to const.");
            (void)ge::AttrUtils::SetBool(geTensorDesc, ge::CONST_ATTR_NAME_INPUT, true);
            ge::ConstGeTensorPtr constTensor = nullptr;
            ACL_MAKE_SHARED(constTensor = std::make_shared<ge::GeTensor>(geTensorDesc,
                static_cast<uint8_t *>(dataBuffer->data), dataBuffer->length), ;);
            (void)ge::AttrUtils::SetTensor(geTensorDesc, ge::ATTR_NAME_WEIGHTS, constTensor);
        }
    }
}

static void OptimizeTensorDescForTransdata(const AclOp &op, const bool isInput, ge::GeTensorDesc &geTensorDesc)
{
    if ((op.opType != "TransData") ||
        (op.inputDesc[0] == nullptr) || (op.outputDesc[0] == nullptr)) {
        return;
    }
    if ((op.inputDesc[0]->storageFormat == ACL_FORMAT_UNDEFINED) &&
        (op.outputDesc[0]->storageFormat == ACL_FORMAT_UNDEFINED)) {
        // This TransData is not set storageFormat, old process
        ge::Format inOriFormat = ge::FORMAT_RESERVED;
        if (op.inputDesc[0]->format != ACL_FORMAT_UNDEFINED) {
            inOriFormat = static_cast<::ge::Format>(op.inputDesc[0]->format);
        }
        ge::Format outOriFormat = ge::FORMAT_RESERVED;
        if (op.outputDesc[0]->format != ACL_FORMAT_UNDEFINED) {
            outOriFormat = static_cast<::ge::Format>(op.outputDesc[0]->format);
        }
        ACL_LOG_INFO("Find input origin format %d, output origin format %d", inOriFormat, outOriFormat);
        ge::Format transdataOriFormat = ge::FORMAT_RESERVED;
        // if output is oringin format,input is not, need update input
        if ((ge::TypeUtilsInner::IsInternalFormat(inOriFormat)) &&
            (!ge::TypeUtilsInner::IsInternalFormat(outOriFormat))) {
            transdataOriFormat = outOriFormat;
            if (isInput) {
                geTensorDesc.SetOriginFormat(transdataOriFormat);
                std::vector<int64_t> dims;
                ConvertSvecToVec(op.outputDesc[0]->dims, dims);
                geTensorDesc.SetOriginShape(ge::GeShape(dims));
            }
        // update output
        } else {
            transdataOriFormat = inOriFormat;
            if (!isInput) {
                geTensorDesc.SetOriginFormat(transdataOriFormat);
                std::vector<int64_t> dims;
                ConvertSvecToVec(op.inputDesc[0]->dims, dims);
                geTensorDesc.SetOriginShape(ge::GeShape(dims));
            }
        }
        ACL_LOG_INFO("Find origin format is %d", transdataOriFormat);
    }
    return;
}

static aclError MakeInputCompileParam(const AclOp &op, CompileParam &param,
                                      ge::OpDesc *const opDesc, const int32_t compileFlag)
{
    for (int32_t i = 0; i < op.numInputs; ++i) {
        const aclTensorDesc *const desc = op.inputDesc[i];
        if (!desc->CheckShapeRange()) {
            ACL_LOG_ERROR("[Check][InputShapeRange]Dims of shape is not equal to dims of shape range in input[%d]", i);
            acl::AclErrorLogManager::ReportInputError(acl::INVALID_PARAM_MSG,
                std::vector<const char *>({"param", "value", "reason"}),
                std::vector<const char *>({"InputShapeRange", ("index[" + std::to_string(i) + "]").c_str(),
                ("the number of shapeRange is not equal to number of dims in input[" + std::to_string(i) + "].").c_str()}));
            return ACL_ERROR_INVALID_PARAM;
        }
        ge::Format geFormat = ge::FORMAT_RESERVED;
        if (desc->format != ACL_FORMAT_UNDEFINED) {
            geFormat = static_cast<::ge::Format>(desc->format);
        }
        ge::DataType geDataType = ge::DT_UNDEFINED;
        if (desc->dataType != ACL_DT_UNDEFINED) {
            geDataType = static_cast<::ge::DataType>(desc->dataType);
        }
        std::vector<int64_t> dims;
        ConvertSvecToVec(desc->dims, dims);
        ge::GeTensorDesc geTensorDesc(ge::GeShape(dims),
                                  geFormat,
                                  geDataType);
        geTensorDesc.SetOriginFormat(geFormat);
        geTensorDesc.SetOriginShape(ge::GeShape(dims));
        if (op.opType == "TransData") {
            ACL_LOG_INFO("This op is TransData of input");
            if (desc->storageFormat != ACL_FORMAT_UNDEFINED) {
                std::vector<int64_t> storageDims;
                ConvertSvecToVec(desc->storageDims, storageDims);
                ACL_LOG_INFO("TransData create aclop inputDesc");
                geTensorDesc.SetShape(ge::GeShape(storageDims));
                geTensorDesc.SetFormat(static_cast<::ge::Format>(desc->storageFormat));
                geTensorDesc.SetDataType(geDataType);
                geTensorDesc.SetOriginDataType(geDataType);
            }
        }
        if (geTensorDesc.SetShapeRange(desc->shapeRange) != ge::GRAPH_SUCCESS) {
            ACL_LOG_INNER_ERROR("set shape range fail, opType: %s", op.opType.c_str());
            return ACL_ERROR_GE_FAILURE;
        }
        (void)ge::AttrUtils::SetInt(geTensorDesc, ge::ATTR_NAME_PLACEMENT, static_cast<int64_t>(desc->memtype));
        ge::TensorUtils::SetRealDimCnt(geTensorDesc, static_cast<uint32_t>(desc->dims.size()));
        ge::TensorUtils::SetInputTensor(geTensorDesc, true);
        ge::TensorUtils::SetOutputTensor(geTensorDesc, false);
        if (desc->storageFormat != ACL_FORMAT_UNDEFINED) {
            (void)ge::AttrUtils::SetInt(geTensorDesc, ge::ATTR_NAME_STORAGE_FORMAT,
                static_cast<int64_t>(desc->storageFormat));
            std::vector<int64_t> storageDims;
            ConvertSvecToVec(desc->storageDims, storageDims);
            (void)ge::AttrUtils::SetListInt(geTensorDesc, ge::ATTR_NAME_STORAGE_SHAPE, storageDims);
        }
        if ((op.inputs != nullptr) && (op.inputs[i] != nullptr)) {
            MakeHostMemTensor(desc, op.inputs[i], compileFlag, geTensorDesc);
        }

        if (desc->isConst) {
            (void)ge::AttrUtils::SetBool(geTensorDesc, ge::CONST_ATTR_NAME_INPUT, true);
            ge::ConstGeTensorPtr constTensor = nullptr;
            ACL_MAKE_SHARED(constTensor = std::make_shared<ge::GeTensor>(geTensorDesc,
                static_cast<uint8_t *>(desc->constDataBuf.get()), desc->constDataLen), ;);
            (void)ge::AttrUtils::SetTensor(geTensorDesc, ge::ATTR_NAME_WEIGHTS, constTensor);
        }

        OptimizeTensorDescForTransdata(op, true, geTensorDesc);
        if (!desc->valRange.empty()) {
            (void)geTensorDesc.SetValueRange(desc->valRange);
        }
        if (!desc->name.empty()) {
            (void)opDesc->AddInputDesc(desc->name, geTensorDesc);
        } else {
            (void)opDesc->AddInputDesc(geTensorDesc);
        }
        param.inputs.emplace_back(geTensorDesc);
    }
    return ACL_SUCCESS;
}

static aclError MakeOutputCompileParam(const AclOp &op, CompileParam &param,
                                       ge::OpDesc *const opDesc, const int32_t compileFlag)
{
    for (int32_t i = 0; i < op.numOutputs; ++i) {
        const aclTensorDesc *const desc = op.outputDesc[i];
        const ge::Format geFormat = (desc->format == ACL_FORMAT_UNDEFINED) ? ge::FORMAT_RESERVED : \
                              static_cast<::ge::Format>(desc->format);
        const ge::DataType geDataType = (desc->dataType == ACL_DT_UNDEFINED) ? ge::DT_UNDEFINED : \
                                  static_cast<::ge::DataType>(desc->dataType);
        std::vector<int64_t> dims;
        ConvertSvecToVec(desc->dims, dims);
        ge::GeTensorDesc geTensorDesc(ge::GeShape(dims), geFormat, geDataType);
        geTensorDesc.SetOriginFormat(geFormat);
        geTensorDesc.SetOriginShape(ge::GeShape(dims));
        if (op.opType == "TransData") {
            ACL_LOG_INFO("This op is TransData of output");
            if (desc->storageFormat != ACL_FORMAT_UNDEFINED) {
                ACL_LOG_INFO("TransData create aclop outputDesc");
                std::vector<int64_t> storageDims;
                ConvertSvecToVec(desc->storageDims, storageDims);
                geTensorDesc.SetShape(ge::GeShape(storageDims));
                geTensorDesc.SetFormat(static_cast<::ge::Format>(desc->storageFormat));
                geTensorDesc.SetDataType(geDataType);
                geTensorDesc.SetOriginDataType(geDataType);
            }
        }
        if (geTensorDesc.SetShapeRange(desc->shapeRange) != ge::GRAPH_SUCCESS) {
            ACL_LOG_INNER_ERROR("set shape range fail, opType: %s", op.opType.c_str());
            return ACL_ERROR_GE_FAILURE;
        }

        (void)ge::AttrUtils::SetInt(geTensorDesc, ge::ATTR_NAME_PLACEMENT, static_cast<int64_t>(desc->memtype));
        ge::TensorUtils::SetRealDimCnt(geTensorDesc, static_cast<uint32_t>(desc->dims.size()));
        ge::TensorUtils::SetInputTensor(geTensorDesc, false);
        ge::TensorUtils::SetOutputTensor(geTensorDesc, true);
        if (desc->storageFormat != ACL_FORMAT_UNDEFINED) {
            (void)ge::AttrUtils::SetInt(geTensorDesc, ge::ATTR_NAME_STORAGE_FORMAT,
                static_cast<int64_t>(desc->storageFormat));
            std::vector<int64_t> storageDims;
            ConvertSvecToVec(desc->storageDims, storageDims);
            (void)ge::AttrUtils::SetListInt(geTensorDesc, ge::ATTR_NAME_STORAGE_SHAPE, storageDims);
        }
        if ((op.outputs != nullptr) && (op.outputs[i] != nullptr)) {
            MakeHostMemTensor(desc, op.outputs[i], compileFlag, geTensorDesc);
        }
        if (desc->isConst) {
            (void)ge::AttrUtils::SetBool(geTensorDesc, ge::CONST_ATTR_NAME_INPUT, true);
            ge::ConstGeTensorPtr constTensor = nullptr;
            ACL_MAKE_SHARED(constTensor = std::make_shared<ge::GeTensor>(geTensorDesc,
                static_cast<uint8_t *>(desc->constDataBuf.get()), desc->constDataLen), ;);
            (void)ge::AttrUtils::SetTensor(geTensorDesc, ge::ATTR_NAME_WEIGHTS, constTensor);
        }

        OptimizeTensorDescForTransdata(op, false, geTensorDesc);
        if (!desc->valRange.empty()) {
            (void)geTensorDesc.SetValueRange(desc->valRange);
        }
        if (!desc->name.empty()) {
            (void)opDesc->AddOutputDesc(desc->name, geTensorDesc);
        } else {
            (void)opDesc->AddOutputDesc(geTensorDesc);
        }
        param.outputs.emplace_back(geTensorDesc);
    }
    return ACL_SUCCESS;
}

aclError OpCompiler::MakeCompileParam(const AclOp &op, CompileParam &param, const int32_t compileFlag)
{
    ge::OpDescPtr opDesc = nullptr;
    static std::atomic<uint64_t> id{0UL};
    std::string opName = op.opType + std::to_string(++id);
    ACL_MAKE_SHARED(opDesc = std::make_shared<ge::OpDesc>(opName, op.opType), return ACL_ERROR_BAD_ALLOC);
    ACL_CHECK_MALLOC_RESULT(opDesc);

    const aclError inputRet = MakeInputCompileParam(op, param, opDesc.get(), compileFlag);
    if (inputRet != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("make input compile param failed, result = %d", inputRet);
        return inputRet;
    }

    const aclError outputRet = MakeOutputCompileParam(op, param, opDesc.get(), compileFlag);
    if (outputRet != ACL_SUCCESS) {
        ACL_LOG_INNER_ERROR("make output compile param failed, result = %d", outputRet);
        return outputRet;
    }

    std::string attrTypeList;
    if (op.opAttr != nullptr) {
        for (const auto &it : op.opAttr->Attrs()) {
            (void)opDesc->SetAttr(it.first, it.second);
            if (op.compileType == OP_COMPILE_UNREGISTERED) {
                const ge::GeAttrValue::ValueType valType = it.second.GetValueType();
                const auto valTypeIt = ATTR_TYPES_MAP.find(valType);
                if (valTypeIt == ATTR_TYPES_MAP.end()) {
                    ACL_LOG_INNER_ERROR("Invalid attr value type, valType: %d", static_cast<int32_t>(valType));
                    return ACL_ERROR_INVALID_PARAM;
                }
                (void)attrTypeList.append(it.first).append(":")
                            .append(valTypeIt->second).append(";");
            }
        }
        // delete ; from end of attrTypeList
        if (attrTypeList.length() != 0U) {
            attrTypeList = attrTypeList.substr(0U, attrTypeList.length() - 1U);
        }
    }

    if (op.compileType == OP_COMPILE_UNREGISTERED) {
        (void)ge::AttrUtils::SetStr(opDesc, ge::ATTR_NAME_UNREGST_OPPATH, op.opPath);
        (void)ge::AttrUtils::SetStr(opDesc, ge::ATTR_NAME_UNREGST_ATTRLIST, attrTypeList);
    }

    // set dynamic input attr
    array_utils::DynamicInputIndexPair indexPair;
    const bool ret = array_utils::GetDynamicInputIndex(op.numInputs, op.inputDesc, indexPair);
    if (!ret) {
        ACL_LOG_ERROR("Can not get dynamic input index, invalid dynamic input attr, op type: %s",
            op.opType.c_str());
        return ACL_ERROR_INVALID_PARAM;
    }

    if (indexPair.first.size() > 0U) {
        (void)ge::AttrUtils::SetListInt(opDesc, ge::ATTR_NAME_DYNAMIC_INPUT_START, indexPair.first);
        (void)ge::AttrUtils::SetListInt(opDesc, ge::ATTR_NAME_DYNAMIC_INPUT_END, indexPair.second);
    }

    param.opDesc = std::move(opDesc);
    param.engineType = static_cast<ge::OpEngineType>(op.engineType);
    param.compileFlag = compileFlag;
    return ACL_SUCCESS;
}

aclError OpCompiler::CompileOp(const AclOp &op, std::shared_ptr<void> &modelData, size_t &modelSize)
{
    const int32_t compileFlag = GetGlobalCompileFlag();
    ACL_LOG_INFO("To compile op: %s, compileFlag is %d", op.opType.c_str(), compileFlag);
    CompileParam param;
    ACL_REQUIRES_OK(MakeCompileParam(op, param, compileFlag));
    const auto start = std::chrono::steady_clock::now();
    ACL_REQUIRES_OK(DoCompile(param, modelData, modelSize));
    const auto end = std::chrono::steady_clock::now();
    ACL_LOG_EVENT("To compile op: %s, compile time is %lf millisecond",
                  op.opType.c_str(), std::chrono::duration<double, std::milli>(end - start).count());
    return ACL_SUCCESS;
}

aclError OpCompiler::CompileAndDumpGraph(const AclOp &op, const char_t *const graphDumpPath,
    const aclGraphDumpOption *const dumpOpt)
{
    const int32_t compileFlag = GetGlobalCompileFlag();
    ACL_LOG_INFO("To compile op: %s, compileFlag is %d", op.opType.c_str(), compileFlag);
    CompileParam param;
    ACL_REQUIRES_OK(MakeCompileParam(op, param, compileFlag));
    ACL_REQUIRES_OK(GenGraphAndDump(param, graphDumpPath, dumpOpt));
    return ACL_SUCCESS;
}
} // namespace acl

