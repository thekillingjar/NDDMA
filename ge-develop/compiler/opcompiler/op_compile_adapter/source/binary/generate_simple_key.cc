/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary/generate_simple_key.h"
#include <unordered_map>
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_util_constants.h"
#include "graph/ascend_string.h"
#include "graph/utils/type_utils.h"
#include "register/op_check.h"
#include "exe_graph/lowering/kernel_run_context_builder.h"

#include <registry/op_impl_space_registry_v2.h>

namespace te {
namespace fusion {
using namespace ge;
enum OpImplModeInt {
    DEFAULT = 0,
    HIGH_PERFORMANCE = 1,
    HIGH_PRECISION = 2,
    SUPER_PERFORMANCE = 3,
    SUPPORT_OUT_OF_BOUND_INDEX = 4,
    ENABLE_FLOAT_32 = 5,
    ENABLE_HI_FLOAT_32 = 6,
    KEEP_FP_16 = 7
};

static const std::unordered_map<std::string, ge::DataType> DTYPE_STR_TO_INT {
                    {"float32", ge::DataType::DT_FLOAT},                   
                    {"float64", ge::DataType::DT_DOUBLE},
                };

static const std::unordered_map<std::string, ge::Format> FORMAT_STR_TO_INT {
                    {"NCHW", ge::Format::FORMAT_NCHW},
                    {"NHWC", ge::Format::FORMAT_NHWC},
                    {"ND", ge::Format::FORMAT_ND},
                    {"NC1HWC0", ge::Format::FORMAT_NC1HWC0},
                    {"FRACTAL_Z", ge::Format::FORMAT_FRACTAL_Z},
                    {"NC1C0HWPAD", ge::Format::FORMAT_NC1C0HWPAD},
                    {"NHWC1C0", ge::Format::FORMAT_NHWC1C0},
                    {"FSR_NCHW", ge::Format::FORMAT_FSR_NCHW},
                    {"FRACTAL_DECONV", ge::Format::FORMAT_FRACTAL_DECONV},
                    {"C1HWNC0", ge::Format::FORMAT_C1HWNC0},
                    {"FRACTAL_DECONV_TRANSPOSE", ge::Format::FORMAT_FRACTAL_DECONV_TRANSPOSE},
                    {"FRACTAL_DECONV_SP_STRIDE_TRANS", ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
                    {"NC1HWC0_C04", ge::Format::FORMAT_NC1HWC0_C04},
                    {"FRACTAL_Z_C04", ge::Format::FORMAT_FRACTAL_Z_C04},
                    {"CHWN", ge::Format::FORMAT_CHWN},
                    {"FRACTAL_DECONV_SP_STRIDE8_TRANS", ge::Format::FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
                    {"HWCN", ge::Format::FORMAT_HWCN},
                    {"NC1KHKWHWC0", ge::Format::FORMAT_NC1KHKWHWC0},
                    {"BN_WEIGHT", ge::Format::FORMAT_BN_WEIGHT},
                    {"FILTER_HWCK", ge::Format::FORMAT_FILTER_HWCK},
                    {"HASHTABLE_LOOKUP_LOOKUPS", ge::Format::FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
                    {"HASHTABLE_LOOKUP_KEYS", ge::Format::FORMAT_HASHTABLE_LOOKUP_KEYS},
                    {"HASHTABLE_LOOKUP_VALUE", ge::Format::FORMAT_HASHTABLE_LOOKUP_VALUE},
                    {"HASHTABLE_LOOKUP_OUTPUT", ge::Format::FORMAT_HASHTABLE_LOOKUP_OUTPUT},
                    {"HASHTABLE_LOOKUP_HITS", ge::Format::FORMAT_HASHTABLE_LOOKUP_HITS},
                    {"C1HWNCoC0", ge::Format::FORMAT_C1HWNCoC0},
                    {"MD", ge::Format::FORMAT_MD},
                    {"NDHWC", ge::Format::FORMAT_NDHWC},
                    {"FRACTAL_ZZ", ge::Format::FORMAT_FRACTAL_ZZ},
                    {"FRACTAL_NZ", ge::Format::FORMAT_FRACTAL_NZ},
                    {"NCDHW", ge::Format::FORMAT_NCDHW},
                    {"DHWCN", ge::Format::FORMAT_DHWCN},
                    {"NDC1HWC0", ge::Format::FORMAT_NDC1HWC0},
                    {"FRACTAL_Z_3D", ge::Format::FORMAT_FRACTAL_Z_3D},
                    {"CN", ge::Format::FORMAT_CN},
                    {"NC", ge::Format::FORMAT_NC},
                    {"DHWNC", ge::Format::FORMAT_DHWNC},
                    {"FRACTAL_Z_3D_TRANSPOSE", ge::Format::FORMAT_FRACTAL_Z_3D_TRANSPOSE},
                    {"FRACTAL_ZN_LSTM", ge::Format::FORMAT_FRACTAL_ZN_LSTM},
                    {"FRACTAL_Z_G", ge::Format::FORMAT_FRACTAL_Z_G},
                    {"RESERVED", ge::Format::FORMAT_RESERVED},
                    {"ALL", ge::Format::FORMAT_ALL},
                    {"NULL", ge::Format::FORMAT_NULL},
                    {"ND_RNN_BIAS", ge::Format::FORMAT_ND_RNN_BIAS},
                    {"FRACTAL_ZN_RNN", ge::Format::FORMAT_FRACTAL_ZN_RNN},
                    {"NYUV", ge::Format::FORMAT_NYUV},
                    {"NYUV_A", ge::Format::FORMAT_NYUV_A},
                    {"NCL", ge::Format::FORMAT_NCL},
                    {"FRACTAL_Z_WINO", ge::Format::FORMAT_FRACTAL_Z_WINO},
                    {"C1HWC0", ge::Format::FORMAT_C1HWC0},
                    {"FRACTAL_NZ_C0_16", ge::Format::FORMAT_FRACTAL_NZ_C0_16},
                    {"FRACTAL_NZ_C0_32", ge::Format::FORMAT_FRACTAL_NZ_C0_32}
                };

static const std::unordered_map<std::string, OpImplModeInt> IMPL_MODE_STR_TO_INT {
                    {"default", DEFAULT},
                    {"high_performance", HIGH_PERFORMANCE},
                    {"high_precision", HIGH_PRECISION},
                    {"super_performance", SUPER_PERFORMANCE},
                    {"support_out_of_bound_index", SUPPORT_OUT_OF_BOUND_INDEX},
                    {"enable_float_32_execution", ENABLE_FLOAT_32},
                    {"enable_hi_float_32_execution", ENABLE_HI_FLOAT_32},
                    {"keep_fp16", KEEP_FP_16}
                };

static const std::set<std::string> DTYPE_MODE_64_SET {"int64", "uint64", "float64", "double", "complex64"};
static const std::set<std::string> DTYPE_MODE_32_SET {"int32", "uint32", "float32", "float", "complex32"};
static const std::set<std::string> DTYPE_MODE_16_SET {"int16", "float16", "bfloat16", "uint16"};
static const std::set<std::string> DTYPE_MODE_8_SET {"int8", "uint8", "bool"};
static const std::set<std::string> FORMAT_MODE_PRIVATE_FORMAT_SET {"NDHWC", "NCDHW", "DHWCN",
                                                                   "NHWC", "NCHW", "HWCN"};
static const std::map<int, std::string> DTYPE_MODE_BIT_MAP {
    {8, "int64"}, {4, "int32"}, {2, "int16"}, {1, "int8"}, {1006, "float6_e3m2"}, {1004, "float4_e2m1"}};
constexpr size_t MAX_CUSTOMIZED_SIMPLIFIED_KEY_LEN = 256;

namespace {
void TransStringToDtype(const string &dtypeStr, ge::DataType &dtype) {
  string feDtypeStr = dtypeStr;
  transform(feDtypeStr.begin(), feDtypeStr.end(), feDtypeStr.begin(), ::toupper);
  std::string geDtypeString = "DT_" + feDtypeStr;
  dtype = ge::TypeUtils::SerialStringToDataType(geDtypeString);
}

std::string DTypeTransToSimple(const std::string dataType)
{
    std::string dtype;
    auto iter = DTYPE_STR_TO_INT.find(dataType);
    if (iter != DTYPE_STR_TO_INT.end()) {
        dtype = std::to_string(iter->second);
    } else {
        TE_WARNLOG("dtypeStr = [%s] is not found in the dataType map, will continue the translation using ge_dtype_map.", dataType.c_str());
        ge::DataType geDtype;
        TransStringToDtype(dataType, geDtype);
        dtype = std::to_string(geDtype);
    }
    return dtype;
}

std::string FormatTransToSimple(const std::string formatStr)
{
    std::string format;
    auto iter = FORMAT_STR_TO_INT.find(formatStr);
    if (iter != FORMAT_STR_TO_INT.end()) {
        format = std::to_string(iter->second);
    } else {
        TE_ERRLOG("formatStr = [%s], not in format map.", formatStr.c_str());
    }
    return format;
}
}

void GenerateSimpleKey::SetInputs(const std::vector<TbeOpParam> &inputs)
{
    inputs_.assign(inputs.begin(), inputs.end());
}

void GenerateSimpleKey::SetOutputs(const std::vector<TbeOpParam> &outputs)
{
    outputs_.assign(outputs.begin(), outputs.end());
}

void GenerateSimpleKey::SetAttrs(const std::vector<TbeAttrValue> &attrs)
{
    attrs_.assign(attrs.begin(), attrs.end());
}

void GenerateSimpleKey::SetBinaryInfoPtr(const BinaryInfoBasePtr binaryInfo)
{
    binaryInfo_ = binaryInfo;
}

void GenerateSimpleKey::SetOpInfoPtr(const TbeOpInfoPtr &opInfo)
{
    opInfo_ = opInfo;
}

std::string GenerateSimpleKey::GenerateImplMode(const std::string implMode) const
{
    std::string implModeStr;
    auto iter = IMPL_MODE_STR_TO_INT.find(implMode);
    if (iter != IMPL_MODE_STR_TO_INT.end()) {
        implModeStr = "p=" + std::to_string(iter->second);
    }
    return implModeStr;
}

bool GenerateSimpleKey::FormatNormalize(const int index, const DtypeFormatMode &mode, std::string &format) const
{
    std::string formatMode = mode.formatMode[index];
    if (formatMode == FORMAT_MODE_NORMAL) {
        return true;
    } else if (formatMode == FORMAT_MODE_AGNOSTIC) {
        format = "ND";
    } else if (formatMode == FORMAT_MODE_ND_AGNOSTIC) {
        auto iter = FORMAT_MODE_PRIVATE_FORMAT_SET.find(format);
        if (iter != FORMAT_MODE_PRIVATE_FORMAT_SET.end()) {
            format = "ND";
        }
    } else if (formatMode == FORMAT_STATIC_ND_AGNOSTIC) {
        if (isUnknowShape_) {  // dynamic
            format = "ND";
        } else { // static
            auto iter = FORMAT_MODE_PRIVATE_FORMAT_SET.find(format);
            if (iter != FORMAT_MODE_PRIVATE_FORMAT_SET.end()) {
                format = "ND";
            }
        }
    }
    return true;
}

bool GenerateSimpleKey::DtypeNormalize(const int index, const DtypeFormatMode &mode, std::string &dtype) const
{
    std::string dtypeMode = mode.dTypeMode[index];
    if (dtypeMode == DTYPE_MODE_NORMAL) {
        return true;
    } else if (dtypeMode == DTYPE_MODE_BIT) {
        ge::DataType geDtype;
        TransStringToDtype(dtype, geDtype);
        std::int32_t dateSize = ge::GetSizeByDataType(geDtype);
        auto iter = DTYPE_MODE_BIT_MAP.find(dateSize);
        if (iter != DTYPE_MODE_BIT_MAP.end()) {
            dtype = iter->second;
            return true;
        }
        TE_ERRLOG("opType [%s], dtype = [%s].", opType_.c_str(), dtype.c_str());
        return false;
    } else if (dtypeMode == DTYPE_MODE_BOOL) {
        if (dtype == "bool") {
            dtype = "int8";
            return true;
        }
        TE_ERRLOG("dtype mode is bool. opType [%s], dtype = [%s].", opType_.c_str(), dtype.c_str());
        return false;
    }
    return false;
}

bool GenerateSimpleKey::Normalization(const int index, std::string putsType,
                                      std::string &format, std::string &dtype) const
{
    DtypeFormatMode mode;
    if (putsType == INPUTS) {
        if (!binaryInfo_->GetInputMode(opType_, mode)) {
            TE_ERRLOG("opType [%s], inputMode not found in inputMode map.", opType_.c_str());
            return false;
        }
    } else if (putsType == OUTPUTS) {
        if (!binaryInfo_->GetOutputMode(opType_, mode)) {
            TE_ERRLOG("opType [%s], inputMode not found in outputMode map.", opType_.c_str());
            return false;
        }
    }
    TE_DBGLOG("before normalization, opType [%s], format[%s], dtype[%s].",
              opType_.c_str(), format.c_str(), dtype.c_str());
    if (!FormatNormalize(index, mode, format)) {
        TE_ERRLOG("opType [%s], FormatNormalize failed.", opType_.c_str());
        return false;
    }
    if (!DtypeNormalize(index, mode, dtype)) {
        TE_ERRLOG("opType [%s], DtypeNormalize failed.", opType_.c_str());
        return false;
    }
    TE_DBGLOG("After normalization, opType [%s], format[%s], dtype[%s].",
              opType_.c_str(), format.c_str(), dtype.c_str());
    return true;
}

bool GenerateSimpleKey::MatchInOutPutCnt(const std::vector<TbeOpParam> &inOutPuts,
                                         const std::string &putsType) const
{
    DtypeFormatMode mode;
    if (putsType == INPUTS) {
        if (!binaryInfo_->GetInputMode(opType_, mode)) {
            TE_ERRLOG("opType [%s] is not in input mode map.", opType_.c_str());
            return false;
        }
        if (inOutPuts.size() != mode.count) {
            TE_ERRLOG("opType[%s] input currSize[%d] can't match mode count[%d].",
                      opType_.c_str(), inOutPuts.size(), mode.count);
            return false;
        }
    } else if (putsType == OUTPUTS) {
        if (!binaryInfo_->GetOutputMode(opType_, mode)) {
            TE_ERRLOG("opType [%s] is not in the output mode map.", opType_.c_str());
            return false;
        }
        if (inOutPuts.size() != mode.count) {
            TE_ERRLOG("opType[%s] output currSize[%d] can't match mode count[%d].",
                      opType_.c_str(), inOutPuts.size(), mode.count);
            return false;
        }
    }
    return true;
}

bool GenerateSimpleKey::GenerateInOutPut(const std::vector<TbeOpParam> &inOutPuts, const std::string &putsType,
                                         std::string &inOutPutStr) const
{
    if (!MatchInOutPutCnt(inOutPuts, putsType)) {
        TE_ERRLOG("opType[%s] [%s] param count check failed.", opType_.c_str(), putsType.c_str());
        return false;
    }
    for (unsigned int index = 0; index < inOutPuts.size(); index++) {
        TensorType tensorType;
        inOutPuts[index].GetType(tensorType);
        const std::vector<TbeOpTensor> &tensors = inOutPuts[index].GetTensors();
        if (tensors.size() == 0) {
            TE_INFOLOG("There's no tensors.putsType[%s], index = [%u].", putsType.c_str(), index);
            // if optional input is not filled. tensors.size() == 0
            if (simpleKeyMode_ == SimpleKeyModeType::COMPATIBLE_MODE) {
                inOutPutStr.append("/,");
            }
            continue;
        }
        std::string curFormat;
        std::string curDType;
        TbeOpTensor tensor = tensors[0];
        tensor.GetFormat(curFormat);
        tensor.GetType(curDType);

        int optionalInputMode = 1;  // default no_placeholder: 1
        if (!binaryInfo_->GetOptionalInputMode(opType_, optionalInputMode)) {
            TE_ERRLOG("opType[%s] GetOptionalInputMode failed.", opType_.c_str());
            return false;
        }

        if (tensorType == TT_OPT) {
            bool optionalAddToSimpleKey = (simpleKeyMode_ == SimpleKeyModeType::SIMPLE_MODE) ||
                                    (simpleKeyMode_ == SimpleKeyModeType::COMPATIBLE_MODE && optionalInputMode == 1);
            // when simplekeyMode == 1 && optinalInputMode == no_placeholder. optional and to simpleKey
            if (optionalAddToSimpleKey) {
                continue;
            }
        }

        if (!Normalization(index, putsType, curFormat, curDType)) {
            TE_ERRLOG("opType[%s], curFormat = [%s], or curDType = [%s] is empty.",
                      opType_.c_str(), curFormat.c_str(), curDType.c_str());
            return false;
        }

        TE_DBGLOG("curFormat = [%s], curDType = [%s].", curFormat.c_str(), curDType.c_str());
        std::string dType = DTypeTransToSimple(curDType);
        std::string format = FormatTransToSimple(curFormat);
        if (dType.empty() || format.empty()) {
            TE_ERRLOG("curFormat = [%s], or curDType = [%s] is empty.", curFormat.c_str(), curDType.c_str());
            return false;
        }
        inOutPutStr.append("/");
        inOutPutStr.append(dType);
        inOutPutStr.append(",");
        inOutPutStr.append(format);
        auto dynamicParamMode = DynamicParamModeEnum::UNDEFINED;
        if (!binaryInfo_->GetDynParamModeByOpType(opType_, dynamicParamMode)) {
            TE_ERRLOG("opType[%s] GetDynParamModeByOpType failed.", opType_.c_str());
            return false;
        }
        // dynamic and keyMode=1, dtype,format,count
        if (tensorType == TT_DYN && simpleKeyMode_ == SimpleKeyModeType::COMPATIBLE_MODE &&
            dynamicParamMode != DynamicParamModeEnum::FOLDED_WITH_DESC) {
            std::string count = std::to_string(tensors.size());
            inOutPutStr.append(",");
            inOutPutStr.append(count);
        }
        TE_DBGLOG("inOutPutStr = [%s].", inOutPutStr.c_str());
    }
    return true;
}

std::string GenerateSimpleKey::GenerateDetermin(const std::string deterministic) const
{
    std::string determinStr;
    if (deterministic == "true") {
        determinStr.append("d=1");
    } else {
        determinStr.append("d=0");
    }
    TE_DBGLOG("determinStr = [%s].", determinStr.c_str());
    return determinStr;
}

std::string GenerateSimpleKey::GenerateAttrs(const std::vector<TbeAttrValue> &attrValues) const
{
    std::string attrsStr;
    for (auto &attr : attrValues) {
        ATTR_DTYPE attrType = attr.GetType();
        if (attrType == ATTR_STR) {
            std::string value;
            attr.GetValue(value);
            TE_DBGLOG("value = [%s].", value.c_str());
            attrsStr.append(value);
        } else if (attrType == ATTR_BOOL) {
            bool value = false;
            attr.GetValue(value);
            if (value) {
                attrsStr.append(std::to_string(1));
            } else {
                attrsStr.append(std::to_string(0));
            }
        }
        attrsStr.append(",");
    }
    attrsStr = "/" + attrsStr;
    attrsStr.pop_back();
    TE_DBGLOG("attrsStr = [%s].", attrsStr.c_str());
    return attrsStr;
}

bool GenerateSimpleKey::CheckParamSetDefaultVal()
{
    if (opType_.empty()) {
        TE_WARNLOG("opType is empty.");
        return false;
    }

    if (simpleKeyMode_ != SimpleKeyModeType::SIMPLE_MODE
        && simpleKeyMode_ != SimpleKeyModeType::COMPATIBLE_MODE
        && simpleKeyMode_ != SimpleKeyModeType::CUSTOM_MODE) {
        TE_ERRLOG("opType [%s] simpleKeyMode [%d] is not supported.", opType_.c_str(), simpleKeyMode_);
        return false;
    }
    if (implMode_.empty()) {
        implMode_ = "default";
    }
    auto iter = IMPL_MODE_STR_TO_INT.find(implMode_);
    if (iter == IMPL_MODE_STR_TO_INT.end()) {
        TE_WARNLOG("opType [%s] implMode [%s] is not supported.", opType_.c_str(), implMode_.c_str());
        return false;
    }
    if (deterministic_.empty()) {
        deterministic_ = "false";
    }
    if (deterministic_ != "true" && deterministic_ != "false" && deterministic_ != "ignore") {
        TE_WARNLOG("opType [%s] deterministic [%s] is not supported.", opType_.c_str(), deterministic_.c_str());
        return false;
    }
    return true;
}

bool GenerateSimpleKey::GenerateCustomizeSimplifiedKey(std::string &simpleKeystr) const
{
    if (opInfo_ == nullptr) {
        TE_WARNLOG("Failed to get op_info for opType = [%s] when generating simplified key in custom mode.", opType_.c_str());
        return false;
    }
    ge::ConstNodePtr nodePtr = opInfo_->GetNode();
    if (nodePtr == nullptr) {
        TE_WARNLOG("opType = [%s] cannot get node.", opType_.c_str());
        return false;
    }
    int64_t binSrcEnum = binaryInfo_->GetOppBinFlag() ? OPP_LATEST_ENUM : 0;
    auto registry = gert::DefaultOpImplSpaceRegistryV2::GetInstance().GetSpaceRegistry(
        static_cast<gert::OppImplVersionTag>(static_cast<ge::OppImplVersion>(binSrcEnum)));
    if (registry == nullptr) {
        TE_WARNLOG("Cannot find implfuncs registry by parameter %ld, operation type is %s.", binSrcEnum, opType_.c_str());
        return false;
    }
    auto funcs = registry->GetOpImpl(opType_.c_str());
    if (funcs == nullptr || funcs->gen_simplifiedkey == nullptr) {
        TE_WARNLOG("No implfuncs or gen_simplifiedkey_func from registry, op type is %s.", opType_.c_str());
        return false;
    }
    simpleKeystr.append("/");
    char simplifiedKeyRes[MAX_CUSTOMIZED_SIMPLIFIED_KEY_LEN] = {0};
    if (strcpy_s(simplifiedKeyRes, strlen(simpleKeystr.c_str()) + 1, simpleKeystr.c_str()) != 0) {
        TE_WARNLOG("Nullptr or bad copy is unexpected.");
        return false;
    }
    auto kernelContextHolder = gert::KernelRunContextBuilder().Build(nodePtr->GetOpDesc());
    auto tilingContext = reinterpret_cast<gert::TilingContext *>(kernelContextHolder.context_);
    if (funcs->gen_simplifiedkey(tilingContext, simplifiedKeyRes) != SUCCESS) {
        TE_WARNLOG("Cannot genSimplifiedKey by registered func, op type is %s.", opType_.c_str());
        return false;
    }
    simpleKeystr = simplifiedKeyRes;
    TE_INFOLOG("GenSimplifiedKey by registered func, res is: %s, optype: %s.", simpleKeystr.c_str(), opType_.c_str());
    return true;
}

bool GenerateSimpleKey::GenerateSimpleKeyStr(std::string &simpleKeystr)
{
    if (!CheckParamSetDefaultVal()) {
        return false;
    }
    std::string determiStr = GenerateDetermin(deterministic_);
    std::string implModeStr = GenerateImplMode(implMode_);

    simpleKeystr.append(opType_);
    simpleKeystr.append("/");
    simpleKeystr.append(determiStr);
    simpleKeystr.append(",");
    simpleKeystr.append(implModeStr);
    if (simpleKeyMode_ == SimpleKeyModeType::CUSTOM_MODE) {
        if (!GenerateCustomizeSimplifiedKey(simpleKeystr)) {
            return false;
        }
        TE_DBGLOG("GenerateSimpleKey by op registered func, simpleKeystr = [%s].", simpleKeystr.c_str());
        return true;
    }

    std::string inputStr;
    if (!GenerateInOutPut(inputs_, INPUTS, inputStr)) {
        TE_WARNLOG("Generate inputStr failed.");
        return false;
    }
    std::string outputStr;
    if (!GenerateInOutPut(outputs_, OUTPUTS, outputStr)) {
        TE_WARNLOG("Generate outputStr failed.");
        return false;
    }
    if (!inputStr.empty()) {
        simpleKeystr.append(inputStr);
    }
    if (!outputStr.empty()) {
        simpleKeystr.append(outputStr);
    }

    if (simpleKeyMode_ == SimpleKeyModeType::COMPATIBLE_MODE) {
        std::string attrStr = GenerateAttrs(attrs_);
        if (!attrStr.empty()) {
            simpleKeystr.append(attrStr);
        }
    }
    TE_DBGLOG("simpleKeystr = [%s].", simpleKeystr.c_str());
    return true;
}


} // namespace fusion
} // namespace te
