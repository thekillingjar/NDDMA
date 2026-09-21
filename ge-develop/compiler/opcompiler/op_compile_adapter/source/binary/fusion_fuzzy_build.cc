/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary/binary_manager.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <fcntl.h>
#include "graph/anchor.h"
#include "graph/utils/type_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "python_adapter/python_api_call.h"
#include "inc/te_fusion_check.h"
#include "common/tbe_op_info_cache.h"
#include "common/fusion_common.h"
#include "common/common_utils.h"
#include "common/te_context_utils.h"
#include "assemble_json/te_json_utils.h"
#include "cache/te_cache_manager.h"

namespace te {
namespace fusion {
using namespace ge;
using namespace nlohmann;
namespace {
const std::string RANGE_LIMITED = "limited";
}

void BinaryManager::ShapeLimitedGeneralize(int64_t &shapeValue, const int64_t &valueMax,
                                             const std::vector<std::pair<int64_t, int64_t>> &rangeSection,
                                             std::vector<std::pair<int64_t, int64_t>> &newShapeRange)
{
    if (shapeValue == 0 || shapeValue == 1) {
        newShapeRange.emplace_back(std::pair<int64_t, int64_t>(shapeValue, shapeValue));
        return;
    }

    if (rangeSection.empty()) {
        TE_WARNLOGF("rangeSection is null.");
        return;
    }

    if (shapeValue >= valueMax) {
        newShapeRange.emplace_back(rangeSection.back());
    } else {
        auto item = std::find_if(rangeSection.begin(), rangeSection.end(),
                                 [&](std::pair<int64_t, int64_t> range) -> bool {
                                     return (shapeValue >= range.first && shapeValue <= range.second);
                                });
        if (item != rangeSection.end()) {
            newShapeRange.emplace_back(*item);
        } else {
            TE_WARNLOGF("Not find range for %ld in rangeSection.", shapeValue);
            return;
        }
    }
    shapeValue = -1;
}

void BinaryManager::GeneralizeWitLimitedRule(std::vector<int64_t> &shape, std::string &format,
                                               std::vector<std::pair<int64_t, int64_t>> &newShapeRange)
{
    TE_DBGLOGF("shape is %s, range is %s, format is %s",
               ShapeToString(shape).c_str(), RangeToString(newShapeRange).c_str(), format.c_str());
    if (format == FORMAT_ND) {
        for (size_t idx = 0; idx < shape.size(); ++idx) {
            ShapeLimitedGeneralize(shape[idx], DIM_DHW_MAX, DIM_DHW, newShapeRange);
        }
        TE_DBGLOGF("after generalize shape is %s, range is %s.",
                   ShapeToString(shape).c_str(), RangeToString(newShapeRange).c_str());
    } else {
        if (shape.size() != format.size()) {
            TE_WARNLOGF("shape.size(%zu) not equal to format.size(%zu)", shape.size(), format.size());
            for (size_t idx = 0; idx < shape.size(); ++idx) {
                ShapeLimitedGeneralize(shape[idx], DIM_DHW_MAX, DIM_DHW, newShapeRange);
            }
            TE_DBGLOGF("After generalize shape is %s, range is %s",
                       ShapeToString(shape).c_str(), RangeToString(newShapeRange).c_str());
            return;
        }
        for (size_t idx = 0; idx < shape.size(); ++idx) {
            if (format[idx] == 'C') {
                newShapeRange.emplace_back(std::make_pair(shape[idx], shape[idx]));
                continue;
            } else if (format[idx] == 'N') {
                ShapeLimitedGeneralize(shape[idx], DIM_N_MAX, DIM_N, newShapeRange);
            } else {
                ShapeLimitedGeneralize(shape[idx], DIM_DHW_MAX, DIM_DHW, newShapeRange);
            }
        }
        TE_DBGLOGF("after generalize shape is %s, range is %s",
                   ShapeToString(shape).c_str(), RangeToString(newShapeRange).c_str());
    }
}

void BinaryManager::ShapeGeneralization(std::vector<int64_t> &shape, std::string &format, const bool &isLimited,
    bool toDynamicRank, std::vector<std::pair<int64_t, int64_t>> &newShapeRange)
{
    std::vector<int64_t> scalarShape = {1};
    if (shape == scalarShape) {
        TE_INFOLOGF("shape[1] do not need generalize");
        return;
    }

    if (shape.empty()) {
        TE_INFOLOGF("shape[] do not need generalize");
        return;
    }

    if (shape == unknownRankShape) {
        TE_INFOLOGF("shape[-2] do not need generalize.");
        return;
    }

    if (toDynamicRank) {
        if (std::find_if(shape.begin(), shape.end(), [] (int64_t i) { return i == 0 || i == 1; }) == shape.end()) {
            shape.clear();
            shape.emplace_back(DYNAMIC_SHAPE_DIM);
            newShapeRange.clear();
            return;
        }
    }

    if (isLimited) {
        GeneralizeWitLimitedRule(shape, format, newShapeRange);
        return;
    }

    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == 0 || shape[i] == 1) {
            newShapeRange.push_back(std::pair<int64_t, int64_t>(shape[i], shape[i]));
            continue;
        }
        shape[i] = -1;
        newShapeRange.push_back(std::pair<int64_t, int64_t>(0, -1));
    }
}

void BinaryManager::ShapeGeneralizeToDynamicRank(std::vector<int64_t> &shape,
                                                   std::vector<std::pair<int64_t, int64_t>> &newShapeRange)
{
    std::vector<int64_t> scalarShape = {1};
    if (shape == scalarShape) {
        TE_INFOLOGF("shape[1] do not need generalize");
        return;
    }

    if (shape.empty()) {
        TE_INFOLOGF("shape[] do not need generalize.");
        return;
    }

    if (std::find_if(shape.begin(), shape.end(), [] (int64_t i) { return i == 0 || i == 1; }) == shape.end()) {
        shape.clear();
        shape.emplace_back(DYNAMIC_SHAPE_DIM);
        newShapeRange.clear();
    }
}

void BinaryManager::GetGeneralizedResultFromReturnTensor(nlohmann::json &tensorJson, nlohmann::json &generalizedInfo)
{
    json tensorResJson;
    if (tensorJson.is_null()) {
        return;
    }

    if (tensorJson.find("shape") != tensorJson.end()) {
        tensorResJson["shape"] = tensorJson["shape"];
    }
    if (tensorJson.find("ori_shape") != tensorJson.end()) {
        tensorResJson["ori_shape"] = tensorJson["ori_shape"];
    }
    if (tensorJson.find("range") != tensorJson.end()) {
        tensorResJson["range"] = tensorJson["range"];
    }
    if (tensorJson.find("ori_range") != tensorJson.end()) {
        tensorResJson["ori_range"] = tensorJson["ori_range"];
    }
    if (tensorJson.find("const_value") != tensorJson.end()) {
        tensorResJson["const_value"] = tensorJson["const_value"];
    }
    if (tensorJson.find("const_value_range") != tensorJson.end()) {
        tensorResJson["const_value_range"] = tensorJson["const_value_range"];
    }
    if (!tensorResJson.is_null()) {
        generalizedInfo.push_back(tensorResJson);
    } else {
        TE_WARNLOGF("There is no shape or ori_shape or range or ori_range in tensorJson.");
    }
}

bool BinaryManager::ParseGeneralizedResultFromTbe(nlohmann::json &generalizedRes, const std::string &opName,
                                                    nlohmann::json &generalizedInfo)
{
    if (!generalizedRes.is_array()) {
        REPORT_TE_INNER_WARN("Node %s returned non-array result from tbe register function.", opName.c_str());
        return false;
    }
    for (auto singleInfo = generalizedRes.begin(); singleInfo != generalizedRes.end(); ++singleInfo) {
        json singleInfoJson = *singleInfo;
        if (singleInfoJson.find("result") != singleInfoJson.end()) {
            REPORT_TE_INNER_WARN("Node %s current shape or range not supported by tbe.", opName.c_str());
            return false;
        }

        if (!singleInfoJson.is_array()) {
            REPORT_TE_INNER_WARN("Node %s generalized result from TBE is not an array.", opName.c_str());
            return false;
        }

        for (auto inputOrOutput = singleInfoJson.begin(); inputOrOutput != singleInfoJson.end(); ++inputOrOutput) {
            json inputOrOutputJson = *inputOrOutput;
            if (inputOrOutputJson.is_array()) {
                for (auto tensor = inputOrOutputJson.begin(); tensor != inputOrOutputJson.end(); ++tensor) {
                    json tensorJson = *tensor;
                    GetGeneralizedResultFromReturnTensor(tensorJson, generalizedInfo);
                }
            } else {
                GetGeneralizedResultFromReturnTensor(inputOrOutputJson, generalizedInfo);
            }
        }
    }
    return true;
}

void BinaryManager::SetGeneralizeInfoToJson(const std::vector<TbeOpTensor> &tensors, OP_VALUE_DEPEND &valueDepend,
                                              const bool &isLimited, const bool &toDynamicRank,
                                              nlohmann::json &generalizedValidInfo)
{
    std::string oriFormat;
    std::vector<int64_t> oriShape = {};
    std::vector<std::pair<int64_t, int64_t>> oriShapeRange;
    bool isConst = false;
    std::vector<int64_t> scalarShape = {1};

    for (size_t i = 0; i < tensors.size(); i++) {
        json singleGeneralizeInfo;

        (void)tensors[i].GetOriginShape(oriShape);
        (void)tensors[i].GetOriginShapeRange(oriShapeRange);
        (void)tensors[i].GetOriginFormat(oriFormat);
        (void)tensors[i].GetConstFlag(isConst);
        json temp;
        temp["ori_shape"] = oriShape;
        temp["ori_shape_range"] = oriShapeRange;
        temp["ori_format"] = oriFormat;
        TE_DBGLOGF("before generalize is %s", temp.dump().c_str());

        if (valueDepend != VALUE_DEPEND_IGNORE) {
            TE_DBGLOGF("input is value depend %d", valueDepend);
            singleGeneralizeInfo["ori_shape"] = oriShape;
            singleGeneralizeInfo["ori_range"] = oriShapeRange;
            if (valueDepend == VALUE_DEPEND_OPTIONAL && isConst) {
                json jsonNull;
                singleGeneralizeInfo["const_value"] = jsonNull;
            }
            generalizedValidInfo.push_back(singleGeneralizeInfo);
            continue;
        }

        std::vector<std::pair<int64_t, int64_t>> newshapeRange;
        ShapeGeneralization(oriShape, oriFormat, isLimited, toDynamicRank, newshapeRange);
        singleGeneralizeInfo["ori_range"] = newshapeRange;
        singleGeneralizeInfo["ori_shape"] = oriShape;

        if (isConst) {
            json jsonNull;
            singleGeneralizeInfo["const_value"] = jsonNull;
        }
        TE_DBGLOGF("After generalize is %s.", singleGeneralizeInfo.dump().c_str());
        generalizedValidInfo.push_back(singleGeneralizeInfo);
    }
}

void BinaryManager::GetHighPerformanceResFromJson(const nlohmann::json &jsonValue, bool &isHighPerformaceRes)
{
    if (jsonValue.find("implMode") == jsonValue.end()) {
        isHighPerformaceRes = true;
    } else {
        std::string implMode = jsonValue.at("implMode").get<string>();
        if (implMode.empty() || implMode != "high_performance") {
            isHighPerformaceRes = false;
        }
    }
}

void BinaryManager::GetHighPerformanceResFromJsonList(const nlohmann::json &kernelList, bool &isHighPerformaceRes)
{
    for (auto it = kernelList.begin(); it != kernelList.end(); ++it) {
        json singleKernel = *it;
        GetHighPerformanceResFromJson(singleKernel, isHighPerformaceRes);
        if (!isHighPerformaceRes) {
            break;
        }
    }
}

void BinaryManager::SetOpResAccordingCompileInfo(const OpBuildTaskPtr &opTask,
                                                   const nlohmann::json &kernelsCompileInfo)
{
    OpBuildTaskResultPtr opResPtr = opTask->opRes;
    if (opResPtr == nullptr) {
        opResPtr = std::make_shared<OpBuildTaskResult>();
        TE_FUSION_CHECK((opResPtr == nullptr), {
            TE_WARNLOG("opResPtr is null, taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
            return ;
        })
    }

    std::string strKernelCompileInfo;
    if (kernelsCompileInfo.is_string()) {
        strKernelCompileInfo = kernelsCompileInfo.get<std::string>();
        TE_DBGLOG("Got string str_kernel_compile_info: %s.", strKernelCompileInfo.c_str());
    } else {
        strKernelCompileInfo = kernelsCompileInfo.dump();
        TE_DBGLOG("Dump json str_kernel_compile_info: %s", strKernelCompileInfo.c_str());
    }

    opResPtr->compile_info_str = strKernelCompileInfo;
    std::string compileInfoKey;
    bool bres = PythonApiCall::Instance().GenerateStrSha256HashValue(opResPtr->compile_info_str, compileInfoKey);
    TE_FUSION_CHECK(!bres, {
        TE_WARNLOG("Failed to generate the compile_info_key.");
    });
    opResPtr->compile_info_key = compileInfoKey;
    opTask->opRes = opResPtr;
}

void BinaryManager::SetMaxKernelIdAndIsHighPerformanceRes(const nlohmann::json &jsonValue,
                                                            const OpBuildTaskPtr &opTask)
{
    if (jsonValue.find("maxKernelId") != jsonValue.end()) {
        TE_DBGLOG("start to set maxKernelId");
        uint64_t maxKernelId = jsonValue.at("maxKernelId").get<uint64_t>();
        opTask->maxKernelId = maxKernelId;
    } else {
        TE_DBGLOG("maxKernelId does not exist or is empty");
    }
    bool high_performance_res = true;
    if (jsonValue.find("kernelList") != jsonValue.end()) {
        json kernelList = jsonValue.at("kernelList");
        GetHighPerformanceResFromJsonList(kernelList, high_performance_res);
    }
    opTask->isHighPerformaceRes = high_performance_res;
}

void BinaryManager::SetShapeAndValueToTensor(nlohmann::json &singleGeneralizeRes,
                                               ge::GeTensorDescPtr &tenosrDescPtr)
{
    try {
        if (singleGeneralizeRes.find("ori_shape") != singleGeneralizeRes.end()) {
            std::vector<int64_t> oriShape = singleGeneralizeRes.at("ori_shape").get<std::vector<int64_t>>();
            tenosrDescPtr->SetOriginShape(ge::GeShape(oriShape));
        }

        if (singleGeneralizeRes.find("ori_range") != singleGeneralizeRes.end()) {
            std::vector<std::pair<int64_t, int64_t>> oriRange =
                                    singleGeneralizeRes.at("ori_range").get<std::vector<std::pair<int64_t, int64_t>>>();
            tenosrDescPtr->SetOriginShapeRange(oriRange);
        }

        if (singleGeneralizeRes.find("const_value_range") != singleGeneralizeRes.end()) {
            std::vector<std::pair<int64_t, int64_t>> valueRange =
                            singleGeneralizeRes.at("const_value_range").get<std::vector<std::pair<int64_t, int64_t>>>();
            ChangeShapeRangeExpressFromUnlimitedToMinMAx(valueRange);
            tenosrDescPtr->SetValueRange(valueRange);
        }

        if (singleGeneralizeRes.find("const_value") != singleGeneralizeRes.end()) {
            if (singleGeneralizeRes["const_value"].is_null() &&
                ge::AttrUtils::HasAttr(tenosrDescPtr, ge::ATTR_NAME_VALUE)) {
                TE_DBGLOGF("SetShapeAndValueToTensor delete ge::ATTR_NAME_VALUE.");
                tenosrDescPtr->DelAttr(ge::ATTR_NAME_VALUE);
            }
        }
    } catch (std::exception &e) {
        TE_WARNLOG("SetShapeAndValueToTensor failed, json string is: %s, reason is: %s",
                   singleGeneralizeRes.dump().c_str(), e.what());
    }
}

void BinaryManager::SetGeneralizedInfoToNode(nlohmann::json &generalizedValidInfo, const ge::NodePtr &nodePtr)
{
    uint32_t index = 0;
    json singleGeneralizeRes;
    auto opDescPtr = nodePtr->GetOpDesc();
    std::string opName = opDescPtr->GetName();
    for (auto &tenosrDescPtr : opDescPtr->GetAllInputsDescPtr()) {
        if (tenosrDescPtr == nullptr) {
            continue;
        }
        singleGeneralizeRes = generalizedValidInfo[index];
        SetShapeAndValueToTensor(singleGeneralizeRes, tenosrDescPtr);
        index++;
    }

    for (auto &tenosrDescPtr : opDescPtr->GetAllOutputsDescPtr()) {
        if (tenosrDescPtr == nullptr) {
            continue;
        }
        singleGeneralizeRes = generalizedValidInfo[index];
        SetShapeAndValueToTensor(singleGeneralizeRes, tenosrDescPtr);
        index++;
    }
}

bool BinaryManager::TeGeneralizeWithRegisterFunc(const TbeOpInfo &tbeOpInfo, const ge::NodePtr &nodePtr)
{
    nlohmann::json generalizedRes;
    const std::string &opName = tbeOpInfo.GetName();
    if (!PythonApiCall::Instance().GetCheckAndGeneralizedResFromTbe(tbeOpInfo, FUZZILY_BUILD_TYPE, generalizedRes)) {
        TE_WARNLOG("GetCheckAndGeneralizedResFromTbe failed for node %s.", opName.c_str());
        return false;
    }

    if (generalizedRes.is_null()) {
        bool isLimitedGraph = false;
        (void)ge::AttrUtils::GetBool(nodePtr->GetOpDesc(), ATTR_NAME_IS_LIMITED_GRAPH, isLimitedGraph);
        TE_INFOLOGF("Node %s func generalize res is null, use default %s generalize rule",
                    opName.c_str(), isLimitedGraph ? "limited" : "unlimited");
        if (isLimitedGraph) {
            TeGeneralizeWithDefaultRuleByTbeOpInfo(tbeOpInfo, true, nodePtr);
        } else {
            TeGeneralizeWithDefaultRuleByTbeOpInfo(tbeOpInfo, false, nodePtr);
        }
        return true;
    }
    TE_DBGLOGF("Node %s generalizedRes is: %s", opName.c_str(), generalizedRes.dump().c_str());

    json generalizedValidInfo;
    if (!ParseGeneralizedResultFromTbe(generalizedRes, opName, generalizedValidInfo)) {
        TE_WARNLOG("ParseGeneralizedResultFromTbe for node %s failed", opName.c_str());
        return false;
    }

    TE_INFOLOGF("Node %s generalizedValidInfo is: %s", opName.c_str(), generalizedValidInfo.dump().c_str());
    SetGeneralizedInfoToNode(generalizedValidInfo, nodePtr);

    return true;
}

void BinaryManager::TeGeneralizeWithDefaultRuleByTbeOpInfo(const TbeOpInfo &tbeOpInfo, const bool &isLimited,
                                                             const ge::NodePtr &nodePtr)
{
    const std::vector<TbeOpParam> &inputs = tbeOpInfo.GetInputs();
    const std::vector<TbeOpParam> &outputs = tbeOpInfo.GetOutputs();

    json generalizedValidInfo;
    OP_VALUE_DEPEND valueDepend = VALUE_DEPEND_IGNORE;
    bool toDynamicRank = (tbeOpInfo.IsSingleOpScene() &&
        tbeOpInfo.GetDynamicRankType() == DynamicRankType::UPGRADE_TO_SUPPORT);
    for (size_t i = 0; i < inputs.size(); i++) {
        const std::vector<TbeOpTensor> &tensors = inputs.at(i).GetTensors();
        if (tensors.size() == 0) {
            continue;
        }
        valueDepend = inputs[i].GetValueDepend();
        SetGeneralizeInfoToJson(tensors, valueDepend, isLimited, toDynamicRank, generalizedValidInfo);
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        const std::vector<TbeOpTensor> &tensors = outputs.at(i).GetTensors();
        if (tensors.size() == 0) {
            continue;
        }
        SetGeneralizeInfoToJson(tensors, valueDepend, isLimited, toDynamicRank, generalizedValidInfo);
    }

    TE_INFOLOGF("Node %s generalizedValidInfo is: %s", tbeOpInfo.GetName().c_str(), generalizedValidInfo.dump().c_str());
    SetGeneralizedInfoToNode(generalizedValidInfo, nodePtr);
}

void BinaryManager::GeneralizeShapeAndValueByTensor(ge::GeTensorDescPtr &tenosrDescPtr)
{
    std::vector<std::pair<int64_t, int64_t>> newOriShapeRange;
    std::vector<std::pair<int64_t, int64_t>> oriShapeRange;

    std::vector<int64_t> oriShape = tenosrDescPtr->GetOriginShape().GetDims();
    (void)tenosrDescPtr->GetOriginShapeRange(oriShapeRange);
    std::string oriFormat = ge::TypeUtils::FormatToSerialString(tenosrDescPtr->GetOriginFormat());

    json temp;
    temp["ori_shape"] = oriShape;
    temp["ori_shape_range"] = oriShapeRange;
    temp["ori_format"] = oriFormat;
    TE_DBGLOGF("before generalize is %s", temp.dump().c_str());

    const bool isLimited = false;
    ShapeGeneralization(oriShape, oriFormat, isLimited, false, newOriShapeRange);

    tenosrDescPtr->SetOriginShape(ge::GeShape(oriShape));
    if (oriShapeRange.size() == 0) {
        tenosrDescPtr->SetOriginShapeRange(newOriShapeRange);
    }

    temp["ori_shape"] = oriShape;
    temp["new_ori_shape_range"] = newOriShapeRange;
    TE_DBGLOGF("after generalize is %s", temp.dump().c_str());
}

void BinaryManager::TeGeneralizeWithDefaultRuleByNode(const ge::NodePtr &nodePtr)
{
    auto opDescPtr = nodePtr->GetOpDesc();
    for (auto &tenosrDescPtr : opDescPtr->GetAllInputsDescPtr()) {
        if (tenosrDescPtr == nullptr) {
            continue;
        }
        GeneralizeShapeAndValueByTensor(tenosrDescPtr);
    }

    for (auto &tenosrDescPtr : opDescPtr->GetAllOutputsDescPtr()) {
        if (tenosrDescPtr == nullptr) {
            continue;
        }
        GeneralizeShapeAndValueByTensor(tenosrDescPtr);
    }
}

bool BinaryManager::TeGeneralize(const TbeOpInfo &tbeOpInfo, const TE_GENERALIZE_TYPE &generalizeType,
                                   const ge::NodePtr &nodePtr)
{
    std::string opName = nodePtr->GetOpDesc()->GetName();
    TE_DBGLOGF("Node[%s] begin TeGeneralize, generalizeType is %d", opName.c_str(), generalizeType);
    if (generalizeType == REGISTER_FUNC) {
        TE_FUSION_CHECK(!TeGeneralizeWithRegisterFunc(tbeOpInfo, nodePtr), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "TeGeneralizeWithRegisterFunc failed.");
            return false;
        });
    } else if (generalizeType == DEFAULT_TBE_OP_INFO) {
        TeGeneralizeWithDefaultRuleByTbeOpInfo(tbeOpInfo, false, nodePtr);
    } else if (generalizeType == DEFAULT_LIMITED_TBE_OP_INFO) {
        TeGeneralizeWithDefaultRuleByTbeOpInfo(tbeOpInfo, true, nodePtr);
    } else if (generalizeType == DEFAULT_NODE) {
        TeGeneralizeWithDefaultRuleByNode(nodePtr);
    } else {
        REPORT_TE_INNER_WARN("Node [%s] has an invalid generalizeType [%d].", opName.c_str(), generalizeType);
        return false;
    }

    return true;
}

void BinaryManager::ChangeShapeRangeExpressFromUnlimitedToMinMAx(
    std::vector<std::pair<int64_t, int64_t>> &shapeRange)
{
    for (size_t i = 0; i < shapeRange.size(); i++) {
        if ((shapeRange[i].first == RANGE_UNLIMITED_LOW_ONE || shapeRange[i].first == RANGE_UNLIMITED_LOW_ZERO) &&
            shapeRange[i].second == RANGE_UNLIMITED_UP) {
            shapeRange[i].first = INT64_MIN;
            shapeRange[i].second = INT64_MAX;
        }
    }
}

bool BinaryManager::GetRangeLimitedType(const TbeOpInfo &tbeOpInfo, bool &isRangeLimited)
{
    RangeLimitType rangeLimitType = tbeOpInfo.GetRangeLimitType();
    if (rangeLimitType == RangeLimitType::DYNAMIC) {
        std::string opSpecificInfo;
        bool res = PythonApiCall::Instance().GetOpSpecificInfo(tbeOpInfo, opSpecificInfo);
        TE_FUSION_CHECK(!res, {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to get op specific info.");
            return false;
        });

        try {
            nlohmann::json specificInfo = nlohmann::json::parse(opSpecificInfo);
            std::string rangeLimit = specificInfo.at("rangeLimit").get<std::string>();
            if (rangeLimit == RANGE_LIMITED) {
                rangeLimitType = RangeLimitType::LIMITED;
            }
        } catch (std::exception &e) {
            TE_WARNLOGF("Parse json_str failed, string is %s and the reason is %s.",
                        opSpecificInfo.c_str(), e.what());
            return false;
        }
    }

    isRangeLimited = rangeLimitType == RangeLimitType::LIMITED;
    return true;
}

bool BinaryManager::IsOpCanSecondaryGeneralization(const OpBuildTaskPtr &opTask, bool &isFormatAgnostic,
    bool &isRangeLimited, bool &toDynamicRank, bool &isRangeAgnostic)
{
    const TbeOpInfo &opInfo = *opTask->pPrebuildOp;
    std::string opStorePattern;
    opInfo.GetOpStorePattern(opStorePattern);
    isRangeAgnostic = (opStorePattern.find("rangeAgnostic") != std::string::npos);
    if ((opTask->buildType != FUZZILY_BUILD) && (!TeContextUtils::EnableShapeGeneralized())) {
        if (isRangeAgnostic) {
            TE_DBGLOG("Current task[%lu:%lu] is range agnostic.", opTask->graphId, opTask->taskId);
            return true;
        }
        TE_DBGLOG("Current task[%lu: %lu] is not fuzz build and shape_generalized flag is false.",
                  opTask->graphId, opTask->taskId);
        return false;
    }

    toDynamicRank = opInfo.GetDynamicRankType() == DynamicRankType::UPGRADE_TO_SUPPORT;
    isFormatAgnostic = (opStorePattern.find("formatAgnostic") != std::string::npos);

    if (!GetRangeLimitedType(opInfo, isRangeLimited)) {
        return false;
    }

    TE_DBGLOG("FormatAgnostic[%s], RangeUnlimited[%s], DynamicRank[%s], RangeAgnostic[%s].",
              (isFormatAgnostic) ? "true" : "false", (isRangeLimited) ? "true" : "false",
              (toDynamicRank) ? "true" : "false", (isRangeAgnostic) ? "true" : "false");

    if (!isFormatAgnostic && isRangeLimited && !toDynamicRank && !isRangeAgnostic) {
        TE_DBGLOG("No need to do secondary generalization.");
        return false;
    }

    return true;
}

void BinaryManager::GeneralizeShapeToUnlimited(const std::vector<int64_t> &shape,
                                                 std::vector<std::pair<int64_t, int64_t>> &shapeRange)
{
    if (shape == unknownRankShape) {
        TE_INFOLOGF("shape == unknownRankShape.");
        return;
    }

    if (shape.size() != shapeRange.size()) {
        TE_WARNLOG("ShapeSize[%zu] and shapeRangeSize[%zu] are not equal.", shape.size(), shapeRange.size());
        return;
    }

    for (size_t idx = 0; idx < shape.size(); ++idx) {
        if (shape[idx] == -1) {
            shapeRange[idx].first = RANGE_UNLIMITED_LOW_ZERO;
            shapeRange[idx].second = RANGE_UNLIMITED_UP;
        }
    }
}

void BinaryManager::SetGeneralizeInfoToTensors(std::vector<TbeOpTensor> &tensors, const bool &isFormatAgnostic,
                                                 const bool &isRangeLimited, const bool &toDynamicRank,
                                                 const bool &isRangeAgnostic)
{
    std::string format;
    std::string oriFormat;
    std::vector<int64_t> shape = {};
    std::vector<int64_t> oriShape = {};
    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    std::vector<std::pair<int64_t, int64_t>> oriShapeRange;

    for (size_t i = 0; i < tensors.size(); i++) {
        tensors[i].GetFormat(format);
        tensors[i].GetOriginFormat(oriFormat);
        tensors[i].GetShape(shape);
        tensors[i].GetOriginShape(oriShape);
        (void)tensors[i].GetShapeRange(shapeRange);
        (void)tensors[i].GetOriginShapeRange(oriShapeRange);

        TE_DBGLOGF("Before generalize:format=%s, oriFormat=%s, shape=%s, oriShape=%s, shapeRange=%s, oriShapeRange=%s.",
                   format.c_str(), oriFormat.c_str(), ShapeToString(shape).c_str(), ShapeToString(oriShape).c_str(),
                   RangeToString(shapeRange).c_str(), RangeToString(oriShapeRange).c_str());
        if (isFormatAgnostic) {
            format = "ND";
            oriFormat = "ND";
        }

        if (!isRangeLimited || isRangeAgnostic) {
            GeneralizeShapeToUnlimited(shape, shapeRange);
            GeneralizeShapeToUnlimited(oriShape, oriShapeRange);
        }

        if (toDynamicRank) {
            ShapeGeneralizeToDynamicRank(shape, shapeRange);
            ShapeGeneralizeToDynamicRank(oriShape, oriShapeRange);
        }

        tensors[i].SetFormat(format);
        tensors[i].SetOriginFormat(oriFormat);
        tensors[i].SetShape(shape);
        tensors[i].SetOriginShape(oriShape);
        tensors[i].SetShapeRange(shapeRange);
        tensors[i].SetOriginShapeRange(oriShapeRange);
        TE_DBGLOGF("After generalize:format=%s, oriFormat=%s, shape=%s, oriShape=%s, shapeRange=%s, oriShapeRange=%s",
                   format.c_str(), oriFormat.c_str(), ShapeToString(shape).c_str(), ShapeToString(oriShape).c_str(),
                   RangeToString(shapeRange).c_str(), RangeToString(oriShapeRange).c_str());
    }
}

bool BinaryManager::FuzzyBuildSecondaryGeneralization(const OpBuildTaskPtr &opTask, const std::string &opName,
                                                        TbeOpInfoPtr &selectedOpInfo)
{
    bool isFormatAgnostic = false;
    bool isRangeLimited = false;
    bool toDynamicRank = false;
    bool isRangeAgnostic = false;
    if (!IsOpCanSecondaryGeneralization(opTask, isFormatAgnostic, isRangeLimited, toDynamicRank, isRangeAgnostic)) {
        TE_DBGLOG("Node [%s] cannot perform secondary generalization", opName.c_str());
        return true;
    }

    TE_DBGLOG("Node[%s] FormatAgnostic[%s], RangeLimited[%s], DynamicRank[%s], RangeAgnostic[%s].",
              opName.c_str(), (isFormatAgnostic) ? "true" : "false", (isRangeLimited) ? "true" : "false",
              (toDynamicRank) ? "true" : "false", (isRangeAgnostic) ? "true" : "false");
    TbeOpInfo &opInfo = *opTask->pPrebuildOp;
    TbeOpInfoPtr opInfoPtr = opInfo.shared_from_this();
    TE_FUSION_CHECK((opInfoPtr == nullptr), {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Record secondary generalization info failed, it is NULL!");
        return true;
    });

    bool isUnknowShape = opInfoPtr->GetIsUnknownShape();
    if (!isUnknowShape && !isFormatAgnostic) {
        TE_DBGLOG("Node[%s], shape is static and format is not agnostic, no need to do secondary generalization",
                  opName.c_str());
        return true;
    }

    std::vector<TbeOpParam> &inputs = opInfoPtr->MutableInputs();
    for (size_t i = 0; i < inputs.size(); i++) {
        std::vector<TbeOpTensor> &tensors = inputs[i].MutableTensors();
        if (tensors.size() == 0) {
            continue;
        }
        bool tensorToDynamicRank = (toDynamicRank && inputs[i].GetValueDepend() == VALUE_DEPEND_IGNORE);
        SetGeneralizeInfoToTensors(tensors, isFormatAgnostic, isRangeLimited, tensorToDynamicRank, isRangeAgnostic);
    }

    std::vector<TbeOpParam> &outputs = opInfoPtr->MutableOutputs();
    for (size_t i = 0; i < outputs.size(); i++) {
        std::vector<TbeOpTensor> &tensors = outputs[i].MutableTensors();
        if (tensors.size() == 0) {
            continue;
        }
        SetGeneralizeInfoToTensors(tensors, isFormatAgnostic, isRangeLimited, toDynamicRank, isRangeAgnostic);
    }

    TbeOpInfoCache::Instance().UpdateSecondTbeOpInfo(opName, opInfoPtr);
    selectedOpInfo = opInfoPtr;
    return true;
}
} // namespace fusion
} // namespace te
