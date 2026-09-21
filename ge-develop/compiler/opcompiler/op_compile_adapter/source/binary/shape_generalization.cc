/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary/shape_generalization.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include "dfxinfo_manager/dfxinfo_manager.h"
#include "graph/anchor.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/operator.h"
#include "inc/te_fusion_check.h"
#include "python_adapter/python_api_call.h"
#include "common/common_utils.h"
#include "common/fusion_common.h"
#include "common/te_config_info.h"
#include "common/te_context_utils.h"
#include "common/tbe_op_info_cache.h"
#include "assemble_json/te_attr_utils.h"
#include "binary/generate_simple_key.h"
#include "binary/fusion_binary_info.h"
#include "assemble_json/te_attr_utils.h"
#include "assemble_json/te_json_utils.h"
#include "assemble_json/te_json_assemble.h"

namespace te {
namespace fusion {
using namespace ge;
using namespace nlohmann;
namespace {
// opc atts dtypes support these
std::map<ATTR_DTYPE, std::string> g_attrDtypeToString = {
    {ATTR_INT64,           "int"},
    {ATTR_FLOAT32,         "float"},
    {ATTR_BOOL,            "bool"},
    {ATTR_STR,             "string"},
    {ATTR_LIST_INT64,      "list_int"},
    {ATTR_LIST_FLOAT32,    "list_float"},
    {ATTR_LIST_BOOL,       "list_bool"},
    {ATTR_LIST_STR,        "list_string"},
    {ATTR_LIST_LIST_INT64, "list_list_int"},
    {ATTR_INT32,           "data_type"},
};

void JudgeJsonExistAndSet(json &inputDynamicJson, json &inputJsonTmp, const string &jsonName)
{
    if (inputJsonTmp.find(jsonName) == inputJsonTmp.end()) {
        return;
    }
    inputDynamicJson[jsonName] = inputJsonTmp[jsonName];
}

void JudgeJsonExistAndSetByDiffKey(json &inputJson, json &inputJsonTmp,
                                   const string &inputJsonKeyName,
                                   const string &inputJsonTmpKeyName)
{
    if (inputJsonTmp.find(inputJsonTmpKeyName) == inputJsonTmp.end()) {
        return;
    }
    inputJson[inputJsonKeyName] = inputJsonTmp[inputJsonTmpKeyName];
}
}

bool ShapeGeneralization::GeneralizeOps(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    bool res;
    if (opTask->opNodes.size() == 1) {
        res = GeneralizeSingleOp(opTask, generalizedResult);
    } else {
        res = GeneralizeFusionOps(opTask, generalizedResult);
    }

    if (!res) {
        TE_DBGLOG("Generalize ops(%s) not success.", GetTaskNodeName(opTask).c_str());
    } else {
        TE_DBGLOG("Generalize ops(%s) successfully.", GetTaskNodeName(opTask).c_str());
    }

    return res;
}

bool ShapeGeneralization::GeneralizeSingleOp(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    auto pNode = opTask->opNodes[0];
    if (pNode == nullptr) {
        TE_WARNLOGF("pNode is null.");
        return false;
    }
    ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(pNode);
    if (tbeOpInfo == nullptr) {
        TE_WARNLOGF("Node %s GetTbeOpInfoByNode failed.", pNode->GetName().c_str());
        return false;
    }

    bool hasRegisteredFunc = false;
    if (!PythonApiCall::Instance().CheckIsTbeGeneralizeFuncRegistered(*tbeOpInfo, hasRegisteredFunc)) {
        return false;
    }
    TE_DBGLOGF("Node[%s] hasRegisteredFunc = %u.", pNode->GetName().c_str(), hasRegisteredFunc);

    GetOptionalInputIdxByTbeOpInfo(tbeOpInfo, generalizedResult);
    if (!BinaryGeneralizeWithDefaultRule(tbeOpInfo, generalizedResult)) {
        TE_DBGLOGF("BinaryGeneralizeWithDefaultRule did not succeed.");
        return false;
    }

    if (hasRegisteredFunc) {
        bool ret = BinaryGeneralizeWithRegisterFunc(tbeOpInfo, generalizedResult);
        if (ret) {
            TE_DBGLOGF("Node[%s] BinaryGeneralizeWithRegisterFunc success.", pNode->GetName().c_str());
            return true;
        }
        TE_DBGLOGF("Node[%s] BinaryGeneralizeWithRegisterFunc did not succeed.", pNode->GetName().c_str());
        return false;
    }

    TE_DBGLOGF("The staticJson is: %s, and the dynamicJson is %s.",
               generalizedResult.staticJson.dump().c_str(), generalizedResult.staticJson.dump().c_str());

    return true;
}

bool ShapeGeneralization::GeneralizeFusionOps(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult)
{
    auto teGraphNode = opTask->opNodes;
    TE_FUSION_CHECK((teGraphNode.empty()), {
        REPORT_TE_INNER_ERROR("Task[%lu] nodes from fe is empty.", opTask->taskId);
        return false;
    });

    std::unordered_set<ge::Node *> allNodes;
    for (auto &ele : teGraphNode) {
        allNodes.emplace(ele);
    }

    for (auto &currentNode : teGraphNode) {
        std::string opName = currentNode->GetName();
        ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(currentNode);
        if (tbeOpInfo == nullptr) {
            return false;
        }

        nlohmann::json &generalizedRes = generalizedResult.generalizedAllJson;
        BinaryGeneralizeWithRegisterFuncGetGenerateRes(tbeOpInfo, generalizedRes);

        TE_FUSION_CHECK((!GenFusionInputTmpJson(teGraphNode, currentNode, allNodes, generalizedResult)), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "GenFusionInputTmpJson fail op name is [%s].",
                               opName.c_str());
            return false;
        });

        TE_FUSION_CHECK((!GenerateNormalizeFusionOutputTmpJson(currentNode, allNodes, generalizedResult)), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "GenerateNormalizeFusionOutputTmpJson fail op name is [%s].",
                               opName.c_str());
            return false;
        });
        TE_FUSION_CHECK((!GenerateNormalizeFusionAttrTmpJson(tbeOpInfo, generalizedResult, false, generalizedRes)), {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "GenerateNormalizeFusionAttrTmpJson fail op name is [%s].",
                               opName.c_str());
            return false;
        });
    }

    TE_DBGLOGF("Fusion op staticJson:%s, dynamicJson:%s",
               generalizedResult.staticJson.dump().c_str(),
               generalizedResult.dynamicJson.dump().c_str());
    return true;
}

void ShapeGeneralization::GetOptionalInputIdxByTbeOpInfo(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                     GeneralizedResult &generalizedResult)
{
    const std::vector<TbeOpParam> &inputs = tbeOpInfo->GetInputs();
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (inputs[i].GetType() == TT_OPT) {
            generalizedResult.optionalInputIdx.emplace_back(static_cast<uint32_t>(i));
        }
    }
}

bool ShapeGeneralization::BinaryGeneralizeWithDefaultRule(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                      GeneralizedResult &generalizedResult)
{
    bool dynamicRankSupport = tbeOpInfo->IsSupportDynamicRank();
    const std::vector<TbeOpParam> &inputs = tbeOpInfo->GetInputs();
    const std::vector<TbeOpParam> &outputs = tbeOpInfo->GetOutputs();
    const std::string &opPattern = tbeOpInfo->GetOpStorePattern();

    for (const TbeOpParam &input : inputs) {
        GeneralizeInOrOutput(generalizedResult.staticJson[INPUTS], generalizedResult.dynamicJson[INPUTS], input,
                             dynamicRankSupport, opPattern);
    }

    for (const TbeOpParam &output : outputs) {
        GeneralizeInOrOutput(generalizedResult.staticJson[OUTPUTS], generalizedResult.dynamicJson[OUTPUTS], output,
                             dynamicRankSupport, opPattern);
    }

    nlohmann::json generalizedRes;
    return GenerateNormalizeFusionAttrTmpJson(tbeOpInfo, generalizedResult, true, generalizedRes);
}

void ShapeGeneralization::GeneralizeInOrOutput(nlohmann::json &inputsJson, nlohmann::json &dyInputsInfo,
                                           const TbeOpParam &input, const bool &dynamicRankSupport,
                                           const std::string &opPattern)
{
    const std::vector<TbeOpTensor> &tensors = input.GetTensors();
    if (tensors.size() == 0) {
        inputsJson.emplace_back();
        dyInputsInfo.emplace_back();
        return;
    }

    OP_VALUE_DEPEND valueDependFlag = input.GetValueDepend();
    TensorType tensorType;
    input.GetType(tensorType);
    bool case1 = (valueDependFlag == VALUE_DEPEND_OPTIONAL);
    bool case2 = ((valueDependFlag == VALUE_DEPEND_IGNORE) && (tensorType == TT_OPT || tensorType == TT_REQ));
    TE_DBGLOG("ValueDependFlag is %u, tensorType is %u.", valueDependFlag, tensorType);

    json inputJson;
    json dyInputInfo;
    if (case1 || case2) {
        // generalize
        GeneralizeWithoutValueDepend(tensors, inputJson, dyInputInfo, opPattern, dynamicRankSupport);
    } else {
        // no generalize
        FeedCurInputShapeToJson(tensors, inputJson, dyInputInfo);
    }

    TE_DBGLOG("inputJson=%s, dyInputInfo=%s.", inputJson.dump().c_str(), dyInputInfo.dump().c_str());
    if (tensorType == TT_DYN) {
        inputsJson.emplace_back(inputJson);
        dyInputsInfo.emplace_back(dyInputInfo);
    } else {
        inputsJson.emplace_back(inputJson.back());
        dyInputsInfo.emplace_back(dyInputInfo.back());
    }
    TE_DBGLOG("inputsJson=%s, dyInputsInfo=%s.", inputsJson.dump().c_str(), dyInputsInfo.dump().c_str());
}

void ShapeGeneralization::GeneralizeWithoutValueDepend(const std::vector<TbeOpTensor> &tensors, nlohmann::json &inputJson,
                                                   nlohmann::json &dyInputInfo, const std::string &opPattern,
                                                   const bool &dynamicRankSupport)
{
    for (const TbeOpTensor &tensor : tensors) {
        std::string curFormat;
        std::string curOriFormat;
        std::vector<int64_t> curShape;
        std::vector<int64_t> curOriShape;
        std::string dataType;
        nlohmann::json curTensorShapeInfo;
        nlohmann::json curTensorDyInfo;

        if (opPattern == FORMAT_AGNOSTIC) {
            curFormat = FORMAT_ND;
        } else {
            tensor.GetFormat(curFormat);
        }

        tensor.GetOriginFormat(curOriFormat);
        tensor.GetOriginShape(curOriShape);
        tensor.GetType(dataType);
        curTensorShapeInfo[DTYPE] = dataType;
        curTensorShapeInfo[FORMAT] = curFormat;
        curTensorDyInfo[ORI_FORMAT] = curOriFormat;
        curTensorDyInfo[ORI_SHAPE] = curOriShape;

        // generalize shape with default binary rule
        if (dynamicRankSupport) {
            // shape=(-2,)
            curShape.emplace_back(DYNAMIC_SHAPE_DIM);
        } else {
            tensor.GetShape(curShape);

            // deal with shape=(1,) or (32,)
            if (curShape.size() > 0) {
                std::fill(curShape.begin(), curShape.end(), -1);
            } else {
                // deal with shape = []
                curShape.push_back(-1);
            }
        }
        curTensorShapeInfo[SHAPE] = curShape;

        curTensorDyInfo.update(curTensorShapeInfo);

        FeedDynamicInOutInfo(tensor, curTensorDyInfo);

        TE_DBGLOG("After generalize. (opPattern=%s, dynamicRankSupport=%u, curTensorShapeInfo=%s, curTensorDyInfo=%s)",
                  opPattern.c_str(), dynamicRankSupport,
                  curTensorShapeInfo.dump().c_str(), curTensorDyInfo.dump().c_str());
        dyInputInfo.emplace_back(curTensorDyInfo);
        inputJson.emplace_back(curTensorShapeInfo);
    }
    return;
}

void ShapeGeneralization::FeedCurInputShapeToJson(const std::vector<TbeOpTensor> &tensors, nlohmann::json &inputJson,
                                              nlohmann::json &dyInputInfo)
{
    for (const auto &tensor : tensors) {
        nlohmann::json curTensorShapeInfo;
        nlohmann::json curTensorDyInfo;
        curTensorShapeInfo[SHAPE] = tensor.GetShape();
        curTensorDyInfo[ORI_SHAPE] = tensor.GetShape();
        curTensorShapeInfo[FORMAT] = tensor.GetFormat();
        curTensorDyInfo[ORI_FORMAT] = tensor.GetFormat();
        curTensorShapeInfo[DTYPE] = tensor.GetType();
        FeedDynamicInOutInfo(tensor, curTensorDyInfo);
        curTensorDyInfo.update(curTensorShapeInfo);
        dyInputInfo.emplace_back(curTensorDyInfo);
        inputJson.emplace_back(curTensorShapeInfo);
    }
}

void ShapeGeneralization::FeedDynamicInOutInfo(const TbeOpTensor &tensor, nlohmann::json &curDynamicInfo)
{
    curDynamicInfo[IS_FIRST_LAYER] = tensor.GetFirstLayer();
    curDynamicInfo[ADDR_TYPE] = tensor.GetAddrType();
    curDynamicInfo[L1_WORKSPACE_SIZE] = tensor.GetL1WorkspaceSize();
    curDynamicInfo[L1_FUSION_TYPE] = tensor.GetL1FusionType();
    curDynamicInfo[SPLIT_INDEX] = tensor.GetSplitIndex();
    curDynamicInfo[L1_ADDR_FLAG] = tensor.GetL1AddrFlag();
    curDynamicInfo[L1_ADDR_OFFSET] = tensor.GetAddrOffset();
    curDynamicInfo[L1_VALID_SIZE] = tensor.GetL1ValidSize();
    TE_FUSION_CHECK(tensor.HasConstValue(), SetBinaryConstValue(tensor, curDynamicInfo));
    TE_FUSION_CHECK(tensor.GetValidShape().size() > 0, curDynamicInfo[VALID_SHAPE] = tensor.GetValidShape());
    TE_FUSION_CHECK(tensor.GetSliceOffset().size() > 0, curDynamicInfo[SLICE_OFFSET] = tensor.GetSliceOffset());
    TE_FUSION_CHECK(tensor.GetTotalShape().size() > 0, curDynamicInfo[TOTAL_SHAPE] = tensor.GetTotalShape());

    GeneralizeAndSetRange(curDynamicInfo, tensor);
}

void ShapeGeneralization::SetBinaryConstValue(const TbeOpTensor &tensor, nlohmann::json &curDynamicInfo)
{
    bool isConstValueNone = false;
    nlohmann::json curConstValueJson;
    std::string dtypeStr;
    ATTR_DTYPE dtype;
    TbeAttrValue constValueRange;

    (void)tensor.IsConstValueNone(isConstValueNone);
    if (isConstValueNone) {
        bool isConstValueRange = false;
        (void)tensor.GetConstValueRangeFlag(isConstValueRange);
        if (isConstValueRange) {
            (void)tensor.GetConstValueRange(constValueRange);
            json constValueRangeJson;
            NullDesc nullDesc;
            GetAttrValueToJson(constValueRange, constValueRangeJson, nullDesc);
            curConstValueJson[VALUE] = constValueRangeJson;
            dtype = constValueRange.GetType();
            TE_FUSION_CHECK(TbeAttrDtypeToString(dtype, dtypeStr), curConstValueJson[DTYPE] = dtypeStr);

            curDynamicInfo[CONST_VALUE_RANGE] = curConstValueJson;
        }
    } else {
        TbeAttrValue constValue;
        (void)tensor.GetConstValue(constValue);
        json constValueJson;
        NullDesc nullDesc;
        GetAttrValueToJson(constValue, constValueJson, nullDesc);
        curConstValueJson[VALUE] = constValueJson;
        dtype = constValueRange.GetType();
        TE_FUSION_CHECK(TbeAttrDtypeToString(dtype, dtypeStr), curConstValueJson[DTYPE] = dtypeStr);

        curDynamicInfo[CONST_VALUE] = curConstValueJson;

        bool hasValidNullDesc = false;
        json nullDescJsonList = json::array({});
        TE_DBGLOG("nullDesc.nullType is %d.", nullDesc.nullType);
        for (auto iter = nullDesc.nullDesc.begin();
            iter != nullDesc.nullDesc.end();
            iter++) {
            json itemValue;
            // nan/inf/-inf
            if (*iter == KEY_INF || *iter == KEY_NEGTIVE_INF || *iter == KEY_NAN) {
                itemValue = *iter;
                hasValidNullDesc = true;
            }
            nullDescJsonList.push_back(itemValue);
        }

        if (hasValidNullDesc) {
            curDynamicInfo[CONST_VALUE_NULL_DESC] = nullDescJsonList;
        }
    }
}

void ShapeGeneralization::GeneralizeAndSetRange(nlohmann::json &desc, const TbeOpTensor &tensor)
{
    std::vector<int64_t> currShape;
    std::vector<int64_t> oriShape;
    tensor.GetShape(currShape);
    tensor.GetOriginShape(oriShape);

    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    bool hasSet = tensor.GetShapeRange(shapeRange);
    GetGeneralizedRange(hasSet, currShape, shapeRange);
    desc[RANGE] = shapeRange;

    std::vector<std::pair<int64_t, int64_t>> oriShapeRange;
    hasSet = tensor.GetOriginShapeRange(oriShapeRange);
    GetGeneralizedRange(hasSet, oriShape, oriShapeRange);
    desc[ORI_RANGE] = oriShapeRange;

    std::vector<std::pair<int64_t, int64_t>> valueRange;
    hasSet = tensor.GetValueRange(valueRange);
    if (hasSet) {
        desc[VALUE_RANGE] = valueRange;
    }
}

void ShapeGeneralization::GetGeneralizedRange(const bool &hasSet, const std::vector<int64_t> &currShape,
                                          std::vector<std::pair<int64_t, int64_t>> &shapeRange)
{
    if (hasSet && (shapeRange.size() > 0)) {
        return;
    }

    for (const auto &shape : currShape) {
        if (shape > 0) {
            shapeRange.emplace_back(shape, shape);
        } else {
            shapeRange.emplace_back(0, -1);
        }
    }
}

void ShapeGeneralization::FillValueNullDesc(const NullDesc &nullDesc, nlohmann::json &valueStaticJson,
                                      nlohmann::json &valueDynamicJson, bool &thisAttrHasNullDesc)
{
    if (nullDesc.nullType == NullType::LIST_VALUE) {
        json nullDescJsonList = json::array({});
        for (auto it = nullDesc.nullDesc.begin(); it != nullDesc.nullDesc.end(); it++) {
            json itemValue; // default is null
            // inf/-inf/nan
            if (*it == KEY_NAN || *it == KEY_INF || *it == KEY_NEGTIVE_INF) {
                itemValue = *it;
                thisAttrHasNullDesc = true;
            }
            nullDescJsonList.push_back(itemValue);
        }
        if (thisAttrHasNullDesc) {
            valueDynamicJson["value_null_desc"] = nullDescJsonList;
            valueStaticJson["value_null_desc"] = nullDescJsonList;
        }
    } else if (nullDesc.nullType == NullType::SINGLE_VALUE) {
        std::string value = nullDesc.nullDesc[0];
        json nullDescJson;
        // inf/-inf/nan
        if (value == KEY_NAN || value == KEY_INF || value == KEY_NEGTIVE_INF) {
            nullDescJson = value;
            thisAttrHasNullDesc = true;
        }

        if (thisAttrHasNullDesc) {
            valueDynamicJson["value_null_desc"] = nullDescJson;
            valueStaticJson["value_null_desc"] = nullDescJson;
        }
    }
}

void ShapeGeneralization::GeneralizeFusionOpShapeAndFormat(const std::string &opPattern,
                                                       const TensorType &tensorType,
                                                       const OP_VALUE_DEPEND &valueDepend,
                                                       const bool &dynamicRankFlag,
                                                       nlohmann::json &tensorJson)
{
    if (tensorJson.empty()) {
        return;
    }
    if (tensorJson.find(FORMAT) == tensorJson.end()) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "Json data contains no 'format'. \n%s", tensorJson.dump().c_str());
        return;
    }
    if (tensorJson.find(SHAPE) == tensorJson.end()) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "Json data contains no 'shape'. \n%s", tensorJson.dump().c_str());
        return;
    }

    bool case1 = (valueDepend == VALUE_DEPEND_OPTIONAL);
    bool case2 = ((valueDepend == VALUE_DEPEND_IGNORE) && (tensorType == TT_OPT || tensorType == TT_REQ));
    if (case1 || case2) {
        if (opPattern == FORMAT_AGNOSTIC) {
            tensorJson[FORMAT] = FORMAT_ND;
        }
        if (dynamicRankFlag) {
            json newShape;
            newShape.emplace_back(DYNAMIC_SHAPE_DIM);
            tensorJson[SHAPE] = newShape;
        } else {
            std::fill(tensorJson[SHAPE].begin(), tensorJson[SHAPE].end(), -1);
        }
    }
    return;
}

bool ShapeGeneralization::BinaryGeneralizeWithRegisterFunc(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                       GeneralizedResult &generalizedResult)
{
    std::string opName;
    (void)tbeOpInfo->GetName(opName);
    nlohmann::json &generalizedRes = generalizedResult.generalizedAllJson;
    if (!PythonApiCall::Instance().GetCheckAndGeneralizedResFromTbe(*tbeOpInfo, BINARY_BUILD_TYPE, generalizedRes)) {
        TE_DBGLOG("GetCheckAndGeneralizedResFromTbe for node %s did not succeed.", opName.c_str());
        return false;
    }

    if (generalizedRes.is_null()) {
        TE_DBGLOGF("Node %s func generalize res is null, use default generalize rule.", opName.c_str());
        return true;
    }

    TE_DBGLOGF("Single op %s, generalizedRes is: %s.", opName.c_str(), generalizedRes.dump().c_str());

    TE_DBGLOGF("Before parse generalize op %s, staticInfo is: %s.", opName.c_str(),
               generalizedResult.staticJson.dump().c_str());
    if (!ParseGeneralizedResToBinaryJson(generalizedResult.generalizedAllJson, true,
                                         generalizedResult.staticJson, opName)) {
        return false;
    }

    // dynamicInfo attrs may be generalized by default rule, need to changed by op generalized result
    TE_DBGLOGF("Before parse generalize op %s, dynamicInfo is: %s.", opName.c_str(),
               generalizedResult.dynamicJson.dump().c_str());
    if (!ParseGeneralizedResToBinaryJson(generalizedResult.generalizedAllJson, false,
                                         generalizedResult.dynamicJson, opName)) {
        return false;
    }

    TE_DBGLOGF("Single op %s after generalized staticInfoJson is: %s, dynamicJson is: %s.", opName.c_str(),
               generalizedResult.staticJson.dump().c_str(),
               generalizedResult.dynamicJson.dump().c_str());

    return true;
}

/*
* generalizedRes:
* [input, output, 1.0, True]
* [input, output, null, True]
* [input, output, null, [1,8]]
*/
bool ShapeGeneralization::ParseGeneralizedResToBinaryJson(const nlohmann::json &generalizedRes,
                                                      bool isStaticInfo,
                                                      nlohmann::json &binaryInfo,
                                                      const std::string &opName)
{
    if (!generalizedRes.is_array()) {
        TE_WARNLOGF("Node %s shape and range generalization failed.", opName.c_str());
        return false;
    }

    for (auto singleInfo = generalizedRes.begin(); singleInfo != generalizedRes.end(); ++singleInfo) {
        json singleInfoJson = *singleInfo;
        if (singleInfoJson.find("result") != singleInfoJson.end()) {
            TE_WARNLOGF("Node %s current shape or range not supported by tbe.", opName.c_str());
            return false;
        }

        TE_DBGLOGF("Node %s single info json is: %s, binaryInfo %s.",
                   opName.c_str(), singleInfoJson.dump().c_str(), binaryInfo.dump().c_str());

        if (!singleInfoJson.is_array()) {
            TE_WARNLOGF("Node %s shape and range generalization failed.", opName.c_str());
            return false;
        }

        nlohmann::json &inputsJson = binaryInfo[INPUTS];
        nlohmann::json &outputsJson = binaryInfo[OUTPUTS];
        nlohmann::json::iterator inputOrOutput = singleInfoJson.begin();
        bool res0 = ParseInOrOutputJsonByTensor(singleInfoJson, inputOrOutput, inputsJson);
        bool res1 = ParseInOrOutputJsonByTensor(singleInfoJson, inputOrOutput, outputsJson);
        if (!(res0 && res1)) {
            TE_WARNLOGF("Parse generalized res from json failed.");
            return false;
        }

        auto attrsIter = binaryInfo.find(ATTRS);
        if (attrsIter != binaryInfo.end()) {
            nlohmann::json &attrsJson = attrsIter.value();
            ParseAttrsJsonByTensor(singleInfoJson, isStaticInfo, inputOrOutput, attrsJson);
        }
    }
    return true;
}

void ShapeGeneralization::ParseAttrsJsonByTensor(nlohmann::json &singleInfoJson, const bool &isStaticInfo,
                                             nlohmann::json::iterator &iterTensor,
                                             nlohmann::json &attrsJson)
{
    auto iterAttrs = attrsJson.begin();
    while (iterTensor != singleInfoJson.end() && iterAttrs != attrsJson.end()) {
        TE_DBGLOGF("Cur iterTensor json is %s, iterAttrs json is %s.", (*iterTensor).dump().c_str(),
            (*iterAttrs).dump().c_str());

        GetAttrGeneralizedResFromTensor(*iterTensor, *iterAttrs, isStaticInfo);

        ++iterTensor;
        ++iterAttrs;
    }
}

void ShapeGeneralization::GetAttrGeneralizedResFromTensor(const nlohmann::json &tensorJson,
                                                      nlohmann::json &binaryInfo,
                                                      const bool &isStaticInfo)
{
    if (binaryInfo.find(VALUE) == binaryInfo.end()) {
        TE_DBGLOGF("There is no value in binaryInfo %s.", binaryInfo.dump().c_str());
        return;
    }

    TE_DBGLOGF("Change static key json attr by generalize json(%s).", tensorJson.dump().c_str());

    if (isStaticInfo) {
        auto iter = binaryInfo.find(DTYPE);
        if (iter != binaryInfo.end()) {
            std::string dtype;
            try {
                dtype = iter.value().get<std::string>();
            } catch (const std::exception &e) {
                TE_WARNLOG("Failed to get bin attr dtype value. reason is %s.", e.what());
                return;
            }
            if (dtype == FLOAT || dtype == ATTR_LIST_FLOAT) {
                binaryInfo[VALUE] = nullptr;
                return;
            }
        }
    }
    binaryInfo[VALUE] = tensorJson;
}

bool ShapeGeneralization::ParseInOrOutputJsonByTensor(nlohmann::json &singleInfoJson,
                                                  nlohmann::json::iterator &iterTensor,
                                                  nlohmann::json &inOrOutputsJson)
{
    TE_DBGLOGF("Cur iterTensor singleInfoJson: %s, inOrOutputsJson: %s.",
               singleInfoJson.dump().c_str(), inOrOutputsJson.dump().c_str());
    auto iterInOrOutput = inOrOutputsJson.begin();
    while (iterTensor != singleInfoJson.end() && iterInOrOutput != inOrOutputsJson.end()) {
        TE_DBGLOGF("Cur iterTensor json is: %s.", (*iterTensor).dump().c_str());
        if (!(*iterTensor).is_null()) {
            if (!GetGeneralizedResFromTensor(*iterTensor, *iterInOrOutput)) {
                return false;
            }
        }
        ++iterTensor;
        ++iterInOrOutput;
    }
    return true;
}

void ShapeGeneralization::GetGeneralizedRes(const nlohmann::json &tensorJson, nlohmann::json &staticTensorJson)
{
    const auto tensorIter = tensorJson.find(ORI_FORMAT);
    auto staticTensorIter = staticTensorJson.find(ORI_FORMAT);
    if ((staticTensorIter != staticTensorJson.end()) && (tensorIter != tensorJson.end())) {
        staticTensorIter.value() = tensorIter.value();
    }
    SetJsonValue(SHAPE, tensorJson, staticTensorJson);
    SetJsonValue(DTYPE, tensorJson, staticTensorJson);
    SetJsonValue(FORMAT, tensorJson, staticTensorJson);
}

void ShapeGeneralization::SetJsonValue(const std::string &key, const nlohmann::json &srcJson, nlohmann::json &dstJson)
{
    const auto iter = srcJson.find(key);
    if (iter != srcJson.end()) {
        dstJson[key] = iter.value();
    }
}

bool ShapeGeneralization::GetGeneralizedResFromTensor(const nlohmann::json &tensorJson, nlohmann::json &staticTensorJson)
{
    TE_DBGLOGF("FromTensor tensorJson: %s, staticTensorJson: %s.",
               tensorJson.dump().c_str(), staticTensorJson.dump().c_str());
    if (tensorJson.is_array()) {
        if (tensorJson.size() != staticTensorJson.size()) {
            TE_DBGLOGF("tensorJson size(%zu) is not same as staticTensorJson size(%zu).",
                tensorJson.size(), staticTensorJson.size());
            return false;
        }
        for (size_t i = 0; i < tensorJson.size(); ++i) {
            GetGeneralizedRes(tensorJson[i], staticTensorJson[i]);
        }
    } else {
        GetGeneralizedRes(tensorJson, staticTensorJson);
    }

    return true;
}

void ShapeGeneralization::BinaryGeneralizeWithRegisterFuncGetGenerateRes(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                                     nlohmann::json &generalizedRes)
{
    bool hasRegisteredFunc;
    if (!PythonApiCall::Instance().CheckIsTbeGeneralizeFuncRegistered(*tbeOpInfo, hasRegisteredFunc)) {
        return;
    }
    if (!hasRegisteredFunc) {
        return;
    }
    std::string opName;
    (void)tbeOpInfo->GetName(opName);
    if (!PythonApiCall::Instance().GetCheckAndGeneralizedResFromTbe(*tbeOpInfo, BINARY_BUILD_TYPE, generalizedRes)) {
        TE_DBGLOG("GetCheckAndGeneralizedResFromTbe for node %s did not succeed.", opName.c_str());
        return;
    }
    if (generalizedRes.is_null()) {
        TE_DBGLOGF("Node %s func generalize res is null, use default generalize rule.", opName.c_str());
        return;
    }
    TE_DBGLOGF("Single op in fusion process %s, generalizedRes is: %s.", opName.c_str(), generalizedRes.dump().c_str());
    return;
}

bool ShapeGeneralization::GenerateNormalizeFusionOutputTmpJson(const ge::Node *currentNode,
                                                           std::unordered_set<ge::Node *> &allNodes,
                                                           GeneralizedResult &generalizedResult)
{
    TE_DBGLOG("GenerateNormalizeFusionOutputTmpJson Enter %s", currentNode->GetName().c_str());
    auto nodeDesc = currentNode->GetOpDesc();
    auto allOutputDesc = nodeDesc->GetAllOutputsDesc();
    auto allOutputAnchors = currentNode->GetAllOutDataAnchors();
    if (allOutputDesc.size() != allOutputAnchors.size()) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Output description and out anchor size not match: %zu %zu",
                           allOutputDesc.size(), allOutputAnchors.size());
    }

    json outputStaticJson;
    json outputDynamicJson;
    for (auto &anchor : allOutputAnchors) {
        if (!FormatFusionOutputTmpJson(currentNode, anchor, outputStaticJson, outputDynamicJson, allNodes,
                                       generalizedResult.generalizedAllJson)) {
            return false;
        }
    }

    CollectOutputJsonInfo(outputStaticJson, generalizedResult.staticJson);
    CollectOutputJsonInfo(outputDynamicJson, generalizedResult.dynamicJson);

    TE_DBGLOG("GenerateNormalizeFusionOutputTmpJson success %s", currentNode->GetName().c_str());
    return true;
}

void ShapeGeneralization::CollectOutputJsonInfo(const nlohmann::json &curOutputJson,
                                            nlohmann::json &oriOutputJson)
{
    if (curOutputJson.is_null()) {
        return;
    }
    if (curOutputJson.is_array()) {
        for (auto singleOutput = curOutputJson.begin(); singleOutput != curOutputJson.end(); ++singleOutput) {
            json singleOutputJson = *singleOutput;
            if (!singleOutputJson.is_null()) {
                oriOutputJson[OUTPUTS].emplace_back(singleOutputJson);
            }
        }
    } else {
        oriOutputJson[OUTPUTS].emplace_back(curOutputJson);
    }
}

bool ShapeGeneralization::FormatFusionOutputTmpJson(const ge::Node *currentNode, ge::OutDataAnchorPtr anchor,
                                                nlohmann::json &staticJson, nlohmann::json &dynamicJson,
                                                std::unordered_set<ge::Node *> &allNodes,
                                                const nlohmann::json &generalizedRes)
{
    ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(currentNode);
    if (tbeOpInfo == nullptr) {
        return false;
    }
    const std::vector<TbeOpParam> &outputs = tbeOpInfo->GetOutputs();
    auto idx = anchor->GetIdx();
    int outIdx = idx + static_cast<int>(tbeOpInfo->GetInputs().size());
    auto peers = anchor->GetPeerInDataAnchors();
    size_t refCnt = 0;
    bool hasOut = false;
    for (size_t i = 0; i < peers.size(); ++i) {
        std::string suffix;
        TeJsonAssemble::GetOutputSuffix(peers.at(i), allNodes, suffix, refCnt);
        if (hasOut && suffix == "out") {
            continue;
        }
        if (suffix == "out") {
            hasOut = true;
        }
        std::string opPattern;
        tbeOpInfo->GetOpStorePattern(opPattern);
        bool dynamicRankSupport = tbeOpInfo->IsSupportDynamicRank();
        if (suffix == "out") {
            GeneralizeInOrOutput(staticJson, dynamicJson, outputs[idx], dynamicRankSupport, opPattern);
            TE_DBGLOGF("Before generalize, staticJson:%s", staticJson.dump().c_str());
            auto &toBeGeneralizedJson = staticJson.back();
            if (!BinaryGeneralizeWithRegisterFuncFusionOp(generalizedRes,
                                                          toBeGeneralizedJson, outIdx, tbeOpInfo)) {
                TE_DBGLOGF("BinaryGeneralizeWithRegisterFuncFusionOp did not succeed.");
                return false;
            }
            TE_DBGLOGF("After generalize, staticJson:%s, toBeGeneralizedJson:%s", staticJson.dump().c_str(),
                       toBeGeneralizedJson.dump().c_str());
        }
    }
    return true;
}

bool ShapeGeneralization::GenerateNormalizeFusionAttrTmpJson(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                         GeneralizedResult &generalizedResult,
                                                         const bool &isSingle,
                                                         const nlohmann::json &generalizedRes)
{
    TE_DBGLOG("GenerateNormalizeFusionAttrTmpJson Enter %s", tbeOpInfo->GetName().c_str());
    const std::vector<TbeAttrValue> &attrValues = tbeOpInfo->GetAttrValues();
    json staticTmpJson;
    json dynamicTmpJson;
    int pos = static_cast<int>(tbeOpInfo->GetInputs().size()) + static_cast<int>(tbeOpInfo->GetOutputs().size());
    for (const TbeAttrValue &item : attrValues) {
        json valueStaticJson;
        json valueDynamicJson;
        if (!GenerateNormalizeFusionAttrTmpJsonPart1(tbeOpInfo, item, valueStaticJson, valueDynamicJson,
                                                     pos, generalizedRes)) {
            return false;
        }
        if (isSingle) {
            generalizedResult.staticJson[ATTRS].emplace_back(valueStaticJson);
            generalizedResult.dynamicJson[ATTRS].emplace_back(valueDynamicJson);
        } else {
            staticTmpJson[ATTRS].emplace_back(valueStaticJson);
            dynamicTmpJson[ATTRS].emplace_back(valueDynamicJson);
        }
        pos++;
    }
    if (!isSingle) {
        const std::string &opImplMode = tbeOpInfo->GetOpImplMode();
        if (!opImplMode.empty()) {
            staticTmpJson[IMPL_MODE] = opImplMode;
        }
        if (staticTmpJson.is_null()) {
            staticTmpJson = json::object();
        }
        generalizedResult.staticJson[GRAPH_OP_PARAMS].emplace_back(staticTmpJson);
        generalizedResult.dynamicJson[GRAPH_OP_PARAMS].emplace_back(dynamicTmpJson);
        generalizedResult.dynamicJson[INT64_MODE] = tbeOpInfo->GetFlagUseInt64();
    }

    std::string deterministic = TeContextUtils::GetDeterministic();
    if (deterministic == STR_TRUE) {
        generalizedResult.dynamicJson[DETERMINISTIC] = deterministic;
    }

    TE_DBGLOGF("Fusion op[%s] staticJson:%s, dynamicJson:%s",
               tbeOpInfo->GetName().c_str(), generalizedResult.staticJson.dump().c_str(),
               generalizedResult.dynamicJson.dump().c_str());
    return true;
}

bool ShapeGeneralization::GenerateNormalizeFusionAttrTmpJsonPart1(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                            const TbeAttrValue &item,
                                                            nlohmann::json &valueStaticJson,
                                                            nlohmann::json &valueDynamicJson,
                                                            const int &attrIdx,
                                                            const nlohmann::json &generalizedRes)
{
    nlohmann::json valueJson;
    bool isSupportAll = item.GetAttrSupAllFlag();
    ATTR_DTYPE attrType = item.GetType();
    std::map<ATTR_DTYPE, std::string>::const_iterator iter = g_attrDtypeToString.find(attrType);
    if (iter == g_attrDtypeToString.end()) {
        TE_WARNLOG("Not support attr dtype: %u.", attrType);
        return false;
    }
    std::string attrDtype = iter->second;
    NullDesc nullDesc;
    GetAttrValueToJson(item, valueJson, nullDesc);
    valueDynamicJson[DTYPE] = attrDtype;
    valueStaticJson[DTYPE] = attrDtype;
    valueDynamicJson[VALUE] = valueJson;
    valueStaticJson[VALUE] = valueJson;

    bool thisAttrHasNullDesc = false;
    TE_DBGLOG("nullDesc.nullType is %d.", nullDesc.nullType);
    FillValueNullDesc(nullDesc, valueStaticJson, valueDynamicJson, thisAttrHasNullDesc);

    TE_DBGLOG("Dynamic attrDtype is [%s], attr_val is %s.", attrDtype.c_str(), valueJson.dump().c_str());
    bool strOrBool = (attrType == ATTR_STR || attrType == ATTR_BOOL);
    if (isSupportAll && !strOrBool) {
        valueDynamicJson[VALUE] = nullptr;
        valueStaticJson[VALUE] = nullptr;
        if (thisAttrHasNullDesc) {
            valueStaticJson.erase("value_null_desc");
            valueDynamicJson.erase("value_null_desc");
        }
    }
    TE_DBGLOG("Static attrDtype is [%s], attr_val is %s.", attrDtype.c_str(), valueJson.dump().c_str());
    TE_DBGLOGF("Before generalize attr, staticJson:%s", valueStaticJson.dump().c_str());
    if (!BinaryGeneralizeWithRegisterFuncFusionOp(generalizedRes, valueStaticJson, attrIdx, tbeOpInfo)) {
        TE_DBGLOGF("BinaryGeneralizeWithRegisterFuncFusionOp did not succeed.");
        return false;
    }
    if (attrType == ATTR_FLOAT32 || attrType == ATTR_LIST_FLOAT32) {
        valueStaticJson[VALUE] = nullptr;
        if (thisAttrHasNullDesc) {
            valueStaticJson.erase("value_null_desc");
        }
    }
    TE_DBGLOGF("After generalize attr, staticJson:%s.", valueStaticJson.dump().c_str());

    TE_DBGLOGF("Before generalize attr, valueDynamicJson:%s", valueDynamicJson.dump().c_str());
    if (!BinaryGeneralizeWithRegisterFuncFusionOp(generalizedRes, valueDynamicJson, attrIdx, tbeOpInfo)) {
        TE_DBGLOGF("BinaryGeneralizeWithRegisterFuncFusionOp did not succeed.");
        return false;
    }
    TE_DBGLOGF("After generalize attr, valueDynamicJson:%s", valueDynamicJson.dump().c_str());

    return true;
}

bool ShapeGeneralization::GenFusionInputTmpJson(std::vector<ge::Node *> &teGraphNode, ge::Node *currentNode,
                                            std::unordered_set<ge::Node *> &allNodes,
                                            GeneralizedResult &generalizedResult)
{
    ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(currentNode);
    if (tbeOpInfo == nullptr) {
        TE_WARNLOG("Node %s GetTbeOpInfoByNode failed.", currentNode->GetName().c_str());
        return false;
    }

    const std::vector<TbeOpParam> &inputs = tbeOpInfo->GetInputs();
    map<int, int> inputOrder;
    uint32_t inputTensorSize = 0;
    if (!GenerateNormalizeFusionInputTmpJsonCheckSize(currentNode, inputs, inputOrder, inputTensorSize)) {
        TE_WARNLOG("Node %s GenerateNormalizeFusionInputTmpJsonCheckSize failed.", currentNode->GetName().c_str());
        return false;
    }

    TE_DBGLOGF("Node[%s] gen fusion input, input tensor size is %u.", currentNode->GetName().c_str(), inputTensorSize);
    uint32_t fusionInputIdx = 0;
    for (uint32_t idx = 0; idx < inputTensorSize; ++idx) {
        map<int, int>::iterator iter;
        bool isNull = false;
        if (!FusionInputCheckTensorNull(iter, inputOrder, idx, inputs, isNull)) {
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "FusionInputCheckTensorNull Check fail");
            return false;
        }
        if (!isNull) {
            ge::Node *preNode = GetPreviousNode(currentNode, idx);
            TE_FUSION_CHECK(preNode == nullptr, {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to get peer node, curr node name=[%s], index=[%d].",
                                   currentNode->GetName().c_str(), idx);
                return false;
            });
            if (allNodes.count(preNode) == 0) {
                TE_DBGLOGF("Node[%s] input[%u] is out input.", currentNode->GetName().c_str(), idx);
                json inputJsonTmp;
                bool bres = TeJsonAssemble::GenNodeDataJson(teGraphNode, currentNode, idx, iter->second, inputJsonTmp,
                                                       false);
                TE_FUSION_CHECK(!bres, {
                    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to gen data json, index=[%d].", iter->second);
                    return false;
                });
                bres = FusionInputFillJson(tbeOpInfo, inputJsonTmp, iter, generalizedResult, fusionInputIdx);
                TE_FUSION_CHECK(!bres, {
                    TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to fill input json, index=[%d].", iter->second);
                    return false;
                });
                fusionInputIdx++;
            }
        } else {
            TE_DBGLOGF("Fusion op node[%s] input[%u] is null.", currentNode->GetName().c_str(), idx);
        }
    }
    return true;
}

bool ShapeGeneralization::GenerateNormalizeFusionInputTmpJsonCheckSize(const ge::Node *currentNode,
                                                                 const std::vector<TbeOpParam> &inputs,
                                                                 map<int, int> &inputOrder,
                                                                 uint32_t &inputTensorSize)
{
    auto opDesc = currentNode->GetOpDesc();
    uint32_t inputDescSize = opDesc->GetInputsSize();
    for (size_t inputIdx = 0; inputIdx < inputs.size(); ++inputIdx) {
        TensorType type = inputs[inputIdx].GetType();
        if (type == TT_REQ || type == TT_OPT) {
            inputOrder.insert(pair<int, int>(inputTensorSize, inputIdx));
            inputTensorSize++;
        } else {
            // TT_DYN input tensor is list, need to count all list members
            const std::vector<TbeOpTensor> &tensors = inputs[inputIdx].GetTensors();
            for (size_t idx = 0; idx < tensors.size(); ++idx) {
                inputOrder.insert(pair<int, int>(inputTensorSize + idx, inputIdx));
            }
            inputTensorSize += tensors.size();
        }
    }

    if (inputDescSize > inputTensorSize) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_INFO, "Graph input size[%d] should be <= input size=[%d].", inputDescSize,
                           inputTensorSize);
        return false;
    }
    return true;
}

bool ShapeGeneralization::FusionInputCheckTensorNull(map<int, int>::iterator &iter, map<int, int> &inputOrder,
                                                 const uint32_t &idx, const std::vector<TbeOpParam> &inputs,
                                                 bool &isNull)
{
    iter = inputOrder.find(idx);
    if (iter == inputOrder.end()) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Failed to get input index pair, index=[%d].", idx);
        return false;
    }

    if (inputs[iter->second].GetTensors().size() == 0) {
        isNull = true;
    }
    return true;
}

bool ShapeGeneralization::FusionInputFillJson(const ConstTbeOpInfoPtr &tbeOpInfo,
                                          json &inputJsonTmp, map<int, int>::iterator &iter,
                                          GeneralizedResult &generalizedResult,
                                          const uint32_t &fusionInputIdx)
{
    TE_DBGLOG("FusionInputFillJson Enter");
    json inputStaicJson;
    json inputDynamicJson;
    // tefusion use key "data_type", binary opc use key "dtype"
    JudgeJsonExistAndSetByDiffKey(inputStaicJson, inputJsonTmp, DTYPE, "data_type");
    JudgeJsonExistAndSet(inputStaicJson, inputJsonTmp, FORMAT);
    JudgeJsonExistAndSet(inputStaicJson, inputJsonTmp, SHAPE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, ORI_FORMAT);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, ORI_SHAPE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, RANGE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, ORI_RANGE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, CONST_VALUE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, CONST_VALUE_RANGE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, IS_FIRST_LAYER);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, ADDR_TYPE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, SLICE_OFFSET);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, L1_WORKSPACE_SIZE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, L1_FUSION_TYPE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, L1_ADDR_OFFSET);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, L1_ADDR_FLAG);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, TOTAL_SHAPE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, VALID_SHAPE);
    JudgeJsonExistAndSet(inputDynamicJson, inputJsonTmp, SPLIT_INDEX);

    const std::vector<TbeOpParam> &inputs = tbeOpInfo->GetInputs();
    TensorType type = inputs[iter->second].GetType();
    if (type == TT_OPT) {
        generalizedResult.optionalInputIdx.emplace_back(fusionInputIdx);
    }
    OP_VALUE_DEPEND valueDepend = inputs[iter->second].GetValueDepend();
    bool dynamicRankFlag = tbeOpInfo->IsSupportDynamicRank();
    std::string opPattern;
    tbeOpInfo->GetOpStorePattern(opPattern);

    GeneralizeFusionOpShapeAndFormat(opPattern, type, valueDepend, dynamicRankFlag, inputStaicJson);
    if (!BinaryGeneralizeWithRegisterFuncFusionOp(generalizedResult.generalizedAllJson, inputStaicJson,
                                                  iter->second, tbeOpInfo)) {
        TE_DBGLOGF("BinaryGeneralizeWithRegisterFuncFusionOp did not succeed.");
        return false;
    }
    inputDynamicJson.update(inputStaicJson);
    generalizedResult.staticJson[INPUTS].emplace_back(inputStaicJson);
    generalizedResult.dynamicJson[INPUTS].emplace_back(inputDynamicJson);

    TE_DBGLOGF("After generalize, staticJson:%s", inputStaicJson.dump().c_str());
    return true;
}

bool ShapeGeneralization::BinaryGeneralizeWithRegisterFuncFusionOp(const nlohmann::json &generalizedRes,
                                                               nlohmann::json &binaryInfo,
                                                               const int &idx,
                                                               const ConstTbeOpInfoPtr &tbeOpInfo)
{
    std::string opName;
    (void)tbeOpInfo->GetName(opName);

    for (auto singleInfo = generalizedRes.begin(); singleInfo != generalizedRes.end(); ++singleInfo) {
        json singleInfoJson = *singleInfo;
        if (!singleInfoJson.is_array()) {
            TE_WARNLOGF("Node %s shape and range generalization failed.", opName.c_str());
            return false;
        }
        if (singleInfoJson.is_null()) {
            return false;
        }
        int pos = 0;
        for (auto tensorInfo = singleInfoJson.begin(); tensorInfo != singleInfoJson.end(); ++tensorInfo) {
            if (pos != idx) {
                pos++;
                continue;
            }
            json &singleTensorJson = *tensorInfo;
            if ((singleTensorJson.find(SHAPE) != singleTensorJson.end()) &&
                (singleTensorJson.find(FORMAT) != singleTensorJson.end()) &&
                (singleTensorJson.find(DTYPE) != singleTensorJson.end())) {
                binaryInfo[FORMAT] = singleTensorJson[FORMAT];
                binaryInfo[SHAPE] = singleTensorJson[SHAPE];
                binaryInfo[DTYPE] = singleTensorJson[DTYPE];
                TE_DBGLOGF("BinaryGeneralizeWithRegisterFuncFusionOp success opName:%s, binaryInfo:%s.", opName.c_str(),
                           binaryInfo.dump().c_str());
            } else {
                GetAttrGeneralizedResFromTensor(singleTensorJson, binaryInfo);
            }
            pos++;
        }
    }
    TE_DBGLOGF("BinaryGeneralizeWithRegisterFuncFusionOp opName:%s, binaryInfo:%s.", opName.c_str(),
               binaryInfo.dump().c_str());
    return true;
}

}
}