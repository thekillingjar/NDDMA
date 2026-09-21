/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_SHAPE_GENERALIZATION_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_SHAPE_GENERALIZATION_H_
#include "binary/fusion_binary_info.h"
#include "inc/te_fusion_types.h"
#include "inc/te_fusion_util_constants.h"
#include "binary/binary_manager.h"
#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <vector>

namespace te {
namespace fusion {
using nlohmann::json;

class ShapeGeneralization {
public:
    static bool GeneralizeOps(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult);
private:
    static bool GeneralizeSingleOp(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult);
    static bool GeneralizeFusionOps(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult);
    static void GetOptionalInputIdxByTbeOpInfo(const ConstTbeOpInfoPtr &tbeOpInfo,
                                               GeneralizedResult &generalizedResult);
    static bool BinaryGeneralizeWithDefaultRule(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                GeneralizedResult &generalizedResult);
    static void GeneralizeInOrOutput(nlohmann::json &inputsJson, nlohmann::json &dyInputsInfo,
                                     const TbeOpParam &input, const bool &dynamicRankSupport,
                                     const std::string &opPattern);
    static void GeneralizeWithoutValueDepend(const std::vector<TbeOpTensor> &tensors, nlohmann::json &inputJson,
                                             nlohmann::json &dyInputInfo, const std::string &opPattern,
                                             const bool &dynamicRankSupport);
    static void FeedCurInputShapeToJson(const std::vector<TbeOpTensor> &tensors, nlohmann::json &inputJson,
                                        nlohmann::json &dyInputInfo);
    static void FeedDynamicInOutInfo(const TbeOpTensor &tensor, nlohmann::json &curDynamicInfo);
    static void SetBinaryConstValue(const TbeOpTensor &tensor, nlohmann::json &curDynamicInfo);
    static void GeneralizeAndSetRange(nlohmann::json &desc, const TbeOpTensor &tensor);
    static void GetGeneralizedRange(const bool &hasSet, const std::vector<int64_t> &currShape,
                                    std::vector<std::pair<int64_t, int64_t>> &shapeRange);
    static void FillValueNullDesc(const NullDesc &nullDesc, nlohmann::json &valueStaticJson,
                                  nlohmann::json &valueDynamicJson, bool &thisAttrHasNullDesc);
    static void GeneralizeFusionOpShapeAndFormat(const std::string &opPattern, const TensorType &tensorType,
                                                 const OP_VALUE_DEPEND &valueDepend, const bool &dynamicRankFlag,
                                                 nlohmann::json &tensorJson);
    static bool BinaryGeneralizeWithRegisterFunc(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                 GeneralizedResult &generalizedResult);
    static bool ParseGeneralizedResToBinaryJson(const nlohmann::json &generalizedRes, bool isStaticInfo,
                                                nlohmann::json &binaryInfo, const std::string &opName);
    static void ParseAttrsJsonByTensor(nlohmann::json &singleInfoJson, const bool &isStaticInfo,
                                       nlohmann::json::iterator &iterTensor, nlohmann::json &attrsJson);
    static void GetAttrGeneralizedResFromTensor(const nlohmann::json &tensorJson, nlohmann::json &binaryInfo,
                                                const bool &isStaticInfo = false);
    static bool ParseInOrOutputJsonByTensor(nlohmann::json &singleInfoJson, nlohmann::json::iterator &iterTensor,
                                            nlohmann::json &inOrOutputsJson);
    static void GetGeneralizedRes(const nlohmann::json &tensorJson, nlohmann::json &staticTensorJson);
    static void SetJsonValue(const std::string &key, const nlohmann::json &srcJson, nlohmann::json &dstJson);
    static bool GetGeneralizedResFromTensor(const nlohmann::json &tensorJson, nlohmann::json &staticTensorJson);
    static void BinaryGeneralizeWithRegisterFuncGetGenerateRes(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                               nlohmann::json &generalizedRes);
    static bool GenerateNormalizeFusionOutputTmpJson(const ge::Node *currentNode,
                                                     std::unordered_set<ge::Node *> &allNodes,
                                                     GeneralizedResult &generalizedResult);
    static void CollectOutputJsonInfo(const nlohmann::json &curOutputJson, nlohmann::json &oriOutputJson);
    static bool GenerateNormalizeFusionAttrTmpJson(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                   GeneralizedResult &generalizedResult, const bool &isSingle,
                                                   const nlohmann::json &generalizedRes);
    static bool GenerateNormalizeFusionAttrTmpJsonPart1(const ConstTbeOpInfoPtr &tbeOpInfo,
                                                        const TbeAttrValue &item,
                                                        nlohmann::json &valueStaticJson,
                                                        nlohmann::json &valueDynamicJson,
                                                        const int &attrIdx,
                                                        const nlohmann::json &generalizedRes);
    static bool GenFusionInputTmpJson(std::vector<ge::Node *> &teGraphNode, ge::Node *currentNode,
                                      std::unordered_set<ge::Node *> &allNodes, GeneralizedResult &generalizedResult);
    static bool GenerateNormalizeFusionInputTmpJsonCheckSize(const ge::Node *currentNode,
                                                             const std::vector<TbeOpParam> &inputs,
                                                             map<int, int> &inputOrder, uint32_t &inputTensorSize);
    static bool FusionInputCheckTensorNull(map<int, int>::iterator &iter, map<int, int> &inputOrder,
                                           const uint32_t &idx, const std::vector<TbeOpParam> &inputs, bool &isNull);
    static bool FusionInputFillJson(const ConstTbeOpInfoPtr &tbeOpInfo, json &inputJsonTmp,
        map<int, int>::iterator &iter, GeneralizedResult &generalizedResult, const uint32_t &fusionInputIdx);
    static bool FormatFusionOutputTmpJson(const ge::Node *currentNode, ge::OutDataAnchorPtr anchor,
                                          nlohmann::json &staticJson, nlohmann::json &dynamicJson,
                                          std::unordered_set<ge::Node *> &allNodes,
                                          const nlohmann::json &generalizedRes);
    static bool BinaryGeneralizeWithRegisterFuncFusionOp(const nlohmann::json &generalizedRes,
        nlohmann::json &binaryInfo, const int &idx, const ConstTbeOpInfoPtr &tbeOpInfo);
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_SHAPE_GENERALIZATION_H_