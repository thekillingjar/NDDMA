/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_BINARY_MANAGER_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_BINARY_MANAGER_H_
#include "binary/fusion_binary_info.h"
#include "inc/te_fusion_types.h"
#include "inc/te_fusion_util_constants.h"
#include <nlohmann/json.hpp>
#include <string>
#include <map>
#include <vector>
#include <mutex>

namespace te {
namespace fusion {
using nlohmann::json;
struct GeneralizedResult {
    nlohmann::json generalizedAllJson;
    nlohmann::json staticJson;
    nlohmann::json dynamicJson;
    std::vector<uint32_t> optionalInputIdx;

    GeneralizedResult()
    {
        staticJson[INPUTS] = json::array();
        staticJson[OUTPUTS] = json::array();
        dynamicJson[INPUTS] = json::array();
        dynamicJson[OUTPUTS] = json::array();
    }
    ~GeneralizedResult() = default;
};

class BinaryManager {
public:
    static BinaryManager& Instance();
    void Initialize(const std::map<std::string, std::string>& options);
    bool JudgeBinKernelInstalled(bool isOm, int64_t implType) const;
    bool CanReuseBinaryKernel(const OpBuildTaskPtr &opTask);
    void SetBinaryReuseAttr(const OpBuildTaskPtr &opTask) const;
    static bool TeGeneralize(const TbeOpInfo &tbeOpInfo, const TE_GENERALIZE_TYPE &generalizeType,
                             const ge::NodePtr &nodePtr);
    static bool FuzzyBuildSecondaryGeneralization(const OpBuildTaskPtr &opTask, const std::string &opName,
                                                  TbeOpInfoPtr &selectedOpInfo);
    void Finalize();

private:
   bool IsBinaryFileValid(bool isOm, const OpBuildTaskPtr &opTask, const json &binListJson,
                           std::string &jsonFilePath) const;
    static bool CheckDebugMode(const TbeOpInfo &opInfo);
    static bool IsAllDimsDyn(const ge::GeTensorDescPtr &tenosrDescPtr);
    static bool CheckIsSpecShape(const ge::OpDescPtr &opDesc);
    static bool IsSpecShapeAndRange(const OpBuildTaskPtr &opTask);
    static bool IsDynamicNodes(const OpBuildTaskPtr &opTask);
    static bool GetJsonFilePathInBinList(const OpBuildTaskPtr &opTask, const json &binListJson, std::string &jsonFilePath);
    bool GetJsonValueByJsonFilePath(const std::string &jsonFilePath, nlohmann::json &jsonValue);
    bool GetHighPerformaceAndCompileInfoFromJson(const OpBuildTaskPtr &opTask, const std::string &jsonFilePath);
    static bool MatchSingleOpAttrs(const json &opParamsJson, const json &binJson);
    bool MatchDeterministic(const json &opParamsJson, const json &binJson) const;
    static void GenerateOpBinaryJsonKey(const OpBuildTaskPtr &opTask, bool hasOmKeyId,
                                        bool isOm, std::string &binaryJsonKey);
    static void ShapeLimitedGeneralize(int64_t &shapeValue, const int64_t &valueMax,
                                       const std::vector<std::pair<int64_t, int64_t>> &rangeSection,
                                       std::vector<std::pair<int64_t, int64_t>> &newShapeRange);
    static void GeneralizeWitLimitedRule(std::vector<int64_t> &shape, std::string &format,
                                         std::vector<std::pair<int64_t, int64_t>> &newShapeRange);
    static void ShapeGeneralization(std::vector<int64_t> &shape, std::string &format, const bool &isLimited,
                                    bool toDynamicRank, std::vector<std::pair<int64_t, int64_t>> &newShapeRange);
    static void ShapeGeneralizeToDynamicRank(std::vector<int64_t> &shape,
                                             std::vector<std::pair<int64_t, int64_t>> &newShapeRange);
    static void GetGeneralizedResultFromReturnTensor(nlohmann::json &tensorJson, nlohmann::json &generalizedInfo);
    static bool ParseGeneralizedResultFromTbe(nlohmann::json &generalizedRes, const std::string &opName,
                                              nlohmann::json &generalizedInfo);
    static void SetGeneralizeInfoToJson(const std::vector<TbeOpTensor> &tensors, OP_VALUE_DEPEND &valueDepend,
                                        const bool &isLimited, const bool &toDynamicRank,
                                        nlohmann::json &generalizedValidInfo);
    static void GetHighPerformanceResFromJson(const nlohmann::json &jsonValue, bool &isHighPerformaceRes);
    static void GetHighPerformanceResFromJsonList(const nlohmann::json &kernelList, bool &isHighPerformaceRes);
    static void SetOpResAccordingCompileInfo(const OpBuildTaskPtr &opTask, const nlohmann::json &kernelsCompileInfo);
    static void SetMaxKernelIdAndIsHighPerformanceRes(const nlohmann::json &jsonValue, const OpBuildTaskPtr &opTask);
    static void SetShapeAndValueToTensor(nlohmann::json &singleGeneralizeRes, ge::GeTensorDescPtr &tenosrDescPtr);
    static void SetGeneralizedInfoToNode(nlohmann::json &generalizedValidInfo, const ge::NodePtr &nodePtr);
    static bool TeGeneralizeWithRegisterFunc(const TbeOpInfo &tbeOpInfo, const ge::NodePtr &nodePtr);
    static void TeGeneralizeWithDefaultRuleByTbeOpInfo(const TbeOpInfo &tbeOpInfo, const bool &isLimited,
                                                       const ge::NodePtr &nodePtr);
    static void GeneralizeShapeAndValueByTensor(ge::GeTensorDescPtr &tenosrDescPtr);
    static void TeGeneralizeWithDefaultRuleByNode(const ge::NodePtr &nodePtr);
    static void ChangeShapeRangeExpressFromUnlimitedToMinMAx(std::vector<std::pair<int64_t, int64_t>> &shapeRange);
    static bool GetRangeLimitedType(const TbeOpInfo &tbeOpInfo, bool &isRangeLimited);
    static bool IsOpCanSecondaryGeneralization(const OpBuildTaskPtr &opTask, bool &isFormatAgnostic,
                                               bool &isRangeLimited, bool &toDynamicRank, bool &isRangeAgnostic);
    static void GeneralizeShapeToUnlimited(const std::vector<int64_t> &shape,
                                           std::vector<std::pair<int64_t, int64_t>> &shapeRange);
    static void SetGeneralizeInfoToTensors(std::vector<TbeOpTensor> &tensors, const bool &isFormatAgnostic,
                                           const bool &isRangeLimited, const bool &toDynamicRank,
                                           const bool &isRangeAgnostic);
    void GetBinaryPath(const OpBuildTaskPtr &opTask, bool isOm, bool isKernelOrModel, std::string &binaryPath) const;
    static bool CheckIsFuzzyBuild(const OpBuildTaskPtr &opTask);
    static bool CheckMatchInBinListByKey(const OpBuildTaskPtr &opTask, const std::string &key,
                                         const std::string &keyValue, json &binListJson);
    static bool GetBinaryVersionInfo(const std::string &verFilePath, std::string &binAdkVersion,
                                     std::string &binOppVersion);
    static bool GenBuildOptions(const OpBuildTaskPtr &opTask, json &staticKeyJson);
    static bool MatchRangeOriRange(const std::string &key, const json &opValue, const json &binValue);
    static void GetConstValue(const std::string &key, std::string &dtype, json &valueJson, const json &constJson);
    static bool MatchConstValue(const json &binValueJson);
    static bool MatchL1Params(const json &binValue, const json &opValue);
    static bool MatchNonRangeParams(const json &binValue, const json &opValue);
    static bool MatchSingleInputOutput(const json &opValue, const json &binValue,
                                       const std::vector<uint32_t> &optionalInputIdx, const uint32_t &idx);
    static bool MatchInputsOutputs(const std::string &key, const json &dynamicJson, const json &binJson,
                                   const std::vector<uint32_t> &optionalInputIdx);
    static bool CompareSingleAttr(const std::string &dtype, const json &opAttr, const json &binAttr);
    static bool MatchSingleAttr(const json &opAttr, const json &binAttr);
    static bool MatchFusionOpsAttrs(const json &opParamsJson, const json &binJson);
    static bool MatchAttrs(const OpBuildTaskPtr &opTask, const json &opParamsJson, const json &binJson);
    bool MatchOpParams(const OpBuildTaskPtr &opTask, json &binListJson, GeneralizedResult &generalizedResult) const;
    bool BinaryMatchWithStaticKeyAndDynInfo(const OpBuildTaskPtr &opTask, GeneralizedResult &generalizedResult,
                                            json &binListJson) const;
    bool SetOmBinaryReuseResult(const OpBuildTaskPtr &opTask, const std::string &jsonFilePath) const;
    static bool GetNodeStrAttr(const OpBuildTaskPtr &opTask, const std::string &attrKey, std::string &attrValue);
    static bool CheckIsCanReuseOmBinaryCompileRes(const OpBuildTaskPtr &opTask);
    void SetBinaryConfigPaths(const int64_t &imply_type, std::string &file_path);
    void SetRelocatableBinaryConfigPaths(const int64_t &imply_type, std::string &file_path);
    void GetCustomBinaryPath(const std::map<std::string, std::string>& options,
                             const std::string &parentPath, const bool isOm);
    void GetBuiltinBinaryPath(const std::map<std::string, std::string>& options,
                              const std::string &parentPath, const std::string &shortSocVersion, bool isOm);
    void GetBinaryOppPath(const std::map<std::string, std::string>& options);
    void ParseAllBinaryInfoConfigPath();
    void GetAllBinaryVersionInfo(bool isOm);
    bool GetBinaryFileName(const OpBuildTaskPtr &opTask, std::string &binaryFileRealPath,
                           bool hasOmKeyId, bool isOm) const;
    bool MatchFusionOpGraphPattern(const OpBuildTaskPtr &opTask, json &binListJson) const;
    bool MatchStaticKey(const OpBuildTaskPtr &opTask, json &staticKeyJson, json &binListJson) const;
    bool SetBinaryReuseResult(const OpBuildTaskPtr &opTask, const std::string &jsonFilePath);
    bool MatchStaticKeyWithOptionalInputNull(const OpBuildTaskPtr &opTask, json &binListJson,
                                             GeneralizedResult &generalizedResult) const;
    bool MatchSimplifiedKey(const OpBuildTaskPtr &opTask, std::string &jsonFilePath) const;
    bool ReuseBinKernelBySimpleKey(const OpBuildTaskPtr &opTask);
    bool ReuseKernelBinaryCompileRes(const OpBuildTaskPtr &opTask);
    bool FusionOpReuseOmBinary(const OpBuildTaskPtr &opTask, json &binListJson, std::string &omKeyId) const;
    bool ReuseOmBinaryCompileRes(const OpBuildTaskPtr &opTask, bool &hasOmKeyId);
    bool MatchAndReuseBinRes(const OpBuildTaskPtr &opTask);
    bool BackToSingleCheck(const OpBuildTaskPtr &opTask);
    void SetBuiltInOppLatestFlag(bool isOppLatest);
    bool GetBuiltInOppLatestFlag() const;
    bool CheckBinaryVersionMatch(bool isOmVersion, const OpBuildTaskPtr &opTask) const;
    bool CheckReuseBinaryCondition(const OpBuildTaskPtr &opTask) const;
    bool CheckAndReadBinaryFile(const OpBuildTaskPtr &opTask, json &binListJson, bool hasOmKeyId, bool isOm);
    bool CheckConditionsForReuse(const OpBuildTaskPtr &opTask) const;
    bool SingleOpReuseOmBinary(const OpBuildTaskPtr &opTask, json &binListJson) const;
    bool GetOpJitCompileForBinReuse(const OpBuildTaskPtr &opTask, bool &needCheckNext) const;
    bool CheckJitCompileForBinReuse(const OpBuildTaskPtr &opTask) const;
    void GetBinaryVersion(const OpBuildTaskPtr &opTask, bool isOm, std::string &adkVrsion,
                          std::string &oppVersion) const;
    bool GetSingleOpSceneFlag(const OpBuildTaskPtr &opTask, bool &isSingleOpScene) const;

    std::map<int64_t, std::string> binaryOppPath_; // binary opp path
    std::map<int64_t, std::string> binaryOppKernelPath_; // binary opp kernel path
    std::map<int64_t, std::string> binaryOmPath_; // binary om path
    std::map<int64_t, std::string> binaryOmModelPath_; // binary om model path
    std::map<int64_t, std::string> binAdkVersionMap_;
    std::map<int64_t, std::string> binOppVersionMap_;
    std::map<int64_t, std::string> binOmAdkVersionMap_;
    std::map<int64_t, std::string> binOmVersionMap_;
    std::map<int64_t, std::string> binaryConfigPathMap_;
    std::map<int64_t, std::string> relocatableBinaryConfigPathMap_;
    std::map<uint64_t, BinaryInfoBasePtr> binaryInfoPtrMap_{};
    std::map<uint64_t, BinaryInfoBasePtr> relocatableBinaryInfoPtrMap_{};
    bool builtInOppLatestFlag_ = false;
    bool initFlag_ = false;
    // op json file path -> json map
    std::unordered_map<std::string, nlohmann::json> opFileJsonMap_;
    mutable std::mutex opFileJsonMutex_;
    // op binary file path -> json map
    std::unordered_map<std::string, nlohmann::json> opBinaryJsonMap_;
    mutable std::mutex opBinaryJsonMutex_;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_BINARY_MANAGER_H_