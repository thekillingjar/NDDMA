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
#include <algorithm>
#include <fstream>
#include <sstream>
#include "graph/anchor.h"
#include "inc/te_fusion_check.h"
#include "python_adapter/python_api_call.h"
#include "common/common_utils.h"
#include "common/te_file_utils.h"
#include "common/fusion_common.h"
#include "common/tbe_op_info_cache.h"
#include "common/te_config_info.h"
#include "common/te_context_utils.h"
#include "assemble_json/te_attr_utils.h"
#include "assemble_json/te_json_utils.h"
#include "assemble_json/te_json_assemble.h"
#include "binary/generate_simple_key.h"
#include "ge_common/ge_api_types.h"

namespace te {
namespace fusion {
using namespace ge;
using namespace nlohmann;
namespace {
    constexpr const char *JSON_FILE_PATH = "jsonFilePath";
    constexpr const char *BINARY_OM_KEY_ID = "omKeyId";
    constexpr const char *BIN_INFO = "binInfo";
    constexpr const char *BIN_LIST = "binList";
}

void BinaryManager::GenerateOpBinaryJsonKey(const OpBuildTaskPtr &opTask, bool hasOmKeyId,
                             bool isOm, std::string &binaryJsonKey)
{
    std::ostringstream binaryJsonKeyOss;
    for (const auto &node : opTask->opNodes) {
        if (node == nullptr) {
            continue;
        }
        binaryJsonKeyOss << node->GetType();
        int64_t implType = -1;
        if (ge::AttrUtils::GetInt(node->GetOpDesc(), FE_IMPLY_TYPE, implType)) {
            binaryJsonKeyOss << "_" << implType;
        }
    }
    binaryJsonKeyOss << "_" << opTask->opNodes.size() << "_" << hasOmKeyId << isOm;
    binaryJsonKey = binaryJsonKeyOss.str();
}

bool BinaryManager::CheckBinaryVersionMatch(bool isOmVersion,
                                              const OpBuildTaskPtr &opTask) const
{
    std::string binAdkVersion;
    std::string binOppVersion;
    GetBinaryVersion(opTask, isOmVersion, binAdkVersion, binOppVersion);

    TE_DBGLOG("Get %s versions. binAdkVersion: [%s], curAdkVersion: [%s], binOppVersion:[%s], curOppVersion: [%s]",
              isOmVersion ? "om" : "opp", binAdkVersion.c_str(), TeConfigInfo::Instance().GetAdkVersion().c_str(),
              binOppVersion.c_str(), TeConfigInfo::Instance().GetOppVersion().c_str());

    if (binAdkVersion != TeConfigInfo::Instance().GetAdkVersion() ||
        binOppVersion != TeConfigInfo::Instance().GetOppVersion()) {
        TE_DBGLOG("Binary version mismatch detected, attempting to reuse binary from a different version");
        return false;
    }

    TE_DBGLOG("Node[%s] get binary version matched", GetTaskNodeName(opTask).c_str());
    return true;
}

bool BinaryManager::CheckIsFuzzyBuild(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() == 1) {
        return opTask->buildType == FUZZILY_BUILD;
    }
    bool bres = false;
    for (auto &node : opTask->opNodes) {
        ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        if (tbeOpInfo == nullptr) {
            TE_WARNLOGF("Node %s GetTbeOpInfoByNode failed.", opTask->outNode->GetName().c_str());
            return false;
        }

        bres = (bres || (tbeOpInfo->GetBuildType() == FUZZILY_BUILD));
    }
    return bres;
}

bool BinaryManager::CheckReuseBinaryCondition(const OpBuildTaskPtr &opTask) const
{
    std::string performanceMode = TeContextUtils::GetPerformanceMode();
    if (!performanceMode.empty() && performanceMode != "normal") {
        TE_DBGLOGF("Node(%s) performance mode(%s) is not normal. Need to compile",
            GetTaskNodeName(opTask).c_str(), performanceMode.c_str());
        return false;
    }

    if (!CheckBinaryVersionMatch(false, opTask)) {
        TE_WARNLOG("Node(%s) opp version does not match with opp-kernel version, but it has no effect on binary reuse",
            GetTaskNodeName(opTask).c_str());
    }

    TE_DBGLOG("Node[%s] successfully checked reuse binary condition", GetTaskNodeName(opTask).c_str());
    return true;
}

bool BinaryManager::CheckAndReadBinaryFile(const OpBuildTaskPtr &opTask, json &binListJson,
                                             bool hasOmKeyId, bool isOm)
{
    std::string opBinaryJsonKey;
    GenerateOpBinaryJsonKey(opTask, hasOmKeyId, isOm, opBinaryJsonKey);
    std::lock_guard<std::mutex> lock_guard(opBinaryJsonMutex_);
    auto iter = opBinaryJsonMap_.find(opBinaryJsonKey);
    if (iter != opBinaryJsonMap_.cend()) {
        TE_DBGLOG("Op binary json is found for node[%s]", GetTaskNodeName(opTask).c_str());
        binListJson = iter->second;
        return true;
    }

    std::string binaryFileRealPath;
    if (!GetBinaryFileName(opTask, binaryFileRealPath, hasOmKeyId, isOm)) {
        TE_DBGLOG("Node(%s) do not get binary file name", GetTaskNodeName(opTask).c_str());
        return false;
    }

    // read file json content
    json jsonValue;
    if (!TeFileUtils::GetJsonValueFromJsonFile(binaryFileRealPath, jsonValue)) {
        TE_DBGLOG("Node(%s) not read binary file(%s).",
                  GetTaskNodeName(opTask).c_str(), binaryFileRealPath.c_str());
        return false;
    }

    // resolve json
    const auto binListJsonIter = jsonValue.find(BIN_LIST);
    if (binListJsonIter == jsonValue.end()) {
        TE_WARNLOG("Node(%s) binary json file(%s) has no bin list.",
            GetTaskNodeName(opTask).c_str(), binaryFileRealPath.c_str());
        return false;
    }

    binListJson = binListJsonIter.value();
    if (!binListJson.is_array()) {
        TE_WARNLOG("Node(%s) binary json file format is incorrect! 'binList' should be array.",
            GetTaskNodeName(opTask).c_str());
        return false;
    }
    opBinaryJsonMap_.emplace(opBinaryJsonKey, binListJson);
    TE_DBGLOG("Node[%s] resolve binary file successfully", GetTaskNodeName(opTask).c_str());
    return true;
}

bool BinaryManager::CheckMatchInBinListByKey(const OpBuildTaskPtr &opTask, const std::string &key,
                                               const std::string &keyValue, json &binListJson)
{
    if (!TeJsonUtils::FilterJsonValueFromJsonListByKey(key, keyValue, binListJson)) {
        TE_DBGLOGF("The node %s %s(%s) does not match in binary file",
            GetTaskNodeName(opTask).c_str(), key.c_str(), keyValue.c_str());
        return false;
    }

    if (key == BINARY_OM_KEY_ID) {
        binListJson = binListJson.back();
    } else {
        binListJson = binListJson;
    }

    TE_DBGLOGF("The %s(%s) match in binary file. Matched bin json is %s", key.c_str(), keyValue.c_str(),
        binListJson.dump().c_str());
    return true;
}

bool BinaryManager::IsBinaryFileValid(bool isOm, const OpBuildTaskPtr &opTask,
                                        const json &binListJson,
                                        std::string &jsonFilePath) const
{
    // kernel path + binFilePath
    GetBinaryPath(opTask, isOm, true, jsonFilePath);

    std::string opName = GetTaskNodeName(opTask);
    if (!GetJsonFilePathInBinList(opTask, binListJson, jsonFilePath)) {
        TE_DBGLOGF("Node(%s) get json file path did not succeed.", opName.c_str());
        return false;
    }
    if (!TeFileUtils::IsObjFileExsit(opName, jsonFilePath)) {
        TE_DBGLOG("Node(%s) check op .o did not succeed.", opName.c_str());
        return false;
    }
    TE_DBGLOG("Node(%s) check binary file success", opName.c_str());
    return true;
}

bool BinaryManager::CheckDebugMode(const TbeOpInfo &opInfo)
{
    return TeConfigInfo::Instance().GetOpDebugLevel() != OpDebugLevel::Disable || !opInfo.GetOpDebugConfig().empty();
}

bool BinaryManager::IsAllDimsDyn(const ge::GeTensorDescPtr &tenosrDescPtr)
{
    const ge::GeShape &shapeDims = tenosrDescPtr->MutableShape();
    if (shapeDims.IsUnknownDimNum()) {
        return true;
    }

    const size_t &shapeDimNum = shapeDims.GetDimNum();
    if (shapeDimNum == 0U) {
        return false;
    }

    for (size_t idx = 0; idx < shapeDimNum; ++idx) {
        if (shapeDims.GetDim(idx) != UNKNOWN_DIM) {
            return false;
        }
    }

    std::vector<std::pair<int64_t, int64_t>> shapeRange;
    (void)tenosrDescPtr->GetShapeRange(shapeRange);
    bool isUnknownRange = std::all_of(shapeRange.begin(), shapeRange.end(),
        [](const std::pair<int64_t, int64_t> &dimRange) {
            return ((dimRange.first == RANGE_UNLIMITED_LOW_ZERO || dimRange.first == RANGE_UNLIMITED_LOW_ONE) &&
                    (dimRange.second == RANGE_UNLIMITED_UP));
        });
    return isUnknownRange;
}

bool BinaryManager::CheckIsSpecShape(const ge::OpDescPtr &opDesc)
{
    // op can't reuse binary when jit compile is set to true and in static shape(there is no input/output)
    bool hasInputOutput = false;
    for (const auto &tenosrDescPtr : opDesc->GetAllInputsDescPtr()) {
        if (tenosrDescPtr == nullptr) {
            continue;
        }
        hasInputOutput = true;
        if (!IsAllDimsDyn(tenosrDescPtr)) {
            return false;
        }
    }

    for (auto &tenosrDescPtr : opDesc->GetAllOutputsDescPtr()) {
        if (tenosrDescPtr == nullptr) {
            continue;
        }

        hasInputOutput = true;
        if (!IsAllDimsDyn(tenosrDescPtr)) {
            return false;
        }
    }

    return hasInputOutput;
}

bool BinaryManager::IsSpecShapeAndRange(const OpBuildTaskPtr &opTask)
{
    for (const auto &node : opTask->opNodes) {
        auto opDesc = node->GetOpDesc();
        if (!CheckIsSpecShape(opDesc)) {
            TE_DBGLOG("Node(%s) is not spec shape[-2,] or all dims is -1",
                      node->GetName().c_str());
            return false;
        }
    }

    TE_DBGLOG("Node is unknown shape.");
    return true;
}

bool BinaryManager::IsDynamicNodes(const OpBuildTaskPtr &opTask)
{
    for (const auto &node : opTask->opNodes) {
        ConstTbeOpInfoPtr tbeOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        if (tbeOpInfo == nullptr) {
            TE_WARNLOG("The tbeOpInfo of Node(%s) is null, cannot reuse binary result. Need to compile.",
                       node->GetName().c_str());
            return false;
        }
        if (!tbeOpInfo->GetIsUnknownShape()) {
            TE_DBGLOG("Node(%s) is not unknown shape, cannot reuse binary result. Need to compile.",
                      node->GetName().c_str());
            return false;
        }
    }
    return true;
}

bool BinaryManager::CheckConditionsForReuse(const OpBuildTaskPtr &opTask) const
{
    if (TeContextUtils::GetBuildInnerModel() == STR_FALSE) {
        TE_DBGLOG("Build inner model is false. Node(%s) no reuse binary files. Need to compile",
            GetTaskNodeName(opTask).c_str());
        return false;
    }

    if (CheckIsFuzzyBuild(opTask)) {
        TE_DBGLOGF("Node(%s) is fuzzy build mode, ready to reuse binary result.", GetTaskNodeName(opTask).c_str());
        return true;
    }
    TE_DBGLOG("Node(%s) is not fuzzy build mode", GetTaskNodeName(opTask).c_str());
    return CheckJitCompileForBinReuse(opTask);
}

bool BinaryManager::GetJsonFilePathInBinList(const OpBuildTaskPtr &opTask, const json &binListJson,
                                             std::string &jsonFilePath)                         
{
    std::string opName = GetTaskNodeName(opTask);
    auto iter = binListJson.find(BIN_INFO);
    if (iter == binListJson.end()) {
        TE_DBGLOGF("Node(%s) do not get bin info.", opName.c_str());
        return false;
    }

    const json &filePathJson = iter.value();
    iter = filePathJson.find(JSON_FILE_PATH);
    if (iter == filePathJson.end()) {
        TE_DBGLOGF("Node(%s) do not get bin info json file path.", opName.c_str());
        return false;
    }

    std::string binFilePath;
    try {
        binFilePath = iter.value().get<std::string>();
    } catch (const std::exception &e) {
        TE_WARNLOG("Not get node(%s) binFilePath value. reason is %s.", opName.c_str(), e.what());
        return false;
    }
    TE_DBGLOG("Node(%s) get bin info file path:%s.", opName.c_str(), binFilePath.c_str());
    std::string opsPathNamePrefix = "";
    if (opTask->opNodes.size() == 1 && opTask->opNodes[0] != nullptr) {
        (void)ge::AttrUtils::GetStr(opTask->opNodes[0]->GetOpDesc(), OPS_PATH_NAME_PREFIX, opsPathNamePrefix);
    }
    AssembleJsonPath(opsPathNamePrefix, jsonFilePath, binFilePath);
    return true;
}

bool BinaryManager::GetJsonValueByJsonFilePath(const std::string &jsonFilePath,
                                                 nlohmann::json &jsonValue)
{
    std::lock_guard<std::mutex> lock_guard(opFileJsonMutex_);
    const auto iter = opFileJsonMap_.find(jsonFilePath);
    if (iter == opFileJsonMap_.end()) {
        std::string jsonFileRealPath = RealPath(jsonFilePath);
        if (jsonFileRealPath.empty()) {
            TE_INFOLOGF("Json file path(%s) does not exist.", jsonFilePath.c_str());
            return false;
        }

        if (!TeFileUtils::GetJsonValueFromJsonFile(jsonFileRealPath, jsonValue)) {
            TE_DBGLOG("Get json_value from json file did not succeed.");
            return false;
        }
        opFileJsonMap_.emplace(jsonFilePath, jsonValue);
    } else {
        jsonValue = iter->second;
    }
    return true;
}

bool BinaryManager::GetHighPerformaceAndCompileInfoFromJson(const OpBuildTaskPtr &opTask,
                                                              const std::string &jsonFilePath)
{
    nlohmann::json jsonValue;
    if (!GetJsonValueByJsonFilePath(jsonFilePath, jsonValue)) {
        return false;
    }

    // save jsonValue to use in future for binary kernel cce build
    opTask->generalizedJson = jsonValue;
    nlohmann::json kernelsCompileInfo;
    if (jsonValue.find("compileInfo") != jsonValue.end()) {
        kernelsCompileInfo = jsonValue.at("compileInfo");
        TE_DBGLOG("Get compileInfo. (kernelsCompileInfo = %s)", kernelsCompileInfo.dump().c_str());
    } else {
        TE_WARNLOG("Find no compileInfo in json.");
        return false;
    }
    SetOpResAccordingCompileInfo(opTask, kernelsCompileInfo);

    SetMaxKernelIdAndIsHighPerformanceRes(jsonValue, opTask);

    return true;
}
} // namespace fusion
} // namespace te