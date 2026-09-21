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
#include <dfxinfo_manager/dfxinfo_manager.h>
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
#include "binary/shape_generalization.h"

namespace te {
namespace fusion {
using namespace ge;
using namespace nlohmann;
namespace {
constexpr const char *BINARY_VERSION_FILE = "version.info";
constexpr const char *BINARY_ADK_VERSION = "adk_version";
constexpr const char *BINARY_OPS_VERSION = "ops_version";
constexpr const char *BINARY_FUSION_OPS_JSON_FILE_NAME = "fusion_ops";
constexpr const char *BINARY_GRAPH_PATTERN_ATTR = "graph_pattern";
constexpr const char *BINARY_GRAPH_PATTERN = "graphPattern";
constexpr const char *BUILD_OPTIONS = "buildOptions";
constexpr const char *BINARY_OM_KEY_ID_ATTR = "om_key_id";
constexpr const char *BINARY_OM_KEY_ID = "omKeyId";
constexpr const char *BINARY_CONFIG_MIDDLE_PAHT = "ai_core/tbe/kernel/config/";
constexpr const char *AI_CORE_TYPE = "ai_core";
constexpr const char *TRUE = "true";
}

BinaryManager& BinaryManager::Instance()
{
    static BinaryManager binManager;
    return binManager;
}

void BinaryManager::Initialize(const std::map<std::string, std::string>& options)
{
    if (initFlag_) {
        return;
    }
    GetBinaryOppPath(options);
    // get binary file version and cur version
    GetAllBinaryVersionInfo(true);
    GetAllBinaryVersionInfo(false);
    initFlag_ = true;
}

void BinaryManager::Finalize()
{
    initFlag_ = false;
    binaryOppPath_.clear();
    binaryOppKernelPath_.clear();
    binaryOmPath_.clear();
    binaryOmModelPath_.clear();
    binAdkVersionMap_.clear();
    binOppVersionMap_.clear();
    binOmAdkVersionMap_.clear();
    binOmVersionMap_.clear();
    binaryConfigPathMap_.clear();
    relocatableBinaryConfigPathMap_.clear();
    binaryInfoPtrMap_.clear();
    relocatableBinaryInfoPtrMap_.clear();
    opFileJsonMap_.clear();
    opBinaryJsonMap_.clear();
}

void BinaryManager::SetBinaryConfigPaths(const int64_t &imply_type, std::string &file_path)
{
    TE_DBGLOG("The current file path is %s.", file_path.c_str());
    size_t pos = file_path.find(AI_CORE_TYPE);
    if (pos != string::npos) {
        std::string binary_config_path = file_path.substr(0, pos) + BINARY_CONFIG_MIDDLE_PAHT;
        binaryConfigPathMap_.emplace(imply_type, binary_config_path);
        TE_DBGLOG("tbe current imply_type is [%lld], binary_config_path is [%s].",
                  imply_type, binary_config_path.c_str());
    }
}

void BinaryManager::SetRelocatableBinaryConfigPaths(const int64_t &imply_type, std::string &file_path)
{
    TE_DBGLOG("The current file path is: %s.", file_path.c_str());
    size_t pos = file_path.find(AI_CORE_TYPE);
    if (pos != string::npos) {
        std::string binary_config_path = file_path.substr(0, pos) + BINARY_CONFIG_MIDDLE_PAHT;
        relocatableBinaryConfigPathMap_.emplace(imply_type, binary_config_path);
        TE_DBGLOG("tbe current imply_type is [%lld], relocatable_kernel_info_config_path is [%s]",
                  imply_type, binary_config_path.c_str());
    }
}

void BinaryManager::GetCustomBinaryPath(const std::map<std::string, std::string>& options,
                                          const std::string &parentPath, const bool isOm)
{
    const std::string key = isOm ? OM_CUSTOM_BINARY_KEY : OP_CUSTOM_BINARY_KEY;
    std::map<std::string, std::string>::const_iterator it = options.find(key);
    if (it == options.end() || it->second.empty()) {
        TE_DBGLOGF("Not found custom binary key:%s in options", key.c_str());
        return;
    }
    const std::vector<std::string> pathVec = Split(it->second, ',');
    for (size_t i = 0; i < pathVec.size(); ++i) {
        size_t pos = pathVec[i].find('|');
        if (pos == std::string::npos) {
            TE_WARNLOGF("Config path:%s format is wrong.", pathVec[i].c_str());
            continue;
        }
        std::string impl = pathVec[i].substr(0, pos);
        int64_t implType;
        std::istringstream(impl) >> implType;
        std::string kernelPath = parentPath + pathVec[i].substr(pos + 1);
        SetBinaryConfigPaths(implType, kernelPath);
        SetRelocatableBinaryConfigPaths(implType, kernelPath);
        std::stringstream cfgPath;
        cfgPath << kernelPath << "config/";
        if (isOm) {
            binaryOmModelPath_.emplace(implType, kernelPath);
            binaryOmPath_.emplace(implType, cfgPath.str());
        } else {
            binaryOppKernelPath_.emplace(implType, kernelPath);
            binaryOppPath_.emplace(implType, cfgPath.str());
        }
        TE_DBGLOGF("Get binary custom key(%ld) of path(%s) and cfg(%s)", implType, kernelPath.c_str(),
                   cfgPath.str().c_str());
    }
}

void BinaryManager::GetBuiltinBinaryPath(const std::map<std::string, std::string>& options,
                                           const std::string &parentPath, const std::string &shortSocVersion, bool isOm)
{
    const std::string key = isOm ? OM_BUILTIN_BINARY_KEY : OP_BUILTIN_BINARY_KEY;
    std::map<std::string, std::string>::const_iterator it = options.find(key);
    if (it == options.end() || it->second.empty()) {
        TE_WARNLOGF("Not found builtin binary key: %s in options.", key.c_str());
        return;
    }
    size_t pos = it->second.find('|');
    if (pos == std::string::npos) {
        TE_WARNLOGF("Config format is wrong");
        return;
    }
    int64_t implType;
    std::istringstream(it->second.substr(0, pos)) >> implType;
    std::string kernelPath = parentPath + it->second.substr(pos + 1);
    SetBinaryConfigPaths(implType, kernelPath);
    SetRelocatableBinaryConfigPaths(implType, kernelPath);
    std::stringstream cfgPath;
    cfgPath << kernelPath;
    cfgPath << "config/" << shortSocVersion << "/";
    if (isOm) {
        binaryOmModelPath_.emplace(implType, kernelPath);
        binaryOmPath_.emplace(implType, cfgPath.str());
    } else {
        binaryOppKernelPath_.emplace(implType, kernelPath);
        binaryOppPath_.emplace(implType, cfgPath.str());
    }
    TE_DBGLOGF("Get binary opp key(%ld) of path(%s) and cfg(%s)", implType, kernelPath.c_str(),
               cfgPath.str().c_str());
    return;
}

/*
    op.binary.builtin, 6|/built-in/op_impl/ai_core/tbe/kernel/
    op.binary.custom,  2|/vendors/customize/op_impl/ai_core/tbe/kernel/, x|/vendors/mdc/op_impl/ai_core/tbe/kernel/
    om.binary.builtin, 6|/built-in/op_impl/ai_core/tbe/model/
    om.binary.custom,  2|/vendors/customize/op_impl/ai_core/tbe/model/, x|/vendors/mdc/op_impl/ai_core/tbe/model/
*/
void BinaryManager::GetBinaryOppPath(const std::map<std::string, std::string>& options)
{
    std::string oppParentPath = TeConfigInfo::Instance().GetOppRealPath();
    if (oppParentPath.empty()) {
        TE_WARNLOGF("ASCEND_OPP_PATH is invalid, unable to get binary opp path");
        return;
    }
    std::string shortSocVersion = TeConfigInfo::Instance().GetShortSocVersion();
    std::transform(shortSocVersion.begin(), shortSocVersion.end(), shortSocVersion.begin(), ::tolower);
    TE_DBGLOGF("Get short soc version(%s)", shortSocVersion.c_str());
    GetCustomBinaryPath(options, oppParentPath, true);
    GetCustomBinaryPath(options, oppParentPath, false);

    std::string oppLatestPath = RealPath(TeConfigInfo::Instance().GetEnvHomePath() + OPP_LATEST_PATH);
    if (!oppLatestPath.empty()) {
        TE_INFOLOGF("ASCEND_HOME_PATH/opp_latest is valid, ready to get binary info from this dictionary[%s]",
                    oppLatestPath.c_str());
        oppParentPath = std::move(oppLatestPath);
        SetBuiltInOppLatestFlag(true);
    }
    GetBuiltinBinaryPath(options, oppParentPath, shortSocVersion, true);
    GetBuiltinBinaryPath(options, oppParentPath, shortSocVersion, false);
    ParseAllBinaryInfoConfigPath();
}

void BinaryManager::SetBuiltInOppLatestFlag(bool isOppLatest)
{
    builtInOppLatestFlag_ = isOppLatest;
}

void BinaryManager::ParseAllBinaryInfoConfigPath()
{
    for (std::pair<uint64_t, string> opConfigPath : binaryConfigPathMap_) {
        BinaryInfoBasePtr binaryInfo = nullptr;
        TE_FUSION_MAKE_SHARED(binaryInfo = std::make_shared<BinaryInfoBase>(), return);
        if (!binaryInfo->ParseBinaryInfoFile(opConfigPath.second, false)) {
            TE_WARNLOG("Can not ParseBinaryInfoFile file_path is [%s].", opConfigPath.second.c_str());
            binaryInfo.reset();
            binaryInfo = nullptr;
            continue;
        }
        if (opConfigPath.second.find(OPP_LATEST_PATH) != std::string::npos) {
            binaryInfo->SetOppBinFlag(true);
        }
        binaryInfoPtrMap_.emplace(opConfigPath.first, binaryInfo);
    }
    // super kernel binary
    for (std::pair<uint64_t, string> opConfigPath : relocatableBinaryConfigPathMap_) {
        BinaryInfoBasePtr relocatableBinaryInfo = nullptr;
        TE_FUSION_MAKE_SHARED(relocatableBinaryInfo = std::make_shared<BinaryInfoBase>(), return);
        if (!relocatableBinaryInfo->ParseBinaryInfoFile(opConfigPath.second, true)) {
            TE_WARNLOG("Can not ParseBinaryInfoFile file_path is [%s].", opConfigPath.second.c_str());
            relocatableBinaryInfo.reset();
            relocatableBinaryInfo = nullptr;
            continue;
        }
        if (opConfigPath.second.find(OPP_LATEST_PATH) != std::string::npos) {
            relocatableBinaryInfo->SetOppBinFlag(true);
        }
        relocatableBinaryInfoPtrMap_.emplace(opConfigPath.first, relocatableBinaryInfo);
    }
}

bool BinaryManager::JudgeBinKernelInstalled(bool isOm, int64_t implType) const
{
    if (isOm) {
        auto iter_0 = binOmAdkVersionMap_.find(implType);
        auto iter_1 = binOmVersionMap_.find(implType);
        return !(iter_0 == binOmAdkVersionMap_.end() || iter_1 == binOmVersionMap_.end());
    }
    auto iter_0 = binAdkVersionMap_.find(implType);
    auto iter_1 = binOppVersionMap_.find(implType);
    return !(iter_0 == binAdkVersionMap_.end() || iter_1 == binOppVersionMap_.end());
}

void BinaryManager::GetAllBinaryVersionInfo(bool isOm)
{
    std::map<int64_t, std::string> &pathMap = isOm ? binaryOmModelPath_ : binaryOppKernelPath_;
    std::map<int64_t, std::string>::const_iterator it = pathMap.cbegin();
    for (; it != pathMap.cend(); ++it) {
        if (it->second.empty()) {
            continue;
        }
        std::string binAdkVersion;
        std::string binOppVersion;
        if (!GetBinaryVersionInfo(it->second, binAdkVersion, binOppVersion)) {
            continue;
        }
        if (isOm) {
            binOmAdkVersionMap_.emplace(it->first, binAdkVersion);
            binOmVersionMap_.emplace(it->first, binOppVersion);
        } else {
            binAdkVersionMap_.emplace(it->first, binAdkVersion);
            binOppVersionMap_.emplace(it->first, binOppVersion);
        }
        TE_DBGLOGF("Add impl: %ld. (adkversion = %s, oppversion = %s)", it->first,
                   binAdkVersion.c_str(), binOppVersion.c_str());
    }
    return;
}

bool BinaryManager::GetBinaryVersionInfo(const std::string &verFilePath, std::string &binAdkVersion,
                                           std::string &binOppVersion)
{
    TE_DBGLOG("Begin to get opp-kernel version. (versionFilePath=%s).", verFilePath.c_str());
    std::string binVerFilePath = RealPath(verFilePath + BINARY_VERSION_FILE);
    if (binVerFilePath.empty()) {
        TE_DBGLOGF("Binary version file path invalid. (versionFilePath=%s).", verFilePath.c_str());
        return false;
    }

    std::ifstream ifs(binVerFilePath);
    if (!ifs.is_open()) {
        TE_WARNLOGF("Open binary version file(%s) failed, not exist or has been opened.", binVerFilePath.c_str());
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line.find('#') == 0) {
            continue;
        }
        size_t equalPos = line.find('=');
        if (equalPos == string::npos) {
            continue;
        }
        std::string key = line.substr(0, equalPos);
        Trim(key);
        std::string value = line.substr(equalPos + 1, line.length() - equalPos - 1);
        Trim(value);
        if (key == BINARY_ADK_VERSION) {
            binAdkVersion = value;
        } else if (key == BINARY_OPS_VERSION) {
            binOppVersion = value;
        }
    }
    ifs.close();

    if (binAdkVersion.empty() || binOppVersion.empty()) {
        TE_DBGLOGF("Get binary version did not succeed. (binAdkVersion = %s, binOppVersion = %s, binVerFilePath = %s).",
            binAdkVersion.c_str(), binOppVersion.c_str(), binVerFilePath.c_str());
        return false;
    }

    TE_DBGLOG("Binary adk version is [%s], bin opp version is [%s]", binAdkVersion.c_str(), binOppVersion.c_str());
    return true;
}

bool BinaryManager::GetBinaryFileName(const OpBuildTaskPtr &opTask, std::string &binaryFileRealPath,
                                        bool hasOmKeyId, bool isOm) const
{
    std::string filePath;
    GetBinaryPath(opTask, isOm, false, filePath);

    if (filePath.empty()) {
        TE_DBGLOGF("Binary file path is empty");
        return false;
    }
    std::string binaryFileName = BINARY_FUSION_OPS_JSON_FILE_NAME;
    std::string binaryFileNameByOpFile = BINARY_FUSION_OPS_JSON_FILE_NAME;
    if (!hasOmKeyId && opTask->opNodes.size() == 1) {
        auto node = opTask->opNodes[0];
        if (node == nullptr) {
            TE_WARNLOGF("Node is null.");
            return false;
        }
        ConstTbeOpInfoPtr pOpInfo = TbeOpInfoCache::Instance().GetTbeOpInfoByNode(node);
        if (pOpInfo == nullptr) {
            TE_WARNLOGF("Node %s get tbe opInfo failed.", opTask->outNode->GetName().c_str());
            return false;
        }
        (void)pOpInfo->GetOpFileName(binaryFileNameByOpFile);
        pOpInfo->GetOpType(binaryFileName);
        pOpInfo->NormalizeFuncName(binaryFileName);
    } 
    std::string opsPathNamePrefix = "";
    std::string binaryFile;
    if (opTask->opNodes.size() == 1) {
        (void)ge::AttrUtils::GetStr(opTask->opNodes[0]->GetOpDesc(), OPS_PATH_NAME_PREFIX, opsPathNamePrefix);
    } else {
        opsPathNamePrefix = "ops_legacy";
    }
    if (opsPathNamePrefix != "") {
        binaryFile = filePath + opsPathNamePrefix + "/" + binaryFileName + ".json";
    } else {
        binaryFile = filePath + binaryFileName + ".json";
    }
    binaryFileRealPath = RealPath(binaryFile);
    if (binaryFileRealPath.empty()) {
        TE_DBGLOG("Node(%s) binary file(%s) does not exist, try to find the binary file which assembly by opFile",
                  GetTaskNodeName(opTask).c_str(), binaryFile.c_str());
        if (opsPathNamePrefix != "") {
            binaryFile = filePath + opsPathNamePrefix + "/" + binaryFileNameByOpFile + ".json";
        } else {
            binaryFile = filePath + binaryFileNameByOpFile + ".json";
        }
        binaryFileRealPath = RealPath(binaryFile);
        if (binaryFileRealPath.empty()) {
            TE_DBGLOG("Node(%s) binary file(%s) does not exist", GetTaskNodeName(opTask).c_str(), binaryFile.c_str());
            return false;
        }
    }
    TE_DBGLOG("Node[%s] get binary file(%s) success",
              binaryFileRealPath.c_str(), GetTaskNodeName(opTask).c_str());
    return true;
}


bool BinaryManager::MatchFusionOpGraphPattern(const OpBuildTaskPtr &opTask, json &binListJson) const
{
    // only fusion ops need to check graph_pattern
    if (opTask->opNodes.size() > 1) {
        std::string graphPattern;
        if (!GetNodeStrAttr(opTask, BINARY_GRAPH_PATTERN_ATTR, graphPattern)) {
            return false;
        }

        return CheckMatchInBinListByKey(opTask, BINARY_GRAPH_PATTERN, graphPattern, binListJson);
    }
    return true;
}

template <typename T>
void GenOptionsNoneDefaultParamJson(const std::string &key, T &value, T &defaultValue, json &genJson)
{
    if (value != defaultValue) {
        genJson[key] = value;
    }
}

bool BinaryManager::GenBuildOptions(const OpBuildTaskPtr &opTask, json &staticKeyJson)
{
    json buildOptionsJson;
    std::string strDft = "0";
    std::string strValue = TeConfigInfo::Instance().GetL2Mode();
    GenOptionsNoneDefaultParamJson("l2_mode", strValue, strDft, buildOptionsJson);

    // ignore status_check while static key match
    buildOptionsJson[STATUS_CHECK] = TRUE;

    // get op impl mode, fusion ops impl mode is generate in graphOpParams when generalize ops
    if (opTask->opNodes.size() == 1) {
        const auto node = opTask->opNodes[0];
        if (node == nullptr) {
            TE_WARNLOGF("Node is null");
            return false;
        }
        std::string implMode;
        if (!TbeOpInfoCache::Instance().GetOpImplModeByOpNode(node, implMode)) {
            TE_DBGLOGF("Get op implMode did not succeed.");
            return false;
        }
        strDft = "";
        GenOptionsNoneDefaultParamJson(IMPL_MODE, implMode, strDft, buildOptionsJson);
    }

    if (buildOptionsJson.empty()) {
        TE_DBGLOGF("Build options are default values, no need to add to json. Build options json is empty.");
    } else {
        TE_DBGLOGF("Build options json is %s", buildOptionsJson.dump().c_str());
        staticKeyJson[BUILD_OPTIONS] = buildOptionsJson;
    }

    return true;
}

bool BinaryManager::MatchStaticKey(const OpBuildTaskPtr &opTask, json &staticKeyJson, json &binListJson) const
{
    TE_DBGLOG("Get generalized ops JSON: %s",
              staticKeyJson.dump().c_str());

    if (!GenBuildOptions(opTask, staticKeyJson)) {
        TE_DBGLOG("Node(%s) not generate build options", GetTaskNodeName(opTask).c_str());
        return false;
    }

    const std::string &staticKeyJsonStr = staticKeyJson.dump();
    std::string staticKey;
    if (!PythonApiCall::Instance().GenerateStrSha256HashValue(staticKeyJsonStr, staticKey)) {
        TE_DBGLOG("Node(%s) do not generate static key hash by json(%s).",
            GetTaskNodeName(opTask).c_str(), staticKeyJsonStr.c_str());
        return false;
    }

    TE_DBGLOGF("Generate static key is %s, json is %s", staticKey.c_str(), staticKeyJsonStr.c_str());

    return CheckMatchInBinListByKey(opTask, STATIC_KEY, staticKey, binListJson);
}

namespace {
enum DeterministicValue {
    DETERMINISTIC_IGNORE,
    DETERMINISTIC_TRUE,
    DETERMINISTIC_FALSE
};

/**
 *  读取json中的deterministic，没有时返回ignore。
 *  其他场景raise错误。
 */
bool GetDeterministic(const json& opJson, DeterministicValue& deterministic)
{
    auto iter = opJson.find(DETERMINISTIC);
    if (iter == opJson.end()) {
        deterministic = DETERMINISTIC_FALSE;
        return true;
    }

    std::string deterministicString;
    try {
        deterministicString = iter.value().get<std::string>();
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to get deterministic value from json(%s).", e.what());
        return false;
    }

    if (deterministicString == "true") {
        deterministic = DETERMINISTIC_TRUE;
    } else if (deterministicString == "false") {
        deterministic = DETERMINISTIC_FALSE;
    } else if (deterministicString == "ignore") {
        deterministic = DETERMINISTIC_IGNORE;
    } else {
        TE_WARNLOG("Deterministic string value '%s' is invalid.", deterministicString.c_str());
        return false;
    }

    return true;
}
}

bool BinaryManager::MatchDeterministic(const json &opParamsJson, const json &binJson) const
{
    bool matchResult = false;

    DeterministicValue paramDeterministic, binDeterministic;
    if (!GetDeterministic(opParamsJson, paramDeterministic)) {
        TE_WARNLOG("Failed to get deterministic in operator params.");
        return false;
    }

    if (!GetDeterministic(binJson, binDeterministic)) {
        TE_WARNLOG("Failed to get deterministic in binary.");
        return false;
    }

    if (paramDeterministic == DETERMINISTIC_TRUE) {
        if (binDeterministic == DETERMINISTIC_TRUE || binDeterministic == DETERMINISTIC_IGNORE) {
            matchResult = true;
        }
    } else if (paramDeterministic == DETERMINISTIC_FALSE) {
        if (binDeterministic == DETERMINISTIC_FALSE || binDeterministic == DETERMINISTIC_IGNORE) {
            matchResult = true;
        }
    } else if (paramDeterministic == DETERMINISTIC_IGNORE) {
        matchResult = true;
    }

    static const string deterministicString[] = {
        [DETERMINISTIC_IGNORE] = "ignore",
        [DETERMINISTIC_TRUE] = "deterministic",
        [DETERMINISTIC_FALSE] = "nondeterministic"
    };
    TE_DBGLOGF("Deterministic match params:%s, bin:%s, result is %d",
               deterministicString[paramDeterministic].c_str(),
               deterministicString[binDeterministic].c_str(),
               matchResult);
    return matchResult;
}

bool BinaryManager::MatchRangeOriRange(const std::string &key, const json &opValue, const json &binValue)
{
    TE_DBGLOG("Match %s params", key.c_str());
    std::vector<std::pair<int64_t, int64_t>> binRange;
    auto iter = binValue.find(key);
    if (iter != binValue.end()) {
        try {
            binRange = iter.value().get<std::vector<std::pair<int64_t, int64_t>>>();
        } catch (const std::exception &e) {
            TE_WARNLOG("Failed to get bin range value. reason is %s", e.what());
            return false;
        }
    } else {
        TE_DBGLOGF("Binary has no %s. Supported all", key.c_str());
        return true;
    }

    std::vector<std::pair<int64_t, int64_t>> opRange;
    iter = opValue.find(key);
    if (iter != opValue.end()) {
        try {
            opRange = iter.value().get<std::vector<std::pair<int64_t, int64_t>>>();
        } catch (const std::exception &e) {
            TE_WARNLOG("Failed to get op range value. reason is %s", e.what());
            return false;
        }
    } else {
        TE_DBGLOGF("Binary has %s, but node does not have. Not supported.", key.c_str());
        return false;
    }

    ChangeShapeRangeExpressFromUnlimitedToMinMAx(opRange);
    ChangeShapeRangeExpressFromUnlimitedToMinMAx(binRange);

    for (size_t i = 0; i < opRange.size(); ++i) {
        bool isMatch = false;
        for (size_t j = 0; j < binRange.size(); ++j) {
            if (opRange[i].first < binRange[j].first || opRange[i].second > binRange[j].second) {
                continue;
            }
            isMatch = true;
            break;
        }
        if (!isMatch) {
            TE_DBGLOGF("Match %s params range did not succeed", key.c_str());
            return false;
        }
    }
    TE_DBGLOGF("Match %s params successfully", key.c_str());
    return true;
}

void BinaryManager::GetConstValue(const std::string &key, std::string &dtype, json &valueJson, const json &constJson)
{
    auto iter = constJson.find(key);
    if (iter == constJson.end()) {
        return;
    }
    const json &constValue = iter.value();

    iter = constValue.find(DTYPE);
    if (iter != constValue.end()) {
        try {
            dtype = iter.value().get<std::string>();
        } catch (const std::exception &e) {
            TE_WARNLOG("Failed to retrieve dtype value: reason is %s", e.what());
            return;
        }
    } else {
        TE_DBGLOGF("Get const value dtype did not succeed.");
    }
    iter = constValue.find(key);
    if (iter != constValue.end()) {
        valueJson = iter.value();
    } else {
        TE_DBGLOGF("Get const value value did not succeed.");
    }
}

bool BinaryManager::MatchConstValue(const json &binValueJson)
{
    TE_DBGLOG("Match const value param");

    json binValue;
    json binValueRange;
    std::string binValueDdype;
    std::string binValueRangeDdype;
    GetConstValue(CONST_VALUE, binValueDdype, binValue, binValueJson);
    GetConstValue(CONST_VALUE_RANGE, binValueRangeDdype, binValueRange, binValueJson);
    if (binValue.size() == 0 && binValueRange.size() == 0) {
        TE_DBGLOG("Binary JSON does not have a constant value. Supports all constants. Successfully matched the constant value parameter.");
        return true;
    }
    TE_WARNLOG("Binary json has const value. Not support match yet.");
    return false;
}

template <typename T>
bool MatchParamFromJson(const std::string &key, T &dftValue,
                        const json &binValueJson, const json &opValueJson)
{
    auto iter = binValueJson.find(key);
    if (iter == binValueJson.end()) {
        // not include in binary, means insensitive to the param, can reuse
        return true;
    }

    // only need to compare explicit set params in binary
    try {
        T binValue = iter.value().get<T>();
        if (binValue == dftValue) {
            return true;
        }

        iter = opValueJson.find(key);
        if (iter == opValueJson.end()) {
            return false;
        }
        T opValue = iter.value().get<T>();

        return binValue == opValue;
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve dtype value: reason is %s", e.what());
        return false;
    }
}

template <typename T>
bool MatchAttrParamFromJson(const std::string &key, T &dftValue,
                            const json &binValueJson, const json &opValueJson)
{
    T binValue = dftValue;
    T opValue = dftValue;
    try {
        auto iter = binValueJson.find(key);
        if (iter != binValueJson.end()) {
            binValue = iter.value().get<T>();
        }

        iter = opValueJson.find(key);
        if (iter != opValueJson.end()) {
            opValue = iter.value().get<T>();
        }

        return binValue == opValue;
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve dtype value: reason is %s", e.what());
        return false;
    }
}

template <typename T>
bool MatchParamFromJsonWithTwoDft(const std::string &key, T &dftValue, T &dftValue1,
                                  const json &binValueJson, const json &opValueJson)
{
    auto iter = binValueJson.find(key);
    if (iter == binValueJson.end()) {
        // not include in binary, means insensitive to the param, can reuse
        return true;
    }

    // only need to compare explicit set params in binary
    try {
        T binValue = iter.value().get<T>();
        // dftValue is {}, dftValue1 is {-2}, all means support all
        if (binValue == dftValue || binValue == dftValue1) {
            return true;
        }

        iter = opValueJson.find(key);
        if (iter == opValueJson.end()) {
            return false;
        }

        T opValue = iter.value().get<T>();
        if (opValue == dftValue || opValue == dftValue1) {
            return false;
        }

        if (binValue.size() != opValue.size()) {
            return false;
        }

        // compare every dim between bin ori shape[binValue] and request ori shape[opValue]
        for (size_t i = 0; i < binValue.size(); i++) {
            if (binValue[i] == opValue[i]) {
                continue;
            }
            if (binValue[i] != -1) {
                return false;
            }
        }

        return true;
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve dtype value with two defaults. Reason: %s", e.what());
        return false;
    }
}

bool BinaryManager::MatchL1Params(const json &binValue, const json &opValue)
{
    int64_t defaultInt64 = 0;
    if (!MatchAttrParamFromJson(L1_ADDR_OFFSET, defaultInt64, binValue, opValue)) {
        TE_DBGLOGF("Match L1_addr_offset param did not succeed.");
        return false;
    }

    size_t defaultSizet = 0;
    if (!MatchAttrParamFromJson(USE_L1_WORKSPACE, defaultSizet, binValue, opValue)) {
        TE_DBGLOGF("Match use_l1_workspace param did not succeed.");
        return false;
    }

    int64_t l1AddrFlagDft = -1;
    if (!MatchAttrParamFromJson(L1_ADDR_FLAG, l1AddrFlagDft, binValue, opValue)) {
        TE_DBGLOGF("Match l1_addr_flag param did not succeed.");
        return false;
    }

    int32_t l1FusionTypeDft = -1;
    if (!MatchAttrParamFromJson(L1_FUSION_TYPE, l1FusionTypeDft, binValue, opValue)) {
        TE_DBGLOGF("Match l1_fusion_type param did not succeed.");
        return false;
    }

    int64_t l1WorkspaceSizeDft = -1;
    if (!MatchAttrParamFromJson(L1_WORKSPACE_SIZE, l1WorkspaceSizeDft, binValue, opValue)) {
        TE_DBGLOGF("Match l1_workspace_size param did not succeed.");
        return false;
    }

    int64_t l1ValidSizeDft = -1;
    if (!MatchAttrParamFromJson(L1_VALID_SIZE, l1ValidSizeDft, binValue, opValue)) {
        TE_DBGLOGF("Match l1_valid_size param did not succeed.");
        return false;
    }

    return true;
}

bool BinaryManager::MatchNonRangeParams(const json &binValue, const json &opValue)
{
    // get ori_format, ori_shape, addr_type, use_L1_workspace const_value and other none default params
    std::string defaultStr = "";
    if (!MatchParamFromJson(ORI_FORMAT, defaultStr, binValue, opValue)) {
        TE_DBGLOGF("Match ori_format param did not succeed.");
        return false;
    }

    // ori_shape -2 or has no, support all
    std::vector<int64_t> defaultVectorInt64;
    std::vector<int64_t> defaultDynamicShape;
    defaultDynamicShape.emplace_back(DYNAMIC_SHAPE_DIM);
    if (!MatchParamFromJsonWithTwoDft(ORI_SHAPE, defaultVectorInt64, defaultDynamicShape, binValue, opValue)) {
        TE_DBGLOGF("Match ori_shape param did not succeed.");
        return false;
    }

    size_t defaultSizet = 0;
    if (!MatchAttrParamFromJson(ADDR_TYPE, defaultSizet, binValue, opValue)) {
        TE_DBGLOGF("Match addr_type param did not succeed.");
        return false;
    }

    uint32_t defaultUint32 = 0;
    if (!MatchAttrParamFromJson(SPLIT_INDEX, defaultUint32, binValue, opValue)) {
        TE_DBGLOGF("Match split_index param did not succeed.");
        return false;
    }

    bool isFirstLayerDft = false;
    if (!MatchAttrParamFromJson(IS_FIRST_LAYER, isFirstLayerDft, binValue, opValue)) {
        TE_DBGLOGF("Match is_first_layer param did not succeed.");
        return false;
    }

    if (!MatchAttrParamFromJson(SLICE_OFFSET, defaultVectorInt64, binValue, opValue)) {
        TE_DBGLOGF("Match slice_offset param did not succeed.");
        return false;
    }

    if (!MatchAttrParamFromJson(VALID_SHAPE, defaultVectorInt64, binValue, opValue)) {
        TE_DBGLOGF("Match valid_shape param did not succeed.");
        return false;
    }

    if (!MatchL1Params(binValue, opValue)) {
        return false;
    }
    return true;
}

bool BinaryManager::MatchSingleInputOutput(const json &opValue, const json &binValue,
                                             const std::vector<uint32_t> &optionalInputIdx,
                                             const uint32_t &idx)
{
    // check range
    if (!MatchRangeOriRange(RANGE, opValue, binValue)) {
        return false;
    }
    // check ori_range
    if (!MatchRangeOriRange(ORI_RANGE, opValue, binValue)) {
        return false;
    }
    // check const_value
    if (!MatchConstValue(binValue)) {
        return false;
    }

    /* if params not in binValue, no need to compare, support reuse
        * if params in binValue, need compare, reuse only completely same
    */
    TE_DBGLOGF("Match bin and op non-range params");
    if (!MatchNonRangeParams(binValue, opValue)) {
        return false;
    }
    TE_DBGLOGF("Match bin and op non-range params successfully");
    if (std::find(optionalInputIdx.begin(), optionalInputIdx.end(), idx) != optionalInputIdx.end()) {
        std::string defaultStr = "";
        if (!MatchParamFromJson(DTYPE, defaultStr, binValue, opValue)) {
            TE_DBGLOGF("Match dtype param did not succeed.");
            return false;
        }
        if (!MatchParamFromJson(FORMAT, defaultStr, binValue, opValue)) {
            TE_DBGLOGF("Match format param did not succeed.");
            return false;
        }
        std::vector<int64_t> defaultVectorInt64;
        if (!MatchParamFromJson(SHAPE, defaultVectorInt64, binValue, opValue)) {
            TE_DBGLOGF("Match shape param did not succeed.");
            return false;
        }
    }
    return true;
}

bool BinaryManager::MatchInputsOutputs(const std::string &key,
                                         const json &dynamicJson,
                                         const json &binJson,
                                         const std::vector<uint32_t> &optionalInputIdx)
{
    TE_DBGLOG("Match %s params", key.c_str());

    const auto opKeyIter = dynamicJson.find(key);
    const auto binKeyIter = binJson.find(key);
    if (opKeyIter == dynamicJson.end() && binKeyIter == binJson.end()) {
        TE_DBGLOGF("Both json has no %s.", key.c_str());
        return true;
    }
    if (opKeyIter == dynamicJson.end() || binKeyIter == binJson.end()) {
        TE_DBGLOGF("One json has no %s. Match did not succeed.", key.c_str());
        return false;
    }

    const json &opKeyJson = opKeyIter.value();
    const json &binKeyJson = binKeyIter.value();
    if (!opKeyJson.is_array() || !binKeyJson.is_array()) {
        TE_DBGLOGF("Format error! '%s' should be array. Match did not succeed.", key.c_str());
        return false;
    }

    if (opKeyJson.size() != binKeyJson.size()) {
        TE_DBGLOGF("Op %s size(%zu) is not same as binary size(%zu). Match did not succeed.", key.c_str(), opKeyJson.size(),
            binKeyJson.size());
        return false;
    }

    TE_DBGLOGF("Op %s is %s, bin is %s.", key.c_str(), opKeyJson.dump().c_str(), binKeyJson.dump().c_str());

    uint32_t opIndex = 0;
    for (auto opIter = opKeyJson.begin(), binIter = binKeyJson.begin();
        opIter != opKeyJson.end() && binIter != binKeyJson.end(); ++opIter, ++binIter) {
        const json &opValue = *opIter;
        const json &binValue = *binIter;
        if (opValue.is_null() && \
            (std::find(optionalInputIdx.begin(),  optionalInputIdx.end(), opIndex) != optionalInputIdx.end())) {
            continue;
        }
        if (!opValue.is_null() && binValue.is_null()) {
            TE_DBGLOGF("Op %s bin value should not be None! Match did not succeed.", key.c_str());
            return false;
        }
        if (!MatchSingleInputOutput(opValue, binValue, optionalInputIdx, opIndex)) {
            return false;
        }
        opIndex++;
    }

    TE_DBGLOG("Matched %s params successfully", key.c_str());
    return true;
}

template <typename T> bool CompareAttrValueFloat(const json &opJson, const json &binJson)
{
    try {
        T binValue = binJson.get<T>();
        T opValue = opJson.get<T>();
        return std::fabs(opValue - binValue) < FLOAT_ESP;
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve attribute value. Reason: %s", e.what());
        return false;
    }
}

template <typename T> bool CompareAttrValue(const json &opJson, const json &binJson)
{
    try {
        T opValue = opJson.get<T>();
        T binValue = binJson.get<T>();
        return opValue == binValue;
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve attribute value. Reason: %s", e.what());
        return false;
    }
}

template <typename T> bool CompareAttrValueInt(const json &opJson, const json &binJson)
{
    try {
        T binValue = binJson.get<T>();
        if (static_cast<int>(binValue) == -1) {
            TE_DBGLOGF("binValue is -1, support all.");
            return true;
        }
        T opValue = opJson.get<T>();
        return opValue == binValue;
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve attribute value. Reason: %s", e.what());
        return false;
    }
}

/*
 "dtype": "list_int32",
 "value": null

 "dtype": "list_int32",
 "value": [1,
           2]
*/
template <typename T> bool CompareAttrValueListInt(const json &opJson, const json &binJson)
{
    T binValue;
    T opValue;
    try {
        binValue = binJson.get<T>();
        opValue = opJson.get<T>();
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve the list_int attribute value. Reason: %s", e.what());
        return false;
    }
    if (opValue.size() != binValue.size()) {
        return false;
    }

    for (size_t i = 0; i < opValue.size(); ++i) {
        if (binValue[i] == -1) {
            continue;
        }
        if (opValue[i] != binValue[i]) {
            return false;
        }
    }
    return true;
}

template <typename T> bool CompareAttrValueListFloat(const json &opJson, const json &binJson)
{
    T binValue;
    T opValue;
    try {
        binValue = binJson.get<T>();
        opValue = opJson.get<T>();
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve the list_float attribute value. Reason: %s", e.what());
        return false;
    }
    if (opValue.size() != binValue.size()) {
        return false;
    }

    for (size_t i = 0; i < opValue.size(); ++i) {
        if (std::fabs(binValue[i] - opValue[i]) >= FLOAT_ESP) {
            return false;
        }
    }
    return true;
}

template <typename T> bool CompareAttrValueListBoolOrStr(const json &opJson, const json &binJson)
{
    T binValue;
    T opValue;
    try {
        binValue = binJson.get<T>();
        opValue = opJson.get<T>();
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to retrieve attribute value. Reason: %s", e.what());
        return false;
    }

    if (opValue.size() != binValue.size()) {
        return false;
    }
    for (size_t i = 0; i < opValue.size(); ++i) {
        if (opValue[i] != binValue[i]) {
            return false;
        }
    }

    return true;
}

/*
 "dtype": "list_list_int64",
 "value": null

 "dtype": "list_list_int64",
 "value": [1,
           2],
          [4,
           5]
*/
template <typename T> bool CompareAttrValueListList(const json &opJson, const json &binJson)
{
    T binValue;
    T opValue;
    try {
        binValue = binJson.get<T>();
        opValue = opJson.get<T>();
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to get list attribute value. Reason is: %s", e.what());
        return false;
    }

    if (opValue.size() != binValue.size()) {
        return false;
    }

    for (size_t i = 0; i < opValue.size(); ++i) {
        if (opValue[i].size() != binValue[i].size()) {
            return false;
        }
        for (size_t j = 0; j < opValue[i].size(); ++j) {
            if (binValue[i][j] == -1) {
                continue;
            }
            if (opValue[i][j] != binValue[i][j]) {
                return false;
            }
        }
    }
    return true;
}

std::map<std::string, std::function<bool(const json &, const json &)>> g_attrValueCompFuncs = {
    {"data_type",     CompareAttrValueInt<int32_t>},
    {"int",           CompareAttrValueInt<int64_t>},
    {"float",         CompareAttrValueFloat<float>},
    {"bool",          CompareAttrValue<bool>},
    {"string",        CompareAttrValue<string>},
    {"list_int",      CompareAttrValueListInt<vector<int64_t>>},
    {"list_float",    CompareAttrValueListFloat<vector<float>>},
    {"list_bool",     CompareAttrValueListBoolOrStr<vector<bool>>},
    {"list_string",   CompareAttrValueListBoolOrStr<vector<string>>},
    {"list_list_int", CompareAttrValueListList<vector<vector<int64_t>>>}
};

bool BinaryManager::CompareSingleAttr(const std::string &dtype,
                                        const json &opAttr,
                                        const json &binAttr)
{
    const std::map<std::string, std::function<bool(const json &, const json &)>>::const_iterator func =
        g_attrValueCompFuncs.find(dtype);
    if (func != g_attrValueCompFuncs.cend()) {
        return (func->second)(opAttr, binAttr);
    }

    TE_DBGLOGF("Wrong dtype(%s).", dtype.c_str());
    return false;
}

bool BinaryManager::MatchSingleAttr(const json &opAttr, const json &binAttr)
{
    const auto opIter = opAttr.find(DTYPE);
    if (opIter == opAttr.end()) {
        TE_DBGLOGF("Get op attr dtype did not succeed.");
        return false;
    }
    const auto &binIter = binAttr.find(DTYPE);
    if (binIter == binAttr.end()) {
        TE_DBGLOGF("Get bin attr dtype did not succeed.");
        return false;
    }

    std::string opDtype;
    std::string binDtype;
    try {
        opDtype = opIter.value().get<std::string>();
        binDtype = binIter.value().get<std::string>();
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to get attribute dtype value. Reason: %s", e.what());
        return false;
    }

    TE_DBGLOGF("Get op attr dtype %s, bin attr dtype %s.",
               opDtype.c_str(), binDtype.c_str());
    if (opDtype != binDtype) {
        TE_DBGLOGF("Match op and bin attr dtype did not succeed.");
        return false;
    }

    if (opDtype != FLOAT && opDtype != ATTR_LIST_FLOAT) {
        return true;
    }

    auto iter = binAttr.find(VALUE);
    if (iter == binAttr.end() || iter.value().is_null()) {
        TE_DBGLOGF("Bin attr has no value or value is null, support all.");
        return true;
    }
    const json &binValue = iter.value();
    iter = opAttr.find(VALUE);
    if (iter == opAttr.end() || iter.value().is_null()) {
        // attr vaule may be erased by generalized parse, it means support all
        TE_DBGLOGF("Op attr has no value or value is null, support all. But bin not support all.");
        return false;
    }
    const json &opValue = iter.value();

    return CompareSingleAttr(opDtype, opValue, binValue);
}

bool BinaryManager::MatchFusionOpsAttrs(const json &opParamsJson, const json &binJson)
{
    TE_DBGLOGF("Match fusion ops attrs.");

    const auto opKeyIter = opParamsJson.find(GRAPH_OP_PARAMS);
    const auto binKeyIter = binJson.find(GRAPH_OP_PARAMS);
    if (opKeyIter == opParamsJson.end() && binKeyIter == binJson.end()) {
        TE_DBGLOGF("Both json has no graph_op_params. Match did not succeed.");
        return true;
    }
    if (opKeyIter == opParamsJson.end() || binKeyIter == binJson.end()) {
        TE_DBGLOGF("One json has no graph_op_params. Match did not succeed.");
        return false;
    }

    const json &opGraphOpParams = opKeyIter.value();
    const json &binGraphOpParams = binKeyIter.value();
    if (!opGraphOpParams.is_array() || !binGraphOpParams.is_array()) {
        TE_DBGLOGF("Format error! 'graph_op_params' should be array. Match failed.");
        return false;
    }

    if (opGraphOpParams.size() != binGraphOpParams.size()) {
        TE_DBGLOGF("Op GraphOpParams size(%zu) is not same as binary size(%zu). Match did not succeed.", opGraphOpParams.size(),
            binGraphOpParams.size());
        return false;
    }

    for (auto opIter = opGraphOpParams.cbegin(), binIter = binGraphOpParams.cbegin();
        opIter != opGraphOpParams.cend() && binIter != binGraphOpParams.cend(); ++opIter, ++binIter) {
        const json &opValue = *opIter;
        const json &binValue = *binIter;

        if (!MatchSingleOpAttrs(opValue, binValue)) {
            TE_DBGLOGF("Match fusion ops attrs did not succeed.");
            return false;
        }
    }

    TE_DBGLOGF("Match fusion ops attrs successfully");
    return true;
}

bool BinaryManager::MatchAttrs(const OpBuildTaskPtr &opTask,
                                 const json &opParamsJson,
                                 const json &binJson)
{
    TE_DBGLOG("Match attrs");
    bool res;
    if (opTask->opNodes.size() == 1) {
        res = MatchSingleOpAttrs(opParamsJson, binJson);
    } else {
        res = MatchFusionOpsAttrs(opParamsJson, binJson);
    }

    if (!res) {
        TE_DBGLOGF("Match attrs did not succeed.");
        return false;
    }

    TE_DBGLOG("Match attrs successfully.");
    return true;
}

bool BinaryManager::MatchOpParams(const OpBuildTaskPtr &opTask,
                                    json &binListJson,
                                    GeneralizedResult &generalizedResult) const
{
    TE_DBGLOGF("Start to match op params. (dynamicJson = %s, binJson = %s)",
        generalizedResult.dynamicJson.dump().c_str(), binListJson.dump().c_str());
    for (auto iter = binListJson.begin(); iter != binListJson.end(); ++iter) {
        const json binJson = *iter;
        if (!MatchDeterministic(generalizedResult.dynamicJson, binJson)) {
            TE_DBGLOG("Node(%s) not match deterministic mode.", GetTaskNodeName(opTask).c_str());
            continue;
        }

        if (!MatchInputsOutputs(INPUTS, generalizedResult.dynamicJson, binJson, generalizedResult.optionalInputIdx)) {
            TE_DBGLOG("Node(%s) not match inputs params", GetTaskNodeName(opTask).c_str());
            continue;
        }

        // current not support optional output, in order to share functions, define an empty vector
        std::vector<uint32_t> optionalOutputIdx;
        if (!MatchInputsOutputs(OUTPUTS, generalizedResult.dynamicJson, binJson, optionalOutputIdx)) {
            TE_DBGLOG("Node(%s) not match outputs params", GetTaskNodeName(opTask).c_str());
            continue;
        }

        // Success is returned only if the last match is found
        if (MatchAttrs(opTask, generalizedResult.dynamicJson, binJson)) {
            binListJson.clear();
            binListJson = binJson;
            TE_DBGLOGF("Node(%s) match op params successfully with json(%s).",
                GetTaskNodeName(opTask).c_str(), binListJson.dump().c_str());
            return true;
        } else {
            TE_DBGLOG("Node(%s) match attrs did not succeed.", GetTaskNodeName(opTask).c_str());
            continue;
        }
    }

    TE_DBGLOG("Node(%s) match op params did not succeed.", GetTaskNodeName(opTask).c_str());
    return false;
}

bool BinaryManager::SetBinaryReuseResult(const OpBuildTaskPtr &opTask, const std::string &jsonFilePath)
{
    if (!GetHighPerformaceAndCompileInfoFromJson(opTask, jsonFilePath)) {
        TE_DBGLOG("Get node(%s) binary json file path no compile info.", GetTaskNodeName(opTask).c_str());
        opTask->opRes = nullptr;
        opTask->maxKernelId = 0;
        opTask->isHighPerformaceRes = false;
        return false;
    }

    if (opTask->opRes != nullptr) {
        opTask->opRes->statusCode = 0;
        // save binary json file path
        TE_DBGLOG("Save binary json file path(%s) to task result", jsonFilePath.c_str());
        opTask->opRes->jsonFilePath = jsonFilePath;
        opTask->opRes->compileRetType = CompileResultType::Binary;
    }

    std::vector<Node *> &teGraphNode = opTask->opNodes;
    Node *pNode = teGraphNode[0];
    TE_FUSION_CHECK(pNode == nullptr, {
        opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;
        TE_INFOLOG("Get node not successfully: can't compile cce, continue to reuse binary.");
        return true;
    });

    TbeOpInfoPtr pOpInfo;
    TE_FUSION_CHECK(!TbeOpInfoCache::Instance().GetTbeOpInfoByNode(pNode, pOpInfo), {
        opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;
        TE_INFOLOG("Get opinfo of node %s not successfully: can't compile cce, continue to reuse binary.",
            pNode->GetName().c_str());
        return true;
    });

    if (CheckDebugMode(*pOpInfo)) {
        opTask->status = OP_TASK_STATUS::OP_TASK_PENDING;
        TE_DBGLOG("Debug mode check result is need to compile cce. Need to compile");
    } else {
        opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;
        TE_DBGLOG("The node(%s) binary json file(%s) can be reused", pNode->GetName().c_str(), jsonFilePath.c_str());
    }
    return true;
}

bool BinaryManager::MatchStaticKeyWithOptionalInputNull(const OpBuildTaskPtr &opTask,
                                                          json &binListJson,
                                                          GeneralizedResult &generalizedResult) const
{
    auto inputs = generalizedResult.staticJson.find(INPUTS);
    if (inputs == generalizedResult.staticJson.end()) {
        TE_DBGLOG("Node(%s) has no inputs info in static key json.", GetTaskNodeName(opTask).c_str());
        return false;
    }

    json &inputsJson = *inputs;
    for (size_t i = 0; i < inputsJson.size(); ++i) {
        if (std::find(generalizedResult.optionalInputIdx.begin(), generalizedResult.optionalInputIdx.end(),
            static_cast<uint32_t>(i)) != generalizedResult.optionalInputIdx.end()) {
            json jsonNull;
            inputsJson[i] = jsonNull;
        }
    }
    const std::string &staticKeyJsonStr = generalizedResult.staticJson.dump();
    std::string staticKey;
    if (!PythonApiCall::Instance().GenerateStrSha256HashValue(staticKeyJsonStr, staticKey)) {
        TE_DBGLOG("Node(%s) do not generate static key hash by json(%s).",
            GetTaskNodeName(opTask).c_str(), staticKeyJsonStr.c_str());
        return false;
    }

    TE_DBGLOG("Generate static key is %s, json is %s",
              staticKey.c_str(), staticKeyJsonStr.c_str());
    if (!CheckMatchInBinListByKey(opTask, STATIC_KEY, staticKey, binListJson)) {
        return false;
    }
    for (const auto &supportInfo : binListJson) {
        auto iter = supportInfo.find(OPT_INPUT_MODE);
        if (iter == supportInfo.end()) {
            continue;
        }
        const std::string &optInputMode = iter.value().get<std::string>();
        if (optInputMode == GEN_PLACEHOLDER) {
            TE_DBGLOG("Node (%s) matched staticKey [%s] while optionalInputMode of bin file is %s",
                GetTaskNodeName(opTask).c_str(), staticKey.c_str(), optInputMode.c_str());
            return true;
        }
    }
    TE_DBGLOG("Node (%s) cannot match staticKey [%s], optionalInputMode of bin file is not gen_placeholder",
              GetTaskNodeName(opTask).c_str(), staticKey.c_str());
    return false;
}

bool BinaryManager::BinaryMatchWithStaticKeyAndDynInfo(const OpBuildTaskPtr &opTask,
                                                         GeneralizedResult &generalizedResult, json &binListJson) const
{
    const std::string opName = GetTaskNodeName(opTask);
    json binListJsonTmp = binListJson;
    if (!MatchStaticKey(opTask, generalizedResult.staticJson, binListJson)) {
        // miss match full inputs, try match again with optional input placeholder
        const json &tmp = generalizedResult.optionalInputIdx;
        TE_DBGLOGF("Node(%s) optiona input is %s.", opName.c_str(), tmp.dump().c_str());
        if (!generalizedResult.optionalInputIdx.empty()) {
            if (!MatchStaticKeyWithOptionalInputNull(opTask, binListJsonTmp, generalizedResult)) {
                TE_DBGLOGF("Node(%s) static key(%s) with optional input null not match binJson(%s). Need to compile.",
                    opName.c_str(), generalizedResult.staticJson.dump().c_str(), binListJsonTmp.dump().c_str());
                return false;
            }
            binListJson = binListJsonTmp;
        } else {
            TE_DBGLOGF("Node(%s) static key(%s) not match binJson(%s). Need to compile.",
                opName.c_str(), generalizedResult.staticJson.dump().c_str(), binListJson.dump().c_str());
            return false;
        }
    }

    if (!MatchOpParams(opTask, binListJson, generalizedResult)) {
        TE_DBGLOGF("Node(%s) not match dynamic info(%s) binJson(%s). Need to compile",
            opName.c_str(), generalizedResult.dynamicJson.dump().c_str(), binListJson.dump().c_str());
        return false;
    }
    return true;
}

bool BinaryManager::MatchSimplifiedKey(const OpBuildTaskPtr &opTask,
                                       string &jsonFilePath) const
{
    auto opNode = opTask->opNodes[0];
    TbeOpInfoPtr opInfo = TbeOpInfoCache::Instance().MutableTbeOpInfoByNode(opNode);
    if (opInfo == nullptr) {
        TE_ERRLOG("Node[%s] get tbe op information by node failed!", opNode->GetName().c_str());
        return false;
    }

    int64_t implType = -1;
    (void)ge::AttrUtils::GetInt(opNode->GetOpDesc(), FE_IMPLY_TYPE, implType);
    int skpSubId = -1;
    bool isSuperKernel = ge::AttrUtils::GetInt(opNode->GetOpDesc(), ASCENDC_SPK_SUB_ID, skpSubId) && (skpSubId != -1);
    TE_INFOLOG("Node[%s] super_kernel_sub_flag is %d", opNode->GetName().c_str(), isSuperKernel);
    auto iter = binaryInfoPtrMap_.find(implType);
    if (isSuperKernel) {
        iter = relocatableBinaryInfoPtrMap_.find(implType);
        if (iter == relocatableBinaryInfoPtrMap_.end()) {
            TE_WARNLOG("Node[%s] binaryInfo is null, whose implye_type is %lu.", opNode->GetName().c_str(), implType);
            return false;
        }
    } else {
        if (iter == binaryInfoPtrMap_.end()) {
            TE_WARNLOG("Node[%s] binaryInfo is null, whose implye_type is %lu.", opNode->GetName().c_str(), implType);
            return false;
        }
    }

    const std::string &opType = opInfo->GetOpType();
    std::string deterministic = TeContextUtils::GetDeterministic();
    const std::vector<TbeOpParam> &inputs = opInfo->GetInputs();
    const std::vector<TbeOpParam> &outputs = opInfo->GetOutputs();
    const std::vector<TbeAttrValue> &attrValues = opInfo->GetAttrValues();
    const std::string &opImplMode = opInfo->GetOpImplMode();
    bool isUnknowShape = opInfo->GetIsUnknownShape();
    if (CheckMemoryL1MemoryL2(opType, inputs) || CheckMemoryL1MemoryL2(opType, outputs)) {
        TE_INFOLOG("Node[%s], opType[%s] has memoryL1 or MemoryL2, Not support simpleKey binary.",
            opNode->GetName().c_str(), opType.c_str());
        return false;
    }
    SimpleKeyModeType simpleKeyMode;
    BinaryInfoBasePtr binaryInfoPtr = iter->second;
    if (!binaryInfoPtr->GetSimpleKeyMode(opType, simpleKeyMode)) {
        TE_WARNLOG("Node[%s], opType[%s] get simpleKeyMode failed.", opNode->GetName().c_str(), opType.c_str());
        return false;
    }
    TE_DBGLOG("opType = [%s], simpleKeyMode = [%d], opImplMode = [%s], deterministic = [%s]",
              opType.c_str(), simpleKeyMode, opImplMode.c_str(), deterministic.c_str());
    GenerateSimpleKey generateKey = {opType, simpleKeyMode, opImplMode, deterministic, isUnknowShape};
    generateKey.SetInputs(inputs);
    generateKey.SetOutputs(outputs);
    generateKey.SetAttrs(attrValues);
    generateKey.SetBinaryInfoPtr(binaryInfoPtr);
    generateKey.SetOpInfoPtr(opInfo);
    std::string simpleKeystr;
    if (!generateKey.GenerateSimpleKeyStr(simpleKeystr)) {
        TE_WARNLOG("Node[%s] generate simplified key failed.", opNode->GetName().c_str());
        return false;
    }
    if (!binaryInfoPtr->MatchSimpleKey(opType, simpleKeystr, jsonFilePath)) {
        TE_WARNLOG("Node[%s], opType[%s] MatchSimpleKey failed. current simpleKey = [%s].",
            opNode->GetName().c_str(), opType.c_str(), simpleKeystr.c_str());
        return false;
    }
    if(isSuperKernel) {
        // SK node that reused binary cache must go tiling sink, here add an attribute for these nodes.
        ge::AttrUtils::SetBool(opNode->GetOpDesc(), SPK_REUSED_BINARY, true);
        TE_INFOLOG("Current superkernel op[%s] was tagged super_kernel_reuse_binary true.", opNode->GetName().c_str());
    }
    return true;
}

bool BinaryManager::ReuseBinKernelBySimpleKey(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() != 1) {
        TE_DBGLOG("Op nodes size is %zu", opTask->opNodes.size());
        return false;
    }

    auto opNode = opTask->opNodes[0];
    TbeOpInfoPtr pOpInfo;
    TE_FUSION_CHECK(!TbeOpInfoCache::Instance().GetTbeOpInfoByNode(opNode, pOpInfo), {
        TE_DBGLOG("Get opinfo of node %s not successfully:", opNode->GetName().c_str());
        return false;
    });
    if (CheckDebugMode(*pOpInfo) && TeContextUtils::GetJitCompile() != JIT_MODE::JIT_USE_BINARY) {
        TE_DBGLOG("Debug mode compile node %s:", opNode->GetName().c_str());
        return false;
    }

    std::string opName = GetTaskNodeName(opTask);
    std::string jsonFilePath;
    if (!MatchSimplifiedKey(opTask, jsonFilePath)) {
        TE_INFOLOG("Node[%s] can not MatchSimplifiedKey.", opNode->GetName().c_str());
        return false;
    }
    std::string binaryPath;
    GetBinaryPath(opTask, false, true, binaryPath);
    if (binaryPath.empty()) {
        TE_INFOLOG("Binary file path is empty.");
        return false;
    }
    TE_DBGLOG("Node(%s) jsonFilePath = [%s], binaryPath = [%s]",
              opName.c_str(), jsonFilePath.c_str(), binaryPath.c_str());
    std::string opsPathNamePrefix;
    (void)ge::AttrUtils::GetStr(opTask->opNodes[0]->GetOpDesc(), OPS_PATH_NAME_PREFIX, opsPathNamePrefix);
    AssembleJsonPath(opsPathNamePrefix, binaryPath, jsonFilePath);
    std::string jsonFileRealPath = RealPath(binaryPath);
    if (jsonFileRealPath.empty()) {
        TE_DBGLOG("Node(%s) json file(%s) does not exist.", opName.c_str(), jsonFilePath.c_str());
        return false;
    }
    if (!SetBinaryReuseResult(opTask, jsonFileRealPath)) {
        TE_INFOLOG("Node(%s) not set binary reuse result.", opName.c_str());
        return false;
    }
    return true;
}

bool BinaryManager::ReuseKernelBinaryCompileRes(const OpBuildTaskPtr &opTask)
{
    json binListJson;
    if (!CheckAndReadBinaryFile(opTask, binListJson, false, false)) {
        TE_INFOLOG("Node(%s) do not check and read binary file. Need to compile",
            GetTaskNodeName(opTask).c_str());
        return false;
    }

    if (!MatchFusionOpGraphPattern(opTask, binListJson)) {
        TE_INFOLOG("Node(%s) graph pattern not match. Need to compile", GetTaskNodeName(opTask).c_str());
        return false;
    }

    GeneralizedResult generalizedResult;
    if (!ShapeGeneralization::GeneralizeOps(opTask, generalizedResult)) {
        TE_INFOLOG("Generalize ops(%s) not success. Need to compile", GetTaskNodeName(opTask).c_str());
        return false;
    }

    if (!BinaryMatchWithStaticKeyAndDynInfo(opTask, generalizedResult, binListJson)) {
        TE_INFOLOG("Node(%s) staticKey or dynInfo not match. Need to compile", GetTaskNodeName(opTask).c_str());
        return false;
    }

    std::string jsonFilePath;
    if (!IsBinaryFileValid(false, opTask, binListJson, jsonFilePath)) {
        TE_INFOLOG("Node(%s) binary file invalid. Need to compile.", GetTaskNodeName(opTask).c_str());
        return false;
    }

    // set build result
    if (!SetBinaryReuseResult(opTask, jsonFilePath)) {
        TE_INFOLOG("Node(%s) not set binary reuse result. Need to compile.", GetTaskNodeName(opTask).c_str());
        return false;
    }
    return true;
}

bool BinaryManager::SetOmBinaryReuseResult(const OpBuildTaskPtr &opTask, const std::string &jsonFilePath) const
{
    const std::string realPath = RealPath(jsonFilePath);
    if (realPath.empty()) {
        TE_INFOLOG("Node(%s)'s om json file path(%s) does not exist.",
            GetTaskNodeName(opTask).c_str(), jsonFilePath.c_str());
        return false;
    }

    if (opTask->opRes == nullptr) {
        TE_FUSION_MAKE_SHARED(opTask->opRes = std::make_shared<OpBuildTaskResult>(), return false);
    }

    opTask->opRes->statusCode = 0;
    // save binary json file path
    TE_DBGLOG("Save om binary json file path(%s) to task result", realPath.c_str());
    opTask->opRes->jsonFilePath = realPath;
    opTask->opRes->compileRetType = CompileResultType::Binary;
    opTask->status = OP_TASK_STATUS::OP_TASK_SUCC;

    TE_DBGLOG("Set om binary reuse task result successfully");
    return true;
}

bool BinaryManager::GetNodeStrAttr(const OpBuildTaskPtr &opTask, const std::string &attrKey,
                                     std::string &attrValue)
{
    for (const auto &node : opTask->opNodes) {
        if (node == nullptr) {
            TE_DBGLOGF("Node is null.");
            continue;
        }
        (void)ge::AttrUtils::GetStr(node->GetOpDesc(), attrKey, attrValue);
        if (!attrValue.empty()) {
            TE_DBGLOGF("Get node %s(%s)", attrKey.c_str(), attrValue.c_str());
            break;
        }
    }

    if (attrValue.empty()) {
        TE_DBGLOGF("Get node(%s) %s empty", GetTaskNodeName(opTask).c_str(), attrKey.c_str());
        return false;
    }
    return true;
}

bool BinaryManager::FusionOpReuseOmBinary(const OpBuildTaskPtr &opTask, json &binListJson,
                                          std::string &omKeyId) const
{
    TE_FUSION_TIMECOST_START(FusionOpReuseOmBinary);
    if (omKeyId.empty()) {
        TE_DBGLOG("Node(%s) has no omKeyId. Not reuse om binary files", GetTaskNodeName(opTask).c_str());
        return false;
    }

    if (!CheckMatchInBinListByKey(opTask, BINARY_OM_KEY_ID, omKeyId, binListJson)) {
        TE_DBGLOG("Node(%s) omKeyId(%s) not match in binary file. Not reuse om binary files",
            GetTaskNodeName(opTask).c_str(), omKeyId.c_str());
        return false;
    }

    std::string jsonFilePath;
    if (!IsBinaryFileValid(true, opTask, binListJson, jsonFilePath)) {
        TE_DBGLOG("Node(%s) binary om file invalid. Not reuse om binary files.", GetTaskNodeName(opTask).c_str());
        return false;
    }

    // set build result
    if (!SetOmBinaryReuseResult(opTask, jsonFilePath)) {
        TE_DBGLOG("Node(%s) not set om binary reuse result. Not reuse om binary files.",
            GetTaskNodeName(opTask).c_str());
        return false;
    }

    TE_INFOLOG("The node(%s) binary om json file(%s) can be reused. No need to compile.",
        GetTaskNodeName(opTask).c_str(), jsonFilePath.c_str());
    TE_FUSION_TIMECOST_END(FusionOpReuseOmBinary, "FusionOpReuseOmBinary");
    return true;
}

bool BinaryManager::CheckIsCanReuseOmBinaryCompileRes(const OpBuildTaskPtr &opTask)
{
    if (opTask->opNodes.size() > 0) {
        if (opTask->opNodes[0]->GetOpDesc()->HasAttr(ATTR_NAME_CAN_NOT_REUSE_OM)) {
            TE_DBGLOG("Node(%s) is ffts+ auto mode and shape is static, can not reuse om. Need to compile.",
                GetTaskNodeName(opTask).c_str());
            return false;
        }
        return true;
    }
    TE_DBGLOGF("There are no nodes in opTask. taskID[%lu:%lu].", opTask->graphId, opTask->taskId);
    return false;
}

bool BinaryManager::ReuseOmBinaryCompileRes(const OpBuildTaskPtr &opTask, bool &hasOmKeyId)
{
    TE_DBGLOG("Process binary om file resuse. (Node = %s).", GetTaskNodeName(opTask).c_str());
    if (!CheckIsCanReuseOmBinaryCompileRes(opTask)) {
        TE_DBGLOG("Node(%s) cat not reuse om BinaryCompileRes. Need to compile.", GetTaskNodeName(opTask).c_str());
        return false;
    }

    std::string omKeyId;
    if (GetNodeStrAttr(opTask, BINARY_OM_KEY_ID_ATTR, omKeyId)) {
        hasOmKeyId = true;
    } else {
        hasOmKeyId = false;
    }

    json binListJson;
    if (!CheckAndReadBinaryFile(opTask, binListJson, hasOmKeyId, true)) {
        TE_DBGLOG("Node(%s) do not check and read binary file. Not reuse om binary files",
            GetTaskNodeName(opTask).c_str());
        return false;
    }

    if (!hasOmKeyId && opTask->opNodes.size() == 1) {
        return SingleOpReuseOmBinary(opTask, binListJson);
    }

    return FusionOpReuseOmBinary(opTask, binListJson, omKeyId);
}

bool BinaryManager::MatchAndReuseBinRes(const OpBuildTaskPtr &opTask)
{
    TE_FUSION_TIMECOST_START(ReuseOmBinaryCompileRes);
    bool hasOmKeyId = false;
    if (ReuseOmBinaryCompileRes(opTask, hasOmKeyId)) {
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::REUSE_SUCC);
        return true;
    }
    TE_FUSION_TIMECOST_END(ReuseOmBinaryCompileRes, "ReuseOmBinaryCompileRes");
    if (hasOmKeyId) {
        TE_INFOLOG("Binary om with omKeyid not matched. No need to match binary kernel files. (Node = %s)",
            GetTaskNodeName(opTask).c_str());
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::OM_MISMATCH);
        return false;
    }

    TE_DBGLOG("Binary om not matched. Continue to match binary kernel files. (Node = %s)",
        GetTaskNodeName(opTask).c_str());

    TE_DBGLOG("Process binary kernel file resuse. (Node = %s)", GetTaskNodeName(opTask).c_str());
    if (!CheckReuseBinaryCondition(opTask)) {
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::RESUE_CHECK_FAIL);
        return false;
    }

    bool res = ReuseBinKernelBySimpleKey(opTask);
    if (res) {
        TE_INFOLOG("Node(%s) reuse kernel binary by simpliedKey.", GetTaskNodeName(opTask).c_str());
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::REUSE_SUCC);
        return res;
    }
    DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::SIMKEY_MISMATCH);
    res = ReuseKernelBinaryCompileRes(opTask);
    if (!res) {
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::STATICKEY_MISMATCH);
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::REUSE_FAIL);
    } else {
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::REUSE_SUCC);
    }
    return res;
}

bool BinaryManager::BackToSingleCheck(const OpBuildTaskPtr &opTask)
{
    std::vector<ge::Node *> opNodes = opTask->opNodes;
    TE_INFOLOG("Node [%s] is ready to check if all nodes can reuse the op binarykernel.", GetTaskNodeName(opTask).c_str());
    for (const auto &node : opNodes) {
        std::vector<ge::Node *> teGraphNode;
        teGraphNode.emplace_back(node);
        OpBuildTaskPtr opTempTask = nullptr;
        TE_FUSION_MAKE_SHARED(opTempTask = std::make_shared<OpBuildTask>(0, 0, 0, teGraphNode,
                nullptr, OP_TASK_STATUS::OP_TASK_PENDING), return false);
        if (!MatchAndReuseBinRes(opTempTask)) {
            TE_INFOLOG("Node [%s] cannot find the binarykernel; needs to compile the fusion op online.", node->GetName().c_str());
            return false;
        }
    }
    TE_INFOLOG("Node[%s] all nodes has binarykernel, Back to single op.", GetTaskNodeName(opTask).c_str());
    return true;
}

bool BinaryManager::CanReuseBinaryKernel(const OpBuildTaskPtr &opTask)
{
    TE_DBGLOG("Start to process binary file resuse. (Node = %s).", GetTaskNodeName(opTask).c_str());
    TE_FUSION_TIMECOST_START(CanReuseBinaryKernel);
    DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::MATCH);
    if (!CheckConditionsForReuse(opTask)) {
        TE_DBGLOG("Cannot reuse binary result for node(%s)", GetTaskNodeName(opTask).c_str());
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::BINARY_REUSE, RecordEventType::RESUE_CHECK_FAIL);
        return false;
    }

    bool bres = MatchAndReuseBinRes(opTask);
    std::string key = GetTaskNodeName(opTask) + ": CanReuseBinaryKernel";
    TE_FUSION_TIMECOST_END(CanReuseBinaryKernel, key.c_str());
    if (opTask->opNodes.size() == 1 || bres) {
        return bres;
    }

    if (opTask->opRes == nullptr) {
        TE_FUSION_MAKE_SHARED(opTask->opRes = std::make_shared<OpBuildTaskResult>(), return false);
    }

    opTask->opRes->backToSingleOpBinReuse = BackToSingleCheck(opTask);
    return bres;
}

bool BinaryManager::GetBuiltInOppLatestFlag() const
{
    return builtInOppLatestFlag_;
}

void BinaryManager::SetBinaryReuseAttr(const OpBuildTaskPtr &opTask) const
{
    auto firstNode = opTask->opNodes[0];
    TE_DBGLOG("Ready to set opp binary path for node[%s, %s]",
              firstNode->GetNamePtr(), firstNode->GetTypePtr());
    int64_t implType = -1;
    (void)ge::AttrUtils::GetInt(firstNode->GetOpDesc(), FE_IMPLY_TYPE, implType);
    if (implType == AICORE_IMPL_TYPE_NUM && GetBuiltInOppLatestFlag()) {
        ge::AttrUtils::SetInt(firstNode->GetOpDesc(), OPP_PATH, OPP_LATEST_ENUM);
        TE_INFOLOG("Set opp binary path enum[%ld] for node[%s, %s].", OPP_LATEST_ENUM,
                   firstNode->GetNamePtr(), firstNode->GetTypePtr());
    }
}

void BinaryManager::GetBinaryPath(const OpBuildTaskPtr &opTask, bool isOm,
                                  bool isKernelOrModel, std::string &binaryPath) const
{
    TE_DBGLOG("isOm %d, isKernelOrModel %d", isOm, isKernelOrModel);
    auto path = &binaryOppPath_;
    if (isOm) {
        if (isKernelOrModel) {
            path = &binaryOmModelPath_;
        } else {
            path = &binaryOmPath_;
        }
    } else {
        if (isKernelOrModel) {
            path = &binaryOppKernelPath_;
        }
    }

    for (const auto &currentNode : opTask->opNodes) {
        TE_FUSION_CHECK(currentNode == nullptr, continue);
        int64_t implType = -1;
        if (!ge::AttrUtils::GetInt(currentNode->GetOpDesc(), FE_IMPLY_TYPE, implType)) {
            continue;
        }

        std::map<int64_t, std::string>::const_iterator iter = path->find(implType);
        if (iter == path->end() || iter->second.empty()) {
            TE_DBGLOG("Node[%s] impltype[%ld] has no binary path.", currentNode->GetName().c_str(), implType);
            return;
        }
        binaryPath = iter->second;
        TE_DBGLOG("Node[%s] impltype[%ld] get binary path [%s].", currentNode->GetName().c_str(),
                  implType, binaryPath.c_str());
        return;
    }
    return;
}

void BinaryManager::GetBinaryVersion(const OpBuildTaskPtr &opTask, bool isOm,
                                     std::string &adkVrsion, std::string &oppVersion) const
{
    TE_DBGLOG("Binary version is %s module", isOm ? "offline" : "online");
    auto adkMap = &binAdkVersionMap_;
    auto oppMap = &binOppVersionMap_;

    if (isOm) {
        adkMap = &binOmAdkVersionMap_;
        oppMap = &binOmVersionMap_;
    }

    for (const auto &currentNode : opTask->opNodes) {
        TE_FUSION_CHECK(currentNode == nullptr, continue);
        int64_t implType = -1;
        if (!ge::AttrUtils::GetInt(currentNode->GetOpDesc(), FE_IMPLY_TYPE, implType)) {
            continue;
        }
        TE_DBGLOG("Node[%s] impltype[%ld]", currentNode->GetName().c_str(), implType);
        std::map<int64_t, std::string>::const_iterator iter = adkMap->find(implType);
        if (iter == adkMap->end() || iter->second.empty()) {
            TE_DBGLOGF("Node[%s] impltype[%ld] has no adk version.", currentNode->GetName().c_str(), implType);
            return;
        }
        adkVrsion = iter->second;

        std::map<int64_t, std::string>::const_iterator iter1 = oppMap->find(implType);
        if (iter1 == oppMap->end() || iter1->second.empty()) {
            TE_DBGLOG("Node[%s] impltype[%ld] has no opp version.", currentNode->GetName().c_str(), implType);
            return;
        }
        oppVersion = iter1->second;
        TE_DBGLOG("Node[%s] impltype[%ld] get adkVrsion[%s] oppVersion[%s]", currentNode->GetName().c_str(),
                  implType, adkVrsion.c_str(), oppVersion.c_str());
        return;
    }
    TE_DBGLOG("Not find op version!");
    return;
}
} // namespace fusion
} // namespace te
