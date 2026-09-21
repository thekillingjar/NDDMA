/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "binary/fusion_binary_info.h"
#include <fcntl.h>
#include <fstream>
#include <cerrno>
#include <dirent.h>
#include <sys/stat.h>
#include <nlohmann/json.hpp>
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"
#include "common/te_config_info.h"
#include "common/te_file_utils.h"

namespace te {
namespace fusion {
constexpr const char *SIMPLIFIED_KEY = "simplifiedKey";
constexpr const char *SIMPLIFIED_KEY_MODE = "simplifiedKeyMode";
constexpr const char *OP_BINARY_LIST = "binaryList";
constexpr const char *OP_JSON_PATH = "jsonPath";
constexpr const char *BINARY_INFO_CONFIG_PARAMS = "params";
constexpr const char *OP_DTYPE_MODE = "dtypeMode";
constexpr const char *OP_FORMAT_MODE = "formatMode";
constexpr const char *BINARY_INFO_CONFIG_JSON = "binary_info_config.json";
constexpr const char *RELOCATABLE_KERNEL_INFO_CONFIG_JSON = "relocatable_kernel_info_config.json";
constexpr const char *FOLDED = "folded_with_desc";
constexpr const char *UNFOLDED = "unfolded";
constexpr const int NUM_4 = 4;

static const std::set<std::string> DTYPE_MODE_SET {DTYPE_MODE_NORMAL, DTYPE_MODE_BIT, DTYPE_MODE_BOOL};
static const std::set<std::string> FROMAT_MODE_SET {FORMAT_MODE_NORMAL, FORMAT_MODE_AGNOSTIC,
                                                    FORMAT_MODE_ND_AGNOSTIC, FORMAT_STATIC_ND_AGNOSTIC};
static std::unordered_set<std::string> OPT_INPUT_MODE_SET = {GEN_PLACEHOLDER, NO_PLACEHOLDER};
static std::unordered_set<std::string> DYN_PARAM_MODE_SET = {FOLDED, UNFOLDED};
static const std::unordered_map<std::string, DynamicParamModeEnum> DYN_PARAM_MODE_STR_2_ENUM = {
    std::make_pair(FOLDED, DynamicParamModeEnum::FOLDED_WITH_DESC),
    std::make_pair(UNFOLDED, DynamicParamModeEnum::UNFOLDED)
};

bool BinaryInfoBase::GetBinaryInfoConfigPath(const string &binaryConfigPath, std::string &configRealPath,
                                             const bool &isSuperKernel, const std::string &path) const
{
    std::string shortSocVersion = TeConfigInfo::Instance().GetShortSocVersion();
    std::transform(shortSocVersion.begin(), shortSocVersion.end(), shortSocVersion.begin(), ::tolower);
    std::string binaryInfoJsonPath = isSuperKernel ? RELOCATABLE_KERNEL_INFO_CONFIG_JSON : BINARY_INFO_CONFIG_JSON;
    std::string binaryInfoConfigPath;
    if (path.empty()) {
        binaryInfoConfigPath = binaryConfigPath + shortSocVersion + "/" + binaryInfoJsonPath;
    } else {
        binaryInfoConfigPath = binaryConfigPath + shortSocVersion + "/" + path + "/" + binaryInfoJsonPath;
    }
    configRealPath = RealPath(binaryInfoConfigPath);
    if (configRealPath.empty()) {
        TE_WARNLOG("binary_info_config.json file[%s] does not exist", binaryInfoConfigPath.c_str());
        return false;
    }
    return true;
}

bool BinaryInfoBase::ReadOpBinaryInfoConfig(const std::string &binaryInfoConfigPath,
                                            nlohmann::json &binaryInfoConfig) const
{
    TE_INFOLOG("Start to read binary info file [%s].", binaryInfoConfigPath.c_str());
    FILE* fp = TeFileUtils::LockFile(binaryInfoConfigPath, false, F_RDLCK);
    if (fp == nullptr) {
        TE_WARNLOG("fp is NULL.binaryInfoConfigPath=%s.", binaryInfoConfigPath.c_str());
        return false;
    }
    std::ifstream ifs(binaryInfoConfigPath);

    try {
        if (!ifs.is_open()) {
            TE_WARNLOG("Open binary_info_config.json[%s] failed. reason [%s]",
                       binaryInfoConfigPath.c_str(), strerror(errno));
            TeFileUtils::ReleaseLockFile(fp, binaryInfoConfigPath);
            return false;
        }
        ifs >> binaryInfoConfig;
        if (!binaryInfoConfig.is_object()) {
            TE_WARNLOG("binaryInfoConfigPath [%s] has an invalid JSON format", binaryInfoConfigPath.c_str());
            ifs.close();
            TeFileUtils::ReleaseLockFile(fp, binaryInfoConfigPath);
            return false;
        }
        ifs.close();
        TeFileUtils::ReleaseLockFile(fp, binaryInfoConfigPath);
        TE_INFOLOGF("Read binary info file[%s] success", binaryInfoConfigPath.c_str());
    } catch (const nlohmann::json::exception &e) {
        TE_WARNLOG("Reading binaryInfoConfigPath [%s] failed, reason: %s", binaryInfoConfigPath.c_str(), e.what());
        ifs.close();
        TeFileUtils::ReleaseLockFile(fp, binaryInfoConfigPath);
        return false;
    }
    return true;
}

bool BinaryInfoBase::MatchSimpleKey(const std::string opType, const std::string simpleKey,
                                    std::string &jsonFilePath) const
{
    std::lock_guard<std::mutex> lockGuard(simpleKeyMutex_);
    auto iter = simpleKeyMap_.find(opType);
    if (iter == simpleKeyMap_.end()) {
        TE_WARNLOG("OpType [%s] is not found in simpleKeyMap.", opType.c_str());
        return false;
    }
    std::unordered_map<std::string, std::string> simpleKeyInfoMap = iter->second;
    auto simpleKeyIter = simpleKeyInfoMap.find(simpleKey);
    if (simpleKeyIter == simpleKeyInfoMap.end()) {
        TE_DBGLOG("opType[%s] simpleKey[%s] is not in simpleKeyMap", opType.c_str(), simpleKey.c_str());
        return false;
    }
    jsonFilePath = simpleKeyIter->second;
    TE_DBGLOG("Match opType[%s], simpleKey[%s] succeeded. binJsonPath is [%s].",
              opType.c_str(), simpleKey.c_str(), jsonFilePath.c_str());
    return true;
}

void BinaryInfoBase::InsertBinaryInfoIntoMap(const std::string opType, const SimpleKeyModeType simpleKeyMode,
                                             const std::string &optionalInputMode, const std::string &dynamicParamMode,
                                             const std::unordered_map<std::string, std::string> &simpleKeyInfoMap)
{
    std::lock_guard<std::mutex> lockGuard(simpleKeyMutex_);
    int inputMode = 0;  // default no_placehold
    if (optionalInputMode == GEN_PLACEHOLDER) {
        inputMode = 1;
    }
    DynamicParamModeEnum dynParamMode = DynamicParamModeEnum::UNDEFINED;
    if (!dynamicParamMode.empty()) {
        try {
            dynParamMode = DYN_PARAM_MODE_STR_2_ENUM.at(dynamicParamMode);
        } catch (const nlohmann::json::exception &e) {
            TE_WARNLOG("Trans dynamic param mode str [%s] to enum failed, reason %s",
                       dynamicParamMode.c_str(), e.what());
        }
    }
    if (simpleKeyMode == SimpleKeyModeType::SIMPLE_MODE) {
        dynParamMode = DynamicParamModeEnum::FOLDED_WITH_DESC;
    }
    simpleKeyModeMap_[opType] = simpleKeyMode;
    optionalInputMode_[opType] = inputMode;
    simpleKeyMap_[opType] = simpleKeyInfoMap;
    opDynParamMode_[opType] = dynParamMode;
    return;
}

bool BinaryInfoBase::GetInputMode(const std::string &opType, DtypeFormatMode &mode) const
{
    auto iter = inputMode_.find(opType);
    if (iter == inputMode_.end()) {
        TE_WARNLOG("OpType [%s] is not supported in the current inputMode.", opType.c_str());
        return false;
    }
    mode = iter->second;
    return true;
}

bool BinaryInfoBase::GetSimpleKeyMode(const std::string &opType, SimpleKeyModeType &simplekeyMode) const
{
    auto iter = simpleKeyModeMap_.find(opType);
    if (iter == simpleKeyModeMap_.end()) {
        TE_WARNLOG("OpType[%s] is not found in simpleKeyModeMap", opType.c_str());
        return false;
    }
    simplekeyMode = iter->second;
    return true;
}

bool BinaryInfoBase::GetOutputMode(const std::string &opType, DtypeFormatMode &mode) const
{
    auto iter = outputMode_.find(opType);
    if (iter == outputMode_.end()) {
        TE_WARNLOG("OpType[%s] is not in outputMode_", opType.c_str());
        return false;
    }
    mode = iter->second;
    return true;
}

bool BinaryInfoBase::GetOptionalInputMode(const std::string &opType, int &optionalInputMode) const
{
    auto iter = optionalInputMode_.find(opType);
    if (iter == optionalInputMode_.end()) {
        TE_WARNLOG("OpType[%s] is not in optionalInputMode_", opType.c_str());
        return false;
    }
    optionalInputMode = iter->second;
    return true;
}

bool BinaryInfoBase::GetDynParamModeByOpType(const std::string &opType, DynamicParamModeEnum &dynParamMode) const
{
    auto iter = opDynParamMode_.find(opType);
    if (iter == opDynParamMode_.end()) {
        TE_WARNLOG("OpType [%s] is not found in opDynParamMode", opType.c_str());
        return false;
    }
    dynParamMode = iter->second;
    return true;
}

bool BinaryInfoBase::GenerateInOutPutMode(const std::string opType, const std::string &type,
                                          const nlohmann::json &binaryInfoParams, DtypeFormatMode &mode) const
{
    const auto inputOutputIter = binaryInfoParams.find(type);
    if (inputOutputIter == binaryInfoParams.end()) {
        TE_DBGLOG("params json has no %s", type.c_str());
        return true;
    }

    const nlohmann::json &inputOutputJson = inputOutputIter.value();
    if (!inputOutputJson.is_array()) {
        TE_DBGLOG("Format wrong! '%s' should be array.", type.c_str());
        return false;
    }
    uint32_t size = inputOutputJson.size();
    std::vector<std::string> dTypeModeVec;
    std::vector<std::string> formatModeVec;
    for (auto iter = inputOutputJson.begin(); iter != inputOutputJson.end(); ++iter) {
        nlohmann::json iterJsonValue = iter.value();
        TE_DBGLOG("jsonValue: %s.", iterJsonValue.dump().c_str());
        if (iterJsonValue.is_null()) {
            dTypeModeVec.emplace_back("normal");
            formatModeVec.emplace_back("normal");
            continue;
        }
        if (iterJsonValue.is_array()) {
            iterJsonValue = iterJsonValue[0];
            TE_DBGLOG("jsonValue_new: %s.", iterJsonValue.dump().c_str());
        }
        std::string dtypeModeStr;
        if (iterJsonValue.find(OP_DTYPE_MODE) == iterJsonValue.end()) {
            dtypeModeStr = "normal";
        } else {
            try {
                dtypeModeStr = iterJsonValue[OP_DTYPE_MODE].get<string>();
            } catch (const std::exception &e) {
                TE_ERRLOG("opType [%s], failed to get dTypeMode. Error message is: %s", opType.c_str(), e.what());
                return false;
            }
        }
        if (DTYPE_MODE_SET.find(dtypeModeStr) == DTYPE_MODE_SET.end()) {
            TE_ERRLOG("opType [%s] dtypeModeStr [%s] not in DTYPE_MODE_SET.", opType.c_str(), dtypeModeStr.c_str());
            return false;
        }
        TE_DBGLOG("Op [%s] [%s] dtypeMode [%s]", opType.c_str(), type.c_str(), dtypeModeStr.c_str());
        dTypeModeVec.emplace_back(dtypeModeStr);

        std::string formatModeStr;
        if (iterJsonValue.find(OP_FORMAT_MODE) == iterJsonValue.end()) {
            formatModeStr = "normal";
        } else {
            try {
                formatModeStr = iterJsonValue[OP_FORMAT_MODE].get<string>();
            } catch (const std::exception &e) {
                TE_ERRLOG("opType [%s], failed to get formatMode. Error message is: %s", opType.c_str(), e.what());
                return false;
            }
        }
        if (FROMAT_MODE_SET.find(formatModeStr) == FROMAT_MODE_SET.end()) {
            TE_ERRLOG("opType [%s], formatModeStr [%s] is not in FROMAT_MODE_SET.",
                      opType.c_str(), formatModeStr.c_str());
            return false;
        }
        TE_DBGLOG("Op [%s] [%s] formatMode [%s].", opType.c_str(), type.c_str(), formatModeStr.c_str());
        formatModeVec.emplace_back(formatModeStr);
    }
    mode.opType = opType;
    mode.count = size;
    mode.dTypeMode = dTypeModeVec;
    mode.formatMode = formatModeVec;
    return true;
}

bool BinaryInfoBase::GenerateSimpleKeyList(const std::string &opType, const nlohmann::json &binaryList,
                                           std::unordered_map<std::string, std::string>  &simpleKeyInfoMap) const
{
    if (!binaryList.is_array()) {
        TE_WARNLOG("binaryList [%s] is not an array.", binaryList.dump().c_str());
        return false;
    }
    for (auto binInfo : binaryList) {
        if (binInfo.find(SIMPLIFIED_KEY) == binInfo.end()) {
            TE_WARNLOG("opType [%s] is not contain simplifiedKey", opType.c_str());
            return false;
        }
        auto simplifiedKeyList = binInfo[SIMPLIFIED_KEY];
        if (!simplifiedKeyList.is_array()) {
            TE_WARNLOG("opType [%s] simplifiedKey is not an array", opType.c_str());
            return false;
        }

        if (binInfo.find(OP_JSON_PATH) == binInfo.end()) {
            TE_WARNLOG("opType [%s] does not contain jsonPath", opType.c_str());
            return false;
        }
        std::string jsonPath;
        try {
            jsonPath = binInfo[OP_JSON_PATH].get<std::string>();
        } catch (const std::exception &e) {
            TE_WARNLOG("opType [%s] failed to get jsonPath. reason is %s", opType.c_str(), e.what());
            return false;
        }
        for (auto simplifiedKey : simplifiedKeyList) {
            if (simpleKeyInfoMap.find(simplifiedKey) == simpleKeyInfoMap.end()) {
                simpleKeyInfoMap[simplifiedKey] = jsonPath;
            } else {
                TE_WARNLOG("SimplifiedKey: [%s] already exists; it should be unique. jsonPath: [%s]",
                    simplifiedKey.dump().c_str(), jsonPath.c_str());
                return false;
            }
        }
    }
    return true;
}

bool BinaryInfoBase::GenerateDtypeFormatMode(const std::string &opType, const nlohmann::json &binaryInfoParams)
{
    TE_DBGLOGF("opType:[%s], params:[%s].", opType.c_str(), binaryInfoParams.dump().c_str());
    DtypeFormatMode inputMode;
    DtypeFormatMode outputMode;
    if (binaryInfoParams.empty()) {
        TE_ERRLOG("opType [%s], binaryInfo params is empty.", opType.c_str());
        return false;
    }
    GenerateInOutPutMode(opType, INPUTS, binaryInfoParams, inputMode);
    GenerateInOutPutMode(opType, OUTPUTS, binaryInfoParams, outputMode);

    inputMode_[opType] = inputMode;
    outputMode_[opType] = outputMode;
    return true;
}

template <typename T>
bool TryGetValueByKey(const nlohmann::json &jsonValue, const std::string &opType, const std::string &key, T &val)
{
    try {
        val = jsonValue[key].get<T>();
    } catch (const std::exception &e) {
        TE_WARNLOG("opType [%s] failed to get simpleKeyMode. Reason is %s", opType.c_str(), e.what());
        return false;
    }
    return true;
}

bool BinaryInfoBase::GetParamModeFromJson(const nlohmann::json &jsonValue, const std::string &opType,
    const std::string &key, std::string &paramMode) const
{
    if (jsonValue.find(key) == jsonValue.end()) {
        return true;
    } else {
        if (!TryGetValueByKey(jsonValue, opType, key, paramMode)) {
            return false;
        }
    }
    if (key == OPT_INPUT_MODE) {
        if (OPT_INPUT_MODE_SET.count(paramMode) == 0) {
            TE_ERRLOG("optionalInputMode value [%s] is not supported.", paramMode.c_str());
            return false;
        }
    } else {
        if (DYN_PARAM_MODE_SET.count(paramMode) == 0) {
            TE_ERRLOG("Unsupported dynamicParamMode value: [%s].", paramMode.c_str());
            return false;
        }
    }
    return true;
}

bool BinaryInfoBase::GenerateBinaryInfo(nlohmann::json &binaryInfoConfig)
{
    if (binaryInfoConfig.empty()) {
        TE_WARNLOG("binaryInfoConfig is empty.");
        return false;
    }
    std::string opType;
    for (auto iter = binaryInfoConfig.begin(); iter != binaryInfoConfig.end(); ++iter) {
        opType = iter.key();
        std::unordered_map<std::string, std::string> simpleKeyInfoMap;
        auto iterJsonValue = iter.value();
        if (iterJsonValue.find(SIMPLIFIED_KEY_MODE) == iterJsonValue.end()) {
            TE_INFOLOG("opType [%s] does not contain simplifiedKeyMode.", opType.c_str());
            continue;
        }
        int simpleKeyModeValue = 0;
        if (!TryGetValueByKey(iterJsonValue, opType, SIMPLIFIED_KEY_MODE, simpleKeyModeValue)) {
            return false;
        }
        SimpleKeyModeType simpleKeyMode = static_cast<SimpleKeyModeType>(simpleKeyModeValue);
        std::vector<SimpleKeyModeType> vecMode = {SimpleKeyModeType::SIMPLE_MODE,
            SimpleKeyModeType::COMPATIBLE_MODE, SimpleKeyModeType::CUSTOM_MODE};
        if (std::find(vecMode.begin(), vecMode.end(), simpleKeyMode) == vecMode.end()) {
            TE_ERRLOG("simpleKeyMode value [%d] is out of range.", simpleKeyMode);
            return false;
        }
        std::string optionalInputMode = NO_PLACEHOLDER;
        if (!GetParamModeFromJson(iterJsonValue, opType, OPT_INPUT_MODE, optionalInputMode)) {
            return false;
        }
        std::string dynamicParamMode;
        if (!GetParamModeFromJson(iterJsonValue, opType, DYN_PARAM_MODE, dynamicParamMode)) {
            return false;
        }
        if (iterJsonValue.find(BINARY_INFO_CONFIG_PARAMS) == iterJsonValue.end()) {
            TE_WARNLOG("opType [%s] does not contain params.", opType.c_str());
            continue;
        }
        nlohmann::json binaryInfoParams = iterJsonValue[BINARY_INFO_CONFIG_PARAMS];
        if (!GenerateDtypeFormatMode(opType, binaryInfoParams)) {
            TE_ERRLOG("opType [%s], params is empty. return failed.", opType.c_str());
            return false;
        }
        if (iterJsonValue.find(OP_BINARY_LIST) == iterJsonValue.end()) {
            TE_ERRLOG("opType [%s] does not contain binaryList.", opType.c_str());
            return false;
        }
        nlohmann::json binaryList = iterJsonValue[OP_BINARY_LIST];
        if (!GenerateSimpleKeyList(opType, binaryList, simpleKeyInfoMap)) {
            TE_WARNLOG("OpType [%s], GenerateSimpleKeyList operation failed", opType.c_str());
            continue;
        }
        InsertBinaryInfoIntoMap(opType, simpleKeyMode, optionalInputMode, dynamicParamMode, simpleKeyInfoMap);
    }
    return true;
}

void BinaryInfoBase::GetAllOpsPathNamePrefix(const std::string &binaryConfigJsonPath,
                                             std::vector<std::string> &binaryConfigPathPrefix) {
    DIR *dir = nullptr;
    struct dirent *dirp = nullptr;
    std::string realBinaryConfigJsonPath = RealPath(binaryConfigJsonPath);
    if (realBinaryConfigJsonPath.empty()) {
        TE_WARNLOG("Original dir path %s is invalid.", binaryConfigJsonPath.c_str());
        return;
    }

    dir = opendir(realBinaryConfigJsonPath.c_str());
    if (dir == nullptr) {
        TE_DBGLOG("Unable to open directory %s.", realBinaryConfigJsonPath.c_str());
        return;
    }

    while ((dirp = readdir(dir)) != nullptr) {
        if (dirp->d_type == DT_DIR) {
            std::string fileName(dirp->d_name);
            if (fileName.length() >= NUM_4 && fileName.substr(0, NUM_4) == "ops_") {
                binaryConfigPathPrefix.emplace_back(fileName);
            }
        }
    }
    closedir(dir);
    std::sort(binaryConfigPathPrefix.begin(), binaryConfigPathPrefix.end(), compareStrings);
}

bool BinaryInfoBase::ParseBinaryInfoFile(const string &binaryConfigPath, const bool &isSuperKernel)
{
    TE_INFOLOG("start to ParseBinaryInfoFile.");
    std::vector<std::string> binaryConfigPathPrefix = {""};
    std::string shortSocVersion = TeConfigInfo::Instance().GetShortSocVersion();
    std::transform(shortSocVersion.begin(), shortSocVersion.end(), shortSocVersion.begin(), ::tolower);
    string binaryConfigJsonPath = binaryConfigPath + shortSocVersion;
    GetAllOpsPathNamePrefix(binaryConfigJsonPath, binaryConfigPathPrefix);
    for (auto &path : binaryConfigPathPrefix) {
        std::string binaryInfoConfigPath;
        nlohmann::json binaryInfoConfig;
        if (!GetBinaryInfoConfigPath(binaryConfigPath, binaryInfoConfigPath, isSuperKernel, path)) {
            TE_WARNLOG("binary_info_config.json file does not exist.");
            continue;
        };
        if (!ReadOpBinaryInfoConfig(binaryInfoConfigPath, binaryInfoConfig)) {
            TE_WARNLOG("Failed to read binary_info_config.json");
            return false;
        }
        if (!GenerateBinaryInfo(binaryInfoConfig)) {
            TE_WARNLOG("GenerateBinaryInfo failed");
            return false;
        }
    }
    return true;
}

bool BinaryInfoBase::GetOppBinFlag() const
{
    return oppBinFlag_;
}

void BinaryInfoBase::SetOppBinFlag(bool flag)
{
    oppBinFlag_ = flag;
}
} // namespace fusion
} // namespace te
