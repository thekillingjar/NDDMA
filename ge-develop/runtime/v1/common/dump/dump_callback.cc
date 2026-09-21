/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include "executor/ge_executor.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "adump_api.h"
#include "adump_pub.h"
#include "mmpa/mmpa_api.h"
#include "dump_callback.h"

using namespace nlohmann;

namespace ge {

std::string GetCfgStrByKey(const nlohmann::json &js, const std::string &key) {
    return js.at(key).get<std::string>();
}

bool ContainKey(const nlohmann::json &js, const std::string &key) {
    return (js.find(key) != js.end());
}

void from_json(const nlohmann::json &js, DumpBlacklist &blacklist) {
    if (ContainKey(js, GE_DUMP_BLACKLIST_NAME)) {
        blacklist.name = GetCfgStrByKey(js, GE_DUMP_BLACKLIST_NAME);
    }
    if (ContainKey(js, GE_DUMP_BLACKLIST_POS)) {
        blacklist.pos = js.at(GE_DUMP_BLACKLIST_POS).get<std::vector<std::string>>();
    }
}

void from_json(const nlohmann::json &js, ModelDumpConfig &info) {
    if (ContainKey(js, GE_DUMP_MODEL_NAME)) {
        info.model_name = GetCfgStrByKey(js, GE_DUMP_MODEL_NAME);
    }
    if (ContainKey(js, GE_DUMP_LAYER) && js[GE_DUMP_LAYER].is_array()) {
        info.layers = js.at(GE_DUMP_LAYER).get<std::vector<std::string>>();
    }
    if (ContainKey(js, GE_DUMP_WATCHER_NODES) && js[GE_DUMP_WATCHER_NODES].is_array()) {
        info.watcher_nodes = js.at(GE_DUMP_WATCHER_NODES).get<std::vector<std::string>>();
    }
    if (ContainKey(js, GE_DUMP_OPTYPE_BLACKLIST) && js[GE_DUMP_OPTYPE_BLACKLIST].is_array()) {
        info.optype_blacklist = js[GE_DUMP_OPTYPE_BLACKLIST].get<std::vector<DumpBlacklist>>();
    }
    if (ContainKey(js, GE_DUMP_OPNAME_BLACKLIST) && js[GE_DUMP_OPNAME_BLACKLIST].is_array()) {
        info.opname_blacklist = js[GE_DUMP_OPNAME_BLACKLIST].get<std::vector<DumpBlacklist>>();
    }
    if (ContainKey(js, GE_DUMP_OPNAME_RANGE) && js[GE_DUMP_OPNAME_RANGE].is_array()) {
        info.dump_op_ranges.clear();
        for (const auto& rangeJson : js[GE_DUMP_OPNAME_RANGE]) {
            if (rangeJson.contains(GE_DUMP_OPNAME_RANGE_BEGIN) &&
                rangeJson.contains(GE_DUMP_OPNAME_RANGE_END)) {
                std::string begin = rangeJson[GE_DUMP_OPNAME_RANGE_BEGIN].get<std::string>();
                std::string end = rangeJson[GE_DUMP_OPNAME_RANGE_END].get<std::string>();
                info.dump_op_ranges.emplace_back(begin, end);
            }
        }
    }
}


bool DumpConfigValidator::IsValidDumpConfig(const nlohmann::json &js) {
    GELOGI("start to execute IsValidDumpConfig.");

    if (!js.contains(GE_DUMP)) {
        GELOGI("no dump item, no need to do dump!");
        return true;
    }

    const nlohmann::json &jsDumpConfig = js[GE_DUMP];

    if (jsDumpConfig.contains(GE_DUMP_SCENE)) {
        std::string dumpScene;
        return CheckDumpSceneSwitch(jsDumpConfig, dumpScene);
    }

    if (IsDumpDebugEnabled(jsDumpConfig)) {
        return DumpDebugCheck(jsDumpConfig);
    }

    return ValidateNormalDumpConfig(jsDumpConfig);
}

bool DumpConfigValidator::IsDumpDebugEnabled(const nlohmann::json &jsDumpConfig) {
    std::string dumpDebug = GE_DUMP_DEBUG_DEFAULT;
    if (jsDumpConfig.contains(GE_DUMP_DEBUG)) {
        dumpDebug = jsDumpConfig[GE_DUMP_DEBUG].get<std::string>();
    }
    return (dumpDebug == GE_DUMP_STATUS_ON);
}

bool DumpConfigValidator::ValidateNormalDumpConfig(const nlohmann::json &jsDumpConfig) {
    if (!ValidateDumpPath(jsDumpConfig)) {
        return false;
    }

    if (!ValidateDumpMode(jsDumpConfig)) {
        return false;
    }

    if (!ValidateDumpLevel(jsDumpConfig)) {
        return false;
    }

    return ValidateOtherDumpConfigs(jsDumpConfig);
}

bool DumpConfigValidator::ValidateDumpPath(const nlohmann::json &jsDumpConfig) {
    if (!jsDumpConfig.contains(GE_DUMP_PATH)) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpConfig]dump_path field in dump config does not exist");
        REPORT_INNER_ERR_MSG("E19999", "dump_path field in dump config does not exist");
        return false;
    }

    const std::string dumpPath = jsDumpConfig[GE_DUMP_PATH].get<std::string>();
    if (dumpPath.empty()) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpConfig]dump_path field is null in config");
        REPORT_INNER_ERR_MSG("E19999", "dump_path field is null in config");
        return false;
    }

    return CheckDumpPath(jsDumpConfig);
}

bool DumpConfigValidator::ValidateDumpMode(const nlohmann::json &jsDumpConfig) {
    std::string dumpMode = GE_DUMP_MODE_DEFAULT;
    if (jsDumpConfig.contains(GE_DUMP_MODE)) {
        dumpMode = jsDumpConfig[GE_DUMP_MODE].get<std::string>();
    }

    const std::vector<std::string> validModes = {
        GE_DUMP_MODE_INPUT, GE_DUMP_MODE_OUTPUT, GE_DUMP_MODE_ALL
    };

    if (std::find(validModes.begin(), validModes.end(), dumpMode) == validModes.end()) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG,
               "[Check][DumpConfig]dump_mode value[%s] error in config, only supports input/output/all",
               dumpMode.c_str());
        REPORT_INNER_ERR_MSG("E19999",
                            "dump_mode value[%s] error in config, only supports input/output/all",
                            dumpMode.c_str());
        return false;
    }

    return true;
}

bool DumpConfigValidator::ValidateDumpLevel(const nlohmann::json &jsDumpConfig) {
    std::string dumpLevel = GE_DUMP_LEVEL_DEFAULT;
    if (jsDumpConfig.contains(GE_DUMP_LEVEL)) {
        dumpLevel = jsDumpConfig[GE_DUMP_LEVEL].get<std::string>();
    }

    const std::vector<std::string> validLevels = {
        GE_DUMP_LEVEL_OP, GE_DUMP_LEVEL_KERNEL, GE_DUMP_LEVEL_ALL
    };

    if (std::find(validLevels.begin(), validLevels.end(), dumpLevel) == validLevels.end()) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG,
               "[Check][DumpConfig]dump_level value[%s] error in config, only supports op/kernel/all",
               dumpLevel.c_str());
        REPORT_INNER_ERR_MSG("E19999",
                            "dump_level value[%s] error in config, only supports op/kernel/all",
                            dumpLevel.c_str());
        return false;
    }

    return true;
}

bool DumpConfigValidator::ValidateOtherDumpConfigs(const nlohmann::json &jsDumpConfig) {
    return CheckDumpStep(jsDumpConfig) &&
           CheckDumplist(jsDumpConfig, GetDumpLevel(jsDumpConfig)) &&
           DumpStatsCheck(jsDumpConfig);
}

std::string DumpConfigValidator::GetDumpLevel(const nlohmann::json &jsDumpConfig) {
    if (jsDumpConfig.contains(GE_DUMP_LEVEL)) {
        return jsDumpConfig[GE_DUMP_LEVEL].get<std::string>();
    }
    return GE_DUMP_LEVEL_DEFAULT;
}

bool DumpConfigValidator::NeedDump(const DumpConfig &cfg) {
    if ((!cfg.dump_list.empty()) ||
        (cfg.dump_op_switch == GE_DUMP_STATUS_ON) ||
        (cfg.dump_debug == GE_DUMP_STATUS_ON) ||
        (!cfg.dump_exception.empty())) {
        return true;
    }
    GELOGI("no need to dump anything after check dump config");
    return false;
}

bool DumpConfigValidator::ParseDumpConfig(const char* dumpData, int32_t size, DumpConfig& dumpConfig) {
    if (dumpData == nullptr || size <= 0) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Parse][DumpConfig]Invalid dump data: data=%p, size=%d", dumpData, size);
        REPORT_INNER_ERR_MSG("E19999", "Invalid dump data: data=%p, size=%d", dumpData, size);
        return false;
    }
    try {
        std::string configStr(dumpData, size);
        GELOGD("Parsing JSON string: %s", configStr.c_str());
        nlohmann::json js = nlohmann::json::parse(configStr);
        if (!IsValidDumpConfig(js)) {
            GELOGI("Dump config validation failed");
            return false;
        }
        const auto& dumpJson = js[GE_DUMP];
        return ParseDumpConfigFromJson(dumpJson, dumpConfig);
    } catch (const std::exception& e) {
        GELOGE(INTERNAL_ERROR, "[Parse][DumpConfig]Parse dump config from memory failed, error: %s", e.what());
        REPORT_INNER_ERR_MSG("E19999", "Parse dump config from memory failed, error: %s", e.what());
        return false;
    }
}

bool DumpConfigValidator::CheckDumpPath(const nlohmann::json &jsDumpConfig) {
    GELOGI("start to execute CheckDumpPath.");

    DumpConfig config;
    if (jsDumpConfig.contains(GE_DUMP_PATH)) {
        config.dump_path = jsDumpConfig[GE_DUMP_PATH].get<std::string>();
    }

    if (config.dump_path.length() > MAX_DUMP_PATH_LENGTH) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpPath]the length[%zu] of dump_path is larger than MAX_DUMP_PATH_LENGTH[%u]",
               config.dump_path.length(), MAX_DUMP_PATH_LENGTH);
        REPORT_INNER_ERR_MSG("E19999", "the length[%zu] of dump_path is larger than MAX_DUMP_PATH_LENGTH[%u]",
                            config.dump_path.length(), MAX_DUMP_PATH_LENGTH);
        return false;
    }

    const size_t colonPos = config.dump_path.find_first_of(":");
    bool isCutDumpPathFlag = CheckIpAddress(config);
    if (isCutDumpPathFlag) {
        config.dump_path = config.dump_path.substr(colonPos + 1U);
        if (!IsValidDirStr(config.dump_path)) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpPath]dump_path[%s] contains invalid characters after IP address",
                   config.dump_path.c_str());
            REPORT_INNER_ERR_MSG("E19999", "dump_path[%s] contains invalid characters after IP address",
                                config.dump_path.c_str());
            return false;
        }
    } else {
        if (!IsDumpPathValid(config.dump_path)) {
            return false;
        }
    }

    GELOGI("successfully execute CheckDumpPath to verify dump_path field, it's valid.");
    return true;
}

bool DumpConfigValidator::CheckIpAddress(const DumpConfig &config) {
    GELOGI("start to execute CheckIpAddress.");

    const size_t colonPos = config.dump_path.find_first_of(":");
    if (colonPos == std::string::npos) {
        GELOGI("the dump_path field does not contain ip address.");
        return false;
    }

    if ((colonPos + 1U) == config.dump_path.size()) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG,
               "[Check][DumpPath]dump_path field is invalid, no path after IP address");
        REPORT_INNER_ERR_MSG("E19999",
                            "dump_path field is invalid, no path after IP address");
        return false;
    }

    GELOGI("dump_path field contains ip address.");
    const std::string ipAddress = config.dump_path.substr(0U, colonPos);

    return ValidateIpAddress(ipAddress);
}

bool DumpConfigValidator::ValidateIpAddress(const std::string& ipAddress) {
    const std::vector<std::string> ipRet = Split(ipAddress, ".");
    if (ipRet.size() != static_cast<size_t>(MAX_IPV4_ADDRESS_LENGTH)) {
        GELOGD("ip address[%s] does not comply with the ipv4 rule", ipAddress.c_str());
        return false;
    }

    try {
        for (const auto &ret : ipRet) {
            const int32_t val = std::stoi(ret);
            if ((val < 0) || (val > MAX_IPV4_ADDRESS_VALUE)) {
                GELOGD("ip address[%s] is invalid in dump_path field", ipAddress.c_str());
                return false;
            }
        }
        GELOGI("ip address[%s] is valid in dump_path field.", ipAddress.c_str());
        return true;
    } catch (...) {
        GELOGD("ip address[%s] can not convert to digital in dump_path field", ipAddress.c_str());
        return false;
    }
}

bool DumpConfigValidator::CheckDumpStep(const nlohmann::json &jsDumpConfig) {
    GELOGI("start to execute CheckDumpStep");

    std::string dumpStep = GE_DUMP_STEP_DEFAULT;
    if (jsDumpConfig.contains(GE_DUMP_STEP)) {
        dumpStep = jsDumpConfig[GE_DUMP_STEP].get<std::string>();
    }

    if (dumpStep.empty() || dumpStep == GE_DUMP_STEP_DEFAULT) {
        return true;
    }

    std::vector<std::string> matchVecs = Split(dumpStep, "|");
    if (matchVecs.size() > 100U) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpStep]dump_step value:%s, only support dump <= 100 sets of data.", dumpStep.c_str());
        REPORT_INNER_ERR_MSG("E19999", "dump_step value:%s, only support dump <= 100 sets of data.", dumpStep.c_str());
        return false;
    }

    for (const auto &it : matchVecs) {
        std::vector<std::string> steps = Split(it, "-");
        if (steps.size() > 2U) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpStep]dump_step value:%s, step is not a range or a digit, correct example:'3|5-10'.",
                   dumpStep.c_str());
            REPORT_INNER_ERR_MSG("E19999", "dump_step value:%s, step is not a range or a digit, correct example:'3|5-10'.",
                                dumpStep.c_str());
            return false;
        }

        for (const auto &step : steps) {
            if (!std::all_of(step.begin(), step.end(), ::isdigit)) {
                GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpStep]dump_step value:%s, step is not a digit, correct example:'3|5-10'.",
                       dumpStep.c_str());
                REPORT_INNER_ERR_MSG("E19999", "dump_step value:%s, step is not a digit, correct example:'3|5-10'.",
                                    dumpStep.c_str());
                return false;
            }
        }
    }

    return true;
}

bool DumpConfigValidator::CheckDumplist(const nlohmann::json &jsDumpConfig, const std::string& dumpLevel) {
    GELOGI("start to execute CheckDumpListValidity.");

    std::vector<ModelDumpConfig> dumpList;
    if (jsDumpConfig.contains(GE_DUMP_LIST)) {
        const auto& dumpListJson = jsDumpConfig[GE_DUMP_LIST];
        if (dumpListJson.is_array()) {
            if (dumpListJson.empty()) {
                GELOGI("dump_list is empty array, allow dump to proceed");
                return true;
            }
            bool isEmptyObjectArray = true;
            for (const auto& item : dumpListJson) {
                if (item.is_object() && !item.empty()) {
                    isEmptyObjectArray = false;
                    break;
                }
            }

            if (isEmptyObjectArray) {
                GELOGI("dump_list contains only empty objects, allow dump to proceed");
                return true;
            }

            dumpList = dumpListJson.get<std::vector<ModelDumpConfig>>();
        }
    }

    std::string dumpOpSwitch = GE_DUMP_OP_SWITCH_DEFAULT;
    if (jsDumpConfig.contains(GE_DUMP_OP_SWITCH)) {
        dumpOpSwitch = jsDumpConfig[GE_DUMP_OP_SWITCH].get<std::string>();
    }

    if (!CheckDumpOpSwitch(dumpOpSwitch)) {
        return false;
    }

    if (!CheckDumpOpNameRange(dumpList, dumpLevel)) {
        return false;
    }

    GELOGI("end to check the validity of dump_list and dump_op_switch field.");
    return true;
}

bool DumpConfigValidator::CheckDumpOpSwitch(const std::string& dumpOpSwitch) {
    if ((dumpOpSwitch != GE_DUMP_STATUS_ON) &&
        (dumpOpSwitch != GE_DUMP_STATUS_OFF)) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpOpSwitch]dump_op_switch value[%s] is invalid in config, "
               "only supports on/off", dumpOpSwitch.c_str());
        REPORT_INNER_ERR_MSG("E19999", "dump_op_switch value[%s] is invalid in config, only supports on/off",
                            dumpOpSwitch.c_str());
        return false;
    }
    return true;
}

bool DumpConfigValidator::CheckDumpOpNameRange(const std::vector<ModelDumpConfig> &dumpConfigList,
                                             const std::string& dumpLevel) {
    GELOGI("start to execute CheckDumpOpNameRange");
    for (const auto &dumpConfig : dumpConfigList) {
        if (!dumpConfig.dump_op_ranges.empty()) {
            if (dumpLevel != GE_DUMP_LEVEL_OP) {
                GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpOpNameRange]dump level only support op when op name range is enable.");
                REPORT_INNER_ERR_MSG("E19999", "dump level only support op when op name range is enable.");
                return false;
            }

            if (dumpConfig.model_name.empty()) {
                GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpOpNameRange]model name cannot be empty when op name range is enable.");
                REPORT_INNER_ERR_MSG("E19999", "model name cannot be empty when op name range is enable.");
                return false;
            }

            for (const auto &range : dumpConfig.dump_op_ranges) {
                if (range.first.empty() || range.second.empty()) {
                    GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpOpNameRange]op name range is imcomplete, "
                           "op name range begin[%s] end[%s].", range.first.c_str(), range.second.c_str());
                    REPORT_INNER_ERR_MSG("E19999", "op name range is imcomplete, op name range begin[%s] end[%s]",
                                        range.first.c_str(), range.second.c_str());
                    return false;
                }
            }
        }
    }

    GELOGI("successfully execute CheckDumpOpNameRange.");
    return true;
}

bool DumpConfigValidator::IsDumpPathValid(const std::string& dumpPath) {
    if (dumpPath.empty()) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpPath]dump_path is empty");
        REPORT_INNER_ERR_MSG("E19999", "dump_path is empty");
        return false;
    }

    if (dumpPath.length() > MAX_DUMP_PATH_LENGTH) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpPath]dump_path length %zu exceeds maximum %u",
               dumpPath.length(), MAX_DUMP_PATH_LENGTH);
        REPORT_INNER_ERR_MSG("E19999", "dump_path length %zu exceeds maximum %u",
                            dumpPath.length(), MAX_DUMP_PATH_LENGTH);
        return false;
    }

    if (!IsValidDirStr(dumpPath)) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpPath]dump_path[%s] contains invalid characters", dumpPath.c_str());
        REPORT_INNER_ERR_MSG("E19999", "dump_path[%s] contains invalid characters", dumpPath.c_str());
        return false;
    }

    return true;
}

bool DumpConfigValidator::IsValidDirStr(const std::string &dumpPath) {
    const std::string pathWhiteList = "-=[];\\,./!@#$%^&*()_+{}:?";
    const size_t len = dumpPath.length();

    for (size_t i = 0U; i < len; ++i) {
        const char c = dumpPath[i];
        if (!std::islower(c) && !std::isupper(c) && !std::isdigit(c) &&
            (pathWhiteList.find(c) == std::string::npos)) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpPath]invalid dump_path [%s] in dump config at %zu", dumpPath.c_str(), i);
            REPORT_INNER_ERR_MSG("E19999", "invalid dump_path [%s] in dump config at %zu", dumpPath.c_str(), i);
            return false;
        }
    }

    GELOGI("the dump result path[%s] is valid.", dumpPath.c_str());
    return true;
}


bool DumpConfigValidator::DumpDebugCheck(const nlohmann::json &jsDumpConfig) {
    GELOGI("start to execute DumpDebugCheck");

    std::string dumpPath;
    if (jsDumpConfig.contains(GE_DUMP_PATH)) {
        dumpPath = jsDumpConfig[GE_DUMP_PATH].get<std::string>();
    } else {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpDebug]dump_path field does not exist in dump config");
        REPORT_INNER_ERR_MSG("E19999", "dump_path field does not exist in dump config");
        return false;
    }

    if (dumpPath.empty() || (!CheckDumpPath(jsDumpConfig))) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpDebug]dump path is invalid in dump config");
        REPORT_INNER_ERR_MSG("E19999", "dump path is invalid in dump config");
        return false;
    }

    if (jsDumpConfig.contains(GE_DUMP_LIST)) {
        std::vector<ModelDumpConfig> dumpList;
        dumpList = jsDumpConfig[GE_DUMP_LIST].get<std::vector<ModelDumpConfig>>();
        if (!dumpList.empty()) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpDebug]nonsupport dump model when open dump overflow detection");
            REPORT_INNER_ERR_MSG("E19999", "nonsupport dump model when open dump overflow detection");
            return false;
        }
    }

    GELOGI("successfully execute DumpDebugCheck");
    return true;
}

bool DumpConfigValidator::DumpStatsCheck(const nlohmann::json &jsDumpConfig) {
    GELOGI("start to execute DumpStatsCheck");

    if (jsDumpConfig.contains(GE_DUMP_STATS)) {
        std::vector<std::string> dumpStats = jsDumpConfig[GE_DUMP_STATS].get<std::vector<std::string>>();
        if (dumpStats.empty()) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpStats]dump_stats exists but is empty.");
            REPORT_INNER_ERR_MSG("E19999", "dump_stats exists but is empty.");
            return false;
        }

        std::string dumpData = GE_DUMP_DATA_DEFAULT;
        if (jsDumpConfig.contains(GE_DUMP_DATA)) {
            dumpData = jsDumpConfig[GE_DUMP_DATA].get<std::string>();
        }

        if (dumpData != "stats") {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpStats]nonsupport to enable dump_stats when dump_data is tensor.");
            REPORT_INNER_ERR_MSG("E19999", "nonsupport to enable dump_stats when dump_data is tensor.");
            return false;
        }
    }

    GELOGI("successfully execute DumpStatsCheck");
    return true;
}

std::vector<std::string> DumpConfigValidator::Split(const std::string &str, const char_t * const delimiter) {
    std::vector<std::string> resVec;
    if (str.empty() || delimiter == nullptr) {
        return resVec;
    }

    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, *delimiter)) {
        resVec.emplace_back(token);
    }
    return resVec;
}

bool DumpConfigValidator::ParseDumpConfigFromJson(const nlohmann::json& dumpJson, DumpConfig& dumpConfig) {
    ParseBasicConfigurations(dumpJson, dumpConfig);
    ParseComplexConfigs(dumpJson, dumpConfig);

    GELOGI("Parse dump config successfully");
    return true;
}

void DumpConfigValidator::ParseBasicConfigurations(const nlohmann::json& dumpJson, DumpConfig& dumpConfig) {
    if (dumpJson.contains(GE_DUMP_PATH)) {
        dumpConfig.dump_path = dumpJson[GE_DUMP_PATH].get<std::string>();
    }

    dumpConfig.dump_exception = GetConfigWithDefault(dumpJson, GE_DUMP_SCENE, GE_DUMP_SCENE_DEFAULT);
    dumpConfig.dump_mode = GetConfigWithDefault(dumpJson, GE_DUMP_MODE, GE_DUMP_MODE_DEFAULT);
    dumpConfig.dump_status = GetConfigWithDefault(dumpJson, GE_DUMP_STATUS, GE_DUMP_STATUS_DEFAULT);
    dumpConfig.dump_op_switch = GetConfigWithDefault(dumpJson, GE_DUMP_OP_SWITCH, GE_DUMP_OP_SWITCH_DEFAULT);
    dumpConfig.dump_debug = GetConfigWithDefault(dumpJson, GE_DUMP_DEBUG, GE_DUMP_DEBUG_DEFAULT);
    dumpConfig.dump_step = GetConfigWithDefault(dumpJson, GE_DUMP_STEP, GE_DUMP_STEP_DEFAULT);
    dumpConfig.dump_data = GetConfigWithDefault(dumpJson, GE_DUMP_DATA, GE_DUMP_DATA_DEFAULT);
    dumpConfig.dump_level = GetConfigWithDefault(dumpJson, GE_DUMP_LEVEL, GE_DUMP_LEVEL_DEFAULT);
}

Status HandleDumpExceptionConfig(DumpConfig &dumpCfg) {
    GELOGI("start to execute HandleDumpExceptionConfig");

    if (geDumpSceneExceptions.find(dumpCfg.dump_exception) == geDumpSceneExceptions.end()) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG,
               "[Handle][DumpException]Invalid dump_scene: %s",
               dumpCfg.dump_exception.c_str());
        REPORT_INNER_ERR_MSG("E19999",
                            "Invalid dump_scene: %s",
                            dumpCfg.dump_exception.c_str());
        return FAILED;
    }

    const char* ascendWorkPath = nullptr;
    MM_SYS_GET_ENV(MM_ENV_ASCEND_WORK_PATH, ascendWorkPath);
    if (ascendWorkPath != nullptr) {
        dumpCfg.dump_path = ascendWorkPath;
        GELOGI("Get env ASCEND_WORK_PATH %s", ascendWorkPath);
    }
    dumpCfg.dump_status = GE_DUMP_STATUS_ON;

    GELOGI("Handle exception dump config successfully: scene=%s, path=%s",
           dumpCfg.dump_exception.c_str(), dumpCfg.dump_path.c_str());
    return SUCCESS;
}

Status HandleDumpDebugConfig(DumpConfig &dumpCfg) {
    GELOGI("start to execute HandleDumpDebugConfig");

    if (dumpCfg.dump_debug != GE_DUMP_STATUS_ON) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG,
               "[Handle][DumpDebug]dump_debug is not ON: %s",
               dumpCfg.dump_debug.c_str());
        REPORT_INNER_ERR_MSG("E19999",
                            "dump_debug is not ON: %s",
                            dumpCfg.dump_debug.c_str());
        return FAILED;
    }

    dumpCfg.dump_status = GE_DUMP_STATUS_OFF;

    GELOGI("Handle debug dump config successfully: path=%s, step=%s",
           dumpCfg.dump_path.c_str(), dumpCfg.dump_step.c_str());
    return SUCCESS;
}

bool DumpConfigValidator::CheckDumpSceneSwitch(const nlohmann::json &jsDumpConfig, std::string &dumpScene) {
    GELOGI("start to execute CheckDumpSceneSwitch.");

    if (!jsDumpConfig.contains(GE_DUMP_SCENE)) {
        GELOGI("dump_scene does not exist, no need to check.");
        return true;
    }
    dumpScene = jsDumpConfig[GE_DUMP_SCENE].get<std::string>();
    if ((geDumpSceneExceptions.find(dumpScene) == geDumpSceneExceptions.end()) && (dumpScene != GE_DUMP_SCENE_WATCHER)) {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpScene]Invalid value of dump_scene[%s]. Supported: lite_exception/aic_err_brief_dump/aic_err_norm_dump/aic_err_detail_dump/watcher",
               dumpScene.c_str());
        REPORT_INNER_ERR_MSG("E19999", "Invalid value of dump_scene[%s].",
                            dumpScene.c_str());
        return false;
    }

    if (jsDumpConfig.contains(GE_DUMP_DEBUG)) {
        std::string dumpDebug = jsDumpConfig[GE_DUMP_DEBUG].get<std::string>();
        if (dumpDebug == GE_DUMP_STATUS_ON) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpScene]dump_scene is %s when dump_debug is on is unsupported.",
                   dumpScene.c_str());
            REPORT_INNER_ERR_MSG("E19999", "dump_scene is %s when dump_debug is on is unsupported.",
                                dumpScene.c_str());
            return false;
        }
    }

    if (jsDumpConfig.contains(GE_DUMP_OP_SWITCH)) {
        std::string dumpOpSwitch = jsDumpConfig[GE_DUMP_OP_SWITCH].get<std::string>();
        if (dumpOpSwitch == GE_DUMP_STATUS_ON) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpScene]dump_scene is %s when dump_op_switch is on is unsupported.",
                   dumpScene.c_str());
            REPORT_INNER_ERR_MSG("E19999", "dump_scene is %s when dump_op_switch is on is unsupported.",
                                dumpScene.c_str());
            return false;
        }
    }

    if (jsDumpConfig.contains(GE_DUMP_MODE)) {
        std::string dumpMode = jsDumpConfig[GE_DUMP_MODE].get<std::string>();
        if ((dumpMode != GE_DUMP_MODE_OUTPUT) && (dumpScene == GE_DUMP_SCENE_WATCHER)) {
            GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Check][DumpScene]dump_mode only supports output when dump_scene is %s.",
                   dumpScene.c_str());
            REPORT_INNER_ERR_MSG("E19999", "dump_mode only supports output when dump_scene is %s.",
                                dumpScene.c_str());
            return false;
        }
    }

    GELOGI("Valid dump scene config: scene=%s", dumpScene.c_str());
    return true;
}

std::string DumpConfigValidator::GetConfigWithDefault(const nlohmann::json& json,
                                                     const std::string& key,
                                                     const std::string& defaultValue) {
    return json.contains(key) ? json[key].get<std::string>() : defaultValue;
}

void DumpConfigValidator::ParseComplexConfigs(const nlohmann::json& dumpJson, DumpConfig& dumpConfig) {
    if (dumpJson.contains(GE_DUMP_STATS) && dumpJson[GE_DUMP_STATS].is_array()) {
        ParseStringArray(dumpJson[GE_DUMP_STATS], dumpConfig.dump_stats);
    }

    if (dumpJson.contains(GE_DUMP_LIST) && dumpJson[GE_DUMP_LIST].is_array()) {
        ParseModelDumpConfigList(dumpJson[GE_DUMP_LIST], dumpConfig.dump_list);
    }
    dumpConfig.dump_status = ((dumpConfig.dump_level == GE_DUMP_LEVEL_OP) ||
                            (dumpConfig.dump_level == GE_DUMP_LEVEL_ALL))
                            ? GE_DUMP_STATUS_ON : GE_DUMP_STATUS_OFF;
}

void DumpConfigValidator::ParseStringArray(const nlohmann::json& jsonArray,
                                         std::vector<std::string>& output) {
    output.clear();
    for (const auto& item : jsonArray) {
        output.push_back(item.get<std::string>());
    }
}

void DumpConfigValidator::ParseModelDumpConfigList(const nlohmann::json& jsonArray,
                                                 std::vector<ModelDumpConfig>& output) {
    output.clear();

    if (jsonArray.empty()) {
        GELOGI("dump_list is empty array, add an empty ModelDumpConfig");
        output.push_back(ModelDumpConfig());
        return;
    }

    for (const auto& modelJson : jsonArray) {
        ModelDumpConfig modelConfig;
        if (ParseModelDumpConfig(modelJson, modelConfig)) {
            output.push_back(modelConfig);
        }
    }
}

bool DumpConfigValidator::CheckDumpModelConfig(const nlohmann::json& modelJson) {
    if (modelJson.contains(GE_DUMP_MODEL_NAME)) {
        const auto& modelNameField = modelJson[GE_DUMP_MODEL_NAME];
        if (modelNameField.is_string() && modelNameField.get<std::string>().empty()) {
            GELOGW("[Check][modelName]the modelName field is null");
            return false;
        }
    }

    if (modelJson.contains(GE_DUMP_LAYER)) {
        const auto& layerField = modelJson[GE_DUMP_LAYER];
        if (layerField.is_array() && layerField.empty()) {
            std::string modelName = modelJson.contains(GE_DUMP_MODEL_NAME) 
                ? modelJson[GE_DUMP_MODEL_NAME].get<std::string>() 
                : "unknown";
            GELOGW("[Check][Layer]Layer field is null in model %s", modelName.c_str());
            return false;
        }
    }

    return true;
}

bool DumpConfigValidator::ParseModelDumpConfig(const nlohmann::json& modelJson, ModelDumpConfig& modelConfig) {
    if (!CheckDumpModelConfig(modelJson)) {
        return false;
    }

    if (modelJson.contains(GE_DUMP_MODEL_NAME)) {
        modelConfig.model_name = modelJson[GE_DUMP_MODEL_NAME].get<std::string>();
    }

    if (modelJson.contains(GE_DUMP_LAYER) && modelJson[GE_DUMP_LAYER].is_array()) {
        modelConfig.layers.clear();
        for (const auto& layer : modelJson[GE_DUMP_LAYER]) {
            modelConfig.layers.push_back(layer.get<std::string>());
        }
    }

    if (modelJson.contains(GE_DUMP_WATCHER_NODES) && modelJson[GE_DUMP_WATCHER_NODES].is_array()) {
        modelConfig.watcher_nodes.clear();
        for (const auto& node : modelJson[GE_DUMP_WATCHER_NODES]) {
            modelConfig.watcher_nodes.push_back(node.get<std::string>());
        }
    }

    if (modelJson.contains(GE_DUMP_OPTYPE_BLACKLIST) && modelJson[GE_DUMP_OPTYPE_BLACKLIST].is_array()) {
        modelConfig.optype_blacklist.clear();
        for (const auto& blacklistJson : modelJson[GE_DUMP_OPTYPE_BLACKLIST]) {
            DumpBlacklist blacklist;
            ParseBlacklist(blacklistJson, blacklist);
            modelConfig.optype_blacklist.push_back(blacklist);
        }
    }

    if (modelJson.contains(GE_DUMP_OPNAME_BLACKLIST) && modelJson[GE_DUMP_OPNAME_BLACKLIST].is_array()) {
        modelConfig.opname_blacklist.clear();
        for (const auto& blacklistJson : modelJson[GE_DUMP_OPNAME_BLACKLIST]) {
            DumpBlacklist blacklist;
            ParseBlacklist(blacklistJson, blacklist);
            modelConfig.opname_blacklist.push_back(blacklist);
        }
    }

    if (modelJson.contains(GE_DUMP_OPNAME_RANGE) && modelJson[GE_DUMP_OPNAME_RANGE].is_array()) {
        modelConfig.dump_op_ranges.clear();
        for (const auto& rangeJson : modelJson[GE_DUMP_OPNAME_RANGE]) {
            if (rangeJson.contains(GE_DUMP_OPNAME_RANGE_BEGIN) && rangeJson.contains(GE_DUMP_OPNAME_RANGE_END)) {
                std::string begin = rangeJson[GE_DUMP_OPNAME_RANGE_BEGIN].get<std::string>();
                std::string end = rangeJson[GE_DUMP_OPNAME_RANGE_END].get<std::string>();
                modelConfig.dump_op_ranges.emplace_back(begin, end);
            }
        }
    }
    return true;
}

void DumpConfigValidator::ParseBlacklist(const nlohmann::json& blacklistJson, DumpBlacklist& blacklist) {
    if (blacklistJson.contains(GE_DUMP_BLACKLIST_NAME)) {
        blacklist.name = blacklistJson[GE_DUMP_BLACKLIST_NAME].get<std::string>();
    }

    if (blacklistJson.contains(GE_DUMP_BLACKLIST_POS) && blacklistJson[GE_DUMP_BLACKLIST_POS].is_array()) {
        blacklist.pos.clear();
        for (const auto& pos : blacklistJson[GE_DUMP_BLACKLIST_POS]) {
            blacklist.pos.push_back(pos.get<std::string>());
        }
    }
}

DumpCallbackManager& DumpCallbackManager::GetInstance() {
    static DumpCallbackManager instance;
    return instance;
}

bool DumpCallbackManager::RegisterDumpCallbacks(uint32_t module_id) const {
    int32_t result = Adx::AdumpRegisterCallback(
        module_id,
        reinterpret_cast<Adx::AdumpCallback>(EnableDumpCallback),\
        reinterpret_cast<Adx::AdumpCallback>(DisableDumpCallback));
    if (result != ADUMP_SUCCESS) {
        GELOGE(INTERNAL_ERROR, "[Register][DumpCallbacks]Register dump callbacks failed, result: %d", result);
        REPORT_INNER_ERR_MSG("E19999", "Register dump callbacks failed, result: %d", result);
        return false;
    }

    GELOGI("Register dump callbacks successfully for module_id: %u", module_id);
    return true;
}

bool DumpCallbackManager::IsEnableExceptionDumpBySwitch(uint64_t dumpSwitch) {
    return ((dumpSwitch & AIC_ERR_NORM_DUMP_BIT) != 0) || 
           ((dumpSwitch & AIC_ERR_BRIEF_DUMP_BIT) != 0);
}

std::string DumpCallbackManager::BuildExceptionDumpJsonBySwitch(uint64_t dumpSwitch) {
    std::string exceptionScene;
    
    if ((dumpSwitch & AIC_ERR_NORM_DUMP_BIT) != 0) {
        exceptionScene = "aic_err_norm_dump";
    } else if ((dumpSwitch & AIC_ERR_BRIEF_DUMP_BIT) != 0) {
        exceptionScene = "aic_err_brief_dump";
    } else {
        return "";
    }

    nlohmann::json js;
    nlohmann::json jsDump;
    
    jsDump[GE_DUMP_SCENE] = exceptionScene;
    jsDump[GE_DUMP_PATH] = "";
    
    js[GE_DUMP] = jsDump;
    
    return js.dump();
}

bool DumpCallbackManager::ProcessExceptionDumpBySwitch(uint64_t dumpSwitch) {
    std::string configStr = BuildExceptionDumpJsonBySwitch(dumpSwitch);
    if (configStr.empty()) {
        GELOGE(INTERNAL_ERROR, "[Process][ExceptionDump]Failed to build exception dump config by switch");
        REPORT_INNER_ERR_MSG("E19999", "Failed to build exception dump config by switch");
        return false;
    }
    
    GELOGI("Processing exception dump by switch, config: %s", configStr.c_str());

    Status ret = HandleEnableDump(configStr.c_str(), static_cast<int32_t>(configStr.size()));
    if (ret == SUCCESS) {
        GELOGI("Enable exception dump by switch processed successfully");
        return true;
    } else {
        GELOGE(ret, "[Process][ExceptionDump]Enable exception dump by switch failed, ret=%u", ret);
        REPORT_INNER_ERR_MSG("E19999", "Enable exception dump by switch failed, ret=%u", ret);
        return false;
    }
}

int32_t DumpCallbackManager::EnableDumpCallback(uint64_t dumpSwitch, const char* dumpData, int32_t size) {
    GELOGI("Enable dump callback triggered, dumpSwitch=%lu, size=%d", dumpSwitch, size);

    if ((dumpData == nullptr || size <= 0) && IsEnableExceptionDumpBySwitch(dumpSwitch)) {
        GELOGI("dumpData is null or size is invalid (size=%d) but exception dump bit is set, processing exception dump by switch", size);
        return ProcessExceptionDumpBySwitch(dumpSwitch) ? ADUMP_SUCCESS : ADUMP_FAILED;
    }

    Status ret = HandleEnableDump(dumpData, size);
    if (ret == SUCCESS) {
        GELOGI("Enable dump callback processed successfully");
        return ADUMP_SUCCESS;
    } else {
        GELOGE(ret, "[Handle][EnableDump]Enable dump callback failed, ret=%u", ret);
        REPORT_INNER_ERR_MSG("E19999", "Enable dump callback failed, ret=%u", ret);
        return ADUMP_FAILED;
    }
}

int32_t DumpCallbackManager::DisableDumpCallback(uint64_t dumpSwitch, const char* dumpData, int32_t size) {
    (void)dumpData;
    GELOGI("Disable dump callback triggered, dumpSwitch=%lu, size=%d", dumpSwitch, size);

    Status ret = HandleDisableDump();
    if (ret == SUCCESS) {
        GELOGI("Disable dump callback processed successfully");
        return ADUMP_SUCCESS;
    } else {
        GELOGE(ret, "[Handle][DisableDump]Disable dump callback failed, ret=%u", ret);
        REPORT_INNER_ERR_MSG("E19999", "Disable dump callback failed, ret=%u", ret);
        return ADUMP_FAILED;
    }
}

Status DumpCallbackManager::HandleEnableDump(const char* dumpData, int32_t size) {
    if (dumpData == nullptr || size <= 0) {
        GELOGW("[Handle][EnableDump]Invalid dump data in enable callback: data=%p, size=%d",
               dumpData, size);
        return SUCCESS;
    }

    DumpConfig dumpConfig;
    if (DumpConfigValidator::ParseDumpConfig(dumpData, size, dumpConfig)) {
        if (!DumpConfigValidator::NeedDump(dumpConfig)) {
            GELOGI("No need to dump based on config");
            return SUCCESS;
        }

        Status result = SUCCESS;

        if (geDumpSceneExceptions.find(dumpConfig.dump_exception) != geDumpSceneExceptions.end()) {
            result = HandleDumpExceptionConfig(dumpConfig);
        }
        else if (dumpConfig.dump_debug == GE_DUMP_STATUS_ON) {
            result = HandleDumpDebugConfig(dumpConfig);
        }
        if (result != SUCCESS) {
            return result;
        }
        ge::GeExecutor geExecutor;
        const ge::Status geRet = geExecutor.SetDump(dumpConfig);
        if (geRet == SUCCESS) {
            GELOGI("Enable dump configuration successfully");
            return SUCCESS;
        } else {
            GELOGE(geRet, "[Handle][EnableDump]Enable dump configuration failed, ret=%u", geRet);
            REPORT_INNER_ERR_MSG("E19999", "Enable dump configuration failed, ret=%u", geRet);
            return geRet;
        }
    } else {
        GELOGE(ACL_GE_INVALID_DUMP_CONFIG, "[Handle][EnableDump]Parse dump config failed in enable callback");
        REPORT_INNER_ERR_MSG("E19999", "Parse dump config failed in enable callback");
        return FAILED;
    }
}

Status DumpCallbackManager::HandleDisableDump() {
    DumpConfig disableConfig;
    disableConfig.dump_status = GE_DUMP_STATUS_OFF;
    disableConfig.dump_debug = GE_DUMP_STATUS_OFF;

    ge::GeExecutor geExecutor;
    const ge::Status geRet = geExecutor.SetDump(disableConfig);
    if (geRet == SUCCESS) {
        GELOGI("Disable dump configuration successfully");
        return SUCCESS;
    } else {
        GELOGE(geRet, "[Handle][DisableDump]Disable dump configuration failed, ret=%u", geRet);
        REPORT_INNER_ERR_MSG("E19999", "Disable dump configuration failed, ret=%u", geRet);
        return geRet;
    }
}

} // namespace ge