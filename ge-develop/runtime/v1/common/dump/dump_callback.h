/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_DUMP_DUMP_CALLBACK_H_
#define GE_COMMON_DUMP_DUMP_CALLBACK_H_

#include <cstdint>
#include <string>
#include <set>
#include <vector>
#include "framework/common/ge_types.h"
#include "nlohmann/json.hpp"

namespace ge {

const uint32_t ACL_GE_INVALID_DUMP_CONFIG = 100009U;
constexpr int32_t ADUMP_SUCCESS = 0;
constexpr int32_t ADUMP_FAILED = -1;
constexpr uint32_t MAX_DUMP_PATH_LENGTH = 512;
constexpr int32_t MAX_IPV4_ADDRESS_LENGTH = 4;
constexpr int32_t MAX_IPV4_ADDRESS_VALUE = 255;
constexpr int32_t DECIMAL = 10;
const uint64_t AIC_ERR_BRIEF_DUMP_BIT = 0x4;
const uint64_t AIC_ERR_NORM_DUMP_BIT = 0x8;

const std::string GE_DUMP_MODE_INPUT = "input";
const std::string GE_DUMP_MODE_OUTPUT = "output";
const std::string GE_DUMP_MODE_ALL = "all";
const std::string GE_DUMP_STATUS_ON = "on";
const std::string GE_DUMP_STATUS_OFF = "off";
const std::string GE_DUMP_LEVEL_OP = "op";
const std::string GE_DUMP_LEVEL_KERNEL = "kernel";
const std::string GE_DUMP_LEVEL_ALL = "all";
const std::string GE_DUMP = "dump";

const std::string GE_DUMP_PATH = "dump_path";
const std::string GE_DUMP_MODE = "dump_mode";
const std::string GE_DUMP_STATUS = "dump_status";
const std::string GE_DUMP_OP_SWITCH = "dump_op_switch";
const std::string GE_DUMP_DEBUG = "dump_debug";
const std::string GE_DUMP_STEP = "dump_step";
const std::string GE_DUMP_DATA = "dump_data";
const std::string GE_DUMP_LEVEL = "dump_level";
const std::string GE_DUMP_STATS = "dump_stats";
const std::string GE_DUMP_LIST = "dump_list";
const std::string GE_DUMP_MODEL_NAME = "model_name";
const std::string GE_DUMP_LAYER = "layer";
const std::string GE_DUMP_WATCHER_NODES = "watcher_nodes";
const std::string GE_DUMP_OPTYPE_BLACKLIST = "optype_blacklist";
const std::string GE_DUMP_OPNAME_BLACKLIST = "opname_blacklist";
const std::string GE_DUMP_BLACKLIST_NAME = "name";
const std::string GE_DUMP_BLACKLIST_POS = "pos";
const std::string GE_DUMP_OPNAME_RANGE = "opname_range";
const std::string GE_DUMP_OPNAME_RANGE_BEGIN = "begin";
const std::string GE_DUMP_OPNAME_RANGE_END = "end";
const std::string GE_DUMP_SCENE = "dump_scene";
const std::string GE_DUMP_LITE_EXCEPTION = "lite_exception";
const std::string GE_DUMP_EXCEPTION_AIC_ERR_BRIEF = "aic_err_brief_dump";    // l0 exception dump
const std::string GE_DUMP_EXCEPTION_AIC_ERR_NORM = "aic_err_norm_dump";      // l1 exception dump
const std::string GE_DUMP_EXCEPTION_AIC_ERR_DETAIL = "aic_err_detail_dump";  // npu coredump
const std::string GE_DUMP_SCENE_WATCHER = "watcher";

const std::set<std::string> geDumpSceneExceptions = {
    GE_DUMP_LITE_EXCEPTION, GE_DUMP_EXCEPTION_AIC_ERR_BRIEF,
    GE_DUMP_EXCEPTION_AIC_ERR_NORM, GE_DUMP_EXCEPTION_AIC_ERR_DETAIL,
};

const std::string GE_DUMP_MODE_DEFAULT = "output";
const std::string GE_DUMP_STATUS_DEFAULT = "on";
const std::string GE_DUMP_OP_SWITCH_DEFAULT = "off";
const std::string GE_DUMP_DEBUG_DEFAULT = "off";
const std::string GE_DUMP_STEP_DEFAULT = "";
const std::string GE_DUMP_DATA_DEFAULT = "tensor";
const std::string GE_DUMP_LEVEL_DEFAULT = "all";
const std::string GE_DUMP_SCENE_DEFAULT = "";

std::string GetCfgStrByKey(const nlohmann::json &js, const std::string &key);
bool ContainKey(const nlohmann::json &js, const std::string &key);

void from_json(const nlohmann::json &js, DumpBlacklist &blacklist);
void from_json(const nlohmann::json &js, ModelDumpConfig &info);

class DumpConfigValidator {
public:
    static bool IsValidDumpConfig(const nlohmann::json &js);
    static bool NeedDump(const DumpConfig &cfg);
    static bool ParseDumpConfig(const char* dumpData, int32_t size, DumpConfig& dumpConfig);

private:
    static bool CheckDumpSceneSwitch(const nlohmann::json &jsDumpConfig, std::string &dumpScene);
    static bool CheckDumpPath(const nlohmann::json &jsDumpConfig);
    static bool CheckDumpStep(const nlohmann::json &jsDumpConfig);
    static bool CheckDumplist(const nlohmann::json &jsDumpConfig, const std::string& dumpLevel);
    static bool CheckDumpOpSwitch(const std::string& dumpOpSwitch);
    static bool CheckDumpOpNameRange(const std::vector<ModelDumpConfig> &dumpConfigList,
                                   const std::string& dumpLevel);
    static bool IsValidDirStr(const std::string &dumpPath);
    static bool CheckIpAddress(const DumpConfig &config);
    static bool ValidateIpAddress(const std::string& ipAddress);
    static bool IsDumpPathValid(const std::string& dumpPath);
    static bool DumpDebugCheck(const nlohmann::json &jsDumpConfig);
    static bool DumpStatsCheck(const nlohmann::json &jsDumpConfig);

    static bool ParseDumpConfigFromJson(const nlohmann::json& dumpJson, DumpConfig& dumpConfig);
    static void ParseBasicConfigurations(const nlohmann::json& dumpJson, DumpConfig& dumpConfig);
    static std::string GetConfigWithDefault(const nlohmann::json& json,
                                           const std::string& key,
                                           const std::string& defaultValue);
    static bool CheckDumpModelConfig(const nlohmann::json& modelJson);
    static bool ParseModelDumpConfig(const nlohmann::json& modelJson, ModelDumpConfig& modelConfig);
    static void ParseBlacklist(const nlohmann::json& blacklistJson, DumpBlacklist& blacklist);
    static bool IsDumpDebugEnabled(const nlohmann::json &jsDumpConfig);
    static bool ValidateNormalDumpConfig(const nlohmann::json &jsDumpConfig);
    static bool ValidateDumpPath(const nlohmann::json &jsDumpConfig);
    static bool ValidateDumpMode(const nlohmann::json &jsDumpConfig);
    static bool ValidateDumpLevel(const nlohmann::json &jsDumpConfig);
    static bool ValidateOtherDumpConfigs(const nlohmann::json &jsDumpConfig);
    static std::string GetDumpLevel(const nlohmann::json &jsDumpConfig);

    static void ParseComplexConfigs(const nlohmann::json& dumpJson, DumpConfig& dumpConfig);
    static void ParseStringArray(const nlohmann::json& jsonArray, std::vector<std::string>& output);
    static void ParseModelDumpConfigList(const nlohmann::json& jsonArray, std::vector<ModelDumpConfig>& output);

    static std::vector<std::string> Split(const std::string &str, const char_t * const delimiter);
};

class DumpCallbackManager {
public:
    static DumpCallbackManager& GetInstance();
    bool RegisterDumpCallbacks(uint32_t module_id) const;

private:
    DumpCallbackManager() = default;
    ~DumpCallbackManager() = default;
    static bool IsEnableExceptionDumpBySwitch(uint64_t dumpSwitch);
    static std::string BuildExceptionDumpJsonBySwitch(uint64_t dumpSwitch);
    static bool ProcessExceptionDumpBySwitch(uint64_t dumpSwitch);
    static int32_t EnableDumpCallback(uint64_t dumpSwitch, const char* dumpData, int32_t size);
    static int32_t DisableDumpCallback(uint64_t dumpSwitch, const char* dumpData, int32_t size);

    static Status HandleEnableDump(const char* dumpData, int32_t size);
    static Status HandleDisableDump();
};

} // namespace ge

#endif // GE_COMMON_DUMP_DUMP_CALLBACK_H_