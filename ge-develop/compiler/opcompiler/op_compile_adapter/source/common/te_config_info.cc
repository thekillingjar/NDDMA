/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/te_config_info.h"
#include <fstream>
#include "mmpa/mmpa_api.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"
#include "ge_common/ge_api_types.h"
#include "graph/ge_context.h"

namespace te {
namespace fusion {
namespace {
const size_t kHardwareInfoLen = 2;
const std::string kParentDirRelativePath = "../";
const std::string kCompilerPkgRelativePath = "../../compiler/";
const std::string kVersionFileName = "version.info";
}
const std::map<TeConfigInfo::EnvItem, std::string> TeConfigInfo::kEnvItemKeyMap {
    {TeConfigInfo::EnvItem::Home, "HOME"},
    {TeConfigInfo::EnvItem::Path, "PATH"},
    {TeConfigInfo::EnvItem::NpuCollectPath, "NPU_COLLECT_PATH"},
    {TeConfigInfo::EnvItem::ParaDebugPath, "PARA_DEBUG_PATH"},
    {TeConfigInfo::EnvItem::TeParallelCompiler, "TE_PARALLEL_COMPILER"},
    {TeConfigInfo::EnvItem::TeNewDfxinfo, "TEFUSION_NEW_DFXINFO"},
    {TeConfigInfo::EnvItem::MinCompileResourceUsageCtrl, "MIN_COMPILE_RESOURCE_USAGE_CTRL"},
    {TeConfigInfo::EnvItem::OppPath, "ASCEND_OPP_PATH"},
    {TeConfigInfo::EnvItem::WorkPath, "ASCEND_WORK_PATH"},
    {TeConfigInfo::EnvItem::CachePath, "ASCEND_CACHE_PATH"},
    {TeConfigInfo::EnvItem::MaxOpCacheSize, "ASCEND_MAX_OP_CACHE_SIZE"},
    {TeConfigInfo::EnvItem::RemainCacheSizeRatio, "ASCEND_REMAIN_CACHE_SIZE_RATIO"},
    {TeConfigInfo::EnvItem::OpCompilerWorkPathInKernelMeta, "ASCEND_OP_COMPILER_WORK_PATH_IN_KERNEL_META"},
    {TeConfigInfo::EnvItem::HomePath, "ASCEND_HOME_PATH"},
    {TeConfigInfo::EnvItem::AscendCoreDumpSignal, "ASCEND_COREDUMP_SIGNAL"},
    {TeConfigInfo::EnvItem::AscendSaveKernelMeta, "ASCEND_OP_COMPILE_SAVE_KERNEL_META"}
};

const std::map<TeConfigInfo::ConfigEnumItem, std::tuple<int64_t, std::string, std::map<std::string, int64_t>>>
    TeConfigInfo::kConfigEnumItemKeyMap {
    {TeConfigInfo::ConfigEnumItem::CacheMode,
     {static_cast<int64_t>(CompileCacheMode::Enable), ge::OP_COMPILER_CACHE_MODE, kCompileCacheModeStrMap}},
    {TeConfigInfo::ConfigEnumItem::Deterministic, {0, ge::DETERMINISTIC, kSwitchStrMap}},
    {TeConfigInfo::ConfigEnumItem::OpDebugLevel,
     {static_cast<int64_t>(OpDebugLevel::Disable), ge::OP_DEBUG_LEVEL, kOpDebugLevelStrMap}}
};

const std::map<TeConfigInfo::ConfigStrItem, std::string> TeConfigInfo::kConfigStrItemKeyMap {
    {TeConfigInfo::ConfigStrItem::SocVersion, ge::SOC_VERSION},
    {TeConfigInfo::ConfigStrItem::ShortSocVersion, "short_soc_version"},
    {TeConfigInfo::ConfigStrItem::AiCoreNum, ge::AICORE_NUM},
    {TeConfigInfo::ConfigStrItem::BufferOptimize, ge::BUFFER_OPTIMIZE},
    {TeConfigInfo::ConfigStrItem::CoreType, ge::CORE_TYPE},
    {TeConfigInfo::ConfigStrItem::CacheDir, ge::OP_COMPILER_CACHE_DIR},
    {TeConfigInfo::ConfigStrItem::DeviceId, ge::OPTION_EXEC_DEVICE_ID},
    {TeConfigInfo::ConfigStrItem::FpCeilingMode, "ge.fpCeilingMode"},
    {TeConfigInfo::ConfigStrItem::L2Mode, "l2_mode"},
    {TeConfigInfo::ConfigStrItem::MdlBankPath, ge::MDL_BANK_PATH_FLAG},
    {TeConfigInfo::ConfigStrItem::OpBankPath, ge::OP_BANK_PATH_FLAG},
    {TeConfigInfo::ConfigStrItem::OpDebugDir, ge::DEBUG_DIR},
    {TeConfigInfo::ConfigStrItem::OpDebugConfig, "op_debug_config_te"},
    {TeConfigInfo::ConfigStrItem::HardwareInfo, "ge.hardwareInfo"}
};

TeConfigInfo::TeConfigInfo()
    : init_flag_(false), isOpCompileDisabled_(false), isUbFusionDisabled_(false) {}

TeConfigInfo::~TeConfigInfo()
{
    init_flag_ = false;
    isUbFusionDisabled_ = false;
    isOpCompileDisabled_ = false;
}

TeConfigInfo& TeConfigInfo::Instance()
{
    static TeConfigInfo te_config_info;
    return te_config_info;
}

bool TeConfigInfo::Initialize(const std::map<std::string, std::string> &options)
{
    if (init_flag_) {
        return true;
    }
    if (!InitLibPath()) {
        TE_WARNLOG("Failed to initialize current lib path.");
        return false;
    }

    InitEnvItem();
    InitOppPath();
    InitKernelMetaHashAndRelativePath();
    InitAdkAndOppVersion();

    if (!InitConfigItemsFromOptions(options)) {
        TE_WARNLOG("Failed to initialize config params from options.");
        return false;
    }

    if (!InitOrRefreshDebugDir()) {
        TE_ERRLOG("Failed to initialize debugDir from ge options.");
        return false;
    }
    init_flag_ = true;
    return true;
}

bool TeConfigInfo::InitLibPath()
{
    Dl_info dlInfo;
    TeConfigInfo &(*instance_ptr)() = &TeConfigInfo::Instance;
    if (dladdr(reinterpret_cast<void *>(instance_ptr), &dlInfo) == 0) {
        TE_WARNLOG("Failed to read the so file path.");
        return false;
    }
    std::string soPath = dlInfo.dli_fname;
    TE_DBGLOG("Current lib path from dl info is [%s].", soPath.c_str());
    if (soPath.empty()) {
        TE_WARNLOG("So file path is empty.");
        return false;
    }
    lib_path_ = RealPath(soPath);
    int32_t pos = lib_path_.rfind('/');
    if (pos < 0) {
        TE_WARNLOG("The path of current so file does not contain /.");
        return false;
    }
    lib_path_ = lib_path_.substr(0, pos + 1);
    TE_DBGLOG("Current lib real path is [%s]", lib_path_.c_str());
    return true;
}

bool TeConfigInfo::RefreshConfigItems()
{
    std::lock_guard<std::mutex> lock_guard(config_item_mutex_);
    if (!InitConfigItemsFromContext()) {
        return false;
    }
    if (!InitOrRefreshDebugDir(false)) {
        return false;
    }
    return true;
}

bool TeConfigInfo::Finalize()
{
    return true;
}

void TeConfigInfo::InitEnvItem()
{
    for (const std::pair<const EnvItem, std::string> &item : kEnvItemKeyMap) {
        char env_value[MMPA_MAX_PATH];
        INT32 ret = mmGetEnv(item.second.c_str(), env_value, MMPA_MAX_PATH);
        if (ret == EN_OK) {
            TE_DBGLOG("The value of env[%s] is [%s].", item.second.c_str(), env_value);
            std::string env_str_value = std::string(env_value);
            Trim(env_str_value);
            env_item_vec_[static_cast<size_t>(item.first)] = env_str_value;
        } else {
            TE_DBGLOG("Env[%s] is not found.", item.second.c_str());
        }
    }
    ParseCompileControlParams();
}

void TeConfigInfo::InitOppPath()
{
    oppPath_ = RealPath(GetEnvOppPath());
    TE_DBGLOG("Opp package path is [%s].", oppPath_.c_str());

    if (!oppPath_.empty()) {
        oppBinaryPath_ = RealPath(oppPath_ + "/built-in/op_impl/ai_core/tbe/kernel");
    }
    TE_DBGLOG("Opp binary kernel path is [%s].", oppBinaryPath_.c_str());
}

void TeConfigInfo::InitAdkAndOppVersion()
{
    std::string adkVersionDirPath = lib_path_ + kParentDirRelativePath;
    adkVersion_ = GetVersionFromVersionFile(adkVersionDirPath);
    if (adkVersion_.empty()) {
        adkVersionDirPath = lib_path_ + kCompilerPkgRelativePath;
        adkVersion_ = GetVersionFromVersionFile(adkVersionDirPath);
        if (adkVersion_.empty()) {
            TE_WARNLOG("ASCEND_ADK_PATH is invalid, unable to get current adk version info.");
        }
    }

    oppVersion_ = GetVersionFromVersionFile(GetEnvOppPath());
    if (oppVersion_.empty()) {
        TE_WARNLOG("ASCEND_OPP_PATH is invalid, Unable to get current opp version info.");
    }

    TE_DBGLOGF("Current adk version is [%s], current opp version is [%s].", adkVersion_.c_str(), oppVersion_.c_str());
}

std::string TeConfigInfo::GetVersionFromVersionFile(const std::string &versionFileDir)
{
    std::string file_real_path = RealPath(versionFileDir + "/" + kVersionFileName);
    if (file_real_path.empty()) {
        TE_DBGLOG("version.info with path [%s] does not exist.", versionFileDir.c_str());
        return "";
    }

    std::ifstream ifs(file_real_path);
    if (!ifs.is_open()) {
        TE_WARNLOGF("Open version file failed, it does not exist or has been opened.");
        return "";
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line.find('#') == 0) {
            continue;
        }
        size_t pos_of_equal = line.find('=');
        if (pos_of_equal == string::npos) {
            continue;
        }
        std::string key = line.substr(0, pos_of_equal);
        Trim(key);
        std::string value = line.substr(pos_of_equal + 1, line.length() - pos_of_equal - 1);
        Trim(value);
        if (!key.empty() && key == "Version") {
            ifs.close();
            return value;
        }
    }
    TE_WARNLOGF("There is no version info in version.info.");
    ifs.close();
    return "";
}

bool TeConfigInfo::InitConfigItemsFromOptions(const std::map<std::string, std::string> &options)
{
    config_enum_item_vec_.fill(0);
    for (const auto &item : kConfigEnumItemKeyMap) {
        config_enum_item_vec_[static_cast<size_t>(item.first)] = std::get<0>(item.second);
        const std::string &itemKey = std::get<1>(item.second);
        const auto iter = options.find(itemKey);
        if (iter == options.cend() || iter->second.empty()) {
            TE_DBGLOG("The value of [%s] is not found or the value is empty.", itemKey.c_str());
            continue;
        }
        const std::map<std::string, int64_t> &itemStrMap = std::get<2>(item.second);
        const auto iterValue = itemStrMap.find(iter->second);
        if (iterValue == itemStrMap.cend()) {
            // report error
            std::string configName = ge::GetContext().GetReadableName(itemKey);
            std::map<std::string, std::string> errMsgMap;
            errMsgMap.emplace("invalid_value", iter->second);
            errMsgMap.emplace("argument", configName);
            errMsgMap.emplace("valid_range", GetMapKeyToString(itemStrMap).c_str());
            TeErrMessageReport(EM_PARAMETER_INVALID_ERROR, errMsgMap);
            TE_ERRLOG("The value[%s] of param[%s] is invalid.", iter->second.c_str(), itemKey.c_str());
            return false;
        }
        TE_DBGLOG("The value of param[%s] is [%ld].", itemKey.c_str(), iterValue->second);
        config_enum_item_vec_[static_cast<size_t>(item.first)] = iterValue->second;
    }
    for (const auto &item : kConfigStrItemKeyMap) {
        const auto iter = options.find(item.second);
        if (iter != options.cend() && !iter->second.empty()) {
            config_str_item_vec_[static_cast<size_t>(item.first)] = iter->second;
            TE_DBGLOG("The value of param[%s] is [%s].", item.second.c_str(), iter->second.c_str());
        }
    }
    ParseHardwareInfo(false);
    return true;
}

bool TeConfigInfo::InitConfigItemsFromContext()
{
    for (const auto &item : kConfigEnumItemKeyMap) {
        const std::string &itemKey = std::get<1>(item.second);
        std::string paramValue;
        (void)ge::GetContext().GetOption(itemKey, paramValue);
        if (paramValue.empty()) {
            TE_DBGLOG("The value of [%s] is not found or the value is empty.", itemKey.c_str());
            continue;
        }
        const std::map<std::string, int64_t> &itemStrMap = std::get<2>(item.second);
        const auto iterValue = itemStrMap.find(paramValue);
        if (iterValue == itemStrMap.cend()) {
            std::string configName = ge::GetContext().GetReadableName(itemKey);
            std::map<std::string, std::string> errMsgMap;
            errMsgMap.emplace("invalid_value", paramValue);
            errMsgMap.emplace("argument", configName);
            errMsgMap.emplace("valid_range", GetMapKeyToString(itemStrMap).c_str());
            TeErrMessageReport(EM_PARAMETER_INVALID_ERROR, errMsgMap);
            TE_ERRLOG("The value[%s] of param[%s] is invalid.", paramValue.c_str(), itemKey.c_str());
            return false;
        }
        TE_DBGLOG("The value of param[%s] is [%ld].", itemKey.c_str(), iterValue->second);
        config_enum_item_vec_[static_cast<size_t>(item.first)] = iterValue->second;
    }
    for (const auto &item : kConfigStrItemKeyMap) {
        std::string paramValue;
        (void)ge::GetContext().GetOption(item.second, paramValue);
        if (!paramValue.empty()) {
            TE_DBGLOG("The value of param[%s] is [%s].", item.second.c_str(), paramValue.c_str());
            config_str_item_vec_[static_cast<size_t>(item.first)] = paramValue;
        }
    }
    ParseHardwareInfo(true);
    return true;
}

const std::string& TeConfigInfo::GetEnvHome() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::Home)];
}
const std::string& TeConfigInfo::GetEnvPath() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::Path)];
}
const std::string& TeConfigInfo::GetEnvNpuCollectPath() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::NpuCollectPath)];
}
const std::string& TeConfigInfo::GetEnvParaDebugPath() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::ParaDebugPath)];
}
const std::string& TeConfigInfo::GetEnvTeParallelCompiler() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::TeParallelCompiler)];
}
const std::string& TeConfigInfo::GetEnvTeNewDfxinfo() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::TeNewDfxinfo)];
}
const std::string& TeConfigInfo::GetEnvMinCompileResourceUsageCtrl() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::MinCompileResourceUsageCtrl)];
}
const std::string& TeConfigInfo::GetEnvOppPath() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::OppPath)];
}
const std::string& TeConfigInfo::GetEnvHomePath() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::HomePath)];
}
const std::string& TeConfigInfo::GetEnvSaveKernelMeta() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::AscendSaveKernelMeta)];
}
std::string TeConfigInfo::GetOppRealPath() const
{
    return RealPath(GetEnvOppPath());
}
const std::string& TeConfigInfo::GetEnvWorkPath() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::WorkPath)];
}
const std::string& TeConfigInfo::GetEnvCachePath() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::CachePath)];
}
const std::string& TeConfigInfo::GetEnvMaxOpCacheSize() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::MaxOpCacheSize)];
}
const std::string& TeConfigInfo::GetEnvRemainCacheSizeRatio() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::RemainCacheSizeRatio)];
}
const std::string& TeConfigInfo::GetEnvOpCompilerWorkPathInKernelMeta() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::OpCompilerWorkPathInKernelMeta)];
}
const std::string& TeConfigInfo::GetEnvAscendCoreDumpSignal() const
{
    return env_item_vec_[static_cast<size_t>(EnvItem::AscendCoreDumpSignal)];
}
CompileCacheMode TeConfigInfo::GetCompileCacheMode() const
{
    return static_cast<CompileCacheMode>(config_enum_item_vec_[static_cast<size_t>(ConfigEnumItem::CacheMode)]);
}
OpDebugLevel TeConfigInfo::GetOpDebugLevel() const
{
    return static_cast<OpDebugLevel>(config_enum_item_vec_[static_cast<size_t>(ConfigEnumItem::OpDebugLevel)]);
}
std::string TeConfigInfo::GetOpDebugLevelStr() const
{
    return std::to_string(config_enum_item_vec_[static_cast<size_t>(ConfigEnumItem::OpDebugLevel)]);
}
const std::string& TeConfigInfo::GetSocVersion() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::SocVersion)];
}
const std::string& TeConfigInfo::GetShortSocVersion() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::ShortSocVersion)];
}
const std::string& TeConfigInfo::GetAiCoreNum() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::AiCoreNum)];
}
const std::string& TeConfigInfo::GetCoreType() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::CoreType)];
}
const std::string& TeConfigInfo::GetDeviceId() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::DeviceId)];
}
const std::string& TeConfigInfo::GetFpCeilingMode() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::FpCeilingMode)];
}
std::string TeConfigInfo::GetL1Fusion() const
{
    const std::string &bufferOptimize = config_str_item_vec_[static_cast<size_t>(ConfigStrItem::BufferOptimize)];
    return (bufferOptimize == "l1_optimize" || bufferOptimize == "l1_and_l2_optimize") ? "true" : "false";
}
std::string TeConfigInfo::GetL2Fusion() const
{
    const std::string &bufferOptimize = config_str_item_vec_[static_cast<size_t>(ConfigStrItem::BufferOptimize)];
    return (bufferOptimize == "l2_optimize" || bufferOptimize == "l1_and_l2_optimize") ? "true" : "false";
}
const std::string& TeConfigInfo::GetL2Mode() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::L2Mode)];
}
const std::string& TeConfigInfo::GetMdlBankPath() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::MdlBankPath)];
}
const std::string& TeConfigInfo::GetOpBankPath() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::OpBankPath)];
}
const std::string& TeConfigInfo::GetOpDebugConfig() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::OpDebugConfig)];
}
void TeConfigInfo::SetOpDebugConfig(const std::string &opDebugConfig)
{
    if (opDebugConfig.empty()) {
        return;
    }
    std::lock_guard<std::mutex> lock_guard(config_item_mutex_);
    if (config_str_item_vec_[static_cast<size_t>(ConfigStrItem::OpDebugConfig)] == opDebugConfig) {
        return;
    }
    config_str_item_vec_[static_cast<size_t>(ConfigStrItem::OpDebugConfig)] = opDebugConfig;
}
const std::string& TeConfigInfo::GetCompileCacheDir() const
{
    return config_str_item_vec_[static_cast<size_t>(ConfigStrItem::CacheDir)];
}
const std::string& TeConfigInfo::GetLibPath() const
{
    return lib_path_;
}

void TeConfigInfo::InitKernelMetaHashAndRelativePath()
{
    // use machine/docker id + process id + timestamp to generate a unique string kernel_metaHash
    std::string pid = std::to_string(static_cast<int>(getpid()));
    // generate hashValue current time stamp
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t ts = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    std::string hashId = pid + '_' + std::to_string(ts) + GenerateMachineDockerId();
    std::hash<std::string> hashFunc;
    size_t hashValue = hashFunc(hashId);
    kernelMetaUniqueHash_ = std::to_string(hashValue);
    TE_DBGLOG("Generate kernel meta unique hash [%s].", kernelMetaUniqueHash_.c_str());

    std::string dirStr = KERNEL_META;
    const std::string &testEnableStr = env_item_vec_[static_cast<size_t>(EnvItem::OpCompilerWorkPathInKernelMeta)];
    if (testEnableStr.empty()) {
        dirStr += KERNEL_META;
    }
    kernelMetaRelativePath_ = dirStr + "_" + kernelMetaUniqueHash_;
    TE_DBGLOG("Generate kernel meta relative path [%s].", kernelMetaRelativePath_.c_str());
}

void TeConfigInfo::ParseCompileControlParams()
{
    // check MIN_COMPILE_RESOURCE_USAGE_CTRL
    const std::string &ctrlConfigStr = GetEnvMinCompileResourceUsageCtrl();
    if (ctrlConfigStr.empty()) {
        return;
    }

    std::vector<std::string> ctrlOptions;
    SplitString(ctrlConfigStr, ",", ctrlOptions);
    for (std::string &option : ctrlOptions) {
        DeleteSpaceInString(option);
        if (option == "ub_fusion") {
            isUbFusionDisabled_ = true;
            TE_INFOLOG("ub_fusion is off");
        } else if (option == "op_compile") {
            isOpCompileDisabled_ = true;
            TE_INFOLOG("op_compile is off");
        } else {
            TE_WARNLOG("Unsupported resource usage ctrl option: %s, supported options are: 'ub_fusion', 'op_compile'", option.c_str());
        }
    }
}

bool TeConfigInfo::InitOrRefreshDebugDir(const bool isInit)
{
    std::string debugDir = config_str_item_vec_[static_cast<size_t>(ConfigStrItem::OpDebugDir)];
    TE_DBGLOG("Get ge.debugDir is [%s].", debugDir.c_str());
    if (!debugDir.empty() && debugDir != "" && debugDir != GetCurrAbsPath()) {
        std::string pathOwner = ge::GetContext().GetReadableName(ge::DEBUG_DIR);
        if (!CheckPathValid(debugDir, pathOwner)) {
            TE_ERRLOG("Check debug_dir failed. %s", debugDir.c_str());
            return false;
        }
    } else {
        const std::string &ascendWorkPath = env_item_vec_[static_cast<size_t>(EnvItem::WorkPath)];
        if (!ascendWorkPath.empty()) {
            debugDir = ascendWorkPath;
            TE_DBGLOG("Get ASCEND_WORK_PATH is [%s], take it as debug dir.", debugDir.c_str());
        } else {
            debugDir = GetCurrAbsPath();
        }
    }
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    if (isInit || (!debugDirs_.empty() && debugDir != debugDirs_.top())) {
        debugDirs_.push(debugDir);
        kernelMetaParentDir_ = debugDir + kernelMetaRelativePath_;
        kernelMetaTempDir_ = debugDir + KERNEL_META + KERNEL_META_TEMP + kernelMetaUniqueHash_;
        TE_DBGLOG("Tefusion set or refresh debug dir [%s], kernel_meta_parent_dir[%s], kernel_temp_dir[%s].",
                  debugDir.c_str(), kernelMetaParentDir_.c_str(), kernelMetaTempDir_.c_str());
    }
    return true;
}

void TeConfigInfo::ParseHardwareInfo(const bool initFromContext)
{
    uint64_t session_id = initFromContext ? ge::GetContext().SessionId() : 0xFFFFFFFFFFFFFFFFULL;
    std::lock_guard<std::mutex> lock_guard(hardwareInfoMutex_);
    if (!hardwareInfoMap_.empty() && !hardwareInfoMap_[session_id].empty()) {
        TE_DBGLOG("HardwareInfo with the session id[%llu] has been parsed.", session_id);
        return;
    }
    const string &hardwareInfoStr = config_str_item_vec_[static_cast<size_t>(ConfigStrItem::HardwareInfo)];
    if (hardwareInfoStr.empty()) {
        return;
    }
    std::map<string, string> tmpHardwareInfoMap;
    std::vector<string> hardwareInfos = Split(hardwareInfoStr, ';');
    for (const auto &hardwareInfo : hardwareInfos) {
        std::vector<string> hardwarePair = Split(hardwareInfo, ':');
        if (hardwarePair.size() != kHardwareInfoLen) {
            continue;
        }
        if (hardwarePair[1].empty() || hardwarePair[1] == "0") {
            continue;
        }
        tmpHardwareInfoMap.insert(std::pair<string, string>(hardwarePair[0], hardwarePair[1]));
    }
    TE_DBGLOG("ParseHardwareInfo with the session id[%llu].", session_id);
    hardwareInfoMap_[session_id] = std::move(tmpHardwareInfoMap);
}

void TeConfigInfo::GetHardwareInfoMap(std::map<std::string, std::string> &hardwareInfoMap) const
{
    uint64_t session_id = ge::GetContext().SessionId();
    std::lock_guard<std::mutex> lock_guard(hardwareInfoMutex_);
    if (!hardwareInfoMap_.empty()) {
        auto it = hardwareInfoMap_.find(session_id);
        if (it == hardwareInfoMap_.end() || it->second.empty()) {
            TE_WARNLOG("Failed to GetHardwareInfoMap with the session id[%llu].", session_id);
            return;
        } else {
            TE_DBGLOG("GetHardwareInfoMap with the session id[%llu].", session_id);
            hardwareInfoMap.insert(it->second.begin(), it->second.end());
        }
    }
}

const std::string& TeConfigInfo::GetUniqueKernelMetaHash() const
{
    return kernelMetaUniqueHash_;
}

std::string TeConfigInfo::GetDebugDir() const
{
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    const std::string& debugDir = debugDirs_.empty() ? "." : debugDirs_.top();
    return debugDir;
}

const std::stack<std::string>& TeConfigInfo::GetAllDebugDirs() const
{
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    return debugDirs_;
}

const std::string& TeConfigInfo::GetKernelMetaTempDir() const
{
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    return kernelMetaTempDir_;
}

const std::string& TeConfigInfo::GetKernelMetaParentDir() const
{
    std::lock_guard<std::mutex> lock_guard(kernelMetaMutex_);
    return kernelMetaParentDir_;
}

const std::string& TeConfigInfo::GetAdkVersion() const
{
    return adkVersion_;
}

const std::string& TeConfigInfo::GetOppVersion() const
{
    return oppVersion_;
}

bool TeConfigInfo::IsDisableUbFusion() const
{
    return isUbFusionDisabled_;
}

bool TeConfigInfo::IsDisableOpCompile() const
{
    return isOpCompileDisabled_;
}

bool TeConfigInfo::IsBinaryInstalled() const
{
    return !oppBinaryPath_.empty();
}
}
}
