/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cache/te_cache_manager.h"
#include <fcntl.h>
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"
#include "common/te_file_utils.h"
#include "common/compile_result_utils.h"
#include "common/te_config_info.h"
#include "cache/te_cache_utils.h"
#include "cache/te_cache_space_manager.h"
#include "dfxinfo_manager/dfxinfo_manager.h"
#include "python_adapter/python_api_call.h"
#include "graph/ge_context.h"

namespace te {
namespace fusion {
TeCacheManager &TeCacheManager::Instance()
{
    static TeCacheManager teCacheManager;
    return teCacheManager;
}

bool TeCacheManager::Initialize()
{
    std::lock_guard<std::mutex> lock_guard(cache_param_mutex_);
    if (TeConfigInfo::Instance().GetOpDebugLevel() != OpDebugLevel::Disable) {
        TE_DBGLOG("Op debug level is [%s], disable cache mode.", TeConfigInfo::Instance().GetOpDebugLevelStr().c_str());
        cache_mode_ = CompileCacheMode::Disable;
        return true;
    }
    if (!TeConfigInfo::Instance().GetOpDebugConfig().empty()) {
        TE_DBGLOG(
            "Op debug config is [%s], set cache mode to disable.", TeConfigInfo::Instance().GetOpDebugConfig().c_str());
        cache_mode_ = CompileCacheMode::Disable;
        return true;
    }
    if (TeConfigInfo::Instance().GetCompileCacheMode() == CompileCacheMode::Disable) {
        TE_DBGLOG("Compile cache mode is disable.");
        cache_mode_ = CompileCacheMode::Disable;
        return true;
    }

    if (cache_mode_ == TeConfigInfo::Instance().GetCompileCacheMode() &&
        cache_parent_dir_path_ == TeConfigInfo::Instance().GetCompileCacheDir()) {
        TE_DBGLOG("Compile cache mode and cache directory have not been changed; no need to manage cache space.");
        return true;
    }
    cache_mode_ = TeConfigInfo::Instance().GetCompileCacheMode();
    cache_parent_dir_path_ = TeConfigInfo::Instance().GetCompileCacheDir();
    TE_DBGLOG("Current compile cache mode and cache dir is [%ld] and [%s].",
        static_cast<int64_t>(cache_mode_),
        cache_parent_dir_path_.c_str());
    if (!SetCacheDirPath()) {
        TE_WARNLOG("Failed to set cache dir path.");
        return false;
    }
    TeCacheSpaceManager::Instance().InitCacheSpace(cache_dir_path_, cache_mode_);
    TE_DBGLOG("Finish cache space initialization.");
    TE_DBGLOG("Finish initialization of te cache manager.");
    return true;
}

void TeCacheManager::CheckAndMakeCacheDir()
{
    bool needRealPath = true;
    if (RealPath(cache_dir_path_).empty()) {
        if (!TeFileUtils::CreateMultiLevelDir(cache_dir_path_)) {
            needRealPath = false;
            TE_WARNLOG("Create cache directory path [%s] unsuccessful.", cache_dir_path_.c_str());
        }
    }
    if (needRealPath) {
        cache_dir_path_ = RealPath(cache_dir_path_);
    }
    TE_DBGLOG("Real cache dir path is [%s].", cache_dir_path_.c_str());
}

bool TeCacheManager::SetCacheDirPath()
{
    cache_dir_path_ = cache_parent_dir_path_;
    if (cache_dir_path_.empty()) {
        cache_dir_path_ = TeConfigInfo::Instance().GetEnvCachePath();
        TE_DBGLOG("Cache parent directory path from env is [%s].", cache_dir_path_.c_str());
    }
    if (!cache_dir_path_.empty() && !IsFilePathValid(cache_dir_path_)) {
        std::string configName = ge::GetContext().GetReadableName("ge.op_compiler_cache_dir");
        std::map<std::string, std::string> cacheDirMapArgs = {{"path", cache_dir_path_},
                                                              {"arg", configName},
                                                              {"result", "unable to cache the compile results"},
                                                              {"reason", "the path contains invalid characters"}};
        TeErrMessageReport(EM_PATH_INVALID_ERROR, cacheDirMapArgs);
        TE_ERRLOG("Cache directory path [%s] is invalid.", cache_dir_path_.c_str());
        return false;
    }
    if (cache_dir_path_.empty()) {
        // if do not get cache parent dir from context or env, using default
        const std::string &homeEnv = TeConfigInfo::Instance().GetEnvHome();
        cache_dir_path_ = homeEnv + "/atc_data";
    }
    // sub dir
    cache_dir_path_ += "/kernel_cache/" + TeConfigInfo::Instance().GetSocVersion();
    TE_DBGLOG("Cache directory path is [%s].", cache_dir_path_.c_str());
    CheckAndMakeCacheDir();
    return true;
}

bool TeCacheManager::IsCacheEnable() const
{
    return cache_mode_ != CompileCacheMode::Disable;
}

CompileResultPtr TeCacheManager::MatchCompileCache(const std::string &kernelName,
                                                   bool spkMode)
{
    if (!IsCacheEnable()) {
        TE_DBGLOG("Compile cache mode is disable.");
        return nullptr;
    }
    TE_DBGLOG("Start to match cache of kernel[%s] in cache dir[%s].", kernelName.c_str(), cache_dir_path_.c_str());
    if (kernelName.empty()) {
        return nullptr;
    }

    // read from memory cache
    CompileResultPtr compileRetPtr = GetCompileResultFromCacheMap(kernelName);
    if (compileRetPtr != nullptr) {
        TE_DBGLOGF("Cache is matched for kernel name[%s], json file is [%s], bin file is [%s].",
                   kernelName.c_str(), compileRetPtr->jsonPath.c_str(), compileRetPtr->binPath.c_str());
        return compileRetPtr;
    }

    if (spkMode) {
        TE_INFOLOG("In SPK mode, the json file for kernel [%s] does not match the current process.", kernelName.c_str());
        return nullptr;
    }

    std::string jsonFilePath = RealPath(cache_dir_path_ + "/" + kernelName + ".json");
    if (jsonFilePath.empty()) {
        TE_DBGLOG("Kernel json file for [%s] does not match.", kernelName.c_str());
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::CACHE_NOT_EXIST);
        return nullptr;
    }

    compileRetPtr = CompileResultUtils::ParseCompileResult(jsonFilePath);
    if (compileRetPtr == nullptr) {
        TE_INFOLOG("Parsing compile result from json file path [%s] was not successfully.", jsonFilePath.c_str());
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::JSON_INVALID);
        return compileRetPtr;
    }

    // verify sha256 of bin file
    if (!VerifyBinFileSha256(compileRetPtr)) {
        TE_INFOLOGF("Verify sha256 of json file[%s] and bin file[%s] not success.",
                     compileRetPtr->jsonPath.c_str(), compileRetPtr->binPath.c_str());
        DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::SHA256_FAIL);
        return nullptr;
    }

    // update access time of json and bin file
    if (TeCacheSpaceManager::Instance().GetMaxOpCacheSize() != CACHE_AGING_FUCNTION_SWITCH) {
        if (!TeFileUtils::UpdateFileAccessTime(compileRetPtr->jsonPath) ||
            !TeFileUtils::UpdateFileAccessTime(compileRetPtr->binPath)) {
            TE_INFOLOGF("Update access time for json file[%s] or bin file[%s] not success.",
                        compileRetPtr->jsonPath.c_str(), compileRetPtr->binPath.c_str());
            DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::UPDATE_ACCESS_FAIL);
            return nullptr;
        }
    } else {
        TE_DBGLOG("Cache aging function is disabled, the access time of cached files is not updated.");
    }

    SetCompileResultIntoCacheMap(kernelName, compileRetPtr);
    TE_DBGLOGF("Cache is matched for kernel name[%s], json file is [%s], bin file is [%s].",
               kernelName.c_str(), compileRetPtr->jsonPath.c_str(), compileRetPtr->binPath.c_str());
    DfxInfoManager::Instance().RecordStatistics(StatisticsType::DISK_CACHE, RecordEventType::REUSE_SUCC);
    return compileRetPtr;
}

bool TeCacheManager::SetCompileResult(const CompileResultPtr &compileResultPtr)
{
    if (!IsCacheEnable()) {
        TE_DBGLOG("Compile cache mode is disable.");
        return true;
    }

    if (compileResultPtr == nullptr) {
        TE_INFOLOG("The compile result is null.");
        return false;
    }
    TE_DBGLOG("Begin to set compile result json file[%s] and bin file[%s] in cache dir[%s].",
              compileResultPtr->jsonPath.c_str(), compileResultPtr->binPath.c_str(), cache_dir_path_.c_str());
    std::string kernelName;
    if (!CompileResultUtils::GetCompileResultKernelName(compileResultPtr, kernelName) || kernelName.empty()) {
        TE_INFOLOG("Unable to get kernel name from compile result.");
        return false;
    }
    TE_DBGLOG("Kernel name in json file [%s] is [%s]", compileResultPtr->jsonPath.c_str(), kernelName.c_str());

    // set compile result in memory
    SetCompileResultIntoCacheMap(kernelName, compileResultPtr);

    // check if json file exist in cache dir
    size_t json_pos = compileResultPtr->jsonPath.rfind('/');
    if (json_pos == std::string::npos) {
        TE_INFOLOG("Json file path[%s] is invalid.", compileResultPtr->jsonPath.c_str());
        return false;
    }
    std::string cacheJsonFilePath = cache_dir_path_ + compileResultPtr->jsonPath.substr(json_pos);

    size_t bin_pos = compileResultPtr->binPath.rfind('/');
    if (bin_pos == std::string::npos) {
        TE_INFOLOG("Bin file path[%s] is invalid.", compileResultPtr->binPath.c_str());
        return false;
    }
    std::string cacheBinFilePath = cache_dir_path_ + compileResultPtr->binPath.substr(bin_pos);
    if (!RealPath(cacheJsonFilePath).empty() || !RealPath(cacheBinFilePath).empty()) {
        TE_DBGLOGF("Json file path[%s] or bin file path[%s] is already existed in cache dir.",
                   cacheJsonFilePath.c_str(), cacheBinFilePath.c_str());
        return true;
    }

    // if json file is not existed in cache dir, copy json and bin file to cache dir
    FILE *fp = TeCacheUtils::LockAndOpenCacheFile(cache_dir_path_, kernelName);
    if (fp == nullptr) {
        TE_INFOLOG("Can not lock and open cache file for cacheKernelName [%s].", kernelName.c_str());
        return true;
    }
    bool ret = CopyCompileRetIntoCacheDir(compileResultPtr, cacheJsonFilePath, cacheBinFilePath);
    TE_DBGLOG("Try to unlock cache file of kernel[%s] in cache dir[%s].", kernelName.c_str(), cache_dir_path_.c_str());
    TeCacheUtils::UnlockAndCloseCacheFile(fp);
    return ret;
}

bool TeCacheManager::CopyCompileRetIntoCacheDir(const CompileResultPtr &compileResultPtr,
                                                const std::string &cacheJsonFilePath,
                                                const std::string &cacheBinFilePath)
{
    TE_DBGLOGF("Begin to copy json file from [%s] to [%s] and bin file from [%s] to [%s].",
               compileResultPtr->jsonPath.c_str(), cacheJsonFilePath.c_str(),
               compileResultPtr->binPath.c_str(), cacheBinFilePath.c_str());
    // check file existed again in case other process copy the file before current process lock the file
    if (!RealPath(cacheJsonFilePath).empty() || !RealPath(cacheBinFilePath).empty()) {
        TE_INFOLOGF("Json file[%s] or bin file[%s] is already existed in cache dir.",
                    cacheJsonFilePath.c_str(), cacheBinFilePath.c_str());
        return true;
    }
    // copy bin file to cache dir
    if (!TeFileUtils::CopyFileToCacheDir(compileResultPtr->binPath, cacheBinFilePath)) {
        TE_INFOLOGF("Can not to copy bin file from[%s] to [%s].",
                    compileResultPtr->binPath.c_str(), cacheBinFilePath.c_str());
        return false;
    }

    // copy header file to cache dir if compile result has header file
    if (!compileResultPtr->headerPath.empty()) {
        size_t pos = compileResultPtr->headerPath.find_last_of('.');
        if (pos != std::string::npos) {
            std::string cacheHeaderFilePath = cacheBinFilePath.substr(0, cacheBinFilePath.find_last_of('.')) +
                compileResultPtr->headerPath.substr(pos);
            TE_DBGLOGF("Begin to copy head file from [%s] to [%s].",
                       compileResultPtr->headerPath.c_str(), cacheHeaderFilePath.c_str());
            if (!TeFileUtils::CopyFileToCacheDir(compileResultPtr->headerPath, cacheHeaderFilePath)) {
                TE_INFOLOGF("Can not to copy header file from [%s] to [%s] .",
                            compileResultPtr->headerPath.c_str(), cacheHeaderFilePath.c_str());
                return false;
            }
        }
    }

    // copy json file to cache dir
    if (!TeFileUtils::CopyFileToCacheDir(compileResultPtr->jsonPath, cacheJsonFilePath)) {
        TE_INFOLOGF("Can not copy json file from[%s] to [%s] did not succeed.",
                    compileResultPtr->jsonPath.c_str(), cacheJsonFilePath.c_str());
        return false;
    }
    TE_DBGLOGF("Finish copying json file from [%s] to [%s] and bin file from [%s] to [%s].",
               compileResultPtr->jsonPath.c_str(), cacheJsonFilePath.c_str(),
               compileResultPtr->binPath.c_str(), cacheBinFilePath.c_str());
    return true;
}

bool TeCacheManager::VerifyBinFileSha256(const CompileResultPtr &compileResultPtr)
{
    if (compileResultPtr == nullptr || compileResultPtr->jsonInfo == nullptr ||
        compileResultPtr->kernelBin == nullptr) {
        return false;
    }
    if (!compileResultPtr->jsonInfo->contains("sha256")) {
        TE_INFOLOG("Json info does not contain sha256.");
        return false;
    }
    std::string sha256InJson = compileResultPtr->jsonInfo->at("sha256");
    TE_DBGLOG("The Sha256 in the json is [%s].", sha256InJson.c_str());

    const char *binData = reinterpret_cast<const char *>(compileResultPtr->kernelBin->GetBinData());
    size_t binSize = compileResultPtr->kernelBin->GetBinDataSize();
    std::string sha256InBin;
    // generate sha256
    if (!PythonApiCall::Instance().GetBinFileSha256Value(binData, binSize, sha256InBin)) {
        TE_INFOLOG("Getting sha256 of bin file was unsuccessful.");
        return false;
    }
    TE_DBGLOG("Sha256 of bin file is [%s].", sha256InBin.c_str());

    // verify sha256
    if (sha256InJson.empty() || sha256InBin.empty()) {
        TE_INFOLOG("The sha256[%s] from the json file or sha256[%s] from the bin file is empty.",
            sha256InJson.c_str(),
            sha256InBin.c_str());
        return false;
    }

    if (sha256InJson != sha256InBin) {
        TE_INFOLOG("The sha256 [%s] of the binary file does not match the sha256 [%s] from the json file.",
            sha256InBin.c_str(),
            sha256InJson.c_str());
        return false;
    }
    return true;
}

PreCompileResultPtr TeCacheManager::MatchPreCompileCache(const std::string &kernelName)
{
    if (!IsCacheEnable()) {
        TE_DBGLOG("Compile cache mode is disable.");
        return nullptr;
    }
    TE_DBGLOG("Begin to match pre-compiled result cache for kernel [%s] with cache directory [%s].",
              kernelName.c_str(), cache_dir_path_.c_str());
    PreCompileResultPtr preCompileRetPtr = GetPreCompileResultFromCacheMap(kernelName);
    if (preCompileRetPtr != nullptr) {
        TE_DBGLOG("Pre-compile result matched for kernel [%s], op pattern [%s], core type [%s], and pre-options [%s].",
                  kernelName.c_str(), preCompileRetPtr->opPattern.c_str(), preCompileRetPtr->coreType.c_str(),
                  preCompileRetPtr->prebuiltOptions.c_str());
        return preCompileRetPtr;
    }
    std::string jsonFilePath = RealPath(cache_dir_path_ + "/" + kernelName + ".json");
    if (jsonFilePath.empty()) {
        TE_DBGLOG("The json file for kernel [%s] was not found during the matching of pre-compile results.", kernelName.c_str());
        return nullptr;
    }
    nlohmann::json jsonInfo;
    if (!TeFileUtils::GetJsonValueFromJsonFile(jsonFilePath, jsonInfo)) {
        TE_INFOLOG("Cannot get json value from json file [%s].", jsonFilePath.c_str());
        return nullptr;
    }

    if (!jsonInfo.contains("pattern")) {
        TE_INFOLOG("The json file[%s] does not contain pattern.", jsonFilePath.c_str());
        return nullptr;
    }
    std::string pattern = jsonInfo.at("pattern").get<std::string>();
    TE_DBGLOG("Pattern of json file[%s] is [%s].", jsonFilePath.c_str(), pattern.c_str());
    TE_FUSION_MAKE_SHARED(preCompileRetPtr = std::make_shared<PreCompileResult>(pattern), return nullptr);

    if (jsonInfo.contains("coreType")) {
        preCompileRetPtr->coreType = jsonInfo.at("coreType").get<std::string>();
        TE_DBGLOG("Core type of the json file [%s] is [%s].", jsonFilePath.c_str(), preCompileRetPtr->coreType.c_str());
    }

    if (jsonInfo.contains("prebuilt_options")) {
        preCompileRetPtr->prebuiltOptions = jsonInfo.at("prebuilt_options").get<std::string>();
        TE_DBGLOG("Pre-built options in the json file [%s] are [%s].",
                  jsonFilePath.c_str(), preCompileRetPtr->prebuiltOptions.c_str());
    }

    if (TeCacheSpaceManager::Instance().GetMaxOpCacheSize() != CACHE_AGING_FUCNTION_SWITCH) {
        if (!TeFileUtils::UpdateFileAccessTime(jsonFilePath)) {
            TE_INFOLOG("Can not Update AccessTime for json file[%s].", jsonFilePath.c_str());
            return nullptr;
        }
    } else {
        TE_DBGLOG("Cache aging function is disabled, the access time of cached files is not updated.");
    }
    SetPreCompileResultIntoCacheMap(kernelName, preCompileRetPtr);
    TE_DBGLOG("Pre-compile result matched for kernel [%s], op pattern [%s], core type [%s], and pre-options [%s].",
              kernelName.c_str(), preCompileRetPtr->opPattern.c_str(), preCompileRetPtr->coreType.c_str(),
              preCompileRetPtr->prebuiltOptions.c_str());
    return preCompileRetPtr;
}

bool TeCacheManager::SetPreCompileResult(const std::string &kernelName, const PreCompileResultPtr &preCompileResultPtr)
{
    if (!IsCacheEnable()) {
        TE_DBGLOG("Compile cache mode is disable.");
        return true;
    }
    TE_DBGLOG("Begin to set pre compile result for kernel[%s] in cache dir[%s].",
        kernelName.c_str(),
        cache_dir_path_.c_str());
    if (kernelName.empty()) {
        TE_DBGLOG("Skipping pre-build cache save due to missing kernelName.");
        return true;
    }
    if (preCompileResultPtr == nullptr || preCompileResultPtr->opPattern.empty()) {
        TE_INFOLOG("Pre compile result is null or op pattern pre compile result is empty.");
        return false;
    }
    SetPreCompileResultIntoCacheMap(kernelName, preCompileResultPtr);

    std::string jsonCachePath = cache_dir_path_ + "/" + kernelName + ".json";
    TE_DBGLOG("Begin to save pre-compile result into json file [%s].", jsonCachePath.c_str());
    if (!RealPath(jsonCachePath).empty()) {
        TE_DBGLOG("Json cache file [%s] is already existed.", jsonCachePath.c_str());
        return true;
    }
    // if json file is not existed in cache dir, copy json and bin file to cache dir
    FILE *fp = TeCacheUtils::LockAndOpenCacheFile(cache_dir_path_, kernelName);
    if (fp == nullptr) {
        TE_INFOLOG("Can not lock and open cache file for cacheKernelName [%s].", kernelName.c_str());
        return true;
    }
    bool ret = SavePreCompileRetIntoCacheDir(preCompileResultPtr, jsonCachePath);
    TeCacheUtils::UnlockAndCloseCacheFile(fp);
    return ret;
}

bool TeCacheManager::SavePreCompileRetIntoCacheDir(const PreCompileResultPtr &preCompileResultPtr,
                                                   const std::string &jsonCachePath)
{
    std::string realPath = RealPath(jsonCachePath);
    // check whether json file is existed again
    if (!realPath.empty()) {
        TE_DBGLOG("Json cache file [%s] is already existed.", jsonCachePath.c_str());
        return true;
    }
    nlohmann::json jsonInfo;
    jsonInfo["pattern"] = preCompileResultPtr->opPattern;
    jsonInfo["coreType"] = preCompileResultPtr->coreType;
    jsonInfo["prebuilt_options"] = preCompileResultPtr->prebuiltOptions;
    std::string fileStr = jsonInfo.dump(SERIALIZATION_JSON_FORMAT);
    if (!TeFileUtils::WriteNewFileInCache(jsonCachePath, fileStr)) {
        TE_INFOLOGF("Can not Write file[%s].", jsonCachePath.c_str());
        return false;
    }
    TE_DBGLOGF("Json cache file[%s] has been saved in cache dir. Op pattern[%s], core type[%s], options[%s].",
               jsonCachePath.c_str(), preCompileResultPtr->opPattern.c_str(), preCompileResultPtr->coreType.c_str(),
               preCompileResultPtr->prebuiltOptions.c_str());
    return true;
}

CompileResultPtr TeCacheManager::GetCompileResultFromCacheMap(const std::string &kernelName)
{
    if (kernelName.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock_guard(compile_cache_mutex_);
    auto iter = compile_ret_cache_map_.find(kernelName);
    if (iter == compile_ret_cache_map_.cend()) {
        return nullptr;
    }
    return iter->second;
}

void TeCacheManager::SetCompileResultIntoCacheMap(
    const std::string &kernelName, const CompileResultPtr &compileResultPtr)
{
    if (kernelName.empty() || compileResultPtr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock_guard(compile_cache_mutex_);
    compile_ret_cache_map_.emplace(kernelName, compileResultPtr);
}

PreCompileResultPtr TeCacheManager::GetPreCompileResultFromCacheMap(const std::string &kernelName)
{
    if (kernelName.empty()) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock_guard(precompile_cache_mutex_);
    auto iter = precompile_ret_cache_map_.find(kernelName);
    if (iter == precompile_ret_cache_map_.cend()) {
        return nullptr;
    }
    return iter->second;
}

void TeCacheManager::SetPreCompileResultIntoCacheMap(
    const std::string &kernelName, const PreCompileResultPtr &preCompileResultPtr)
{
    if (kernelName.empty() || preCompileResultPtr == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock_guard(precompile_cache_mutex_);
    precompile_ret_cache_map_.emplace(kernelName, preCompileResultPtr);
}

void TeCacheManager::SetOpArgsCache(const std::string &key_name, const std::string &op_args) {
    std::lock_guard<std::mutex> lock_guard(op_args_mutex_);
    auto it = op_args_cache_map_.find(key_name);
    if (it != op_args_cache_map_.end()) {
        it->second = op_args;
    } else {
        op_args_cache_map_.emplace(key_name, op_args);
    }
}

bool TeCacheManager::GetOpArgsCache(const std::string &key_name, std::string &op_args) {
    std::lock_guard<std::mutex> lock_guard(op_args_mutex_);
    auto it = op_args_cache_map_.find(key_name);
    if (it != op_args_cache_map_.end()) {
        op_args = it->second;
        return true;
    }
    return false;
}

void TeCacheManager::ClearOpArgsCache(const std::string &session_graph_id) {
    std::lock_guard<std::mutex> lock_guard(op_args_mutex_);
    for (auto it = op_args_cache_map_.begin(); it != op_args_cache_map_.end();) {
        if (0 == std::strncmp(it->first.c_str(), session_graph_id.c_str(), std::min(session_graph_id.length(), it->first.length()))) {
            TE_DBGLOG("ClearOpArgsCache erased op %s", it->first.c_str());
            it = op_args_cache_map_.erase(it);
        } else {
            TE_DBGLOG("ClearOpArgsCache op %s", it->first.c_str());
            it++;
        }
    }
}

void TeCacheManager::Finalize()
{
    compile_ret_cache_map_.clear();
    precompile_ret_cache_map_.clear();
}
}
}
