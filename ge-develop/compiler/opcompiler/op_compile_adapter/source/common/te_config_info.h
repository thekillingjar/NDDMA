/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_CONFIG_INFO_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_CONFIG_INFO_H_

#include <string>
#include <map>
#include <tuple>
#include <vector>
#include <stack>
#include <mutex>
#include "inc/te_fusion_types.h"

namespace te {
namespace fusion {
class TeConfigInfo {
public:
    TeConfigInfo(const TeConfigInfo &) = delete;
    TeConfigInfo &operator=(const TeConfigInfo &) = delete;

    static TeConfigInfo &Instance();
    bool Initialize(const std::map<std::string, std::string> &options);
    bool RefreshConfigItems();
    bool Finalize();

    const std::string& GetEnvHome() const;
    const std::string& GetEnvPath() const;
    const std::string& GetEnvNpuCollectPath() const;
    const std::string& GetEnvParaDebugPath() const;
    const std::string& GetEnvTeParallelCompiler() const;
    const std::string& GetEnvTeNewDfxinfo() const;
    const std::string& GetEnvMinCompileResourceUsageCtrl() const;
    const std::string& GetEnvOppPath() const;
    const std::string& GetEnvWorkPath() const;
    const std::string& GetEnvCachePath() const;
    const std::string& GetEnvMaxOpCacheSize() const;
    const std::string& GetEnvRemainCacheSizeRatio() const;
    const std::string& GetEnvOpCompilerWorkPathInKernelMeta() const;
    const std::string& GetEnvHomePath() const;
    const std::string& GetEnvAscendCoreDumpSignal() const;
    const std::string& GetEnvSaveKernelMeta() const;

    std::string GetOppRealPath() const;

    CompileCacheMode GetCompileCacheMode() const;
    OpDebugLevel GetOpDebugLevel() const;
    std::string GetOpDebugLevelStr() const;
    const std::string& GetSocVersion() const;
    const std::string& GetShortSocVersion() const;
    const std::string& GetAiCoreNum() const;
    const std::string& GetCoreType() const;
    const std::string& GetDeviceId() const;
    const std::string& GetFpCeilingMode() const;
    std::string GetL1Fusion() const;
    std::string GetL2Fusion() const;
    const std::string& GetL2Mode() const;
    const std::string& GetMdlBankPath() const;
    const std::string& GetOpBankPath() const;
    const std::string& GetOpDebugConfig() const;
    void SetOpDebugConfig(const std::string &opDebugConfig);
    const std::string& GetCompileCacheDir() const;
    const std::string& GetLibPath() const;
    std::string GetDebugDir() const;
    const std::stack<std::string>& GetAllDebugDirs() const;
    const std::string& GetUniqueKernelMetaHash() const;
    const std::string& GetKernelMetaTempDir() const;
    const std::string& GetKernelMetaParentDir() const;
    const std::string& GetAdkVersion() const;
    const std::string& GetOppVersion() const;
    bool IsDisableUbFusion() const;
    bool IsDisableOpCompile() const;
    bool IsBinaryInstalled() const;
    void GetHardwareInfoMap(std::map<std::string, std::string> &hardwareInfoMap) const;
private:
    TeConfigInfo();
    ~TeConfigInfo();
    bool InitLibPath();
    void InitEnvItem();
    void InitOppPath();
    void InitAdkAndOppVersion();
    static std::string GetVersionFromVersionFile(const std::string &versionFileDir);
    bool InitConfigItemsFromOptions(const std::map<std::string, std::string> &options);
    bool InitConfigItemsFromContext();
    void ParseCompileControlParams();
    void InitKernelMetaHashAndRelativePath();
    bool InitOrRefreshDebugDir(const bool isInit = true);
    void ParseHardwareInfo(const bool initFromContext);

private:
    enum class EnvItem {
        Home = 0,
        Path,
        NpuCollectPath,
        ParaDebugPath,
        TeParallelCompiler,
        TeNewDfxinfo,
        MinCompileResourceUsageCtrl,
        OppPath,
        WorkPath,
        CachePath,
        MaxOpCacheSize,
        RemainCacheSizeRatio,
        OpCompilerWorkPathInKernelMeta,
        HomePath,
        AscendCoreDumpSignal,
        AscendSaveKernelMeta,
        ItemBottom
    };
    enum class ConfigEnumItem {
        CacheMode = 0,
        Deterministic,
        OpDebugLevel,
        ItemBottom
    };
    enum class ConfigStrItem {
        SocVersion = 0,
        ShortSocVersion,
        AiCoreNum,
        BufferOptimize,
        CacheDir,
        CoreType,
        DeviceId,
        FpCeilingMode,
        HardwareInfo,
        L2Mode,
        MdlBankPath,
        OpBankPath,
        OpDebugDir,
        OpDebugConfig,
        ItemBottom
    };
    bool init_flag_;
    bool refresh_flag_;
    std::string lib_path_;
    std::string kernelMetaUniqueHash_;
    std::string kernelMetaParentDir_;
    std::string kernelMetaRelativePath_;
    std::string kernelMetaTempDir_;
    std::string oppPath_;
    std::string oppBinaryPath_;
    std::string adkVersion_;
    std::string oppVersion_;
    bool isOpCompileDisabled_ = false;
    bool isUbFusionDisabled_ = false;
    mutable std::mutex kernelMetaMutex_;
    mutable std::mutex config_item_mutex_;
    std::array<std::string, static_cast<size_t>(EnvItem::ItemBottom)> env_item_vec_;
    std::array<int64_t, static_cast<size_t>(ConfigEnumItem::ItemBottom)> config_enum_item_vec_;
    std::array<std::string, static_cast<size_t>(ConfigStrItem::ItemBottom)> config_str_item_vec_;
    std::map<uint64_t, std::map<string, string>> hardwareInfoMap_;
    mutable std::mutex hardwareInfoMutex_;
    std::stack<std::string> debugDirs_;
    static const std::map<EnvItem, std::string> kEnvItemKeyMap;
    static const std::map<ConfigEnumItem, std::tuple<int64_t, std::string, std::map<std::string, int64_t>>>
        kConfigEnumItemKeyMap;
    static const std::map<ConfigStrItem, std::string> kConfigStrItemKeyMap;
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_CONFIG_INFO_H_