/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_CACHE_TE_CACHE_SPACE_MANAGER_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_CACHE_TE_CACHE_SPACE_MANAGER_H_

#include <string>
#include <vector>
#include "inc/te_fusion_types.h"
#include "inc/te_fusion_util_constants.h"

namespace te {
namespace fusion {
struct FileInfo {
    std::string filePath;
    uint32_t fileSize;
};

struct CacheFileSizeInfo {
    std::string jsonFilePath;
    vector<FileInfo> totalFileSizeInfos;
};

/**
 * @brief: the Fusion serial Class
 */
class TeCacheSpaceManager {
  public:
    TeCacheSpaceManager();
    ~TeCacheSpaceManager();
    static TeCacheSpaceManager &Instance();

    void InitCacheSpace(const std::string &cacheDirPath, const CompileCacheMode cacheMode);

    int64_t GetMaxOpCacheSize() const;

  private:
    static bool GetFileSizeInfo(const std::string &filePath, FileInfo &fileInfo);

    bool LoadIniFile(const std::string &fileRealPath,
                     std::map<std::string, std::multimap<std::string, std::string>> &contentInfoMap) const;

    bool GetCacheInitFileCfg(const std::string &cacheFileRealPath, std::string &iniAdkVersion,
                             std::string &iniOppVersion, std::string &opMaxCacheSize,
                             std::string &opRemainCacheRadio) const;

    void UpdateOpCacheIniFile(std::string cacheFileRealPath, const std::string &curAdkVersion,
                              const std::string &curOppVersion) const;

    void UpdateOpCacheSizeCfg(std::string &opMaxCacheSize, std::string &opRemainCacheRadio);

    static void CreatOpCacheIniFile(
        const std::string &cacheFileRealPath, const std::string &curAdkVersion, const std::string &curOppVersion);

    void DelCachedFiles(const std::string &cacheDir, const CompileCacheMode cacheMode);

    static void AddBinFileInfoFromJson(const std::string jsonFileDirPath, const nlohmann::json &jsonInfo,
                                       CacheFileSizeInfo &cacheFileInfo);

    static void GetFilesAccessTime(const vector<std::string> &cacheJsonFilePaths,
                                   std::multimap<uint64_t, CacheFileSizeInfo> &filesStatInfo);

    void DelCacheFileByAccessTime(const std::string &cachePath, const uint64_t &sizeToDel) const;

    void AgingCacheFileByAccessTime(std::multimap<uint64_t, CacheFileSizeInfo> &filesStatInfo,
                                    const uint64_t sizeToDel) const;

    void CollectCacheJsonFileAccessTime(const std::string &cachePath,
                                        std::vector<std::string> &cachedJsonFiles) const;

    int64_t GetCacheSpaceMaxSizeCfg();

    int64_t GetCacheRemainSizeRadio() const;

    bool CacheSpaceInitialize(const std::string &cacheDir);

    void GetNewVersionString(char *tmpLineData, const uint32_t dataLength,
                             std::map<std::string, std::string> &contentToUpdate, std::string &strFileData) const;

    int64_t maxOpCacheSize_{OP_CACHE_CFG_DEFAULT_VALUE};
    int64_t remainCacheSizeRatio_{OP_CACHE_CFG_DEFAULT_VALUE};
};
}  // namespace fusion
}  // namespace te
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_CACHE_TE_CACHE_SPACE_MANAGER_H_
