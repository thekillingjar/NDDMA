/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cache/te_cache_space_manager.h"
#include <climits>
#include <fstream>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"
#include "common/te_config_info.h"
#include "common/te_file_utils.h"
#include "cache/te_cache_utils.h"
#include "dfxinfo_manager/dfxinfo_manager.h"
#include "graph/ge_context.h"

namespace te {
namespace fusion {
using namespace ge;
using namespace nlohmann;
namespace {
const uint32_t DEFAULT_MAX_OP_CACHE_SIZE = 500;                // unit is MB
const int64_t DEFAULT_REMAIN_CACHE_SIZE_RATIO = 50;           // unit is Percent
const uint64_t SIZE_MB_UNIT = 1048576;
const uint32_t PERCENT_UNIT = 100;
const uint32_t EACH_LINE_MAX_SIZE = 1024;
const std::string OP_CACHE_INI_SUB_DIR = "/op_cache.ini";
const std::string OP_CACHE_INI_SECTION = "op_compiler_cache";
const std::string OP_MAX_CACHE_SIZE = "max_op_cache_size";
const std::string OP_REMAIN_CACHE_RADIO = "remain_cache_size_ratio";
const std::string ADK_VERSION = "adk_version";
const std::string OPP_VERSION = "ops_version";
constexpr int MAX_REMAIN_CACHE_RADIO = 100;
constexpr int MIN_CACHE_AGING_TIME = 1800;
const std::string kPreCompileFileSuffix = "_pre.json";
}

TeCacheSpaceManager::TeCacheSpaceManager() {}

TeCacheSpaceManager::~TeCacheSpaceManager() {}

TeCacheSpaceManager &TeCacheSpaceManager::Instance()
{
    static TeCacheSpaceManager serialInstance;
    return serialInstance;
}

bool TeCacheSpaceManager::LoadIniFile(const std::string &fileRealPath,
    std::map<std::string, std::multimap<std::string, std::string>> &contentInfoMap) const
{
    std::ifstream ifs(fileRealPath);
    if (!ifs.is_open()) {
        TE_WARNLOGF("Open conf file failed, it does not exist or has been opened.");
        return false;
    }

    std::multimap<string, string> contentMap;
    contentMap.clear();
    std::string line;
    std::string mapKey;
    while (std::getline(ifs, line)) {
        if (line.empty() || line.find('#') == 0) {
            continue;
        }

        if (line.find('[') == 0) {
            if (!mapKey.empty() && !contentMap.empty()) {
                contentInfoMap.emplace(make_pair(mapKey, contentMap));
                contentMap.clear();
            }
            size_t pos = line.rfind(']');
            if (pos == string::npos) {
                continue;
            }
            mapKey = line.substr(1, pos - 1);
            Trim(mapKey);
            continue;
        }

        size_t posOfEqual = line.find('=');
        if (posOfEqual == string::npos) {
            continue;
        }

        std::string pos_key = line.substr(0, posOfEqual);
        Trim(pos_key);
        std::string pos_value = line.substr(posOfEqual + 1, line.length() - posOfEqual - 1);
        Trim(pos_value);
        if (!pos_key.empty() && !pos_value.empty()) {
            contentMap.insert(std::make_pair(pos_key, pos_value));
        }
    }

    if (!contentMap.empty() && !mapKey.empty()) {
        contentInfoMap.emplace(std::make_pair(mapKey, contentMap));
        contentMap.clear();
    }

    ifs.close();
    return true;
}

void TeCacheSpaceManager::InitCacheSpace(const std::string &cacheDirPath, const CompileCacheMode cacheMode)
{
    DelCachedFiles(cacheDirPath, cacheMode);
    (void)CacheSpaceInitialize(cacheDirPath);
}

int64_t TeCacheSpaceManager::GetMaxOpCacheSize() const
{
    return maxOpCacheSize_;
}

bool TeCacheSpaceManager::GetCacheInitFileCfg(const std::string &cacheFileRealPath, std::string &iniAdkVersion,
                                              std::string &iniOppVersion, std::string &opMaxCacheSize,
                                              std::string &opRemainCacheRadio) const
{
    std::map<std::string, std::multimap<std::string, std::string>> contentInfoMap;
    if (!LoadIniFile(cacheFileRealPath, contentInfoMap)) {
        TE_WARNLOG("LoadIniFile [%s] failed.", cacheFileRealPath.c_str());
        return false;
    }

    std::map<std::string, std::multimap<std::string, std::string>>::iterator iter;
    for (iter = contentInfoMap.begin(); iter != contentInfoMap.end(); ++iter) {
        if (iter->first != OP_CACHE_INI_SECTION) {
            TE_WARNLOGF("It is not contains op_compiler_cache in ini file.");
            return false;
        }
        std::multimap<std::string, std::string>::iterator multimapIter;
        for (multimapIter = iter->second.begin(); multimapIter != iter->second.end(); ++multimapIter) {
            if (multimapIter->first == ADK_VERSION) {
                iniAdkVersion = multimapIter->second;
            } else if (multimapIter->first == OPP_VERSION) {
                iniOppVersion = multimapIter->second;
            } else if (multimapIter->first == OP_MAX_CACHE_SIZE) {
                opMaxCacheSize = multimapIter->second;
            } else if (multimapIter->first == OP_REMAIN_CACHE_RADIO) {
                opRemainCacheRadio = multimapIter->second;
            }
        }
        TE_DBGLOG("ini_adk_version is [%s], ini_opp_version is [%s].", iniAdkVersion.c_str(), iniOppVersion.c_str());
        TE_DBGLOG("max_op_cache_size is [%s], remain_cache_size_ratio is [%s].",
                  opMaxCacheSize.c_str(), opRemainCacheRadio.c_str());
        if (!iniAdkVersion.empty() && !iniOppVersion.empty()) {
            return true;
        }
    }
    return false;
}

void TeCacheSpaceManager::GetNewVersionString(char *tmpLineData, const uint32_t dataLength,
                                              std::map<std::string, std::string> &contentToUpdate,
                                              std::string &strFileData) const
{
    char *tmpBuffer = new char[EACH_LINE_MAX_SIZE];
    if (tmpBuffer == nullptr) {
        TE_WARNLOGF("New buffer failed, unable to cache compile results.");
        return;
    }

    if (memset_s(tmpBuffer, EACH_LINE_MAX_SIZE, 0, EACH_LINE_MAX_SIZE) != EOK) {
        TE_WARNLOGF("Initialize buffer failed, unable to cache compile results.");
        delete[] tmpBuffer;
        return;
    }

    errno_t ret = sscanf_s(tmpLineData, "%[^=]", tmpBuffer, dataLength);
    if (ret == -1) {
        TE_WARNLOGF("Sscanf tmpLineData to buffer1 failed, unable to cache compile results.");
        delete[] tmpBuffer;
        return;
    }
    if (strcmp(ADK_VERSION.c_str(), tmpBuffer) == EOK || strcmp(OPP_VERSION.c_str(), tmpBuffer) == EOK) {
        std::string lineStart = tmpBuffer;
        strFileData += lineStart + "=" + contentToUpdate[lineStart];
        strFileData += "\n";
    } else {
        strFileData += tmpLineData;
        strFileData += "\n";
    }

    delete[] tmpBuffer;

    if (memset_s(tmpLineData, dataLength, 0, dataLength) != EOK) {
        TE_WARNLOGF("Memset tmpLineData failed, unable to cache compile results.");
    }
}

void TeCacheSpaceManager::UpdateOpCacheSizeCfg(std::string &opMaxCacheSize, std::string &opRemainCacheRadio)
{
    if (opMaxCacheSize.empty()) {
        maxOpCacheSize_ = OP_CACHE_CFG_DEFAULT_VALUE;
    }
    if (opRemainCacheRadio.empty()) {
        remainCacheSizeRatio_ = OP_CACHE_CFG_DEFAULT_VALUE;
    }
    if (opMaxCacheSize.empty() || opRemainCacheRadio.empty()) {
        return;
    }
    int64_t maxCacheSize = 0;
    int64_t remainCacheRadio = 0;
    try {
        maxCacheSize = std::stoi(opMaxCacheSize);
        remainCacheRadio = std::stoi(opRemainCacheRadio);
    } catch (const std::exception &e) {
        TE_WARNLOG("Failed to obtain max_op_cache_size [%s] or remain_cache_size_ratio [%s]. Reason: %s",
                   opMaxCacheSize.c_str(), opRemainCacheRadio.c_str(), e.what());
        return;
    }
    if (maxCacheSize == 0 || maxCacheSize < CACHE_AGING_FUCNTION_SWITCH || maxCacheSize > INT_MAX) {
        TE_WARNLOG("max_op_cache_size[%s] is invalid, it should be [1, %d]", opMaxCacheSize.c_str(), INT_MAX);
        std::map<std::string, std::string> maxSizeMap = {{"invalid_value", opMaxCacheSize},
                                                            {"argument", "max_op_cache_size"}};
        maxSizeMap["valid_range"] = "[1, " + std::to_string(INT_MAX) + "]";
        TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, maxSizeMap);
        return;
    }

    if (remainCacheRadio <= 0 || remainCacheRadio > MAX_REMAIN_CACHE_RADIO) {
        TE_WARNLOG("max_op_cache_size[%s] is invalid; it should be between in [1, 100].", opRemainCacheRadio.c_str());
        std::map<std::string, std::string> maxSizeMap = {{"invalid_value", opRemainCacheRadio},
                                                         {"argument", "remain_cache_size_ratio"}};
        maxSizeMap["valid_range"] = "[1, 100]";
        TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, maxSizeMap);
        return;
    }

    maxOpCacheSize_ =
        maxCacheSize == CACHE_AGING_FUCNTION_SWITCH ? CACHE_AGING_FUCNTION_SWITCH : maxCacheSize * SIZE_MB_UNIT;
    remainCacheSizeRatio_ = remainCacheRadio;
    TE_INFOLOG("maxOpCacheSize_ = [%lld], remainCacheSizeRatio_ = [%lld].", maxOpCacheSize_, remainCacheSizeRatio_);
    return;
}

void TeCacheSpaceManager::UpdateOpCacheIniFile(std::string cacheFileRealPath, const std::string &curAdkVersion,
                                               const std::string &curOppVersion) const
{
    std::map<std::string, std::string> contentToUpdate;
    contentToUpdate[ADK_VERSION] = curAdkVersion;
    contentToUpdate[OPP_VERSION] = curOppVersion;

    std::ifstream inFileStream;
    inFileStream.open(cacheFileRealPath, std::ios::in);
    if (!inFileStream) {
        TE_WARNLOGF("Open file[%s] with in failed, unable to cache compile results.", cacheFileRealPath.c_str());
        return;
    }

    std::string strFileData = "";
    char *tmpLineData = new char[EACH_LINE_MAX_SIZE];
    if (tmpLineData == nullptr) {
        TE_WARNLOGF("New buffer failed, unable to cache compile results.");
        inFileStream.close();
        return;
    }
    if (memset_s(tmpLineData, EACH_LINE_MAX_SIZE, 0, EACH_LINE_MAX_SIZE) != EOK) {
        TE_WARNLOGF("Initialize tmpLineData buffer failed, unable to cache compile results.");
        inFileStream.close();
        delete[] tmpLineData;
        return;
    }
    while (inFileStream.getline(tmpLineData, EACH_LINE_MAX_SIZE)) {
        GetNewVersionString(tmpLineData, EACH_LINE_MAX_SIZE, contentToUpdate, strFileData);
    }
    delete[] tmpLineData;
    inFileStream.close();

    std::ofstream outFileStream;
    outFileStream.open(cacheFileRealPath, std::ios::out);
    if (!outFileStream) {
        TE_WARNLOGF("Open file[%s] with out failed, unable to cache compile results.", cacheFileRealPath.c_str());
        return;
    }
    outFileStream.flush();
    outFileStream << strFileData;
    outFileStream.close();

    return;
}

void TeCacheSpaceManager::CreatOpCacheIniFile(
    const std::string &cacheFileRealPath, const std::string &curAdkVersion, const std::string &curOppVersion)
{
    int fd;
    std::string str_section = "[" + OP_CACHE_INI_SECTION + "]\n";
    std::string str_adk = ADK_VERSION + "=" + curAdkVersion + "\n";
    std::string str_opp = OPP_VERSION + "=" + curOppVersion + "\n";
    TE_DBGLOG("Create op_cache.ini detail is [%s, %s, %s].", str_section.c_str(), str_adk.c_str(), str_opp.c_str());
    fd = open(cacheFileRealPath.c_str(), O_CREAT | O_RDWR, FILE_AUTHORITY);
    if (fd < 0) {
        TE_WARNLOGF("Creat op_cache.ini with path[%s] failed, %s.", cacheFileRealPath.c_str(), strerror(errno));
        return;
    }

    int ret = write(fd, str_section.c_str(), str_section.length());
    if (ret < 0) {
        TE_WARNLOGF("write file with [%s] failed, %s", str_section.c_str(), strerror(errno));
        close(fd);
        return;
    }
    ret = write(fd, str_adk.c_str(), str_adk.length());
    if (ret < 0) {
        TE_WARNLOGF("write file with [%s] failed, %s", str_adk.c_str(), strerror(errno));
        close(fd);
        return;
    }
    ret = write(fd, str_opp.c_str(), str_opp.length());
    if (ret < 0) {
        TE_WARNLOGF("write file with [%s] failed, %s", str_opp.c_str(), strerror(errno));
        close(fd);
        return;
    }
    close(fd);
    return;
}

void TeCacheSpaceManager::DelCachedFiles(const std::string &cacheDir, const CompileCacheMode cacheMode)
{
    if (cacheMode == CompileCacheMode::Disable) {
        return;
    }
    const std::string &curAdkVersion = TeConfigInfo::Instance().GetAdkVersion();
    const std::string &curOppVersion = TeConfigInfo::Instance().GetOppVersion();
    if (curAdkVersion.empty() || curOppVersion.empty()) {
        TeFileUtils::DeleteFilesInDir(cacheDir, false);
        REPORT_TE_INNER_WARN("Failed to get current version info; skipping delete cache directory %s.", cacheDir.c_str());
        TE_INFOLOG("Get current version info not successfully. Skipping deletion of cache directory: %s.", cacheDir.c_str());
        return;
    }
    TE_DBGLOG("Start to check ini version in path %s with cache mode %d.", cacheDir.c_str(),
              static_cast<int32_t>(cacheMode));
    size_t pos = cacheDir.rfind('/');
    if (pos == std::string::npos) {
        TE_WARNLOG("Cache dir[%s] does not contain /.", cacheDir.c_str());
        return;
    }

    // 不能多实例配置cacheMode为force
    if (cacheMode == CompileCacheMode::Force) {
        TeFileUtils::DeleteFilesInDir(cacheDir, false);
        TE_INFOLOG("Cache mode is set to force; need to delete cache directory %s", cacheDir.c_str());
    }

    std::string cacheInitFilePath = cacheDir.substr(0, pos) + OP_CACHE_INI_SUB_DIR;
    std::string cacheInitFileRealPath = RealPath(cacheInitFilePath);
    if (cacheInitFileRealPath.empty()) {
        CreatOpCacheIniFile(cacheInitFilePath, curAdkVersion, curOppVersion);
        TeFileUtils::DeleteFilesInDir(cacheDir, false);
        maxOpCacheSize_ = OP_CACHE_CFG_DEFAULT_VALUE;
        remainCacheSizeRatio_ = OP_CACHE_CFG_DEFAULT_VALUE;
        TE_INFOLOG("Cache initialization file real path is empty. Starting to delete cache directory: %s.", cacheDir.c_str());
        return;
    }
    std::string iniAdkVersion;
    std::string iniOppVersion;
    std::string opMaxCacheSize;
    std::string opRemainCacheRadio;
    if (GetCacheInitFileCfg(cacheInitFileRealPath, iniAdkVersion, iniOppVersion, opMaxCacheSize, opRemainCacheRadio)) {
        UpdateOpCacheSizeCfg(opMaxCacheSize, opRemainCacheRadio);
        if (cacheMode == CompileCacheMode::Enable &&
            iniAdkVersion == curAdkVersion && iniOppVersion == curOppVersion) {
            return;
        }
        REPORT_TE_INNER_WARN(
            "The cache version has expired[old_adk:%s-old_opp:%s-new_adk:%s-new_opp:%s] and will be reinitialized",
            iniAdkVersion.c_str(), iniOppVersion.c_str(), curAdkVersion.c_str(), curOppVersion.c_str());
        UpdateOpCacheIniFile(cacheInitFileRealPath, curAdkVersion, curOppVersion);
        TE_INFOLOG("Need to delete cache file; path is %s.", cacheDir.c_str());
        TeFileUtils::DeleteFilesInDir(cacheDir, false);
    }
}

bool TeCacheSpaceManager::GetFileSizeInfo(const std::string &filePath, FileInfo &fileInfo)
{
    struct stat fileStat;
    if (stat(filePath.c_str(), &fileStat) != 0) {
        TE_WARNLOGF("Get file[%s] stat failed, %s.", filePath.c_str(), strerror(errno));
        return false;
    }
    fileInfo.filePath = filePath;
    fileInfo.fileSize = fileStat.st_size;
    return true;
}

void TeCacheSpaceManager::AddBinFileInfoFromJson(const std::string jsonFileDirPath, const nlohmann::json &jsonInfo,
                                                 CacheFileSizeInfo &cacheFileInfo)
{
    if (jsonInfo.find("binFileName") == jsonInfo.end() || jsonInfo.find("binFileSuffix") == jsonInfo.end()) {
        TE_WARNLOGF("The json does not have no binFileName or binFileSuffix.");
        return;
    }
    std::string binFileName = jsonInfo.at("binFileName").get<std::string>();
    std::string binFileSuffix = jsonInfo.at("binFileSuffix").get<std::string>();
    std::string objFileFullPath = jsonFileDirPath + "/" + binFileName + binFileSuffix;

    FileInfo fileInfo = FileInfo{objFileFullPath, 0};
    if (GetFileSizeInfo(objFileFullPath, fileInfo)) {
        cacheFileInfo.totalFileSizeInfos.push_back(fileInfo);
    }
}

void TeCacheSpaceManager::GetFilesAccessTime(const vector<std::string> &cacheJsonFilePaths,
                                             std::multimap<uint64_t, CacheFileSizeInfo> &filesStatInfo)
{
    for (const auto &cachedJsonFilePath : cacheJsonFilePaths) {
        CacheFileSizeInfo cacheFileInfo;
        cacheFileInfo.jsonFilePath = cachedJsonFilePath;

        struct stat fileStat;
        if (stat(cachedJsonFilePath.c_str(), &fileStat) != 0) {
            TE_WARNLOGF("Get file[%s] stat failed, %s.", cachedJsonFilePath.c_str(), strerror(errno));
        }

        FileInfo fileInfo;
        if (GetFileSizeInfo(cachedJsonFilePath, fileInfo)) {
            cacheFileInfo.totalFileSizeInfos.push_back(fileInfo);
        }
        if (IsStrEndWith(cachedJsonFilePath, kPreCompileFileSuffix)) {
            continue;
        }
        nlohmann::json jsonInfo;
        if (!TeFileUtils::GetJsonValueFromJsonFile(cachedJsonFilePath, jsonInfo)) {
            TE_WARNLOGF("Get jsonValue from jsonFile[%s] failed.", cachedJsonFilePath.c_str());
            continue;
        }

        std::string jsonFileDirPath = DirPath(cachedJsonFilePath);
        if (jsonInfo.find("jsonList") == jsonInfo.end()) { // traditional pattern
            AddBinFileInfoFromJson(jsonFileDirPath, jsonInfo, cacheFileInfo);
        } else {
            auto &jsonList = jsonInfo.at("jsonList");
            for (json::iterator it = jsonList.begin(); it != jsonList.end(); ++it) {
                std::string jsonFileName = (*it)["jsonFileName"];
                std::string curJsonFilePath = jsonFileDirPath + "/" + jsonFileName;
                struct stat st;
                if (stat(curJsonFilePath.c_str(), &st) == 0) {
                    if (st.st_mode & S_IFDIR) {
                        TE_WARNLOG("curJsonFilePath [%s] is a directory, jsonFileDirPath: [%s], jsonFileName: [%s].",
                                   curJsonFilePath.c_str(), jsonFileDirPath.c_str(), jsonFileName.c_str());
                        continue;
                    }
                }
                FileInfo curFileInfo;
                if (GetFileSizeInfo(curJsonFilePath, curFileInfo)) {
                    cacheFileInfo.totalFileSizeInfos.push_back(curFileInfo);
                }

                // read object file from json path
                nlohmann::json subJsonInfo;
                if (!TeFileUtils::GetJsonValueFromJsonFile(curJsonFilePath, subJsonInfo)) {
                    TE_WARNLOGF("Get jsonValue from jsonFile[%s] failed!", curJsonFilePath.c_str());
                    continue;
                }
                AddBinFileInfoFromJson(jsonFileDirPath, subJsonInfo, cacheFileInfo);
            }
        }
        TE_DBGLOG("File [%s]: access time is %ld seconds.", cacheFileInfo.jsonFilePath.c_str(), fileStat.st_atime);
        filesStatInfo.insert(std::make_pair(fileStat.st_atime, cacheFileInfo));
    }
}

void TeCacheSpaceManager::AgingCacheFileByAccessTime(std::multimap<uint64_t, CacheFileSizeInfo> &filesStatInfo,
                                                     const uint64_t sizeToDel) const
{
    std::multimap<uint64_t, CacheFileSizeInfo>::iterator mapIter;
    uint64_t totalDelSize = 0;
    struct stat fileStat;
    for (mapIter = filesStatInfo.begin(); mapIter != filesStatInfo.end(); ++mapIter) {
        std::vector<FileInfo>::iterator iterTotalFiles;
        std::time_t currTime = std::time(nullptr);
        double diffTime = difftime(currTime, mapIter->first);
        if (diffTime < MIN_CACHE_AGING_TIME) {
            TE_INFOLOG("currTime is [%ld], accessTime is [%ld], diffTime is [%f]", currTime, mapIter->first, diffTime);
            break;
        }
        auto fileInfo = mapIter->second.totalFileSizeInfos[0];
        if (stat(fileInfo.filePath.c_str(), &fileStat) != 0) {
            TE_WARNLOGF("Get file[%s] stat failed, %s.", fileInfo.filePath.c_str(), strerror(errno));
        } else {
            uint64_t accessTime = fileStat.st_atime;
            TE_INFOLOG("Aged file [%s], new accessTime is [%ld]", fileInfo.filePath.c_str(), accessTime)
            if (accessTime == mapIter->first) {
                // need delete all files associate with this json file
                for (auto needDelFile : mapIter->second.totalFileSizeInfos) {
                    TE_INFOLOG(
                        "Aged file is [%s], file accessTime is [%ld]", needDelFile.filePath.c_str(), mapIter->first);
                    TeFileUtils::DeleteFile(needDelFile.filePath);
                    totalDelSize += needDelFile.fileSize;
                }
            }
        }
        if (totalDelSize >= sizeToDel) {
            TE_DBGLOG("totalDelSize [%ld] sizeToDel is [%ld].", totalDelSize, sizeToDel);
            return;
        }
    }
    return;
}

void TeCacheSpaceManager::CollectCacheJsonFileAccessTime(const std::string &cachePath,
                                                         std::vector<std::string> &cachedJsonFiles) const
{
    DIR *dir = nullptr;
    struct dirent *dirp = nullptr;
    const std::string fileSuffix = ".json";
    const std::string fileSuffixMixAic = "_mix_aic.json";
    const std::string fileSuffixMixAiv = "_mix_aiv.json";
    const size_t fileSuffixSize = fileSuffix.length();
    const size_t subfileSuffixSize = fileSuffixMixAic.length();
    dir = opendir(cachePath.c_str());
    if (dir == nullptr) {
        TE_WARNLOGF("Fail to open directory %s.", cachePath.c_str());
        return;
    }
    while ((dirp = readdir(dir)) != nullptr) {
        std::string fileName(dirp->d_name);
        auto fileNameSize = fileName.length();
        if (fileName == "." || fileName == "..") {
            continue;
        }
        if (dirp->d_type == DT_DIR) {
            TE_INFOLOG("Find cache dir. Cache path is %s, and file name is %s.", cachePath.c_str(), fileName.c_str());
            std::string subCachePath = cachePath + "/" + fileName;
            CollectCacheJsonFileAccessTime(subCachePath, cachedJsonFiles);
            continue;
        }
        if (fileNameSize <= fileSuffixSize) {
            continue;
        }
        // only process main json file
        if (fileName.substr(fileNameSize - fileSuffixSize, fileSuffixSize) != fileSuffix) {
            continue;
        }
        if (fileNameSize > subfileSuffixSize) {
            if (fileName.substr(fileNameSize - subfileSuffixSize, subfileSuffixSize) == fileSuffixMixAic) {
                continue;
            }
            if (fileName.substr(fileNameSize - subfileSuffixSize, subfileSuffixSize) == fileSuffixMixAiv) {
                continue;
            }
        }
        cachedJsonFiles.push_back(cachePath + "/" + fileName);
    }
    closedir(dir);
    return;
}

void TeCacheSpaceManager::DelCacheFileByAccessTime(const std::string &cachePath, const uint64_t &sizeToDel) const
{
    std::multimap<uint64_t, CacheFileSizeInfo> filesStatInfo;
    std::vector<std::string> cachedJsonFiles;
    CollectCacheJsonFileAccessTime(cachePath, cachedJsonFiles);
    GetFilesAccessTime(cachedJsonFiles, filesStatInfo);
    AgingCacheFileByAccessTime(filesStatInfo, sizeToDel);
    return;
}

int64_t TeCacheSpaceManager::GetCacheSpaceMaxSizeCfg()
{
    if (maxOpCacheSize_ != OP_CACHE_CFG_DEFAULT_VALUE && remainCacheSizeRatio_ != OP_CACHE_CFG_DEFAULT_VALUE) {
        TE_INFOLOG("maxCacheSizeCfg is: [%lld]", maxOpCacheSize_);
        return maxOpCacheSize_;
    }
    const std::string &maxCacheSizeCfg = TeConfigInfo::Instance().GetEnvMaxOpCacheSize();
    if (maxCacheSizeCfg.empty()) {
        return DEFAULT_MAX_OP_CACHE_SIZE * SIZE_MB_UNIT;
    }
    TE_INFOLOG("The maxCacheSizeCfg is set to %s.", maxCacheSizeCfg.c_str());
    int64_t maxSizeIntCfg;
    try {
        maxSizeIntCfg = std::stoi(maxCacheSizeCfg);
    } catch (...) {
        TE_WARNLOGF("ASCEND_MAX_OP_CACHE_SIZE[%s] is invalid.", maxCacheSizeCfg.c_str());
        std::map<std::string, std::string> maxSizeMap = {
            {"invalid_value", maxCacheSizeCfg}, {"argument", "ASCEND_MAX_OP_CACHE_SIZE"}};
        maxSizeMap["valid_range"] = "[1, " + std::to_string(DEFAULT_MAX_OP_CACHE_SIZE) + ") or -1";
        maxSizeMap["default_value"] = std::to_string(DEFAULT_MAX_OP_CACHE_SIZE);
        TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, maxSizeMap);
        return DEFAULT_MAX_OP_CACHE_SIZE * SIZE_MB_UNIT;
    }
    if (maxSizeIntCfg == 0 || maxSizeIntCfg < CACHE_AGING_FUCNTION_SWITCH || maxSizeIntCfg >= INT_MAX) {
        TE_WARNLOGF("ASCEND_MAX_OP_CACHE_SIZE[%s] is invalid, it should be -1 or [1, %ld), use default size:500.",
            maxCacheSizeCfg.c_str(),
            INT_MAX);
        std::map<std::string, std::string> maxSizeMap = {
            {"invalid_value", maxCacheSizeCfg}, {"argument", "ASCEND_MAX_OP_CACHE_SIZE"}};
        maxSizeMap["valid_range"] = "[1, " + std::to_string(INT_MAX) + ") or -1 ";
        maxSizeMap["default_value"] = std::to_string(DEFAULT_MAX_OP_CACHE_SIZE);
        TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, maxSizeMap);
        return DEFAULT_MAX_OP_CACHE_SIZE * SIZE_MB_UNIT;
    }
    maxOpCacheSize_ = maxSizeIntCfg == CACHE_AGING_FUCNTION_SWITCH
                          ? CACHE_AGING_FUCNTION_SWITCH
                          : static_cast<uint64_t>(maxSizeIntCfg) * SIZE_MB_UNIT;
    TE_DBGLOG("The max cache size is %llu.", maxOpCacheSize_);
    return maxOpCacheSize_;
}

int64_t TeCacheSpaceManager::GetCacheRemainSizeRadio() const
{
    int64_t sizeRatio = 0;
    if (maxOpCacheSize_ != OP_CACHE_CFG_DEFAULT_VALUE && remainCacheSizeRatio_ != OP_CACHE_CFG_DEFAULT_VALUE) {
        sizeRatio = remainCacheSizeRatio_;
        TE_INFOLOG("remain_cache_size_ratio is: [%d]", sizeRatio);
        return sizeRatio;
    }
    const std::string &remainCacheSizeRatioStr = TeConfigInfo::Instance().GetEnvRemainCacheSizeRatio();
    if (remainCacheSizeRatioStr.empty()) {
        sizeRatio = DEFAULT_REMAIN_CACHE_SIZE_RATIO;
    } else {
        std::map<std::string, std::string> cacheSizeRatioMap = {{"invalid_value", remainCacheSizeRatioStr},
            {"argument", "ASCEND_REMAIN_CACHE_SIZE_RATIO"}, {"valid_range", "[0, 100]"}, {"default_value", "50"}};
        try {
            sizeRatio = std::atoi(remainCacheSizeRatioStr.c_str());
        } catch (...) {
            TE_WARNLOGF("ASCEND_REMAIN_CACHE_SIZE_RATIO[%s] is invalid.", remainCacheSizeRatioStr.c_str());
            TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, cacheSizeRatioMap);
            sizeRatio = DEFAULT_REMAIN_CACHE_SIZE_RATIO;
        }
        if (sizeRatio > static_cast<int>(PERCENT_UNIT) || sizeRatio < 0) {
            TE_WARNLOGF("sizeRatio[%d] is larger than 100 or less than zero.", sizeRatio);
            TeErrMessageReport(EM_PARAMETER_INVALID_WARNING, cacheSizeRatioMap);
            sizeRatio = DEFAULT_REMAIN_CACHE_SIZE_RATIO;
        }
    }
    return sizeRatio;
}

bool TeCacheSpaceManager::CacheSpaceInitialize(const std::string &cacheDir)
{
    int64_t maxSize  = GetCacheSpaceMaxSizeCfg();
    if (maxOpCacheSize_ == CACHE_AGING_FUCNTION_SWITCH) {
        TE_DBGLOG("The log aging function is disabled.");
        return true;
    }
    int64_t totalUsedSize = TeFileUtils::GetDirectorySize(cacheDir);
    int64_t remainSizeRadio = GetCacheRemainSizeRadio();

    TE_INFOLOG("Max op_cache_size is [%llu], and total_used_size is [%llu].", maxSize, totalUsedSize);
    if (totalUsedSize >= maxSize) {
        TE_WARNLOG("The cache space is insufficient. The max size is %llu and the total_used_size is %llu.",
            maxSize, totalUsedSize);
        const std::string delCacheLockName = "op_cache_file_del";
        FILE *fp = TeCacheUtils::LockAndOpenCacheFile(cacheDir, delCacheLockName);
        if (fp == nullptr) {
            TE_INFOLOGF("Other processes are deleting excess cache files.");
            return true;
        }

        if (!CheckInt64MulOverflow(maxSize, remainSizeRadio)) {
            TeCacheUtils::UnlockAndCloseCacheFile(fp);
            return false;
        }
        uint64_t sizeToDel = totalUsedSize - maxSize * remainSizeRadio / PERCENT_UNIT;
        TE_INFOLOG("The remainSizeRadio is %d. [%llu] bytes of space need to be deleted.", remainSizeRadio, sizeToDel);
        DelCacheFileByAccessTime(cacheDir, sizeToDel);
        TeCacheUtils::UnlockAndCloseCacheFile(fp);
    }
    return true;
}
} // namespace fusion
} // namespace te
