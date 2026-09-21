/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "file_handle/te_file_handle.h"
#include <dirent.h>
#include <unistd.h>
#include <sstream>
#include <fstream>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <nlohmann/json.hpp>
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"
#include "common/te_file_utils.h"
#include "common/te_config_info.h"
#include "graph/ge_context.h"

using namespace ge;
namespace te {
namespace fusion{
namespace {
constexpr const char *COMPUTE_JSON_KEY = "_compute.json";
const std::string PID_FILE_NAME = "buildPidInfo.json";
const std::string PID_DELETED = "pidDeleted";
}

void GetCurrProcesses(std::vector<uint32_t> &pids)
{
    // run in linux, if in windows need to use other system interfaces
    static const char procDir[] = "/proc/";

    DIR *dirProc = opendir(procDir);
    if (dirProc == nullptr) {
        return;
    }

    struct dirent *direntry = nullptr;
    while ((direntry = readdir(dirProc)) != nullptr) {
        if (direntry->d_type != DT_DIR || !IsPid(direntry->d_name)) {
            continue;
        }

        pids.push_back(static_cast<uint32_t>(atoi(direntry->d_name)));
    }
    closedir(dirProc);

    if (!pids.empty()) {
        nlohmann::json pidsJson = pids;
        TE_DBGLOGF("The current number of processes is :%zu.", pids.size());
    }
}

bool IsPid(const char *dname)
{
    for (size_t i = 0; i < strlen(dname); i++) {
        if (isdigit(dname[i]) == 0) {
            return false;
        }
    }
    return true;
}

void PushPidTimeId(std::vector<std::pair<uint32_t, std::string>> &pidTimeId,
                   uint32_t pid, const std::string &kernelParentDir)
{
    if (pidTimeId.empty()) {
        pidTimeId.emplace(pidTimeId.begin(), std::make_pair(pid, kernelParentDir));
        return;
    }

    for (auto iter = pidTimeId.end() - 1; iter != pidTimeId.end() && iter >= pidTimeId.begin(); --iter) {
        if (iter->first < pid) {
            pidTimeId.emplace(iter + 1, std::make_pair(pid, kernelParentDir));
            return;
        }
        if (iter->first == pid) {
            TE_DBGLOGF("Pid(%u) exist, old kernelParentDir is %s, new kernelParentDir is %s.",
                pid, iter->second.c_str(), kernelParentDir.c_str());
            return;
        }
        if (iter == pidTimeId.begin()) {
            pidTimeId.emplace(pidTimeId.begin(), std::make_pair(pid, kernelParentDir));
            return;
        }
    }
}

bool CheckFileEmpty(const std::string &realPath)
{
    std::ifstream ifs(realPath);
    if (!ifs.is_open()) {
        TE_DBGLOGF("Open %s did not succeed, file is already open.", realPath.c_str());
        return false;
    }
    char c;
    ifs >> c;
    if (ifs.eof()) {
        ifs.close();
        return true;
    }
    ifs.close();
    return false;
}

bool ReadPidTimeIdFromJson(const std::string &realPath, std::vector<std::pair<uint32_t, std::string>> &pidTimeId)
{
    if (CheckFileEmpty(realPath)) {
        TE_DBGLOGF("File %s is empty.", realPath.c_str());
        return true;
    }

    std::ifstream ifs(realPath);
    try {
        if (!ifs.is_open()) {
            TE_DBGLOGF("Open %s did not succeed, file is already open.", realPath.c_str());
            return false;
        }
        nlohmann::json pidTimeIdJson;
        ifs >> pidTimeIdJson;
        ifs.close();
        pidTimeId = pidTimeIdJson.get<std::vector<std::pair<uint32_t, std::string>>>();
    } catch (const std::exception &e) {
        TE_DBGLOGF("Failed to convert file[%s] to Json. Error message is %s.", realPath.c_str(), e.what());
        ifs.close();
        return false;
    }
    return true;
}

void WritePidTimeIdToJson(const std::string &realPath, uint32_t pid, const std::string &kernelParentDir)
{
    std::vector<std::pair<uint32_t, std::string>> pidTimeId;
    if (!ReadPidTimeIdFromJson(realPath, pidTimeId)) {
        return;
    }

    FILE* fp = TeFileUtils::LockFile(realPath, true, F_WRLCK);
    if (fp == nullptr) {
        return;
    }

    std::ofstream ofstream(realPath, std::ios::out);
    try {
        if (!ofstream.is_open()) {
            TE_DBGLOGF("Open %s did not succeed, file is already opened.", realPath.c_str());
            TeFileUtils::ReleaseLockFile(fp, realPath);
            return;
        }

        PushPidTimeId(pidTimeId, pid, kernelParentDir);

        nlohmann::json newPidTimeIdJson = pidTimeId;
        std::string jsonStr = newPidTimeIdJson.dump(SERIALIZATION_JSON_FORMAT);

        ofstream << jsonStr.c_str();
        ofstream.close();
        TeFileUtils::ReleaseLockFile(fp, realPath);
    } catch (const std::exception &e) {
        TE_DBGLOGF("Unable to write file[%s]. reason is %s.", realPath.c_str(), e.what());
        ofstream.close();
        TeFileUtils::ReleaseLockFile(fp, realPath);
        return;
    }
}

void RecordPidTimeIdInfo()
{
    std::string pidFilePath = TeConfigInfo::Instance().GetDebugDir() + KERNEL_META;
    std::string realPath = RealPath(pidFilePath);
    if (realPath.empty()) {
        TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "pidFile path[%s] does not exist, try to create one.",
                           pidFilePath.c_str());
        bool ret = mkdir(pidFilePath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
        if (ret != 0) {
            TE_WARNLOGF("Creat dir[%s] did not succeed, number: %d.", pidFilePath.c_str(), errno);
        }
        return;
    }

    std::string jsonFilePath = pidFilePath + "/" + PID_FILE_NAME;
    if (!CreateFile(jsonFilePath)) {
        TE_WARNLOGF("Creat file[%s] failed.", jsonFilePath.c_str());
        return;
    }

    uint32_t pid = static_cast<uint32_t>(getpid());
    WritePidTimeIdToJson(jsonFilePath, pid, TeConfigInfo::Instance().GetKernelMetaParentDir());
}

bool IsPidActive(uint32_t pid, std::vector<uint32_t> &pids)
{
    for (size_t i = 0; i < pids.size(); i++) {
        if (pids[i] == pid) {
            return true;
        }
    }
    return false;
}

void RemovePidTimeIdAndFile(std::vector<std::pair<uint32_t, std::string>> &pidTimeId)
{
    uint32_t localPid = static_cast<uint32_t>(getpid());
    std::vector<uint32_t> pids = {};

    GetCurrProcesses(pids);

    // del file and set kernelMetaPath value to "deleted"
    for (auto &item : pidTimeId) {
        uint32_t pid = item.first;
        if (pid != localPid && IsPidActive(pid, pids)) {
            TE_DBGLOG("Pid %u is active.", pid);
            continue;
        }
        if (pid != localPid) {
            // localPid already processed before
            std::string kernelParentPath = item.second;
            std::string kernelFilePath = kernelParentPath + KERNEL_META;
            (void)TeFileUtils::DeleteFilesInDir(kernelFilePath, false);
            RemoveKernelMetaDir(kernelParentPath);
            TE_DBGLOG("Delete pid(%u) file successfully.", pid);
        }
        item.second = PID_DELETED;
    }
}

void MgrPidTimeIdInfo(std::vector<std::pair<uint32_t, std::string>> &delPidTimeId,
    std::vector<std::pair<uint32_t, std::string>> &curPidTimeId,
    std::vector<std::pair<uint32_t, std::string>> &mgrPidTimeId)
{
    // pid is increased when write to json
    auto curIter = curPidTimeId.begin();
    auto delIter = delPidTimeId.begin();
    while (curIter != curPidTimeId.end() && delIter != delPidTimeId.end()) {
        uint32_t curPid = curIter->first;
        uint32_t delPid = delIter->first;
        if (curPid < delPid) {
            mgrPidTimeId.emplace_back(std::make_pair(curIter->first, curIter->second));
            curIter++;
        } else if (curPid == delPid) {
            if (delIter->second != PID_DELETED) {
                mgrPidTimeId.emplace_back(std::make_pair(curIter->first, curIter->second));
            }
            curIter++;
            delIter++;
        } else {
            delIter++;
        }
    }

    for (; curIter != curPidTimeId.end(); ++curIter) {
        mgrPidTimeId.emplace_back(std::make_pair(curIter->first, curIter->second));
    }
}

void UpdatePidTimeIdToJson(const std::string &jsonFileDir, const std::string &realPath,
    std::vector<std::pair<uint32_t, std::string>> &pidTimeId)
{
    // reread info lock file, merge, delete file if nothing in it
    std::vector<std::pair<uint32_t, std::string>> curPidTimeId;
    if (!ReadPidTimeIdFromJson(realPath, curPidTimeId)) {
        return;
    }

    FILE* fp = TeFileUtils::LockFile(realPath, false, F_WRLCK);
    if (fp == nullptr) {
        TE_DBGLOGF("Unable to lock file(%s) and return.", realPath.c_str());
        return;
    }

    std::ofstream ofs(realPath, std::ios::out);
    try {
        if (!ofs.is_open()) {
            TE_DBGLOGF("Open %s did not succeed, file is already open.", realPath.c_str());
            TeFileUtils::ReleaseLockFile(fp, realPath);
            return;
        }

        std::vector<std::pair<uint32_t, std::string>> mgrPidTimeId;
        MgrPidTimeIdInfo(pidTimeId, curPidTimeId, mgrPidTimeId);

        if (mgrPidTimeId.empty()) {
            std::string keyStr = PID_FILE_NAME;
            ofs.close();
            TeFileUtils::ReleaseLockFile(fp, realPath);
            TE_DBGLOGF("MgrPidTimeId is empty, keyStr: %s", keyStr.c_str());
            (void)TeFileUtils::DeleteFilesInDir(jsonFileDir, false, keyStr);
        } else {
            nlohmann::json mgrPidTimeIdJson = mgrPidTimeId;
            std::string jsonStr = mgrPidTimeIdJson.dump(SERIALIZATION_JSON_FORMAT);
            ofs << jsonStr.c_str();
            ofs.close();
            TeFileUtils::ReleaseLockFile(fp, realPath);
        }
    } catch (const std::exception &e) {
        TE_DBGLOGF("Failed to convert file[%s] to Json. Error message is %s.", realPath.c_str(), e.what());
        ofs.close();
        TeFileUtils::ReleaseLockFile(fp, realPath);
    }
}

void RemovePidTimeIdInfo()
{
    std::vector<std::pair<uint32_t, std::string>> pidTimeId;
    std::string pidJsonPath = TeConfigInfo::Instance().GetDebugDir() + KERNEL_META;
    std::string jsonFilePath = pidJsonPath + "/" + PID_FILE_NAME;
    std::string realPath = RealPath(jsonFilePath);
    if (realPath.empty()) {
        TE_DBGLOG("File path[%s] is not valid", jsonFilePath.c_str());
        return;
    }
    TE_DBGLOG("jsonFilePath:%s, realPath:%s", jsonFilePath.c_str(), realPath.c_str());

    // read from json file
    if (!ReadPidTimeIdFromJson(realPath, pidTimeId)) {
        return;
    }

    // remove remained pid and delete kernel file
    RemovePidTimeIdAndFile(pidTimeId);

    // write json, json maybe rewrited by other process, need to reread and merge
    UpdatePidTimeIdToJson(pidJsonPath, realPath, pidTimeId);
}

std::string GetKernelMetaDir()
{
    std::string kernelDir = KERNEL_META;
    const std::string &testEnableStr = TeConfigInfo::Instance().GetEnvOpCompilerWorkPathInKernelMeta();
    if (testEnableStr.empty()) {
        kernelDir += KERNEL_META;
    }
    return kernelDir + "_" + TeConfigInfo::Instance().GetUniqueKernelMetaHash();
}

void RemoveAllWorkFiles(bool keepProcessKernelMeta)
{
    bool saveKernelMeta = (TeConfigInfo::Instance().GetEnvSaveKernelMeta() == "1");
    if (saveKernelMeta) {
        TE_INFOLOG("No need to remove files in the kernel metadata directory.");
        return;
    }
    auto allDebugDirs = TeConfigInfo::Instance().GetAllDebugDirs();
    while(!allDebugDirs.empty()) {
        const std::string &debugDirTmp = allDebugDirs.top();
        allDebugDirs.pop();
        const std::string &kernelMetaParentDir = debugDirTmp + GetKernelMetaDir();
        const std::string &kernelMetaTempPath = debugDirTmp + KERNEL_META +
                                                KERNEL_META_TEMP + TeConfigInfo::Instance().GetUniqueKernelMetaHash();
        const std::string &parentKernelMetaPath = debugDirTmp + KERNEL_META;
        TE_INFOLOG("Rm files in dir:%s, kernelMetaParentDir:%s, kernelMetaTempPath:%s, parentKernelMetaPath:%s.",
                   debugDirTmp.c_str(), kernelMetaParentDir.c_str(), kernelMetaTempPath.c_str(),
                   parentKernelMetaPath.c_str());
        RemoveBuildLockFile(kernelMetaParentDir);
        RemoveKernelTempDir(kernelMetaTempPath);
        if (!keepProcessKernelMeta) {
            RemoveProcessKernelMetaDir(kernelMetaParentDir);
        }
        RemoveParentKernelMetaDir(parentKernelMetaPath);
    }
}

void RemoveBuildLockFile(const std::string &kernelMetaParentDir)
{
    std::string keyStr = ".lock";
    // remove .lock files used in build binary kernel
    (void)TeFileUtils::DeleteFilesInDir(kernelMetaParentDir, true, keyStr);
}

void RemoveKernelTempDir(const std::string &kernelMetaTempPath)
{
    std::string kernelTempPath = RealPath(kernelMetaTempPath);
    if (kernelTempPath.empty()) {
        TE_DBGLOG("kernelMetaTempPath[%s] is not valid", kernelMetaTempPath.c_str());
        return;
    }
    (void)TeFileUtils::DeleteFilesInDir(kernelTempPath, false);
    if (rmdir(kernelTempPath.c_str()) != 0) {
        TE_WARNLOG("Remove kernel_meta_temp Path[%s] failed.", kernelTempPath.c_str());
    }
    TE_DBGLOG("Remove kernel_meta_temp Path[%s] successfully.", kernelTempPath.c_str());
}

void RemoveParentKernelMetaDir(const std::string &parentKernelMetaPath)
{
    std::string parentkernelPath = RealPath(parentKernelMetaPath);
    if (parentkernelPath.empty()) {
        TE_DBGLOG("The parentKernelPath [%s] is invalid.", parentkernelPath.c_str());
        return;
    }
    if (rmdir(parentkernelPath.c_str()) != 0) {
        TE_DBGLOG("Cannot remove parent_kernel_meta path [%s].", parentkernelPath.c_str());
    } else {
        TE_DBGLOG("Parent_kernel_meta path [%s] is empty and removal was successful.", parentkernelPath.c_str());
    }
}

void RemoveKernelMetaDir(const std::string &kernelMetaParentDir)
{
    std::string kernelMetaPath = kernelMetaParentDir + KERNEL_META;
    std::string kernelRealPath = RealPath(kernelMetaPath);
    if (kernelRealPath.empty()) {
        TE_WARNLOG("kernelMetaPath [%s] is invalid", kernelMetaPath.c_str());
    } else {
        if (rmdir(kernelRealPath.c_str()) != 0) {
            TE_WARNLOG("Remove kernel_meta Path[%s] failed, cur dir is locking by other process.",
                       kernelRealPath.c_str());
        } else {
            TE_INFOLOG("Remove kernel_meta Path[%s] successfully.", kernelRealPath.c_str());
        }
    }

    std::string kernelMetaPidTimeIdPath = RealPath(kernelMetaParentDir);
    if (kernelMetaPidTimeIdPath.empty()) {
        TE_WARNLOG("kernelMetaParentDir[%s] is not valid", kernelMetaParentDir.c_str());
        return;
    }
    if (rmdir(kernelMetaPidTimeIdPath.c_str()) != 0) {
        TE_WARNLOG("Remove kernelMetaPidTimeIdPath[%s] failed.", kernelMetaPidTimeIdPath.c_str());
    } else {
        TE_INFOLOG("Remove kernelMetaPidTimeIdPath[%s] successfully.", kernelMetaPidTimeIdPath.c_str());
    }

    return;
}

void RemoveProcessKernelMetaDir(const std::string &kernelMetaParentDir)
{
    std::string kernelMetaWorkDir = RealPath(kernelMetaParentDir + KERNEL_META);
    if (kernelMetaWorkDir.empty()) {
        TE_DBGLOG("kernelMetaParentDir[%s] is not valid", kernelMetaParentDir.c_str());
        return;
    }
    if (TeFileUtils::DeleteFilesInDir(kernelMetaWorkDir, false)) {
        RemoveKernelMetaDir(kernelMetaParentDir);
    }
}

void CreateKernelMetaTempDir()
{
    const std::string& kernelTempDir = TeConfigInfo::Instance().GetKernelMetaTempDir();
    std::string realPath = RealPath(kernelTempDir);
    if (realPath.empty()) {
        bool bres = TeFileUtils::CreateMultiLevelDir(kernelTempDir);
        if (!bres) {
            REPORT_TE_INNER_WARN("Create kernel_meta_temp[%s] failed.", kernelTempDir.c_str());
            return;
        }
    }
    TE_DBGLOG("Create kernel_meta_temp[%s] successfully.", kernelTempDir.c_str());
}

std::string GetNpuCollectPath()
{
    std::string result;
    /* 1. Get path from environment para and check whether it is valid. */
    const std::string &npuCollectPathEnv = TeConfigInfo::Instance().GetEnvNpuCollectPath();
    if (!npuCollectPathEnv.empty()) {
        std::string npuCollectPath(npuCollectPathEnv);

        npuCollectPath = RealPath(npuCollectPath);
        if (npuCollectPath.empty()) {
            std::map<std::string, std::string> mapArgs = {{"path", npuCollectPathEnv},
                                                          {"arg", "NPU_COLLECT_PATH"},
                                                          {"result", "unable to copy files to npu path"},
                                                          {"reason", "path is invalid"}};
            TeErrMessageReport(EM_PATH_INVALID_WARNING, mapArgs);
            TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "NPU_COLLECT_PATH %s is invalid", npuCollectPathEnv.c_str());
            return npuCollectPath;
        }

        /* Create directory, pad /kernel_meta/pid_device_id/ */
        std::stringstream streamFileName;
        streamFileName << npuCollectPath << "/extra-info/ops/";
        streamFileName << ge::GetContext().DeviceId();
        result = streamFileName.str();
    }

    return result;
}

void ClearKernelMetaProc(const std::string &kernelMetaPath)
{
    if (TeConfigInfo::Instance().GetOpDebugLevel() != OpDebugLevel::Disable||
        !TeConfigInfo::Instance().GetOpDebugConfig().empty()) {
        return;
    }
    const std::string &opDebugConfigStr = TeConfigInfo::Instance().GetOpDebugConfig();
    std::vector<std::string> configItemVec = Split(opDebugConfigStr, ',');
    bool isDumpOpCompute = std::find(configItemVec.begin(), configItemVec.end(), kDumpUbFusion) != configItemVec.end();
    bool isDumpBin = std::find(configItemVec.begin(), configItemVec.end(), kDumpBin) != configItemVec.end() ||
                     std::find(configItemVec.begin(), configItemVec.end(), kDumpCce) != configItemVec.end();
    TE_DBGLOG("Op debug config is [%s], dump compute flag is [%d] and bin flag is [%d].",
              opDebugConfigStr.c_str(), isDumpOpCompute, isDumpBin);
    if (isDumpOpCompute || isDumpBin) {
        if (!isDumpBin) {
            (void)TeFileUtils::DeleteFilesInDir(kernelMetaPath, false, "", COMPUTE_JSON_KEY);
        }
    } else {
        (void)TeFileUtils::DeleteFilesInDir(kernelMetaPath, false);
        RemoveKernelMetaDir(TeConfigInfo::Instance().GetKernelMetaParentDir());
    }
}
} // namespace fusion
} // namespace te
