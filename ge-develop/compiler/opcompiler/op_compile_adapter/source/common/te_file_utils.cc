/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/te_file_utils.h"
#include <climits>
#include <fstream>
#include <thread>
#include <dirent.h>
#include <fcntl.h>
#include <utime.h>
#include <sys/stat.h>
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "common/common_utils.h"

namespace te {
namespace fusion {
namespace {
const int RECURSIVE_INTVL = 10; // 10 milliseconds
const int RECURSIVE_CNT_MAX = 6000; // 1mins 6000*10 milliseconds
}
bool TeFileUtils::CreateMultiLevelDir(const std::string &directoryPath)
{
    auto dir_path_len = directoryPath.length();
    if (dir_path_len >= PATH_MAX) {
        TE_WARNLOGF("Path[%s] len is too long, it must be less than %d", directoryPath.c_str(), PATH_MAX);
        return false;
    }
    char tmpDirPath[PATH_MAX] = {0};
    int32_t ret;
    for (size_t i = 0; i < dir_path_len; ++i) {
        tmpDirPath[i] = directoryPath[i];
        if ((tmpDirPath[i] == '\\') || (tmpDirPath[i] == '/')) {
            if (access(tmpDirPath, F_OK) == 0) {
                continue;
            }
            bool res = JudgeEmptyAndCreateDir(tmpDirPath, directoryPath);
            if (!res) {
                return false;
            }
        }
    }

    std::string path = RealPath(directoryPath);
    if (path.empty()) {
        ret = mkdir(directoryPath.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);  // 750
        if (ret != 0 && errno != EEXIST) {
            REPORT_TE_INNER_WARN("Creat dir[%s] failed, reason is: %s", directoryPath.c_str(), strerror(errno));
            return false;
        }
    }

    TE_DBGLOG("Create multi level dir [%s] successfully.", directoryPath.c_str());
    return true;
}

bool TeFileUtils::JudgeEmptyAndCreateDir(char tmpDirPath[], const std::string &directoryPath)
{
    std::string realPath = RealPath(tmpDirPath);
    if (realPath.empty()) {
        int32_t ret = 0;
        ret = mkdir(tmpDirPath, S_IRWXU | S_IRGRP | S_IXGRP);  // 750
        if (ret != 0 && errno != EEXIST) {
            TE_WARNLOGF("Creat dir[%s] failed, %s.", directoryPath.c_str(), strerror(errno));
            return false;
        }
    }
    return true;
}

bool TeFileUtils::CopyDirFileToNewDir(const std::string &dirPath, const std::string &newDirPath)
{
    DIR *dir = nullptr;
    struct dirent *dirp = nullptr;
    std::string oriRealPath = RealPath(dirPath);
    std::string newRealPath = RealPath(newDirPath);
    if (oriRealPath.empty()) {
        TE_WARNLOG("Original dir path %s is invalid.", dirPath.c_str())
        return true;
    }

    if (newRealPath.empty()) {
        TE_WARNLOG("New dir path %s is invalid.", newDirPath.c_str())
        return true;
    }

    dir = opendir(oriRealPath.c_str());
    TE_FUSION_CHECK(dir == nullptr, {
        TE_DBGLOG("Unable to open directory %s.", oriRealPath.c_str());
        return false;
    });

    while ((dirp = readdir(dir)) != nullptr) {
        std::string fileName(dirp->d_name);
        if (fileName == "." || fileName == "..") {
            continue;
        }
        std::string oriFilePath = RealPath(oriRealPath + "/" + fileName);
        std::string newFilePath = newRealPath + "/" + fileName;
        if (dirp->d_type == DT_DIR) {
            if (CreateMultiLevelDir(newFilePath)) {
                CopyDirFileToNewDir(oriFilePath, newFilePath);
            }
        }
        CopyFileToNewPath(oriFilePath, newFilePath);
    }

    closedir(dir);
    return true;
}

bool TeFileUtils::CopyFileToNewPath(const std::string &filePath, const std::string &dstPath)
{
    FILE *fp1 = fopen(filePath.c_str(), "rb");
    if (fp1 == nullptr) {
        TE_WARNLOGF("Open file[%s] failed, %s.", filePath.c_str(), strerror(errno));
        return false;
    }

    FILE *fp2 = fopen(dstPath.c_str(), "wb");
    if (fp2 == nullptr) {
        TE_WARNLOGF("Open file[%s] failed, %s.", dstPath.c_str(), strerror(errno));
        fclose(fp1);
        return false;
    }
    int res = chmod(dstPath.c_str(), FILE_AUTHORITY);
    if (res == -1) {
        TE_WARNLOGF("Update file[%s] authority failed.", dstPath.c_str());
    }

    if (!FcntlLockFileSet(fileno(fp2), F_WRLCK, 0)) {
        TE_WARNLOGF("Lock file[%s] failed, %s.", dstPath.c_str(), strerror(errno));
        fclose(fp1);
        fclose(fp2);
        return false;
    }

    TE_DBGLOG("Try to lock file [%s] successfully.", dstPath.c_str());

    char buff[1024] = {'\0'};
    int count = 0;
    while ((count = fread(buff, 1, sizeof(buff), fp1)) != 0) {
        fwrite(buff, 1, count, fp2);
    }

    (void)FcntlLockFileSet(fileno(fp2), F_UNLCK, 0);

    TE_DBGLOG("Copy file %s to %s successfully.",  filePath.c_str(), dstPath.c_str());
    fclose(fp1);
    fclose(fp2);
    return true;
}

bool TeFileUtils::IsObjFileExsit(const std::string &opName, const std::string &jsonFilePath)
{
    nlohmann::json jsonInfo;
    if (!GetJsonValueFromJsonFile(jsonFilePath, jsonInfo)) {
        TE_INFOLOG("Cannot get jsonValue from jsonFile [%s].", jsonFilePath.c_str());
        return false;
    }

    if (jsonInfo.find("jsonList") == jsonInfo.end()) { // traditional pattern
        std::string binFileName = jsonInfo.at("binFileName");
        std::string binFileSuffix = jsonInfo.at("binFileSuffix");
        std::string oFilePath = DirPath(jsonFilePath) + "/" + binFileName + binFileSuffix;
        TE_DBGLOGF("Get file path. (jsonFilePath = %s, oFilePath = %s)", jsonFilePath.c_str(), oFilePath.c_str());
        std::string realPath = RealPath(oFilePath);
        if (realPath.empty()) {
            TE_WARNLOG("o file path [%s] does not exist.", oFilePath.c_str());
            return false;
        }
        TE_DBGLOG("Check binary file invalid successfully.");
        return true;
    }
    // mix pattern
    // cache main json file
    // cache sub .json and .o files
    auto jsonList = jsonInfo.at("jsonList");
    for (nlohmann::json::iterator it = jsonList.begin(); it != jsonList.end(); ++it) {
        std::string jsonFileName = (*it)["jsonFileName"];
        std::string dirPath = DirPath(jsonFilePath);
        std::string subJsonFilePath = dirPath + "/" + jsonFileName;
        TE_DBGLOG("subJsonFilePath is [%s].", subJsonFilePath.c_str());
        if (!IsObjFileExsit(opName, subJsonFilePath)) {
            return false;
        }
    }
    TE_DBGLOG("binary file from jsonFilePath [%s] already exists.", jsonFilePath.c_str());
    return true;
}

// copy file to cache dir
bool TeFileUtils::CopyFileToCacheDir(const std::string &srcPath, const std::string &dstPath)
{
    TE_DBGLOG("Start copy %s to %s", srcPath.c_str(), dstPath.c_str());
    int fp1 = open(srcPath.c_str(), O_RDONLY);
    if (fp1 == -1) {
        TE_ERRLOG("Failed to open file %s, error message: [%s], error number: [%d]", srcPath.c_str(), strerror(errno), errno);
        return false;
    }
    std::string dstPathTemp = ConstructTempFileName(dstPath);
    TE_DBGLOG("Start write temp file %s", dstPathTemp.c_str());
    int flag = O_CREAT | O_RDWR;
    int fp2 = open(dstPathTemp.c_str(), flag, FILE_AUTHORITY);
    if (fp2 == -1) {
        TE_ERRLOG("Failed to open file %s, error message: [%s], error number: [%d]", dstPathTemp.c_str(), strerror(errno), errno);
        close(fp1);
        return false;
    }
    int res = fchmod(fp2, FILE_AUTHORITY);
    if (res == -1) {
        TE_ERRLOG("Update of file [%s] authority failed: %d", dstPathTemp.c_str(), errno);
        close(fp1);
        close(fp2);
        return false;
    }

    char buff[1024] = {'\0'};
    int count = 0;
    int wcount = 0;
    while ((wcount >= 0) && (count = read(fp1, buff, sizeof(buff))) > 0) {
        int nw = 0;
        while ((count > nw) && (wcount = write(fp2, buff + nw, count - nw)) >= 0) {
            nw += wcount;
        }
    }
    close(fp1);
    close(fp2);
    if (wcount < 0 || count < 0) {
        // we treat read/write errno of EAGAIN/EINTR as fail too, as failure here is not fatal.
        TE_ERRLOG("read/write error. src:%s, dst:%s, errno:%d", srcPath.c_str(), dstPathTemp.c_str(), errno);
        return false;
    }
    TE_DBGLOG("Successfully wrote temp file %s", dstPathTemp.c_str());
    res = rename(dstPathTemp.c_str(), dstPath.c_str());
    if (res != 0) {
        TE_ERRLOG("rename error. src:%s, dst:%s, errno:%d", dstPathTemp.c_str(), dstPath.c_str(), errno);
        return false;
    }
    TE_DBGLOG("Success copy %s to %s", srcPath.c_str(), dstPath.c_str());
    return true;
}

bool TeFileUtils::WriteNewFileInCache(const std::string &filePath, const std::string &fileStr)
{
    TE_DBGLOG("Start write data to file %s", filePath.c_str());
    std::string filePathTemp = ConstructTempFileName(filePath);
    int flag = O_CREAT | O_RDWR;
    int fp = open(filePathTemp.c_str(), flag, FILE_AUTHORITY);
    if (fp == -1) {
        TE_ERRLOG("Failed to open file %s, error message: [%s], error number: [%d]", filePathTemp.c_str(), strerror(errno), errno);
        close(fp);
        return false;
    }
    int res = fchmod(fp, FILE_AUTHORITY);
    if (res == -1) {
        TE_ERRLOG("Update of file [%s] authority failed: %d", filePathTemp.c_str(), errno);
        close(fp);
        return false;
    }
    const char *buff = fileStr.c_str();
    int count = fileStr.size();
    int wcount = 0;
    int nw = 0;
    while ((count > nw) && (wcount = write(fp, buff + nw, count - nw)) >= 0) {
        nw += wcount;
    }
    if (wcount < 0) {
        TE_ERRLOG("write error. file:%s, errno:%d.", filePathTemp.c_str(), errno);
        close(fp);
        return false;
    }
    close(fp);
    TE_DBGLOG("Successfully wrote temp file %s", filePathTemp.c_str());
    res = rename(filePathTemp.c_str(), filePath.c_str());
    if (res != 0) {
        TE_ERRLOG("rename error. src:%s, dst:%s, errno:%d.", filePathTemp.c_str(), filePath.c_str(), errno);
        return false;
    }
    TE_DBGLOG("Successfully wrote data to file %s", filePath.c_str());
    return true;
}

bool TeFileUtils::DeleteFilesInDir(const std::string &dirPath, bool needCheckUsing, const std::string &fileToDelete,
                                   const std::string &fileToReserve)
{
    TE_DBGLOG("Start to delete dir %s", dirPath.c_str());
    DIR *dir = nullptr;
    struct dirent *dirp = nullptr;
    std::string realPath = RealPath(dirPath);
    dir = opendir(realPath.c_str());
    if (dir == nullptr) {
        TE_DBGLOG("Unable to open directory[%s].", realPath.c_str());
        return false;
    };

    while ((dirp = readdir(dir)) != nullptr) {
        std::string fileName(dirp->d_name);
        if (fileName == "." || fileName == "..") {
            continue;
        }

        if (!fileToDelete.empty() && fileName.find(fileToDelete) == std::string::npos) {
            continue;
        }
        if (!fileToReserve.empty() && fileName.find(fileToReserve) != std::string::npos) {
            continue;
        }
        std::string filePath = RealPath(realPath + "/" + fileName);
        struct stat info;
        if (stat(filePath.c_str(), &info) != 0) {
            TE_WARNLOG("Stat file[%s] failed", filePath.c_str());
            continue;
        }
        if (S_ISDIR(info.st_mode)) {
            TeFileUtils::DeleteFilesInDir(filePath, needCheckUsing, fileToDelete, fileToReserve);
        } else {
            if (needCheckUsing && IsFileUsed(filePath)) {
                continue;
            }
        }
        if (remove(filePath.c_str()) == 0) {
            TE_DBGLOG("Delete file[%s] success", filePath.c_str());
        } else {
            TE_WARNLOG("Delete file[%s] failed", filePath.c_str());
        }
    }
    TE_DBGLOG("Finish delete dir %s", dirPath.c_str());
    closedir(dir);
    return true;
}

void TeFileUtils::DeleteFile(const std::string &path)
{
    if (path.empty()) {
        TE_WARNLOG("File name is empty.");
        return;
    }
    struct stat statBuf;
    if (lstat(path.c_str(), &statBuf) != 0) {
        TE_WARNLOG("Stat file[%s] failed.", path.c_str());
        return;
    }
    if (S_ISREG(statBuf.st_mode) == 0) {
        TE_WARNLOG("[%s] is not a file.", path.c_str());
        return;
    }
    int res = remove(path.c_str());
    if (res != 0) {
        TE_WARNLOG("Delete file[%s] failed.", path.c_str());
    }
}

bool TeFileUtils::IsFileUsed(std::string &filePath)
{
    // if locked by other process, don't delete the file
    FILE *fp = fopen(filePath.c_str(), "r");
    if (fp == nullptr) {
        TE_WARNLOG("Can not open file[%s].", filePath.c_str());
        return true;
    }

    if (IsFileFcntlLock(fileno(fp))) {
        TE_WARNLOG("File[%s] is locked by other process.", filePath.c_str());
        fclose(fp);
        return true;
    }
    TE_DBGLOG("File [%s] is not locked by other process.", filePath.c_str());
    fclose(fp);
    return false;
}

bool TeFileUtils::IsFileFcntlLock(int fd)
{
    struct flock lock;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_type = F_RDLCK; // F_GETLK, the type should not be F_UNLCK
    if (fcntl(fd, F_GETLK, &lock) != 0) {
        TE_ERRLOG("Get file lock failed.");
        return false;
    }

    if (lock.l_type == F_UNLCK) {
        // locked process get lock will return unlock
        TE_DBGLOG("Get file lock successfully, it is unlocked.");
        return false;
    }
    if (lock.l_type == F_RDLCK) {
        TE_DBGLOG("Get file lock successfully, it is read locked by %d.", lock.l_pid);
    }
    if (lock.l_type == F_WRLCK) {
        TE_DBGLOG("Get file lock successfully, it is write locked by %d.", lock.l_pid);
    }
    return true;
}

bool TeFileUtils::FcntlLockFileSet(int fd, int type, int recursiveCnt)
{
    struct flock lock;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;
    lock.l_type = type;

    // lock or unlock
    if (fcntl(fd, F_SETLK, &lock) == 0) {
        return true;
    }

    if (errno != EACCES && errno != EAGAIN) {
        TE_ERRLOG("fcntl F_SETLK failed. type:%d, recursiveCnt:%d, errno:%d, %s",
                  type, recursiveCnt, errno, strerror(errno));
    }
    if (type == F_UNLCK) {
        TE_WARNLOG("Unlock file failed.");
        return false;
    }

    // get the lock fail reason
    if (recursiveCnt == 0) {
        lock.l_type = F_RDLCK;
        int ret = fcntl(fd, F_GETLK, &lock);
        TE_INFOLOG("Get lock. (ret=%d, type=%d)", ret, type);
        if (lock.l_type == F_RDLCK) {
            TE_INFOLOG("File is read locked by %d.", lock.l_pid);
        }
        if (lock.l_type == F_WRLCK) {
            TE_INFOLOG("File is write locked by %d.", lock.l_pid);
        }
    }

    return false;
}

FILE* TeFileUtils::LockFile(const std::string &filePath, bool needRepeatTry, int lockType)
{
    std::string realPath = RealPath(filePath);
    if (realPath.empty()) {
        TE_WARNLOG("FilePath[%s] is not correct.", filePath.c_str());
        return nullptr;
    }
    std::string openFile = (lockType == F_RDLCK) ? "r" : "w"; // lockType is F_RDLCK or F_WRLCK
    FILE* fp = fopen(realPath.c_str(), openFile.c_str());
    if (fp == nullptr) {
        TE_WARNLOG("Open file[%s] failed.", realPath.c_str());
        return nullptr;
    }

    if (needRepeatTry) {
        int recursiveCnt = 0;
        do {
            if (!FcntlLockFileSet(fileno(fp), lockType, 0)) {
                std::this_thread::sleep_for(std::chrono::microseconds(RECURSIVE_INTVL));
            } else {
                break;
            }
            if (recursiveCnt == RECURSIVE_CNT_MAX) {
                TE_WARNLOG("Lock file(%s) failed, try %u times.", realPath.c_str(), RECURSIVE_CNT_MAX);
                fclose(fp);
                return nullptr;
            }
            recursiveCnt++;
        } while (true);
    } else {
        if (!FcntlLockFileSet(fileno(fp), lockType, 0)) {
            TE_WARNLOG("Lock file(%s) failed.", realPath.c_str());
            fclose(fp);
            return nullptr;
        }
    }

    TE_DBGLOG("Lock file [%s] success", realPath.c_str());
    // release the lock and close the file by caller
    return fp;
}

void TeFileUtils::ReleaseLockFile(FILE *fp, const std::string &path)
{
    if (fp == nullptr) {
        return;
    }
    (void)FcntlLockFileSet(fileno(fp), F_UNLCK, 0);

    TE_DBGLOG("Release lock file (%s) success.", path.c_str());
    fclose(fp);
}

/*
    Update the access time of the JSON file when the cache is hit to prevent the cacheFile being aged.
*/
bool TeFileUtils::UpdateFileAccessTime(const std::string &path)
{
    std::string realPath = RealPath(path);
    if (realPath.empty()) {
        TE_WARNLOG("path[%s] does not exist.", path.c_str());
        return false;
    }
    struct stat st;
    if (stat(realPath.c_str(), &st) != 0) {
        TE_ERRLOG("stat file [%s] failed.", realPath.c_str());
        return false;
    }
    struct utimbuf newTimes;
    time(&newTimes.actime);
    newTimes.modtime = st.st_mtime;
    if (utime(realPath.c_str(), &newTimes) != 0) {
        TE_ERRLOG("utime file[%s] failed. fail reason: [%s].", realPath.c_str(), strerror(errno));
        return false;
    }
    TE_DBGLOG("update atime file[%s] success.", realPath.c_str());
    return true;
}

int64_t TeFileUtils::GetDirectorySize(const std::string& path)
{
    int64_t size = 0;
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        return size;
    }

    struct dirent* entry;
    struct stat info;
    while ((entry = readdir(dir)) != nullptr) {
        std::string fileName = entry->d_name;
        std::string fullPath = path + "/" + fileName;

        if (stat(fullPath.c_str(), &info) != 0) {
            continue;
        }
        if (S_ISDIR(info.st_mode)) {
            if (fileName != "." && fileName != "..") {
                size += GetDirectorySize(fullPath);
            }
        } else if (S_ISREG(info.st_mode)) {
            size += info.st_size;
        }
    }

    closedir(dir);
    return size;
}

bool TeFileUtils::GetJsonValueFromJsonFile(const std::string &jsonFilePath, nlohmann::json &jsonValue)
{
    std::string realPath = RealPath(jsonFilePath);
    if (realPath.empty()) {
        TE_WARNLOG("Json file path [%s] is not valid.", jsonFilePath.c_str());
        return false;
    }
    std::ifstream ifs(realPath);
    try {
        if (!ifs.is_open()) {
            REPORT_TE_INNER_WARN("Opening %s failed, file is already in use.", jsonFilePath.c_str());
            return false;
        }

        ifs >> jsonValue;
        ifs.close();
    } catch (const std::exception &e) {
        REPORT_TE_INNER_WARN("Failed to convert file [%s] to JSON. Reason: %s", jsonFilePath.c_str(), e.what());
        ifs.close();
        return false;
    }

    TE_DBGLOGF("jsonFilePath: [%s], jsonValue: [%s].", jsonFilePath.c_str(), jsonValue.dump().c_str());
    return true;
}

bool TeFileUtils::GetBufferFromBinFile(const std::string &binFilePath, std::vector<char> &buffer)
{
    std::string realPath = RealPath(binFilePath);
    if (realPath.empty()) {
        TE_WARNLOG("Bin file path[%s] is not valid.", binFilePath.c_str());
        return false;
    }

    std::ifstream ifStream(realPath.c_str(), std::ios::binary | std::ios::ate);
    if (!ifStream.is_open()) {
        TE_WARNLOG("read file %s failed.", binFilePath.c_str());
        return false;
    }
    try {
        std::streamsize size = ifStream.tellg();
        if (size <= 0) {
            ifStream.close();
            TE_WARNLOG("file length <= 0, not valid.");
            return false;
        }

        if (size > INT_MAX) {
            ifStream.close();
            TE_WARNLOG("File size %ld is out of limit: %d.", size, INT_MAX);
            return false;
        }
        ifStream.seekg(0, std::ios::beg);

        buffer.resize(size);
        ifStream.read(&buffer[0], size);
        TE_DBGLOG("Release file(%s) handle.", realPath.c_str());
        ifStream.close();
        TE_DBGLOG("Read size:%ld.", size);
    } catch (const std::ifstream::failure& e) {
        TE_WARNLOG("Failed to read file %s. Exception: %s", binFilePath.c_str(), e.what());
        ifStream.close();
        return false;
    }
    return true;
}
}
}
