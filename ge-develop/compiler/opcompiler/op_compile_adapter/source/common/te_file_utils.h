/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_FILE_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_FILE_UTILS_H_

#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace te {
namespace fusion {
class TeFileUtils {
public:
    static bool CreateMultiLevelDir(const std::string &directoryPath);
    static bool CopyDirFileToNewDir(const std::string &dirPath, const std::string &newDirPath);
    static bool IsObjFileExsit(const std::string &opName, const std::string &jsonFilePath);
    static bool CopyFileToCacheDir(const std::string &srcPath, const std::string &dstPath);
    static void DeleteFile(const std::string &path);
    static bool WriteNewFileInCache(const std::string &filePath, const std::string &fileStr);
    static bool DeleteFilesInDir(const std::string &dirPath, bool needCheckUsing, const std::string &fileToDelete = "",
                                 const std::string &fileToReserve = "");
    static bool FcntlLockFileSet(int fd, int type, int recursiveCnt);
    static FILE* LockFile(const std::string &filePath, bool needRepeatTry, int lockType);
    static void ReleaseLockFile(FILE *fp, const std::string &path);
    static bool UpdateFileAccessTime(const std::string &path);
    static int64_t GetDirectorySize(const std::string& path);
    static bool GetJsonValueFromJsonFile(const std::string &jsonFilePath, nlohmann::json &jsonValue);
    static bool GetBufferFromBinFile(const std::string &binFilePath, std::vector<char> &buffer);

private:
    static bool IsFileUsed(std::string &filePath);
    static bool IsFileFcntlLock(int fd);
    static bool JudgeEmptyAndCreateDir(char tmpDirPath[], const std::string &directoryPath);
    static bool CopyFileToNewPath(const std::string &filePath, const std::string &dstPath);
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_TE_FILE_UTILS_H_