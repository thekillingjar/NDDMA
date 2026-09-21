/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/common_utils.h"
#include <chrono>
#include <vector>
#include <map>
#include <climits>
#include <unistd.h>
#include <fstream>
#include <libgen.h>
#include <dirent.h>
#include <netdb.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sstream>
#include <cerrno>
#include "inc/te_fusion_check.h"
#include "inc/te_fusion_util_constants.h"
#include "base/err_msg.h"

namespace te {
namespace fusion {
const char *const REASON = "The path can only contain 'a-z' 'A-Z' '0-9' '-' '.' '_' and chinese character";
int64_t GetMicroSecondTime()
{
    struct timeval tv = {0, 0};
    int ret = gettimeofday(&tv, nullptr);
    if (ret != 0) {
        return 0;
    }
    if (tv.tv_sec < 0 || tv.tv_usec < 0) {
        return 0;
    }
    int64_t microMultiples = 1000000;
    int64_t second = tv.tv_sec;
    TE_FUSION_UINT64_MULCHECK(second, microMultiples);
    int64_t secondToMicro = second * microMultiples;
    TE_FUSION_INT64_ADDCHECK(secondToMicro, tv.tv_usec);
    return secondToMicro + tv.tv_usec;
}

std::string GetCurrAbsPath()
{
    char* curr_path = get_current_dir_name();
    if (curr_path == nullptr) {
        return ".";
    }
    std::string currPath(curr_path);
    free(curr_path);
    curr_path = nullptr;
    return currPath;
}

std::vector<std::string> Split(const std::string &str, char pattern)
{
    std::vector<std::string> resVev;
    if (str.empty()) {
        return resVev;
    }
    std::string strAndPattern = str + pattern;
    size_t pos = strAndPattern.find(pattern);
    size_t size = strAndPattern.size();
    while (pos != std::string::npos) {
        std::string subStr = strAndPattern.substr(0, pos);
        resVev.push_back(subStr);
        strAndPattern = strAndPattern.substr(pos + 1, size);
        pos = strAndPattern.find(pattern);
    }
    return resVev;
}

size_t SplitString(const std::string &fullStr, const std::string &separator, std::vector<std::string> &tokenList)
{
    size_t posA = 0;
    size_t posB = 0;
    std::string token;

    while ((posB = fullStr.find(separator, posA)) != std::string::npos) {
        token = fullStr.substr(posA, posB - posA);
        tokenList.emplace_back(token);
        posA = posB + separator.size();
    }

    token = fullStr.substr(posA);
    if (token.size() > 0) {
        tokenList.emplace_back(token);
    }
    return tokenList.size();
}

void DeleteSpaceInString(std::string &str)
{
    bool loop = true;
    while (loop) {
        size_t pos = str.find(" ");
        if (pos == std::string::npos) {
            loop = false;
        } else {
            str.erase(pos, 1);
        }
    }
}

bool IsStrEndWith(const std::string &str, const std::string &suffix)
{
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length()) == suffix;
}

bool IsChineseChar(const char ch)
{
    return (static_cast<unsigned char>(ch) & 0x80);
}

// A regular matching expression to verify the validity of the input file path
// Path section: Support upper and lower case letters, numbers dots(.) / - _ chinese
bool IsStrValid(const std::string &str)
{
    TE_DBGLOG("Check path [%s].", str.c_str());
    size_t i = 0;
    bool hasInvalidChar = false;
    const size_t CHINESE_CHAR_LENGTH = 2;
    while (i < str.size() && !hasInvalidChar) {
        if (IsChineseChar(str[i])) {
            i += CHINESE_CHAR_LENGTH;
        } else {
            bool check = (isdigit(str[i]) || isalpha(str[i]) || str[i] == '.'
                          || str[i] == '-' || str[i] == '_' || str[i] == '/');
            if (!check) {
                TE_DBGLOG("Invalid char[%c].", str[i]);
                hasInvalidChar = true;
            }
            i++;
        }
    }
    return !hasInvalidChar;
}

bool IsFilePathValid(const std::string &path)
{
    if (path.empty()) {
        TE_WARNLOG("Path value is NULL");
        return false;
    }

    if (!IsStrValid(path)) {
        TE_WARNLOG("Path[%s] is invalid: %s.", path.c_str(), REASON);
        return false;
    }
    return true;
}

std::string RealPath(const std::string &path)
{
    if (path.empty()) {
        TE_DBGLOG("Path is NULL.");
        return "";
    }

    if (path.size() >= PATH_MAX) {
        TE_DBGLOG("Path[%s] is too long.", path.c_str());
        return "";
    }

    char resovedPath[PATH_MAX] = {0};
    std::string res = "";

    if (realpath(path.c_str(), resovedPath) != nullptr) {
        res = resovedPath;
    } else {
        TE_DBGLOG("Path[%s] does not exist.", path.c_str());
    }
    return res;
}

/**
 * @brief: get dir path of the file
 * @param [in] filePath: path of the file
 */
std::string DirPath(const std::string &filePath)
{
    if (filePath.empty()) {
        TE_DBGLOG("Path is NULL.");
        return "";
    }

    if (filePath.size() >= PATH_MAX) {
        TE_DBGLOG("Path[%s] is too long.", filePath.c_str());
        return "";
    }

    char bufFilePath[PATH_MAX + 1] = {0};
    strncpy_s(bufFilePath, PATH_MAX, filePath.c_str(), std::strlen(filePath.c_str()));
    std::string res = dirname(bufFilePath);

    return res;
}

bool CreateFile(const std::string &filePath)
{
    TE_DBGLOG("Create file %s", filePath.c_str());
    std::string realPath = RealPath(filePath);
    if (realPath.empty()) {
        FILE* fp = fopen(filePath.c_str(), "w+");
        if (fp == nullptr) {
            TE_WARNLOG("Open file[%s] failed.", filePath.c_str());
            return false;
        }
        int res = chmod(filePath.c_str(), FILE_AUTHORITY);
        if (res == -1) {
            TE_WARNLOG("Update file[%s] authority failed.", filePath.c_str());
            fclose(fp);
            return false;
        }
        fclose(fp);
    }
    return true;
}

std::string ConstructTempFileName(const std::string &name)
{
    // create temp file
    std::string pid = std::to_string(static_cast<int>(getpid()));
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t ts = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
    std::ostringstream oss;
    oss << name << "_" << pid << "_" << ts << "_" << rand() << ".temp";
    oss.flush();
    return oss.str();
}

void ReadFile(const std::string &filepath, std::string &content)
{
    std::ifstream idfile(filepath);
    if (idfile.is_open()) {
        std::string id;
        getline(idfile, id);
        if (!id.empty()) {
            content = id;
            return;
        }
    }
}

/**
 * @brief check name valid or not
 * @param [in] name                     the string need to check
 * @param [in] sptCh                    support char set
 * @return [out] bool                   the check result
 */
bool IsNameValid(std::string name, std::string sptCh)
{
    TE_FUSION_CHECK((name == ""), {
        REPORT_TE_INNER_ERROR("Parameter name is empty, which is an error.");
        return false;
    });

    for (char chId : name) {
        if (sptCh.find(chId) != std::string::npos) {
            continue;
        }
        TE_FUSION_CHECK(!((chId >= '0' && chId <= '9') || (chId >= 'a' && chId <= 'z') ||
                          (chId >= 'A' && chId <= 'Z') || (chId == '_') || (chId == '-') ||
                          (chId == '>')),
                        {
                            REPORT_TE_INNER_ERROR("Check parameter[%s] error, it's char should be '0'~'9' or 'a'~'z' or 'A'~'Z' or '_' or '-'.", name.c_str());
                            return false;
                        });
    }
    return true;
}

std::string GenerateMachineDockerId()
{
    std::string machineId;
    std::string dockerId;
    std::string machineDockerId;
    try {
        ReadFile("/etc/machine-id", machineId);
        if (!machineId.empty()) {
            machineDockerId = machineId;
            TE_DBGLOG("Machine-Id is[%s]", machineId.c_str());
        }
        ReadFile("/proc/self/cgroup", dockerId);
        if (!dockerId.empty()) {
            char separator = '/';
            size_t pos = dockerId.rfind(separator);
            if (pos != std::string::npos) {
                dockerId = dockerId.substr(pos + 1);
                TE_DBGLOG("dockerId is[%s]", dockerId.c_str());
                machineDockerId = machineDockerId + dockerId;
            }
        }
    } catch (const std::exception &e) {
        TE_WARNLOG("Read File failed, reason %s", e.what());
        return "";
    }
    TE_DBGLOG("Machine-Id and Docker-Id is [%s]", machineDockerId.c_str());
    return machineDockerId;
}

void Trim(std::string &str)
{
    if (str.empty()) {
        return;
    }
    // trim Tabs , spaces at the beginning and end of the string
    str.erase(0, str.find_first_not_of(" \t"));
    str.erase(str.find_last_not_of(" \t") + 1);
    return;
}

bool NotZero(const std::vector<int64_t> &validShape)
{
    if (validShape.empty()) {
        return false;
    }
    for (auto dim : validShape) {
        if (dim == 0) {
            return false;
        }
    }
    return true;
}

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges)
{
    bool first = true;
    std::stringstream ss;
    ss << "[";
    for (const auto &range : ranges) {
        if (first) {
            first = false;
        } else {
            ss << ",";
        }
        ss << "{";
        ss << range.first << "," << range.second;
        ss << "}";
    }
    ss << "]";
    return ss.str();
}

std::string ShapeToString(const std::vector<int64_t> &shapes)
{
    bool first = true;
    std::stringstream ss;
    ss << "[";
    for (const auto &shape : shapes) {
        if (first) {
            first = false;
        } else {
            ss << ",";
        }
        ss << shape;
    }
    ss << "]";
    return ss.str();
}

/**
 * @brief: report TeFusion Error to error manager
 * @param [in] errorCode: errorCode error_code.json
 * @param [in] mapArgs: error info with map
 */
void TeErrMessageReport(const std::string &errorCode, std::map<std::string, std::string> &mapArgs)
{
    std::vector<const char *> keyVec;
    std::vector<const char *> valueVec;
    for (const auto &pair : mapArgs) {
      keyVec.push_back(pair.first.c_str());
      valueVec.push_back(pair.second.c_str());
    }

    int reportRes = REPORT_PREDEFINED_ERR_MSG(errorCode.c_str(), keyVec, valueVec);
    TE_FUSION_CHECK(reportRes == -1,
                    {TE_FUSION_LOG_EXEC(TE_FUSION_LOG_WARNING, "Report warning message failed.");
                    });
}

void TeInnerErrMessageReport(const std::string &errorCode, const std::string &errorMsg)
{
    REPORT_INNER_ERR_MSG(errorCode.c_str(), errorMsg.c_str());
}

bool CheckPathValid(const std::string &path, const std::string &pathOwner)
{
    std::map<std::string, std::string> mapArgs = {{"path", path}, {"arg", pathOwner}};
    if (path.empty()) {
        TE_ERRLOG("Path is none, and it's error.");
        mapArgs.emplace(std::pair<std::string, std::string>("result", "check path failed"));
        mapArgs.emplace(std::pair<std::string, std::string>("reason", "the path is empty"));
        TeErrMessageReport(EM_PATH_INVALID_ERROR, mapArgs);
        return false;
    }

    if (strlen(path.c_str()) >= PATH_MAX) {
        std::string failedReason = "path length exceeded the maximum value " + std::to_string(PATH_MAX) + "";
        mapArgs.emplace(std::pair<std::string, std::string>("result", "check path length failed"));
        mapArgs.emplace(std::pair<std::string, std::string>("reason", failedReason));
        TeErrMessageReport(EM_PATH_INVALID_ERROR, mapArgs);
        return false;
    }

    // Return absolute path when path is accessible
    std::string res;
    char resolvedPath[PATH_MAX] = {0};
    if (realpath(path.c_str(), resolvedPath) != nullptr) {
        res = resolvedPath;
    }

    if (res.empty()) {
        mapArgs.emplace(std::pair<std::string, std::string>("result", "real path get failed"));
        mapArgs.emplace(std::pair<std::string, std::string>("reason",
                                                            "the path does not exist or its access permission is denied"));
        TeErrMessageReport(EM_PATH_INVALID_ERROR, mapArgs);
        TE_ERRLOG("Path[%s] does not exist or has no permission to access.", path.c_str());
        return false;
    }

    if (access(res.c_str(), W_OK | F_OK) != 0) {
        mapArgs.emplace(std::pair<std::string, std::string>("result", "access real path failed"));
        mapArgs.emplace(std::pair<std::string, std::string>("reason", strerror(errno)));
        TeErrMessageReport(EM_PATH_INVALID_ERROR, mapArgs);
        TE_ERRLOG("Path[%s] has no permission to write.", res.c_str());
        return false;
    }

    return true;
}

void AssembleJsonPath(const std::string &opsPathNamePrefix, std::string &jsonFilePath, std::string &binFilePath) {
    const size_t NUM_2 = 2;
    TE_DBGLOG("Before revert opsPathNamePrefix is %s, jsonFilePath is %s, binFilePath is %s",
              opsPathNamePrefix.c_str(), jsonFilePath.c_str(),binFilePath.c_str());
    if (opsPathNamePrefix == "") {
        TE_DBGLOG("After revert opsPathNamePrefix is %s, jsonFilePath is %s, binFilePath is %s",
                  opsPathNamePrefix.c_str(), jsonFilePath.c_str(),binFilePath.c_str());
        jsonFilePath = jsonFilePath + binFilePath;
        return;
    }
    std::stringstream ss(binFilePath);
    std::string token;
    std::vector<std::string> tokens;

    while(std::getline(ss, token, '/')) {
        if (!token.empty()) {
            tokens.emplace_back(token);
        }
    }

    if (tokens.size() < NUM_2) {
        return;
    }
    std::string part = tokens[1];
    if (part == opsPathNamePrefix) {
        jsonFilePath = jsonFilePath + binFilePath;
    } else {
        std::string tempPath = tokens[0] + "/" + opsPathNamePrefix;
        for (size_t i = 1; i < tokens.size(); i++) {
            tempPath += "/" + tokens[i];
        }
        jsonFilePath = jsonFilePath + tempPath;
    }
    TE_DBGLOG("After revert opsPathNamePrefix is %s, jsonFilePath is %s, binFilePath is %s",
              opsPathNamePrefix.c_str(), jsonFilePath.c_str(),binFilePath.c_str());
    return;
}

bool compareStrings(const std::string& a, const std::string& b) {
    if (a.empty()) {
        return true;
    }
    if (b.empty()) {
        return false;
    }
    bool alsOpsLegacy = a.find("ops_legacy") == 0;
    bool blsOpsLegacy = b.find("ops_legacy") == 0;
    if (alsOpsLegacy && !blsOpsLegacy) {
        return true;
    }
    if (!alsOpsLegacy && blsOpsLegacy) {
        return false;
    }
    return a.length() < b.length();
}
} // namespace fusion
} // namespace te
