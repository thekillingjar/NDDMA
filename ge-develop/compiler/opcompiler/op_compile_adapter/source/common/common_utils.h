/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_COMMON_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_COMMON_UTILS_H_

#include <string>
#include <sstream>
#include <sys/time.h>
#include <nlohmann/json.hpp>
#include "inc/te_fusion_log.h"

namespace te {
namespace fusion {
const std::string KERNEL_META = "/kernel_meta";
const std::string KERNEL_META_TEMP = "/kernel_meta_temp_";
// Record the start time of stage.
#define TE_FUSION_TIMECOST_START(stage) int64_t start_usec_##stage = GetMicroSecondTime();

// Print the log of time cost of stage.
#define TE_FUSION_TIMECOST_END(stage, stage_name)                                               \
{                                                                                               \
    int64_t end_usec_##stage = GetMicroSecondTime();                                            \
    TE_INFOLOG("[FE_PERFORMANCE]The time cost of %s is [%ld] micro second.", (stage_name),     \
        (end_usec_##stage - start_usec_##stage));                                               \
}

template<typename T>
std::string GetVectorValueToString(const std::vector<T> &value)
{
    std::ostringstream oss;
    for (const T &val : value) {
        oss << " " << val;
    }
    return oss.str();
}

template<typename K, typename V>
std::string GetMapKeyToString(const std::map<K, V> &value)
{
    std::ostringstream oss;
    bool isFirst = true;
    for (const auto &val : value) {
        if (isFirst) {
            isFirst = false;
        } else {
            oss << ", ";
        }
        oss << val.first;
    }
    return oss.str();
}

int64_t GetMicroSecondTime();

std::string GetCurrAbsPath();

std::vector<std::string> Split(const std::string &str, char pattern);

void DeleteSpaceInString(std::string &str);

bool IsStrEndWith(const std::string &str, const std::string &suffix);

size_t SplitString(const std::string &fullStr, const std::string &separator, std::vector<std::string> &tokenList);

bool IsNameValid(std::string name, std::string sptCh);

bool IsChineseChar(const char ch);

bool IsStrValid(const std::string &str);

bool IsFilePathValid(const std::string &path);

std::string RealPath(const std::string &path);

std::string DirPath(const std::string &filePath);

void ReadFile(const std::string &filepath, std::string &content);

bool CreateFile(const std::string &filePath);

std::string ConstructTempFileName(const std::string &name);

std::string GenerateMachineDockerId();

void Trim(std::string &str);

bool NotZero(const std::vector<int64_t> &validShape);

std::string ShapeToString(const std::vector<int64_t> &shapes);

std::string RangeToString(const std::vector<std::pair<int64_t, int64_t>> &ranges);

void TeErrMessageReport(const std::string &errorCode, std::map<std::string, std::string> &mapArgs);

void TeInnerErrMessageReport(const std::string &errorCode, const std::string &errorMsg);

bool CheckPathValid(const std::string &path, const std::string &pathOwner);

void AssembleJsonPath(const std::string &opsPathNamePrefix, std::string &jsonFilePath, std::string &binFilePath);

bool compareStrings(const std::string& a, const std::string& b);
}  // namespace fusion
}  // namespace te
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_COMMON_COMMON_UTILS_H_
