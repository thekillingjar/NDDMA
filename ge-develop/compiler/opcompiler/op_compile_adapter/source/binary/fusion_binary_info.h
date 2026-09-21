/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_FUSION_BINARY_INFO_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_BINARY_FUSION_BINARY_INFO_H_

#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <nlohmann/json.hpp>

namespace te {
namespace fusion {
constexpr const char *DTYPE_MODE_NORMAL = "normal";
constexpr const char *DTYPE_MODE_BIT = "bit";
constexpr const char *DTYPE_MODE_BOOL = "bool";
constexpr const char *FORMAT_MODE_NORMAL = "normal";
constexpr const char *FORMAT_MODE_AGNOSTIC = "agnostic";
constexpr const char *FORMAT_MODE_ND_AGNOSTIC = "nd_agnostic";
constexpr const char *FORMAT_STATIC_ND_AGNOSTIC = "static_nd_agnostic";

struct DtypeFormatMode {
    std::string opType;
    unsigned int count;
    std::vector<std::string> dTypeMode;
    std::vector<std::string> formatMode;
};

enum class SimpleKeyModeType {SIMPLE_MODE, COMPATIBLE_MODE, CUSTOM_MODE};
enum class DynamicParamModeEnum {
    UNDEFINED = 0,
    FOLDED_WITH_DESC,
    UNFOLDED
};

class BinaryInfoBase {
public:
    BinaryInfoBase() {};
    BinaryInfoBase(const BinaryInfoBase&) = default;
    BinaryInfoBase& operator=(const BinaryInfoBase&) = default;
    ~BinaryInfoBase() {};

    bool ParseBinaryInfoFile(const std::string &binaryConfigPath, const bool &isSuperKernel);
    bool MatchSimpleKey(const std::string opType, const std::string simpleKey, std::string &jsonFilePath) const;
    bool GetInputMode(const std::string &opType, DtypeFormatMode &mode) const;
    bool GetOutputMode(const std::string &opType, DtypeFormatMode &mode) const;
    bool GetSimpleKeyMode(const std::string &opType, SimpleKeyModeType &simplekeyMode) const;
    bool GetOptionalInputMode(const std::string &opType, int &optionalInputMode) const;
    bool GetDynParamModeByOpType(const std::string &opType, DynamicParamModeEnum &dynParamMode) const;
    bool GetOppBinFlag() const;
    void SetOppBinFlag(bool flag);

private:
    void GetAllOpsPathNamePrefix(const std::string &binaryConfigJsonPath,
                                 std::vector<std::string> &binaryConfigPathPrefix);
    void InsertBinaryInfoIntoMap(const std::string opType, const SimpleKeyModeType simpleKeyMode,
                                const std::string &optionalInputMode, const std::string &dynamicParamMode,
                                const std::unordered_map<std::string, std::string> &simpleKeyInfoMap);
    bool GetBinaryInfoConfigPath(const std::string &binaryConfigPath, std::string &configRealPath,
                                 const bool &isSuperKernel, const std::string &path) const;
    bool ReadOpBinaryInfoConfig(const std::string &binaryInfoConfigPath, nlohmann::json &binaryInfoConfig) const;
    bool GenerateBinaryInfo(nlohmann::json &binaryInfoConfig);
    bool GenerateSimpleKeyList(const std::string &opType, const nlohmann::json &binaryList,
                               std::unordered_map<std::string, std::string>  &simpleKeyInfoMap) const;
    bool GenerateInOutPutMode(const std::string opType, const std::string &type,
                              const nlohmann::json &binaryInfoParams, DtypeFormatMode &mode) const;
    bool GenerateDtypeFormatMode(const std::string &opType, const nlohmann::json &binaryInfoParams);
    bool GetParamModeFromJson(const nlohmann::json &jsonValue, const std::string &opType, const std::string &key,
                              std::string &paramMode) const;

    std::unordered_map<std::string, SimpleKeyModeType> simpleKeyModeMap_{};
    std::unordered_map<std::string, int> optionalInputMode_{};    // gen_placeholder: 0   no_placeholder: 1 (default)
    std::unordered_map<std::string, DynamicParamModeEnum> opDynParamMode_{};
    std::unordered_map<std::string, DtypeFormatMode> inputMode_{};
    std::unordered_map<std::string, DtypeFormatMode> outputMode_{};

    mutable std::mutex simpleKeyMutex_;
    // <opType, <simpleKey, jsonpath>>
    std::unordered_map<std::string,  std::unordered_map<std::string, std::string>> simpleKeyMap_{};
    bool oppBinFlag_ = false;      // use opp_latest binary version
};
using BinaryInfoBasePtr = std::shared_ptr<BinaryInfoBase>;
} // namespace fusion
} // namespace te
#endif

