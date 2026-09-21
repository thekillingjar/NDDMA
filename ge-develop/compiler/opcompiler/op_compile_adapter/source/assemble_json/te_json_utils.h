/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_JSON_UTILS_H_
#define ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_JSON_UTILS_H_

#include <string>
#include <nlohmann/json.hpp>

namespace te {
namespace fusion {
class TeJsonUtils {
public:
    static void DeleteValuesFromJson(const std::string &key, nlohmann::json &nodeJson);
    static bool CheckIfOutputNode(const std::string &name, const nlohmann::json &jsonData);
    static void GenerateFusionName(const nlohmann::json &jsonData, std::string &fusionName);
    static bool FilterJsonValueFromJsonListByKey(const std::string &key, const std::string &keyValue,
                                                 nlohmann::json &jsonList);
};
}
}
#endif  // ATC_OPCOMPILER_TE_FUSION_SOURCE_ASSEMBLE_JSON_TE_JSON_UTILS_H_