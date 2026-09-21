/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "assemble_json/te_json_utils.h"
#include "inc/te_fusion_log.h"
#include "inc/te_fusion_check.h"

namespace te {
namespace fusion {
void TeJsonUtils::DeleteValuesFromJson(const std::string &key, nlohmann::json &nodeJson)
{
    if (nodeJson.is_null()) {
        return;
    }
    TE_DBGLOG("before deleting caxis json: %s.", nodeJson.dump().c_str());
    bool erase_flag = true;
    while (erase_flag) {
        erase_flag = false;
        for (auto &desc : nodeJson) {
            if (desc.is_null()) {
                continue;
            }
            if (desc.contains(key)) {
                TE_FUSION_LOG_EXEC(TE_FUSION_LOG_DEBUG, "has %s.", key.c_str());
                desc.erase(key);
                erase_flag = true;
                break;
            }
        }
        if (!erase_flag) {
            break;
        }
    }
    TE_DBGLOG("after delete caxis json: %s", nodeJson.dump().c_str());
}

bool TeJsonUtils::CheckIfOutputNode(const std::string &name, const nlohmann::json &jsonData)
{
    if (jsonData.find("op_list") == jsonData.end()) {
        return false;
    }
    for (const auto &opNode : jsonData["op_list"]) {
        if (opNode.find("input_desc") == opNode.end()) {
            continue;
        }
        for (const auto &input : opNode["input_desc"]) {
            if (name == input["name"]) {
                return true;
            }
        }
    }

    return false;
}

void TeJsonUtils::GenerateFusionName(const nlohmann::json &jsonData, std::string &fusionName)
{
    if (jsonData.find("op_list") == jsonData.end()) {
        return;
    }
    for (auto &opNode : jsonData["op_list"]) {
        // only deal node whose type is not Data
        if (opNode.find("type") != opNode.end()) {
            std::string opType = opNode["type"];
            if (opType == "Data") {
                continue;
            }
        }
        if (opNode.find("func_name") != opNode.end()) {
            std::string funcName = opNode["func_name"];
            fusionName += "_" + funcName;
        }
    }
    TE_DBGLOG("Fusion node name is [%s].", fusionName.c_str());
}

bool TeJsonUtils::FilterJsonValueFromJsonListByKey(const std::string &key, const std::string &keyValue,
                                                   nlohmann::json &jsonList)
{
    TE_DBGLOGF("Filter json value by key %s, and key value is %s", key.c_str(), keyValue.c_str());

    for (auto nodeIter = jsonList.begin(); nodeIter != jsonList.end();) {
        auto field = nodeIter->find(key);
        if (field == nodeIter->end()) {
            nodeIter = jsonList.erase(nodeIter);
            continue;
        }
        std::string binKeyVal = field.value().get<std::string>();
        if (binKeyVal.find(keyValue) == std::string::npos) {
            nodeIter = jsonList.erase(nodeIter);
        } else {
            ++nodeIter;
            TE_DBGLOGF("Bin static key(%s) equals or contains op static key(%s)", binKeyVal.c_str(), keyValue.c_str());
        }
    }

    return !jsonList.empty();
}
}
}