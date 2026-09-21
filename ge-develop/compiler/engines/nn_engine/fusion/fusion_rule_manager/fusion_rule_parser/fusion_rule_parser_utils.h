/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_PARSER_UTILS_H_
#define FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_PARSER_UTILS_H_

#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <mutex>
#include "common/fe_utils.h"
#include "graph/utils/attr_utils.h"
#include "common/opskernel/ops_kernel_info_store.h"
#include "common/opskernel/ops_kernel_info_types.h"
#include "common/util/json_util.h"

namespace fe {
Status GetFromJson(const nlohmann::json &json_object, std::string &str);

bool AnalyseCheckMap(const std::map<std::string, bool> &check_map);

/*
 * @brief: Split input string by format "key:value"
 */
Status SplitKeyValueByColon(const std::string &str, std::string &key, std::string &value);

/*
 * @brief: Split input string by format "key.value"
 */
Status SplitKeyValueByDot(const std::string &str, std::string &key, std::string &value);

Status StringToInt(const std::string &str, int &value);

Status StringToInt64(const std::string &str, int64_t &value);

Status StringToFloat(const std::string &str, float &value);
/*
 * @brief: Get value stored in GeAttrValue, and convert it to string.
 */
std::string GetStrFromAttrValue(ge::GeAttrValue &attr_value);

std::string GetStrOfInt(ge::GeAttrValue &attr_value);

std::string GetStrOfFloat(ge::GeAttrValue &attr_value);

std::string GetStrOfBool(ge::GeAttrValue &attr_value);

std::string GetStrOfListInt(ge::GeAttrValue &attr_value);

std::string GetStrOfListFloat(ge::GeAttrValue &attr_value);

std::string GetStrOfListBool(ge::GeAttrValue &attr_value);
class FusionRuleParserUtils {
 public:
  FusionRuleParserUtils() : engine_name_("") {}

  ~FusionRuleParserUtils() {}

  static FusionRuleParserUtils *Instance() {
    static FusionRuleParserUtils instance;
    return &instance;
  }

  void SetEngineName(const std::string &name) {
    std::lock_guard<std::mutex> lock_guard(parse_mutex_);
    engine_name_ = name;
  }

  std::string GetEngineName() {
    std::lock_guard<std::mutex> lock_guard(parse_mutex_);
    return engine_name_;
  }

 private:
  std::string engine_name_;
  std::mutex parse_mutex_;
};

}  // namespace fe

#endif  // FUSION_ENGINE_FUSION_FUSION_RULE_MANAGER_FUSION_RULE_PARSER_FUSION_RULE_PARSER_UTILS_H_
