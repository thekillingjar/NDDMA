/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ops_parallel_rule_json_file.h"
#include "util/log.h"
#include "util/util.h"

using namespace std;
using namespace nlohmann;
using namespace ::aicpu::FWKAdapter;

namespace aicpu {
OpsParallelRuleJsonFile &OpsParallelRuleJsonFile::Instance()
{
  static OpsParallelRuleJsonFile instance;
  return instance;
}

aicpu::State OpsParallelRuleJsonFile::ParseUnderPath(const string &file_path, json &json_read) const
{
  aicpu::State ret = ReadJsonFile(file_path, json_read);
  if (ret.state != ge::SUCCESS) {
    AICPUE_LOGW("Read kernel json file failed, file_path[%s].",
                file_path.c_str());
    return aicpu::State(ge::SUCCESS);
  }
  // inner error
  return ConvertOpsParallelRuleJsonFormat(json_read) ? aicpu::State(ge::SUCCESS) : aicpu::State(ge::FAILED);
}

bool OpsParallelRuleJsonFile::ConvertOpsParallelRuleJsonFormat(json &json_read) const
{
  AICPUE_LOGI("Start convert ops parallel rule json.");
  for (auto it = json_read.cbegin(); it != json_read.cend(); ++it) {
    json new_json = it.value();
    string ops_parallel_rule = it.key();
    if (strncmp(ops_parallel_rule.c_str(), kOpsParallelRule.c_str(), kOpsParallelRule.size()) != 0) {
      continue;
    }
    new_json[kOpsParallelRule] = ops_parallel_rule;
    json ops_name_list;
    bool ret = ParseOpsNameList(it.value(), ops_name_list);
    AICPU_IF_BOOL_EXEC((!ret),
        AICPU_REPORT_INNER_ERR_MSG("Call OpsParallelRuleJsonFile::ParseOpsNameList failed.");
        return false)
    new_json[kOpNameList] = ops_name_list;
    json_read = new_json;
    AICPUE_LOGI("Convert ops parallel json [rule:%s]success.", kOpsParallelRule.c_str());
    return true;
  }

  AICPUE_LOGI("Convert ops parallel json [rule:%s] fail.", kOpsParallelRule.c_str());
  return false;
}

bool OpsParallelRuleJsonFile::ParseOpsNameList(const json &json_read, json &ops_name_list) const
{
  json list_result;
  const string ops_list_str = kOpNameList;
  for (json::const_iterator iter = json_read.cbegin(); iter != json_read.cend(); ++iter) {
    const string key = iter.key();
    if ((strncmp(key.c_str(), ops_list_str.c_str(), ops_list_str.size()) == 0)) {
      json json_key = json_read[key];
      auto iter_list = json_key.find(kList);
      if (iter_list != json_key.end()) {
        string list = iter_list.value().get<string>();
        AICPUE_LOGI("ops parallel name list is [%s].", list.c_str());
        list_result[key] = list;
      }
    }
  }
  ops_name_list = list_result;
  return true;
}

template <typename T>
inline void Assignment(T &varible, const string &key, const json &json_read)
{
  auto iter = json_read.find(key);
  if (iter != json_read.end()) {
    varible = iter.value().get<T>();
  }
}

void from_json(const nlohmann::json &json_read, OpsParallelInfo &ops_rule_info)
{
  auto iter = json_read.find(kOpNameList);
  if (iter == json_read.end()) {
    return;
  }
  json list_json = iter.value();
  for (json::iterator it = list_json.begin(); it != list_json.end(); ++it) {
    string ops_list = it.value().get<string>();
    ops_list += ",";
    size_t len = ops_list.size();
    size_t left = 0;
    size_t right = 0;
    while (right < len) {
      while (ops_list[right] != ',') {
        right++;
      }
      ops_rule_info.ops_list.push_back(ops_list.substr(left, right - left));
      right++;
      left = right;
    }
  }
}

void from_json(const nlohmann::json &json_read, RuleInfoDesc &rule_info_desc)
{
  Assignment(rule_info_desc.rule_name, kOpsParallelRule, json_read);
  Assignment(rule_info_desc.rule_info, kOpNameList, json_read);
}
}  // namespace aicpu
