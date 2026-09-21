/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_OPS_PARALLEL_RULE_JSON_FILE_H
#define AICPU_OPS_PARALLEL_RULE_JSON_FILE_H

#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include "error_code/error_code.h"

namespace aicpu {
const std::string kOpsParallelRule = "ops_parallel_rule";
const std::string kOpNameList = "ops_name_list";
const std::string kList = "list";

struct OpsParallelInfo {
  std::vector<std::string> ops_list;
};

struct RuleInfoDesc {
  std::string rule_name;      // rule name
  OpsParallelInfo rule_info;  // rule info
};

class OpsParallelRuleJsonFile {
 public:
  /**
   * Get instance
   * @return OpsParallelRuleJsonFile reference
   */
  static OpsParallelRuleJsonFile &Instance();

  /**
   * Destructor
   */
  virtual ~OpsParallelRuleJsonFile() = default;

  /**
   * Read json file in specified path(based on source file's current path)
   * @param file_path json file path
   * @param json_read read json handle
   * @return whether read file successfully
   */
  aicpu::State ParseUnderPath(const std::string &file_path, nlohmann::json &json_read) const;

  // Copy prohibit
  OpsParallelRuleJsonFile(const OpsParallelRuleJsonFile &) = delete;
  // Copy prohibit
  OpsParallelRuleJsonFile &operator=(const OpsParallelRuleJsonFile &) = delete;
  // Move prohibit
  OpsParallelRuleJsonFile(OpsParallelRuleJsonFile &&) = delete;
  // Move prohibit
  OpsParallelRuleJsonFile &operator=(OpsParallelRuleJsonFile &&) = delete;

 private:
  // Constructor
  OpsParallelRuleJsonFile() = default;

  /**
   * convert json file format
   * @param json_read Original format json
   * @return whether convert format successfully
   */
  bool ConvertOpsParallelRuleJsonFormat(nlohmann::json &json_read) const;
  bool ParseOpsNameList(const nlohmann::json &json_read, nlohmann::json &ops_name_list) const;
};
                
void from_json(const nlohmann::json &json_read, OpsParallelInfo &ops_rule_info);
void from_json(const nlohmann::json &json_read, RuleInfoDesc &rule_info_desc);
}  // namespace aicpu

#endif
