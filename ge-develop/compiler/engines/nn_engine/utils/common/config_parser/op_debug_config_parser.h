/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_DEBUG_CONFIG_PARSER_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_DEBUG_CONFIG_PARSER_

#include <vector>
#include <string>
#include <set>
#include <mutex>
#include "common/base_config_parser.h"
#include "common/fe_type_utils.h"
#include "common/platform_utils.h"
#include "common/fe_report_error.h"
#include "ge_common/ge_api_types.h"
#include "common/aicore_util_constants.h"
#include "framework/common/ge_types.h"

namespace fe {
class OpDebugConfigParser : public BaseConfigParser {
 public:
  OpDebugConfigParser();
  explicit OpDebugConfigParser(const string &npu_collect_path);
  ~OpDebugConfigParser() override;

  Status InitializeFromOptions(const std::map<std::string, std::string> &options) override;
  using BaseConfigParser::InitializeFromContext;
  Status InitializeFromContext() override;
  std::set<std::string> GetOpDebugListName() const;
  std::set<std::string> GetOpDebugListType() const;
  std::string GetOpDebugConfig() const;
  bool IsNeedMemoryCheck() const;
  bool IsOpDebugListOp(const ge::OpDescPtr &op_desc_ptr) const;
  void SetOpDebugConfigEnv(const std::string &env);
  std::string npu_collect_path_;

 private:
  bool SetOpDebugList(const std::string &op_debug_list, const std::string &file_path);
  bool SetOpdebugConfig(const std::string &file_path);
  bool GetOpdebugValue(const std::string &line, std::vector<std::string> &res, const std::string &file_path);
  void AssembleOpDebugConfigInfo(std::vector<std::string> &value_vec);
  void AssembleOpDebugListInfo(std::vector<std::string> &value_list_vec);
  std::string op_debug_config_{};
  std::set<std::string> op_debug_list_name_{};
  std::set<std::string> op_debug_list_type_{};
  bool enable_op_memory_check_{false};
};
using OpDebugConfigParserPtr = std::shared_ptr<OpDebugConfigParser>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_UTILS_COMMON_OP_IMPL_MODE_CONFIG_PARSER_
