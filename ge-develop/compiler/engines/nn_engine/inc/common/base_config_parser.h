/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_BASE_CONFIG_PARSER_
#define AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_BASE_CONFIG_PARSER_

#include <map>
#include <memory>
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

namespace fe {
class BaseConfigParser {
public:
  BaseConfigParser() {}
  virtual ~BaseConfigParser() {}

  virtual Status InitializeFromOptions(const std::map<std::string, std::string> &options) {
    (void)options;
    return SUCCESS;
  }
  virtual Status InitializeFromContext() {
    return SUCCESS;
  }
  virtual Status InitializeFromContext(const std::string &combined_params_key) {
    (void)combined_params_key;
    return SUCCESS;
  }
};
using BaseConfigParserPtr = std::shared_ptr<BaseConfigParser>;
}  // namespace fe
#endif  // AIR_COMPILER_GRAPHCOMPILER_ENGINES_NNENG_INC_COMMON_BASE_CONFIG_PARSER_
