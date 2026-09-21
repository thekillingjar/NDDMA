/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/scope/scope_fusion_pass_register.h"
#include "common/ge_common/string_util.h"

namespace ge {
std::string ScopeUtil::StringReplaceAll(std::string str, const std::string &old_value, const std::string &new_value) {
  return ge::StringUtils::ReplaceAll(str, old_value, new_value);
}

AscendString ScopeUtil::StringReplaceAll(const char_t *str, const char_t *old_value, const char_t *new_value) {
  std::string tmp_str;
  if (str != nullptr) {
    tmp_str = str;
  }
  std::string tmp_old_value;
  if (old_value != nullptr) {
    tmp_old_value = old_value;
  }
  std::string tmp_new_value;
  if (new_value != nullptr) {
    tmp_new_value = new_value;
  }
  const std::string ret = ge::StringUtils::ReplaceAll(tmp_str, tmp_old_value, tmp_new_value);
  return AscendString(ret.c_str());
}

void ScopeUtil::FreeScopePatterns(ScopeFusionPatterns &patterns) {
  for (auto &batch_pattern : patterns) {
    FreeOneBatchPattern(batch_pattern);
  }
  patterns.clear();
}

void ScopeUtil::FreeOneBatchPattern(std::vector<ScopePattern *> &one_batch_pattern) {
  for (auto &one_pattern : one_batch_pattern) {
    if (one_pattern != nullptr) {
      delete one_pattern;
      one_pattern = nullptr;
    }
  }
  one_batch_pattern.clear();
}
}  // namespace ge
