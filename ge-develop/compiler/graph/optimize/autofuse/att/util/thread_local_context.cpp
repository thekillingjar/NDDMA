/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "thread_local_context.h"

namespace att {
ThreadLocalContext &GetThreadLocalContext() {
  static thread_local ThreadLocalContext thread_context;
  return thread_context;
}

ge::Status ThreadLocalContext::GetOption(const std::string &key, std::string &option) {
  const std::map<std::string, std::string>::const_iterator global_iter = all_options_.find(key);
  if (global_iter != all_options_.end()) {
    option = global_iter->second;
    return ge::SUCCESS;
  }
  return ge::FAILED;
}

void ThreadLocalContext::SetOption(const std::map<std::string, std::string> &options_map) {
  all_options_ = options_map;
}
}