/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATT_UTIL_GLOBAL_OPTIONS_H_
#define ATT_UTIL_GLOBAL_OPTIONS_H_

#include <map>
#include <string>
#include <mutex>
#include "ge_common/ge_api_error_codes.h"

namespace att {
class ThreadLocalContext {
 public:
  ge::Status GetOption(const std::string &key, std::string &option);
  void SetOption(const std::map<std::string, std::string> &options_map);
 private:
  std::map<std::string, std::string> all_options_;
};
ThreadLocalContext &GetThreadLocalContext();
}
#endif  // ATT_UTIL_GLOBAL_OPTIONS_H_
