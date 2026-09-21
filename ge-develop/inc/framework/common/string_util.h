/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INC_FRAMEWORK_COMMON_STRING_UTIL_H_
#define INC_FRAMEWORK_COMMON_STRING_UTIL_H_
#include "common/ge_common/string_util.h"
namespace ge {
template<typename Iterator>
static std::string StrJoin(Iterator begin, Iterator end, const std::string &separator) {
  if (begin == end) {
    return "";
  }
  std::stringstream str_stream;
  str_stream << *begin;
  for (Iterator it = std::next(begin); it != end; ++it) {
    str_stream << separator << *it;
  }
  return str_stream.str();
}
}
#endif  // INC_FRAMEWORK_COMMON_STRING_UTIL_H_
