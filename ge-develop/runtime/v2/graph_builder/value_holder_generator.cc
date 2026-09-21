/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "value_holder_generator.h"
namespace gert {
namespace bg {
std::vector<DevMemValueHolderPtr> CreateDevMemErrorValueHolders(int64_t stream_id, size_t count, const char *fmt, ...) {
  va_list arg;
  va_start(arg, fmt);
  auto holder = DevMemValueHolder::CreateError(stream_id, fmt, arg);
  va_end(arg);
  return {count, holder};
}
std::vector<ValueHolderPtr> CreateErrorValueHolders(size_t count, const char *fmt, ...) {
  va_list arg;
  va_start(arg, fmt);
  auto holder = ValueHolder::CreateError(fmt, arg);
  va_end(arg);
  return {count, holder};
}
std::vector<bg::ValueHolderPtr> ConvertDevMemValueHoldersToValueHolders(
    const std::vector<bg::DevMemValueHolderPtr> &dev_mem_holders) {
  std::vector<bg::ValueHolderPtr> holders(dev_mem_holders.cbegin(), dev_mem_holders.cend());
  return holders;
}
}  // namespace bg
}  // namespace gert
