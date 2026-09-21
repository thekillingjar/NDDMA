/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_PRELOAD_TASK_INFO_PRE_TASK_STATUS_H_
#define GE_COMMON_PRELOAD_TASK_INFO_PRE_TASK_STATUS_H_
#include "graph/types.h"
#include "common/checker.h"

namespace ge {
ge::char_t *CreateMessage(const ge::char_t * const format, va_list arg);
class PreTaskStatus {
 public:
  bool IsSuccess() const {
    return status_ == nullptr;
  }
  const char_t *GetErrorMessage() const noexcept {
    return status_;
  }
  PreTaskStatus() = default;
  ~PreTaskStatus() {
    delete[] status_;
  }
  PreTaskStatus(const PreTaskStatus &other);
  PreTaskStatus(PreTaskStatus &&other) noexcept;
  PreTaskStatus &operator=(const PreTaskStatus &other);
  PreTaskStatus &operator=(PreTaskStatus &&other) noexcept;

  static PreTaskStatus Success();
  static PreTaskStatus ErrorStatus(const char_t *message, ...) __attribute__((format(printf, 1, 2)));
 private:
  char_t *status_ = nullptr;
};
}  // namespace ge
#endif  // GE_COMMON_PRELOAD_TASK_INFO_PRE_TASK_STATUS_H_