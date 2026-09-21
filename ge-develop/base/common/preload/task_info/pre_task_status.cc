/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/preload/task_info/pre_task_status.h"
#include <cstring>
#include <memory>
#include <securec.h>

namespace ge {
ge::char_t *CreateMessage(const ge::char_t * const format, va_list arg) {
  GE_ASSERT_TRUE(!(format == nullptr));

  va_list arg_copy;
  va_copy(arg_copy, arg);
  const int len = vsnprintf(nullptr, 0, format, arg_copy);
  va_end(arg_copy);

  GE_ASSERT_TRUE(!(len < 0));

  auto task_msg = std::unique_ptr<ge::char_t[]>(new (std::nothrow) ge::char_t[static_cast<size_t>(len) + 1U]);
  GE_ASSERT_TRUE(!(task_msg == nullptr));

  const auto ret = vsnprintf_s(task_msg.get(), len + 1, len, format, arg);
  GE_ASSERT_TRUE(!(ret < 0));

  return task_msg.release();
}

PreTaskStatus::PreTaskStatus(const PreTaskStatus &other) : status_{nullptr} {
  *this = other;
}

PreTaskStatus &PreTaskStatus::operator=(const PreTaskStatus &other) {
  delete [] status_;
  if (other.status_ == nullptr) {
    status_ = nullptr;
  } else {
    const size_t status_len = strlen(other.status_) + 1U;
    status_ = new (std::nothrow) ge::char_t[status_len];
    if (status_ != nullptr) {
      const auto ret = strcpy_s(status_, status_len, other.status_);
      if (ret != EOK) {
        status_[0] = '\0';
      }
    }
  }
  return *this;
}

PreTaskStatus::PreTaskStatus(PreTaskStatus &&other) noexcept {
  status_ = other.status_;
  other.status_ = nullptr;
}

PreTaskStatus &PreTaskStatus::operator=(PreTaskStatus &&other) noexcept {
  delete [] status_;
  status_ = other.status_;
  other.status_ = nullptr;
  return *this;
}

PreTaskStatus PreTaskStatus::Success() {
  return {};
}

PreTaskStatus PreTaskStatus::ErrorStatus(const char_t *message, ...) {
  PreTaskStatus status;
  va_list arg;
  va_start(arg, message);
  status.status_ = CreateMessage(message, arg);
  va_end(arg);
  return status;
}
}  // namespace ge