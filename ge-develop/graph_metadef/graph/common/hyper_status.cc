/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/hyper_status.h"

#include <cstring>
#include <memory>
#include <securec.h>

namespace gert {
ge::char_t *CreateMessage(const ge::char_t *format, va_list arg) {
  if (format == nullptr) {
    return nullptr;
  }

  va_list arg_copy;
  va_copy(arg_copy, arg);
  int len = vsnprintf(nullptr, 0, format, arg_copy);
  va_end(arg_copy);

  if (len < 0) {
    return nullptr;
  }

  auto msg = std::make_unique<ge::char_t[]>(len + 1);
  if (msg == nullptr) {
    return nullptr;
  }

  auto ret = vsnprintf_s(msg.get(), len + 1, len, format, arg);
  if (ret < 0) {
    return nullptr;
  }

  return msg.release();
}
HyperStatus HyperStatus::Success() {
  return {};
}
HyperStatus::HyperStatus(const HyperStatus &other) : status_{nullptr} {
  *this = other;
}
HyperStatus &HyperStatus::operator=(const HyperStatus &other) {
  if (this == &other) {
    return *this;
  }
  if (status_ != nullptr) {
    delete [] status_;
    status_ = nullptr;
  }
  if (other.status_ == nullptr) {
    status_ = nullptr;
  } else {
    size_t status_len = strlen(other.status_) + 1;
    status_ = new (std::nothrow) ge::char_t[status_len];
    if (status_ != nullptr) {
      auto ret = strcpy_s(status_, status_len, other.status_);
      if (ret != EOK) {
        status_[0] = '\0';
      }
    }
  }
  return *this;
}
HyperStatus::HyperStatus(HyperStatus &&other) noexcept {
  status_ = other.status_;
  other.status_ = nullptr;
}
HyperStatus &HyperStatus::operator=(HyperStatus &&other) noexcept {
  if (this != &other) {
    delete [] status_;
    status_ = other.status_;
    other.status_ = nullptr;
  }
  return *this;
}
HyperStatus HyperStatus::ErrorStatus(const ge::char_t *message, ...) {
  HyperStatus status;
  va_list arg;
  va_start(arg, message);
  status.status_ = CreateMessage(message, arg);
  va_end(arg);
  return status;
}
HyperStatus HyperStatus::ErrorStatus(std::unique_ptr<ge::char_t[]> message) {
  HyperStatus status;
  status.status_ = message.release();
  return status;
}
}
