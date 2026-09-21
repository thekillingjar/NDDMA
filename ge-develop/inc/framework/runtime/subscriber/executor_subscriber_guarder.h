/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_GUARDER_H_
#define AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_GUARDER_H_

#include "framework/common/ge_visibility.h"
#include "common/checker.h"
#include "executor_subscriber_c.h"
namespace gert {
template <typename T>
void ObjectDeleter(void *obj) {
  delete static_cast<T *>(obj);
}

class VISIBILITY_EXPORT ExecutorSubscriberGuarder {
 public:
  using ArgDeleter = void (*)(void *);

  ExecutorSubscriberGuarder(::SubscriberFunc func, void *arg, ArgDeleter deleter)
      : subscriber_({func, arg}), arg_deleter_(deleter) {}
  ExecutorSubscriberGuarder(ExecutorSubscriberGuarder &&other) noexcept {
    MoveAssignment(other);
  }
  ExecutorSubscriberGuarder &operator=(ExecutorSubscriberGuarder &&other) noexcept {
    DeleteArg();
    MoveAssignment(other);
    return *this;
  }

  ExecutorSubscriber &GetSubscriber() {
    return subscriber_;
  }

  const ExecutorSubscriber &GetSubscriber() const {
    return subscriber_;
  }

  ~ExecutorSubscriberGuarder() {
    DeleteArg();
  }

  void SetEnabledFunc(const std::function<bool()> &enabled_func) {
    enabled_func_ = enabled_func;
  }

  bool IsEnabled() const {
    if (enabled_func_ == nullptr) {
      return true;
    }
    if (enabled_func_()) {
      return true;
    }
    return false;
  }

  ExecutorSubscriberGuarder(const ExecutorSubscriberGuarder &) = delete;
  ExecutorSubscriberGuarder &operator=(const ExecutorSubscriberGuarder &) = delete;

 private:
  void DeleteArg() {
    if (arg_deleter_ != nullptr) {
      arg_deleter_(subscriber_.arg);
    }
    enabled_func_ = nullptr;
  }
  void MoveAssignment(ExecutorSubscriberGuarder &other) {
    subscriber_ = other.subscriber_;
    arg_deleter_ = other.arg_deleter_;
    enabled_func_ = other.enabled_func_;
    other.subscriber_ = {nullptr, nullptr};
    other.arg_deleter_ = nullptr;
    other.enabled_func_ = nullptr;
  }

 private:
  ExecutorSubscriber subscriber_{nullptr, nullptr};
  ArgDeleter arg_deleter_{nullptr};
  std::function<bool()> enabled_func_{nullptr};
};
using ExecutorSubscriberGuarderPtr = std::shared_ptr<ExecutorSubscriberGuarder>;
}  // namespace gert
#endif  // AIR_CXX_INC_FRAMEWORK_RUNTIME_EXECUTOR_SUBSCRIBER_GUARDER_H_
