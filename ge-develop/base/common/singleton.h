/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GE_COMMON_SINGLETON_H_
#define GE_COMMON_SINGLETON_H_

#include <memory>
#include <mutex>

#define DECLARE_SINGLETON_CLASS(T) friend class Singleton<T>

namespace ge {
// Single thread version single instance template
template <typename T>
class Singleton {
 public:
  Singleton(Singleton const &) = delete;
  Singleton &operator=(Singleton const &) = delete;

  template <typename... _Args>
  static T *Instance(_Args... args) {
    static std::mutex single_mutex_;
    std::lock_guard<std::mutex> lock(single_mutex_);
    if (instance_ == nullptr) {
      // std::nothrow, Nullptr returned when memory request failed
      instance_.reset(new (std::nothrow) T(args...));
    }
    return instance_.get();
  }

  static void Destroy(void) { instance_.reset(); }

  Singleton() = default;
  virtual ~Singleton() = default;

 private:
  static std::unique_ptr<T> instance_;
};

template <typename T>
std::unique_ptr<T> Singleton<T>::instance_;
}  // namespace ge
#endif  // GE_COMMON_SINGLETON_H_
