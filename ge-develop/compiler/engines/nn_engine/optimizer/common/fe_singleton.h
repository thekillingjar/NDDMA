/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FUSION_ENGINE_OPTIMIZER_COMMON_FE_SINGLETON_H_
#define FUSION_ENGINE_OPTIMIZER_COMMON_FE_SINGLETON_H_
#include <mutex>

#define FE_DECLARE_SINGLETON(class_name)           \
 public:                                           \
  static class_name* GetInstance();                \
  /* only be called when system end */             \
  static void DestroyInstance();                   \
                                                   \
 protected:                                        \
  class_name();                                    \
  ~class_name();                                   \
  class_name(const class_name&);                   \
  const class_name& operator=(const class_name&);  \
                                                   \
 private:                                          \
  static void Init() {                             \
    mtx_.lock();                                   \
    if (instance_ == nullptr) {                    \
      instance_ = new (std::nothrow) class_name(); \
      if (instance_ == nullptr) {                  \
        mtx_.unlock();                             \
        return;                                    \
      }                                            \
    }                                              \
    mtx_.unlock();                                 \
  }                                                \
  static void restore() {                          \
    mtx_.lock();                                   \
    if (instance_ == nullptr) {                    \
      mtx_.unlock();                               \
      return;                                      \
    }                                              \
    delete instance_;                              \
    instance_ = nullptr;                           \
    mtx_.unlock();                                 \
  }                                                \
                                                   \
 private:                                          \
  static class_name* instance_;                    \
  static std::mutex mtx_;

#define FE_DEFINE_SINGLETON_EX(class_name)     \
  class_name* class_name::instance_ = nullptr; \
  std::mutex class_name::mtx_;                 \
  class_name* class_name::GetInstance() {      \
    if (instance_ == nullptr) {                \
      class_name::Init();                      \
    }                                          \
    return instance_;                          \
  }                                            \
  void class_name::DestroyInstance() { restore(); }

#define FE_DEFINE_SINGLETON(class_name) \
  FE_DEFINE_SINGLETON_EX(class_name)    \
  class_name::class_name() {}           \
  class_name::~class_name() {}

#endif  // FUSION_ENGINE_OPTIMIZER_COMMON_FE_SINGLETON_H_
