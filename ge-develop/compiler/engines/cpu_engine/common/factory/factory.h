/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_FACTORY_H_
#define AICPU_FACTORY_H_

#include <functional>
#include <map>
#include <memory>
#include <string>

namespace aicpu {
/**
 * Template Factory Class
 * @ClassName Factory
 * @Description Automatic registration factory implement, Store a Map which key
 * is KernelLib Name e.g. CCEKernel TFKernel and value is pointer to a instance
 * of KernelLib. when a KernelLib defined, register a instance to the map
 */
template <typename T>
class Factory {
 public:
  using FactoryType = std::shared_ptr<T>;

  ~Factory() = default;

  template <typename N>
  struct Register {
    Register(const std::string &key) {
      Factory::Instance()->creators_[key] = [] { return N::Instance(); };
    }
  };

  static Factory<T> *Instance() {
    static Factory<T> factory;
    return &factory;
  }

  std::shared_ptr<T> Create(const std::string &key) {
    std::shared_ptr<T> re = nullptr;
    auto iter = creators_.find(key);
    if (iter != creators_.end()) {
      re = (iter->second)();
    }
    return re;
  }

  static std::shared_ptr<T> Produce(const std::string &key) {
    return Factory::Instance()->Create(key);
  }

 private:
  // Contructior
  Factory() = default;
  // Copy prohibit
  Factory(const Factory &) = delete;
  // Move prohibit
  Factory(Factory &&) = delete;
  // Copy prohibit
  Factory &operator=(const Factory &) = delete;
  // Move prohibit
  Factory &operator=(const Factory &&) = delete;
  std::map<std::string, std::function<std::shared_ptr<T>()>> creators_;
};
}  // namespace aicpu
#endif  // AICPU_FACTORY_H_
