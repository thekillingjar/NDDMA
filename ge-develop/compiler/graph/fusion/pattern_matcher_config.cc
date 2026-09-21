/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <memory>
#include "ge/fusion/pattern_matcher_config.h"
#include "common/checker.h"

namespace ge {
namespace fusion {
class PatternMatcherConfig::Impl {
 public:
  bool IsEnableConstValueMatch() const {
    return enable_const_value_match_;
  }
  bool IsEnableIrAttrMatch() const {
    return enable_ir_attr_match_;
  }

 private:
  friend class PatternMatcherConfigBuilderImpl;
  bool enable_const_value_match_ = false;
  bool enable_ir_attr_match_ = false;
};

bool PatternMatcherConfig::IsEnableConstValueMatch() const {
  return !(impl_ == nullptr) && impl_->IsEnableConstValueMatch();
}

bool PatternMatcherConfig::IsEnableIrAttrMatch() const {
  return !(impl_ == nullptr) && impl_->IsEnableIrAttrMatch();
}

PatternMatcherConfig::PatternMatcherConfig() {
  impl_ = std::make_unique<PatternMatcherConfig::Impl>();
}

PatternMatcherConfig::PatternMatcherConfig(const PatternMatcherConfig &other) {
  if (other.impl_ != nullptr) {
    impl_ = std::make_unique<PatternMatcherConfig::Impl>(*other.impl_);
  }
}

PatternMatcherConfig &PatternMatcherConfig::operator=(const PatternMatcherConfig &other) {
  if (this != &other) {
    impl_ = (other.impl_ != nullptr) ? std::make_unique<PatternMatcherConfig::Impl>(*other.impl_) : nullptr;
  }
  return *this;
}

PatternMatcherConfig::PatternMatcherConfig(PatternMatcherConfig &&other) noexcept : impl_(std::move(other.impl_)) {}

PatternMatcherConfig &PatternMatcherConfig::operator=(PatternMatcherConfig &&other) noexcept {
  if (this != &other) {
    impl_ = std::move(other.impl_);
  }
  return *this;
}

PatternMatcherConfig::~PatternMatcherConfig() = default;

class PatternMatcherConfigBuilderImpl {
 public:
  void EnableConstValueMatch() {
    enable_const_value_match_ = true;
  }
  void EnableIrAttrMatch() {
      enable_ir_attr_match_ = true;
  }
  std::unique_ptr<PatternMatcherConfig> Build() const {
    auto config = std::make_unique<PatternMatcherConfig>();
    GE_ASSERT_NOTNULL(config);
    config->impl_->enable_const_value_match_ = enable_const_value_match_;
    config->impl_->enable_ir_attr_match_ = enable_ir_attr_match_;
    return config;
  }
 private:
  bool enable_const_value_match_ = false;
  bool enable_ir_attr_match_ = false;
};

PatternMatcherConfigBuilder &PatternMatcherConfigBuilder::EnableConstValueMatch() {
  if (impl_ != nullptr) {
    impl_->EnableConstValueMatch();
  }
  return *this;
}

PatternMatcherConfigBuilder &PatternMatcherConfigBuilder::EnableIrAttrMatch() {
  if (impl_ != nullptr) {
    impl_->EnableIrAttrMatch();
  }
  return *this;
}

std::unique_ptr<PatternMatcherConfig> PatternMatcherConfigBuilder::Build() const {
  return impl_ == nullptr ? nullptr : impl_->Build();
}

PatternMatcherConfigBuilder::PatternMatcherConfigBuilder() {
  impl_ = std::make_unique<PatternMatcherConfigBuilderImpl>();
}

PatternMatcherConfigBuilder::~PatternMatcherConfigBuilder() = default;
}  // namespace fusion
}  // namespace ge

