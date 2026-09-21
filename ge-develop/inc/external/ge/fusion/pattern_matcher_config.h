/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CANN_GRAPH_ENGINE_PATTERN_MATCHER_CONFIG_H
#define CANN_GRAPH_ENGINE_PATTERN_MATCHER_CONFIG_H

#include <memory>

namespace ge {
namespace fusion {
/**
 * 用于配置PatternMatcher行为的类, 请使用PatternMatcherConfigBuilder来构造PatternMatcherConfig
 */
class PatternMatcherConfig {
 public:
  PatternMatcherConfig();
  ~PatternMatcherConfig();
  PatternMatcherConfig(const PatternMatcherConfig &other);
  PatternMatcherConfig &operator=(const PatternMatcherConfig &other);
  PatternMatcherConfig(PatternMatcherConfig &&other) noexcept;
  PatternMatcherConfig &operator=(PatternMatcherConfig &&other) noexcept;
  /**
   * 返回是否使能const值匹配能力
   * 若返回true，pattern matcher在匹配过程中将对pattern中定义的const/constant上携带的Tensor进行值的匹配，值相等才认为匹配成功。
   * 注意：匹配方式为二进制匹配，这是一种较为严格的匹配方式(对于浮点类数据类型来说）。若不符合预期，请关闭该开关，自行处理值校验逻辑
   * @return
   */
  bool IsEnableConstValueMatch() const;

  /**
   * 返回是否使能ir属性及其值匹配能力
   * 若返回true, pattern matcher将在pattern匹配过程中对pattern中节点上携带的IR属性的值进行匹配。
   * 注意：匹配方式为二进制匹配，这是一种较为严格的匹配方式(对于浮点类数据类型来说）。若不符合预期，请关闭该开关，自行处理值校验逻辑
   * @return
   */
  bool IsEnableIrAttrMatch() const;

 private:
  friend class PatternMatcherConfigBuilderImpl;
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class PatternMatcherConfigBuilderImpl;
/**
 * 构造PatternMatcherConfig的辅助类
 */
class PatternMatcherConfigBuilder {
 public:
  PatternMatcherConfigBuilder();
  ~PatternMatcherConfigBuilder();
  /**
   * 配置使能const值匹配能力
   * 若使能该配置，pattern matcher在匹配过程中将对pattern中定义的const/constant上携带的Tensor进行值的匹配，值相等才认为匹配成功。
   * 注意：匹配方式为二进制匹配，这是一种较为严格的匹配方式。若不符合预期，请关闭该配置，自行处理值校验逻辑
   * @return
   */
  PatternMatcherConfigBuilder &EnableConstValueMatch();

  /**
   * 配置是否使能ir属性及其值匹配能力
   * 若使能该配置， pattern matcher将在pattern匹配过程中对pattern中节点上携带的IR属性的值进行匹配。
   * 注意：匹配方式为二进制匹配，这是一种较为严格的匹配方式。若不符合预期，请关闭该开关，自行处理值校验逻辑
   * @return
   */
  PatternMatcherConfigBuilder &EnableIrAttrMatch();

  /**
   * 构造PatternMatcherConfig
   * @return
   */
  std::unique_ptr<PatternMatcherConfig> Build() const;

 private:
  std::unique_ptr<PatternMatcherConfigBuilderImpl> impl_;
};
}  // namespace fusion
}  // namespace ge
#endif  // CANN_GRAPH_ENGINE_PATTERN_MATCHER_CONFIG_H
