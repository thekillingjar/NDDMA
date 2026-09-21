/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_SYMBOLIZER_SYMBOLIC_UTILS_H_
#define GRAPH_SYMBOLIZER_SYMBOLIC_UTILS_H_

#include <string>
namespace ge {
class Expression;
enum class TriBool : int8_t { kFalse = 0, kTrue = 1, kUnknown = -1 };

class SymbolicUtils {
 public:
  /**
   * @param e 符号
   * @brief 序列化，将表达式转换成字符串
   */
  static std::string ToString(const Expression &e);

  /**
   * @brief 基于之前生成的guard信息判断e1与e2是否相等，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e1 表达式1
   * @param e2 表达式2
   * @return TriBool - 三态布尔返回值：
   *         kTrue: 可确定 e1 == e2
   *         kFalse: 可确定 e1 != e2
   *         kUnknown: 根据现有guard无法确定大小关系
   */
  static TriBool StaticCheckEq(const Expression &e1, const Expression &e2);

  /**
   * @brief 基于之前生成的guard信息判断e1与e2是否不相等，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e1 表达式1
   * @param e2 表达式2
   * @return TriBool - 三态布尔返回值：
   *         kTrue: 可确定 e1 != e2
   *         kFalse: 可确定 e1 == e2
   *         kUnknown: 根据现有guard无法确定大小关系
   */
  static TriBool StaticCheckNe(const Expression &e1, const Expression &e2);

  /**
   * @brief 基于之前生成的guard信息判断e1是否小于e2，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e1 表达式1
   * @param e2 表达式2
   * @return TriBool - 三态布尔返回值：
   *         kTrue: 可确定 e1 < e2
   *         kFalse: 可确定 e1 >= e2
   *         kUnknown: 根据现有guard无法确定大小关系
   */
  static TriBool StaticCheckLt(const Expression &e1, const Expression &e2);

  /**
   * @brief 基于之前生成的guard信息判断e1是否小于等于e2，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e1 表达式1
   * @param e2 表达式2
   * @return TriBool - 三态布尔返回值：
   *         kTrue: 可确定 e1 <= e2
   *         kFalse: 可确定 e1 > e2
   *         kUnknown: 根据现有guard无法确定大小关系
   */
  static TriBool StaticCheckLe(const Expression &e1, const Expression &e2);

  /**
   * @brief 基于之前生成的guard信息判断e1是否大于e2，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e1 表达式1
   * @param e2 表达式2
   * @return TriBool - 三态布尔返回值：
   *         kTrue: 可确定 e1 > e2
   *         kFalse: 可确定 e1 <= e2
   *         kUnknown: 根据现有guard无法确定大小关系
   */
  static TriBool StaticCheckGt(const Expression &e1, const Expression &e2);

  /**
   * @brief 基于之前生成的guard信息判断e1是否大于等于e2，仅基于已有guard做校验，不生产新的guard，主要用于内存优化等编译态优化时判断使用
   * @param e1 表达式1
   * @param e2 表达式2
   * @return TriBool - 三态布尔返回值：
   *         kTrue: 可确定 e1 >= e2
   *         kFalse: 可确定 e1 < e2
   *         kUnknown: 根据现有guard无法确定大小关系
   */
  static TriBool StaticCheckGe(const Expression &e1, const Expression &e2);

  /**
  * @brief 将表达式拆成分子除以分母的形式进行序列化，例如表达式Rational(2, 3) * s1，会序列化成(2 * s1) / 3
  * @param x 要序列化的表达式
  */
  static std::string AsNumerDenomToString(const Expression &x);

 private:
  static TriBool StaticCheckBool(const Expression &expr);
};
}
#endif // GRAPH_SYMBOLIZER_SYMBOLIC_UTILS_H_