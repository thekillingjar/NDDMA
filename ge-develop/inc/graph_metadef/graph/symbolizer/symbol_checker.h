/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GRAPH_SYMBOLIZER_SYMBOL_CHECKER_H_
#define GRAPH_SYMBOLIZER_SYMBOL_CHECKER_H_

/*
 * 校验表达式e0是否与e1相等
 * 如果e0的hint值与e1的hint值相等，则宏返回true，并生成e0 == e1的guard
 * 反之，则返回false，并生成 e0 != e1的guard
 */
#define EXPECT_SYMBOL_EQ(e0, e1)                                                                                       \
  ge::sym::ExpectSymbolEq(e0, e1, __FILE__, __LINE__)

/*
 * 校验表达式e0是否与e1不相等
 * 如果e0的hint值与e1的hint值不相等，则宏返回true，并生成e0 != e1的guard
 * 反之，则返回false，并生成 e0 == e1的guard
 */
#define EXPECT_SYMBOL_NE(e0, e1)                                                                                       \
  ge::sym::ExpectSymbolNe(e0, e1, __FILE__, __LINE__)

/*
 * 校验表达式e0是否小于e1
 * 如果e0的hint值小于e1的hint值，则宏返回true，并生成e0 < e1的guard
 * 反之，则返回false，并生成 e1 <= e0的guard
 */
#define EXPECT_SYMBOL_LT(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Lt(e0, e1), __FILE__, __LINE__)

/*
 * 校验表达式e0是否小于等于e1
 * 如果e0的hint值小于等于e1的hint值，则宏返回true，并生成e0 <= e1的guard
 * 反之，则返回false，并生成 e1 < e0的guard
 */
#define EXPECT_SYMBOL_LE(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Le(e0, e1), __FILE__, __LINE__)

/*
 * 校验表达式e0是否大于e1
 * 如果e0的hint值大于e1的hint值，则宏返回true，并生成e1 < e0的guard
 * 反之，则返回false，并生成 e0 <= e1的guard
 */
#define EXPECT_SYMBOL_GT(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Gt(e0, e1), __FILE__, __LINE__)

/*
 * 校验表达式e0是否大于等于e1
 * 如果e0的hint值大于等于e1的hint值，则宏返回true，并生成e1 <= e0的guard
 * 反之，则返回false，并生成 e0 < e1的guard
 */
#define EXPECT_SYMBOL_GE(e0, e1)                                                                                       \
  EXPECT_SYMBOL_CHECK(ge::sym::Ge(e0, e1), __FILE__, __LINE__)

/*
 * 检查表达式列表是否都为true
 * 如果表达式都为true，则宏返回true，
 * 如果其中有一个是false，则返回false，否则并生成LogicAnd()的guard
 * 例如校验表达式：EXPECT_SYMBOL_AND(Ge(s0, s1), Le(s2, s3), Eq(s4, s5))
 * hint值为true时添加guard：LogicAnd(ExpectEq(s4, s5), ExpectLe(s1, s0), ExpectLe(s2, s3))
 * hint值为false时添加guard：LogicOr(ExpectLt(s0, s1), ExpectLt(s3, s2), ExpectNe(s4, s5))
 */
#define EXPECT_SYMBOL_AND(...)                                                                                         \
  EXPECT_SYMBOL_CHECK(ge::sym::LogicalAnd(std::vector<Expression>{__VA_ARGS__}), __FILE__, __LINE__)

/*
 * 检查表达式列表是否有一个为true
 * 如果表达式全部为false，则宏返回false，
 * 如果其中有一个是true，则返回true，否则并生成LogicOr()的guard
 * 例如校验表达式：EXPECT_SYMBOL_OR(Ge(s0, s1), Le(s2, s3), Eq(s4, s5))
 * hint值为true时添加guard：LogicOr(ExpectEq(s4, s5), ExpectLe(s1, s0), ExpectLe(s2, s3))
 * hint值为false时添加guard：LogicAnd(ExpectLt(s0, s1), ExpectLt(s3, s2), ExpectNe(s4, s5))
 */
#define EXPECT_SYMBOL_OR(...)                                                                                          \
  EXPECT_SYMBOL_CHECK(ge::sym::LogicalOr(std::vector<Expression>{__VA_ARGS__}), __FILE__, __LINE__)

/*
 * 强校验表达式e0是否等于e1
 * 如果e0的hint值等于e1的hint值，则生成e0 == e1的guard
 * 反之，则报错
 */
#define ASSERT_SYMBOL_EQ(e0, e1)                                                                                       \
  do {                                                                                                                 \
    if (!ge::sym::AssertSymbolEq(e0, e1, __FILE__, __LINE__)) {                                                        \
      return ::ErrorResult();                                                                                          \
    }                                                                                                                  \
  } while (false)

/*
 * 强校验表达式e0是否不等于e1
 * 如果e0的hint值不等于e1的hint值，则生成e0 != e1的guard
 * 反之，则报错
 */
#define ASSERT_SYMBOL_NE(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Ne(e0, e1), __FILE__, __LINE__)

/*
 * 强校验表达式e0是否小于e1
 * 如果e0的hint值小于e1的hint值，则生成e0 < e1的guard
 * 反之，则报错
 */
#define ASSERT_SYMBOL_LT(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Lt(e0, e1), __FILE__, __LINE__)

/*
 * 强校验表达式e0是否小于等于e1
 * 如果e0的hint值小于等于e1的hint值，则生成e0 <= e1的guard
 * 反之，则报错
 */
#define ASSERT_SYMBOL_LE(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Le(e0, e1), __FILE__, __LINE__)

/*
 * 强校验表达式e0是否大于e1
 * 如果e0的hint值大于e1的hint值，则生成e1 < e0的guard
 * 反之，则报错
 */
#define ASSERT_SYMBOL_GT(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Gt(e0, e1), __FILE__, __LINE__)

/*
 * 强校验表达式e0是否大于等于e1
 * 如果e0的hint值大于等于e1的hint值，则生成e1 <= e0的guard
 * 反之，则报错
 */
#define ASSERT_SYMBOL_GE(e0, e1)                                                                                       \
  ASSERT_SYMBOL_CHECK(ge::sym::Ge(e0, e1), __FILE__, __LINE__)

#define EXPECT_SYMBOL_CHECK(expr, file, line)                                                                          \
  ge::sym::ExpectSymbolBool(expr, file, line)

#define ASSERT_SYMBOL_CHECK(expr, file, line)                                                                          \
  do {                                                                                                                 \
    if (!ge::sym::AssertSymbolBool(expr, file, line)) {                                                                \
      return ::ErrorResult();                                                                                          \
    }                                                                                                                  \
  } while (false)

namespace ge {
class Expression;
namespace sym {
bool ExpectSymbolEq(const Expression &e0, const Expression &e1,
    const char_t *file, const int64_t line);
bool AssertSymbolEq(const Expression &e0, const Expression &e1,
    const char_t *file, const int64_t line);
bool ExpectSymbolNe(const Expression &e0, const Expression &e1,
    const char_t *file, const int64_t line);
bool ExpectSymbolBool(const Expression &expr,
    const char_t *file, const int64_t line);
bool AssertSymbolBool(const Expression &expr,
    const char_t *file, const int64_t line);
}  // namespace sym
}
#endif // GRAPH_SYMBOLIZER_SYMBOL_CHECKER_H_