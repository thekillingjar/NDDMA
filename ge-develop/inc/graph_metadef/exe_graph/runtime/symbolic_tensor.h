/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_SYMBOL_TENSOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_SYMBOL_TENSOR_H_

#include <array>
#include <vector>
#include <iostream>
#include <cstring>
#include <type_traits>
#include <limits>
#include "utils/extern_math_util.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "symbolic_shape.h"

namespace gert {
class SymbolTensor {
 public:
  SymbolTensor() = default;
  SymbolTensor(const std::initializer_list<ge::Expression> &origin_symbol_shape)
      : origin_symbol_shape_(origin_symbol_shape) {}

  SymbolTensor(const std::initializer_list<ge::Expression> &origin_symbol_shape,
               const std::initializer_list<ge::Expression> &symbolic_values)
      : origin_symbol_shape_(origin_symbol_shape),
        symbolic_values_(ge::ComGraphMakeUnique<std::vector<ge::Expression>>(symbolic_values)) {}

  // 拷贝构造函数
  SymbolTensor(const SymbolTensor &other)
      : origin_symbol_shape_(other.origin_symbol_shape_),
        symbolic_values_(other.symbolic_values_
                           ? ge::ComGraphMakeUnique<std::vector<ge::Expression>>(*other.symbolic_values_)
                           : nullptr) {}

  // 拷贝赋值运算符
  SymbolTensor &operator=(const SymbolTensor &other) {
    if (this != &other) {
      origin_symbol_shape_ = other.origin_symbol_shape_;
      if (other.symbolic_values_) {
        symbolic_values_ = ge::ComGraphMakeUnique<std::vector<ge::Expression>>(*other.symbolic_values_);
      } else {
        symbolic_values_.reset();
      }
    }
    return *this;
  }

  // 移动构造函数
  SymbolTensor(SymbolTensor &&other) noexcept = default;

  // 移动赋值运算符
  SymbolTensor &operator=(SymbolTensor &&other) noexcept = default;

  /**
   * 获取只读的原始格式符号化shape
   * @return 原始格式符号化shape引用
   */
  const SymbolShape &GetOriginSymbolShape() const {
    return origin_symbol_shape_;
  }
  /**
   * 获取可修改的原始格式符号化shape
   * @return 原始格式符号化shape引用
   */
  SymbolShape &MutableOriginSymbolShape() {
    return origin_symbol_shape_;
  }
  /**
   * 设置原始格式符号化shape
   * @return void
   */
  void SetOriginSymbolShape(const SymbolShape &ori_symbol_shape) {
    origin_symbol_shape_ = ori_symbol_shape;
  }

  /**
   * 设置原始格式、存储格式符号化shape
   * @return void
   */
  void SetSymbolShape(const SymbolShape &symbol_shape) {
    SetOriginSymbolShape(symbol_shape);
  }
  /**
   * 获取symbol tensor中存储的符号值，一般在data dependent算子场景使用
   * 1. 返回值为nullptr，表明无符号值
   * 2. 返回值为{}，表明符号值为空
   * 3. 返回值不为空，表明存在符号值
   *
   * @return 只读的symbol tensor符号值;
   */
  const std::vector<ge::Expression> *GetSymbolicValue() const {
    return symbolic_values_.get();
  }
  /**
   * 设置symbol tensor中存储的符号值
   * @param symbolic_values
   */
  void SetSymbolicValue(std::unique_ptr<std::vector<ge::Expression>> symbolic_values) {
    symbolic_values_ = std::move(symbolic_values);
  }
  /**
   * 获取tensor data
   * @return 可写的symbol tensor符号值
   */
  std::vector<ge::Expression> *MutableSymbolicValue() {
    if (!symbolic_values_) {
      symbolic_values_ = ge::ComGraphMakeUnique<std::vector<ge::Expression>>();
    }

    if (!symbolic_values_) {
      return nullptr;
    }

    return symbolic_values_.get();
  }
 private:
  SymbolShape origin_symbol_shape_;
  std::unique_ptr<std::vector<ge::Expression>> symbolic_values_;
};
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_SYMBOL_SHAPE_H_
