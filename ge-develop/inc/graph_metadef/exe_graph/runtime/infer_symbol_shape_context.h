/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SYMBOL_SHAPE_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SYMBOL_SHAPE_CONTEXT_H_

#include <type_traits>
#include "exe_graph/runtime/runtime_attrs.h"
#include "symbolic_tensor.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
class InferSymbolShapeContext : public ExtendedKernelContext {
 public:
  /**
  * 根据输入index，获取输入symbol shape指针，该接口仅在编译态使用；
  * @param index 输入index；
  * @return 输入symbol shape指针，index非法时，返回空指针。
  */
  const SymbolShape *GetInputSymbolShape(const size_t index) const {
    if (GetInputSymbolTensor(index) == nullptr) {
      return nullptr;
    }
    return &(GetInputSymbolTensor(index)->GetOriginSymbolShape());
  }

  /**
  * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入symbol shape指针，该接口仅在编译态使用；
  * @param ir_index IR原型定义中的index；
  * @return symbol shape指针，index非法，或该INPUT没有实例化时，返回空指针。
  */
  const SymbolShape *GetOptionalInputSymbolShape(const size_t ir_index) const {
    if (GetDynamicInputSymbolTensor(ir_index, 0) == nullptr) {
      return nullptr;
    }
    return &(GetDynamicInputSymbolTensor(ir_index, 0)->GetOriginSymbolShape());
  }

  /**
  * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入symbol shape指针，该接口仅在编译态使用；
  * @param ir_index IR原型定义中的index；
  * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]；
  * @return symbol shape指针，index或relative_index非法时，返回空指针。
  */
  const SymbolShape *GetDynamicInputSymbolShape(const size_t ir_index, const size_t relative_index) const {
    if (GetDynamicInputSymbolTensor(ir_index, relative_index) == nullptr) {
      return nullptr;
    }
    return &(GetDynamicInputSymbolTensor(ir_index, relative_index)->GetOriginSymbolShape());
  }

  /**
  * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入symbol shape指针，该接口仅在编译态使用；
  * @param ir_index IR原型定义中的index
  * @return symbol shape指针，index非法，或该INPUT没有实例化时，返回空指针
  */
  const SymbolShape *GetRequiredInputSymbolShape(const size_t ir_index) const {
    if (GetDynamicInputSymbolTensor(ir_index, 0) == nullptr) {
      return nullptr;
    }
    return &(GetDynamicInputSymbolTensor(ir_index, 0)->GetOriginSymbolShape());
  }

  /**
   * 根据输入index，获取输入SymbolTensor指针，该接口仅在编译态使用；
   * 若算子被配置为'data'数据依赖，则返回的SymbolTensor对象中保存了的符号值；反之，内存地址为nullptr。
   * @param index 输入index
   * @return 输入SymbolTensor指针，index非法时，返回空指针
   */
  const SymbolTensor *GetInputSymbolTensor(const size_t index) const {
    return GetInputPointer<SymbolTensor>(index);
  }

  /**
   * 基于算子IR原型定义，获取`OPTIONAL_INPUT`类型的输入SymbolTensor指针，该接口仅在编译态使用；
   * 若算子被配置为'data'数据依赖，则返回的SymbolTensor对象中保存了的符号值；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return SymbolTensor指针，index非法，或该INPUT没有实例化时，返回空指针
   */
  const SymbolTensor *GetOptionalInputSymbolTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<SymbolTensor>(ir_index, 0);
  }

  /**
   * 基于算子IR原型定义，获取`DYNAMIC_INPUT`类型的输入tensor指针，该接口仅在编译态使用；
   * 若算子被配置为'data'数据依赖，则返回的SymbolTensor对象中保存了的符号值；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @param relative_index 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]
   * @return SymbolTensor指针，index或relative_index非法时，返回空指针
   */
  const SymbolTensor *GetDynamicInputSymbolTensor(const size_t ir_index, const size_t relative_index) const {
    return GetDynamicInputPointer<SymbolTensor>(ir_index, relative_index);
  }

  /**
   * 基于算子IR原型定义，获取`REQUIRED_INPUT`类型的输入SymbolTensor指针，该接口仅在编译态使用；
   * 若算子被配置为'data'数据依赖，则返回的SymbolTensor对象中保存了的符号值；反之，内存地址为nullptr。
   * @param ir_index IR原型定义中的index
   * @return SymbolTensor指针，index非法时，返回空指针
   */
  const SymbolTensor *GetRequiredInputSymbolTensor(const size_t ir_index) const {
    return GetDynamicInputPointer<SymbolTensor>(ir_index, 0);
  }

  /**
  * 根据输出index，获取输出符号化Symbolshape指针，该接口仅在编译态使用；
  * @param index 输出index；
  * @return 输出符号化Symbolshape指针，index非法时，返回空指针。
  */
  SymbolShape *GetOutputSymbolShape(const size_t index) {
    return GetOutputPointer<SymbolShape>(index);
  }
};

static_assert(std::is_standard_layout<InferSymbolShapeContext>::value,
              "The class InferSymbolShapeContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_INFER_SYMBOL_SHAPE_CONTEXT_H_
