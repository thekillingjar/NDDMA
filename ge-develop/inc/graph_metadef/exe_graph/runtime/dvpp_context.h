/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_DVPP_CONTEXT_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_DVPP_CONTEXT_H_
#include <type_traits>
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
/**
 * Dvpp kernel的context
 */
class DvppContext : public ExtendedKernelContext {
public:
  /**
   * 获取输入shape，输入shape中包含了原始shape与运行时shape
   * @param index 输入index
   * @return 输入shape指针，index非法时返回空指针
   */
  const StorageShape *GetInputShape(size_t index) const {
    auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetInputsNum()) {
      return nullptr;
    }

    return GetInputPointer<StorageShape>(index);
  }

  /**
   * 获取输入tensor
   *
   * **注意：只有在`IMPL_OP`实现算子时， 将对应输入设置为数据依赖后，
   * 才可以调用此接口获取tensor，否则行为是未定义的。**
   * @param index 输入index
   * @return 输入tensor指针，index非法时返回空指针
   */
  const Tensor *GetInputTensor(size_t index) const {
    return GetInputPointer<Tensor>(index);
  }

  /**
   * 根据输出index，获取输出shape指针，shape中包含了原始shape与运行时shape
   * @param index 输出index
   * @return 输出shape指针，index非法时，返回空指针
   */
  const StorageShape *GetOutputShape(size_t index) const {
    auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }

    if (index >= compute_node_info->GetOutputsNum()) {
      return nullptr;
    }

    size_t offset = compute_node_info->GetInputsNum();
    return GetInputPointer<StorageShape>(offset + index);
  }
};
static_assert(std::is_standard_layout<DvppContext>::value,
              "The class DvppContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_DVPP_CONTEXT_H_
