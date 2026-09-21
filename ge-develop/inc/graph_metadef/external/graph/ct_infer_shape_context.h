/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_GRAPH_CT_INFER_SHAPE_CONTEXT_H_
#define METADEF_CXX_INC_GRAPH_CT_INFER_SHAPE_CONTEXT_H_
#include <type_traits>
#include "exe_graph/runtime/infer_shape_context.h"
#include "graph/inference_context.h"

namespace gert {
/**
 * 在节点输入后的扩展输入的索引，若需要扩展，请新增枚举类型
 */
enum class CtInferShapeInputExternLayout : uint32_t {
  kInferShapeFunc = 0,
  kInferenceContext = 1,
};

class CtInferShapeContext : public InferShapeContext {
 public:
  /**
   * 获取InferenceContext指针
   * @param NA
   * @return 输出InferenceContext指针
   */
  ge::InferenceContext *GetInferenceContext() const {
    const auto compute_node_info = GetComputeNodeInfo();
    if (compute_node_info == nullptr) {
      return nullptr;
    }
    const auto offset =
        compute_node_info->GetInputsNum() + static_cast<size_t>(CtInferShapeInputExternLayout::kInferenceContext);
    return MutableInputPointer<ge::InferenceContext>(offset);
  }
};
static_assert(std::is_standard_layout<CtInferShapeContext>::value, "The class CtInferShapeContext must be a POD");
}  // namespace gert
#endif  // METADEF_CXX_INC_GRAPH_CT_INFER_SHAPE_CONTEXT_H_
