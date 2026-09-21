/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_CXX_INC_EXE_GRAPH_LOWERING_GENERATE_EXE_GRAPH_H_
#define METADEF_CXX_INC_EXE_GRAPH_LOWERING_GENERATE_EXE_GRAPH_H_
#include <vector>

#include "dev_mem_value_holder.h"
#include "graph/compute_graph.h"
#include "lowering_global_data.h"
namespace gert {
namespace bg {
class GenerateExeGraph {
 public:
  struct ExeGraphGenerator {
    using InferShapeFunc = std::vector<ValueHolderPtr> (*)(const ge::NodePtr &node,
                                                           const std::vector<ValueHolderPtr> &shapes,
                                                           LoweringGlobalData &global_data);
    using AllocOutputMemoryFunc = std::vector<DevMemValueHolderPtr> (*)(TensorPlacement placement,
                                                                        const ge::NodePtr &node,
                                                                        const std::vector<ValueHolderPtr> &output_sizes,
                                                                        LoweringGlobalData &global_data);
    using CalcTensorSizeFunc = std::vector<ValueHolderPtr> (*)(const ge::NodePtr &node,
                                                               const std::vector<ValueHolderPtr> &output_shapes);

    InferShapeFunc infer_shape;
    AllocOutputMemoryFunc alloc_output_memory;
    CalcTensorSizeFunc calc_tensor_size;
  };

 public:
  static std::vector<ValueHolderPtr> InferShape(const ge::NodePtr &node, const std::vector<ValueHolderPtr> &shapes,
                                                LoweringGlobalData &global_data) {
    if (generator_.infer_shape == nullptr) {
      return {};
    }
    return generator_.infer_shape(node, shapes, global_data);
  }
  static std::vector<DevMemValueHolderPtr> AllocOutputMemory(TensorPlacement placement, const ge::NodePtr &node,
                                                             const std::vector<ValueHolderPtr> &output_sizes,
                                                             LoweringGlobalData &global_data) {
    if (generator_.alloc_output_memory == nullptr) {
      return {};
    }
    return generator_.alloc_output_memory(placement, node, output_sizes, global_data);
  }
  static std::vector<ValueHolderPtr> CalcTensorSize(const ge::NodePtr &node,
                                                    const std::vector<ValueHolderPtr> &output_shapes) {
    if (generator_.calc_tensor_size == nullptr) {
      return {};
    }
    return generator_.calc_tensor_size(node, output_shapes);
  }

  static void AddBuilderImplement(ExeGraphGenerator generator) {
    generator_ = generator;
  }

  static ValueHolderPtr MakeSureTensorAtHost(const ge::Node *node, LoweringGlobalData &global_data,
                                             const ValueHolderPtr &addr, const ValueHolderPtr &size);

  static ValueHolderPtr CalcTensorSizeFromShape(ge::DataType dt, const ValueHolderPtr &shape);

  static ValueHolderPtr FreeMemoryGuarder(const ValueHolderPtr &resource);

 private:
  static ExeGraphGenerator generator_;
};
}  // namespace bg
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_LOWERING_GENERATE_EXE_GRAPH_H_
