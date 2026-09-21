/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/node_converter_registry.h"
#include "graph/node.h"
#include "graph_builder/bg_infer_shape.h"
#include "kernel/common_kernel_impl/build_tensor.h"

namespace gert {
LowerResult LoweringShapeNode(const ge::NodePtr &node, const LowerInput &input) {
  auto out_shapes = bg::InferStorageShape(node, input.input_shapes, *(input.global_data));
  auto node_outputs_num = node->GetAllOutDataAnchorsSize();
  // todo 如果shape作为网络的最后一个输入，这种方式是无法使能零拷贝能力的
  auto out_tensor_datas = bg::DevMemValueHolder::CreateDataOutput(
      kernel::kBuildShapeTensorData, input.input_shapes, node_outputs_num, node->GetOpDescBarePtr()->GetStreamId());
  std::for_each(out_tensor_datas.begin(), out_tensor_datas.end(), [] (bg::DevMemValueHolderPtr &out_tensor_data) {
    out_tensor_data->SetPlacement(kOnHost);
  });
  return {HyperStatus::Success(),
          {},  // no ordered holder
          out_shapes,
          out_tensor_datas};
}
REGISTER_NODE_CONVERTER("Shape", LoweringShapeNode);
REGISTER_NODE_CONVERTER("ShapeN", LoweringShapeNode);
}  // namespace gert
