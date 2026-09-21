/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "multi_batch_shape_data_kernel.h"
#include <sstream>
#include "register/kernel_registry.h"
#include "common/checker.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/kernel_context.h"
#include "graph/node.h"
#include "utils/extern_math_util.h"
#include "common/plugin/ge_make_unique_util.h"

namespace gert {
namespace kernel {
ge::graphStatus GetCurDynamicShapeFunc(KernelContext *context) {
  const size_t input_size = context->GetInputNum();
  auto input_and_unknown_dim_index =
      context->GetInputValue<ContinuousVectorVector *>(
      static_cast<size_t>(GetCurDynamicShapeInput::kUnknownShapeIndex));
  GE_ASSERT_TRUE(input_size ==
      (input_and_unknown_dim_index->GetSize() + static_cast<size_t>(GetCurDynamicShapeInput::kEnd)));
  auto tensor_data =
      context->GetOutputPointer<GertTensorData>(
      static_cast<size_t>(GetCurDynamicShapeOutputs::kTensorData));
  GE_ASSERT_NOTNULL(tensor_data);
  auto current_shape = reinterpret_cast<int32_t *>(tensor_data->GetAddr());
  GE_ASSERT_NOTNULL(current_shape);
  size_t index = 0UL;
  for (size_t i = 0UL; i < input_and_unknown_dim_index->GetSize(); i++) {
    auto storage_shape =
        context->GetInputPointer<StorageShape>(static_cast<size_t>(GetCurDynamicShapeInput::kEnd) + i);
    GE_ASSERT_NOTNULL(storage_shape);
    auto unknown_dim_index = input_and_unknown_dim_index->Get(i);
    size_t unknown_dim_index_size = unknown_dim_index->GetSize();
    GE_ASSERT_TRUE(unknown_dim_index_size > 0UL);
    auto unknown_dim_index_data = reinterpret_cast<const int32_t *>(unknown_dim_index->GetData());
    const auto shape = storage_shape->GetStorageShape();
    for (size_t j = 0UL; j < unknown_dim_index_size; j++) {
      GE_ASSERT_TRUE(unknown_dim_index_data[j] < static_cast<int32_t>(shape.GetDimNum()));
      GE_ASSERT_TRUE(index * sizeof(int32_t) < tensor_data->GetSize());
      current_shape[index++] = static_cast<int32_t>(shape.GetDim(unknown_dim_index_data[j]));
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildCurDynamicShapesOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  const auto data_type_size = ge::GetSizeByDataType(ge::DataType::DT_INT32);
  GE_ASSERT_TRUE(data_type_size > 0, "get data type size failed, data_type_size:%d", data_type_size);
  auto storage_shape =
      context->GetInputPointer<StorageShape>(static_cast<size_t>(GetCurDynamicShapeInput::kShape));
  GE_ASSERT_NOTNULL(storage_shape);
  auto output_shape = storage_shape->GetStorageShape();
  const auto output_shape_dim = output_shape.GetDimNum();
  GE_ASSERT_TRUE(output_shape_dim == 1UL);
  size_t malloc_buffer_size = 0U;
  GE_ASSERT_TRUE(!ge::MulOverflow(static_cast<size_t>(data_type_size),
      static_cast<size_t>(output_shape.GetDim(0UL)), malloc_buffer_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(malloc_buffer_size, sizeof(GertTensorData), malloc_buffer_size));

  auto chain = context->GetOutput(static_cast<size_t>(GetCurDynamicShapeOutputs::kTensorData));
  GE_ASSERT_NOTNULL(chain);
  auto out_data = ge::MakeUnique<uint8_t[]>(malloc_buffer_size);
  GE_ASSERT_NOTNULL(out_data);
  new (out_data.get())
      GertTensorData(out_data.get() + sizeof(GertTensorData),
      malloc_buffer_size - sizeof(GertTensorData), kOnHost, -1);
  chain->SetWithDefaultDeleter<uint8_t[]>(out_data.release());
  return ge::GRAPH_SUCCESS;
}

std::vector<std::string> PrintCurDynamicShape(const KernelContext *context) {
  std::stringstream ss;
  auto tensor_data =
      context->GetOutputPointer<GertTensorData>(static_cast<size_t>(GetCurDynamicShapeOutputs::kTensorData));
  GE_ASSERT_NOTNULL(tensor_data);
  const auto shape_data = reinterpret_cast<int32_t *>(tensor_data->GetAddr());
  GE_ASSERT_NOTNULL(shape_data);
  const size_t shape_size = tensor_data->GetSize() / sizeof(int32_t);
  ss << "tensor data is: [";
  for (size_t i = 0U; i < shape_size; i++) {
    ss << std::to_string(shape_data[i]);
    if (i < shape_size - 1U) {
      ss << ", ";
    }
  }
  ss << "]";
  return {ss.str()};
}

REGISTER_KERNEL(GetCurDynamicShape)
    .RunFunc(GetCurDynamicShapeFunc)
    .OutputsCreator(BuildCurDynamicShapesOutput)
    .TracePrinter(PrintCurDynamicShape);
}
}
