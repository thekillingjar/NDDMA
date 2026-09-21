/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "size_kernel.h"
#include "securec.h"
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "graph/utils/math_util.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "common/checker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
namespace kernel {
namespace {
constexpr size_t kSizeDTypeAttrIndex = 0U;
ge::graphStatus GetSizeOutputDataType(KernelContext *context, ge::DataType &out_data_type) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto attrs = compute_node_info->GetAttrs();
  const int32_t *const data_type = attrs->GetAttrPointer<int32_t>(kSizeDTypeAttrIndex);
  GE_ASSERT_NOTNULL(data_type, "%s dtype does not exist, ", extend_context->GetKernelName());
  out_data_type = static_cast<ge::DataType>(*data_type);
  GE_ASSERT_TRUE(out_data_type == ge::DataType::DT_INT32 || out_data_type == ge::DataType::DT_INT64,
                 "only support DT_INT32 and DT_INT64, but out_data_type is %s",
                 ge::TypeUtils::DataTypeToSerialString(out_data_type).c_str());
  return ge::GRAPH_SUCCESS;
}
}
template<typename T>
ge::Status SetOutput(int64_t shape_size, TensorAddress output_addr) {
  auto out_data = ge::PtrToPtr<void, T>(output_addr);
  GE_ASSERT_NOTNULL(out_data);
  GE_ASSERT_TRUE(ge::IntegerChecker<T>::Compat(shape_size));
  *out_data = shape_size;
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetShapeSizeKernelCompute(KernelContext *context) {
  auto input_shape = context->GetInputPointer<gert::Shape>(static_cast<size_t>(SizeKernelInputs::kShapeStart));
  GE_ASSERT_NOTNULL(input_shape);
  auto shape_size = input_shape->GetShapeSize();
  ge::DataType out_data_type;
  GE_ASSERT_GRAPH_SUCCESS(GetSizeOutputDataType(context, out_data_type), "get data type failed");
  auto out_tensor_data = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(SizeKernelOutputs::kTensorData));
  GE_ASSERT_NOTNULL(out_tensor_data);
  if (out_data_type == ge::DT_INT64) {
    GE_ASSERT_GRAPH_SUCCESS(SetOutput<int64_t>(shape_size, out_tensor_data->GetAddr()));
  } else if (out_data_type == ge::DT_INT32) {
    GE_ASSERT_GRAPH_SUCCESS(SetOutput<int32_t>(shape_size, out_tensor_data->GetAddr()));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildSizeOutput(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  ge::DataType out_data_type;
  GE_ASSERT_GRAPH_SUCCESS(GetSizeOutputDataType(context, out_data_type), "get data type failed");

  // Op Size output is scalar.
  int32_t data_size = ge::GetSizeByDataType(static_cast<ge::DataType>(out_data_type));
  GE_ASSERT_TRUE(!ge::AddOverflow(data_size, sizeof(GertTensorData), data_size));
  auto chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(chain);
  auto out_data = ge::MakeUnique<uint8_t[]>(data_size);
  GE_ASSERT_NOTNULL(out_data);
  new (out_data.get())
      GertTensorData(out_data.get() + sizeof(GertTensorData), data_size - sizeof(GertTensorData), kOnHost, -1);
  chain->SetWithDefaultDeleter<uint8_t[]>(out_data.release());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GetShapeSizeKernel).RunFunc(GetShapeSizeKernelCompute).OutputsCreator(BuildSizeOutput);
}  // namespace kernel
}  // namespace gert
