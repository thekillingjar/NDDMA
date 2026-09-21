/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gather_shapes_kernel.h"
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
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "graph/node.h"

namespace gert {
namespace kernel {
enum class GatherShapeInputs {
  kShapeStart
};
namespace {
constexpr size_t kAxesAttrIndex = 0U;
constexpr size_t kDTypeAttrIndex = 1U;
ge::graphStatus GetOutputDataType(KernelContext *context, ge::DataType &out_data_type) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto attrs = compute_node_info->GetAttrs();
  const int32_t *const data_type = attrs->GetAttrPointer<int32_t>(kDTypeAttrIndex);
  GE_ASSERT_NOTNULL(data_type, "%s dtype does not exist, ", extend_context->GetKernelName());
  out_data_type =  static_cast<ge::DataType>(*data_type);
  return ge::SUCCESS;
}
ge::graphStatus GetAxes(KernelContext *context, const ContinuousVectorVector **cvv) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto attrs = compute_node_info->GetAttrs();
  *cvv = attrs->GetListListInt(kAxesAttrIndex);
  GE_ASSERT_NOTNULL(*cvv, "%s axes does not exist, ", extend_context->GetKernelName());
  return ge::SUCCESS;
}
std::string PrintContinuousVectorVector(const ContinuousVectorVector *cvv) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0U; i < cvv->GetSize(); ++i) {
    ss << "[";
    const auto cv = cvv->Get(i);
    const uint64_t *const data = reinterpret_cast<const uint64_t *>(cv->GetData());
    for (size_t j = 0U; j < cv->GetSize(); ++j) {
      ss << data[j] << ", ";
    }
    ss << "], ";
  }
  ss << "]";
  return ss.str();
}
}
template<typename T>
ge::Status SetOutput(const int64_t *const output_data, size_t size, TensorAddress output_addr) {
  auto out_dims = ge::PtrToPtr<void, T>(output_addr);
  GE_ASSERT_NOTNULL(out_dims);
  for (size_t i = 0U; i < size; ++i) {
    GE_ASSERT_TRUE(ge::IntegerChecker<int32_t>::Compat(output_data[i]));
    out_dims[i] = static_cast<T>(output_data[i]);
  }
  return ge::SUCCESS;
}

ge::graphStatus GatherShapesKernel(KernelContext *context) {
  const ContinuousVectorVector *cvv = nullptr;
  GE_ASSERT_GRAPH_SUCCESS(GetAxes(context, &cvv));

  ge::DataType out_data_type;
  GE_ASSERT_GRAPH_SUCCESS(GetOutputDataType(context, out_data_type), "get data type failed");

  int64_t output_data[Shape::kMaxDimNum];
  GE_ASSERT_TRUE(cvv->GetSize() <= Shape::kMaxDimNum);
  for (size_t i = 0U; i < cvv->GetSize(); ++i) {
    const auto cv = cvv->Get(i);
    GE_ASSERT_NOTNULL(cv);
    GE_ASSERT_EQ(cv->GetSize(), 2U); // [it_index, dim_index]
    const uint64_t *data = reinterpret_cast<const uint64_t *>(cv->GetData());
    GE_ASSERT_NOTNULL(data);
    const auto input_index = *data + static_cast<size_t>(GatherShapeInputs::kShapeStart);
    GE_ASSERT(input_index < context->GetInputNum(), "input_index[%lu] must less than input num[%zu],"
              " i: %zu, axes: %s", input_index, context->GetInputNum(), i, PrintContinuousVectorVector(cvv).c_str());
    const auto dim_index = *(data + 1U);
    auto in_shape = context->GetInputPointer<StorageShape>(input_index);
    GE_ASSERT_NOTNULL(in_shape);
    GE_ASSERT(dim_index < in_shape->GetOriginShape().GetDimNum(), "dim_index[%lu] must less than input shape"
              " dim number[%zu], i: %zu, axes: %s",
              dim_index, in_shape->GetOriginShape().GetDimNum(), i, PrintContinuousVectorVector(cvv).c_str());
    const auto dim = in_shape->GetOriginShape().GetDim(dim_index);
    GE_ASSERT(dim != Shape::kInvalidDimValue);
    output_data[i] = dim;
  }
  auto out_tensor_data =
      context->GetOutputPointer<GertTensorData>(static_cast<size_t>(GatherShapesKernelOutputs::kTensorData));
  GE_ASSERT_NOTNULL(out_tensor_data);
  if (out_data_type == ge::DT_INT64) {
    GE_ASSERT_GRAPH_SUCCESS(SetOutput<int64_t>(output_data, cvv->GetSize(), out_tensor_data->GetAddr()));
  } else if (out_data_type == ge::DT_INT32) {
    GE_ASSERT_GRAPH_SUCCESS(SetOutput<int32_t>(output_data, cvv->GetSize(), out_tensor_data->GetAddr()));
  } else {
    GELOGE(ge::GRAPH_FAILED, "only support DT_INT32 and DT_INT64, but out_data_type is %s",
           ge::TypeUtils::DataTypeToSerialString(out_data_type).c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildGatherShapesOutput(const ge::FastNode *node, KernelContext *context) {
  GE_ASSERT_NOTNULL(node);

  uint32_t out_type = ge::DT_INT32;
  (void)ge::AttrUtils::GetInt(node->GetOpDescBarePtr(), "dtype", out_type);
  const auto data_type_size = ge::GetSizeByDataType(static_cast<ge::DataType>(out_type));
  GE_ASSERT_TRUE((out_type == ge::DT_INT32) || (out_type == ge::DT_INT64));

  size_t malloc_buffer_size = 0U;
  GE_ASSERT_TRUE(data_type_size > 0, "get data type size failed, data_type_size:%d", data_type_size);
  GE_ASSERT_TRUE(!ge::MulOverflow(static_cast<size_t>(data_type_size), Shape::kMaxDimNum, malloc_buffer_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(malloc_buffer_size, sizeof(GertTensorData), malloc_buffer_size));

  auto chain = context->GetOutput(0U);
  GE_ASSERT_NOTNULL(chain);
  auto out_data = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[malloc_buffer_size]);
  GE_ASSERT_NOTNULL(out_data);
  new (out_data.get())
      GertTensorData(out_data.get() + sizeof(GertTensorData), malloc_buffer_size - sizeof(GertTensorData), kOnHost, -1);
  chain->SetWithDefaultDeleter<uint8_t[]>(out_data.release());
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(GatherShapesKernel).RunFunc(GatherShapesKernel).OutputsCreator(BuildGatherShapesOutput);
}  // namespace kernel
}  // namespace gert
