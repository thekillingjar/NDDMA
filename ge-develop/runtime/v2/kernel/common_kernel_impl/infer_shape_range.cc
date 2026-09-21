/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "infer_shape_range.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/ge_log.h"
#include "kernel/kernel_log.h"
#include "runtime/mem.h"
#include "transfer_shape_according_to_format.h"
#include "infer_shape.h"
#include "base/registry/op_impl_space_registry_v2.h"

namespace gert {
namespace kernel {
namespace {
void ShapeRangeToStringStream(std::stringstream &ss, const Range<Shape> &shape_range) {
  const auto &max = shape_range.GetMax();
  const auto &min = shape_range.GetMin();
  if ((max == nullptr) || (min == nullptr)) {
    ss << "max or min is nullptr. max:" << max << ", min:" << min;
    return;
  }
  if (max->GetDimNum() != min->GetDimNum()) {
    ss << "dim num not match. max dim num: " << max->GetDimNum() << ", min dim num: " << min->GetDimNum();
    return;
  }
  ss << "[";
  for (size_t j = 0U; j < max->GetDimNum(); ++j) {
    ss << "[" << min->GetDim(j) << "," << max->GetDim(j) << "]";
    if (j + 1U < max->GetDimNum()) {
      ss << ", ";
    }
  }
  ss << "]";
}

void PrintFormatDtypeShapeRange(std::stringstream &ss, ge::Format format, ge::DataType type,
                                const Range<Shape> &shape_range) {
  ss << "[";
  ss << ge::TypeUtils::FormatToSerialString(format) << " ";
  ss << ge::TypeUtils::DataTypeToSerialString(type) << " ";
  ShapeRangeToStringStream(ss, shape_range);
  ss << "]";
}

std::vector<std::string> InferShapeRangeKernelTrace(const KernelContext *context) {
  return {PrintNodeType(context),
          PrintInputRangeInfo(context, 0U),
          PrintOutputRangeInfo(context)};
}
} // namespace

std::string PrintInputRangeInfo(const KernelContext *const context, const size_t &input_range_start_index) {
  std::stringstream ss;
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return "compute_node_info is nullptr";
  }
  if (context->GetInputNum() < input_range_start_index) {
    ss << "Trace failed, input num < input_range_start_index, "
       << "context->GetInputNum:" << context->GetInputNum()
       << ", input_range_start_index:" << input_range_start_index;
    return ss.str();
  }
  ss << "input shape ranges : ";
  for (size_t i = 0U; i < compute_node_info->GetInputsNum(); ++i) {
    auto td = compute_node_info->GetInputTdInfo(i);
    if (td == nullptr) {
      return "The " + to_string(i) + "th's input tensor desc is nullptr";
    }
    auto shape_range = context->GetInputPointer<Range<Shape>>(i + input_range_start_index);
    if (shape_range == nullptr) {
      return "The " + to_string(i) + "th's input shape range is nullptr";
    }
    PrintFormatDtypeShapeRange(ss, td->GetOriginFormat(), td->GetDataType(), *shape_range);
    if (i + 1U < compute_node_info->GetInputsNum()) {
      ss << ", ";
    }
  }
  return ss.str();
}

std::string PrintOutputRangeInfo(const KernelContext *context) {
  std::stringstream ss;
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return "compute_node_info is nullptr";
  }
  ss << "output shape ranges : ";
  for (size_t i = 0U; i < compute_node_info->GetOutputsNum(); ++i) {
    auto td = compute_node_info->GetOutputTdInfo(i);
    if (td == nullptr) {
      return "The " + std::to_string(i) + "th's output tensor desc is nullptr";
    }
    auto shape_range = context->GetOutputPointer<Range<Shape>>(i);
    if (shape_range == nullptr) {
      return "The " + std::to_string(i) + "th's output shape range is nullptr";
    }
    PrintFormatDtypeShapeRange(ss, td->GetOriginFormat(), td->GetDataType(), *shape_range);
    if (i + 1U < compute_node_info->GetOutputsNum()) {
      ss << ", ";
    }
  }
  return ss.str();
}

ge::graphStatus TransformAllOutputsMaxShape(const ComputeNodeInfo *compute_node_info, KernelContext *context) {
  const size_t node_outputs_num = compute_node_info->GetOutputsNum();
  for (size_t index = 0U; index < node_outputs_num; ++index) {
    auto output_td = compute_node_info->GetOutputTdInfo(index);
    GE_ASSERT_NOTNULL(output_td);
    auto storage_shape = context->GetOutputPointer<StorageShape>(node_outputs_num * 2U + index);
    GE_ASSERT_NOTNULL(storage_shape);
    GE_CHK_STATUS_RET(TransformOutputShape(compute_node_info, output_td, storage_shape),
                      "Fail to transfer node %s %zu output shape according format.",
                      compute_node_info->GetNodeName(), index);
  }
  return ge::GRAPH_SUCCESS;
}

/*
 * 3n个输出，0~n-1 Range<Shape>, n~2n-1 min shape，2n~3n-1 max shape,n是node输出个数
 */
ge::graphStatus BuildInferShapeRangeOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  size_t node_output_num = static_cast<const ComputeNodeInfo *>(context->GetComputeNodeExtend())->GetOutputsNum();
  GE_ASSERT_EQ(context->GetOutputNum(), node_output_num * kShapeRangeOutputOfNode);
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  for (size_t index = 0U; index < node_output_num; index++) {
    auto min_shape_chain = context->GetOutput(node_output_num + index);
    GE_ASSERT_NOTNULL(min_shape_chain);
    auto output_desc = extend_context->GetOutputDesc(index);
    GE_ASSERT_NOTNULL(output_desc);
    auto min_shape_tensor = new (std::nothrow) Tensor(StorageShape(),
        output_desc->GetFormat(), output_desc->GetDataType());
    GE_ASSERT_NOTNULL(min_shape_tensor);
    min_shape_chain->SetWithDefaultDeleter(min_shape_tensor);

    auto max_shape_chain = context->GetOutput(2U * node_output_num + index);
    GE_ASSERT_NOTNULL(max_shape_chain);
    auto max_shape_tensor = new (std::nothrow) Tensor(StorageShape(),
        output_desc->GetFormat(), output_desc->GetDataType());
    GE_ASSERT_NOTNULL(max_shape_tensor);
    max_shape_chain->SetWithDefaultDeleter(max_shape_tensor);

    auto shape_range_chain = context->GetOutput(index);
    GE_ASSERT_NOTNULL(shape_range_chain);
    auto shape_range = new (std::nothrow) Range<Tensor>(min_shape_tensor,
                                                        max_shape_tensor);
    GE_ASSERT_NOTNULL(shape_range);
    shape_range_chain->SetWithDefaultDeleter(shape_range);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FindInferShapeRangeFunc(KernelContext *context) {
  auto node_type = context->GetInputValue<char *>(0);
  auto space_registry = context->GetInputValue<gert::OpImplSpaceRegistryV2 *>(1);
  auto infer_fun_ptr = context->GetOutputPointer<OpImplKernelRegistry::InferShapeRangeKernelFunc>(0);
  if (node_type == nullptr || infer_fun_ptr == nullptr || space_registry == nullptr) {
    KLOGE("Failed to find infer shape range kernel, input or output is nullptr");
    return ge::GRAPH_FAILED;
  }
  auto op_funcs = space_registry->GetOpImpl(node_type);
  if (op_funcs == nullptr) {
    KLOGE("Failed to find infer shape range kernel, node type %s", node_type);
    return ge::GRAPH_FAILED;
  }
  if (op_funcs->infer_shape_range == nullptr) {
    KLOGE("Failed to find infer shape range kernel for node %s", node_type);
    return ge::GRAPH_FAILED;
  }
  *infer_fun_ptr = op_funcs->infer_shape_range;
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FindInferShapeRangeFunc).RunFunc(FindInferShapeRangeFunc);

/*
 * input: all shape ranges , op_infershaperange_fun;
 * output: all shape ranges
 */
KernelRegistry::KernelFunc GetOpInferShapeRangeFun(const KernelContext *const context) {
  auto input_num = context->GetInputNum();
  if (input_num < 1U) {
    return nullptr;
  }
  return context->GetInputValue<KernelRegistry::KernelFunc>(input_num - 1U);
}

ge::graphStatus InferShapeRange(KernelContext *context) {
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto op_infer_fun = GetOpInferShapeRangeFun(context);
  GE_ASSERT_NOTNULL(op_infer_fun);
  auto ret = op_infer_fun(context);
  if (ret != ge::GRAPH_SUCCESS) {
    KLOGE("infer shape range failed, node type %s, name %s, error-code %u", extend_context->GetNodeType(),
          extend_context->GetNodeName(), ret);
    return ret;
  }
  ret = TransformAllOutputsMaxShape(compute_node_info, context);
  if (ret != ge::GRAPH_SUCCESS) {
    KLOGE("Failed to trans shape to 5D when infer shape for node %s, type %s, error-code %u",
          extend_context->GetNodeName(), extend_context->GetNodeType(), ret);
    return ret;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateRanges(KernelContext *context) {
  size_t output_num = context->GetOutputNum();
  GE_ASSERT_EQ(context->GetInputNum(), output_num);
  for (size_t index = 0U; index < output_num; ++index) {
    auto tensor = context->MutableInputPointer<Tensor>(index);
    GE_ASSERT_NOTNULL(tensor);
    auto out_tensor_range = context->GetOutputPointer<TensorRange>(index);
    GE_ASSERT_NOTNULL(out_tensor_range);
    out_tensor_range->SetMax(tensor);
    out_tensor_range->SetMin(tensor);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildRanges(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  size_t output_num = context->GetOutputNum();
  GE_ASSERT_EQ(context->GetInputNum(), output_num);
  for (size_t index = 0; index < context->GetOutputNum(); index++) {
    auto tensor = context->MutableInputPointer<Tensor>(index);
    GE_ASSERT_NOTNULL(tensor);
    auto chain = context->GetOutput(index);
    GE_ASSERT_NOTNULL(chain);
    chain->SetWithDefaultDeleter(new (std::nothrow) TensorRange(tensor));
  }
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(InferShapeRange).RunFunc(InferShapeRange).OutputsCreator(BuildInferShapeRangeOutputs)
    .TracePrinter(InferShapeRangeKernelTrace);
REGISTER_KERNEL(CreateTensorRangesAndShapeRanges).RunFunc(CreateRanges).OutputsCreator(BuildRanges);
}  // namespace kernel
}  // namespace gert
