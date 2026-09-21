/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom_op_kernel.h"
#include "register/kernel_registry.h"
#include "common/checker.h"
#include "graph/custom_op_factory.h"
#include "graph/custom_op.h"
#include "kernel/memory/multi_stream_mem_block.h"
#include "graph/def_types.h"
#include "graph/utils/type_utils.h"

namespace gert {
namespace kernel {
namespace {
enum class CustomOpInput {
  kAllocator = 0,
  kStream,
  kFunc,
  kEnd
};

std::string PrintNodeType(const KernelContext *context) {
  std::stringstream ss;
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return "compute_node_info is nullptr";
  }
  ss << "node_type[" << compute_node_info->GetNodeType() << "]";
  return ss.str();
}

void ShapeToStringStream(std::stringstream &ss, const Shape &shape) {
  ss << "[";
  for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
    ss << shape.GetDim(j);
    if (j + 1U < shape.GetDimNum()) {
      ss << ", ";
    }
  }
  ss << "]";
}

void PrintTensor(std::stringstream &ss, const gert::Tensor *tensor) {
  ss << "format: ";
  ss << ge::TypeUtils::FormatToSerialString(tensor->GetStorageFormat()) << ", ";
  ss << "datatype: ";
  ss << ge::TypeUtils::DataTypeToSerialString(tensor->GetDataType()) << ", ";
  ss << "shape: ";
  ShapeToStringStream(ss, tensor->GetStorageShape());
  ss << ", addr: " << ge::PtrToValue(tensor->GetAddr());
}
}

ge::graphStatus FindCustomOpFunc(KernelContext *context) {
  const char *node_type = context->GetInputValue<char *>(0);
  GE_ASSERT_NOTNULL(node_type, "Failed to find custom op func, node type is nullptr");
  auto custom_op_ptr = ge::CustomOpFactory::CreateCustomOp(node_type);
  GE_ASSERT_NOTNULL(custom_op_ptr);
  auto chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(chain);
  chain->SetWithDefaultDeleter(custom_op_ptr.release());
  return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CreateWorkspacesMemory(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto *extended_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  const size_t node_output_num = extended_kernel_context->GetComputeNodeOutputNum();
  for (size_t index = 0; index < node_output_num; index++) {
    auto chain = context->GetOutput(index);
    GE_ASSERT_NOTNULL(chain);
    auto output_desc = extended_kernel_context->GetOutputDesc(index);
    GE_ASSERT_NOTNULL(output_desc);
    chain->SetWithDefaultDeleter(new (std::nothrow) Tensor(StorageShape(),
        output_desc->GetFormat(), output_desc->GetDataType()));
  }
  auto workspace_memory_av = context->GetOutput(node_output_num);
  auto workspace_memory_holder = new (std::nothrow) std::vector<memory::MultiStreamMemBlock *>();
  GE_ASSERT_NOTNULL(workspace_memory_holder);
  // 节省MallocWorkspace时vector添加元素时动态扩容的开销
  workspace_memory_holder->reserve(1U);
  workspace_memory_av->SetWithDefaultDeleter(workspace_memory_holder);
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus ExecuteCustomOpFunc(KernelContext *context) {
  // OpExecuteContext *op_execute_context = reinterpret_cast<OpExecuteContext *>(context);
  auto *extended_kernel_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extended_kernel_context);
  const size_t node_input_num = extended_kernel_context->GetComputeNodeInputNum();
  auto custom_op_ptr =
      context->GetInputValue<ge::BaseCustomOp *>(node_input_num + static_cast<size_t>(CustomOpInput::kFunc));
  GE_ASSERT_NOTNULL(custom_op_ptr);
  GE_ASSERT_SUCCESS(custom_op_ptr->Execute(reinterpret_cast<EagerOpExecutionContext *>(context)));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus FreeCustomOpWorkspacesFunc(KernelContext *context) {
  auto memory_vec =
      context->MutableInputPointer<std::vector<GertMemBlock *>>(0);
  GE_ASSERT_NOTNULL(memory_vec);
  auto gert_allocator = context->GetInputValue<GertAllocator *>(1);
  GE_ASSERT_NOTNULL(gert_allocator);
  auto stream_id = gert_allocator->GetStreamId();
  for (size_t i = 0UL; i < memory_vec->size(); i++) {
    if (memory_vec->at(i) != nullptr) {
      memory_vec->at(i)->Free(stream_id);
    }
  }
  memory_vec->clear();
  return ge::GRAPH_SUCCESS;
}

static std::vector<std::string> CustomOpExecuteKernelTrace(const KernelContext *context) {
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  if (compute_node_info == nullptr) {
    return {PrintNodeType(context), "compute_node_info is nullptr"};
  }
  auto *eager_op_context = reinterpret_cast<const EagerOpExecutionContext *>(context);
  std::stringstream input_tensor_ss;
  input_tensor_ss << "input tensor: ";
  for (size_t i = 0U; i < compute_node_info->GetInputsNum(); ++i) {
    auto tensor = eager_op_context->GetInputTensor(i);
    if (tensor == nullptr) {
      return {PrintNodeType(context), "The " + std::to_string(i) + "th's input tensor is nullptr"};
    }
    input_tensor_ss << "[" << i << "]: [";
    PrintTensor(input_tensor_ss, tensor);
    input_tensor_ss << "]";
    if ((i + 1U) < compute_node_info->GetInputsNum()) {
      input_tensor_ss << ", ";
    }
  }
  std::stringstream output_tensor_ss;
  output_tensor_ss << "output tensor: ";
  for (size_t i = 0U; i < compute_node_info->GetOutputsNum(); ++i) {
    auto tensor = eager_op_context->GetOutputTensor(i);
    if (tensor == nullptr) {
      return {PrintNodeType(context), "The " + std::to_string(i) + "th's output tensor is nullptr"};
    }
    output_tensor_ss << "[" << i << "]: [";
    PrintTensor(output_tensor_ss, tensor);
    output_tensor_ss << "]";
    if ((i + 1U) < compute_node_info->GetOutputsNum()) {
      output_tensor_ss << ", ";
    }
  }
  return {PrintNodeType(context), input_tensor_ss.str(), output_tensor_ss.str()};  
}

ge::graphStatus CustomOpProfilingDataFill(const KernelContext *context, ProfilingInfoWrapper &prof_info) {
  prof_info.SetBlockDim(std::numeric_limits<uint32_t>::max());
  auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info);
  auto node_input_num = compute_node_info->GetInputsNum();
  const auto eager_context = reinterpret_cast<const EagerOpExecutionContext *>(context);
  GE_ASSERT_NOTNULL(eager_context);
  std::vector<std::vector<int64_t>> input_shapes;
  for (size_t i = 0UL; i < node_input_num; i++) {
    auto tensor = eager_context->GetInputTensor(i);
    GE_ASSERT_NOTNULL(tensor);
    auto shape = tensor->GetStorageShape();
    std::vector<int64_t> dims;
    for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
      dims.emplace_back(shape.GetDim(j));
    }
    input_shapes.emplace_back(dims);
  }
  auto node_output_num = compute_node_info->GetOutputsNum();
  std::vector<std::vector<int64_t>> output_shapes;
  for (size_t i = 0UL; i < node_output_num; i++) {
    auto tensor = eager_context->GetOutputTensor(i);
    GE_ASSERT_NOTNULL(tensor);
    auto shape = tensor->GetStorageShape();
    std::vector<int64_t> dims;
    for (size_t j = 0U; j < shape.GetDimNum(); ++j) {
      dims.emplace_back(shape.GetDim(j));
    }
    output_shapes.emplace_back(dims);
  }
  GE_ASSERT_SUCCESS(prof_info.FillShapeInfo(input_shapes, output_shapes));
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(FindCustomOp).RunFunc(FindCustomOpFunc);
REGISTER_KERNEL(ExecuteCustomOp).OutputsCreator(CreateWorkspacesMemory)
    .RunFunc(ExecuteCustomOpFunc).TracePrinter(CustomOpExecuteKernelTrace)
    .ProfilingInfoFiller(CustomOpProfilingDataFill);
REGISTER_KERNEL(FreeCustomOpWorkspaces).RunFunc(FreeCustomOpWorkspacesFunc);
}
}