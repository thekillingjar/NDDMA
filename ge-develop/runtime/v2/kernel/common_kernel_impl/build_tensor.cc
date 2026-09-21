/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "build_tensor.h"
#include <unordered_set>
#include "securec.h"
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "common/checker.h"
#include "exe_graph/runtime/extended_kernel_context.h"
#include "exe_graph/lowering/shape_utils.h"
#include "graph/types.h"
#include "graph/utils/type_utils.h"
#include "common/plugin/ge_make_unique_util.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "core/utils/tensor_utils.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "base/err_msg.h"

namespace gert {
namespace kernel {
namespace {
constexpr size_t kShapeDataTypeAttrIndex = 0U;
ge::graphStatus GetOutputDataType(const KernelContext *context, ge::DataType &out_data_type) {
  const auto extend_context = reinterpret_cast<const ExtendedKernelContext *>(context);
  const auto compute_node_info = extend_context->GetComputeNodeInfo();
  GE_ASSERT_NOTNULL(compute_node_info, "%s node info does not exist", extend_context->GetKernelName());
  const auto attrs = compute_node_info->GetAttrs();
  const int32_t *const data_type = attrs->GetAttrPointer<int32_t>(kShapeDataTypeAttrIndex);
  GE_ASSERT_NOTNULL(data_type, "%s dtype does not exist, ", extend_context->GetKernelName());
  out_data_type = static_cast<ge::DataType>(*data_type);
  return ge::GRAPH_SUCCESS;
}

template <typename T>
void GetTensorValue(TensorAddress addr, size_t num, std::stringstream &ss) {
  auto value_ptr = ge::PtrToPtr<void, T>(addr);
  for (size_t i = 0U; i < num; ++i) {
    ss << value_ptr[i] << ", ";
  }
}

ge::graphStatus GetShapeOutputValue(const KernelContext *context, std::stringstream &ss) {
  ge::DataType out_data_type;
  GE_ASSERT_GRAPH_SUCCESS(GetOutputDataType(context, out_data_type), "get data type failed");
  ss << "dtype: " << ge::TypeUtils::DataTypeToSerialString(out_data_type);
  auto inputs_begin = static_cast<size_t>(BuildShapeTensorDataInputs::kNum);
  for (size_t i = inputs_begin; i < context->GetInputNum(); ++i) {
    ss << " , output[" << (i - inputs_begin) << "], value: [";
    auto in_shape = context->GetInputPointer<StorageShape>(i);
    GE_ASSERT_NOTNULL(in_shape);
    auto tensor_data = context->GetOutputPointer<GertTensorData>(i - inputs_begin);
    GE_ASSERT_NOTNULL(tensor_data);
    if (out_data_type == ge::DataType::DT_INT32) {
      GetTensorValue<uint32_t>(tensor_data->GetAddr(), in_shape->GetOriginShape().GetDimNum(), ss);
    } else if (out_data_type == ge::DataType::DT_INT64) {
      GetTensorValue<uint64_t>(tensor_data->GetAddr(), in_shape->GetOriginShape().GetDimNum(), ss);
    } else {
      ss << "not support dtype";
    }
    ss << "]";
  }
  return ge::GRAPH_SUCCESS;
}

std::vector<std::string> ShapeTracer(const KernelContext *context) {
  std::stringstream ss;
  (void)GetShapeOutputValue(context, ss);
  return {ss.str()};
}
}  // namespace
ge::graphStatus BuildTensor(KernelContext *context) {
  auto storage_shape = context->GetInputPointer<StorageShape>(0U);
  auto tensor_data = context->MutableInputPointer<GertTensorData>(1U);
  auto tensor_attr = context->GetInputPointer<BuildTensorAttr>(2U);
  if ((storage_shape == nullptr) || (tensor_data == nullptr) || (tensor_attr == nullptr)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto tensor = context->GetOutputPointer<Tensor>(0U);
  GE_ASSERT_NOTNULL(tensor);
  tensor->MutableStorageShape() = storage_shape->GetStorageShape();
  tensor->MutableOriginShape() = storage_shape->GetOriginShape();
  tensor->SetDataType(tensor_attr->data_type);
  tensor->SetPlacement(tensor_attr->placement);
  tensor->MutableFormat() = tensor_attr->storage_format;
  TensorUtils::ShareGtdToTd(*tensor_data, tensor->MutableTensorData());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildRefTensor(KernelContext *context) {
  auto storage_shape = context->GetInputPointer<StorageShape>(0U);
  auto tensor_data = context->MutableInputPointer<GertTensorData>(1U);
  auto tensor_attr = context->GetInputPointer<BuildTensorAttr>(2U);
  if ((storage_shape == nullptr) || (tensor_data == nullptr) || (tensor_attr == nullptr)) {
    return ge::GRAPH_PARAM_INVALID;
  }
  auto tensor = context->GetOutputPointer<Tensor>(0U);
  GE_ASSERT_NOTNULL(tensor);
  tensor->MutableStorageShape() = storage_shape->GetStorageShape();
  tensor->MutableOriginShape() = storage_shape->GetOriginShape();
  tensor->SetDataType(tensor_attr->data_type);
  tensor->SetPlacement(tensor_attr->placement);
  tensor->MutableFormat() = tensor_attr->storage_format;
  TensorUtils::RefGtdToTd(*tensor_data, tensor->MutableTensorData());
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateBuildTensorOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto chain = context->GetOutput(0);
  GE_ASSERT_NOTNULL(chain);
  auto tensor = new (std::nothrow) Tensor();
  GE_ASSERT_NOTNULL(tensor);
  chain->SetWithDefaultDeleter(tensor);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(BuildTensorPureShape)
    .RunFunc(BuildTensor)
    .OutputsCreator(CreateBuildTensorOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(BuildTensorStorage)
    .RunFunc(BuildTensor)
    .OutputsCreator(CreateBuildTensorOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(BuildTensor).RunFunc(BuildTensor)
    .OutputsCreator(CreateBuildTensorOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);
REGISTER_KERNEL(BuildRefTensor).RunFunc(BuildRefTensor)
    .OutputsCreator(CreateBuildTensorOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

/*
 * SplitTensor不准备提供Guarder机制，后续如果需要Guarder机制，可以考虑实现SplitTensorWithGuarder
 * 不提供Guarder机制时，使用SplitTensor考虑如下场景：
 *                   Tensor
 *                 /   |    \
 *                /    |     FreeMemory1(Guarder)
 *               /     |     /c
 *          Shape  TensorData
 *                  |     \
 *                  |     FreeMemory2(Guarder)
 *                  |     /c
 *                  Launch
 *  如果当前在图中构造了一个Tensor并增加了Guarder(FreeMemory1)，上图场景能正常工作，因为TensorData也增加了Guarder，
 *  能保证laucnh时的地址是有效的，如果去除了SplitTensor的Guarder后：
 *
 *                   Tensor
 *                 /   |    \
 *                /    |     FreeMemory1(Guarder)
 *               /     |     /c
 *          Shape  TensorData
 *                     |
 *                     |
 *                  Launch
 *  当TensorData调度执行完，并不能保证Launch在FreeMemory1之前执行。
 *  但是现在由于SplitTensor是针对输入处理的，地址由执行器用户管理，执行器调度未结束,data节点内存不会释放，因此不会有问题。
 *  后续如果存在图中间产生了Tensor的场景，或者说用户需要GE在图中释放输入内存的场景，需要新增SplitTensorWithGuarder
 *
 * */
ge::graphStatus SplitDataTensor(KernelContext *context) {
  auto tensor = context->GetInputPointer<Tensor>(0UL);
  auto allocator = context->MutableInputPointer<GertAllocator>(1UL);
  GE_ASSERT_NOTNULL(allocator);
  auto shape_tensor = context->GetOutputPointer<Tensor>(static_cast<size_t>(SplitTensorOutputs::kShape));
  auto gtd = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(SplitTensorOutputs::kTensorData));
  if ((tensor == nullptr) || (shape_tensor == nullptr) || (gtd == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  shape_tensor->SetDataType(tensor->GetDataType());
  shape_tensor->MutableFormat() = tensor->GetFormat();
  shape_tensor->GetShape() = tensor->GetShape();
  // data tensor's TensorData has manager, so use ShareTensorToGtd.
  GE_ASSERT_SUCCESS(TensorUtils::ShareTensorToGtd(*tensor, *allocator, *gtd));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitConstTensor(KernelContext *context) {
  auto tensor = context->GetInputPointer<Tensor>(0UL);
  auto stream_id = context->GetInputPointer<int64_t>(1UL);
  GE_ASSERT_NOTNULL(stream_id);
  auto shape_tensor = context->GetOutputPointer<Tensor>(static_cast<size_t>(SplitTensorOutputs::kShape));
  auto gtd = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(SplitTensorOutputs::kTensorData));
  if ((tensor == nullptr) || (shape_tensor == nullptr) || (gtd == nullptr)) {
    return ge::GRAPH_FAILED;
  }
  shape_tensor->SetDataType(tensor->GetDataType());
  shape_tensor->MutableFormat() = tensor->GetFormat();
  shape_tensor->GetShape() = tensor->GetShape();
  // const tensor's TensorData doesn't have manager, so use RefTensorToGtd.
  GE_ASSERT_SUCCESS(TensorUtils::RefTensorToGtd(*tensor, *stream_id, *gtd));
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitTensorOutputsCreator(const ge::FastNode *, KernelContext *context) {
  auto shape_chain = context->GetOutput(static_cast<size_t>(SplitTensorOutputs::kShape));
  auto tensor_data_chain = context->GetOutput(static_cast<size_t>(SplitTensorOutputs::kTensorData));
  GE_ASSERT_NOTNULL(shape_chain);
  GE_ASSERT_NOTNULL(tensor_data_chain);
  auto shape_tensor = new (std::nothrow) Tensor();
  GE_ASSERT_NOTNULL(shape_tensor);
  shape_chain->SetWithDefaultDeleter(shape_tensor);

  auto gert_td = new (std::nothrow) GertTensorData;
  GE_ASSERT_NOTNULL(gert_td);
  tensor_data_chain->SetWithDefaultDeleter(gert_td);

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SplitDataTensor)
    .RunFunc(SplitDataTensor)
    .OutputsCreator(SplitTensorOutputsCreator)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

REGISTER_KERNEL(SplitConstTensor)
    .RunFunc(SplitConstTensor)
    .OutputsCreator(SplitTensorOutputsCreator);

/**
 * SplitTensorForOutputData专门为OutputData开发，因此SplitTensorForOutputData与SplitTensor的不同点：
 * 1. SplitTensorForOutputData 会校验Shape与TensorData的匹配度（TensorData的size，需要是shape和dtype对应的大小+padding）
 * 2. SplitTensorForOutputData 不会增加Tensor的引用计数，这就意味着后面不需要跟一个FreeMemory去释放内存
 * todo 本kernel需要校验一下placement，在用户传入错误的placement时，报错退出。
 *      校验palcement时，需要把allocator传入，通过allocator获取placement与用户传入的Tensor作比较
 * @param context
 * @return
 */
ge::graphStatus SplitTensorForOutputData(KernelContext *context) {
  auto tensor = context->GetInputPointer<Tensor>(static_cast<size_t>(SplitTensorForOutputDataInputs::kTensor));
  auto gert_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(SplitTensorForOutputDataInputs::kAllocator));
  auto out_shape_tensor = context->GetOutputPointer<Tensor>(static_cast<size_t>(SplitTensorOutputs::kShape));
  auto tensor_data = context->GetOutputPointer<GertTensorData>(static_cast<size_t>(SplitTensorOutputs::kTensorData));
  GE_ASSERT_NOTNULL(gert_allocator);
  if (out_shape_tensor == nullptr || tensor_data == nullptr) {
    return ge::GRAPH_FAILED;
  }

  if (tensor == nullptr || tensor->GetAddr() == nullptr) {
    GELOGE(ge::PARAM_INVALID,
           "In the `always_zero_copy` mode, the output tensor and tensor data must be allocated by the called");
    const std::string reason = "The output tensor and its address must be allocated and passed by the caller";
    const std::vector<std::string> key{"reason"};
    const std::vector<std::string> val{reason};
    REPORT_PREDEFINED_ERR_MSG("E13031", std::vector<const ge::char_t *>({"reason"}),
                               std::vector<const ge::char_t *>({reason.c_str()}));
    return ge::PARAM_INVALID;
  }

  // 外部申请的tensor内存要大于等于tensor的大小
  auto shape_size = tensor->GetShape().GetStorageShape().GetShapeSize();
  const auto storage_shape_size = ge::GetSizeInBytes(shape_size, tensor->GetDataType());
  if (tensor->GetSize() < static_cast<size_t>(storage_shape_size)) {
    GELOGE(ge::PARAM_INVALID,
           "The output tensor memory size[%zu] is smaller than the actual size[%" PRId64 "] required by tensor.",
           tensor->GetSize(), storage_shape_size);
    const std::string reason = "The output tensor memory size " + std::to_string(tensor->GetSize()) +
                               " is smaller than the actual size " + std::to_string(storage_shape_size) +
                               " required by tensor";
    const std::vector<std::string> key{"reason"};
    const std::vector<std::string> val{reason};
    REPORT_PREDEFINED_ERR_MSG("E13031", std::vector<const ge::char_t *>({"reason"}),
                               std::vector<const ge::char_t *>({reason.c_str()}));
    return ge::PARAM_INVALID;
  }
  out_shape_tensor->SetDataType(tensor->GetDataType());
  out_shape_tensor->MutableFormat() = tensor->GetFormat();
  out_shape_tensor->GetShape() = tensor->GetShape();
  // todo 当前PTA传入的size不满足32对齐+32，因此此处不做校验了，待后续结论明确后确定是否做校验
  // SplitTensorForOutputData在pass里产生，使用的是
  GE_ASSERT_SUCCESS(TensorUtils::ShareTdToGtd(tensor->GetTensorData(), *gert_allocator, *tensor_data));

  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(SplitTensorForOutputData)
    .RunFunc(SplitTensorForOutputData)
    .OutputsCreator(SplitTensorOutputsCreator)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);;

ge::graphStatus BuildShapeTensorData(KernelContext *context) {
  ge::DataType out_data_type;
  GE_ASSERT_GRAPH_SUCCESS(GetOutputDataType(context, out_data_type), "get data type failed");
  if (out_data_type != ge::DataType::DT_INT32) {
    return ge::GRAPH_SUCCESS;
  }
  for (size_t i = 0U; i < context->GetInputNum(); ++i) {
    auto in_shape = context->GetInputPointer<StorageShape>(i);
    GE_ASSERT_NOTNULL(in_shape);
    const int64_t *const in_shape_dims = &(in_shape->GetOriginShape()[0]);
    GE_ASSERT_NOTNULL(in_shape_dims);
    auto gtd = context->GetOutputPointer<GertTensorData>(i);
    GE_ASSERT_NOTNULL(gtd);
    auto out_dims = ge::PtrToPtr<void, int32_t>(gtd->GetAddr());
    GE_ASSERT_NOTNULL(out_dims);
    for (size_t j = 0U; j < in_shape->GetOriginShape().GetDimNum(); ++j) {
      out_dims[j] = static_cast<int32_t>(in_shape_dims[j]);
    }
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildShapeTensorDataCreateOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  GE_ASSERT_EQ(context->GetInputNum(), context->GetOutputNum());

  ge::DataType out_data_type;
  GE_ASSERT_GRAPH_SUCCESS(GetOutputDataType(context, out_data_type), "get data type failed");
  GE_ASSERT_TRUE(out_data_type == ge::DataType::DT_INT32 || out_data_type == ge::DataType::DT_INT64,
                 "only support DT_INT32 and DT_INT64, but out_data_type is %s",
                 ge::TypeUtils::DataTypeToSerialString(out_data_type).c_str());
  const auto is_malloc = (out_data_type == ge::DataType::DT_INT32);

  const auto data_type_size = ge::GetSizeByDataType(out_data_type);
  size_t malloc_buffer_size = 0U;
  GE_ASSERT_TRUE(data_type_size > 0, "get data type size failed, data_type_size:%d", data_type_size);
  GE_ASSERT_TRUE(!ge::MulOverflow(static_cast<size_t>(data_type_size), Shape::kMaxDimNum, malloc_buffer_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(malloc_buffer_size, sizeof(GertTensorData), malloc_buffer_size));

  for (size_t i = 0U; i < context->GetInputNum(); ++i) {
    auto chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(chain);
    if (is_malloc) {
      auto out_data = ge::MakeUnique<uint8_t[]>(malloc_buffer_size);
      GE_ASSERT_NOTNULL(out_data);
      new (out_data.get()) GertTensorData(out_data.get() + sizeof(GertTensorData),
                                          malloc_buffer_size - sizeof(GertTensorData), kOnHost, -1);
      chain->SetWithDefaultDeleter<uint8_t[]>(out_data.release());
    } else {
      auto shape = context->GetInputPointer<StorageShape>(i);
      GE_ASSERT_NOTNULL(shape);
      auto gtd = new (std::nothrow) GertTensorData();
      GE_ASSERT_NOTNULL(gtd);
      *gtd = GertTensorData{const_cast<void *>(reinterpret_cast<const void *>(&(shape->GetOriginShape()[0]))),
                            Shape::kMaxDimNum * sizeof(int64_t), kOnHost, -1};
      chain->SetWithDefaultDeleter(gtd);
    }
  }

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(BuildShapeTensorData)
    .RunFunc(BuildShapeTensorData)
    .OutputsCreator(BuildShapeTensorDataCreateOutputs)
    .TracePrinter(ShapeTracer);
}  // namespace kernel
}  // namespace gert
