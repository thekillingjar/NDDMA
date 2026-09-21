/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstddef>
#include <iomanip>
#include <cinttypes>
#include "graph/ge_error_codes.h"
#include "graph/def_types.h"
#include "register/kernel_registry.h"
#include "framework/common/debug/log.h"
#include "exe_graph/runtime/tensor.h"
#include "kernel/memory/mem_block.h"
#include "runtime/mem.h"
#include "common/checker.h"
#include "runtime/kernel.h"
#include "proto/task.pb.h"
#include "common/plugin/ge_make_unique_util.h"
#include "core/debug/kernel_tracing.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "kernel/kernel_log.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "core/utils/tensor_utils.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/lowering/shape_utils.h"

using namespace ge;
namespace gert {
namespace kernel {
const int32_t kInputNum = 2;
const int32_t kDimNum = 2;
enum class BuildAddrFromTensorComputeInputs {
  kTensor,
  kAllocator,
};
enum class SplitToSequenceArgIndex {
  KAxis,
  KInputXShape,
  KInputXAddr,
  KSplitShapes,
  KInnerSplitAddrs
};

enum class GetSplitVecArgIndex {
  KAxis,
  KInputNum,
  KInputXShape,
  KSplitShapes,
  KSplitAddr,
  KInnerSplitAddrs
};

struct CopyTensorArg {
  void* src;
  void* dst;
  int32_t type_size;
};

const std::set<ge::DataType> kSupportedDataType = {
    DT_UINT8,
    DT_UINT16,
    DT_UINT32,
    DT_UINT64,
    DT_INT8,
    DT_INT16,
    DT_INT32,
    DT_INT64,
    DT_FLOAT16,
    DT_FLOAT,
    DT_DOUBLE,
    DT_BOOL,
    DT_COMPLEX64,
    DT_COMPLEX128
};

ge::graphStatus AllocSpecifiedMem(KernelContext *context) {
  auto gert_allocator = context->GetInputValue<GertAllocator *>(0);
  GE_ASSERT_NOTNULL(gert_allocator);
  auto shape_sizes = context->GetInputPointer<TypedContinuousVector<uint64_t>>(1);
  GE_ASSERT_NOTNULL(shape_sizes);

  auto tensor_num = shape_sizes->GetSize();
  auto tensor_data_addrs = ContinuousVector::Create<TensorData*>(tensor_num);
  auto tensor_data_vec = reinterpret_cast<ContinuousVector*>(tensor_data_addrs.get());
  GE_ASSERT_NOTNULL(tensor_data_vec);
  tensor_data_vec->SetSize(tensor_num);

  auto shape_sizes_data = shape_sizes->GetData();
  auto tensor_data_info = static_cast<TensorData**>(tensor_data_vec->MutableData());
  for (size_t i = 0; i < tensor_num; ++i) {
    auto mem_block = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(shape_sizes_data[i]));
    KERNEL_CHECK_NOTNULL(mem_block);
    KERNEL_CHECK(mem_block->GetAddr() != nullptr, "malloc failed, tensor size=%zu", shape_sizes_data[i]);
    tensor_data_info[i] = new (std::nothrow) TensorData();
    GE_ASSERT_NOTNULL(tensor_data_info[i]);
    *(tensor_data_info[i]) =
        TensorUtils::ToTensorData(mem_block->GetMemBlock(), mem_block->GetSize(), gert_allocator->GetPlacement());

    KERNEL_TRACE(TRACE_STR_ALLOC_MEM", index %zu, tensor data address %p", 0, mem_block, mem_block->GetAddr(),
                 shape_sizes_data[i], i, tensor_data_info[i]->GetAddr());
  }

  auto tensor_data_av = context->GetOutput(0);
  GE_ASSERT_NOTNULL(tensor_data_av);
  tensor_data_av->SetWithDefaultDeleter<uint8_t[]>(tensor_data_addrs.release());
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(AllocSpecifiedMem).RunFunc(AllocSpecifiedMem).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus FreeSpecifiedMem(KernelContext *context) {
  auto addrs = context->MutableInputPointer<TypedContinuousVector<TensorData*>>(0);
  GE_ASSERT_NOTNULL(addrs);
  for (size_t i = 0; i < addrs->GetSize(); ++i) {
    if (addrs->MutableData()[i] != nullptr) {
      KERNEL_TRACE("[MEM]Free memory, index %zu, address %p", i, addrs->MutableData()[i]->GetAddr());
      delete addrs->MutableData()[i];
      addrs->MutableData()[i] = nullptr;
    }
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(FreeSpecifiedMem).RunFunc(FreeSpecifiedMem).ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus CalcTensorSizeFromSpecifiedShape(KernelContext *context) {
  auto datatype = context->GetInputValue<ge::DataType>(0);
  auto shapes = context->GetInputPointer<TypedContinuousVector<Shape>>(1);
  auto addrs_av = context->GetOutput(0);
  GE_ASSERT_NOTNULL(shapes);
  GE_ASSERT_NOTNULL(addrs_av);

  auto shape_num = shapes->GetSize();
  auto shape_data = shapes->GetData();
  auto shape_sizes_addrs = ContinuousVector::Create<uint64_t>(shape_num);
  GE_ASSERT_NOTNULL(shape_sizes_addrs);
  auto shapes_addrs_vec = reinterpret_cast<ContinuousVector*>(shape_sizes_addrs.get());
  shapes_addrs_vec->SetSize(shape_num);
  auto shape_addrs_data = static_cast<uint64_t*>(shapes_addrs_vec->MutableData());
  for (size_t i = 0; i < shape_num; i++) {
    GE_ASSERT_GRAPH_SUCCESS(CalcAlignedSizeByShape(shape_data[i], datatype, shape_addrs_data[i]));
  }

  addrs_av->SetWithDefaultDeleter<uint8_t[]>(shape_sizes_addrs.release());
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(CalcTensorSizeFromSpecifiedShape).RunFunc(CalcTensorSizeFromSpecifiedShape);

template <typename T>
ge::graphStatus BuildSplitVecWithInput(const gert::StorageShape &split_storeage_shape,
    const gert::GertTensorData *split_addr,  std::vector<int64_t> &split_vec, const int32_t input_x_axis_dim) {
  gert::Shape split_shape = split_storeage_shape.GetStorageShape();
  auto split_rank = split_shape.GetDimNum();
  GE_ASSERT_TRUE((split_rank < kDimNum), "split must be scalar or 1D, but given %u", split_rank);
  uint32_t split_num = 0;
  if (split_rank == 0) {
    T split_scalar;
    GE_ASSERT_RT_OK(rtMemcpy(&split_scalar, sizeof(T), split_addr->GetAddr(), sizeof(T),
        RT_MEMCPY_DEVICE_TO_HOST));
    GE_ASSERT_TRUE((split_scalar > 0), "split scalar must be greater than 0,%d", split_scalar);
    auto num_even_split = input_x_axis_dim / split_scalar;
    auto num_remain = input_x_axis_dim % split_scalar;
    split_num = num_even_split;
    if (num_remain != 0) {
      split_num += 1;
    }
    split_vec.resize(split_num);
    std::fill(split_vec.begin(), split_vec.begin() + num_even_split, split_scalar);
    std::fill(split_vec.begin() + num_even_split, split_vec.end(), num_remain);
  } else if (split_rank == 1) {
    split_num = split_shape.GetDim(0);
    split_vec.resize(split_num);
    auto temp_array = MakeUnique<T[]>(split_num);
    GE_ASSERT_RT_OK(rtMemcpy(temp_array.get(), sizeof(T) * split_num,
        split_addr->GetAddr(), sizeof(T) * split_num, RT_MEMCPY_DEVICE_TO_HOST));
    auto split_sum = 0;
    for (size_t i = 0; i < split_num; i++) {
      GE_ASSERT_TRUE((temp_array[i] > 0), "split must be greater than 0,but get:%ld", temp_array[i]);
      split_vec[i] = temp_array[i];
      split_sum += split_vec[i];
    }
    GE_ASSERT_EQ(split_sum, input_x_axis_dim);
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetSplitVec(gert::KernelContext *context) {
  auto axis = context->GetInputValue<int32_t>(static_cast<size_t>(GetSplitVecArgIndex::KAxis));
  auto input_num = context->GetInputValue<uint32_t>(static_cast<size_t>(GetSplitVecArgIndex::KInputNum));
  auto input_storage_shape = context->GetInputPointer<gert::StorageShape>(
        static_cast<size_t>(GetSplitVecArgIndex::KInputXShape));
  GE_ASSERT_NOTNULL(input_storage_shape);
  gert::Shape input_x_shape = input_storage_shape->GetStorageShape();
  auto input_tensor_rank = static_cast<int32_t>(input_x_shape.GetDimNum());
  GE_ASSERT_TRUE((axis >= -input_tensor_rank) && (axis <= (input_tensor_rank - 1)),
      "axis %d, is out of range:[-%d,%d].", axis, input_tensor_rank, input_tensor_rank - 1);
  if (axis < 0) {
    axis += input_tensor_rank;
  }
  const int64_t input_x_axis_dim = input_x_shape.GetDim(axis);
  std::vector<int64_t> split_sizes;
  auto extend_ctx = reinterpret_cast<gert::ExtendedKernelContext *>(context);
  if (input_num == kInputNum) {
    auto split_storage_shape = *context->GetInputPointer<gert::StorageShape>(
        static_cast<size_t>(GetSplitVecArgIndex::KSplitShapes));
    auto split_addr =
        context->GetInputPointer<gert::GertTensorData>(static_cast<size_t>(GetSplitVecArgIndex::KSplitAddr));
    auto split_desc = extend_ctx->GetInputDesc(1);
    GE_ASSERT_NOTNULL(split_desc);
    ge::DataType split_data_type = split_desc->GetDataType();
    GE_ASSERT_TRUE((split_data_type == DT_INT32) || (split_data_type == DT_INT64),
        "only support int32_t or int64_t for split, %u", split_data_type);
    switch (split_data_type) {
      case DT_INT32:
        GE_ASSERT_GRAPH_SUCCESS(BuildSplitVecWithInput<int32_t>(split_storage_shape, split_addr,
            split_sizes, input_x_axis_dim), "Build split vec err");
        break;
      case DT_INT64:
        GE_ASSERT_GRAPH_SUCCESS(BuildSplitVecWithInput<int64_t>(split_storage_shape, split_addr,
            split_sizes, input_x_axis_dim), "Build split vec err");
        break;
      default:
        return GRAPH_FAILED;
    }
  } else {
    split_sizes.resize(input_x_axis_dim);
    std::fill(split_sizes.begin(), split_sizes.begin() + input_x_axis_dim, 1);
  }

  auto output_num = split_sizes.size();
  std::vector<gert::Shape> shape_after_split(output_num, input_x_shape);
  for (size_t i = 0; i < output_num; i++) {
    shape_after_split[i].SetDim(axis, split_sizes[i]);
  }

  auto addrs_av = context->GetOutput(0);
  GE_ASSERT_NOTNULL(addrs_av);
  auto addrs = gert::ContinuousVector::Create<gert::Shape>(output_num);
  auto addrs_vec = reinterpret_cast<gert::ContinuousVector*>(addrs.get());
  addrs_vec->SetSize(output_num);
  auto shapes_addrs = reinterpret_cast<gert::Shape*>(addrs_vec->MutableData());
  for (size_t i = 0; i < output_num; i++) {
    shapes_addrs[i] = shape_after_split[i];
  }

  addrs_av->SetWithDefaultDeleter<uint8_t[]>(addrs.release());
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(GetSplitVecWithInput).RunFunc(GetSplitVec);

int64_t CaculateSize(const gert::Shape &shape, size_t start, size_t end) {
  int64_t size = 1;
  if (end > shape.GetDimNum()) {
    return size;
  }
  for (auto i = start; i < end; i++) {
    size *= shape.GetDim(i);
  }
  return size;
}

int64_t SizeToSplitAxis(const gert::Shape &shape, int32_t axis) {
  return CaculateSize(shape, 0, axis);
}

int64_t SizeFromSplitAxis(const gert::Shape &shape, int32_t axis) {
  return CaculateSize(shape, axis, shape.GetDimNum());
}

ge::graphStatus CopyTensor(int64_t M, int64_t N, int lda, int64_t ldb, const CopyTensorArg cpy_arg) {
  if (lda == N && ldb == N) {
    GE_ASSERT_RT_OK(rtMemcpy(cpy_arg.dst, N * cpy_arg.type_size, cpy_arg.src,
        N * cpy_arg.type_size, RT_MEMCPY_DEVICE_TO_DEVICE));
    return ge::GRAPH_SUCCESS;
  }

  GELOGD("M:%ld, N:%ld, lda:%d, ldb:%ld, type_size:%d", M, N, lda, ldb, cpy_arg.type_size);
  for (size_t i = 0; i < static_cast<size_t>(M); i++) {
    GE_ASSERT_RT_OK(rtMemcpy(static_cast<uint8_t*>(cpy_arg.dst) + ldb * i * cpy_arg.type_size,
                             N * cpy_arg.type_size,
                             static_cast<uint8_t*>(cpy_arg.src) + lda * i * cpy_arg.type_size,
                             N * cpy_arg.type_size, RT_MEMCPY_DEVICE_TO_DEVICE));
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus SplitToSequenceComputeKernel(gert::KernelContext *context) {
  auto input_storage_shape = context->GetInputPointer<gert::StorageShape>(
      static_cast<size_t>(SplitToSequenceArgIndex::KInputXShape));
  GE_ASSERT_NOTNULL(input_storage_shape);
  auto input_addr = context->GetInputPointer<gert::GertTensorData>(
      static_cast<size_t>(SplitToSequenceArgIndex::KInputXAddr));
  GE_ASSERT_NOTNULL(input_addr);
  auto extend_ctx = reinterpret_cast<gert::ExtendedKernelContext *>(context);
  auto input_x_desc = extend_ctx->GetInputDesc(0);
  GE_ASSERT_NOTNULL(input_x_desc);
  ge::DataType input_data_type = input_x_desc->GetDataType();
  GE_ASSERT_TRUE(kSupportedDataType.count(input_data_type) != 0,
      "SplitToSequence kernel data type [%d] not support.", input_data_type);
  int32_t type_size = GetSizeByDataType(input_data_type);
  auto axis = context->GetInputValue<int32_t>(static_cast<size_t>(SplitToSequenceArgIndex::KAxis));
  auto split_shapes = context->GetInputPointer<gert::TypedContinuousVector<gert::Shape>>(
      static_cast<size_t>(SplitToSequenceArgIndex::KSplitShapes));
  GE_ASSERT_NOTNULL(split_shapes);
  gert::Shape input_x_shape = input_storage_shape->GetStorageShape();
  auto input_x_rank = static_cast<int32_t>(input_x_shape.GetDimNum());
  if (axis < 0) {
    axis += input_x_rank;
  }

  auto before_dims = SizeToSplitAxis(input_x_shape, axis);
  auto after_dims_include_axis = SizeFromSplitAxis(input_x_shape, axis);
  auto after_dims_exclude_axis = ((axis + 1) == input_x_rank) ? 1 : SizeFromSplitAxis(input_x_shape, axis + 1);
  auto inner_tensor_addrs = context->MutableInputPointer<gert::TypedContinuousVector<
        gert::TensorData*>>(static_cast<size_t>(SplitToSequenceArgIndex::KInnerSplitAddrs));
  // tensor vec after split
  auto split_num = split_shapes->GetSize();
  auto split_shape_data = split_shapes->GetData();
  int64_t input_offset = 0;
  CopyTensorArg cpy_arg;
  for (size_t i = 0; i < split_num; i++) {
    gert::Shape each_shape_data = split_shape_data[i];
    auto split_size = each_shape_data.GetDim(axis);
    cpy_arg.src = reinterpret_cast<uint8_t*>(input_addr->GetAddr()) + input_offset;
    GE_ASSERT_NOTNULL(inner_tensor_addrs->MutableData()[i]);
    cpy_arg.dst = inner_tensor_addrs->MutableData()[i]->GetAddr();
    cpy_arg.type_size = type_size;
    GE_ASSERT_GRAPH_SUCCESS(CopyTensor(before_dims, split_size * after_dims_exclude_axis,
        after_dims_include_axis, split_size * after_dims_exclude_axis,
        cpy_arg), "copy tensor err.");
    input_offset += split_size * after_dims_exclude_axis * type_size;
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(SplitToSequenceDoCompute).RunFunc(SplitToSequenceComputeKernel);

ge::graphStatus BuildInferShapeFromTensorDoCompute(gert::KernelContext *context) {
  GELOGD("Enter BuildInferShapeFromTensorDoCompute");
  auto input_tensor = context->GetInputPointer<gert::Tensor>(0);
  GE_ASSERT_NOTNULL(input_tensor);

  auto storage_shape = context->GetOutputPointer<StorageShape>(0);
  GE_ASSERT_NOTNULL(storage_shape);
  storage_shape->MutableStorageShape() = input_tensor->GetStorageShape();
  storage_shape->MutableOriginShape() = input_tensor->GetStorageShape();
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus BuildSingleInferShapeOutputs(const ge::FastNode *node, KernelContext *context) {
  GELOGD("Enter BuildSingleInferShapeOutputs");
  (void)node;
  auto output0 = context->GetOutput(0);
  GE_ASSERT_NOTNULL(output0);
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  auto output_desc = extend_context->GetOutputDesc(0UL);
  GE_ASSERT_NOTNULL(output_desc);
  auto shape_tensor = new (std::nothrow) Tensor(StorageShape(),
      output_desc->GetFormat(), output_desc->GetDataType());
  GE_ASSERT_NOTNULL(shape_tensor);
  output0->SetWithDefaultDeleter(shape_tensor);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(BuildInferShapeFromTensorCompute)
    .RunFunc(BuildInferShapeFromTensorDoCompute)
    .OutputsCreator(BuildSingleInferShapeOutputs);

ge::graphStatus BuildAddrFromTensorDoCompute(gert::KernelContext *context) {
  GELOGD("Enter BuildAddrFromTensorDoCompute");
  auto input_tensor =
      context->GetInputPointer<gert::Tensor>(static_cast<size_t>(BuildAddrFromTensorComputeInputs::kTensor));
  auto gert_allocator =
      context->GetInputValue<GertAllocator *>(static_cast<size_t>(BuildAddrFromTensorComputeInputs::kAllocator));
  GE_ASSERT_NOTNULL(input_tensor);

  auto gtd = context->GetOutputPointer<GertTensorData>(0);
  GE_ASSERT_NOTNULL(gtd);
  return TensorUtils::ShareTdToGtd(input_tensor->GetTensorData(), *gert_allocator, *gtd);
}

ge::graphStatus AllocSingleOutputTensorData(const ge::FastNode *node, KernelContext *context) {
  GELOGD("Enter AllocSingleOutputTensorData");
  (void)node;
  auto output0 = context->GetOutput(0);
  GE_ASSERT_NOTNULL(output0);
  auto tensor_data = new (std::nothrow) GertTensorData();
  GE_ASSERT_NOTNULL(tensor_data);
  output0->SetWithDefaultDeleter(tensor_data);
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(BuildAddrFromTensorCompute)
    .RunFunc(BuildAddrFromTensorDoCompute)
    .OutputsCreator(AllocSingleOutputTensorData)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus GetSequenceHandleShapeKernel(KernelContext *context) {
  auto output_storage_shape = context->GetOutputPointer<gert::StorageShape>(0);
  GE_ASSERT_NOTNULL(output_storage_shape);

  gert::Shape scalar_shape;
  scalar_shape.SetScalar();
  output_storage_shape->MutableStorageShape() = scalar_shape;
  output_storage_shape->MutableOriginShape() = scalar_shape;
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(GetSequenceHandleShape)
    .RunFunc(GetSequenceHandleShapeKernel)
    .OutputsCreator(BuildSingleInferShapeOutputs);
}  // namespace kernel
}  // namespace gert
