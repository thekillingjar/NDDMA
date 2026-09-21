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
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/runtime/tensor.h"
#include "register/kernel_registry_impl.h"
#include "tensor_sequence.h"
#include "resource_mgr.h"
#include "framework/common/debug/ge_log.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/extended_kernel_context.h"

using namespace ge;
namespace gert {
namespace {
enum class SplitArgIndex {
  kSessionId,
  kContainerId,
  kInnerTensorDatas,
  kInnerShapes,
  kInputNum,
  kKeepDims,
  kAxis,
  kInputXShape,
  kOutputTensor
};
} // namespace

Shape UpdateShape(const Shape old_shape, int32_t axis) {
  Shape each_shape_data;
  auto dim_num = old_shape.GetDimNum();
  auto k = 0;
  each_shape_data.SetDimNum(dim_num - 1);
  for (size_t n = 0; n < dim_num; n++) {
    if (n == static_cast<size_t>(axis)) {
      GELOGD("skip dim val %lld", old_shape.GetDim(n));
      continue;
    }
    each_shape_data.SetDim(k++, old_shape.GetDim(n));
  }
  return each_shape_data;
}

ge::graphStatus StoreTensorInfoToSequence(KernelContext *context) {
  GELOGD("StoreTensorInfoToSequence begin");
  auto inner_tensor_shape = context->GetInputPointer<TypedContinuousVector<Shape>>
        (static_cast<size_t>(SplitArgIndex::kInnerShapes));
  if (inner_tensor_shape == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get inner shape failed");
    REPORT_INNER_ERR_MSG("E39999", "Get inner shape failed");
    return ge::PARAM_INVALID;
  }

  auto input_num = context->GetInputValue<int32_t>(static_cast<size_t>(SplitArgIndex::kInputNum));
  auto keep_dims = context->GetInputValue<int32_t>(static_cast<size_t>(SplitArgIndex::kKeepDims));
  auto input_x_storage_shape = context->GetInputPointer<StorageShape>(
      static_cast<size_t>(SplitArgIndex::kInputXShape));
  if (input_x_storage_shape == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get tensor shape failed");
    REPORT_INNER_ERR_MSG("E39999", "Get tensor shape failed");
    return ge::PARAM_INVALID;
  }

  auto axis = context->GetInputValue<int32_t>(static_cast<size_t>(SplitArgIndex::kAxis));
  Shape input_x_shape = input_x_storage_shape->GetStorageShape();
  auto input_tensor_rank = static_cast<int32_t>(input_x_shape.GetDimNum());
  GELOGD("input_tensor_rank:%d, axis=%d", input_tensor_rank, axis);
  if (axis < 0) {
    axis += input_tensor_rank;
  }
  // tensor vec after split
  auto shape_num = inner_tensor_shape->GetSize();
  auto shape_data = inner_tensor_shape->GetData();
  auto inner_tensor_addrs = context->MutableInputPointer<TypedContinuousVector<TensorData*>>(
        static_cast<size_t>(SplitArgIndex::kInnerTensorDatas));
  if (inner_tensor_addrs == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get tensor data addrs failed");
    REPORT_INNER_ERR_MSG("E39999", "Get tensor data addrs failed");
    return ge::PARAM_INVALID;
  }

  auto extend_ctx = reinterpret_cast<ExtendedKernelContext*>(context);
  auto input_x_desc = extend_ctx->GetInputDesc(0);
  if (input_x_desc == nullptr) {
    GELOGE(ge::PARAM_INVALID, "Get input x desc failed");
    REPORT_INNER_ERR_MSG("E39999", "Get input x desc failed");
    return ge::PARAM_INVALID;
  }

  ge::DataType data_type = input_x_desc->GetDataType();
  TensorSeqPtr tensor_seq = std::make_shared<TensorSeq>(data_type);
  if (tensor_seq == nullptr) {
    GELOGE(ge::PARAM_INVALID, "tensor_seq is a null pointer");
    REPORT_INNER_ERR_MSG("E39999", "tensor_seq is a null pointer");
    return ge::PARAM_INVALID;
  }

  auto session_id = context->GetInputValue<size_t>(static_cast<size_t>(SplitArgIndex::kSessionId));
  auto container_id = context->GetInputValue<size_t>(static_cast<size_t>(SplitArgIndex::kContainerId));
  ResourceMgrPtr rm;
  SessionMgr::GetInstance()->GetRm(session_id, container_id, rm);
  uint64_t handle = rm->GetHandle();
  GELOGD("handle=%llu, shape_num=%u, session_id=%u, container_id=%u",
         handle, shape_num, session_id, container_id);
  rm->StoreStepHandle(handle);
  rm->Create(handle, tensor_seq);
  StorageShape store_shape;
  for (size_t i = 0; i < shape_num; i++) {
    // split is not given, step is 1, drop it
    if ((keep_dims == 0) && (input_num == 1)) {
      store_shape.MutableOriginShape() = UpdateShape(shape_data[i], axis);
      store_shape.MutableStorageShape() = UpdateShape(shape_data[i], axis);
      if (inner_tensor_addrs->MutableData()[i] == nullptr) {
        GELOGE(ge::INTERNAL_ERROR, "inner addr is a null pointer");
        REPORT_INNER_ERR_MSG("E39999", "inner addr is a null pointer");
        return ge::INTERNAL_ERROR;
      }  
      tensor_seq->Add(data_type, *inner_tensor_addrs->MutableData()[i], store_shape);
      continue;
    }
    store_shape.MutableOriginShape() = shape_data[i];
    store_shape.MutableStorageShape() = shape_data[i];
    for (size_t j = 0; j < shape_data[i].GetDimNum(); j++) {
      GELOGD("dim = %lld", shape_data[i].GetDim(j));
    }
    if (inner_tensor_addrs->MutableData()[i] == nullptr) {
      GELOGE(ge::INTERNAL_ERROR, "inner addr is a null pointer");
      REPORT_INNER_ERR_MSG("E39999", "inner addr is a null pointer");
      return ge::INTERNAL_ERROR;
    }  
    tensor_seq->Add(data_type, *inner_tensor_addrs->MutableData()[i], store_shape);
  }

  auto output_tensor = context->MutableInputPointer<Tensor>(
      static_cast<size_t>(SplitArgIndex::kOutputTensor));
  if ((output_tensor == nullptr) || (output_tensor->GetData<uint64_t>() == nullptr)) {
    GELOGE(ge::PARAM_INVALID, "rm is a null pointer");
    REPORT_INNER_ERR_MSG("E39999", "rm is a null pointer");
    return ge::PARAM_INVALID;
  }
  uint64_t* data_ptr = output_tensor->GetData<uint64_t>();
  *data_ptr = handle;
  GELOGD("StoreTensorInfoToSequence end");
  return ge::GRAPH_SUCCESS;
}

REGISTER_KERNEL(StoreSplitTensorToSequence).RunFunc(StoreTensorInfoToSequence);
}  // namespace gert