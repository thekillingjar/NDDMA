/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bg_tensor.h"
#include "common/checker.h"
#include "bg_memory.h"
#include "storage_format.h"
#include "kernel/tensor_attr.h"
namespace gert {
namespace bg {
ValueHolderPtr CreateBuildTensorAttr(const ge::NodePtr &node, int32_t index, TensorPlacement placement) {
  const auto &tensor_desc = node->GetOpDescBarePtr()->GetOutputDescPtr(index);
  GE_ASSERT_NOTNULL(tensor_desc, "Failed to get output tensor desc from node %s(%s), index %d", node->GetNamePtr(),
                    node->GetTypePtr(), index);

  kernel::BuildTensorAttr attr{};
  attr.placement = placement;
  attr.data_type = tensor_desc->GetDataType();
  attr.storage_format.SetStorageFormat(tensor_desc->GetFormat());
  attr.storage_format.SetOriginFormat(tensor_desc->GetOriginFormat());
  // todo padding

  return ValueHolder::CreateConst(&attr, sizeof(attr));
}
ValueHolderPtr BuildTensorWithoutGuarder(const ge::NodePtr &node, int32_t index, TensorPlacement placement,
                                         const ValueHolderPtr &storage_shape, const ValueHolderPtr &addr) {
  auto tensor_attr = CreateBuildTensorAttr(node, index, placement);
  return DevMemValueHolder::CreateSingleDataOutput("BuildTensor", {storage_shape, addr, tensor_attr},
                                                   node->GetOpDescBarePtr()->GetStreamId());
}
ValueHolderPtr BuildTensor(const ge::NodePtr &node, int32_t index, TensorPlacement placement,
                           const ValueHolderPtr &storage_shape, const ValueHolderPtr &addr) {
  auto tensor = BuildTensorWithoutGuarder(node, index, placement, storage_shape, addr);
  GE_ASSERT_NOTNULL(ValueHolder::CreateVoidGuarder("FreeTensorMemory", tensor, {}));
  return tensor;
}

ValueHolderPtr BuildRefTensor(const ge::NodePtr &node, int32_t index, TensorPlacement placement,
                                         const ValueHolderPtr &storage_shape, const ValueHolderPtr &addr) {
  DevMemValueHolderPtr tensor = nullptr;
  auto tensor_attr = CreateBuildTensorAttr(node, index, placement);
  tensor = DevMemValueHolder::CreateSingleDataOutput("BuildRefTensor", {storage_shape, addr, tensor_attr},
                                                     node->GetOpDescBarePtr()->GetStreamId());
  return tensor;
}

ValueHolderPtr CopyD2H(const ge::NodePtr &node, int32_t out_index, const ValueHolderPtr &storage_shape,
                       const ValueHolderPtr &addr) {
  if (node->GetAllOutDataAnchorsSize() <= static_cast<size_t>(out_index)) {
    return nullptr;
  }
  const auto &tensor_desc = node->GetOpDescBarePtr()->GetOutputDescPtr(out_index);
  if (tensor_desc == nullptr) {
    return nullptr;
  }

  auto tensor_size = CalcOutTensorSize(node, out_index, storage_shape);
  auto host_addr = ValueHolder::CreateSingleDataOutput("CopyD2H", {addr, tensor_size});
  GE_ASSERT_NOTNULL(ValueHolder::CreateVoidGuarder("FreeMemory", host_addr, {}));
  return BuildTensor(node, out_index, kOnHost, storage_shape, host_addr);
}
}  // namespace bg
}  // namespace gert
