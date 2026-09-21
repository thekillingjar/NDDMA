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
#include "graph/ge_error_codes.h"
#include "register/kernel_registry.h"
#include "kernel/tensor_attr.h"
#include "common/checker.h"
#include "core/executor/multi_thread_topological/executor/schedule/producer/producers/kernel_tags/critical_section_config.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/extended_kernel_context.h"

namespace gert {
namespace kernel {
ge::graphStatus IdentityShape(KernelContext *context) {
  for (size_t i = 0U; i < context->GetOutputNum(); i++) {
    auto in_shape = context->GetInputPointer<StorageShape>(i);
    auto out_shape = context->GetOutputPointer<StorageShape>(i);
    GE_ASSERT_NOTNULL(in_shape);
    GE_ASSERT_NOTNULL(out_shape);
    *out_shape = *in_shape;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateIdentityShapeOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  for (size_t i = 0U; i < context->GetOutputNum(); i++) {
    auto chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(chain);
    // 该kernel是为while子图的NetOutput算子插入identity，需要使用输入desc，netOutput没有输出
    auto input_desc = extend_context->GetInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc);
    auto shape_tensor = new (std::nothrow) Tensor(StorageShape(),
        input_desc->GetFormat(), input_desc->GetDataType());
    GE_ASSERT_NOTNULL(shape_tensor);
    chain->SetWithDefaultDeleter(shape_tensor);
  }

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(IdentityShape).RunFunc(IdentityShape).OutputsCreator(CreateIdentityShapeOutputs);

ge::graphStatus IdentityAddr(KernelContext *context) {
  for (size_t i = 0U; i < context->GetOutputNum(); i++) {
    auto in_tensor_data = context->GetInputPointer<GertTensorData>(i);
    auto out_tensor_data = context->GetOutputPointer<GertTensorData>(i);
    GE_ASSERT_NOTNULL(in_tensor_data);
    GE_ASSERT_NOTNULL(out_tensor_data);
    out_tensor_data->ShareFrom(*in_tensor_data);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus IdentityShapeAndAddr(KernelContext *context) {
  size_t num_shapes = context->GetOutputNum() >> 1U;
  for (size_t i = 0U; i < num_shapes; i++) {
    auto in_shape = context->GetInputPointer<StorageShape>(i);
    auto out_shape = context->GetOutputPointer<StorageShape>(i);
    GE_ASSERT_NOTNULL(in_shape);
    GE_ASSERT_NOTNULL(out_shape);
    *out_shape = *in_shape;
  }
  for (size_t i = num_shapes; i < context->GetOutputNum(); i++) {
    auto in_tensor_data = context->GetInputPointer<GertTensorData>(i);
    auto out_tensor_data = context->GetOutputPointer<GertTensorData>(i);
    GE_ASSERT_NOTNULL(in_tensor_data);
    GE_ASSERT_NOTNULL(out_tensor_data);
    out_tensor_data->ShareFrom(*in_tensor_data);
  }

  return ge::GRAPH_SUCCESS;
}

ge::graphStatus CreateIdentityShapeAndAddrOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  size_t num_shapes = context->GetOutputNum() >> 1U;
  auto extend_context = reinterpret_cast<ExtendedKernelContext *>(context);
  GE_ASSERT_NOTNULL(extend_context);
  for (size_t i = 0U; i < num_shapes; i++) {
    auto chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(chain);
    // 该kernel是为while子图的NetOutput算子插入identity，需要使用输入desc，netOutput没有输出
    auto input_desc = extend_context->GetInputDesc(i);
    GE_ASSERT_NOTNULL(input_desc);
    auto shape_tensor = new (std::nothrow) Tensor(StorageShape(),
        input_desc->GetFormat(), input_desc->GetDataType());
    GE_ASSERT_NOTNULL(shape_tensor);
    chain->SetWithDefaultDeleter(shape_tensor);
  }
  for (size_t i = num_shapes; i < context->GetOutputNum(); i++) {
    auto chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(chain);
    auto td = new (std::nothrow) GertTensorData();
    GE_ASSERT_NOTNULL(td);
    chain->SetWithDefaultDeleter(td);
  }

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(IdentityShapeAndAddr)
    .RunFunc(IdentityShapeAndAddr)
    .OutputsCreator(CreateIdentityShapeAndAddrOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus CreateIdentityAddrOutputs(const ge::FastNode *node, KernelContext *context) {
  (void)node;
  for (size_t i = 0U; i < context->GetOutputNum(); i++) {
    auto chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(chain);

    auto td = new (std::nothrow) GertTensorData();
    GE_ASSERT_NOTNULL(td);
    chain->SetWithDefaultDeleter(td);
  }

  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(IdentityAddr)
    .RunFunc(IdentityAddr)
    .OutputsCreator(CreateIdentityAddrOutputs)
    .ConcurrentCriticalSectionKey(kKernelUseMemory);

ge::graphStatus PointFromInputs(KernelContext *context) {
  for (size_t i = 0U; i < context->GetOutputNum(); i++) {
    auto in_chain = context->GetInput(i);
    auto out_chain = context->GetOutput(i);
    GE_ASSERT_NOTNULL(in_chain);
    GE_ASSERT_NOTNULL(out_chain);
    out_chain->Set(in_chain->GetValue<void *>(), nullptr);
  }
  return ge::GRAPH_SUCCESS;
}
REGISTER_KERNEL(PointFromInputs).RunFunc(PointFromInputs);
}  // namespace kernel
}  // namespace gert
