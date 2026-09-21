/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_NODES_FAKER_FAKE_SHAPE_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_NODES_FAKER_FAKE_SHAPE_H_
#include "faker/base_node_exe_faker.h"
#include "faker/fake_kernel_definitions.h"
#include "kernel/memory/caching_mem_allocator.h"
#include "kernel/memory/host_mem_allocator.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "core/utils/tensor_utils.h"

namespace gert {
class FakedShapeImpl : public BaseNodeExeFaker {
 public:
  TensorPlacement GetOutPlacement() const override {
    return kOnHost;
  }
  ge::graphStatus RunFunc(KernelContext *context) override {
    auto gert_allocator =
        context->GetInputValue<GertAllocator *>(kFkiAllocator);
    GE_ASSERT_NOTNULL(gert_allocator);
    auto in_shape = context->GetInputPointer<StorageShape>(kFkiStart);
    auto out_shape = context->GetOutputPointer<StorageShape>(0);
    auto out_addr = context->GetOutputPointer<GertTensorData>(1);
    GE_ASSERT_NOTNULL(in_shape);
    GE_ASSERT_NOTNULL(out_shape);
    GE_ASSERT_NOTNULL(out_addr);

    auto dim_num = in_shape->GetStorageShape().GetDimNum();

    out_shape->MutableStorageShape().SetDimNum(1);
    out_shape->MutableStorageShape().SetDim(0, static_cast<int64_t>(dim_num));
    out_shape->MutableOriginShape().SetDimNum(1);
    out_shape->MutableOriginShape().SetDim(0, static_cast<int64_t>(dim_num));

    auto out_data_size = out_shape->GetStorageShape().GetShapeSize() * sizeof(int64_t);
    if (out_addr->GetAddr() == nullptr || out_addr->GetSize() < out_data_size) {
      auto out_data = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(out_data_size));
      GE_ASSERT_NOTNULL(out_data);
      *out_addr = TensorUtils::ToGertTensorData(
          out_data, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());
    }

    memcpy(out_addr->GetAddr(), &(in_shape->GetStorageShape()[0]), out_data_size);

    return ge::GRAPH_SUCCESS;
  }
};
class FakedRankImpl : public BaseNodeExeFaker {
 public:
  TensorPlacement GetOutPlacement() const override {
    return kOnHost;
  }
  ge::graphStatus RunFunc(KernelContext *context) override {
    auto gert_allocator =
        context->GetInputValue<GertAllocator *>(kFkiAllocator);
    GE_ASSERT_NOTNULL(gert_allocator);
    auto in_shape = context->GetInputPointer<StorageShape>(kFkiStart);
    auto out_shape = context->GetOutputPointer<StorageShape>(0);
    auto out_addr = context->GetOutputPointer<GertTensorData>(1);
    GE_ASSERT_NOTNULL(in_shape);
    GE_ASSERT_NOTNULL(out_shape);
    GE_ASSERT_NOTNULL(out_addr);

    out_shape->MutableStorageShape().SetDimNum(0);
    out_shape->MutableOriginShape().SetDimNum(0);

    auto out_data_size = out_shape->GetStorageShape().GetShapeSize() * sizeof(size_t);
    if (out_addr->GetAddr() == nullptr || out_addr->GetSize() < out_data_size) {
      auto out_data = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(out_data_size));
      GE_ASSERT_NOTNULL(out_data);
      std::cout << "Rank: Alloc output memory " << out_data
                << ", count " << out_data->GetCount(gert_allocator->GetStreamId()) << std::endl;
      *out_addr = TensorUtils::ToGertTensorData(
          out_data, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());
    }

    *static_cast<size_t *>(out_addr->GetAddr()) = in_shape->GetStorageShape().GetDimNum();

    return ge::GRAPH_SUCCESS;
  }
};
class FakedSizeImpl : public BaseNodeExeFaker {
 public:
  TensorPlacement GetOutPlacement() const override {
    return kOnHost;
  }
  ge::graphStatus RunFunc(KernelContext *context) override {
    auto gert_allocator =
        context->GetInputValue<GertAllocator *>(kFkiAllocator);
    GE_ASSERT_NOTNULL(gert_allocator);
    auto in_shape = context->GetInputPointer<StorageShape>(kFkiStart);
    auto out_shape = context->GetOutputPointer<StorageShape>(0);
    auto out_addr = context->GetOutputPointer<GertTensorData>(1);
    GE_ASSERT_NOTNULL(in_shape);
    GE_ASSERT_NOTNULL(out_shape);
    GE_ASSERT_NOTNULL(out_addr);

    out_shape->MutableStorageShape().SetDimNum(0);
    out_shape->MutableOriginShape().SetDimNum(0);

    auto out_data_size = out_shape->GetStorageShape().GetShapeSize() * sizeof(size_t);
    if (out_addr->GetAddr() == nullptr || out_addr->GetSize() < out_data_size) {
      auto out_data = reinterpret_cast<memory::MultiStreamMemBlock *>(gert_allocator->Malloc(out_data_size));
      GE_ASSERT_NOTNULL(out_data);
      std::cout << "Rank: Alloc output memory " << out_data << ", count " << out_data->GetCount(gert_allocator->GetStreamId()) << std::endl;
      *out_addr = TensorUtils::ToGertTensorData(
          out_data, gert_allocator->GetPlacement(), gert_allocator->GetStreamId());
    }

    *static_cast<int64_t *>(out_addr->GetAddr()) = in_shape->GetStorageShape().GetShapeSize();

    return ge::GRAPH_SUCCESS;
  }
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_SRC_NODES_FAKER_FAKE_SHAPE_H_
