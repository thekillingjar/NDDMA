/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MULTI_STREAM_ALLOCATOR_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MULTI_STREAM_ALLOCATOR_FAKER_H_
#include "kernel/memory/multi_stream_l2_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"
#include "exe_graph/runtime/kernel_context.h"
#include "runtime/stream.h"
namespace gert {
namespace memory {
class MultiStreamAllocatorFaker {
 public:
  using MultiStreamMemBlockPtr = std::unique_ptr<MultiStreamMemBlock, std::function<void(MultiStreamMemBlock*)>>;
  struct Holder {
    MultiStreamL2Allocator *Allocator(int64_t stream_id);
    GertTensorData Alloc(int64_t stream_id, size_t size);
    MultiStreamMemBlockPtr AllocBlock(int64_t stream_id, size_t size);

    Holder() = default;
    ~Holder();
    Holder(Holder &&) = default;
    Holder &operator=(Holder &&) = default;

    MultiStreamL2Allocator *at(size_t index) {
      return stream_ids_to_allocator.at(index).get();
    }

    std::shared_ptr<ge::Allocator> l1_allocator;
    std::vector<std::unique_ptr<MultiStreamL2Allocator>> stream_ids_to_allocator;
    std::vector<rtStream_t> streams;
    std::unique_ptr<uint8_t[]> l2_allocators_holder;
    std::unique_ptr<uint8_t[]> all_l2_mem_pool_holder;
  };
  MultiStreamAllocatorFaker() = default;
  MultiStreamAllocatorFaker &StreamNum(int64_t num);
  MultiStreamAllocatorFaker &Placement(TensorPlacement placement);
  MultiStreamAllocatorFaker &L1Allocator(std::shared_ptr<ge::Allocator> l1_allocator);
  Holder Build() const;

 private:
  int64_t stream_num_{1};
  TensorPlacement placement_{kOnDeviceHbm};
  std::shared_ptr<ge::Allocator> l1_allocator_;
};
}  // namespace memory
}  // namespace gert

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MULTI_STREAM_ALLOCATOR_FAKER_H_
