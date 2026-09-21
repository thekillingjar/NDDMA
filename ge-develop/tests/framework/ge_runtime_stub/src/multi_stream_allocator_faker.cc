/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/multi_stream_allocator_faker.h"
#include "faker/fake_allocator.h"
namespace gert {
namespace memory {
MultiStreamAllocatorFaker &MultiStreamAllocatorFaker::StreamNum(int64_t num) {
  stream_num_ = num;
  return *this;
}
MultiStreamAllocatorFaker &MultiStreamAllocatorFaker::Placement(TensorPlacement placement) {
  placement_ = placement;
  return *this;
}
MultiStreamAllocatorFaker::Holder MultiStreamAllocatorFaker::Build() const {
  MultiStreamAllocatorFaker::Holder holder;
  if (l1_allocator_ == nullptr) {
    holder.l1_allocator = std::make_shared<ge::FakeAllocator>();
  } else {
    holder.l1_allocator = l1_allocator_;
  }

  holder.l2_allocators_holder = ContinuousVector::Create<memory::MultiStreamL2Allocator *>(stream_num_);
  auto l2_allocators = reinterpret_cast<TypedContinuousVector<memory::MultiStreamL2Allocator *> *>(holder.l2_allocators_holder.get());
  l2_allocators->SetSize(static_cast<size_t>(stream_num_));
  auto l2_allocators_vec = l2_allocators->MutableData();

  holder.all_l2_mem_pool_holder = ContinuousVector::Create<memory::L2MemPool *>(stream_num_);
  auto all_l2_mem_pool = reinterpret_cast<TypedContinuousVector<memory::L2MemPool *> *>(holder.all_l2_mem_pool_holder.get());
  all_l2_mem_pool->SetSize(static_cast<size_t>(stream_num_));
  auto all_l2_mem_pool_vec = all_l2_mem_pool->MutableData();

  for (int64_t i = 0; i < stream_num_; ++i) {
    rtStream_t stream;
    rtStreamCreate(&stream, static_cast<int32_t>(RT_STREAM_PRIMARY_DEFAULT));
    auto allocator = std::make_unique<MultiStreamL2Allocator>(holder.l1_allocator.get(), placement_, i, stream,
                                                              l2_allocators, all_l2_mem_pool);
    l2_allocators_vec[i] = allocator.get();
    all_l2_mem_pool_vec[i] = allocator->GetL2MemPool();
    holder.stream_ids_to_allocator.emplace_back(std::move(allocator));
    holder.streams.emplace_back(stream);
  }
  return holder;
}
MultiStreamAllocatorFaker &MultiStreamAllocatorFaker::L1Allocator(std::shared_ptr<ge::Allocator> l1_allocator) {
  l1_allocator_ = std::move(l1_allocator);
  return *this;
}
MultiStreamL2Allocator *MultiStreamAllocatorFaker::Holder::Allocator(int64_t stream_id) {
  return stream_ids_to_allocator.at(stream_id).get();
}
GertTensorData MultiStreamAllocatorFaker::Holder::Alloc(int64_t stream_id, size_t size) {
  return {size, Allocator(stream_id)->GetPlacement(), stream_id, AllocBlock(stream_id, size).release()};
}
MultiStreamAllocatorFaker::Holder::~Holder() {
  for (auto stream : streams) {
    rtStreamDestroy(stream);
  }
  auto deleter = [](void *ptr) {
    delete [] static_cast<uint8_t *>(ptr);
  };
  deleter(l2_allocators_holder.release());
}
MultiStreamAllocatorFaker::MultiStreamMemBlockPtr MultiStreamAllocatorFaker::Holder::AllocBlock(int64_t stream_id,
                                                                                                size_t size) {
  auto block = reinterpret_cast<MultiStreamMemBlock *>(Allocator(stream_id)->Malloc(size));
  return {block, [this, stream_id](MultiStreamMemBlock *b) { Allocator(stream_id)->BirthRecycle(b); }};
}
}  // namespace memory
}  // namespace gert
