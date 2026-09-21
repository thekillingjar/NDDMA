/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_ALLOCATOR_FACKER_H_
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_FAKER_ALLOCATOR_FACKER_H_

#include <vector>
#include "ge/ge_allocator.h"
#include "exe_graph/runtime/gert_mem_block.h"
#include "exe_graph/runtime/gert_mem_allocator.h"
#include "exe_graph/runtime/gert_tensor_data.h"

namespace gert {
class GertMemBlockFaker : public GertMemBlock {
  public:
  explicit GertMemBlockFaker(void *addr) : addr_(addr) {}
  ~GertMemBlockFaker() override {
      addr_ = nullptr;
  };
  void Free(int64_t stream_id) override {
    free(addr_);
    addr_ = nullptr;
  }
  void *GetAddr() override { return addr_; }

  void AddCount() {
    use_count_++;
  }

  private:
  void *addr_;
  size_t use_count_=0u;
};
class GertTensorDataFaker : public GertTensorData {

};
class AllocatorFaker : public GertAllocator {
 public:
  AllocatorFaker() = default;
  ~AllocatorFaker() override;
  GertMemBlock *Malloc(size_t size) override;
  void Free(GertMemBlock *block) override;

  GertTensorData MallocTensorData(size_t size) override;

  TensorData MallocTensorDataFromL1(size_t size) override {
    return TensorData();
  }
  ge::graphStatus ShareFromTensorData(const TensorData &td, GertTensorData &gtd) override {
    return 0;
  }
  ge::graphStatus SetL1Allocator(ge::Allocator *allocator) override {
    return ge::GRAPH_SUCCESS;
  }
  ge::graphStatus FreeAt(int64_t stream_id, GertMemBlock *block) override {
    return ge::GRAPH_SUCCESS;
  }
  int64_t GetStreamNum() override {
    return 0;
  }
 private:
  std::vector<GertMemBlock *> block_pool_;
};
}
#endif
