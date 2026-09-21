/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MEM_ALLOCATION_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MEM_ALLOCATION_FAKER_H_
#include <vector>
#include "graph/load/model_manager/task_info/task_info.h"
#include "graph/load/model_manager/model_utils.h"
namespace ge {
class MemAllocationFaker {
 public:
  MemAllocationFaker &FeatureMap(uint64_t logical_addr, uint64_t len) {
    allocation_table_.push_back({
          static_cast<uint32_t>(allocation_table_.size()),
          logical_addr,
          len,
          MemAllocation::Type::FEATURE_MAP,
          0U,    // model io 特有，fm始终填0
          kFmMemType
      });
    return *this;
  }
  MemAllocationFaker &ModelInput(uint32_t input_index, uint64_t logical_addr, uint64_t len) {
    allocation_table_.push_back({
        static_cast<uint32_t>(allocation_table_.size()),
        logical_addr,
        len,
        MemAllocation::Type::INPUT,
        input_index,
        RT_MEMORY_HBM
    });
    return *this;
  }
  MemAllocationFaker &ModelOutput(uint32_t output_index, uint64_t logical_addr, uint64_t len) {
    allocation_table_.push_back({
        static_cast<uint32_t>(allocation_table_.size()),
        logical_addr,
        len,
        MemAllocation::Type::OUTPUT,
        output_index,
        RT_MEMORY_HBM
    });
    return *this;
  }

  MemAllocationFaker &Absolute(uint64_t logical_addr, uint64_t len) {
    allocation_table_.push_back({
        static_cast<uint32_t>(allocation_table_.size()),
        logical_addr,
        len,
        MemAllocation::Type::ABSOLUTE,
        0,
      kAbsoluteMemType
    });
    return *this;
  }
  std::vector<MemAllocation> Build() const {
    return allocation_table_;
  }
 private:
  std::vector<MemAllocation> allocation_table_;
};
}

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_MEM_ALLOCATION_FAKER_H_
