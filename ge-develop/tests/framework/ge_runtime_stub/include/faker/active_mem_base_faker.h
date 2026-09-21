/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_ACTIVE_MEM_BASE_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_ACTIVE_MEM_BASE_FAKER_H_
#include <vector>
#include "davinci_model_faker.h"
namespace ge {
class ActiveMemBaseFaker {
 public:
  ActiveMemBaseFaker(size_t model_fusion_num, size_t model_fm_num, size_t model_input_num, size_t model_output_num);
  ActiveMemBaseFaker(size_t model_fm_num, size_t model_input_num, size_t model_output_num);
  ActiveMemBaseFaker(size_t model_input_num, size_t model_output_num);
  ActiveMemBaseFaker(std::vector<int64_t> model_input_sizes, std::vector<int64_t> model_output_sizes);
  ActiveMemBaseFaker(std::vector<int64_t> model_fm_sizes, std::vector<int64_t> model_input_sizes,
                     std::vector<int64_t> model_output_sizes);
  ActiveMemBaseFaker(std::vector<int64_t> model_fusion_sizes, std::vector<int64_t> model_fm_sizes,
                     std::vector<int64_t> model_input_sizes, std::vector<int64_t> model_output_sizes);
  ActiveMemBaseFaker &FusionBaseIndex(uint64_t index);
  ActiveMemBaseFaker &FmBaseIndex(uint64_t index);
  ActiveMemBaseFaker &ModelIoBaseIndex(uint64_t index);

  std::vector<uint64_t> Build() const;

 private:
  std::vector<int64_t> model_fusion_sizes_;
  std::vector<int64_t> model_fm_sizes_{0x100};
  std::vector<int64_t> model_input_sizes_;
  std::vector<int64_t> model_output_sizes_;
  uint64_t fusion_base_index_{0};
  uint64_t fm_base_index_{0};
  uint64_t model_io_base_index_{1};
};
}  // namespace ge

#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_ACTIVE_MEM_BASE_FAKER_H_
