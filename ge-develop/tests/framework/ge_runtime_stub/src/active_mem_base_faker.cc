/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "faker/active_mem_base_faker.h"
namespace ge {
ActiveMemBaseFaker::ActiveMemBaseFaker(std::vector<int64_t> model_input_sizes, std::vector<int64_t> model_output_sizes)
    : model_input_sizes_(std::move(model_input_sizes)), model_output_sizes_(std::move(model_output_sizes)) {}

ActiveMemBaseFaker::ActiveMemBaseFaker(size_t model_input_num, size_t model_output_num)
    : ActiveMemBaseFaker(std::vector<int64_t>(model_input_num, 0x100), std::vector<int64_t>(model_output_num, 0x100)) {}

ActiveMemBaseFaker::ActiveMemBaseFaker(std::vector<int64_t> model_fm_sizes, std::vector<int64_t> model_input_sizes,
                                       std::vector<int64_t> model_output_sizes)
    : model_fm_sizes_(std::move(model_fm_sizes)),
      model_input_sizes_(std::move(model_input_sizes)),
      model_output_sizes_(std::move(model_output_sizes)) {}

ActiveMemBaseFaker::ActiveMemBaseFaker(size_t model_fm_num, size_t model_input_num, size_t model_output_num)
    : ActiveMemBaseFaker(std::vector<int64_t>(model_fm_num, 0x100), std::vector<int64_t>(model_input_num, 0x100),
                         std::vector<int64_t>(model_output_num, 0x100)) {}

ActiveMemBaseFaker::ActiveMemBaseFaker(std::vector<int64_t> model_fusion_sizes, std::vector<int64_t> model_fm_sizes,
                                       std::vector<int64_t> model_input_sizes, std::vector<int64_t> model_output_sizes)
    : model_fusion_sizes_(std::move(model_fusion_sizes)),
      model_fm_sizes_(std::move(model_fm_sizes)),
      model_input_sizes_(std::move(model_input_sizes)),
      model_output_sizes_(std::move(model_output_sizes)) {}

ActiveMemBaseFaker::ActiveMemBaseFaker(size_t model_fusion_num, size_t model_fm_num,
                                       size_t model_input_num, size_t model_output_num)
    : ActiveMemBaseFaker(std::vector<int64_t>(model_fusion_num, 0x100), std::vector<int64_t>(model_fm_num, 0x100),
                         std::vector<int64_t>(model_input_num, 0x100), std::vector<int64_t>(model_output_num, 0x100)) {}

ActiveMemBaseFaker &ActiveMemBaseFaker::FusionBaseIndex(uint64_t index) {
  fusion_base_index_ = index;
  return *this;
}

ActiveMemBaseFaker &ActiveMemBaseFaker::FmBaseIndex(uint64_t index) {
  fm_base_index_ = index;
  return *this;
}
ActiveMemBaseFaker &ActiveMemBaseFaker::ModelIoBaseIndex(uint64_t index) {
  model_io_base_index_ = index;
  return *this;
}

std::vector<uint64_t> ActiveMemBaseFaker::Build() const {
  std::vector<uint64_t> allocation_ids_to_base;
  allocation_ids_to_base.reserve(model_fm_sizes_.size() + model_input_sizes_.size() + model_output_sizes_.size() + 1UL);

  // fusion 的device地址和 fm段是一样的
  auto model_fusion_addr = DavinciModelFaker::GetFmDevBase(fusion_base_index_);
  for (const auto fusion_size : model_fusion_sizes_) {
      allocation_ids_to_base.emplace_back(model_fusion_addr);
      model_fusion_addr += fusion_size;
  }

  auto model_fm_addr = DavinciModelFaker::GetFmDevBase(fm_base_index_);
  for (const auto fm_size : model_fm_sizes_) {
    allocation_ids_to_base.emplace_back(model_fm_addr);
    model_fm_addr += fm_size;
  }

  auto model_io_addr = DavinciModelFaker::GetModelIoDevBase(model_io_base_index_);
  for (const auto input_size : model_input_sizes_) {
    allocation_ids_to_base.emplace_back(model_io_addr);
    model_io_addr += input_size;
  }
  for (const auto output_size : model_output_sizes_) {
    allocation_ids_to_base.emplace_back(model_io_addr);
    model_io_addr += output_size;
  }

  allocation_ids_to_base.emplace_back(0);

  return allocation_ids_to_base;
}
}  // namespace ge
