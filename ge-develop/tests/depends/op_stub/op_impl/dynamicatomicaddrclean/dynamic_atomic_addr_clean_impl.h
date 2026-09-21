/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_ATOMIC_ADDR_CLEAN_IMPL_H_
#define AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_ATOMIC_ADDR_CLEAN_IMPL_H_
#include <cstdint>
#include <vector>
#include "register/op_compile_info_base.h"

namespace gert {
struct DynamicAtomicAddrCleanParam {
  // common input scalar
  int32_t select_key_input_scalar;
  int32_t need_core_num_input_scalar;
  int32_t ele_num_full_mask_full_repeat_time_input_scalar;
  int32_t burst_len_full_mask_full_repeat_time_input_scalar;

  // init input scalar
  // front core
  int32_t ele_num_front_core_input_scalar;
  int32_t init_times_full_mask_full_repeat_time_front_core_input_scalar;
  int32_t ele_num_front_part_front_core_input_scalar;
  int32_t burst_len_last_part_front_core_input_scalar;
  int32_t repeat_time_last_part_front_core_input_scalar;
  // last core
  int32_t ele_num_last_core_input_scalar;
  int32_t init_times_full_mask_full_repeat_time_last_core_input_scalar;
  int32_t ele_num_front_part_last_core_input_scalar;
  int32_t burst_len_last_part_last_core_input_scalar;
  int32_t repeat_time_last_part_last_core_input_scalar;
};

struct DynamicAtomicAddrCleanCompileInfo : public optiling::CompileInfoBase {
  uint32_t workspace_num = 0;
  uint32_t core_num = 0;
  uint32_t ub_size = 0;
  std::vector<int64_t> _workspace_index_list;
};
}  // namespace gert
#endif  // AIR_CXX_RUNTIME_V2_OP_IMPL_DYNAMIC_ATOMIC_ADDR_CLEAN_IMPL_H_
