/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

//
// Created by cyrus on 2/6/25.
//

#ifndef AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_TILING_FWK_DATA_HELPER_H
#define AIR_CXX_TESTS_UT_GE_RUNTIME_V2_COMMON_TILING_FWK_DATA_HELPER_H
#include "engine/aicore/launch_kernel/rt_kernel_launch_args_ex.h"

namespace gert {
inline void CreateDefaultArgsInfo(ArgsInfosDesc::ArgsInfo *args_info, size_t input_num, size_t output_num) {
  auto node_io_num = input_num + output_num;
  for (size_t idx = 0U; idx < node_io_num; ++idx) {
    int32_t start_index = (idx < input_num) ? idx : (idx - input_num);
    auto arg_type = (idx < input_num) ? ArgsInfosDesc::ArgsInfo::ArgsInfoType::INPUT
                                      : ArgsInfosDesc::ArgsInfo::ArgsInfoType::OUTPUT;
    args_info[idx].Init(arg_type, ArgsInfosDesc::ArgsInfo::ArgsInfoFormat::DIRECT_ADDR, start_index, 1U);
  }
}
inline std::unique_ptr<uint8_t[]> CreateDefaultArgsInfoDesc(size_t input_num, size_t output_num) {
  const size_t args_info_num = input_num + output_num;
  ArgsInfosDesc::ArgsInfo args_info[args_info_num];
  CreateDefaultArgsInfo(args_info, input_num, output_num);
  size_t total_size = 0U;
  const size_t args_info_size = args_info_num * sizeof(ArgsInfosDesc::ArgsInfo);
  auto args_info_desc_holder = ArgsInfosDesc::Create(args_info_size, total_size);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  if (args_info_size > 0) {
    GELOGD("Copy args info to compute node extended desc mem, size:%lld", args_info_size);
    if (memcpy_s(args_info_desc->MutableArgsInfoBase(), args_info_desc->GetArgsInfoSize(), args_info, args_info_size) !=
        EOK) {
      return nullptr;
        }
  }
  args_info_desc->Init(input_num, output_num, input_num, output_num);
  return args_info_desc_holder;
}

inline std::unique_ptr<uint8_t[]> CreateLaunchArg(size_t input_num = 1, size_t output_num = 1) {
  size_t compiled_args_size = 64;
  std::unique_ptr<uint8_t[]> node_desc_holder(
      new (std::nothrow) uint8_t[compiled_args_size + sizeof(RtKernelLaunchArgsEx::ComputeNodeDesc)]);
  auto node_desc = reinterpret_cast<RtKernelLaunchArgsEx::ComputeNodeDesc *>(node_desc_holder.get());
  *node_desc = {.input_num = input_num,
                .output_num = output_num,
                .workspace_cap = 8,
                .max_tiling_data = 1024,
                .need_shape_buffer = false,
                .need_overflow = false,
                .compiled_args_size = compiled_args_size};
  auto args_info_desc_holder = CreateDefaultArgsInfoDesc(node_desc->input_num, node_desc->output_num);
  auto args_info_desc = reinterpret_cast<ArgsInfosDesc *>(args_info_desc_holder.get());
  auto launch_arg_holder = RtKernelLaunchArgsEx::Create(*node_desc, *args_info_desc);
  return launch_arg_holder;
}
}

#endif //TILING_FWK_DATA_HELPER_H
