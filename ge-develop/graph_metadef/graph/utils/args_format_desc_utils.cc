/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/utils/args_format_desc_utils.h"
#include <functional>
#include "common/checker.h"
#include "framework/common/debug/ge_log.h"
#include "graph/args_format_desc.h"

namespace ge {

void ArgsFormatDescUtils::Append(std::vector<ArgDesc> &arg_descs, ge::AddrType type, int32_t ir_idx, bool folded) {
  arg_descs.push_back({type, ir_idx, folded, {0}});
}

void ArgsFormatDescUtils::AppendTilingContext(std::vector<ArgDesc> &arg_descs, ge::TilingContextSubType sub_type) {
  arg_descs.push_back({AddrType::TILING_CONTEXT, static_cast<int32_t>(sub_type), false, {0}});
}

graphStatus ArgsFormatDescUtils::InsertHiddenInputs(std::vector<ArgDesc> &arg_descs, int32_t insert_pos,
                                                    HiddenInputsType hidden_type, size_t input_cnt) {
  if (insert_pos < 0) {
    insert_pos = arg_descs.size();
  }
  GE_ASSERT_TRUE(static_cast<size_t>(insert_pos) <= arg_descs.size());
  ArgDesc arg_desc = {AddrType::HIDDEN_INPUT, -1, false, {0}};
  for (size_t i = 0; i < input_cnt; ++i, ++insert_pos) {
    arg_desc.ir_idx = static_cast<int32_t>(i);
    *reinterpret_cast<uint32_t *>(arg_desc.reserved) = static_cast<uint32_t>(hidden_type);
    arg_descs.insert(arg_descs.begin() + insert_pos, arg_desc);
  }
  return GRAPH_SUCCESS;
}

graphStatus ArgsFormatDescUtils::InsertCustomValue(std::vector<ArgDesc> &arg_descs, int32_t insert_pos,
                                                   uint64_t custom_value) {
  ArgDesc arg = {AddrType::CUSTOM_VALUE, static_cast<int32_t>(ArgsFormatWidth::BIT64), false, {0}};
  *reinterpret_cast<uint64_t *>(arg.reserved) = custom_value;
  if (insert_pos < 0) {
    arg_descs.emplace_back(arg);
  } else {
    GE_ASSERT_TRUE(static_cast<size_t>(insert_pos) <= arg_descs.size());
    arg_descs.insert(arg_descs.begin() + insert_pos, arg);
  }
  return GRAPH_SUCCESS;
}

graphStatus ArgsFormatDescUtils::Parse(const std::string &str, std::vector<ArgDesc> &arg_descs) {
  return ArgsFormatDesc::Parse(nullptr, str, arg_descs, true);
}

std::string ArgsFormatDescUtils::Serialize(const std::vector<ArgDesc> &arg_descs) {
  return ArgsFormatDesc::Serialize(arg_descs);
}

std::string ArgsFormatDescUtils::ToString(const std::vector<ArgDesc> &arg_descs) {
  return ArgsFormatDesc::Serialize(arg_descs);
}
}  // namespace ge