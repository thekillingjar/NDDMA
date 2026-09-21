/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "graph/args_format/arg_desc_info_impl.h"
#include <unordered_map>
#include "graph/utils/args_format_desc_utils.h"
#include "common/checker.h"

namespace ge {
namespace {
bool HasIrIndex(AddrType type) {
  return (type == AddrType::INPUT) || (type == AddrType::OUTPUT) ||
      (type == AddrType::INPUT_DESC) || (type == AddrType::OUTPUT_DESC);
}
AddrType TransToAddrType(ArgDescType args_type) {
  static const std::unordered_map<ArgDescType, AddrType> arg_desc_to_addr_type = {
      {ArgDescType::kIrInput, AddrType::INPUT}, {ArgDescType::kIrOutput, AddrType::OUTPUT},
      {ArgDescType::kWorkspace, AddrType::WORKSPACE}, {ArgDescType::kTiling, AddrType::TILING},
      {ArgDescType::kIrInput, AddrType::INPUT_DESC}, {ArgDescType::kIrOutput, AddrType::OUTPUT_DESC},
      {ArgDescType::kHiddenInput, AddrType::HIDDEN_INPUT},
      {ArgDescType::kCustomValue, AddrType::CUSTOM_VALUE},
      {ArgDescType::kIrInputDesc, AddrType::INPUT_DESC},
      {ArgDescType::kIrOutputDesc, AddrType::OUTPUT_DESC},
      {ArgDescType::kInputInstance, AddrType::INPUT_INSTANCE},
      {ArgDescType::kOutputInstance, AddrType::OUTPUT_INSTANCE}};
  auto iter = arg_desc_to_addr_type.find(args_type);
  if (iter != arg_desc_to_addr_type.end()) {
    return iter->second;
  }
  return AddrType::MAX;
}

ArgDescType TransToArgDescType(AddrType addr_type) {
  static const std::unordered_map<AddrType, ArgDescType> addr_to_arg_desc_type = {
      {AddrType::INPUT, ArgDescType::kIrInput}, {AddrType::OUTPUT, ArgDescType::kIrOutput},
      {AddrType::WORKSPACE, ArgDescType::kWorkspace}, {AddrType::TILING, ArgDescType::kTiling},
      {AddrType::HIDDEN_INPUT, ArgDescType::kHiddenInput},
      {AddrType::CUSTOM_VALUE, ArgDescType::kCustomValue},
      {AddrType::INPUT_DESC, ArgDescType::kIrInputDesc},
      {AddrType::INPUT_INSTANCE, ArgDescType::kInputInstance},
      {AddrType::OUTPUT_DESC, ArgDescType::kIrOutputDesc},
      {AddrType::OUTPUT_INSTANCE, ArgDescType::kOutputInstance}
  };
  auto iter = addr_to_arg_desc_type.find(addr_type);
  if (iter != addr_to_arg_desc_type.end()) {
    return iter->second;
  }
  return ArgDescType::kEnd;
}

HiddenInputsType TransToHiddenInputType(HiddenInputSubType hidden_sub_type) {
  static const std::unordered_map<HiddenInputSubType, HiddenInputsType> hidden_sub_types = {
      {HiddenInputSubType::kHcom, HiddenInputsType::HCOM},
      {HiddenInputSubType::kEnd, HiddenInputsType::MAX}
  };
  auto iter = hidden_sub_types.find(hidden_sub_type);
  if (iter != hidden_sub_types.end()) {
    return iter->second;
  }
  return HiddenInputsType::MAX;
}
HiddenInputSubType TransToHiddenInputSubType(HiddenInputsType hidden_type) {
  static const std::unordered_map<HiddenInputsType, HiddenInputSubType> hidden_input_types = {
      {HiddenInputsType::HCOM, HiddenInputSubType::kHcom},
      {HiddenInputsType::MAX, HiddenInputSubType::kEnd}
  };
  auto iter = hidden_input_types.find(hidden_type);
  if (iter != hidden_input_types.end()) {
    return iter->second;
  }
  return HiddenInputSubType::kEnd;
}
}
AscendString ArgsFormatSerializer::Serialize(const std::vector<ArgDescInfo> &args_format) {
  // 将args_desc_info转成arg_desc
  std::vector<ArgDesc> arg_descs;
  int32_t hidden_input_index = 0;
  int32_t input_instance_index = 0;
  int32_t output_instance_index = 0;
  for (const auto &arg_desc_info : args_format) {
    ArgDesc desc;
    desc.addr_type = TransToAddrType(arg_desc_info.GetType());
    // 当内部类型无法被解析出来时，表示这个argDescInfo可能是内部框架生成，尝试使用inner_arg_type做序列化
    if (desc.addr_type == AddrType::MAX) {
      GE_ASSERT_NOTNULL(arg_desc_info.impl_);
      desc.addr_type = arg_desc_info.impl_->GetInnerArgType();
    }
    // kHiddenInput,kInputInstance和kOutputInstance的索引需要单独排序
    if (arg_desc_info.GetType() == ArgDescType::kHiddenInput) {
      desc.ir_idx = hidden_input_index;
      hidden_input_index++;
    } else if (arg_desc_info.GetType() == ArgDescType::kInputInstance) {
      desc.ir_idx = input_instance_index;
      input_instance_index++;
    } else if (arg_desc_info.GetType() == ArgDescType::kOutputInstance) {
      desc.ir_idx = output_instance_index;
      output_instance_index++;
    } else {
      desc.ir_idx = arg_desc_info.GetIrIndex();
    }

    desc.folded = arg_desc_info.IsFolded();
    if (arg_desc_info.GetType() == ArgDescType::kCustomValue) {
      *reinterpret_cast<uint64_t *>(desc.reserved) = arg_desc_info.GetCustomValue();
    } else if (arg_desc_info.GetType() == ArgDescType::kHiddenInput) {
      *reinterpret_cast<uint32_t *>(desc.reserved) =
          static_cast<uint32_t>(TransToHiddenInputType(arg_desc_info.GetHiddenInputSubType()));
    } else {
      // static check
    }
    arg_descs.emplace_back(desc);
  }
  return AscendString(ArgsFormatDescUtils::Serialize(arg_descs).c_str());
}

std::vector<ArgDescInfo> ArgsFormatSerializer::Deserialize(const AscendString &args_str) {
  std::vector<ArgDesc> arg_descs;
  GE_ASSERT_SUCCESS(ArgsFormatDescUtils::Parse(std::string(args_str.GetString()), arg_descs));
  // 将args_desc转成arg_desc_info
  std::vector<ArgDescInfo> args_format;
  for (const auto &desc : arg_descs) {
    auto arg_desc_type = TransToArgDescType(desc.addr_type);
    ArgDescInfo arg_desc(arg_desc_type, -1, desc.folded);
    if (HasIrIndex(desc.addr_type)) {
      // 除了kIrInput,kIrOutput,kIrInputDesc,kIrOutputDesc，其他type没有ir_index
      arg_desc.SetIrIndex(desc.ir_idx);
    }
    GE_ASSERT_NOTNULL(arg_desc.impl_);
    arg_desc.impl_->SetInnerArgType(desc.addr_type);
    if (desc.addr_type == AddrType::CUSTOM_VALUE) {
      arg_desc.SetCustomValue(*reinterpret_cast<const uint64_t *>(desc.reserved));
    } else if (desc.addr_type == AddrType::HIDDEN_INPUT) {
      arg_desc.SetHiddenInputSubType(TransToHiddenInputSubType(
          static_cast<HiddenInputsType>(*reinterpret_cast<const uint32_t *>(desc.reserved))));
    } else {
      // static check
    }
    args_format.emplace_back(arg_desc);
  }
  return args_format;
}
}