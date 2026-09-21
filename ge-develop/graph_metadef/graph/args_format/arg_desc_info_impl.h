/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef METADEF_GRAPH_ARGS_FORMAT_ARG_DESC_INFO_IMPL_H
#define METADEF_GRAPH_ARGS_FORMAT_ARG_DESC_INFO_IMPL_H

#include "graph/arg_desc_info.h"
#include "graph/utils/args_format_desc_utils.h"

namespace ge {
class ArgDescInfoImpl {
 public:
  explicit ArgDescInfoImpl(ArgDescType arg_type,
      int32_t ir_index = -1, bool is_folded = false);
  ~ArgDescInfoImpl() = default;
  ArgDescInfoImpl() = default;
  static ArgDescInfoImplPtr CreateCustomValue(uint64_t custom_value);
  static ArgDescInfoImplPtr CreateHiddenInput(HiddenInputSubType hidden_type);
  ArgDescType GetType() const;
  uint64_t GetCustomValue() const;
  graphStatus SetCustomValue(uint64_t custom_value);
  HiddenInputSubType GetHiddenInputSubType() const;
  graphStatus SetHiddenInputSubType(HiddenInputSubType hidden_type);
  void SetIrIndex(int32_t ir_index);
  int32_t GetIrIndex() const;
  bool IsFolded() const;
  void SetFolded(bool is_folded);
  void SetInnerArgType(AddrType inner_arg_type);
  AddrType GetInnerArgType() const;
 private:
  ArgDescType arg_type_{ArgDescType::kEnd};
  AddrType inner_arg_type_{AddrType::MAX};
  int32_t ir_index_{-1};
  HiddenInputSubType hidden_type_{HiddenInputSubType::kEnd};
  uint64_t custom_value_{0};
  bool is_folded_{false};
};
}

#endif // METADEF_GRAPH_ARGS_FORMAT_ARG_DESC_INFO_IMPL_H