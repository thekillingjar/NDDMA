/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "exe_graph/lowering/tiling_parse_context_builder.h"

#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph_metadef/graph/debug/ge_util.h"
#include "graph/def_types.h"
#include "common/checker.h"
#include "graph_metadef/graph/debug/ge_util.h"

namespace gert {
TilingParseContextBuilder &TilingParseContextBuilder::CompileJson(const ge::char_t *compile_json) {
  compile_json_ = const_cast<ge::char_t *>(compile_json);
  return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::PlatformInfo(void *platform_info) {
  platform_info_ = platform_info;
  return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::CompileInfoCreatorFunc(
    OpImplRegisterV2::CompileInfoCreatorFunc create_func) {
  create_func_ = create_func;
  return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::CompileInfoDeleterFunc(
    OpImplRegisterV2::CompileInfoDeleterFunc delete_func) {
  delete_func_ = delete_func;
  return *this;
}

KernelContextHolder TilingParseContextBuilder::Build(const ge::Operator &op) {
  KernelContextHolder holder{};
  if (compile_json_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Compile info is nullptr.");
    return holder;
  }
  if (platform_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Platform info is nullptr.");
    return holder;
  }
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_CHECK_NOTNULL_EXEC(op_desc, return holder);
  std::vector<std::pair<void *, gert::Chain::Deleter>> tiling_parse_outputs(1, std::make_pair(nullptr, nullptr));
  if (create_func_ != nullptr && delete_func_ != nullptr) {
    tiling_parse_outputs[0].first = create_func_();
    tiling_parse_outputs[0].second = delete_func_;
  }
  return gert::KernelRunContextBuilder()
      .Inputs({compile_json_})
      .Inputs({platform_info_})
      .Inputs({const_cast<ge::char_t *>(op_desc->GetTypePtr())})
      .Outputs(tiling_parse_outputs)
      .Build(op_desc);
}
}  // namespace gert
