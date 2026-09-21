/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "format_selector/builtin/process/format_process_registry.h"
#include "common/fe_type_utils.h"
#include "common/fe_utils.h"
#include "graph/utils/type_utils.h"

namespace fe {
FormatProcessRegistry &GetFormatProcessRegistry() {
  static FormatProcessRegistry registry;
  return registry;
}

FormatProcessBasePtr BuildFormatProcess(const FormatProccessArgs &args) {
  auto registry = GetFormatProcessRegistry();
  const auto dst_builder = registry.src_dst_builder.find(args.op_pattern);
  if (dst_builder == registry.src_dst_builder.end()) {
    return nullptr;
  }
  const auto builder_iter = dst_builder->second.find(args.support_format);
  if (builder_iter == dst_builder->second.end()) {
    return nullptr;
  }
  return builder_iter->second();
}

bool FormatProcessExists(const FormatProccessArgs &args) {
  auto registry = GetFormatProcessRegistry();
  auto dst_builder = registry.src_dst_builder.find(args.op_pattern);
  if (dst_builder == registry.src_dst_builder.end()) {
    return false;
  }
  return dst_builder->second.count(args.support_format) > 0;
}

FormatProcessRegister::FormatProcessRegister(FormatProcessBuildFunc builder, OpPattern op_pattern, ge::Format format) {
  FE_LOGD("The parameter[pattern[%s]], parameter[format[%s]] of FormatProcessRegister",
          GetOpPatternString(op_pattern).c_str(), ge::TypeUtils::FormatToSerialString(format).c_str());
  auto ret = GetFormatProcessRegistry().RegisterBuilder(op_pattern, format, std::move(builder));
  if (ret != SUCCESS) {
    REPORT_FE_ERROR(
        "[GraphOpt][GenBuiltInFmt][FmtProcsReg] Failed to register the builder for the format of op_pattern[%d].",
        op_pattern);
  } else {
    FE_LOGD("Processor of pattern[%s] and format[%s] is registered", GetOpPatternString(op_pattern).c_str(),
            FormatToStr(format).c_str());
  }
}
}  // namespace fe
