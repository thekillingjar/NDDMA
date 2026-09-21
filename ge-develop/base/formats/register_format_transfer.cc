/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "formats/register_format_transfer.h"

#include <map>

namespace ge {
namespace formats {
namespace {
struct FormatTransferRegistry {
  Status RegisterBuilder(const Format src, const Format dst, FormatTransferBuilder builder) {
    src_dst_builder[src][dst] = std::move(builder);
    return SUCCESS;
  }

  std::shared_ptr<FormatTransfer> GenerateFormatTransfer(const Format src, const Format dst) {
    const std::map<Format,
                   std::map<Format, FormatTransferBuilder>>::const_iterator dst_builder = src_dst_builder.find(src);
    if (dst_builder == src_dst_builder.cend()) {
      return nullptr;
    }
    const auto builder_iter = dst_builder->second.find(dst);
    if (builder_iter == dst_builder->second.end()) {
      return nullptr;
    }
    return builder_iter->second();
  }

  bool IsFormatTransferExists(const Format src, const Format dst) {
    const std::map<Format,
                   std::map<Format, FormatTransferBuilder>>::const_iterator dst_builder = src_dst_builder.find(src);
    if (dst_builder == src_dst_builder.cend()) {
      return false;
    }
    return dst_builder->second.count(dst) > 0UL;
  }

private:
  std::map<Format, std::map<Format, FormatTransferBuilder>> src_dst_builder;
};

FormatTransferRegistry &GetFormatTransferRegistry() {
  static FormatTransferRegistry registry;
  return registry;
}
}  // namespace

FormatTransferRegister::FormatTransferRegister(FormatTransferBuilder builder, const Format src,
                                               const Format dst) noexcept {
  (void)GetFormatTransferRegistry().RegisterBuilder(src, dst, std::move(builder));
  // RegisterBuilder() always return success, no need to check value
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::shared_ptr<FormatTransfer> BuildFormatTransfer(
    const TransArgs &args) {
  return GetFormatTransferRegistry().GenerateFormatTransfer(args.src_primary_format, args.dst_primary_format);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool FormatTransferExists(const TransArgs &args) {
  return GetFormatTransferRegistry().IsFormatTransferExists(args.src_primary_format, args.dst_primary_format);
}
}  // namespace formats
}  // namespace ge
