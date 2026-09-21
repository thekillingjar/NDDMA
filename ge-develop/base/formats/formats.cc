/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "formats/formats.h"

#include <securec.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "formats/utils/formats_trans_utils.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/debug/log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/utils/type_utils.h"

namespace ge {
namespace formats {
Status TransDataFormat(const TransArgs &args, TransResult &result) {
  const auto transfer = BuildFormatTransfer(args);
  if (transfer == nullptr) {
    const std::string error = "Failed to trans data from format " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }

  const auto src_shape_size = GetItemNumByShape(args.src_shape);
  if ((args.data == nullptr) && (src_shape_size != 0)) {
    GELOGE(ACL_ERROR_GE_PARAM_INVALID, "[Check][Shape]Failed, input data is null "
           "or shape size not euqal to 0, src_shape %s",
           ShapeToString(args.src_shape).c_str());
    REPORT_INNER_ERR_MSG("E19999", "Failed to check shape, input data is null "
                      "or shape size not equal to 0, src_shape %s",
                      ShapeToString(args.src_shape).c_str());
    return ACL_ERROR_GE_PARAM_INVALID;
  }

  return transfer->TransFormat(args, result);
}

Status TransTensorShape(const Format src_format, const std::vector<int64_t> &src_shape, const DataType data_type,
                        const Format dst_format, std::vector<int64_t> &dst_shape) {
  formats::TransArgs args;
  args.src_format = src_format;
  args.dst_format = dst_format;
  args.src_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(src_format)));
  args.dst_primary_format = static_cast<Format>(GetPrimaryFormat(static_cast<int32_t>(dst_format)));
  const auto transfer = BuildFormatTransfer(args);
  if (transfer == nullptr) {
    const std::string error = "Failed to trans data from format " +
        FmtToStr(TypeUtils::FormatToSerialString(args.src_primary_format)) + " to " +
        FmtToStr(TypeUtils::FormatToSerialString(args.dst_primary_format));
    GE_ERRORLOG_AND_ERRORMSG(ACL_ERROR_GE_FORMAT_INVALID, error.c_str());
    return ACL_ERROR_GE_FORMAT_INVALID;
  }

  return transfer->TransShape(src_format, src_shape, data_type, dst_format, dst_shape);
}

bool IsTransFormatSupport(const TransArgs &args) {
  return FormatTransferExists(args);
}
}  // namespace formats
}  // namespace ge
